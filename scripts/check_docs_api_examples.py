import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

FENCE_PATTERN = re.compile(r"^```([A-Za-z0-9_+-]*)\s*$")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_FIELD_PATTERNS = {
    "legacy_result_field_final_value": re.compile(r"\bresult\.final_value\b"),
    "legacy_result_field_total_return": re.compile(r"\bresult\.total_return\b"),
    # 可视化方法已收敛到 result.viz.* 命名空间(见
    # docs/zh/meta/viz-namespace-and-lwc-review-rfc.md);旧的顶层方法已删除。
    # 负向零宽断言排除新命名空间形式(result.viz.report( 等)。
    "legacy_viz_report": re.compile(r"(?<!viz)\.report\("),
    "legacy_viz_report_quantstats": re.compile(r"\.report_quantstats\("),
    "legacy_viz_plot_indicators": re.compile(r"\.plot_indicators\("),
    "legacy_viz_plot": re.compile(r"(?<![a-z_])result[a-z_]*\.plot\("),
}
RUN_BACKTEST_RENAMES = {
    "cash": "initial_cash",
    "commission": "commission_rate",
    "strategy_class": "strategy",
}
RUN_OPTIMIZATION_RENAMES = {
    "cash": "initial_cash",
    "commission": "commission_rate",
}


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _run_git_command(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            cwd=PROJECT_ROOT,
        )
    except OSError as exc:
        print(
            f"docs api example check fallback: failed to run {' '.join(args)} ({exc})"
        )
        return None


def _extract_func_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _collect_python_blocks(md_text: str) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    lines = md_text.splitlines()
    in_python = False
    start_line = 0
    buf: list[str] = []
    for idx, line in enumerate(lines, 1):
        match = FENCE_PATTERN.match(line)
        if match:
            lang = match.group(1).lower()
            if in_python:
                blocks.append((start_line, "\n".join(buf)))
                in_python = False
                start_line = 0
                buf = []
                continue
            if lang in {"python", "py"}:
                in_python = True
                start_line = idx + 1
                buf = []
            continue
        if in_python:
            buf.append(line)
    return blocks


def _analyze_python_block(
    file_path: Path,
    start_line: int,
    source: str,
) -> list[tuple[Path, int, str, str]]:
    violations: list[tuple[Path, int, str, str]] = []
    lines = source.splitlines()
    for offset, line in enumerate(lines, 0):
        for label, pattern in LEGACY_FIELD_PATTERNS.items():
            if pattern.search(line):
                violations.append((file_path, start_line + offset, label, line.strip()))
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return violations
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = _extract_func_name(node)
        if func_name == "run_backtest":
            rename_map = RUN_BACKTEST_RENAMES
        elif func_name == "run_optimization":
            rename_map = RUN_OPTIMIZATION_RENAMES
        else:
            continue
        for kw in node.keywords:
            if kw.arg is None:
                continue
            if kw.arg not in rename_map:
                continue
            line_text = ""
            line_idx = max(1, node.lineno) - 1
            if line_idx < len(lines):
                line_text = lines[line_idx].strip()
            violations.append(
                (
                    file_path,
                    start_line + node.lineno - 1,
                    "legacy_api_argument",
                    f"{kw.arg} -> {rename_map[kw.arg]} | {line_text}",
                )
            )
    return violations


def _scan_markdown_files(docs_dir: Path) -> list[tuple[Path, int, str, str]]:
    violations: list[tuple[Path, int, str, str]] = []
    for md_file in sorted(docs_dir.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        blocks = _collect_python_blocks(text)
        for start_line, source in blocks:
            violations.extend(_analyze_python_block(md_file, start_line, source))
    return violations


def _scan_target_files(files: list[Path]) -> list[tuple[Path, int, str, str]]:
    violations: list[tuple[Path, int, str, str]] = []
    for md_file in sorted(files):
        text = md_file.read_text(encoding="utf-8")
        blocks = _collect_python_blocks(text)
        for start_line, source in blocks:
            violations.extend(_analyze_python_block(md_file, start_line, source))
    return violations


def _resolve_scan_files(docs_dir: Path, files: list[str]) -> list[Path]:
    scan_files: list[Path] = []
    for raw in files:
        candidate = _resolve_project_path(raw)
        if not candidate.exists():
            continue
        if candidate.suffix != ".md":
            continue
        if docs_dir not in candidate.parents:
            continue
        scan_files.append(candidate)
    return sorted(set(scan_files))


def _changed_files_between_revs(
    docs_dir: Path,
    from_rev: str,
    to_rev: str,
) -> list[Path] | None:
    from_rev_result = _run_git_command(
        ["git", "rev-parse", "--verify", f"{from_rev}^{{commit}}"]
    )
    to_rev_result = _run_git_command(
        ["git", "rev-parse", "--verify", f"{to_rev}^{{commit}}"]
    )
    if from_rev_result is None or to_rev_result is None:
        print(
            "docs api example check fallback: "
            f"revision check failed ({from_rev}..{to_rev}), scanning all docs"
        )
        return None
    from_ok = from_rev_result.returncode == 0
    to_ok = to_rev_result.returncode == 0
    if not from_ok or not to_ok:
        print(
            "docs api example check fallback: "
            f"revision not found ({from_rev}..{to_rev}), scanning all docs"
        )
        return None

    result = _run_git_command(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{from_rev}...{to_rev}"]
    )
    if result is None:
        print(
            "docs api example check fallback: "
            "git diff command failed, scanning all docs"
        )
        return None
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if stderr:
            print(stderr)
        print("docs api example check fallback: git diff failed, scanning all docs")
        return None
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return _resolve_scan_files(docs_dir, files)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs-dir",
        default="docs",
        help="Docs root directory to scan",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Specific markdown files to scan",
    )
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Scan changed markdown files between revisions",
    )
    parser.add_argument(
        "--from-rev",
        default="HEAD~1",
        help="Git base revision for --changed-only mode",
    )
    parser.add_argument(
        "--to-rev",
        default="HEAD",
        help="Git target revision for --changed-only mode",
    )
    args = parser.parse_args()
    docs_dir = _resolve_project_path(args.docs_dir)
    if not docs_dir.exists():
        print(f"docs directory not found: {docs_dir}")
        return 2

    scan_files: list[Path] | None = None
    if args.files:
        scan_files = _resolve_scan_files(docs_dir, args.files)
    elif args.changed_only:
        scan_files = _changed_files_between_revs(docs_dir, args.from_rev, args.to_rev)

    if scan_files is None:
        violations = _scan_markdown_files(docs_dir)
    else:
        if not scan_files:
            print(
                "docs api example check skipped: "
                f"no markdown files selected under {docs_dir}"
            )
            return 0
        violations = _scan_target_files(scan_files)
    if not violations:
        print(f"docs api example check passed: {docs_dir}")
        return 0

    print(f"docs api example check failed: {len(violations)} issue(s)")
    for path, line_no, label, text in violations:
        print(f"{path}:{line_no}: {label}: {text}")
    return 1


if __name__ == "__main__":
    sys.exit(_main())
