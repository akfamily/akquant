"""发布前校验版本号自洽，供 release.yml 调用.

PyPI 的版本号一经占用即不可重用，发错只能改用下一个号，因此在构建之前就要拦下。
版本号散在四处，必须一致：pyproject.toml、Cargo.toml、Cargo.lock 中的 akquant
条目、CHANGELOG.md 的版本标题；给出 tag 时 tag 也必须与之一致。

另有两个可选动作（只在 release job 里用）：
    --dist DIR       校验构建产物文件名都带着这个版本号
                     （构建后端取错版本时只有看产物才能发现）
    --notes-out FILE 把 CHANGELOG 中该版本的段落写出，作为 GitHub Release 正文

只依赖标准库（tomllib 需 Python 3.11+），CI 中无需安装任何依赖。
"""

import argparse
import re
import sys
from pathlib import Path

import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "akquant"
TAG_RE = re.compile(r"^v(\d+\.\d+\.\d+)$")
CHANGELOG_HEADING_RE = re.compile(r"^## \[(?P<version>[^\]]+)\]", re.MULTILINE)


def parse_tag(tag: str) -> str:
    """从形如 v1.2.3 的 tag 取出版本号."""
    match = TAG_RE.match(tag)
    if not match:
        raise ValueError(f"tag 格式不合法，应为 vX.Y.Z，实际为 {tag!r}")
    return match.group(1)


def read_declared_versions(root: Path) -> dict[str, str]:
    """读取仓库内各处声明的版本号，键为来源说明."""
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    cargo = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((root / "Cargo.lock").read_text(encoding="utf-8"))
    lock_versions = [
        pkg["version"] for pkg in lock.get("package", []) if pkg["name"] == PACKAGE
    ]
    if len(lock_versions) != 1:
        raise ValueError(
            f"Cargo.lock 中 {PACKAGE} 条目应恰有 1 个，实际 {len(lock_versions)} 个"
        )
    return {
        "pyproject.toml": pyproject["project"]["version"],
        "Cargo.toml": cargo["package"]["version"],
        "Cargo.lock": lock_versions[0],
    }


def extract_changelog_section(text: str, version: str) -> str | None:
    """返回 CHANGELOG 中该版本标题下的正文（不含标题行），没有该条目返回 None."""
    headings = list(CHANGELOG_HEADING_RE.finditer(text))
    for index, heading in enumerate(headings):
        if heading.group("version") != version:
            continue
        body_start = text.find("\n", heading.end()) + 1
        body_end = (
            headings[index + 1].start() if index + 1 < len(headings) else len(text)
        )
        return text[body_start:body_end].strip()
    return None


def check_dist(dist: Path, version: str) -> list[str]:
    """校验产物文件名都带着该版本号，且 wheel 与 sdist 都存在；返回错误列表."""
    files = sorted(p.name for p in dist.iterdir() if p.is_file())
    errors = [
        f"产物 {name} 与版本 {version} 不匹配"
        for name in files
        if not (
            (name.startswith(f"{PACKAGE}-{version}-") and name.endswith(".whl"))
            or name == f"{PACKAGE}-{version}.tar.gz"
        )
    ]
    if not any(name.endswith(".whl") for name in files):
        errors.append(f"{dist} 中没有 wheel")
    if f"{PACKAGE}-{version}.tar.gz" not in files:
        errors.append(f"{dist} 中没有 sdist {PACKAGE}-{version}.tar.gz")
    return errors


def main(argv: list[str] | None = None) -> int:
    """命令行入口，校验通过返回 0."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "tag", nargs="?", help="git tag，如 v0.3.72；省略则只校验仓库内一致"
    )
    parser.add_argument("--dist", type=Path, help="构建产物目录")
    parser.add_argument(
        "--notes-out", type=Path, help="写出该版本 CHANGELOG 段落的文件"
    )
    parser.add_argument(
        "--root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    declared = read_declared_versions(args.root)
    version = declared["pyproject.toml"]
    if args.tag:
        try:
            version = parse_tag(args.tag)
        except ValueError as exc:
            print(f"[release] {exc}")
            return 1
    for source, value in declared.items():
        if value != version:
            errors.append(f"{source} 版本为 {value}，期望 {version}")

    changelog = (args.root / "CHANGELOG.md").read_text(encoding="utf-8")
    notes = extract_changelog_section(changelog, version)
    if notes is None:
        errors.append(f"CHANGELOG.md 中没有 ## [{version}] 条目")

    if args.dist is not None:
        errors.extend(check_dist(args.dist, version))

    if errors:
        for error in errors:
            print(f"[release] {error}")
        return 1

    if args.notes_out is not None and notes is not None:
        args.notes_out.write_text(notes + "\n", encoding="utf-8")
    print(f"[release] 版本 {version} 校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
