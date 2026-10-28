import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("tomllib", reason="check_release.py 需要 Python 3.11+")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_release.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_release", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_release"] = module
    spec.loader.exec_module(module)
    return module


check_release = _load_script()

CHANGELOG = """# Changelog

## [Unreleased]

## [1.2.3] - 2026-10-28

### Fixed
- 修了一个问题。

## [1.2.2] - 2026-10-01

- 更早的条目。
"""


def _make_repo(
    root: Path,
    *,
    pyproject: str = "1.2.3",
    cargo: str = "1.2.3",
    lock: str = "1.2.3",
    changelog: str = CHANGELOG,
) -> Path:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "akquant"\nversion = "{pyproject}"\n', encoding="utf-8"
    )
    (root / "Cargo.toml").write_text(
        f'[package]\nname = "akquant"\nversion = "{cargo}"\n', encoding="utf-8"
    )
    (root / "Cargo.lock").write_text(
        "version = 4\n\n"
        '[[package]]\nname = "polars"\nversion = "0.55.2"\n\n'
        f'[[package]]\nname = "akquant"\nversion = "{lock}"\n',
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    return root


def _run(root: Path, *args: str) -> int:
    return int(check_release.main([*args, "--root", str(root)]))


def test_consistent_versions_pass_with_and_without_tag(tmp_path: Path) -> None:
    """各处版本一致时，带不带 tag 都通过."""
    root = _make_repo(tmp_path)
    assert _run(root) == 0
    assert _run(root, "v1.2.3") == 0


@pytest.mark.parametrize(
    ("field", "label"),
    [("pyproject", "pyproject.toml"), ("cargo", "Cargo.toml"), ("lock", "Cargo.lock")],
)
def test_any_drifted_source_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str, label: str
) -> None:
    """任一处版本漂移都失败，并指出是哪一处."""
    root = _make_repo(tmp_path, **{field: "1.2.4"})
    assert _run(root, "v1.2.3") == 1
    assert label in capsys.readouterr().out


def test_lock_without_package_entry_raises(tmp_path: Path) -> None:
    """Cargo.lock 里没有 akquant 条目时报错而非静默通过."""
    root = _make_repo(tmp_path)
    (root / "Cargo.lock").write_text(
        'version = 4\n\n[[package]]\nname = "polars"\nversion = "0.55.2"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Cargo.lock"):
        _run(root)


def test_tag_not_matching_declared_version_fails(tmp_path: Path) -> None:
    """仓库内版本与 tag 不一致时失败."""
    root = _make_repo(tmp_path)
    assert _run(root, "v1.2.4") == 1


@pytest.mark.parametrize("tag", ["1.2.3", "v1.2", "v1.2.3rc1", "release-v1.2.3"])
def test_malformed_tag_fails(tmp_path: Path, tag: str) -> None:
    """非 vX.Y.Z 格式的 tag 被拒."""
    root = _make_repo(tmp_path)
    assert _run(root, tag) == 1


def test_missing_changelog_entry_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CHANGELOG 没有该版本条目时失败."""
    root = _make_repo(tmp_path, changelog="# Changelog\n\n## [1.2.2] - 2026-10-01\n")
    assert _run(root, "v1.2.3") == 1
    assert "CHANGELOG.md" in capsys.readouterr().out


def test_notes_out_writes_only_that_version_section(tmp_path: Path) -> None:
    """Release 正文只取该版本段落，不含标题和相邻版本."""
    root = _make_repo(tmp_path)
    notes = tmp_path / "notes.md"
    assert _run(root, "v1.2.3", "--notes-out", str(notes)) == 0
    assert notes.read_text(encoding="utf-8") == "### Fixed\n- 修了一个问题。\n"


def test_dist_with_matching_artifacts_passes(tmp_path: Path) -> None:
    """产物文件名都带正确版本且 wheel、sdist 齐全时通过."""
    root = _make_repo(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    for name in [
        "akquant-1.2.3-cp310-abi3-win_amd64.whl",
        "akquant-1.2.3-cp310-abi3-macosx_11_0_arm64.whl",
        "akquant-1.2.3.tar.gz",
    ]:
        (dist / name).touch()
    assert _run(root, "v1.2.3", "--dist", str(dist)) == 0


@pytest.mark.parametrize(
    "names",
    [
        # 产物版本与 tag 不一致
        ["akquant-1.2.2-cp310-abi3-win_amd64.whl", "akquant-1.2.3.tar.gz"],
        # 缺 sdist
        ["akquant-1.2.3-cp310-abi3-win_amd64.whl"],
        # 缺 wheel
        ["akquant-1.2.3.tar.gz"],
        # 混入无关文件
        [
            "akquant-1.2.3-cp310-abi3-win_amd64.whl",
            "akquant-1.2.3.tar.gz",
            "other-1.2.3.tar.gz",
        ],
    ],
)
def test_dist_with_bad_artifacts_fails(tmp_path: Path, names: list[str]) -> None:
    """产物版本错、缺 wheel/sdist 或混入无关文件时失败."""
    root = _make_repo(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    for name in names:
        (dist / name).touch()
    assert _run(root, "v1.2.3", "--dist", str(dist)) == 1


def test_real_repo_is_consistent() -> None:
    """仓库当前的版本声明本身自洽."""
    assert check_release.main([]) == 0
