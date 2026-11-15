"""tests/mcp_server 公共夹具."""

from pathlib import Path
from typing import Callable

import pytest
from akquant.mcp.service.paths import ServiceConfig, make_config


@pytest.fixture
def aktools_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """项目根目录之外的 AKTools 数据目录."""
    return tmp_path_factory.mktemp("aktools-data")


@pytest.fixture
def cfg(tmp_path: Path, aktools_dir: Path) -> ServiceConfig:
    """以 tmp_path 为项目根、tmp_path/.home 为工作目录的配置.

    AKTools 数据目录指向临时目录, 不读本机的 ~/.aktools/data。
    """
    return make_config(root=tmp_path, home=tmp_path / ".home", aktools_dir=aktools_dir)


@pytest.fixture
def write_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """返回一个在 tmp_path 下写 UTF-8 文本文件的函数."""

    def _write(name: str, text: str) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    return _write
