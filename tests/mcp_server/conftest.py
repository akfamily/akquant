"""tests/mcp_server 公共夹具."""

from pathlib import Path
from typing import Callable

import pytest
from akquant.mcp.service.paths import ServiceConfig, make_config


@pytest.fixture
def cfg(tmp_path: Path) -> ServiceConfig:
    """以 tmp_path 为项目根、tmp_path/.home 为工作目录的配置."""
    return make_config(root=tmp_path, home=tmp_path / ".home")


@pytest.fixture
def write_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """返回一个在 tmp_path 下写 UTF-8 文本文件的函数."""

    def _write(name: str, text: str) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    return _write
