"""akquant.mcp.service.data: 三种数据源、规范化与缓存."""

import os
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest
from akquant.mcp.service import ServiceError, data
from akquant.mcp.service.paths import ServiceConfig

WriteFile = Callable[[str, str], Path]

CSV = (
    "date,open,high,low,close,volume\n"
    "2024-01-02,10,10.5,9.8,10.2,1000\n"
    "2024-01-03,10.2,10.6,10.0,10.4,1200\n"
    "2024-01-04,10.4,10.7,10.1,,900\n"
)


def test_synthetic_deterministic_and_cached(cfg: ServiceConfig) -> None:
    """相同参数命中缓存、数据一致; 换 seed 得到新数据集."""
    spec = {"source": "synthetic", "end": "2023-03-31"}
    first = data.prepare_data(cfg, spec)
    second = data.prepare_data(cfg, spec)
    assert first["dataset_id"] == second["dataset_id"]
    assert first["cached"] is False and second["cached"] is True
    assert first["synthetic"] is True
    assert first["symbols"] == ["SYN001"]
    assert first["freq_guess"] == "1d"
    assert first["timezone"] == "Asia/Shanghai"
    other = data.prepare_data(cfg, {**spec, "seed": 1})
    assert other["dataset_id"] != first["dataset_id"]
    a = pd.read_parquet(data.dataset_file(cfg, first["dataset_id"]))
    b = data.synthetic_frame(end="2023-03-31")
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_synthetic_minute_bars() -> None:
    """1m 每个工作日 240 根, 落在 A 股连续竞价时段."""
    frame = data.synthetic_frame(freq="1m", start="2024-01-02", end="2024-01-03")
    assert len(frame) == 480
    times = frame["date"].dt.strftime("%H:%M")
    assert times.iloc[0] == "09:31" and times.iloc[239] == "15:00"
    assert "12:00" not in set(times)


def test_file_csv_naive_and_missing(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """CSV 无时区被本地化, 缺 close 的行被丢弃, symbol 取文件名."""
    write_file("bars/600000.csv", CSV)
    out = data.prepare_data(cfg, {"source": "file", "path": "bars/600000.csv"})
    assert out["rows"] == 2
    assert out["symbols"] == ["600000"]
    assert out["columns"] == data.STANDARD_COLUMNS
    joined = " ".join(out["issues"])
    assert "本地化" in joined and "丢弃" in joined
    assert out["start"].endswith("+08:00")


def test_file_mtime_invalidates(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """文件修改后 dataset_id 改变."""
    path = write_file("a.csv", CSV)
    first = data.prepare_data(cfg, {"source": "file", "path": "a.csv"})
    path.write_text(CSV + "2024-01-05,10,10.5,9.8,10.2,1000\n", encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    second = data.prepare_data(cfg, {"source": "file", "path": "a.csv"})
    assert second["dataset_id"] != first["dataset_id"]
    assert second["rows"] == 3


def test_file_blocking_errors(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """缺 OHLC 列、文件不存在、后缀不支持都报 ServiceError."""
    write_file("bad.csv", "date,open\n2024-01-02,1\n")
    with pytest.raises(ServiceError, match="缺少必需列"):
        data.prepare_data(cfg, {"source": "file", "path": "bad.csv"})
    with pytest.raises(ServiceError, match="不存在"):
        data.prepare_data(cfg, {"source": "file", "path": "nope.csv"})
    write_file("x.txt", "")
    with pytest.raises(ServiceError, match="csv"):
        data.prepare_data(cfg, {"source": "file", "path": "x.txt"})


def test_akshare_source_removed_points_to_aktools(cfg: ServiceConfig) -> None:
    """source=akshare 已移除: 报错说明改用 AKTools 的 save_data + source=file."""
    spec = {
        "source": "akshare",
        "symbol": "600000",
        "start": "20240101",
        "end": "20240131",
    }
    with pytest.raises(ServiceError, match="已移除.*save_data.*source.*file"):
        data.prepare_data(cfg, spec)


def test_spec_validation(cfg: ServiceConfig) -> None:
    """未知字段、未知 source、坏 id 都报 ServiceError."""
    with pytest.raises(ServiceError, match="不支持字段"):
        data.prepare_data(cfg, {"source": "synthetic", "rows": 10})
    with pytest.raises(ServiceError, match="source"):
        data.prepare_data(cfg, {"source": "tushare"})
    with pytest.raises(ServiceError):
        data.dataset_file(cfg, "../../etc")
    with pytest.raises(ServiceError, match="prepare_data"):
        data.dataset_file(cfg, "0123456789ab")


def test_user_input_errors_become_service_error(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """可由用户修正的输入问题, 都转成 ServiceError 而不是 pandas 原始异常."""
    mixed = (
        "date,open,high,low,close,volume\n"
        "2024-01-02 09:31:00+08:00,1,1,1,1,1\n"
        "2024-01-02 09:32:00+00:00,1,1,1,1,1\n"
    )
    write_file("mixed.csv", mixed)
    with pytest.raises(ServiceError):
        data.prepare_data(cfg, {"source": "file", "path": "mixed.csv"})
    write_file("ok.csv", CSV)
    with pytest.raises(ServiceError):
        data.prepare_data(
            cfg, {"source": "file", "path": "ok.csv", "timezone": "No/Such_Zone"}
        )
    with pytest.raises(ServiceError):
        data.prepare_data(cfg, {"source": "synthetic", "start": "not-a-date"})
    write_file("empty.csv", "")
    with pytest.raises(ServiceError):
        data.prepare_data(cfg, {"source": "file", "path": "empty.csv"})
    with pytest.raises(ServiceError):
        data.prepare_data(cfg, {"source": "synthetic", "seed": "abc"})


def test_csv_symbol_keeps_leading_zeros(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """CSV 的 symbol 列按字符串读取, 000001 不变成 1."""
    rows = ["date,open,high,low,close,volume,symbol"]
    for day in range(1, 6):
        rows.append(f"2023-01-0{day},10,11,9,10,1000,000001")
    path = write_file("z.csv", "\n".join(rows) + "\n")
    out = data.prepare_data(cfg, {"source": "file", "path": str(path)})
    assert out["symbols"] == ["000001"]


SAVED_NAME = "stock_zh_a_hist_sina-89238273f5.csv"


def test_file_in_aktools_dir_is_readable(cfg: ServiceConfig, aktools_dir: Path) -> None:
    """AKTools save_data 存下的文件在项目根目录之外, 也能用 source=file 读."""
    saved = aktools_dir / SAVED_NAME
    saved.write_text(CSV, encoding="utf-8")
    out = data.prepare_data(
        cfg, {"source": "file", "path": str(saved), "symbol": "600519"}
    )
    assert out["symbols"] == ["600519"] and out["rows"] == 2


def test_file_in_aktools_dir_requires_symbol(
    cfg: ServiceConfig, aktools_dir: Path
) -> None:
    """AKTools 的文件名是接口名加哈希, 不能当标的名: 没传 symbol 直接报错."""
    saved = aktools_dir / SAVED_NAME
    saved.write_text(CSV, encoding="utf-8")
    with pytest.raises(ServiceError, match="symbol"):
        data.prepare_data(cfg, {"source": "file", "path": str(saved)})
    with_symbol = aktools_dir / "with-symbol.csv"
    with_symbol.write_text(
        "date,open,high,low,close,volume,symbol\n2024-01-02,10,11,9,10,1000,000001\n",
        encoding="utf-8",
    )
    out = data.prepare_data(cfg, {"source": "file", "path": str(with_symbol)})
    assert out["symbols"] == ["000001"]


def test_file_in_root_without_symbol_still_uses_stem(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """项目目录内的文件不传 symbol 时仍用文件名(原有行为不变)."""
    write_file("600000.csv", CSV)
    out = data.prepare_data(cfg, {"source": "file", "path": "600000.csv"})
    assert out["symbols"] == ["600000"]


def test_aktools_dir_only_accepts_data_files(
    cfg: ServiceConfig, aktools_dir: Path
) -> None:
    """AKTools 目录里也只读 .csv / .parquet."""
    other = aktools_dir / "notes.txt"
    other.write_text("x", encoding="utf-8")
    with pytest.raises(ServiceError, match="csv"):
        data.prepare_data(cfg, {"source": "file", "path": str(other), "symbol": "x"})


CN_CSV = (
    "日期,股票代码,开盘,收盘,最高,最低,成交量,成交额\n"
    "2024-01-02,000001,9.39,9.21,9.42,9.20,1158366,1075742252.45\n"
    "2024-01-03,000001,9.19,9.20,9.22,9.15,733610,673673613.54\n"
)


def _frame(cfg: ServiceConfig, out: Any) -> pd.DataFrame:
    return pd.read_parquet(data.dataset_file(cfg, out["dataset_id"]))


def test_chinese_column_names(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """东方财富等接口的中文列名按 akquant.schema 的别名表识别, 代码保留前导零."""
    write_file("em.csv", CN_CSV)
    out = data.prepare_data(cfg, {"source": "file", "path": "em.csv"})
    assert out["symbols"] == ["000001"] and out["rows"] == 2
    first = _frame(cfg, out).iloc[0]
    got = [first[c] for c in ("open", "high", "low", "close", "volume")]
    assert got == [9.39, 9.42, 9.20, 9.21, 1158366.0]


def test_english_column_wins_over_chinese_alias(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """同一字段中英文列都在时用英文列, 不被别名覆盖."""
    text = "date,open,high,low,close,收盘,volume\n2024-01-02,1,3,0.5,2,99,10\n"
    write_file("both.csv", text)
    out = data.prepare_data(cfg, {"source": "file", "path": "both.csv"})
    assert _frame(cfg, out).iloc[0]["close"] == 2.0


@pytest.mark.parametrize("text", [CSV, CN_CSV], ids=["english", "chinese"])
def test_csv_with_bom(cfg: ServiceConfig, tmp_path: Path, text: str) -> None:
    """带 BOM 的 UTF-8 CSV(Excel 另存的常见格式)照常读, 首列列名不带 BOM."""
    (tmp_path / "bom.csv").write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    out = data.prepare_data(
        cfg, {"source": "file", "path": "bom.csv", "symbol": "000001"}
    )
    assert out["rows"] == 2 and out["symbols"] == ["000001"]


def test_aktools_file_with_chinese_symbol_column_needs_no_symbol(
    cfg: ServiceConfig, aktools_dir: Path
) -> None:
    """AKTools 存下的文件里有“股票代码”列时, 不必再传 symbol."""
    saved = aktools_dir / "stock_zh_a_hist_em-0123456789.csv"
    saved.write_text(CN_CSV, encoding="utf-8")
    out = data.prepare_data(cfg, {"source": "file", "path": str(saved)})
    assert out["symbols"] == ["000001"]
