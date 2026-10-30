"""akquant.mcp.service.data: 三种数据源、规范化与缓存."""

import os
from pathlib import Path
from typing import Any, Callable, List

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


def test_akshare_monkeypatched(
    cfg: ServiceConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AKShare 来源: 日期转 YYYYMMDD, 补 symbol 列, 不联网."""
    calls: List[Any] = []

    def fake(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
        calls.append((symbol, start, end, adjust))
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "open": [1.0, 1.1],
                "high": [1.2, 1.2],
                "low": [0.9, 1.0],
                "close": [1.1, 1.15],
                "volume": [100.0, 120.0],
                "amount": [1.0, 2.0],
            }
        )

    monkeypatch.setattr(data, "fetch_akshare_symbol", fake)
    spec = {
        "source": "akshare",
        "symbol": "600000",
        "start": "2024-01-01",
        "end": "20240131",
    }
    out = data.prepare_data(cfg, spec)
    assert calls == [("600000", "20240101", "20240131", "qfq")]
    assert out["symbols"] == ["600000"] and out["synthetic"] is False


def test_akshare_empty_and_spec_validation(
    cfg: ServiceConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AKShare 无数据、未知字段、未知 source、坏 id 都报 ServiceError."""

    def empty(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise ValueError("No data found")

    monkeypatch.setattr(data, "fetch_akshare_symbol", empty)
    spec = {"source": "akshare", "symbol": "600000", "start": "20240101"}
    with pytest.raises(ServiceError, match="end"):
        data.prepare_data(cfg, spec)
    with pytest.raises(ServiceError, match="AKShare"):
        data.prepare_data(cfg, {**spec, "end": "20240131"})
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


def test_akshare_request_failure_is_service_error(
    cfg: ServiceConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AKShare 请求失败(RuntimeError)转成带原因的 ServiceError."""

    def boom(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise RuntimeError("AKShare 请求失败 (sh600000): ConnectionError: down")

    monkeypatch.setattr(data, "fetch_akshare_symbol", boom)
    spec = {
        "source": "akshare",
        "symbol": "600000",
        "start": "20240101",
        "end": "20240131",
    }
    with pytest.raises(ServiceError, match="AKShare 请求失败"):
        data.prepare_data(cfg, spec)
