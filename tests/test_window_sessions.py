"""阶段 5：多周期窗口按交易日与交易块切分."""

from __future__ import annotations

import datetime as dt

import akquant as aq
import pandas as pd

TZ = "Asia/Shanghai"


def _ns(text: str) -> int:
    return int(pd.Timestamp(text, tz=TZ).value)


def test_resolver_trading_blocks_and_block_at() -> None:
    """解析器按交易块返回区间，午休返回 None."""
    r = aq.SessionResolver(timezone=TZ)
    blocks = r.trading_blocks("600000.SH", dt.date(2024, 1, 2), aq.AssetType.Stock)
    assert blocks == [
        (_ns("2024-01-02 09:15"), _ns("2024-01-02 11:30")),
        (_ns("2024-01-02 13:00"), _ns("2024-01-02 15:00")),
    ]
    assert r.block_at("600000.SH", _ns("2024-01-02 12:00"), aq.AssetType.Stock) is None
    hit = r.block_at("600000.SH", _ns("2024-01-02 10:00"), aq.AssetType.Stock)
    assert hit is not None
    day, idx, start, end = hit
    assert (day, idx, start, end) == (
        dt.date(2024, 1, 2),
        0,
        _ns("2024-01-02 09:15"),
        _ns("2024-01-02 11:30"),
    )
