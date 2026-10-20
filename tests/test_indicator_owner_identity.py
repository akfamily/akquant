"""指标定义按 owner 区分身份: ``(owner_strategy_id, indicator_key)``.

主策略与指标脚本用同名键 ``ema`` 时, 定义不能互相吞掉; 图上 series 键只在
同名冲突时才加 ``owner:`` 前缀(见 spec 2026-10-20-indicator-owner-identity §2.1)。
"""

from __future__ import annotations

from typing import Any

import akquant as aq
import pandas as pd
from akquant import Bar, Indicator, Strategy, merge_results, run_backtest
from akquant.indicator_identity import display_label, series_keys

SYMBOL = "OWN_ID"


def _md(start: str = "2024-01-01", n: int = 60) -> pd.DataFrame:
    closes = [10.0 + (i % 7) * 0.5 + i * 0.1 for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=n, freq="D", tz="UTC"),
            "symbol": SYMBOL,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": 100.0,
        }
    )


class MainEma(Strategy):
    """主策略: 主图红色 EMA5."""

    def on_start(self) -> None:
        """声明主图 ema."""
        self.ema = self.I(aq.EMA(5), name="ema", plot=True, pane=0, color="red")

    def on_bar(self, bar: Bar) -> None:
        """不交易."""


class Sub(Indicator):
    """指标脚本: 同名 ema(EMA50), 蓝色, 默认附图."""

    def on_start(self) -> None:
        """声明同名 ema."""
        self.ema = self.I(aq.EMA(50), name="ema", color="blue")


def _run(start: str = "2024-01-01", n: int = 60) -> Any:
    return run_backtest(
        strategy=MainEma,
        data=_md(start, n),
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        timezone="UTC",
        indicators=[Sub],
    )


# ---------------------------------------------------------------- series_keys


def test_series_keys_no_conflict_keeps_plain_key() -> None:
    """无冲突: series 键就是 indicator_key."""
    defs = [
        {"owner_strategy_id": "_default", "indicator_key": "ema"},
        {"owner_strategy_id": "sub", "indicator_key": "rsi"},
    ]
    assert series_keys(defs) == {("_default", "ema"): "ema", ("sub", "rsi"): "rsi"}


def test_series_keys_conflict_prefixes_owner() -> None:
    """同一 key 被多个 owner 使用: 全部加 owner 前缀, 其余键不变."""
    defs = [
        {"owner_strategy_id": "_default", "indicator_key": "ema"},
        {"owner_strategy_id": "sub", "indicator_key": "ema"},
        {"owner_strategy_id": "sub", "indicator_key": "rsi"},
    ]
    assert series_keys(defs) == {
        ("_default", "ema"): "_default:ema",
        ("sub", "ema"): "sub:ema",
        ("sub", "rsi"): "rsi",
    }


def test_series_keys_missing_owner_defaults() -> None:
    """缺失/空 owner 按 _default 处理, 与显式 _default 视为同一 owner."""
    defs: list[dict[str, Any]] = [
        {"indicator_key": "ema"},
        {"owner_strategy_id": "", "indicator_key": "ema"},
        {"owner_strategy_id": None, "indicator_key": "macd"},
        {"owner_strategy_id": "sub", "indicator_key": "macd"},
    ]
    assert series_keys(defs) == {
        ("_default", "ema"): "ema",
        ("_default", "macd"): "_default:macd",
        ("sub", "macd"): "sub:macd",
    }


def test_display_label_rule() -> None:
    """未写 label 且冲突 -> series 键; 写了 label 或无冲突 -> 原 display_name."""
    plain = {"owner_strategy_id": "sub", "indicator_key": "ema", "display_name": "ema"}
    assert display_label(plain, "sub:ema") == "sub:ema"
    assert display_label(plain, "ema") == "ema"
    labeled = {
        "owner_strategy_id": "sub",
        "indicator_key": "ema",
        "display_name": "EMA 50",
    }
    assert display_label(labeled, "sub:ema") == "EMA 50"
    missing = {"owner_strategy_id": "sub", "indicator_key": "ema"}
    assert display_label(missing, "sub:ema") == "sub:ema"


# ------------------------------------------------------------ 端到端: 双 ema


def test_dual_ema_definitions_keep_both_owners() -> None:
    """主策略与指标脚本同名 ema: 定义 2 行, pane/color 各自正确, owner 在第一列."""
    result = _run()
    defs = result.indicator_definitions
    assert list(defs.columns)[0] == "owner_strategy_id"
    assert len(defs) == 2
    rows = [
        (r.owner_strategy_id, r.indicator_key, int(r.pane), r.color)
        for r in defs.itertuples()
    ]
    assert rows == [("_default", "ema", 0, "red"), ("sub", "ema", 1, "blue")]
    payload_defs = result.indicator_outputs["definitions"]
    assert all("owner_strategy_id" in d for d in payload_defs)
    assert series_keys(payload_defs) == {
        ("_default", "ema"): "_default:ema",
        ("sub", "ema"): "sub:ema",
    }
    # 数据点本来就完整, 两个 owner 各一份
    points = result.indicator_df()
    assert set(points["owner_strategy_id"]) == {"_default", "sub"}


# --------------------------------------------------------- merge_results 跨段


def test_merge_results_dedups_definitions_by_owner_and_key() -> None:
    """跨段合并: 同 (owner, key) 只留一行, 不同 owner 的同名键各留一行."""
    first = _run("2024-01-01", 60)
    second = _run("2024-03-01", 60)
    merged = merge_results(first, second)
    defs = merged.indicator_outputs["definitions"]
    identities = sorted((d["owner_strategy_id"], d["indicator_key"]) for d in defs)
    assert identities == [("_default", "ema"), ("sub", "ema")]
