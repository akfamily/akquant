"""成交量限制 volume_limit_pct 的下发与截断规则 (#409)."""

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from akquant import (
    BacktestConfig,
    Bar,
    CurrentClose,
    Strategy,
    StrategyConfig,
    run_backtest,
    run_from_checkpoint,
    save_checkpoint,
)
from akquant.backtest.engine import _resolve_warm_start_volume_limit

SYMBOL = "VOLCAP"


def _bars(volume: float, start: str = "2023-01-03") -> list[Bar]:
    """三根北京时间 10:00(连续竞价)的 bar, 价格恒为 10, 成交量固定."""
    stamps = pd.date_range(start=start, periods=3, freq="D", tz="Asia/Shanghai")
    return [
        Bar(
            timestamp=(ts + pd.Timedelta(hours=10)).value,
            open=10.0,
            high=10.0,
            low=10.0,
            close=10.0,
            volume=volume,
            symbol=SYMBOL,
        )
        for ts in stamps
    ]


class _BuyOnce(Strategy):
    """首根 bar 买 3000 股, 之后不再下单."""

    def __init__(self) -> None:
        super().__init__()
        self.sent = False

    def on_bar(self, bar: Bar) -> None:
        if not self.sent:
            self.sent = True
            self.buy(bar.symbol, 3000)


def _fills(**kwargs: Any) -> list[float]:
    result = run_backtest(
        data=_bars(10_000.0),
        strategy=_BuyOnce,
        symbols=SYMBOL,
        initial_cash=1_000_000.0,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        fill_policy=CurrentClose(),
        lot_size=100,
        show_progress=False,
        **kwargs,
    )
    return [float(trade.quantity) for trade in result.executions]


def test_default_is_unlimited() -> None:
    """不传 volume_limit_pct: 不限制, 一次成交全部 3000 股."""
    assert _fills() == [3000.0]


def test_zero_is_unlimited() -> None:
    """0 与 None 一样表示不限制."""
    assert _fills(volume_limit_pct=0.0) == [3000.0]


def test_explicit_quarter_is_applied() -> None:
    """显式 0.25 必须生效: 原来与默认值相同就被跳过 (#409)."""
    assert _fills(volume_limit_pct=0.25) == [2500.0, 500.0]


def test_capped_quantity_rounds_down_to_lot() -> None:
    """0.2499 * 10000 = 2499 股, 按 lot_size=100 向下取整为 2400."""
    assert _fills(volume_limit_pct=0.2499) == [2400.0, 600.0]


def test_strategy_config_quarter_is_applied() -> None:
    """通过 BacktestConfig.strategy_config 传 0.25 同样生效."""
    config = BacktestConfig(strategy_config=StrategyConfig(volume_limit_pct=0.25))
    assert _fills(config=config) == [2500.0, 500.0]


def test_strategy_config_default_is_unlimited() -> None:
    """StrategyConfig 的 volume_limit_pct 默认 0, 即不限制."""
    assert StrategyConfig().volume_limit_pct == 0.0


def test_broker_profile_quarter_is_applied() -> None:
    """cn_stock_t1_low_fee 模板的 0.25 原来从未生效."""
    assert _fills(broker_profile="cn_stock_t1_low_fee") == [2500.0, 500.0]


def test_resolved_config_records_unlimited_by_default() -> None:
    """已解析配置里默认记录 volume_limit_pct == 0, 而不是旧的 0.25."""
    result = run_backtest(
        data=_bars(10_000.0),
        strategy=_BuyOnce,
        symbols=SYMBOL,
        show_progress=False,
    )
    resolved = result.resolved_config
    assert resolved is not None
    assert resolved["volume_limit_pct"] == 0.0


class _WarmBuyOnce(Strategy):
    """只在热启动恢复后的第一根 bar 买 3000 股."""

    def __init__(self) -> None:
        super().__init__()
        self.sent = False

    def on_bar(self, bar: Bar) -> None:
        if self.is_restored and not self.sent:
            self.sent = True
            self.buy(bar.symbol, 3000)


def test_warm_start_applies_explicit_quarter(tmp_path: Path) -> None:
    """热启动显式传 0.25 必须生效(原来同样被 != 0.25 跳过)."""
    checkpoint = tmp_path / "volume.pkl"
    phase1 = run_backtest(
        data=_bars(10_000.0),
        strategy=_WarmBuyOnce,
        symbols=SYMBOL,
        lot_size=100,
        show_progress=False,
    )
    save_checkpoint(phase1.engine, phase1.strategy, str(checkpoint))  # type: ignore[arg-type]
    phase2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=_bars(10_000.0, start="2023-02-01"),
        symbols=SYMBOL,
        show_progress=False,
        volume_limit_pct=0.25,
    )
    # 恢复后首根下单, 按默认成交策略在后两根成交,
    # 每根最多 0.25 * 10000 = 2500 股, 3000 股分两根成交 2500 + 500
    quantities = [float(trade.quantity) for trade in phase2.executions]
    assert quantities == [2500.0, 500.0]


def _save_quarter_checkpoint(tmp_path: Path) -> Path:
    """跑第一段 (显式 0.25) 并连同已解析配置存档, 返回存档路径."""
    checkpoint = tmp_path / "volume_e2e.pkl"
    phase1 = run_backtest(
        data=_bars(10_000.0),
        strategy=_WarmBuyOnce,
        symbols=SYMBOL,
        lot_size=100,
        volume_limit_pct=0.25,
        show_progress=False,
    )
    save_checkpoint(
        phase1.engine,  # type: ignore[arg-type]
        phase1.strategy,
        str(checkpoint),
        config=phase1,
    )
    return checkpoint


def _resume_fills(checkpoint: Path) -> list[float]:
    """从存档续跑第二段, 不传 volume_limit_pct, 返回各笔成交量."""
    phase2 = run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=_bars(10_000.0, start="2023-02-01"),
        symbols=SYMBOL,
        show_progress=False,
    )
    return [float(trade.quantity) for trade in phase2.executions]


def test_warm_start_inherits_explicit_quarter_from_new_snapshot(
    tmp_path: Path,
) -> None:
    """新快照带 volume_limit_explicit 标记: 续跑不传量限也继承存档里的 0.25."""
    checkpoint = _save_quarter_checkpoint(tmp_path)
    with open(checkpoint, "rb") as f:
        snapshot = pickle.load(f)
    assert snapshot["snapshot_features"]["volume_limit_explicit"] is True
    assert snapshot["backtest_config"]["volume_limit_pct"] == 0.25
    assert _resume_fills(checkpoint) == [2500.0, 500.0]


def test_warm_start_legacy_snapshot_quarter_means_unlimited(tmp_path: Path) -> None:
    """旧快照(无标记)里的 0.25 是当年的默认值, 续跑按不限制, 一次成交 3000 股."""
    checkpoint = _save_quarter_checkpoint(tmp_path)
    with open(checkpoint, "rb") as f:
        snapshot = pickle.load(f)
    del snapshot["snapshot_features"]["volume_limit_explicit"]
    with open(checkpoint, "wb") as f:
        pickle.dump(snapshot, f)
    assert _resume_fills(checkpoint) == [3000.0]


def test_resolve_warm_start_volume_limit_precedence() -> None:
    """优先级: 显式入参 > strategy_config > 快照记录."""
    restored = {"volume_limit_pct": 0.25}
    features = {"volume_limit_explicit": True}
    config = StrategyConfig(volume_limit_pct=0.1)
    assert _resolve_warm_start_volume_limit(None, config, restored, features) == 0.1
    assert _resolve_warm_start_volume_limit(0.3, config, restored, features) == 0.3
    # strategy_config 的 0 同样是显式值, 不会被快照里的 0.25 顶掉
    zero = StrategyConfig(volume_limit_pct=0.0)
    assert _resolve_warm_start_volume_limit(None, zero, restored, features) == 0.0


def test_legacy_snapshot_default_quarter_means_unlimited() -> None:
    """旧快照(无 volume_limit_explicit 标记)里的 0.25 是当年的默认值, 实际不限制."""
    assert (
        _resolve_warm_start_volume_limit(None, None, {"volume_limit_pct": 0.25}, {})
        == 0.0
    )
    assert (
        _resolve_warm_start_volume_limit(
            None, None, {"volume_limit_pct": 0.25}, {"volume_limit_explicit": True}
        )
        == 0.25
    )
    assert (
        _resolve_warm_start_volume_limit(
            0.1, None, {"volume_limit_pct": 0.25}, {"volume_limit_explicit": True}
        )
        == 0.1
    )
    assert _resolve_warm_start_volume_limit(None, None, {}, {}) == 0.0


class _BuyOnceRecordingOrders(_BuyOnce):
    """同 _BuyOnce, 另记下每次 on_order 回调的 (状态, 已成交量)."""

    def __init__(self) -> None:
        super().__init__()
        self.order_events: list[tuple[str, float]] = []

    def on_order(self, order: Any) -> None:
        self.order_events.append((str(order.status), float(order.filled_quantity)))


def test_execution_reject_after_partial_fills_keeps_filled_quantity() -> None:
    """跨 bar 部分成交后剩余部分验资被拒: 已成交量保留, 剩余部分撤销.

    3000 股 @10 下单时资金 31000 够用; 之后价格涨到 12, 每根只能成交 1000 股,
    前两根成交 2000 股后剩 7000 元, 第三根 12000 元的成交在成交时被拒.
    """
    stamps = pd.date_range(start="2023-01-03", periods=4, freq="D", tz="Asia/Shanghai")
    bars = [
        Bar(
            timestamp=(ts + pd.Timedelta(hours=10)).value,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=10_000.0,
            symbol=SYMBOL,
        )
        for ts, price in zip(stamps, [10.0, 12.0, 12.0, 12.0])
    ]
    result = run_backtest(
        data=bars,
        strategy=_BuyOnceRecordingOrders,
        symbols=SYMBOL,
        initial_cash=31_000.0,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        lot_size=100,
        volume_limit_pct=0.1,
        show_progress=False,
    )

    assert [float(t.quantity) for t in result.executions] == [1000.0, 1000.0]
    orders = result.orders_df
    assert len(orders) == 1
    row = orders.iloc[0]
    assert row["status"] == "cancelled"
    assert float(row["filled_quantity"]) == 2000.0
    assert float(row["avg_price"]) == 12.0
    assert "Insufficient margin at execution" in str(row["reject_reason"])

    strategy = result.strategy
    assert strategy is not None
    assert strategy.order_events[-1] == ("OrderStatus.Cancelled", 2000.0)
