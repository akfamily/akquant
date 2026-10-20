"""指标脚本: 只声明指标、不交易的单元(``aq.Indicator``).

复用多 slot 策略拓扑, 不新建管线: ``run_backtest(indicators=[...])`` 与
``run_live(indicators=[...])`` 把每个指标脚本映射成一个 slot, 指标点带各自的
``owner_strategy_id``。``Indicator`` 基类把全部交易 API 换成抛错, 是"只画图不交易"
的硬保证。
"""

from __future__ import annotations

from typing import Any, List

import akquant as aq
import pandas as pd
import pytest
from akquant import (
    Bar,
    Indicator,
    IndicatorCannotTradeError,
    Strategy,
    run_backtest,
)

SYMBOL = "STUDY"


def _md(n: int = 10) -> pd.DataFrame:
    closes = [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0, 16.0, 15.0][:n]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "symbol": SYMBOL,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": 100.0,
        }
    )


class RsiIndicator(Indicator):
    """只画 RSI."""

    def on_start(self) -> None:
        """声明一个副图 RSI."""
        self.rsi = self.I(aq.RSI(3), name="rsi", pane=1)


class MainStrategy(Strategy):
    """会交易的主策略."""

    def on_start(self) -> None:
        """声明主图 SMA."""
        self.sma = self.I(aq.SMA(2), name="sma", pane=0)

    def on_bar(self, bar: Bar) -> None:
        """SMA 抬头且空仓时买入."""
        prev, now = self.sma[1], self.sma[0]
        if prev is not None and now is not None and now > prev:
            if self.get_position(bar.symbol) == 0:
                self.buy(bar.symbol, 100)


def _run(**kwargs: Any) -> Any:
    return run_backtest(
        strategy=MainStrategy,
        data=_md(),
        symbols=[SYMBOL],
        initial_cash=1e5,
        show_progress=False,
        timezone="UTC",
        strategy_id="main",
        **kwargs,
    )


_TRADING_API = [
    ("buy", (SYMBOL, 100)),
    ("sell", (SYMBOL, 100)),
    ("short", (SYMBOL, 100)),
    ("cover", (SYMBOL, 100)),
    ("submit_order", ()),
    ("place_oco", ()),
    ("place_bracket", ()),
    ("place_trailing_stop", ()),
    ("place_trailing_stop_limit", ()),
    ("order_target", (SYMBOL, 100)),
    ("order_target_value", (SYMBOL, 1000.0)),
    ("order_target_percent", (SYMBOL, 0.5)),
    ("rebalance_weights", ({SYMBOL: 1.0},)),
    ("rebalance_positions", ({SYMBOL: 100},)),
    ("rebalance_to_topn", ({SYMBOL: 1.0},)),
    ("close_position", (SYMBOL,)),
    ("cancel_order", ("oid",)),
    ("cancel_group", ("gid",)),
    ("cancel_all_orders", ()),
]


@pytest.mark.parametrize("method,args", _TRADING_API, ids=[m for m, _ in _TRADING_API])
def test_indicator_blocks_every_trading_api(method: str, args: tuple[Any, ...]) -> None:
    """Indicator 上任何交易 API 都抛 IndicatorCannotTradeError, 是硬保证不是约定."""
    script = RsiIndicator()

    with pytest.raises(IndicatorCannotTradeError, match=method):
        getattr(script, method)(*args)


def test_indicator_id_defaults_to_snake_case_class_name() -> None:
    """未显式给 id 时用类名的 snake_case, 稳定且可读."""

    class MACDDivergenceIndicator(Indicator):
        pass

    class Plain(Indicator, id="custom_id"):
        pass

    assert RsiIndicator.resolve_id() == "rsi_indicator"
    assert MACDDivergenceIndicator.resolve_id() == "macd_divergence_indicator"
    assert Plain.resolve_id() == "custom_id"


def test_indicators_param_attaches_indicator_as_slot_with_own_owner() -> None:
    """run_backtest(indicators=[...]) 让指标点带自己的 owner, 主策略照常交易."""
    result = _run(indicators=[RsiIndicator])

    frame = result.indicator_df()
    assert set(frame["owner_strategy_id"]) == {"main", "rsi_indicator"}
    assert set(
        frame.loc[frame.owner_strategy_id == "rsi_indicator", "indicator_key"]
    ) == {"rsi"}
    # 主策略只买不卖, result.trades(闭合回合)为空是正常的; 看订单是否成交
    orders = result.orders_df
    assert (orders["status"] == "filled").any()  # 指标脚本不影响主策略下单
    assert set(orders["owner_strategy_id"]) == {"main"}  # 指标脚本没下过单


def test_indicator_df_filters_by_owner() -> None:
    """indicator_df(owner=) 只留该 owner 的点."""
    result = _run(indicators=[RsiIndicator])

    only_indicator = result.indicator_df(owner="rsi_indicator")
    assert not only_indicator.empty
    assert set(only_indicator["owner_strategy_id"]) == {"rsi_indicator"}
    assert result.indicator_df(owner="nobody").empty


def test_indicators_coexist_with_strategies_by_slot() -> None:
    """Indicators 与 strategies_by_slot 可同时给, 各占各的 slot."""

    class Beta(Strategy):
        def on_bar(self, bar: Bar) -> None:
            pass

    result = _run(indicators=[RsiIndicator], strategies_by_slot={"beta": Beta})

    assert set(result.indicator_df()["owner_strategy_id"]) == {"main", "rsi_indicator"}
    assert set(result.strategy._slot_strategies) == {"beta", "rsi_indicator"}


def test_id_collision_fails_fast() -> None:
    """脚本 id 撞上 strategy_id 或已有 slot key 时报错, 不静默覆盖."""

    class Beta(Strategy):
        def on_bar(self, bar: Bar) -> None:
            pass

    class MainIndicator(Indicator, id="main"):
        pass

    with pytest.raises(ValueError, match="main"):
        _run(indicators=[MainIndicator])
    with pytest.raises(ValueError, match="rsi_indicator"):
        _run(indicators=[RsiIndicator], strategies_by_slot={"rsi_indicator": Beta})


def test_indicators_rejects_plain_strategies() -> None:
    """indicators= 只接受 Indicator 子类/实例 —— 否则'不交易'的保证就没了."""

    class NotAnIndicator(Strategy):
        def on_bar(self, bar: Bar) -> None:
            pass

    with pytest.raises(TypeError, match="Indicator"):
        _run(indicators=[NotAnIndicator])


def test_indicator_trading_inside_engine_loop_raises() -> None:
    """指标脚本在 on_bar 里误下单, 错误要冒出来而不是静默成交."""

    class SneakyIndicator(Indicator):
        def on_bar(self, bar: Bar) -> None:
            self.buy(bar.symbol, 100)

    with pytest.raises(IndicatorCannotTradeError):
        _run(indicators=[SneakyIndicator])


def test_run_live_indicators_param_reports_with_owner() -> None:
    """实盘入口同样接 indicators=, 流事件里带指标脚本的 owner."""
    from akquant import run_live
    from akquant.akquant import AssetType, Instrument

    messages: List[Any] = []

    def on_event(event: Any) -> None:
        if aq.is_indicator_stream_event(event):
            message = aq.to_indicator_message(event)
            if message is not None and message["type"] == "point":
                messages.append(message)

    base = int(pd.Timestamp("2024-03-01 09:30", tz="Asia/Shanghai").value)
    bars = [
        Bar(
            timestamp=base + i * 60_000_000_000,
            open=c,
            high=c + 0.5,
            low=c - 0.5,
            close=float(c),
            volume=100.0,
            symbol="600000",
        )
        for i, c in enumerate([10, 11, 12, 11, 13, 14, 13, 15])
    ]
    run_live(
        strategy_cls=MainStrategy,
        instruments=[
            Instrument(
                symbol="600000",
                asset_type=AssetType.Stock,
                multiplier=1.0,
                margin_ratio=1.0,
                tick_size=0.01,
                lot_size=100,
                option_type=None,
                strike_price=None,
                expiry_date=None,
            )
        ],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars},
        cash=1e6,
        show_progress=False,
        on_event=on_event,
        duration="30s",
        strategy_id="main",
        indicators=[RsiIndicator],
    )

    owners = {m["indicator"]["owner_strategy_id"] for m in messages}
    assert "rsi_indicator" in owners


def test_indicator_class_keywords_are_recorded() -> None:
    """title/overlay/id 写在类定义行, 由 __init_subclass__ 接收."""

    class DualMA(Indicator, title="双均线", overlay=True, id="dual_ma_x"):
        pass

    assert DualMA.resolve_title() == "双均线"
    assert DualMA.resolve_overlay() is True
    assert DualMA.resolve_id() == "dual_ma_x"


def test_indicator_defaults_title_classname_id_snake_overlay_false() -> None:
    """省略时: title 为类名, id 为 snake_case, overlay 为 False."""

    class MACDDivergence(Indicator):
        pass

    assert MACDDivergence.resolve_title() == "MACDDivergence"
    assert MACDDivergence.resolve_id() == "macd_divergence"
    assert MACDDivergence.resolve_overlay() is False


def test_title_and_id_not_inherited_but_overlay_is() -> None:
    """title/id 属于具体类(否则子类与父类撞 id); overlay 是画法, 可继承."""

    class Base(Indicator, title="基类", id="base_x", overlay=True):
        pass

    class Child(Base):
        pass

    assert Child.resolve_title() == "Child"
    assert Child.resolve_id() == "child"
    assert Child.resolve_overlay() is True


def test_unknown_class_keyword_raises_at_definition() -> None:
    """拼错的类定义行关键字在定义时就报错."""
    with pytest.raises(TypeError):

        class Bad(Indicator, overlap=True):  # type: ignore[call-arg]
            pass


def test_overlay_must_be_bool() -> None:
    """类定义行的 overlay 必须是 bool."""
    with pytest.raises(TypeError, match="overlay"):

        class Bad(Indicator, overlay="yes"):  # type: ignore[arg-type]
            pass


def test_strategy_accepts_title_and_id_but_rejects_overlay() -> None:
    """策略可写 title/id, 但 overlay 只属于指标脚本."""

    class Trend(Strategy, title="趋势策略", id="trend"):
        pass

    assert Trend.resolve_title() == "趋势策略"
    assert Trend.resolve_id() == "trend"

    class Plain(Strategy):
        pass

    assert Plain.resolve_id() is None  # 策略未写 id 沿用 "_default", 不能变成类名

    with pytest.raises(TypeError, match="overlay"):

        class Bad(Strategy, overlay=True):  # type: ignore[call-arg]
            pass


@pytest.mark.parametrize("bad", ["", "   ", 123])
def test_title_and_id_must_be_non_empty_str(bad: Any) -> None:
    """title/id 给了就必须是非空字符串."""
    with pytest.raises((TypeError, ValueError)):

        class BadTitle(Indicator, title=bad):
            pass

    with pytest.raises((TypeError, ValueError)):

        class BadId(Indicator, id=bad):
            pass


def test_strategy_class_id_feeds_strategy_id() -> None:
    """类定义行的 id 等价于 strategy_id; 两处都给且不一致时报错."""
    from akquant.strategy import reconcile_strategy_id

    class Trend(Strategy, id="trend"):
        pass

    class Plain(Strategy):
        pass

    assert reconcile_strategy_id(Trend, None) == "trend"
    assert reconcile_strategy_id(Trend(), "trend") == "trend"
    assert reconcile_strategy_id(Plain, "x") == "x"
    assert reconcile_strategy_id(Plain, None) is None
    assert reconcile_strategy_id(None, "x") == "x"
    with pytest.raises(ValueError, match="trend"):
        reconcile_strategy_id(Trend, "other")


def _plotting_trend() -> Any:
    class Trend(Strategy, id="trend"):
        """类定义行带 id 的策略, 画一条 SMA 以便按 owner 观察."""

        def on_start(self) -> None:
            """声明主图 SMA."""
            self.sma = self.I(aq.SMA(2), name="sma", pane=0)

    return Trend


def test_run_backtest_rejects_conflicting_strategy_id() -> None:
    """run_backtest: 类定义行 id 与 strategy_id 不一致时报错."""
    Trend = _plotting_trend()
    with pytest.raises(ValueError, match="trend"):
        run_backtest(
            strategy=Trend,
            data=_md(),
            symbols=[SYMBOL],
            show_progress=False,
            timezone="UTC",
            strategy_id="other",
        )


def test_run_backtest_class_id_becomes_owner() -> None:
    """run_backtest: 不传 strategy_id 时, 类定义行的 id 成为指标 owner."""
    Trend = _plotting_trend()
    result = run_backtest(
        strategy=Trend,
        data=_md(),
        symbols=[SYMBOL],
        show_progress=False,
        timezone="UTC",
    )
    assert set(result.indicator_df()["owner_strategy_id"]) == {"trend"}


def test_run_live_rejects_conflicting_strategy_id() -> None:
    """run_live: 类定义行 id 与 strategy_id 不一致时报错."""
    from akquant import run_live

    Trend = _plotting_trend()
    with pytest.raises(ValueError, match="trend"):
        run_live(
            strategy_cls=Trend,
            instruments=[],
            broker="replay",
            trading_mode="paper",
            strategy_id="other",
        )
