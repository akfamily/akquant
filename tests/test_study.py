"""可插拔 study: 只声明指标、不交易的单元(RFC indicator-tradingview §2.4 / L3).

复用多 slot 策略拓扑, 不新建管线: ``run_backtest(studies=[...])`` 与
``run_live(studies=[...])`` 把每个 study 映射成一个 slot, 指标点带各自的
``owner_strategy_id``。``Study`` 基类把全部交易 API 换成抛错, 是"只画图不交易"
的硬保证。
"""

from __future__ import annotations

from typing import Any, List

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy, Study, StudyCannotTradeError, run_backtest

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


class RsiStudy(Study):
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
def test_study_blocks_every_trading_api(method: str, args: tuple[Any, ...]) -> None:
    """Study 上任何交易 API 都抛 StudyCannotTradeError —— 这是硬保证, 不是约定."""
    study = RsiStudy()

    with pytest.raises(StudyCannotTradeError, match=method):
        getattr(study, method)(*args)


def test_study_id_defaults_to_snake_case_class_name() -> None:
    """未显式给 study_id 时用类名的 snake_case, 稳定且可读."""

    class MACDDivergenceStudy(Study):
        pass

    class Plain(Study):
        study_id = "custom_id"

    assert RsiStudy.resolve_study_id() == "rsi_study"
    assert MACDDivergenceStudy.resolve_study_id() == "macd_divergence_study"
    assert Plain.resolve_study_id() == "custom_id"


def test_studies_param_attaches_study_as_slot_with_own_owner() -> None:
    """run_backtest(studies=[...]) 让 study 的指标点带自己的 owner, 主策略照常交易."""
    result = _run(studies=[RsiStudy])

    frame = result.indicator_df()
    assert set(frame["owner_strategy_id"]) == {"main", "rsi_study"}
    assert set(frame.loc[frame.owner_strategy_id == "rsi_study", "indicator_key"]) == {
        "rsi"
    }
    # 主策略只买不卖, result.trades(闭合回合)为空是正常的; 看订单是否成交
    orders = result.orders_df
    assert (orders["status"] == "filled").any()  # study 不影响主策略下单
    assert set(orders["owner_strategy_id"]) == {"main"}  # study 没下过单


def test_indicator_df_filters_by_owner() -> None:
    """indicator_df(owner=) 只留该 owner 的点."""
    result = _run(studies=[RsiStudy])

    only_study = result.indicator_df(owner="rsi_study")
    assert not only_study.empty
    assert set(only_study["owner_strategy_id"]) == {"rsi_study"}
    assert result.indicator_df(owner="nobody").empty


def test_studies_coexist_with_strategies_by_slot() -> None:
    """Studies 与 strategies_by_slot 可同时给, 各占各的 slot."""

    class Beta(Strategy):
        def on_bar(self, bar: Bar) -> None:
            pass

    result = _run(studies=[RsiStudy], strategies_by_slot={"beta": Beta})

    assert set(result.indicator_df()["owner_strategy_id"]) == {"main", "rsi_study"}
    assert set(result.strategy._slot_strategies) == {"beta", "rsi_study"}


def test_study_id_collision_fails_fast() -> None:
    """study_id 撞上 strategy_id 或已有 slot key 时报错, 不静默覆盖."""

    class Beta(Strategy):
        def on_bar(self, bar: Bar) -> None:
            pass

    class MainStudy(Study):
        study_id = "main"

    with pytest.raises(ValueError, match="main"):
        _run(studies=[MainStudy])
    with pytest.raises(ValueError, match="rsi_study"):
        _run(studies=[RsiStudy], strategies_by_slot={"rsi_study": Beta})


def test_studies_rejects_non_study_classes() -> None:
    """studies= 只接受 Study 子类/实例 —— 否则'不交易'的保证就没了."""

    class NotAStudy(Strategy):
        def on_bar(self, bar: Bar) -> None:
            pass

    with pytest.raises(TypeError, match="Study"):
        _run(studies=[NotAStudy])


def test_study_trading_inside_engine_loop_raises() -> None:
    """Study 在 on_bar 里误下单, 错误要冒出来而不是静默成交."""

    class SneakyStudy(Study):
        def on_bar(self, bar: Bar) -> None:
            self.buy(bar.symbol, 100)

    with pytest.raises(StudyCannotTradeError):
        _run(studies=[SneakyStudy])


def test_run_live_studies_param_reports_with_owner() -> None:
    """实盘入口同样接 studies=, 流事件里带 study 的 owner."""
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
        studies=[RsiStudy],
    )

    owners = {m["indicator"]["owner_strategy_id"] for m in messages}
    assert "rsi_study" in owners
