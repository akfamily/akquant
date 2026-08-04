"""live 的中国市场配置必须按资产类型选择.

``use_china_futures_market()`` 只配期货费率(``stock``/``fund``/``option`` 均为
``None``)，股票订单到达 Rust 撮合时会 panic:
``Stock market configuration not found but received stock order``
(``src/market/china.rs``)。回测早有分支(``backtest/engine.py``)，live 此前无条件
用期货配置。
"""

from typing import Any, List

import pandas as pd
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar
from akquant.live._runner import LiveRunner


def _instrument(symbol: str, asset_type: Any) -> Instrument:
    """构造一个标的."""
    return Instrument(
        symbol=symbol,
        asset_type=asset_type,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=1,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _runner_with(instruments: List[Instrument]) -> LiveRunner:
    """构造只带 instruments 的裸 LiveRunner."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.instruments = instruments
    return runner


def test_all_futures_uses_futures_config() -> None:
    """全期货标的沿用期货配置(保持原行为)."""
    runner = _runner_with([_instrument("IF2506", AssetType.Futures)])

    assert runner._all_instruments_are_futures() is True


def test_stock_instrument_needs_full_config() -> None:
    """含股票标的必须改用全资产配置, 否则 Rust 撮合 panic."""
    runner = _runner_with([_instrument("600000", AssetType.Stock)])

    assert runner._all_instruments_are_futures() is False


def test_mixed_instruments_need_full_config() -> None:
    """期货+股票混合时也必须用全资产配置."""
    runner = _runner_with(
        [
            _instrument("IF2506", AssetType.Futures),
            _instrument("600000", AssetType.Stock),
        ]
    )

    assert runner._all_instruments_are_futures() is False


def test_empty_instruments_keeps_legacy_behavior() -> None:
    """无标的时保持原行为(期货配置), 不改变既有语义."""
    runner = _runner_with([])

    assert runner._all_instruments_are_futures() is True


class _BuyingStrategy(Strategy):
    """会真正下单的策略: 只有下单才会走到撮合与手续费计算(panic 发生处)."""

    def __init__(self) -> None:
        """初始化成交记录."""
        self.trades: list[Any] = []

    def on_bar(self, bar: Bar) -> None:
        """首根 bar 买入一手."""
        if not self.trades and self.get_position(bar.symbol) == 0:
            self.buy(bar.symbol, 1)

    def on_trade(self, trade: Any) -> None:
        """记录成交."""
        self.trades.append(trade)


def test_stock_paper_session_places_order_without_panic() -> None:
    """含股票标的的 paper 会话必须能真正下单成交, 不 panic.

    这是**防回归的关键测试**: 上面 4 个测试只验证纯分类器
    ``_all_instruments_are_futures()``, 若 ``run()`` 不再咨询它(改回无条件调用
    ``use_china_futures_market()``), 那 4 个测试仍会全绿——而本测试会因 Rust 撮合
    层 panic 而失败(``Stock market configuration not found but received stock order``)。

    必须让策略**真正下单**: 只记录事件的策略永远走不到撮合与手续费计算, 因此
    发现不了这个缺陷(Task 4 的股票 e2e 测试就是这样才漏过去的)。
    """
    symbol = "MKTCFG_A"
    stamps = ["2023-01-03 09:30:00", "2023-01-03 10:00:00"]
    bars = [
        Bar(
            timestamp=int(pd.Timestamp(text, tz="Asia/Shanghai").value),
            open=10.0,
            high=10.5,
            low=9.5,
            close=10.0,
            volume=1000.0,
            symbol=symbol,
        )
        for text in stamps
    ]

    strategy = _BuyingStrategy()
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument(symbol, AssetType.Stock)],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars},
        cash=1_000_000.0,
        show_progress=False,
        duration="60s",
    )

    assert strategy.trades, "股票 paper 会话未产生成交, 市场配置可能又退回期货专用"
