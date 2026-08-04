"""live 的中国市场配置必须按资产类型选择.

``use_china_futures_market()`` 只配期货费率(``stock``/``fund``/``option`` 均为
``None``)，股票订单到达 Rust 撮合时会 panic:
``Stock market configuration not found but received stock order``
(``src/market/china.rs``)。回测早有分支(``backtest/engine.py``)，live 此前无条件
用期货配置。
"""

from typing import Any, List

from akquant import AssetType, Instrument
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


class _RecordingEngine:
    """记录市场配置调用的假引擎."""

    def __init__(self) -> None:
        """初始化调用记录."""
        self.calls: List[str] = []

    def use_china_futures_market(self) -> None:
        """记录期货配置调用."""
        self.calls.append("futures")

    def use_china_market(self) -> None:
        """记录全资产配置调用."""
        self.calls.append("all")


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
