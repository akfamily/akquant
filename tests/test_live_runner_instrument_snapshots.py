"""live 下 instrument 快照灌入(此前实盘 get_instrument 对任何 symbol 都 KeyError)."""

from typing import Any, cast

import pytest
from akquant import AssetType, Instrument
from akquant.live._runner import LiveRunner, _instruments_to_snapshots
from akquant.strategy import Strategy


class _DummyEngine:
    def set_strategy_slots(self, slot_ids: list[str]) -> None:
        self.slot_ids = slot_ids

    def set_default_strategy_id(self, strategy_id: str) -> None:
        self.default_strategy_id = strategy_id

    def set_strategy_for_slot(self, slot_index: int, strategy: Any) -> None:
        _ = (slot_index, strategy)


class _DummyStrategy(Strategy):
    def on_bar(self, bar: Any) -> None:
        _ = bar


def _runner(instruments: list[Instrument]) -> LiveRunner:
    runner = LiveRunner.__new__(LiveRunner)
    runner.engine = cast(Any, _DummyEngine())
    runner.context = {}
    runner.instruments = instruments
    return runner


def test_maps_readable_instrument_fields() -> None:
    """Instrument 的可读字段被映射进快照."""
    snapshots = _instruments_to_snapshots(
        [
            Instrument(
                "600000.SH",
                AssetType.Stock,
                multiplier=1.0,
                margin_ratio=1.0,
                tick_size=0.01,
                lot_size=100.0,
            )
        ]
    )
    snap = snapshots["600000.SH"]
    assert snap.symbol == "600000.SH"
    assert snap.asset_type == "STOCK"
    assert snap.tick_size == pytest.approx(0.01)
    assert snap.lot_size == pytest.approx(100.0)


def test_futures_asset_type_mapped() -> None:
    """期货标的 asset_type 映射为 FUTURES."""
    snapshots = _instruments_to_snapshots(
        [Instrument("IF2601", AssetType.Futures, multiplier=300.0, margin_ratio=0.12)]
    )
    snap = snapshots["IF2601"]
    assert snap.asset_type == "FUTURES"
    assert snap.multiplier == pytest.approx(300.0)
    assert snap.margin_ratio == pytest.approx(0.12)


def test_empty_instruments_yields_empty_snapshots() -> None:
    """空标的列表得到空快照(不抛错)."""
    assert _instruments_to_snapshots([]) == {}


def test_configure_slots_populates_strategy_snapshots() -> None:
    """_configure_strategy_slots 给全部 slot 灌入快照, get_instrument 可用."""
    instruments = [
        Instrument("600000.SH", AssetType.Stock, tick_size=0.01, lot_size=100.0),
        Instrument("000012.SZ", AssetType.Stock, tick_size=0.01, lot_size=100.0),
    ]
    runner = _runner(instruments)
    primary = _DummyStrategy()
    secondary = _DummyStrategy()
    runner._configure_strategy_slots(primary, {"beta": secondary}, "alpha")

    for target in (primary, secondary):
        assert target.get_instrument("600000.SH").symbol == "600000.SH"
        assert target.get_instrument("000012.SZ").symbol == "000012.SZ"
        assert set(target.get_instruments()) == {"600000.SH", "000012.SZ"}
        assert target.get_instrument_field("600000.SH", "lot_size") == pytest.approx(
            100.0
        )


def test_unknown_symbol_still_raises() -> None:
    """未配置的标的仍按回测同口径抛 KeyError(不静默返回默认值)."""
    runner = _runner([Instrument("600000.SH", AssetType.Stock)])
    strategy = _DummyStrategy()
    runner._configure_strategy_slots(strategy, {}, "alpha")
    with pytest.raises(KeyError):
        strategy.get_instrument("999999.SH")
