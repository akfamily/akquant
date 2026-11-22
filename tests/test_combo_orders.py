"""多腿组合单 ``submit_combo`` / ``get_combo``: 编组、腿失败时撤同组委托、整组撤单."""

from typing import Any, Dict, List, Optional, Tuple

import akquant
import pandas as pd
import pytest
from akquant import BarBatch, Strategy

DAYS = pd.date_range("2026-11-17 15:00", periods=4, freq="B", tz="Asia/Shanghai")
NEAR, FAR, NO_DATA = "m2601", "m2605", "m2609"


def _bars(symbol: str, price: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": DAYS,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 100000,
            "symbol": symbol,
        }
    )


class _Combo(Strategy):
    """按天调用 ``on_day``; 记录组合单状态与回调里看到的状态."""

    def __init__(self) -> None:
        super().__init__()
        self.day = -1
        self.states: List[str] = []
        self.records: Dict[str, Any] = {}

    def on_bars(self, batch: BarBatch) -> None:
        self.day += 1
        self.on_day(self.day)
        if "group" in self.records:
            self.states.append(self.get_combo(self.records["group"]).state)

    def on_day(self, day: int) -> None:
        """每日回调."""


def _run(strategy: _Combo) -> _Combo:
    configs = [
        akquant.InstrumentConfig(
            symbol=s, asset_type="FUTURES", multiplier=10.0, margin_ratio=0.1
        )
        for s in (NEAR, FAR, NO_DATA)
    ]
    akquant.run_backtest(
        data={NEAR: _bars(NEAR, 3000.0), FAR: _bars(FAR, 3050.0)},
        strategy=strategy,
        config=akquant.BacktestConfig(
            strategy_config=akquant.StrategyConfig(initial_cash=1_000_000.0),
            instruments_config=configs,
        ),
        freq="1d",
        show_progress=False,
    )
    return strategy


class _Spread(_Combo):
    def on_day(self, day: int) -> None:
        if day == 0:
            self.records["group"] = self.submit_combo([(NEAR, 1), (FAR, -1)], 2)


def test_combo_fills_all_legs() -> None:
    """两腿按每组手数 × 组数下单, 全部成交后整组为 filled."""
    strategy = _run(_Spread())
    combo = strategy.get_combo(strategy.records["group"])
    assert combo.state == "filled"
    assert [(leg.symbol, leg.side, leg.quantity, leg.filled) for leg in combo.legs] == [
        (NEAR, "Buy", 2.0, 2.0),
        (FAR, "Sell", 2.0, 2.0),
    ]
    assert strategy.states[0] == "working"
    assert strategy.get_position(NEAR) == 2.0
    assert strategy.get_position(FAR) == -2.0


class _LegRejected(_Combo):
    def on_day(self, day: int) -> None:
        if day == 0:
            # 第一腿限价远离市价不会成交, 第二腿没有行情被引擎拒绝
            self.records["group"] = self.submit_combo(
                [(NEAR, 1), (NO_DATA, -1)], prices={NEAR: 1.0}
            )


def test_rejected_leg_cancels_working_legs() -> None:
    """一腿被拒: 同组未成交的腿被自动撤销, 没有成交时整组为 cancelled."""
    strategy = _run(_LegRejected())
    combo = strategy.get_combo(strategy.records["group"])
    assert combo.state == "cancelled"
    assert [leg.status for leg in combo.legs] == ["failed", "failed"]
    assert strategy.get_open_orders() == []


class _Broken(_Combo):
    def on_day(self, day: int) -> None:
        if day == 0:
            self.records["group"] = self.submit_combo(
                [(NEAR, 1), (FAR, -1)], prices={FAR: 99999.0}
            )
        elif day == 2:
            combo = self.get_combo(self.records["group"])
            self.cancel_order(combo.legs[1].order_ids[0])


def test_failed_leg_after_fill_marks_combo_broken() -> None:
    """已有腿成交后另一腿被撤: 整组为 broken, 已成交的腿保留."""
    strategy = _run(_Broken())
    combo = strategy.get_combo(strategy.records["group"])
    assert combo.state == "broken"
    assert [(leg.status, leg.filled) for leg in combo.legs] == [
        ("filled", 1.0),
        ("failed", 0.0),
    ]
    assert strategy.get_position(NEAR) == 1.0


class _CancelGroup(_Combo):
    def on_day(self, day: int) -> None:
        if day == 0:
            self.records["group"] = self.submit_combo(
                [(NEAR, 1), (FAR, -1)], prices={NEAR: 1.0, FAR: 99999.0}
            )
        elif day == 1:
            self.cancel_group(self.records["group"])


def test_cancel_group_cancels_combo_legs() -> None:
    """cancel_group(组号) 撤销整组未完成的委托."""
    strategy = _run(_CancelGroup())
    assert strategy.get_combo(strategy.records["group"]).state == "cancelled"
    assert strategy.get_open_orders() == []


class _Invalid(_Combo):
    def on_day(self, day: int) -> None:
        if day != 0:
            return
        self.records["errors"] = []
        cases: List[Tuple[List[Tuple[str, float]], float, Optional[Dict[str, float]]]]
        cases = [
            ([(NEAR, 1)], 1, None),
            ([(NEAR, 1), (NEAR, -1)], 1, None),
            ([(NEAR, 1.5), (FAR, -1)], 1, None),
            ([(NEAR, 1), (FAR, -1)], 0, None),
            ([(NEAR, 1), (FAR, -1)], 1, {"m2701": 1.0}),
        ]
        for legs, quantity, prices in cases:
            with pytest.raises(ValueError) as exc:
                self.submit_combo(legs, quantity, prices)
            self.records["errors"].append(str(exc.value))


def test_invalid_combo_arguments() -> None:
    """参数错误报出确切原因, 不报单."""
    strategy = _run(_Invalid())
    assert strategy.records["errors"] == [
        "组合单至少需要 2 条腿, 实际 1 条",
        f"组合单的腿 {NEAR} 重复",
        f"组合单的腿 {NEAR} 每组手数须为非零整数, 实际: 1.5",
        "组合单 quantity(组数) 须为正整数, 实际: 0",
        "prices 中的 m2701 不是组合单的腿",
    ]
    assert strategy.get_position(NEAR) == 0.0


def _fake_strategy(failures: Dict[str, str]) -> Any:
    """实盘下单器替身: ``failures`` 中的腿返回空回执(当场被拒或结果不可知)."""
    import threading
    from types import SimpleNamespace

    from akquant.gateway.order_receipt import OrderReceipt

    submitted: List[str] = []
    cancelled: List[str] = []

    def submit(symbol: str, qty: float, price: Any = None, tag: Any = None) -> Any:
        submitted.append(symbol)
        if symbol in failures:
            return OrderReceipt(
                group_id="", order_ids=(), legs=(), failure=failures[symbol]
            )
        return OrderReceipt(
            group_id=f"g-{symbol}", order_ids=(f"id-{symbol}",), legs=()
        )

    return SimpleNamespace(
        _order_group_lock=threading.RLock(),
        _order_group_seq=0,
        _combos={},
        _combo_order_to_group={},
        _order_group_members={},
        buy=submit,
        sell=submit,
        execution=SimpleNamespace(cancel_order=cancelled.append),
        submitted=submitted,
        cancelled=cancelled,
    )


def test_immediate_leg_failure_stops_remaining_legs() -> None:
    """实盘某腿当场被拒: 余下的腿不报, 已报的腿撤销; 结果不可知时整组为 unknown."""
    from types import SimpleNamespace

    from akquant.akquant import OrderStatus
    from akquant.strategy_combo import get_combo, process_combo_order, submit_combo

    legs = [("a", 1), ("b", -1), ("c", 1)]
    fake = _fake_strategy({"b": "rejected"})
    group = submit_combo(fake, legs)
    assert fake.submitted == ["a", "b"]
    assert fake.cancelled == ["id-a"]
    # 柜台回报撤单成功
    process_combo_order(
        fake,
        SimpleNamespace(id="id-a", status=OrderStatus.Cancelled, filled_quantity=0.0),
    )
    combo = get_combo(fake, group)
    assert combo.state == "cancelled"
    assert [leg.status for leg in combo.legs] == ["failed", "failed", "failed"]

    fake = _fake_strategy({"b": "unknown"})
    assert get_combo(fake, submit_combo(fake, legs)).state == "unknown"
