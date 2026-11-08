"""订单事件回调轨迹场景: 订单账本改造的等价性基线.

每个场景跑一次回测, 把策略收到的 on_order / on_reject / on_trade 按顺序记成轨迹;
订单号是随机 uuid, 轨迹里换成首次出现的序号。

    python tests/_order_event_scenarios.py --write   # 用当前实现重新生成基线

基线只应在旧实现上生成一次(订单账本改造的 Task 1); 改造后只比对、不重写。
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy, TimeInForce, run_from_checkpoint, save_checkpoint

FIXTURE = Path(__file__).parent / "fixtures" / "order_event_traces.json"
TZ = "Asia/Shanghai"
_TERMINAL = {"Filled", "Cancelled", "Rejected", "Expired"}


def minute_frame(symbols: List[str], days: int, seed: int) -> pd.DataFrame:
    """每天取早盘前 30 根与尾盘后 30 根分钟 bar(跨日, 能触发当日单过期)."""
    rng = random.Random(seed)
    dates = pd.bdate_range("2024-01-02", periods=days)
    offs = pd.timedelta_range("09:31:00", "11:30:00", freq="min").append(
        pd.timedelta_range("13:01:00", "15:00:00", freq="min")
    )
    offs = offs[:30].append(offs[-30:])
    idx = pd.DatetimeIndex(
        (dates.values[:, None] + offs.values[None, :]).ravel()
    ).tz_localize(TZ)
    frames = []
    for symbol in symbols:
        price = 10.0
        rows = []
        for ts in idx:
            price = max(
                1.0, round(price + rng.choice([-0.02, -0.01, 0.0, 0.01, 0.02]), 2)
            )
            volume = float(rng.choice([200, 500, 1000, 5000]))
            rows.append((ts, price, price + 0.02, price - 0.02, price, volume))
        frame = pd.DataFrame(
            rows, columns=["date", "open", "high", "low", "close", "volume"]
        )
        frame["symbol"] = symbol
        frames.append(frame)
    return pd.concat(frames).sort_values(["date", "symbol"]).reset_index(drop=True)


def _status(order: Any) -> str:
    return str(getattr(order, "status")).split(".")[-1]


class TracingStrategy(Strategy):
    """记录回调轨迹; 状态挂在实例上, 随存档 pickle."""

    def _ensure(self) -> None:
        if not hasattr(self, "trace_log"):
            self.trace_log: List[list] = []
            self.ordinals: Dict[str, int] = {}
            self.bar_no = 0
            self.open_ids: List[str] = []

    def oid(self, order_id: str) -> int:
        self._ensure()
        return self.ordinals.setdefault(str(order_id), len(self.ordinals))

    def on_start(self) -> None:
        self._ensure()

    def on_bar(self, bar: Bar) -> None:
        self._ensure()
        self.bar_no += 1
        self.act(bar)

    def act(self, bar: Bar) -> None:
        """子类实现每根 bar 的下单逻辑."""

    def react(self, order: Any) -> None:
        """子类可在 on_order 里追加动作."""

    def _record_order(self, kind: str, order: Any) -> None:
        self._ensure()
        avg = getattr(order, "average_filled_price")
        self.trace_log.append(
            [
                kind,
                self.bar_no,
                self.oid(getattr(order, "id")),
                _status(order),
                round(float(getattr(order, "filled_quantity")), 6),
                None if avg is None else round(float(avg), 6),
                str(getattr(order, "reject_reason")),
            ]
        )

    def on_order(self, order: Any) -> None:
        self._record_order("on_order", order)
        oid = str(getattr(order, "id"))
        if _status(order) in _TERMINAL:
            if oid in self.open_ids:
                self.open_ids.remove(oid)
        elif oid not in self.open_ids:
            self.open_ids.append(oid)
        self.react(order)

    def on_reject(self, order: Any) -> None:
        self._record_order("on_reject", order)

    def on_trade(self, trade: Any) -> None:
        self._ensure()
        self.trace_log.append(
            [
                "on_trade",
                self.bar_no,
                self.oid(getattr(trade, "order_id")),
                round(float(getattr(trade, "quantity")), 6),
                round(float(getattr(trade, "price")), 6),
            ]
        )


class CancelResubmit(TracingStrategy):
    def act(self, bar: Bar) -> None:
        if self.bar_no == 1:
            self.oid(self.buy("X", 100, price=5.0).primary)
        if self.bar_no in (3, 6) and self.open_ids:
            self.cancel_order(self.open_ids[0])

    def react(self, order: Any) -> None:
        if _status(order) == "Cancelled" and self.bar_no == 3:
            self.oid(self.buy("X", 100, price=5.0).primary)


class PartialFills(TracingStrategy):
    def act(self, bar: Bar) -> None:
        if self.bar_no == 1:
            self.oid(self.buy("X", 3000).primary)
        if self.bar_no == 20:
            for oid in list(self.open_ids):
                self.cancel_order(oid)


class Rejects(TracingStrategy):
    def act(self, bar: Bar) -> None:
        if self.bar_no == 1:
            self.oid(self.buy("X", 1_000_000).primary)
        if self.bar_no == 2:
            self.oid(self.buy("X", 100).primary)


class DayExpiry(TracingStrategy):
    def act(self, bar: Bar) -> None:
        if self.bar_no == 1:
            self.oid(
                self.buy("X", 100, price=5.0, time_in_force=TimeInForce.Day).primary
            )
            self.oid(self.buy("X", 100, price=5.0).primary)


class BracketOco(TracingStrategy):
    def act(self, bar: Bar) -> None:
        if self.bar_no == 1:
            self.place_bracket(
                "X",
                100,
                stop_trigger_price=bar.close - 0.05,
                take_profit_price=bar.close + 0.05,
            )


class SlotAlpha(TracingStrategy):
    def act(self, bar: Bar) -> None:
        if bar.symbol == "X" and self.bar_no in (1, 30):
            self.oid(self.buy("X", 100).primary)
        if bar.symbol == "X" and self.bar_no == 2:
            self.oid(self.buy("X", 100, price=5.0).primary)
        if bar.symbol == "X" and self.bar_no == 10 and self.open_ids:
            self.cancel_order(self.open_ids[0])


class SlotBeta(TracingStrategy):
    def act(self, bar: Bar) -> None:
        if bar.symbol == "Y" and self.bar_no in (5, 50):
            self.oid(self.buy("Y", 200).primary)
        if bar.symbol == "Y" and self.bar_no == 6:
            self.oid(self.buy("Y", 1_000_000).primary)


class RandomTrader(TracingStrategy):
    """随机下单 / 撤单, 部分回调里补单, 覆盖各类状态组合."""

    seed = 7

    def act(self, bar: Bar) -> None:
        if not hasattr(self, "rng"):
            self.rng = random.Random(self.seed)
        roll = self.rng.random()
        symbol = bar.symbol
        if roll < 0.10:
            self.oid(self.buy(symbol, 100 * self.rng.randint(1, 5)).primary)
        elif roll < 0.18:
            price = round(bar.close - 0.03 * self.rng.randint(0, 3), 2)
            tif = TimeInForce.Day if self.rng.random() < 0.5 else TimeInForce.GTC
            self.oid(self.buy(symbol, 100, price=price, time_in_force=tif).primary)
        elif roll < 0.24 and self.get_position(symbol) > 0:
            self.oid(self.sell(symbol, 100).primary)
        elif roll < 0.30 and self.open_ids:
            self.cancel_order(self.open_ids[self.rng.randrange(len(self.open_ids))])

    def react(self, order: Any) -> None:
        if not hasattr(self, "rng"):
            return
        if _status(order) == "Cancelled" and self.rng.random() < 0.5:
            self.oid(self.buy(str(getattr(order, "symbol")), 100, price=5.0).primary)


def _run(strategy: type, data: pd.DataFrame, symbols: List[str], **kwargs: Any) -> Any:
    return aq.run_backtest(
        data=data, strategy=strategy, symbols=symbols, show_progress=False, **kwargs
    )


def _scenario_cancel_resubmit() -> object:
    return _run(CancelResubmit, minute_frame(["X"], 1, 1), ["X"]).strategy.trace_log


def _scenario_partial_fills() -> object:
    r = _run(PartialFills, minute_frame(["X"], 1, 2), ["X"], volume_limit_pct=0.1)
    return r.strategy.trace_log


def _scenario_rejects() -> object:
    r = _run(Rejects, minute_frame(["X"], 1, 3), ["X"], initial_cash=100_000.0)
    return r.strategy.trace_log


def _scenario_day_expiry() -> object:
    return _run(DayExpiry, minute_frame(["X"], 2, 4), ["X"]).strategy.trace_log


def _scenario_bracket_oco() -> object:
    return _run(BracketOco, minute_frame(["X"], 2, 5), ["X"]).strategy.trace_log


def _scenario_multi_slot() -> object:
    r = aq.run_backtest(
        data=minute_frame(["X", "Y"], 2, 6),
        strategy=SlotAlpha,
        strategy_id="alpha",
        strategies_by_slot={"beta": SlotBeta},
        symbols=["X", "Y"],
        initial_cash=200_000.0,
        show_progress=False,
    )
    # run_backtest 把槽位策略实例挂在主策略的 _slot_strategies 上(backtest/engine.py)
    main: Any = r.strategy
    beta = main._slot_strategies["beta"]
    return {"alpha": main.trace_log, "beta": beta.trace_log}


def _scenario_random() -> object:
    r = _run(
        RandomTrader, minute_frame(["X", "Y"], 3, 8), ["X", "Y"], volume_limit_pct=0.2
    )
    return r.strategy.trace_log


def _scenario_random_checkpoint() -> object:
    data = minute_frame(["X", "Y"], 3, 9)
    split = len(data) // 2
    first = _run(RandomTrader, data.iloc[:split], ["X", "Y"], volume_limit_pct=0.2)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = str(Path(tmp) / "ckpt.pkl")
        save_checkpoint(first.engine, first.strategy, ckpt)  # type: ignore[arg-type]
        resumed = run_from_checkpoint(
            checkpoint_path=ckpt,
            data=data.iloc[split:],
            symbols=["X", "Y"],
            volume_limit_pct=0.2,
            show_progress=False,
        )
    final: Any = resumed.strategy
    return final.trace_log


SCENARIOS: Dict[str, Callable[[], object]] = {
    "cancel_resubmit": _scenario_cancel_resubmit,
    "partial_fills": _scenario_partial_fills,
    "rejects": _scenario_rejects,
    "day_expiry": _scenario_day_expiry,
    "bracket_oco": _scenario_bracket_oco,
    "multi_slot": _scenario_multi_slot,
    "random": _scenario_random,
    "random_checkpoint": _scenario_random_checkpoint,
}


def run_scenario(name: str) -> object:
    # 经 JSON 往返, 与基线文件里的类型一致(tuple→list 等)
    return json.loads(json.dumps(SCENARIOS[name]()))


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args != ["--write"]:
        print("usage: python tests/_order_event_scenarios.py --write")
        return 2
    traces = {name: run_scenario(name) for name in sorted(SCENARIOS)}
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps(traces, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
