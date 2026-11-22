"""实盘策略状态持久化: run_live(state_file=...) 保存与恢复 state_variables."""

import json
from pathlib import Path
from typing import Any, List

import akquant
import pandas as pd
import pytest
from akquant import AssetType, Bar, Instrument, Strategy, run_live
from akquant.live._state_store import LiveStateStore

SYMBOL = "600000"
N_BARS = 6


class _Holder:
    state_variables = ("count", "phase")

    def __init__(self, count: Any = 0, phase: Any = "idle") -> None:
        self.count = count
        self.phase = phase


def test_store_round_trip_by_slot(tmp_path: Path) -> None:
    """按槽位保存与恢复; 文件中缺少的变量保持初始值, 多余的槽位忽略."""
    path = tmp_path / "state.json"
    store = LiveStateStore(path, {"a": _Holder(3, "long"), "b": _Holder(7)})
    store.restore()
    assert store.save()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["slots"] == {
        "a": {"count": 3, "phase": "long"},
        "b": {"count": 7, "phase": "idle"},
    }
    data["slots"]["a"].pop("phase")
    data["slots"]["gone"] = {"count": 1}
    path.write_text(json.dumps(data), encoding="utf-8")
    fresh = {"a": _Holder(), "b": _Holder()}
    LiveStateStore(path, fresh).restore()
    assert (fresh["a"].count, fresh["a"].phase) == (3, "idle")
    assert (fresh["b"].count, fresh["b"].phase) == (7, "idle")


def test_store_does_not_save_before_restore(tmp_path: Path) -> None:
    """恢复完成前(启动失败)不保存, 原文件不被初始值覆盖."""
    path = tmp_path / "state.json"
    path.write_text('{"slots": {"a": {"count": 9}}}', encoding="utf-8")
    assert not LiveStateStore(path, {"a": _Holder()}).save()
    assert json.loads(path.read_text(encoding="utf-8"))["slots"]["a"]["count"] == 9


def test_store_validation_and_unserializable_values(tmp_path: Path) -> None:
    """属性不存在或不能写入 JSON 时启动校验报错; 运行中不能写入时保留原文件."""
    path = tmp_path / "state.json"
    holder = _Holder()
    del holder.phase
    with pytest.raises(ValueError) as exc:
        LiveStateStore(path, {"a": holder}).validate()
    assert str(exc.value) == (
        "策略 a(_Holder) 的 state_variables 中的 phase 不是策略属性"
        "(须在 __init__ 或 on_start 中赋值)"
    )
    with pytest.raises(ValueError, match="^策略 a 的状态变量 count 不能写入 JSON"):
        LiveStateStore(path, {"a": _Holder(count={1, 2})}).validate()

    holder = _Holder()
    store = LiveStateStore(path, {"a": holder})
    store.restore()
    assert store.save()
    holder.count = object()
    assert not store.save()
    assert json.loads(path.read_text(encoding="utf-8"))["slots"]["a"]["count"] == 0


def _bars() -> List[Bar]:
    idx = pd.date_range(
        "2024-01-02 09:31", periods=N_BARS, freq="1min", tz="Asia/Shanghai"
    )
    return [Bar(int(ts.value), 10.0, 10.0, 10.0, 10.0, 1e6, SYMBOL) for ts in idx]


class _Counter(Strategy):
    """首根 bar 买入一次; 计数 bar; on_start 里把状态设为初始值(恢复排在它之后)."""

    state_variables = ("bars_seen", "entered")

    def on_start(self) -> None:
        self.bars_seen = 0
        self.entered = False
        self.first_seen: List[Any] = []

    def on_bar(self, bar: Bar) -> None:
        if not self.first_seen:
            self.first_seen.append((self.bars_seen, self.entered))
        self.bars_seen += 1
        if not self.entered:
            self.buy(SYMBOL, 100)
            self.entered = True


def _run(path: Path) -> _Counter:
    strategy = _Counter()
    run_live(
        strategy_cls=strategy,
        instruments=[
            Instrument(
                symbol=SYMBOL,
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
        gateway_options={"bars": _bars(), "freq": "1min"},
        show_progress=False,
        duration="60s",
        state_file=str(path),
    )
    return strategy


def test_run_live_restores_state_after_restart(tmp_path: Path) -> None:
    """第二次运行在 on_start 之后恢复状态: 不重复开仓, 计数接着上次."""
    path = tmp_path / "live" / "state.json"
    first = _run(path)
    assert first.first_seen == [(0, False)]
    saved = json.loads(path.read_text(encoding="utf-8"))["slots"]["_default"]
    assert saved == {"bars_seen": N_BARS, "entered": True}

    second = _run(path)
    assert second.first_seen == [(N_BARS, True)]
    assert second.get_position(SYMBOL) == 0.0
    saved = json.loads(path.read_text(encoding="utf-8"))["slots"]["_default"]
    assert saved["bars_seen"] == 2 * N_BARS


def test_save_state_without_state_file_returns_false() -> None:
    """未启用状态文件(回测)时 save_state 返回 False."""
    results: List[bool] = []

    class _S(Strategy):
        def on_bar(self, bar: Bar) -> None:
            results.append(self.save_state())

    akquant.run_backtest(
        data=pd.DataFrame(
            {
                "date": pd.date_range(
                    "2024-01-02 15:00", periods=2, freq="B", tz="Asia/Shanghai"
                ),
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 100.0,
                "symbol": SYMBOL,
            }
        ),
        strategy=_S,
        show_progress=False,
    )
    assert results == [False, False]
