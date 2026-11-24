"""on_start 里读账户: 引擎上下文尚未绑定, 读数来自开跑前的账户视图."""

from pathlib import Path
from typing import Any, Dict, List

import akquant as aq
import numpy as np
import pandas as pd
import pytest
from akquant import AssetType, Bar, Instrument, Strategy, run_live
from akquant.config import RiskConfig

SYMBOL = "AAA"


def _frame(periods: int = 20) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=periods, freq="B", tz="Asia/Shanghai")
    close = np.linspace(10.0, 12.0, periods)
    return pd.DataFrame(
        {
            "date": index,
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": 1e6,
            "symbol": SYMBOL,
        }
    )


class _Recorder(Strategy):
    """在各生命周期回调里记下账户读数; 第一根 bar 买入 1000 股."""

    def __init__(self) -> None:
        self.seen: Dict[str, Dict[str, Any]] = {}

    def _record(self, where: str) -> None:
        self.seen[where] = {
            "ctx_bound": self.ctx is not None,
            "account": dict(self.get_account()),
            "cash": self.cash,
            "equity": self.equity,
            "buying_power": self.buying_power,
            "position": self.get_position(SYMBOL),
            "available": self.get_available_position(SYMBOL),
            "positions": dict(self.positions),
        }

    def on_resume(self) -> None:
        self._record("on_resume")

    def on_start(self) -> None:
        self._record("on_start")

    def on_bar(self, bar: Bar) -> None:
        if "first_bar" not in self.seen:
            self._record("first_bar")
        if self.get_position(bar.symbol) == 0:
            self.buy(bar.symbol, 1000)


def _run(**kwargs: Any) -> _Recorder:
    result = aq.run_backtest(
        data=_frame(),
        strategy=_Recorder,
        symbols=[SYMBOL],
        show_progress=False,
        **kwargs,
    )
    return result.strategy  # type: ignore[return-value]


def test_on_start_reads_initial_cash_from_get_account() -> None:
    """全新回测: on_start 里 get_account() 给出初始资金, 不再报 Context not ready."""
    strategy = _run(initial_cash=123456.0)

    start = strategy.seen["on_start"]
    assert start["ctx_bound"] is False
    account = start["account"]
    assert account["cash"] == 123456.0
    assert account["equity"] == 123456.0
    assert account["available_cash"] == 123456.0
    assert account["market_value"] == 0.0
    assert account["margin"] == 0.0
    assert account["frozen_cash"] == 0.0
    assert account["account_mode"] == "cash"


def test_on_start_account_matches_first_bar_account() -> None:
    """开跑前的账户读数与第一根 bar 下单前的读数逐键一致."""
    strategy = _run(initial_cash=123456.0)

    assert strategy.seen["on_start"]["account"] == strategy.seen["first_bar"]["account"]


def test_on_start_cash_and_equity_properties_are_initial_cash() -> None:
    """on_start 里 self.cash / self.equity / self.buying_power 不再静默返回 0."""
    strategy = _run(initial_cash=123456.0)

    start = strategy.seen["on_start"]
    assert start["cash"] == 123456.0
    assert start["equity"] == 123456.0
    assert start["buying_power"] == strategy.seen["first_bar"]["buying_power"]


def test_on_start_positions_are_empty_in_fresh_backtest() -> None:
    """全新回测开跑前没有持仓: self.positions 是空字典而不是报错."""
    strategy = _run(initial_cash=123456.0)

    start = strategy.seen["on_start"]
    assert start["positions"] == {}
    assert start["position"] == 0.0
    assert start["available"] == 0.0


def test_on_start_reports_configured_account_mode() -> None:
    """保证金账户在 on_start 里读到的 account_mode 与运行中一致."""
    strategy = _run(
        initial_cash=50000.0,
        risk_config=RiskConfig(account_mode="margin", enable_short_sell=True),
    )

    assert strategy.seen["on_start"]["account"]["account_mode"] == "margin"
    assert strategy.seen["first_bar"]["account"]["account_mode"] == "margin"


def test_init_still_fails_fast_without_a_run() -> None:
    """没有进入任何一次运行的策略实例没有账户可读, get_account() 仍明确报错."""
    strategy = _Recorder()

    with pytest.raises(RuntimeError, match="Context not ready"):
        strategy.get_account()


def test_on_start_sees_restored_account_when_resuming(tmp_path: Path) -> None:
    """从检查点恢复: on_resume / on_start 读到的是恢复出来的资金与持仓."""
    frame = _frame()
    first = aq.run_backtest(
        data=frame.iloc[:10],
        strategy=_Recorder,
        symbols=[SYMBOL],
        initial_cash=100000.0,
        show_progress=False,
    )
    engine = first.engine
    assert engine is not None
    cash_at_save = float(engine.portfolio.cash)
    equity_at_save = float(engine.get_account_metrics()[0])
    assert dict(engine.portfolio.positions) == {SYMBOL: 1000.0}
    checkpoint = tmp_path / "checkpoint.pkl"
    aq.save_checkpoint(engine, first.strategy, str(checkpoint))  # type: ignore[arg-type]

    second = aq.run_from_checkpoint(
        checkpoint_path=str(checkpoint),
        data=frame.iloc[10:],
        symbols=[SYMBOL],
        show_progress=False,
    )

    resumed: _Recorder = second.strategy  # type: ignore[assignment]
    for where in ("on_resume", "on_start"):
        seen = resumed.seen[where]
        assert seen["cash"] == pytest.approx(cash_at_save)
        assert seen["equity"] == pytest.approx(equity_at_save)
        assert seen["position"] == 1000.0
        assert seen["positions"] == {SYMBOL: 1000.0}
        assert seen["account"]["cash"] == pytest.approx(cash_at_save)
        assert seen["account"]["equity"] == pytest.approx(equity_at_save)
        assert seen["account"]["market_value"] == pytest.approx(
            equity_at_save - cash_at_save
        )


def test_on_start_reads_cash_in_paper_live_session() -> None:
    """Paper 实盘(replay 行情): on_start 里同样读得到账户资金."""
    seen: List[Dict[str, Any]] = []

    class LiveRecorder(Strategy):
        def on_start(self) -> None:
            seen.append(
                {
                    "account_cash": self.get_account()["cash"],
                    "cash": self.cash,
                    "equity": self.equity,
                }
            )

    instrument = Instrument(
        symbol="DEMO",
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=1,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )
    bars = [
        Bar(
            timestamp=int(pd.Timestamp(text, tz="Asia/Shanghai").value),
            open=10.0,
            high=10.5,
            low=9.5,
            close=10.0,
            volume=1000.0,
            symbol="DEMO",
        )
        for text in ("2024-01-02 09:31:00", "2024-01-02 09:32:00")
    ]

    run_live(
        strategy_cls=LiveRecorder,
        instruments=[instrument],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars},
        cash=77000.0,
        show_progress=False,
        duration="20s",
    )

    assert seen == [{"account_cash": 77000.0, "cash": 77000.0, "equity": 77000.0}]
