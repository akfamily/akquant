"""Bare-number slippage is rejected everywhere; None and dict still work."""

from pathlib import Path
from typing import Any

import akquant
import pandas as pd
import pytest
from akquant import Bar, Strategy
from akquant.config import bare_slippage_error

BARE_VALUES = [0.0002, 0.2, 0, 0.0, 1, True]


def _bars(symbol: str = "A", start: str = "2024-01-02 10:00") -> pd.DataFrame:
    ts = pd.date_range(start, periods=4, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 100000,
            "symbol": symbol,
        }
    )


class _Idle(Strategy):
    def on_bar(self, bar: Bar) -> None:
        return None


class _OrderOnce(Strategy):
    """Place one order through the chosen API with a given slippage."""

    method = "buy"
    order_slippage: Any = None

    def on_bar(self, bar: Bar) -> None:
        if getattr(self, "_done", False):
            return
        self._done = True
        fn = getattr(self, self.method)
        fn(symbol="A", quantity=100, slippage=self.order_slippage)


def _assert_hint(excinfo: Any, bare: Any) -> None:
    message = str(excinfo.value)
    assert repr(bare) in message
    assert f"{{'type': 'percent', 'value': {bare!r}}}" in message
    assert "{'type': 'fixed', 'value': ...}" in message
    assert "omit it or pass None" in message


def _run(**kwargs: Any) -> Any:
    return akquant.run_backtest(
        data=_bars(), strategy=_Idle, symbols="A", show_progress=False, **kwargs
    )


def test_helper_message_is_shared() -> None:
    """The helper builds a TypeError with value, migration and None hint."""
    err = bare_slippage_error("Global", 0.0002)
    assert isinstance(err, TypeError)
    assert "Global" in str(err)
    assert "{'type': 'percent', 'value': 0.0002}" in str(err)


@pytest.mark.parametrize("bare", BARE_VALUES)
def test_run_backtest_rejects_bare(bare: Any) -> None:
    """run_backtest(slippage=<number>) raises TypeError."""
    with pytest.raises(TypeError) as excinfo:
        _run(slippage=bare)
    _assert_hint(excinfo, bare)


@pytest.mark.parametrize("bare", BARE_VALUES)
def test_strategy_config_rejects_bare(bare: Any) -> None:
    """StrategyConfig.slippage with a number raises TypeError at run time."""
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(slippage=bare)  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError) as excinfo:
        _run(config=config)
    _assert_hint(excinfo, bare)


def test_strategy_config_default_is_none() -> None:
    """The default StrategyConfig.slippage is None, not 0.0."""
    assert akquant.StrategyConfig().slippage is None


@pytest.mark.parametrize("bare", BARE_VALUES)
def test_instrument_config_rejects_bare(bare: Any) -> None:
    """InstrumentConfig.slippage with a number raises TypeError."""
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(),
        instruments_config=[
            akquant.InstrumentConfig(symbol="A", slippage=bare)  # type: ignore[arg-type]
        ],
    )
    with pytest.raises(TypeError) as excinfo:
        _run(config=config)
    _assert_hint(excinfo, bare)


@pytest.mark.parametrize("bare", BARE_VALUES)
def test_strategy_slippage_map_rejects_bare(bare: Any) -> None:
    """strategy_slippage values that are numbers raise TypeError."""
    with pytest.raises(TypeError) as excinfo:
        _run(strategy_slippage={"_default": bare})
    _assert_hint(excinfo, bare)


@pytest.mark.parametrize("method", ["buy", "sell", "short", "cover"])
@pytest.mark.parametrize("bare", [0.0002, 0.0, 1])
def test_order_level_rejects_bare(method: str, bare: Any) -> None:
    """Order-level slippage with a number raises TypeError for all four APIs."""

    class _S(_OrderOnce):
        pass

    _S.method = method
    _S.order_slippage = bare
    with pytest.raises(TypeError) as excinfo:
        akquant.run_backtest(
            data=_bars(), strategy=_S, symbols="A", show_progress=False
        )
    _assert_hint(excinfo, bare)


def test_none_and_dict_still_work() -> None:
    """None means no slippage; dict policies keep their behavior."""
    _run(slippage=None)
    _run(slippage={"type": "percent", "value": 0.0002})
    _run(strategy_slippage={"_default": {"type": "fixed", "value": 0.1}})
    _run(
        config=akquant.BacktestConfig(
            strategy_config=akquant.StrategyConfig(
                slippage={"type": "percent", "value": 0.001}
            ),
            instruments_config=[
                akquant.InstrumentConfig(
                    symbol="A", slippage={"type": "fixed", "value": 0.1}
                )
            ],
        )
    )


class _WarmBuy(Strategy):
    def on_bar(self, bar: Bar) -> None:
        if self.is_restored and not getattr(self, "_bought", False):
            self._bought = True
            self.buy(symbol=bar.symbol, quantity=1.0)


@pytest.mark.parametrize("via", ["kwarg", "config"])
def test_warm_start_rejects_bare(tmp_path: Path, via: str) -> None:
    """run_from_checkpoint rejects a bare-number slippage (kwarg or config)."""
    result1 = akquant.run_backtest(
        data=_bars("W"), strategy=_WarmBuy, symbols="W", show_progress=False
    )
    ckpt = str(tmp_path / "ckpt.pkl")
    akquant.save_checkpoint(result1.engine, result1.strategy, ckpt)  # type: ignore[arg-type]
    kwargs: dict[str, Any] = {}
    if via == "kwarg":
        kwargs["slippage"] = 0.0002
    else:
        kwargs["config"] = akquant.BacktestConfig(
            strategy_config=akquant.StrategyConfig(slippage=0.0002)  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError) as excinfo:
        akquant.run_from_checkpoint(
            checkpoint_path=ckpt,
            data=_bars("W", "2024-02-01 10:00"),
            symbols="W",
            show_progress=False,
            **kwargs,
        )
    _assert_hint(excinfo, 0.0002)
