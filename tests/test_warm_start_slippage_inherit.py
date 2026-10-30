"""Warm start slippage precedence: explicit arg > config > checkpoint snapshot."""

from pathlib import Path
from typing import Any, Optional

import akquant
import pandas as pd
import pytest
from akquant import Bar, Strategy

SNAPSHOT_SLIPPAGE = {"type": "percent", "value": 0.01}


def _bars(start: str) -> pd.DataFrame:
    ts = pd.date_range(start, periods=4, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 100000,
            "symbol": "W",
        }
    )


class _WarmBuy(Strategy):
    """Buy once, only after being restored from a checkpoint."""

    def on_bar(self, bar: Bar) -> None:
        if self.is_restored and not getattr(self, "_bought", False):
            self._bought = True
            self.buy(symbol=bar.symbol, quantity=100)


@pytest.fixture()
def checkpoint(tmp_path: Path) -> str:
    """Phase-1 run at 1% slippage, checkpoint saved with its resolved config."""
    result1 = akquant.run_backtest(
        data=_bars("2024-01-02 10:00"),
        strategy=_WarmBuy,
        symbols="W",
        slippage=SNAPSHOT_SLIPPAGE,  # type: ignore[arg-type]
        show_progress=False,
    )
    path = str(tmp_path / "ckpt.pkl")
    akquant.save_checkpoint(
        result1.engine,  # type: ignore[arg-type]
        result1.strategy,
        path,
        config=result1,
    )
    return path


def _resume_fill_price(
    ckpt: str,
    *,
    config_slippage: Optional[dict[str, Any]] = None,
    slippage: Optional[dict[str, Any]] = None,
) -> float:
    kwargs: dict[str, Any] = {
        "config": akquant.BacktestConfig(
            strategy_config=akquant.StrategyConfig(slippage=config_slippage)  # type: ignore[arg-type]
        )
    }
    if slippage is not None:
        kwargs["slippage"] = slippage
    result = akquant.run_from_checkpoint(
        checkpoint_path=ckpt,
        data=_bars("2024-02-01 10:00"),
        symbols="W",
        show_progress=False,
        **kwargs,
    )
    filled = result.orders_df[result.orders_df["filled_quantity"] > 0]
    assert len(filled) == 1
    return float(filled["avg_price"].iloc[0])


def test_config_without_slippage_inherits_snapshot(checkpoint: str) -> None:
    """A config that leaves slippage unset falls through to the snapshot's 1%."""
    assert _resume_fill_price(checkpoint) == pytest.approx(101.0)


def test_config_zero_slippage_overrides_snapshot(checkpoint: str) -> None:
    """An explicit zero policy in config overrides the snapshot to no slippage."""
    assert _resume_fill_price(
        checkpoint, config_slippage={"type": "zero"}
    ) == pytest.approx(100.0)


def test_explicit_arg_overrides_config_and_snapshot(checkpoint: str) -> None:
    """The slippage= argument beats both config and the snapshot."""
    price = _resume_fill_price(
        checkpoint,
        config_slippage={"type": "percent", "value": 0.02},
        slippage={"type": "zero"},
    )
    assert price == pytest.approx(100.0)
    price = _resume_fill_price(
        checkpoint,
        config_slippage={"type": "zero"},
        slippage={"type": "percent", "value": 0.03},
    )
    assert price == pytest.approx(103.0)
