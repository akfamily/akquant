"""run_from_checkpoint 续跑段: 期权按张计费、按品种费用覆盖生效."""

from pathlib import Path
from typing import Optional

import akquant
import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest, run_from_checkpoint, save_checkpoint


def _bars(symbol: str, start: str, n: int, price: float) -> pd.DataFrame:
    ts = pd.date_range(f"{start} 10:00", periods=n, freq="1D")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 100000,
            "symbol": symbol,
        }
    )


class _BuyEachPhase(Strategy):
    """每段第一次看到 target 时买 1 单位."""

    target: str

    def on_bar(self, bar: Bar) -> None:
        """在每段首日对 target 下 1 单位买单."""
        day = str(bar.timestamp_iso)[:10]
        if bar.symbol == self.target and day in ("2024-01-02", "2024-01-10"):
            self.buy(symbol=self.target, quantity=1)


def _config(
    instruments: list[akquant.InstrumentConfig],
    china_options: Optional[akquant.ChinaOptionsConfig] = None,
) -> akquant.BacktestConfig:
    return akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(initial_cash=1_000_000.0),
        instruments_config=instruments,
        china_options=china_options,
    )


def _price(ic: akquant.InstrumentConfig) -> float:
    return 0.1 if str(ic.asset_type).upper() == "OPTION" else 10.0


def _phase2_fees(
    tmp_path: Path,
    instruments: list[akquant.InstrumentConfig],
    target: str,
    china_options: Optional[akquant.ChinaOptionsConfig] = None,
) -> list[float]:
    strat = _BuyEachPhase()
    strat.target = target
    data1 = {
        ic.symbol: _bars(ic.symbol, "2024-01-02", 5, _price(ic)) for ic in instruments
    }
    data2 = {
        ic.symbol: _bars(ic.symbol, "2024-01-10", 5, _price(ic)) for ic in instruments
    }
    r1 = run_backtest(
        data=data1,
        strategy=strat,
        config=_config(instruments, china_options),
        show_progress=False,
    )
    ckpt = tmp_path / "c.pkl"
    save_checkpoint(r1.engine, r1.strategy, str(ckpt))  # type: ignore[arg-type]
    r2 = run_from_checkpoint(
        checkpoint_path=str(ckpt),
        data=data2,
        config=_config(instruments, china_options),
        show_progress=False,
    )
    orders = r2.orders_df
    filled = orders[(orders["filled_quantity"] > 0) & (orders["symbol"] == target)]
    return [float(c) for c in filled["commission"]]


def _option_instruments() -> list[akquant.InstrumentConfig]:
    return [
        akquant.InstrumentConfig(
            symbol="OPT",
            asset_type="OPTION",
            option_type="CALL",
            strike_price=2.0,
            expiry_date=20991231,
            underlying_symbol="UL",
            multiplier=10000.0,
            tick_size=0.0001,
        ),
        akquant.InstrumentConfig(symbol="UL", asset_type="FUND"),
    ]


def test_resumed_option_uses_per_contract_fees(tmp_path: Path) -> None:
    """续跑段期权成交按张计费(默认 6.6 元/张), 而非按成交额百分比."""
    fees = _phase2_fees(tmp_path, _option_instruments(), "OPT")
    assert fees
    assert fees[-1] == pytest.approx(6.6)


def test_resumed_custom_option_fee_config(tmp_path: Path) -> None:
    """续跑段沿用 config.china_options 的自定义期权费率, 不回落到默认 6.6."""
    china_options = akquant.ChinaOptionsConfig(
        commission_per_contract=1.0,
        exchange_fee_per_contract=0.0,
        clearing_fee_per_contract=0.0,
    )
    fees = _phase2_fees(tmp_path, _option_instruments(), "OPT", china_options)
    assert fees
    assert fees[-1] == pytest.approx(1.0)


def test_resumed_instrument_fee_override(tmp_path: Path) -> None:
    """续跑段 InstrumentConfig.commission_rate 按品种覆盖仍生效."""
    instruments = [
        akquant.InstrumentConfig(symbol="A", commission_rate=0.01, min_commission=0.0)
    ]
    # 成交额 10 × 1 = 10, 10 × 0.01 = 0.1
    fees = _phase2_fees(tmp_path, instruments, "A")
    assert fees
    assert fees[-1] == pytest.approx(0.1)
