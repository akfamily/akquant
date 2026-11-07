"""回测的全局费率参数必须同样作用于基金(ETF)标的.

此前走 ChinaMarket(``t_plus_one=True``、含期货或期权)时, ``commission_rate`` /
``commission_policy`` / ``transfer_fee_rate`` / ``min_commission`` 只写进股票配置,
基金一直按 Rust 默认(万三 + 十万分之一过户费 + 最低 5 元)收费; 走 SimpleMarket
(``t_plus_one=False``)时才生效。印花税不作用于基金(场内基金不收)。
"""

from pathlib import Path
from typing import Any, Dict, List, Literal

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest, run_from_checkpoint, save_checkpoint

_SYMBOL = "510300"
_AssetType = Literal["STOCK", "FUND"]


def _bars(start: str, n: int = 4) -> pd.DataFrame:
    ts = pd.date_range(f"{start} 15:00", periods=n, freq="B", tz="Asia/Shanghai")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1e9,
            "symbol": _SYMBOL,
        }
    )


class _BuyThenSell(Strategy):
    """首根买入, 第三根卖出; 每段各一次."""

    qty: float = 100_000.0

    def on_bar(self, bar: Bar) -> None:
        """按 bar 序号买卖."""
        self._n = getattr(self, "_n", 0) + 1
        if self._n == 1:
            self.buy(bar.symbol, self.qty)
        elif self._n == 3:
            self.sell(bar.symbol, self.qty)


def _config(asset_type: _AssetType) -> aq.BacktestConfig:
    return aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=10_000_000.0),
        instruments_config=[aq.InstrumentConfig(symbol=_SYMBOL, asset_type=asset_type)],
    )


def _fees(
    asset_type: _AssetType = "FUND", qty: float = 100_000.0, **kw: Any
) -> List[float]:
    strat = _BuyThenSell()
    strat.qty = qty
    result = run_backtest(
        data=_bars("2026-02-02"),
        strategy=strat,
        config=_config(asset_type),
        show_progress=False,
        **kw,
    )
    return [float(e.commission) for e in result.executions]


_EXPLICIT: Dict[str, Any] = {
    "commission_rate": 0.0001,
    "transfer_fee_rate": 0.0,
    "min_commission": 0.0,
}


@pytest.mark.parametrize("t_plus_one", [False, True])
def test_fund_uses_explicit_commission_rate(t_plus_one: bool) -> None:
    """基金按显式佣金率收费, 与是否 T+1 无关(1e6 成交额 × 万一 = 100)."""
    assert _fees(t_plus_one=t_plus_one, **_EXPLICIT) == pytest.approx([100.0, 100.0])


@pytest.mark.parametrize("asset_type", ["STOCK", "FUND"])
def test_stock_and_fund_agree_on_buy_side(asset_type: _AssetType) -> None:
    """同一组费率下, 股票与基金的买入手续费相同(买入不收印花税)."""
    fees = _fees(asset_type, t_plus_one=True, stamp_tax_rate=0.001, **_EXPLICIT)
    assert fees[0] == pytest.approx(100.0)


def test_fund_sell_has_no_stamp_tax() -> None:
    """印花税只作用于股票, 基金卖出不收."""
    fees = _fees(t_plus_one=True, stamp_tax_rate=0.001, **_EXPLICIT)
    assert fees == pytest.approx([100.0, 100.0])


def test_fund_transfer_fee_and_min_commission() -> None:
    """过户费与最低佣金同样作用于基金."""
    fees = _fees(
        qty=1,
        t_plus_one=True,
        commission_rate=0.0001,
        transfer_fee_rate=0.001,
        min_commission=2.0,
    )
    # 1 股 × 10 元: 佣金按最低 2 元, 过户费 0.01
    assert fees[0] == pytest.approx(2.01)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ({"type": "fixed", "value": 3.0}, 3.0),
        ({"type": "per_unit", "value": 0.001}, 100.0),
        ({"type": "percent", "value": 0.0002}, 200.0),
    ],
)
def test_fund_honors_commission_policy(policy: Any, expected: float) -> None:
    """commission_policy 的三种口径(固定/按股/按成交额)都作用于基金."""
    fees = _fees(
        t_plus_one=True,
        commission_policy=policy,
        transfer_fee_rate=0.0,
        min_commission=0.0,
    )
    assert fees[0] == pytest.approx(expected)


def test_fund_default_fees_match_stock() -> None:
    """不传任何费率时, 基金与股票一样不收费(此前基金按默认万三收)."""
    assert _fees("FUND", t_plus_one=True) == pytest.approx([0.0, 0.0])
    assert _fees("STOCK", t_plus_one=True) == pytest.approx([0.0, 0.0])


def test_fund_commission_kwarg_still_overrides() -> None:
    """单独给基金设费率的 fund_commission 仍优先于全局费率."""
    fees = _fees(t_plus_one=True, fund_commission=0.0005, **_EXPLICIT)
    assert fees[0] == pytest.approx(500.0)


def test_resumed_fund_uses_explicit_commission_rate(tmp_path: Path) -> None:
    """热启动续跑段同样按显式费率给基金计费."""
    first = run_backtest(
        data=_bars("2026-02-02"),
        strategy=_BuyThenSell(),
        config=_config("FUND"),
        show_progress=False,
        t_plus_one=True,
        **_EXPLICIT,
    )
    ckpt = tmp_path / "fund.pkl"
    save_checkpoint(first.engine, first.strategy, str(ckpt))  # type: ignore[arg-type]
    resumed = run_from_checkpoint(
        checkpoint_path=str(ckpt),
        data=_bars("2026-02-16"),
        config=_config("FUND"),
        show_progress=False,
        t_plus_one=True,
        **_EXPLICIT,
    )
    fees = [float(e.commission) for e in resumed.executions]
    assert fees
    assert fees[-1] == pytest.approx(100.0)
