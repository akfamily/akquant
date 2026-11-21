"""Strategy 期权方法: 期权链、选合约、Greeks、组合 Greeks 与 Delta 对冲."""

import math
from typing import Any, Dict, List, Literal, Tuple

import akquant
import pandas as pd
import pytest
from akquant import Bar, Strategy
from akquant.option import black76_price

DAYS = ["2023-11-27", "2023-11-28", "2023-11-29"]
EXPIRY = 20231215
RATE = 0.02
SIGMA = 0.2

Setup = Tuple[Dict[str, pd.DataFrame], List[akquant.InstrumentConfig]]
KINDS: Tuple[Tuple[Literal["CALL", "PUT"], str], ...] = (("CALL", "C"), ("PUT", "P"))


def _t(day: str) -> float:
    return (pd.Timestamp(str(EXPIRY)) - pd.Timestamp(day)).days / 365.0


def _daily(symbol: str, closes: List[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([f"{d} 15:00" for d in DAYS]),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1000,
            "symbol": symbol,
        }
    )


def _options(
    series: str,
    underlying: str,
    multiplier: float,
    strikes: List[float],
    forward: float,
) -> Setup:
    """按 Black-76 (σ=SIGMA, 远期恒为 forward) 生成一条期权链的行情与合约配置."""
    data: Dict[str, pd.DataFrame] = {}
    configs: List[akquant.InstrumentConfig] = []
    for k in strikes:
        for kind, flag in KINDS:
            symbol = f"{series}-{flag}-{int(k)}"
            closes = [
                float(black76_price(forward, k, _t(d), RATE, SIGMA, kind)) for d in DAYS
            ]
            data[symbol] = _daily(symbol, closes)
            configs.append(
                akquant.InstrumentConfig(
                    symbol=symbol,
                    asset_type="OPTION",
                    multiplier=multiplier,
                    option_type=kind,
                    strike_price=k,
                    expiry_date=EXPIRY,
                    underlying_symbol=underlying,
                )
            )
    return data, configs


def _commodity() -> Setup:
    """豆粕期权链, 标的期货 m2401 恒为 3000."""
    data, configs = _options("m2401", "m2401", 10.0, [2900.0, 3000.0, 3100.0], 3000.0)
    data["m2401"] = _daily("m2401", [3000.0] * 3)
    configs.append(
        akquant.InstrumentConfig(
            symbol="m2401", asset_type="FUTURES", multiplier=10.0, margin_ratio=0.1
        )
    )
    return data, configs


def _index() -> Setup:
    """股指期权链, 指数 4000、期权隐含远期 3985(贴水), 对冲用期货 IF2312."""
    data, configs = _options("IO2312", "IDX", 100.0, [3900.0, 4000.0, 4100.0], 3985.0)
    data["IDX"] = _daily("IDX", [4000.0] * 3)
    data["IF2312"] = _daily("IF2312", [3985.0] * 3)
    configs += [
        akquant.InstrumentConfig(symbol="IDX", asset_type="STOCK"),
        akquant.InstrumentConfig(
            symbol="IF2312", asset_type="FUTURES", multiplier=300.0, margin_ratio=0.12
        ),
    ]
    return data, configs


class _DayEnd(Strategy):
    """每天数据中全部标的的 bar 到齐后调用 ``on_day``, 保证期权链用的是当日价格."""

    option_rate = RATE
    n_symbols = 0

    def __init__(self) -> None:
        super().__init__()
        self._seen: Dict[str, set] = {}
        self.records: Dict[str, Any] = {}

    def on_bar(self, bar: Bar) -> None:
        day = str(bar.timestamp_iso)[:10]
        seen = self._seen.setdefault(day, set())
        seen.add(bar.symbol)
        if len(seen) == self.n_symbols:
            self.on_day(DAYS.index(day))

    def on_day(self, i: int) -> None:
        """当日 bar 到齐后的回调, ``i`` 为 DAYS 中的序号."""


def _run(strategy: _DayEnd, setup: Setup) -> None:
    data, instruments = setup
    strategy.n_symbols = len(data)
    config = akquant.BacktestConfig(
        strategy_config=akquant.StrategyConfig(initial_cash=10_000_000.0),
        instruments_config=instruments,
        china_options=akquant.ChinaOptionsConfig(
            commission_per_contract=0.0,
            exchange_fee_per_contract=0.0,
            clearing_fee_per_contract=0.0,
            exercise_fee_per_contract=0.0,
        ),
    )
    akquant.run_backtest(
        data=data, strategy=strategy, config=config, show_progress=False, freq="1d"
    )


class _CommodityChain(_DayEnd):
    def on_day(self, i: int) -> None:
        if i != 0:
            return
        self.records.update(
            expiries=self.get_expiries("m2401"),
            chain=self.option_chain("m2401"),
            atm_call=self.select_option("m2401", "CALL"),
            up_call=self.select_option("m2401", "C", strike_offset=1),
            down_put=self.select_option("m2401", "PUT", strike_offset=-1),
            out_of_range=self.select_option("m2401", "C", strike_offset=5),
            delta_put=self.select_option("m2401", "P", delta=-0.2),
            greeks=self.get_greeks("m2401-C-3000"),
        )


def test_commodity_chain_and_selection() -> None:
    """期货标的: 远期取期货价, 隐含波动率还原, 按平值档位与 Delta 选合约."""
    strat = _CommodityChain()
    _run(strat, _commodity())

    rec = strat.records
    assert rec["expiries"] == [EXPIRY]
    chain = rec["chain"]
    assert len(chain) == 6
    assert chain["forward"].tolist() == pytest.approx([3000.0] * 6)
    assert chain["iv"].tolist() == pytest.approx([SIGMA] * 6, abs=1e-6)
    assert chain["t"].iloc[0] == pytest.approx(_t(DAYS[0]))
    assert rec["atm_call"] == "m2401-C-3000"
    assert rec["up_call"] == "m2401-C-3100"
    assert rec["down_put"] == "m2401-P-2900"
    assert rec["out_of_range"] is None
    puts = chain[chain["option_type"] == "PUT"]
    assert (
        rec["delta_put"]
        == puts.loc[(puts["delta"].abs() - 0.2).abs().idxmin(), "symbol"]
    )
    t = _t(DAYS[0])
    d1 = 0.5 * SIGMA * math.sqrt(t)
    assert rec["greeks"]["delta"] == pytest.approx(
        math.exp(-RATE * t) * 0.5 * math.erfc(-d1 / math.sqrt(2)), rel=1e-6
    )


class _CommodityHedge(_DayEnd):
    def on_day(self, i: int) -> None:
        if i == 0:
            self.buy("m2401-C-3000", 2)
        elif i == 1:
            self.records.update(
                before=self.portfolio_greeks("m2401"),
                delta=self.get_greeks("m2401-C-3000")["delta"],
                order=self.hedge_delta("m2401"),
            )
        else:
            self.records.update(
                after=self.portfolio_greeks("m2401"),
                futures=self.get_position("m2401"),
            )


def test_commodity_portfolio_greeks_and_hedge() -> None:
    """持有 2 张平值认购, 次日卖出标的期货把 Delta 对冲到 1 手以内."""
    strat = _CommodityHedge()
    _run(strat, _commodity())

    rec = strat.records
    delta = rec["delta"]
    assert rec["before"]["delta"] == pytest.approx(2 * 10 * delta)
    assert rec["before"]["cash_delta"] == pytest.approx(2 * 10 * delta * 3000.0)
    assert rec["order"] is not None
    assert rec["futures"] == -round(2 * delta)
    assert abs(rec["after"]["delta"]) < 10.0


class _IndexHedge(_DayEnd):
    def on_day(self, i: int) -> None:
        if i == 0:
            self.records["chain"] = self.option_chain("IDX")
            self.buy("IO2312-C-4000", 6)
        elif i == 1:
            self.hedge_delta("IDX", hedge_symbol="IF2312")
        else:
            self.records.update(
                hedged=self.portfolio_greeks("IDX", include=["IF2312"]),
                options_only=self.portfolio_greeks("IDX"),
                futures=self.get_position("IF2312"),
            )


def test_index_options_use_parity_forward_and_futures_hedge() -> None:
    """指数标的: 远期按平价反推(含基差), 用股指期货对冲, include 计入期货 Delta."""
    strat = _IndexHedge()
    _run(strat, _index())

    rec = strat.records
    chain = rec["chain"]
    assert chain["forward"].tolist() == pytest.approx([3985.0] * 6)
    assert chain["iv"].tolist() == pytest.approx([SIGMA] * 6, abs=1e-6)
    assert rec["futures"] < 0
    lot_cash = 300.0 * 3985.0
    assert abs(rec["hedged"]["cash_delta"]) <= 0.5 * lot_cash
    assert rec["hedged"]["cash_delta"] == pytest.approx(
        rec["options_only"]["cash_delta"] + rec["futures"] * lot_cash
    )


def _error(fn: Any, *args: Any) -> str:
    try:
        fn(*args)
    except ValueError as exc:
        return str(exc)
    return ""


class _Failures(_DayEnd):
    """逐项制造算不出 Greeks 的情形, 记录报错文本."""

    def on_day(self, i: int) -> None:
        if i == 0:
            self.buy("m2401-C-3000", 1)
            self.sell("m2401", 1)
            return
        if i != 1:
            return
        prices = self._last_prices
        saved = dict(prices)
        rec = self.records

        prices.pop("m2401-C-3000")
        rec["no_price"] = _error(self.get_greeks, "m2401-C-3000")
        rec["portfolio"] = _error(self.portfolio_greeks, "m2401")
        prices.update(saved)

        prices["m2401-C-2900"] = 50.0  # 内在价值约 100
        rec["below_intrinsic"] = _error(self.get_greeks, "m2401-C-2900")
        floor = float(black76_price(3000.0, 2900.0, _t(DAYS[1]), RATE, 1e-6, "CALL"))
        prices["m2401-C-2900"] = floor + 5e-9
        rec["within_tol"] = _error(self.get_greeks, "m2401-C-2900")
        prices["m2401-C-3100"] = 3000.0  # 远期 3000 的认购不可能值 3000
        rec["above_cap"] = _error(self.get_greeks, "m2401-C-3100")
        prices.update(saved)

        prices.pop("m2401")
        rec["no_forward"] = _error(self.get_greeks, "m2401-C-3000")
        rec["no_linear"] = _error(self.portfolio_greeks, "m2401")
        prices.update(saved)

        rec["expired"] = _error(self.get_greeks, "m2401-C-2800")
        rec["ok"] = _error(self.get_greeks, "m2401-C-3000")


def test_greeks_errors_state_the_exact_cause() -> None:
    """算不出 Greeks 时报出确切原因, 持仓不静默按 0 计."""
    data, configs = _commodity()
    data["m2401-C-2800"] = _daily("m2401-C-2800", [200.0] * 3)
    configs.append(
        akquant.InstrumentConfig(
            symbol="m2401-C-2800",
            asset_type="OPTION",
            multiplier=10.0,
            option_type="CALL",
            strike_price=2800.0,
            expiry_date=20231120,
            underlying_symbol="m2401",
        )
    )
    strat = _Failures()
    _run(strat, (data, configs))

    rec = strat.records
    assert rec["no_price"] == "m2401-C-3000 无法计算 Greeks: 没有最新价"
    assert rec["portfolio"] == "以下持仓无法计算 Greeks: m2401-C-3000: 没有最新价"
    assert (
        "m2401-C-2900 无法计算 Greeks: 最新价 50 不高于波动率 0.0001% 时的理论价"
        in (rec["below_intrinsic"])
    )
    assert "隐含波动率无解" in rec["below_intrinsic"]
    assert rec["within_tol"].startswith("m2401-C-2900 无法计算 Greeks: 时间价值 5e-09")
    assert "不超过求解价格容差 1e-08" in rec["within_tol"]
    assert "最新价 3000 不低于波动率 500% 时的理论价" in rec["above_cap"]
    assert "隐含波动率超出求解上限" in rec["above_cap"]
    assert rec["no_forward"] == (
        "m2401-C-3000 无法计算 Greeks: 标的期货 m2401 没有最新价"
    )
    assert rec["no_linear"] == (
        "以下持仓无法计算 Greeks: m2401-C-3000: 标的期货 m2401 没有最新价; "
        "m2401: 持有该线性品种但没有最新价"
    )
    assert rec["expired"] == "m2401-C-2800 无法计算 Greeks: 已于 20231120 15:00 到期"
    assert rec["ok"] == ""


TICK_NOW = pd.Timestamp("2023-11-27 10:00", tz="Asia/Shanghai")
TICK_EXPIRY = pd.Timestamp("2023-12-15 15:00", tz="Asia/Shanghai")


def _tick_t() -> float:
    from akquant.option import year_fraction

    return year_fraction(TICK_NOW.to_pydatetime(), TICK_EXPIRY.to_pydatetime())


def _quoted_ticks() -> Tuple[List[Any], List[akquant.InstrumentConfig]]:
    """豆粕期货 3000 与 5 张期权的 tick; 成交价故意偏离中间价 3 元.

    报价情形: C-3000 / P-3000 双边, C-3100 只有卖一, P-2900 无报价, C-2900 买价
    高于卖价。
    """
    from akquant.akquant import Tick

    t = _tick_t()
    ns = TICK_NOW.value
    fair = {
        (kind, k): float(black76_price(3000.0, k, t, RATE, SIGMA, kind))
        for kind, k in (
            ("CALL", 3000),
            ("PUT", 3000),
            ("CALL", 3100),
            ("PUT", 2900),
            ("CALL", 2900),
        )
    }
    books = {
        ("CALL", 3000): (-0.5, 0.5),
        ("PUT", 3000): (-0.5, 0.5),
        ("CALL", 3100): (None, 0.5),
        ("PUT", 2900): (None, None),
        ("CALL", 2900): (1.0, -1.0),
    }
    ticks: List[Any] = [Tick(ns, 3000.0, 1.0, "m2401")]
    configs = [
        akquant.InstrumentConfig(
            symbol="m2401", asset_type="FUTURES", multiplier=10.0, margin_ratio=0.1
        )
    ]
    for (kind, k), (bid_off, ask_off) in books.items():
        symbol = f"m2401-{kind[0]}-{k}"
        p = fair[(kind, k)]
        ticks.append(
            Tick(
                ns,
                p + 3.0,
                1.0,
                symbol,
                bid_price=None if bid_off is None else p + bid_off,
                ask_price=None if ask_off is None else p + ask_off,
            )
        )
        configs.append(
            akquant.InstrumentConfig(
                symbol=symbol,
                asset_type="OPTION",
                multiplier=10.0,
                option_type="CALL" if kind == "CALL" else "PUT",
                strike_price=float(k),
                expiry_date=20231215,
                underlying_symbol="m2401",
            )
        )
    return ticks, configs


class _QuoteChain(Strategy):
    option_rate = RATE
    option_price_source: Literal["last", "mid"] = "mid"

    def __init__(self) -> None:
        super().__init__()
        self.seen: set = set()
        self.records: Dict[str, Any] = {}

    def on_tick(self, tick: Any) -> None:
        self.seen.add(tick.symbol)
        if len(self.seen) < 6 or self.records:
            return
        self.records["mid"] = self.option_chain("m2401")
        for symbol in ("m2401-C-3100", "m2401-P-2900", "m2401-C-2900"):
            self.records[symbol] = _error(self.get_greeks, symbol)
        self.option_price_source = "last"
        self.records["last"] = self.option_chain("m2401")


def test_mid_quote_pricing_and_quote_failures() -> None:
    """中间价模式: 双边报价取中间价, 单边或异常报价不算并写明原因; last 模式取成交价."""
    ticks, configs = _quoted_ticks()
    strat = _QuoteChain()
    akquant.run_backtest(
        data=ticks,
        strategy=strat,
        symbols=[str(conf.symbol) for conf in configs],
        config=akquant.BacktestConfig(
            strategy_config=akquant.StrategyConfig(initial_cash=1_000_000.0),
            instruments_config=configs,
        ),
        show_progress=False,
    )
    rec = strat.records
    mid = rec["mid"].set_index("symbol")
    t = _tick_t()
    fair_c = float(black76_price(3000.0, 3000.0, t, RATE, SIGMA, "CALL"))
    assert mid.loc["m2401-C-3000", "price"] == pytest.approx(fair_c)
    assert mid.loc["m2401-C-3000", "bid"] == pytest.approx(fair_c - 0.5)
    assert mid.loc["m2401-C-3000", "iv"] == pytest.approx(SIGMA, abs=1e-6)
    assert mid.loc[["m2401-C-3100", "m2401-P-2900", "m2401-C-2900"], "iv"].isna().all()
    assert rec["m2401-C-3100"].startswith("m2401-C-3100 无法计算 Greeks: 只有卖一价 ")
    assert rec["m2401-C-3100"].endswith(", 没有买一价")
    assert rec["m2401-P-2900"] == "m2401-P-2900 无法计算 Greeks: 没有买一价和卖一价"
    assert "高于卖一价" in rec["m2401-C-2900"]
    last = rec["last"].set_index("symbol")
    assert last.loc["m2401-C-3000", "price"] == pytest.approx(fair_c + 3.0)
