"""策略侧期权工具: 期权链、选合约、Greeks 与 Delta 对冲."""

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd

from .option.pricing import (
    _IV_MAX_ITER,
    _IV_TOL,
    _SIGMA_HIGH,
    _SIGMA_LOW,
    baw_price,
    black76_greeks,
    black76_price,
    implied_forward,
    implied_vol,
    year_fraction,
)

OptionModel = Literal["black76", "baw"]

_CHAIN_COLUMNS = [
    "symbol",
    "option_type",
    "strike",
    "expiry_date",
    "t",
    "bid",
    "ask",
    "price",
    "forward",
    "iv",
    "delta",
    "gamma",
    "vega",
    "theta",
]


def _resolve_rate(strategy: Any, rate: Optional[float]) -> float:
    return float(strategy.option_rate if rate is None else rate)


def _price_source(strategy: Any) -> str:
    source = strategy.option_price_source
    if source not in ("last", "mid"):
        raise ValueError(f"option_price_source 只接受 last/mid, 实际: {source!r}")
    return str(source)


def _quote_price(strategy: Any, symbol: str) -> Tuple[float, float, float]:
    """(买一价, 卖一价, 定价用价格), 缺失为 nan.

    定价用价格: ``option_price_source="last"`` 取最新价; ``"mid"`` 取中间价, 仅当
    买一价、卖一价都为正且买价不高于卖价(Cboe VIX 的取价规则)。
    """
    bid, ask = strategy._last_quotes.get(symbol, (None, None))
    b = math.nan if bid is None else float(bid)
    a = math.nan if ask is None else float(ask)
    if _price_source(strategy) == "last":
        return b, a, float(strategy._last_prices.get(symbol, math.nan))
    valid = b > 0 and a > 0 and b <= a
    return b, a, 0.5 * (b + a) if valid else math.nan


def _mid_failure(bid: float, ask: float) -> str:
    """中间价取不到的确切原因, 与 :func:`_quote_price` 的判定一致."""
    if math.isnan(bid) and math.isnan(ask):
        return "没有买一价和卖一价"
    if math.isnan(bid):
        return f"只有卖一价 {ask:g}, 没有买一价"
    if math.isnan(ask):
        return f"只有买一价 {bid:g}, 没有卖一价"
    if bid <= 0:
        return f"买一价 {bid:g} 不为正"
    if ask <= 0:
        return f"卖一价 {ask:g} 不为正"
    return f"买一价 {bid:g} 高于卖一价 {ask:g}"


def _expiry_t(strategy: Any, expiry_date: int) -> float:
    """当前时刻到到期日 15:00(策略时区)的年化期限; 没有当前时间时为 nan."""
    now = strategy.now
    if now is None:
        return math.nan
    expiry = pd.Timestamp(str(expiry_date)).replace(hour=15)
    expiry = expiry.tz_localize(now.tz) if now.tz is not None else expiry
    return year_fraction(now.to_pydatetime(), expiry.to_pydatetime())


def _option_snapshots(strategy: Any, underlying: str) -> List[Any]:
    return [
        snap
        for snap in strategy.get_instruments().values()
        if snap.asset_type == "OPTION"
        and snap.underlying_symbol == underlying
        and snap.expiry_date is not None
        and snap.strike_price is not None
        and snap.option_type is not None
    ]


def get_expiries(strategy: Any, underlying: str) -> List[int]:
    """返回标的下尚未到期的期权到期日(YYYYMMDD), 由近到远."""
    expiries = {snap.expiry_date for snap in _option_snapshots(strategy, underlying)}
    return sorted(e for e in expiries if _expiry_t(strategy, e) > 0)


def _forward(
    strategy: Any,
    underlying: str,
    strikes: np.ndarray,
    types: np.ndarray,
    prices: np.ndarray,
    t: float,
    rate: float,
) -> float:
    """远期价格: 期货标的取期货价; 否则按平价反推, 缺报价时退回现货按利率前推."""
    underlying_snap = strategy.get_instruments().get(underlying)
    spot = strategy._last_prices.get(underlying)
    if underlying_snap is not None and underlying_snap.asset_type == "FUTURES":
        return float(spot) if spot is not None else math.nan
    priced = np.isfinite(prices)
    calls = dict(
        zip(strikes[priced & (types == "CALL")], prices[priced & (types == "CALL")])
    )
    puts = dict(
        zip(strikes[priced & (types == "PUT")], prices[priced & (types == "PUT")])
    )
    common = sorted(calls.keys() & puts.keys())
    if common:
        return implied_forward(
            common, [calls[k] for k in common], [puts[k] for k in common], t, rate
        )
    return float(spot) * math.exp(rate * t) if spot is not None else math.nan


def option_chain(
    strategy: Any,
    underlying: str,
    expiry: Optional[int] = None,
    rate: Optional[float] = None,
    model: OptionModel = "black76",
) -> pd.DataFrame:
    """标的的期权链: 合约要素、报价、远期、隐含波动率与 Greeks."""
    r = _resolve_rate(strategy, rate)
    snaps = [
        snap
        for snap in _option_snapshots(strategy, underlying)
        if expiry is None or snap.expiry_date == expiry
    ]
    symbols = np.array([snap.symbol for snap in snaps], dtype=object)
    types = np.array([snap.option_type for snap in snaps], dtype=str)
    strikes = np.array([snap.strike_price for snap in snaps], dtype=float)
    expiries = np.array([snap.expiry_date for snap in snaps], dtype=np.int64)
    quotes = np.array(
        [_quote_price(strategy, snap.symbol) for snap in snaps], dtype=float
    ).reshape(-1, 3)
    prices = quotes[:, 2]
    t = np.full(len(snaps), np.nan)
    forward = np.full(len(snaps), np.nan)
    for expiry_date in np.unique(expiries):
        mask = expiries == expiry_date
        t_e = _expiry_t(strategy, int(expiry_date))
        if t_e > 0:
            t[mask] = t_e
            forward[mask] = _forward(
                strategy, underlying, strikes[mask], types[mask], prices[mask], t_e, r
            )
    live = t > 0
    iv = np.asarray(
        implied_vol(
            prices[live], forward[live], strikes[live], t[live], r, types[live], model
        ),
        dtype=float,
    )
    greeks = black76_greeks(forward[live], strikes[live], t[live], r, iv, types[live])
    data: Dict[str, Any] = {
        "symbol": symbols[live],
        "option_type": types[live].astype(object),
        "strike": strikes[live],
        "expiry_date": expiries[live],
        "t": t[live],
        "bid": quotes[live, 0],
        "ask": quotes[live, 1],
        "price": prices[live],
        "forward": forward[live],
        "iv": iv,
    }
    for name in ("delta", "gamma", "vega", "theta"):
        data[name] = np.where(np.isnan(iv), np.nan, greeks[name])
    order = np.lexsort((data["option_type"], data["strike"], data["expiry_date"]))
    return pd.DataFrame({name: data[name][order] for name in _CHAIN_COLUMNS})


def _greeks_failure(
    strategy: Any,
    snap: Any,
    row: Optional[pd.Series],
    rate: float,
    model: OptionModel,
) -> str:
    """期权算不出 Greeks 的确切原因, 判定条件与 option_chain / implied_vol 一致.

    :param row: 该合约在期权链中的行; 不在链中(期限不为正)时为 None
    """
    if strategy.now is None:
        return "当前没有行情时间, 无法计算剩余期限"
    if row is None:
        return f"已于 {snap.expiry_date} 15:00 到期"
    price, forward, t, strike = (
        float(row["price"]),
        float(row["forward"]),
        float(row["t"]),
        float(row["strike"]),
    )
    by_mid = _price_source(strategy) == "mid"
    label = "中间价" if by_mid else "最新价"
    if math.isnan(price):
        return (
            _mid_failure(float(row["bid"]), float(row["ask"]))
            if by_mid
            else ("没有最新价")
        )
    if math.isnan(forward):
        underlying = snap.underlying_symbol
        underlying_snap = strategy.get_instruments().get(underlying)
        if underlying_snap is not None and underlying_snap.asset_type == "FUTURES":
            return f"标的期货 {underlying} 没有最新价"
        return (
            f"到期日 {snap.expiry_date} 没有同一行权价的认购、认沽{label}可反推远期, "
            f"标的 {underlying} 也没有最新价"
        )
    if forward <= 0:
        return f"远期价格 {forward:g} 不为正"
    if strike <= 0:
        return f"行权价 {strike:g} 不为正"
    price_fn = black76_price if model == "black76" else baw_price
    floor = float(price_fn(forward, strike, t, rate, _SIGMA_LOW, snap.option_type))
    time_value = price - floor
    if time_value <= 0:
        return (
            f"{label} {price:g} 不高于波动率 {_SIGMA_LOW:.4%} 时的理论价 {floor:.6g}"
            f"(内在价值, 远期 {forward:g}), 隐含波动率无解"
        )
    if time_value <= _IV_TOL:
        return (
            f"时间价值 {time_value:.3g}({label} {price:g} 减内在价值 {floor:.6g})"
            f"不超过求解价格容差 {_IV_TOL:g}, 价格中没有可分辨的波动率信息"
        )
    cap = float(price_fn(forward, strike, t, rate, _SIGMA_HIGH, snap.option_type))
    if price >= cap:
        return (
            f"{label} {price:g} 不低于波动率 {_SIGMA_HIGH:.0%} 时的理论价 {cap:.6g}"
            f"(远期 {forward:g}), 隐含波动率超出求解上限"
        )
    return (
        f"隐含波动率在 {_IV_MAX_ITER} 次迭代内未收敛"
        f"({label} {price:g}, 远期 {forward:g})"
    )


def _chain_row(chain: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
    rows = chain[chain["symbol"] == symbol]
    return None if rows.empty else rows.iloc[0]


def select_option(
    strategy: Any,
    underlying: str,
    option_type: str,
    expiry: Optional[int] = None,
    strike_offset: int = 0,
    delta: Optional[float] = None,
    rate: Optional[float] = None,
    model: OptionModel = "black76",
) -> Optional[str]:
    """按到期日与平值档位(或目标 Delta)选出一个合约代码; 选不到返回 None."""
    kind = option_type.strip().upper()
    kind = {"C": "CALL", "P": "PUT"}.get(kind, kind)
    if kind not in ("CALL", "PUT"):
        raise ValueError(f"option_type 只接受 CALL/PUT/C/P, 实际: {option_type!r}")
    if expiry is None:
        expiries = get_expiries(strategy, underlying)
        if not expiries:
            return None
        expiry = expiries[0]
    chain = option_chain(strategy, underlying, expiry=expiry, rate=rate, model=model)
    chain = chain[chain["option_type"] == kind]
    if chain.empty:
        return None
    if delta is not None:
        distance = (chain["delta"].abs() - abs(delta)).abs()
        if distance.isna().all():
            return None
        return str(chain.loc[distance.idxmin(), "symbol"])
    forward = float(chain["forward"].iloc[0])
    if math.isnan(forward):
        return None
    strikes = np.sort(chain["strike"].unique())
    idx = int(np.argmin(np.abs(strikes - forward))) + strike_offset
    if not 0 <= idx < len(strikes):
        return None
    return str(chain.loc[chain["strike"] == strikes[idx], "symbol"].iloc[0])


def get_greeks(
    strategy: Any,
    symbol: str,
    rate: Optional[float] = None,
    model: OptionModel = "black76",
) -> Dict[str, float]:
    """单个期权合约的价格、远期、期限、隐含波动率与 Greeks(每单位标的)."""
    snap = strategy.get_instrument(symbol)
    if snap.asset_type != "OPTION" or snap.underlying_symbol is None:
        raise ValueError(f"{symbol} 不是已登记标的的期权合约")
    chain = option_chain(
        strategy,
        snap.underlying_symbol,
        expiry=snap.expiry_date,
        rate=rate,
        model=model,
    )
    row = _chain_row(chain, symbol)
    if row is None or math.isnan(float(row["delta"])):
        reason = _greeks_failure(
            strategy, snap, row, _resolve_rate(strategy, rate), model
        )
        raise ValueError(f"{symbol} 无法计算 Greeks: {reason}")
    return {
        name: float(row[name])
        for name in ("price", "forward", "t", "iv", "delta", "gamma", "vega", "theta")
    }


def portfolio_greeks(
    strategy: Any,
    underlying: str,
    include: Sequence[str] = (),
    rate: Optional[float] = None,
    model: OptionModel = "black76",
) -> Dict[str, float]:
    """标的相关持仓的合计 Greeks, 已乘持仓数量与合约乘数."""
    positions = strategy.positions
    total = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "cash_delta": 0.0}
    held = {
        snap.symbol: snap
        for snap in _option_snapshots(strategy, underlying)
        if positions.get(snap.symbol, 0.0)
    }
    linear = [underlying] + [s for s in include if s != underlying]
    failures = []
    rows = []
    if held:
        r = _resolve_rate(strategy, rate)
        chain = option_chain(strategy, underlying, rate=r, model=model)
        for symbol, snap in sorted(held.items()):
            row = _chain_row(chain, symbol)
            if row is None or math.isnan(float(row["delta"])):
                reason = _greeks_failure(strategy, snap, row, r, model)
                failures.append(f"{symbol}: {reason}")
            else:
                rows.append((positions[symbol] * snap.multiplier, row))
    for symbol in linear:
        if positions.get(symbol, 0.0) and symbol not in strategy._last_prices:
            failures.append(f"{symbol}: 持有该线性品种但没有最新价")
    if failures:
        raise ValueError("以下持仓无法计算 Greeks: " + "; ".join(failures))
    for units, row in rows:
        for name in ("delta", "gamma", "vega", "theta"):
            total[name] += units * float(row[name])
        total["cash_delta"] += units * float(row["delta"]) * float(row["forward"])
    for symbol in linear:
        qty = positions.get(symbol, 0.0)
        if not qty:
            continue
        units = qty * strategy.get_instrument(symbol).multiplier
        total["delta"] += units
        total["cash_delta"] += units * float(strategy._last_prices[symbol])
    return total


def hedge_delta(
    strategy: Any,
    underlying: str,
    hedge_symbol: Optional[str] = None,
    band: float = 0.0,
    rate: Optional[float] = None,
    model: OptionModel = "black76",
    **kwargs: Any,
) -> Optional[str]:
    """用线性品种把标的相关持仓的现金 Delta 调回 0, 返回对冲订单 ID."""
    hedge = hedge_symbol or underlying
    price = strategy._last_prices.get(hedge)
    if price is None:
        raise ValueError(f"对冲品种 {hedge} 没有最新价, 无法计算对冲数量")
    if price <= 0:
        raise ValueError(f"对冲品种 {hedge} 最新价 {price:g} 不为正, 无法计算对冲数量")
    snap = strategy.get_instrument(hedge)
    greeks = portfolio_greeks(
        strategy, underlying, include=[hedge], rate=rate, model=model
    )
    change = -greeks["cash_delta"] / (price * snap.multiplier)
    lot = snap.lot_size or 1.0
    change = round(change / lot) * lot
    if change == 0 or abs(change) <= band:
        return None
    target = strategy.get_position(hedge) + change
    return cast(Optional[str], strategy.order_target(hedge, target, **kwargs))
