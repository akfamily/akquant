"""策略侧期权工具: 期权链、选合约、Greeks 与 Delta 对冲."""

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

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
from .option.surface import SmileModel, VolSurface, fit_surface

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


def _quote_prices(
    strategy: Any, symbols: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(买一价, 卖一价, 定价用价格) 三个数组, 缺失为 nan.

    定价用价格: ``option_price_source="last"`` 取最新价; ``"mid"`` 取中间价, 仅当
    买一价、卖一价都为正且买价不高于卖价(Cboe VIX 的取价规则)。
    """
    source = _price_source(strategy)
    quotes = strategy._last_quotes
    no_quote = (None, None)
    book = np.array(
        [quotes.get(symbol, no_quote) for symbol in symbols], dtype=float
    ).reshape(-1, 2)
    bid, ask = book[:, 0], book[:, 1]
    if source == "last":
        last = strategy._last_prices
        price = np.array(
            [last.get(symbol, math.nan) for symbol in symbols], dtype=float
        )
        return bid, ask, price
    valid = (bid > 0) & (ask > 0) & (bid <= ask)
    return bid, ask, np.where(valid, 0.5 * (bid + ask), np.nan)


def _mid_failure(bid: float, ask: float) -> str:
    """中间价取不到的确切原因, 与 :func:`_quote_prices` 的判定一致."""
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


def _option_snapshots(
    instruments: Dict[str, Any], underlying: Optional[str]
) -> List[Any]:
    """已登记的期权合约; ``underlying`` 为 None 时取全部标的."""
    return [
        snap
        for snap in instruments.values()
        if snap.asset_type == "OPTION"
        and snap.underlying_symbol is not None
        and (underlying is None or snap.underlying_symbol == underlying)
        and snap.expiry_date is not None
        and snap.strike_price is not None
        and snap.option_type is not None
    ]


def get_expiries(strategy: Any, underlying: str) -> List[int]:
    """返回标的下尚未到期的期权到期日(YYYYMMDD), 由近到远."""
    snaps = _option_snapshots(strategy.get_instruments(), underlying)
    expiries = {snap.expiry_date for snap in snaps}
    return sorted(e for e in expiries if _expiry_t(strategy, e) > 0)


def _forward(
    strategy: Any,
    underlying_snap: Any,
    underlying: str,
    strikes: np.ndarray,
    types: np.ndarray,
    prices: np.ndarray,
    t: float,
    rate: float,
) -> float:
    """远期价格: 期货标的取期货价; 否则按平价反推, 缺报价时退回现货按利率前推."""
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


def _price_chains(
    strategy: Any,
    instruments: Dict[str, Any],
    snaps: List[Any],
    rate: float,
    model: OptionModel,
) -> pd.DataFrame:
    """一次批量计算多条期权链, 按标的、到期日、行权价、类型排序, 不含已到期合约."""
    underlyings = np.array([snap.underlying_symbol for snap in snaps], dtype=str)
    symbols = np.array([snap.symbol for snap in snaps], dtype=object)
    types = np.array([snap.option_type for snap in snaps], dtype=str)
    strikes = np.array([snap.strike_price for snap in snaps], dtype=float)
    expiries = np.array([snap.expiry_date for snap in snaps], dtype=np.int64)
    order = np.lexsort((types, strikes, expiries, underlyings))
    underlyings, symbols, types = underlyings[order], symbols[order], types[order]
    strikes, expiries = strikes[order], expiries[order]
    bids, asks, prices = _quote_prices(strategy, symbols)
    n = len(symbols)
    t = np.full(n, np.nan)
    forward = np.full(n, np.nan)
    # 排序后同一(标的, 到期日)的合约连续, 按段计算期限与远期
    starts_mask = np.ones(n, dtype=bool)
    starts_mask[1:] = (underlyings[1:] != underlyings[:-1]) | (
        expiries[1:] != expiries[:-1]
    )
    starts = np.flatnonzero(starts_mask)
    t_of: Dict[int, float] = {}
    for start, end in zip(starts, np.append(starts[1:], n)):
        expiry_date = int(expiries[start])
        if expiry_date not in t_of:
            t_of[expiry_date] = _expiry_t(strategy, expiry_date)
        t_e = t_of[expiry_date]
        if t_e > 0:
            seg = slice(start, end)
            underlying = str(underlyings[start])
            t[seg] = t_e
            forward[seg] = _forward(
                strategy,
                instruments.get(underlying),
                underlying,
                strikes[seg],
                types[seg],
                prices[seg],
                t_e,
                rate,
            )
    live = t > 0
    iv = np.asarray(
        implied_vol(
            prices[live],
            forward[live],
            strikes[live],
            t[live],
            rate,
            types[live],
            model,
        ),
        dtype=float,
    )
    greeks = black76_greeks(
        forward[live], strikes[live], t[live], rate, iv, types[live]
    )
    data: Dict[str, Any] = {
        "underlying": underlyings[live].astype(object),
        "symbol": symbols[live],
        "option_type": types[live].astype(object),
        "strike": strikes[live],
        "expiry_date": expiries[live],
        "t": t[live],
        "bid": bids[live],
        "ask": asks[live],
        "price": prices[live],
        "forward": forward[live],
        "iv": iv,
    }
    for name in ("delta", "gamma", "vega", "theta"):
        data[name] = np.where(np.isnan(iv), np.nan, greeks[name])
    return pd.DataFrame(data, columns=["underlying", *_CHAIN_COLUMNS])


def _chains(
    strategy: Any, underlying: Optional[str], rate: float, model: OptionModel
) -> pd.DataFrame:
    """期权链(含 underlying 列), 调用方不得修改返回值.

    ``option_refresh_interval > 0`` 时整批计算全部标的的期权链, 按行情时间在间隔内
    复用; 缓存按取价方式、利率、模型区分。为 0 时每次只算 ``underlying`` 的链。
    """
    interval = float(strategy.option_refresh_interval)
    if interval < 0:
        raise ValueError(f"option_refresh_interval 不能为负, 实际: {interval:g}")
    now = strategy.now
    if interval == 0 or now is None:
        instruments = strategy.get_instruments()
        snaps = _option_snapshots(instruments, underlying)
        return _price_chains(strategy, instruments, snaps, rate, model)
    key = (_price_source(strategy), rate, model)
    cached = strategy._option_chains_cache.get(key)
    if cached is None or not 0 <= (now - cached[0]).total_seconds() < interval:
        instruments = strategy.get_instruments()
        snaps = _option_snapshots(instruments, None)
        full = _price_chains(strategy, instruments, snaps, rate, model)
        # 已按标的排序, 记下各标的的行区间, 取用时切片
        names = full["underlying"].to_numpy()
        change = np.ones(len(names), dtype=bool)
        change[1:] = names[1:] != names[:-1]
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], len(names))
        spans = {str(names[s]): (int(s), int(e)) for s, e in zip(starts, ends)}
        cached = (now, full, spans)
        strategy._option_chains_cache[key] = cached
    _, full, spans = cached
    if underlying is None:
        return cast(pd.DataFrame, full)
    start, end = spans.get(underlying, (0, 0))
    return cast(pd.DataFrame, full.iloc[start:end])


def option_chains(
    strategy: Any,
    underlyings: Optional[Sequence[str]] = None,
    rate: Optional[float] = None,
    model: OptionModel = "black76",
) -> pd.DataFrame:
    """多个标的的期权链合并为一张表, 首列为 underlying."""
    chains = _chains(strategy, None, _resolve_rate(strategy, rate), model)
    if underlyings is not None:
        chains = chains[chains["underlying"].isin(list(underlyings))]
    return chains.reset_index(drop=True)


def option_chain(
    strategy: Any,
    underlying: str,
    expiry: Optional[int] = None,
    rate: Optional[float] = None,
    model: OptionModel = "black76",
) -> pd.DataFrame:
    """标的的期权链: 合约要素、报价、远期、隐含波动率与 Greeks."""
    chain = _chains(strategy, underlying, _resolve_rate(strategy, rate), model)
    if expiry is not None:
        chain = chain[chain["expiry_date"] == expiry]
    return chain.drop(columns="underlying").reset_index(drop=True)


def vol_surface(
    strategy: Any,
    underlying: Union[str, Sequence[str]],
    model: SmileModel = "svi",
    beta: float = 1.0,
    rate: Optional[float] = None,
    pricing_model: OptionModel = "black76",
) -> VolSurface:
    """由标的的期权链拟合波动率曲面; 与期权链同一批计算时复用上次的拟合结果."""
    names = (underlying,) if isinstance(underlying, str) else tuple(underlying)
    if not names:
        raise ValueError("underlying 不能为空")
    resolved = _resolve_rate(strategy, rate)
    chain = _chains(
        strategy, names[0] if len(names) == 1 else None, resolved, pricing_model
    )
    if len(names) > 1:
        chain = chain[chain["underlying"].isin(names)]
    if float(strategy.option_refresh_interval) == 0 or strategy.now is None:
        return fit_surface(chain, model, beta)
    chain_key = (_price_source(strategy), resolved, pricing_model)
    chain_time = strategy._option_chains_cache[chain_key][0]
    key = (chain_key, names, model, float(beta))
    cached = strategy._vol_surface_cache.get(key)
    if cached is None or cached[0] != chain_time:
        cached = (chain_time, fit_surface(chain, model, beta))
        strategy._vol_surface_cache[key] = cached
    return cast(VolSurface, cached[1])


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
        for snap in _option_snapshots(strategy.get_instruments(), underlying)
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
