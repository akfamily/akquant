"""期权 Greeks 限额: 下单前按标的投影组合 Greeks, 超限且增加暴露的委托拒绝.

限额配置在策略类属性 ``option_greek_limits``(``{标的: {Greek: 上限}}``)。下单前的
组合 Greeks 按标的下各合约的投影持仓(结算持仓 + 本策略全部在途委托, 含本次回调
刚提交的)乘以每手 Greeks 求和, 口径同 ``portfolio_greeks``(已乘数量与合约乘数);
加上本笔委托即下单后的值。下单后绝对值超过上限且大于下单前时拒绝, 减小暴露的
委托放行。回测与实盘共用本函数, 拒单分别经引擎订单账本与柜台下单器回吐
``on_reject``。
"""

import math
from typing import Any, Dict, List, Mapping, Optional

from .strategy_option import get_greeks

GREEK_NAMES = ("delta", "gamma", "vega", "theta", "cash_delta")


def _validated_limits(underlying: str, limits: Any) -> Dict[str, float]:
    if not isinstance(limits, Mapping):
        raise ValueError(
            f"option_greek_limits[{underlying!r}] 须为 {{Greek: 上限}} 字典, "
            f"实际: {type(limits).__name__}"
        )
    checked: Dict[str, float] = {}
    for name, value in limits.items():
        if name not in GREEK_NAMES:
            raise ValueError(
                f"option_greek_limits[{underlying!r}] 的键 {name!r} 不是 "
                f"{', '.join(GREEK_NAMES)} 之一"
            )
        limit = float(value)
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError(
                f"option_greek_limits[{underlying!r}][{name!r}] 须为正数, "
                f"实际: {value!r}"
            )
        checked[name] = limit
    return checked


def _unit_greeks(strategy: Any, symbol: str, underlying: str) -> Dict[str, float]:
    """每手 Greeks(已乘合约乘数); 标的本身 delta 为 1、现金 delta 为最新价."""
    multiplier = float(strategy.get_instrument(symbol).multiplier)
    if symbol == underlying:
        price = strategy._last_prices.get(symbol)
        if price is None:
            raise ValueError(f"{symbol} 没有最新价")
        unit = {"delta": 1.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        unit["cash_delta"] = float(price)
    else:
        greeks = get_greeks(strategy, symbol)
        unit = {name: greeks[name] for name in ("delta", "gamma", "vega", "theta")}
        unit["cash_delta"] = greeks["delta"] * greeks["forward"]
    return {name: value * multiplier for name, value in unit.items()}


def greek_limit_violation(
    strategy: Any, symbol: str, side: str, quantity: float
) -> Optional[str]:
    """校验本笔委托是否使标的组合 Greeks 超限, 返回拒单原因(None 表示通过).

    :param strategy: 策略实例
    :param symbol: 委托标的(期权, 或设了限额的标的本身)
    :param side: ``"Buy"`` / ``"Sell"``
    :param quantity: 委托数量(正数)
    :raises ValueError: ``option_greek_limits`` 配置不合法
    """
    by_underlying = getattr(strategy, "option_greek_limits", None)
    if not by_underlying:
        return None
    instruments = strategy.get_instruments()
    snap = instruments.get(symbol)
    underlying = (
        snap.underlying_symbol
        if snap is not None and snap.asset_type == "OPTION"
        else symbol
    )
    if underlying not in by_underlying:
        return None
    limits = _validated_limits(underlying, by_underlying[underlying])
    related: List[str] = [
        s
        for s, item in instruments.items()
        if item.asset_type == "OPTION" and item.underlying_symbol == underlying
    ]
    if underlying in instruments:
        related.append(underlying)
    before = dict.fromkeys(GREEK_NAMES, 0.0)
    try:
        for held in related:
            projected = float(strategy.execution.get_projected_position(held))
            if projected:
                unit = _unit_greeks(strategy, held, underlying)
                for name in GREEK_NAMES:
                    before[name] += projected * unit[name]
        unit = _unit_greeks(strategy, symbol, underlying)
    except ValueError as exc:
        return f"期权 Greeks 限额无法校验(标的 {underlying}): {exc}"
    sign = 1.0 if str(side).strip().lower() == "buy" else -1.0
    breaches = []
    for name, limit in limits.items():
        after = before[name] + sign * float(quantity) * unit[name]
        if abs(after) > limit and abs(after) > abs(before[name]):
            breaches.append(
                f"{name} 下单后 {after:.6g}, 超过上限 {limit:g}"
                f"(下单前 {before[name]:.6g}, 含在途委托)"
            )
    if not breaches:
        return None
    return f"期权 Greeks 限额(标的 {underlying}): " + "; ".join(breaches)


def greek_limit_reject(
    strategy: Any, symbol: str, side: str, quantity: float, price: Any, tag: Any
) -> Optional[List[str]]:
    """回测下单入口的 Greeks 限额: 对用户提交的这一笔逻辑委托校验(拆腿之前).

    超限时作为风控拒单交给引擎(进订单账本, 触发 ``on_reject``), 不撮合, 返回拒单
    的订单号; 通过时返回 None。
    """
    reason = greek_limit_violation(strategy, symbol, side, float(quantity))
    if reason is None:
        return None
    return [
        str(strategy.ctx.reject_order(symbol, side, quantity, reason, price, tag or ""))
    ]
