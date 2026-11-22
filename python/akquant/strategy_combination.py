"""策略侧组合持仓申请/拆分: 校验持仓后交给执行后端.

实盘(broker_live)经柜台向交易所申请组合或拆分组合, 申请成功后按组合保证金收取;
回测与 paper 没有组合申请这一步, 组合保证金由 ``CHINA_COMMODITY_COMBINATION``
模型按交易所优先级自动配对计算, 这里只做与实盘相同的参数和持仓校验。
"""

from typing import Any, Iterable, List, Tuple


def _comb_legs(
    strategy: Any, code: str, legs: Iterable[Tuple[str, float]], quantity: float
) -> List[Tuple[str, float]]:
    """校验组合代码、两条腿和组数, 返回各腿带符号的总手数(每组手数 × 组数)."""
    if (
        not isinstance(code, str)
        or not code
        or "&" in code
        or any(c.isspace() for c in code)
    ):
        raise ValueError(f"组合代码须为不含空格和 & 的非空字符串, 实际: {code!r}")
    leg_list = list(legs)
    if len(leg_list) != 2:
        raise ValueError(f"组合持仓须恰好 2 条腿, 实际 {len(leg_list)} 条")
    if leg_list[0][0] == leg_list[1][0]:
        raise ValueError(f"组合持仓的两条腿相同: {leg_list[0][0]}")
    count = float(quantity)
    if count <= 0 or not count.is_integer():
        raise ValueError(f"quantity(组数) 须为正整数, 实际: {quantity}")
    result = []
    for symbol, lots in leg_list:
        per = float(lots)
        if per == 0 or not per.is_integer():
            raise ValueError(
                f"组合持仓的腿 {symbol} 每组手数须为非零整数, 实际: {lots}"
            )
        need = per * count
        detail = strategy.execution.get_position_detail(symbol)
        side, held = ("多", detail.long) if need > 0 else ("空", detail.short)
        if held.quantity < abs(need):
            raise ValueError(
                f"{symbol} {side}头持仓 {held.quantity:g} 手, "
                f"不足组合所需 {abs(need):g} 手"
            )
        result.append((symbol, need))
    return result


def combine_positions(
    strategy: Any, code: str, legs: Iterable[Tuple[str, float]], quantity: float = 1
) -> None:
    """校验并申请组合持仓."""
    strategy.execution.combine_positions(
        code, _comb_legs(strategy, code, legs, quantity), False
    )


def uncombine_positions(
    strategy: Any, code: str, legs: Iterable[Tuple[str, float]], quantity: float = 1
) -> None:
    """校验并申请拆分组合持仓."""
    strategy.execution.combine_positions(
        code, _comb_legs(strategy, code, legs, quantity), True
    )
