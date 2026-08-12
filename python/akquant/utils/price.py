"""委托价与最小变动价位(tick size)的对齐工具."""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal

_ROUNDING_BY_DIRECTION = {
    "nearest": ROUND_HALF_UP,
    "down": ROUND_FLOOR,
    "up": ROUND_CEILING,
}


def round_to_tick(price: float, tick: float, direction: str = "nearest") -> float:
    """把委托价对齐到最小变动价位的整数倍.

    框架**不会**在下单路径上自动调用本函数——擅自改价是隐性成本, 见
    docs/zh/advanced/tick_size_alignment.md 的取舍说明。这里提供工具, 由策略
    显式调用。

    :param price: 原始委托价。
    :param tick: 最小变动价位; ``<=0`` 时原样返回 ``price``(部分柜台/模拟环境
        会给出 0, 直接相除会抛 DivisionByZero)。
    :param direction: ``nearest`` 四舍五入 / ``down`` 向下 / ``up`` 向上。买单
        用 ``down``、卖单用 ``up`` 可保证取整不把价格推向不利方向。
    :raises ValueError: ``direction`` 不在词表内。
    """
    rounding = _ROUNDING_BY_DIRECTION.get(str(direction).strip().lower())
    if rounding is None:
        supported = ", ".join(sorted(_ROUNDING_BY_DIRECTION))
        raise ValueError(f"direction 须为 {supported} 之一, 收到: {direction!r}")
    tick_dec = Decimal(str(tick))
    if tick_dec <= 0:
        return float(price)
    price_dec = Decimal(str(price))
    steps = (price_dec / tick_dec).quantize(Decimal("1"), rounding=rounding)
    return float(steps * tick_dec)
