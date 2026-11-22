"""多腿组合单: 一次提交多条腿, 整组编号管理, 一条腿失败时撤掉同组未成交的腿.

国内期权没有组合合约, 各腿作为独立委托报出、分别风控与撮合(盘中柜台按单腿保证金
冻结, 回测相同)。组合单只负责编组与腿风险保护: 任一腿被拒、被撤或过期且未全部
成交时, 撤掉同组其余未完成的委托; 已成交的腿不回滚, ``get_combo`` 显示组合状态。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

TERMINAL_FAILED = ("rejected", "cancelled", "canceled", "expired")


@dataclass(frozen=True)
class ComboLegStatus:
    """组合单一条腿的状态.

    :ivar symbol: 合约
    :ivar side: ``"Buy"`` / ``"Sell"``
    :ivar quantity: 本腿目标数量
    :ivar filled: 已成交数量
    :ivar status: ``working`` 未完成 / ``filled`` 全部成交 / ``failed`` 委托已终结
        且未全部成交(含前面的腿当场失败而未报出) / ``unknown`` 报单结果不可知
        (超时、断连, 不得当作不存在而重发)
    :ivar order_ids: 本腿的委托号(一条腿可能拆成平仓、开仓两笔)
    """

    symbol: str
    side: str
    quantity: float
    filled: float
    status: str
    order_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ComboStatus:
    """组合单状态.

    :ivar group_id: 组号
    :ivar state: ``working`` 进行中 / ``filled`` 全部成交 / ``broken`` 有腿失败且
        已有成交(组合不完整, 需策略自行处理) / ``cancelled`` 有腿失败且没有任何
        成交 / ``unknown`` 有腿报单结果不可知
    :ivar legs: 各腿状态
    """

    group_id: str
    state: str
    legs: Tuple[ComboLegStatus, ...]


def _status_name(status: Any) -> str:
    return str(status).split(".")[-1].strip().lower()


def _validated_legs(
    legs: Sequence[Tuple[str, float]],
    quantity: float,
    prices: Optional[Mapping[str, float]],
) -> List[Tuple[str, str, float]]:
    if len(legs) < 2:
        raise ValueError(f"组合单至少需要 2 条腿, 实际 {len(legs)} 条")
    if quantity <= 0 or quantity != int(quantity):
        raise ValueError(f"组合单 quantity(组数) 须为正整数, 实际: {quantity!r}")
    resolved: List[Tuple[str, str, float]] = []
    seen = set()
    for item in legs:
        symbol, ratio = str(item[0]).strip(), float(item[1])
        if symbol in seen:
            raise ValueError(f"组合单的腿 {symbol} 重复")
        seen.add(symbol)
        if ratio == 0 or ratio != int(ratio):
            raise ValueError(
                f"组合单的腿 {symbol} 每组手数须为非零整数, 实际: {item[1]!r}"
            )
        resolved.append((symbol, "Buy" if ratio > 0 else "Sell", abs(ratio) * quantity))
    for symbol, price in (prices or {}).items():
        if symbol not in seen:
            raise ValueError(f"prices 中的 {symbol} 不是组合单的腿")
        if float(price) <= 0:
            raise ValueError(f"组合单的腿 {symbol} 限价须为正数, 实际: {price!r}")
    return resolved


def submit_combo(
    strategy: Any,
    legs: Sequence[Tuple[str, float]],
    quantity: float = 1,
    prices: Optional[Mapping[str, float]] = None,
    tag: Optional[str] = None,
) -> str:
    """提交组合单, 返回组号."""
    resolved = _validated_legs(legs, quantity, prices)
    with strategy._order_group_lock:
        strategy._order_group_seq += 1
        group_id = f"combo-{strategy._order_group_seq}"
    combo: Dict[str, Any] = {
        "legs": [
            {
                "symbol": symbol,
                "side": side,
                "quantity": qty,
                "order_ids": (),
                "failure": None,
            }
            for symbol, side, qty in resolved
        ],
        "orders": {},
        "breaking": False,
    }
    strategy._combos[group_id] = combo
    for leg in combo["legs"]:
        if combo["breaking"]:
            leg["failure"] = "not_submitted"
            continue
        submit = strategy.buy if leg["side"] == "Buy" else strategy.sell
        receipt = submit(
            leg["symbol"],
            leg["quantity"],
            price=(prices or {}).get(leg["symbol"]),
            tag=tag,
        )
        leg["order_ids"] = tuple(str(i) for i in receipt.order_ids)
        for order_id in leg["order_ids"]:
            combo["orders"][order_id] = ("submitted", 0.0)
            strategy._combo_order_to_group[order_id] = group_id
        if not leg["order_ids"]:
            # 本腿当场被拒或结果不可知: 余下的腿不再报, 撤已报的腿
            leg["failure"] = receipt.failure or "rejected"
            _break_combo(strategy, group_id)
    strategy._order_group_members[group_id] = tuple(combo["orders"])
    return group_id


def _break_combo(strategy: Any, group_id: str) -> None:
    combo = strategy._combos[group_id]
    if combo["breaking"]:
        return
    combo["breaking"] = True
    for order_id, (status, _) in combo["orders"].items():
        if status in ("filled", *TERMINAL_FAILED):
            continue
        strategy.execution.cancel_order(order_id)


def process_combo_order(strategy: Any, order: Any) -> None:
    """委托状态更新: 记录组合腿的状态与成交; 腿失败时撤掉同组未完成的委托."""
    order_id = str(getattr(order, "id", ""))
    group_id = strategy._combo_order_to_group.get(order_id)
    if group_id is None:
        return
    combo = strategy._combos[group_id]
    status = _status_name(order.status)
    filled = float(getattr(order, "filled_quantity", 0.0) or 0.0)
    combo["orders"][order_id] = (status, filled)
    if status in TERMINAL_FAILED:
        _break_combo(strategy, group_id)


def cancel_combo(strategy: Any, group_id: str) -> bool:
    """撤销组合单未完成的委托; ``group_id`` 不是组合单时返回 False."""
    if group_id not in getattr(strategy, "_combos", {}):
        return False
    _break_combo(strategy, group_id)
    return True


def get_combo(strategy: Any, group_id: str) -> ComboStatus:
    """组合单当前状态."""
    combo = strategy._combos.get(group_id)
    if combo is None:
        raise KeyError(f"没有组合单 {group_id}")
    leg_states: List[ComboLegStatus] = []
    for leg in combo["legs"]:
        orders = [combo["orders"][i] for i in leg["order_ids"]]
        filled = sum(f for _, f in orders)
        if leg["failure"] == "unknown":
            status = "unknown"
        elif filled >= leg["quantity"]:
            status = "filled"
        elif leg["failure"] is not None or (
            orders and all(s in (*TERMINAL_FAILED, "filled") for s, _ in orders)
        ):
            status = "failed"
        else:
            status = "working"
        leg_states.append(
            ComboLegStatus(
                symbol=leg["symbol"],
                side=leg["side"],
                quantity=leg["quantity"],
                filled=filled,
                status=status,
                order_ids=leg["order_ids"],
            )
        )
    statuses = {leg.status for leg in leg_states}
    if "unknown" in statuses:
        state = "unknown"
    elif statuses == {"filled"}:
        state = "filled"
    elif "failed" in statuses or combo["breaking"]:
        state = "broken" if any(leg.filled > 0 for leg in leg_states) else "cancelled"
    else:
        state = "working"
    return ComboStatus(group_id=group_id, state=state, legs=tuple(leg_states))
