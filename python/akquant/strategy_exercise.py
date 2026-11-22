"""策略侧期货期权行权与放弃行权申请: 校验后交给执行后端.

回测与 paper 由引擎在当日日终结算时执行; 实盘(broker_live)经柜台提交行权指令。
"""

from typing import Any, Optional

import pandas as pd


def exercise(strategy: Any, symbol: str, quantity: Optional[float] = None) -> None:
    """校验并提交期货期权行权申请; 不满足条件时报出确切原因."""
    strategy.execution.exercise(symbol, _checked_quantity(strategy, symbol, quantity))


def abandon_exercise(
    strategy: Any, symbol: str, quantity: Optional[float] = None
) -> None:
    """校验并提交到期日放弃行权申请; 不满足条件时报出确切原因."""
    expiry = strategy.get_instrument(symbol).expiry_date
    qty = _checked_quantity(strategy, symbol, quantity, action="放弃行权")
    today = int(pd.Timestamp(strategy.now).strftime("%Y%m%d"))
    if expiry is None:
        raise ValueError(f"{symbol} 没有登记到期日, 无法判定今天是否到期日")
    if today != int(expiry):
        raise ValueError(
            f"{symbol} 到期日为 {expiry}, 今天 {today}: 只能在到期日申请放弃行权"
        )
    strategy.execution.abandon_exercise(symbol, qty)


def _checked_quantity(
    strategy: Any, symbol: str, quantity: Optional[float], action: str = "行权"
) -> float:
    """核对期货期权多头可以行权(放弃行权), 返回手数."""
    snap = strategy.get_instrument(symbol)
    if snap.asset_type != "OPTION":
        raise ValueError(f"{symbol} 不是期权合约, 不能{action}")
    underlying = snap.underlying_symbol
    underlying_snap = strategy.get_instruments().get(underlying) if underlying else None
    if underlying_snap is None or underlying_snap.asset_type != "FUTURES":
        raise ValueError(
            f"{symbol} 的标的 {underlying} 不是登记为期货的合约: 只有期货期权(美式)"
            "可以提前行权, 股指期权与 ETF 期权到期按结算价结算"
        )
    now = strategy.now
    if now is None:
        raise ValueError(f"{symbol} {action}失败: 当前没有行情时间")
    if snap.expiry_date is not None and int(pd.Timestamp(now).strftime("%Y%m%d")) > int(
        snap.expiry_date
    ):
        raise ValueError(f"{symbol} 已于 {snap.expiry_date} 到期, 不能再{action}")
    long = float(strategy.get_position(symbol))
    if long <= 0:
        raise ValueError(f"{symbol} 没有多头持仓(当前 {long:g}), 只有买方可以{action}")
    qty = long if quantity is None else float(quantity)
    if qty <= 0 or qty != int(qty):
        raise ValueError(f"{symbol} {action}数量须为正整数手, 实际: {qty:g}")
    if qty > long:
        raise ValueError(f"{symbol} 申请{action} {qty:g} 手, 超过多头持仓 {long:g} 手")
    return qty


def cancel_exercise(strategy: Any, symbol: str) -> float:
    """撤销该期权合约尚未执行的行权与放弃行权, 返回撤销的手数."""
    return float(strategy.execution.cancel_exercise(symbol))
