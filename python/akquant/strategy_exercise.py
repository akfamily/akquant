"""策略侧期货期权行权申请: 校验后交给引擎, 在当日日终结算时执行."""

from typing import Any, Optional

import pandas as pd


def exercise(strategy: Any, symbol: str, quantity: Optional[float] = None) -> None:
    """校验并提交期货期权行权申请; 不满足条件时报出确切原因."""
    if getattr(strategy, "_live_market_data_owner", False):
        raise RuntimeError(
            "实盘不支持由策略提交行权申请(尚未接入 CTP 行权指令), 请在交易终端提交"
        )
    snap = strategy.get_instrument(symbol)
    if snap.asset_type != "OPTION":
        raise ValueError(f"{symbol} 不是期权合约, 不能行权")
    underlying = snap.underlying_symbol
    underlying_snap = strategy.get_instruments().get(underlying) if underlying else None
    if underlying_snap is None or underlying_snap.asset_type != "FUTURES":
        raise ValueError(
            f"{symbol} 的标的 {underlying} 不是登记为期货的合约: 只有期货期权(美式)"
            "可以提前行权, 股指期权与 ETF 期权到期按结算价结算"
        )
    now = strategy.now
    if now is None:
        raise ValueError(f"{symbol} 行权失败: 当前没有行情时间")
    if snap.expiry_date is not None and int(pd.Timestamp(now).strftime("%Y%m%d")) > int(
        snap.expiry_date
    ):
        raise ValueError(f"{symbol} 已于 {snap.expiry_date} 到期, 不能再行权")
    long = float(strategy.get_position(symbol))
    if long <= 0:
        raise ValueError(f"{symbol} 没有多头持仓(当前 {long:g}), 只有买方可以行权")
    qty = long if quantity is None else float(quantity)
    if qty <= 0 or qty != int(qty):
        raise ValueError(f"{symbol} 行权数量须为正整数手, 实际: {qty:g}")
    if qty > long:
        raise ValueError(f"{symbol} 申请行权 {qty:g} 手, 超过多头持仓 {long:g} 手")
    strategy.ctx.request_exercise(symbol, qty)
