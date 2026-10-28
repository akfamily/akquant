"""0.3.71 发布版生成的分钟线旧存档能被当前版本续跑（交易时段重构阶段 3 回归）."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from _checkpoint_0371 import PHASE2, SYMBOL, TZ, HoldThenSell, minute_bars
from akquant import run_from_checkpoint

FIXTURE = Path(__file__).parent / "fixtures" / "checkpoint_0371_minute.pkl"

# 0.3.71 第一段的结果（生成时打印）：09:31 下单、09:32 开盘 10.1 成交 100 股。
CASH_AT_CHECKPOINT = 98_990.0


def test_resume_minute_checkpoint_from_0371(tmp_path: Path) -> None:
    """载入 0.3.71 存档：现金与持仓保留，次日卖出照常成交."""
    ckpt = tmp_path / "ckpt.pkl"
    shutil.copy(FIXTURE, ckpt)
    r = run_from_checkpoint(
        checkpoint_path=str(ckpt),
        data=minute_bars(*PHASE2),
        symbols=[SYMBOL],
        show_progress=False,
    )
    strategy = r.strategy
    assert isinstance(strategy, HoldThenSell)
    # 存档里的持仓（T+1 后可卖）与策略状态都保留下来
    assert strategy.bought is True
    assert strategy.position_at_resume == pytest.approx(100.0)
    orders = r.orders_df
    sells = orders[(orders["side"] == "sell") & (orders["filled_quantity"] > 0)]
    assert len(sells) == 1
    sell = sells.iloc[0]
    assert float(sell["filled_quantity"]) == pytest.approx(100.0)
    # 09:31 下单，NextOpen 在 09:32 按开盘价 11.1 成交
    filled_at = pd.Timestamp(sell["updated_at"]).tz_convert(TZ)
    assert filled_at.strftime("%m-%d %H:%M") == "01-03 09:32"
    assert float(sell["avg_price"]) == pytest.approx(11.1)
    assert strategy.get_position(SYMBOL) == pytest.approx(0.0)
    # 现金 = 存档现金 + 卖出所得 - 费用
    fees = float(sell["commission"])
    assert r.engine is not None
    assert float(r.engine.portfolio.cash) == pytest.approx(
        CASH_AT_CHECKPOINT + 100 * 11.1 - fees
    )
