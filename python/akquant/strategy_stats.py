"""运行时绩效统计 ``Strategy.stats``：数值取自引擎，口径与 ``result.metrics`` 一致.

取数来源有两条（见 spec 5.3）：回调运行期间读 ``ctx.strategy_stats()``，此时引擎被
``run()`` 独占借用；``on_stop`` 中读 ``Engine.strategy_stats()``，
此时 ``run()`` 已返回，期末点的补法与 ``get_results()`` 相同。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional, Sequence

_INT_FIELDS = frozenset(
    {"winning_trades", "losing_trades", "trade_count", "max_wins", "max_losses"}
)
_PCT_FIELDS = frozenset(
    {
        "total_return_pct",
        "max_drawdown_pct",
        "win_rate",
        "loss_rate",
        "current_drawdown_pct",
    }
)

DEFAULT_ROW_FIELDS: tuple[str, ...] = (
    "total_return_pct",
    "max_drawdown_pct",
    "win_rate",
    "profit_factor",
    "trade_count",
    "total_pnl",
)

DEFAULT_LABELS: dict[str, str] = {
    "total_return_pct": "累计收益",
    "total_pnl": "总盈亏",
    "unrealized_pnl": "浮动盈亏",
    "total_commission": "手续费",
    "max_drawdown": "最大回撤(比例)",
    "max_drawdown_pct": "最大回撤",
    "max_drawdown_value": "最大回撤金额",
    "win_rate": "胜率",
    "loss_rate": "败率",
    "winning_trades": "盈利笔数",
    "losing_trades": "亏损笔数",
    "profit_factor": "盈亏比",
    "avg_pnl": "平均盈亏",
    "avg_profit": "平均盈利",
    "avg_loss": "平均亏损",
    "largest_win": "最大单笔盈利",
    "largest_loss": "最大单笔亏损",
    "max_wins": "最大连胜",
    "max_losses": "最大连亏",
    "trade_count": "交易次数",
    "current_drawdown_pct": "当前回撤",
    "equity": "权益",
}


@dataclass(frozen=True)
class StrategyStats:
    """一次读取时的统计快照（冻结）；字段含义见 docs/zh/guide/visualization.md."""

    total_return_pct: Optional[float]
    total_pnl: Optional[float]
    unrealized_pnl: Optional[float]
    total_commission: Optional[float]
    max_drawdown: Optional[float]
    max_drawdown_pct: Optional[float]
    max_drawdown_value: Optional[float]
    win_rate: Optional[float]
    loss_rate: Optional[float]
    winning_trades: Optional[int]
    losing_trades: Optional[int]
    profit_factor: Optional[float]
    avg_pnl: Optional[float]
    avg_profit: Optional[float]
    avg_loss: Optional[float]
    largest_win: Optional[float]
    largest_loss: Optional[float]
    max_wins: Optional[int]
    max_losses: Optional[int]
    trade_count: Optional[int]
    current_drawdown_pct: Optional[float]
    equity: Optional[float]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StrategyStats":
        """从引擎返回的字典构造；缺失字段为 None，计数类字段转 int."""
        values: dict[str, Any] = {}
        for f in fields(cls):
            v = raw.get(f.name)
            if v is None:
                values[f.name] = None
            elif f.name in _INT_FIELDS:
                values[f.name] = int(v)
            else:
                values[f.name] = float(v)
        return cls(**values)

    def as_rows(
        self,
        fields: Optional[Sequence[str]] = None,
        labels: Optional[Mapping[str, str]] = None,
    ) -> list[list[dict[str, Any]]]:
        """转成 ``draw.table(rows=...)`` 可直接用的二维单元格列表（名称、数值两列）."""
        names = list(fields) if fields is not None else list(DEFAULT_ROW_FIELDS)
        label_map = {**DEFAULT_LABELS, **(dict(labels) if labels else {})}
        rows: list[list[dict[str, Any]]] = []
        for name in names:
            if name not in DEFAULT_LABELS:
                raise ValueError(
                    f"未知统计字段 {name!r}，可选: {', '.join(DEFAULT_LABELS)}"
                )
            value = getattr(self, name)
            rows.append(
                [
                    {"value": label_map[name]},
                    {"value": value, "text": _format(name, value)},
                ]
            )
        return rows


def _format(name: str, value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "—"
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    if name in _PCT_FIELDS:
        return f"{value:.2f}%"
    return f"{value:,.2f}"


def build_strategy_stats(strategy: Any) -> StrategyStats:
    """按当前所处阶段选择取数来源，构造 ``StrategyStats``."""
    broker_stats = _broker_live_stats(strategy)
    if broker_stats is not None:
        return broker_stats
    if getattr(strategy, "_stats_final", False):
        engine = getattr(strategy, "_engine", None)
        method = getattr(engine, "strategy_stats", None)
        if callable(method):
            try:
                return StrategyStats.from_mapping(method())
            except RuntimeError:
                # 实盘某些收尾路径里引擎仍被借用：退回 ctx（数值是最后一次回调时的状态）
                pass
    ctx = getattr(strategy, "ctx", None)
    method = getattr(ctx, "strategy_stats", None)
    if callable(method):
        return StrategyStats.from_mapping(method())
    raise RuntimeError("self.stats 只能在策略回调或 on_stop 中读取")


def _broker_live_stats(strategy: Any) -> Optional[StrategyStats]:
    """broker_live：由会话级统计器（柜台成交驱动）给出，见 spec 5.5.

    统计器不可用（启动时柜台权益/持仓查询失败）时只给 equity（柜台权益；
    账户也不可用时为 None），其余为 None。
    浮动盈亏按策略已见到的最新收盘价（``_last_prices``）计算。
    """
    from .gateway.broker_execution import BrokerExecution

    execution = getattr(strategy, "execution", None)
    if not isinstance(execution, BrokerExecution):
        return None
    # 账户从未加载成功（柜台不可用）时权益未知：报 None，不能报 0.0
    account = execution.state_cache.account()
    equity: Optional[float] = (
        None if account is None else float(getattr(account, "equity", 0.0) or 0.0)
    )
    raw: Mapping[str, Any] = {"equity": equity}
    stats = execution.trade_stats
    if stats is not None and equity is not None:
        prices = {
            str(symbol): float(price)
            for symbol, price in (getattr(strategy, "_last_prices", None) or {}).items()
        }
        snap = stats.snapshot(_current_timestamp_ns(strategy), equity, prices)
        if snap is not None:
            raw = snap
    return StrategyStats.from_mapping(raw)


def _current_timestamp_ns(strategy: Any) -> int:
    """当前 bar/tick/ctx 的时间戳；都没有时（如收尾阶段）用墙钟."""
    resolve = getattr(strategy, "_resolve_record_timestamp", None)
    if callable(resolve):
        try:
            return int(resolve())
        except RuntimeError:
            pass
    return time.time_ns()
