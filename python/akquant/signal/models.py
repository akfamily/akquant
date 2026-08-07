"""外部信号的数据契约."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SignalAction(str, Enum):
    """信号动作."""

    BUY = "buy"
    SELL = "sell"


class SignalStatus(str, Enum):
    """信号处理结果."""

    ACCEPTED = "accepted"
    """已投递给引擎/柜台(不代表已成交)."""

    DUPLICATE = "duplicate"
    """signal_id 重复, 已幂等丢弃."""

    REJECTED = "rejected"
    """被风控或柜台拒绝(同步可知的那部分)."""

    ERROR = "error"
    """处理过程抛异常."""


class Signal(BaseModel):
    """一条外部交易指令.

    ``signal_id`` 是幂等键: 平台重连/重推同一 id 只会下一次单。缺失时无法去重,
    调度器会告警并按"每次都是新信号"处理 —— 这是危险的默认, 平台侧应始终提供。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: str = Field(min_length=1, description="幂等键, 平台侧唯一")
    symbol: str = Field(min_length=1, description="标的代码")
    # 标注含 str 是有意的: 平台侧多以 "buy"/"sell" 字符串传入, pydantic 会强制转成
    # 枚举。只标 SignalAction 会让字符串调用点在 mypy 下报错, 而那是主要用法。
    action: Union[SignalAction, str] = Field(description="买/卖")
    quantity: float = Field(gt=0, description="委托数量")
    price: Optional[float] = Field(default=None, gt=0, description="委托价; 省略为市价")
    order_type: Optional[str] = Field(default=None, description="Limit / Market")
    strategy_id: str = Field(default="_default", description="归属策略, 风控按它路由")
    timestamp: Optional[int] = Field(default=None, description="平台生成时刻(纳秒)")
    tag: Optional[str] = Field(default=None, description="附加标记")

    @field_validator("symbol", "signal_id", "strategy_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        """去空白并拒绝纯空白值."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("不能为空白")
        return stripped

    @field_validator("order_type")
    @classmethod
    def _check_order_type(cls, value: Optional[str]) -> Optional[str]:
        """只接受 Limit / Market(大小写不敏感), 归一为首字母大写."""
        if value is None:
            return None
        normalized = value.strip().capitalize()
        if normalized not in {"Limit", "Market"}:
            raise ValueError(f"order_type 仅支持 Limit / Market, 收到 {value!r}")
        return normalized

    @field_validator("action")
    @classmethod
    def _coerce_action(cls, value: Union[SignalAction, str]) -> SignalAction:
        """把字符串归一为枚举, 未知值报错."""
        if isinstance(value, SignalAction):
            return value
        try:
            return SignalAction(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"action 仅支持 buy / sell, 收到 {value!r}") from exc

    @property
    def side(self) -> str:
        """网关口径的方向字符串(``Buy`` / ``Sell``)."""
        return "Buy" if self.action is SignalAction.BUY else "Sell"


class SignalResult(BaseModel):
    """一条信号的处理结果(同步可知的部分).

    柜台异步拒单不在此列 —— 那条路径经 ``SignalDispatcher.handle_reject`` 回吐。
    """

    model_config = ConfigDict(frozen=True)

    signal_id: str
    status: SignalStatus
    order_id: Optional[str] = None
    reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        """是否已被接受."""
        return self.status is SignalStatus.ACCEPTED

    def as_dict(self) -> dict[str, Any]:
        """转普通 dict(便于回执给平台)."""
        return self.model_dump(mode="json")
