"""实盘策略状态持久化: 把策略声明的变量写入 JSON 文件, 重启后恢复.

策略类属性 ``state_variables`` 列出要保存的属性名。``run_live(state_file=...)``
启用后:

- 启动时, 在全部策略的 ``on_start`` 与历史预热回放之后, 校验声明的属性存在且能写入
  JSON, 再从文件恢复(文件不存在则从头开始);
- 每批成交派发 ``on_trade`` 之后、会话收尾 ``on_stop`` 之后、策略调用
  ``save_state()`` 时保存; 恢复完成之前(启动失败)不保存, 以免用初始值覆盖原文件。

持仓以柜台(paper 模式为引擎)为准, 不在此保存; 指标靠历史预热重算。文件按策略槽位
分组, 先写临时文件再替换, 写到一半中断不会损坏原文件。
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict

from ..log import get_logger

logger = get_logger("live.state")


def _declared(strategy: Any) -> tuple[str, ...]:
    return tuple(getattr(type(strategy), "state_variables", ()) or ())


class LiveStateStore:
    """一次实盘会话的策略状态文件."""

    def __init__(self, path: str | os.PathLike[str], targets: Dict[str, Any]) -> None:
        """绑定状态文件与 ``{策略槽位: 策略实例}``."""
        self.path = Path(path)
        self._targets = dict(targets)
        self._lock = threading.Lock()
        # 恢复完成之前不保存: 启动失败时收尾保存会用初始值覆盖上一次的状态
        self._ready = False

    def _snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            slot: {name: getattr(strategy, name) for name in _declared(strategy)}
            for slot, strategy in self._targets.items()
            if _declared(strategy)
        }

    def _unserializable(self, snapshot: dict[str, dict[str, Any]]) -> str | None:
        """第一个不能写入 JSON 的变量, 返回确切原因; 都能写入时为 None."""
        for slot, values in snapshot.items():
            for name, value in values.items():
                try:
                    json.dumps(value)
                except (TypeError, ValueError) as exc:
                    return (
                        f"策略 {slot} 的状态变量 {name} 不能写入 JSON"
                        f"(类型 {type(value).__name__}: {exc})"
                    )
        return None

    def validate(self) -> None:
        """校验各策略声明的状态变量存在且能写入 JSON; 不满足时抛 ``ValueError``."""
        for slot, strategy in self._targets.items():
            for name in _declared(strategy):
                if not hasattr(strategy, name):
                    raise ValueError(
                        f"策略 {slot}({type(strategy).__name__}) 的 state_variables "
                        f"中的 {name} 不是策略属性(须在 __init__ 或 on_start 中赋值)"
                    )
        reason = self._unserializable(self._snapshot())
        if reason is not None:
            raise ValueError(reason)

    def restore(self) -> None:
        """从文件恢复各策略声明的变量; 文件不存在时不做任何事."""
        if not self.path.exists():
            logger.info("策略状态文件 %s 不存在, 从初始状态开始", self.path)
            self._ready = True
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        saved = data.get("slots", {})
        for slot, strategy in self._targets.items():
            values = saved.get(slot, {})
            restored = [name for name in _declared(strategy) if name in values]
            for name in restored:
                setattr(strategy, name, values[name])
            missing = [name for name in _declared(strategy) if name not in values]
            logger.info(
                "策略 %s 从 %s(保存于 %s)恢复状态变量 %s%s",
                slot,
                self.path,
                data.get("saved_at", "未知时间"),
                restored,
                f", 文件中没有 {missing}, 保持初始值" if missing else "",
            )
        for slot in sorted(set(saved) - set(self._targets)):
            logger.warning("策略状态文件中的槽位 %s 不在本次会话中, 已忽略", slot)
        self._ready = True

    def save(self) -> bool:
        """保存各策略声明的变量; 有变量不能写入 JSON 时记录错误并返回 False."""
        with self._lock:
            if not self._ready:
                logger.warning(
                    "策略状态尚未从 %s 恢复(启动未完成), 不保存以免覆盖原文件",
                    self.path,
                )
                return False
            snapshot = self._snapshot()
            reason = self._unserializable(snapshot)
            if reason is not None:
                logger.error("策略状态未保存, 保留上一次的文件: %s", reason)
                return False
            payload = {
                "saved_at": dt.datetime.now().astimezone().isoformat(),
                "slots": snapshot,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            os.replace(tmp, self.path)
            return True
