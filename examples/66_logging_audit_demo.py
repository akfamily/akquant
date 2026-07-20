# -*- coding: utf-8 -*-
"""日志系统演示：订单审计、敏感脱敏与日志语言.

演示目标（全部自包含、无需真实网关，可直接运行）:
- configure_logging 打开控制台 + 独立订单审计文件(order_audit_file)
- 敏感字段脱敏(密钥全掩码 / 账户保留尾 4 位),handler 层兜底
- 订单生命周期审计: submit/fill/cancel 逐笔落 JSON,可脱机重建
- 日志语言: 默认英文(通用契约); language="zh" 只切控制台审计行,文件恒英文

说明: `broker_live` 下审计由 gateway 自动触发; 这里直接调用 order_audit.*
仅为在无网关环境下演示日志系统的能力。
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import akquant
from akquant import LogConfig, configure_logging, get_logger
from akquant.gateway import order_audit
from akquant.gateway.broker_models import UnifiedTrade


def _emit_sample_events(trace_id: str = "GROUP-1") -> None:
    """产生一组样例日志: 一条含敏感字段, 加一笔订单的 submit/fill/cancel 审计."""
    # 普通日志: 敏感字段会被自动脱敏(user_id 保留尾 4 位, password 全掩码)
    get_logger("gateway.live").info(
        "login user_id=88881234 password=secret123",
        extra=akquant.log.build_log_extra(phase="gateway"),
    )
    # 订单审计(broker_live 下自动触发, 此处手动调用仅为演示)
    order_audit.record_submit(
        strategy_id="demo",
        symbol="600000.SH",
        side="Buy",
        quantity=100,
        price=10.55,
        client_order_id="GROUP-1-a-0",
        broker_order_id="B1",
        order_type="Limit",
        trace_id=trace_id,
    )
    order_audit.record_broker_event(
        "trade",
        UnifiedTrade(
            trade_id="T1",
            broker_order_id="B1",
            client_order_id="GROUP-1-a-0",
            symbol="600000.SH",
            side="Buy",
            quantity=100.0,
            price=10.55,
            timestamp_ns=0,
        ),
        owner_strategy_id="demo",
        trace_id=trace_id,
    )
    order_audit.record_cancel(
        broker_order_id="B1", symbol="600000.SH", strategy_id="demo"
    )


def _flush() -> None:
    """刷新所有 handler, 确保文件内容落盘后可读取."""
    for name in ("akquant", "akquant.audit.order"):
        for handler in logging.getLogger(name).handlers:
            handler.flush()


def _dump_audit_file(audit_path: Path) -> None:
    """读取审计 JSON 文件并打印(证明可脱机重建订单生命周期)."""
    print("\n--- 审计文件内容(纯 JSON, 机器可对账; message 恒英文) ---")
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry: dict[str, Any] = json.loads(line)
        print(
            f"  event={entry['event']:<13} "
            f"trace_id={entry.get('trace_id', '-')} "
            f"msg={entry['message']}"
        )


def main() -> None:
    """运行日志系统演示."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="akquant_log_demo_"))
    audit_path = tmp_dir / "orders_audit.log"

    # 1) 默认英文控制台 + 独立审计文件 + 脱敏(默认开启)
    print("=" * 72)
    print("[1] language='en'(默认): 控制台英文, 审计另落 JSON 文件, 敏感字段脱敏")
    print("=" * 72)
    configure_logging(
        LogConfig(
            profile="live",
            level="INFO",
            console=True,
            order_audit_file=str(audit_path),
            order_audit_level="INFO",
            language="en",
        )
    )
    _emit_sample_events()
    _flush()

    # 2) 切中文控制台: 只影响控制台审计行, 文件仍英文
    print("\n" + "=" * 72)
    print("[2] language='zh': 控制台审计行渲染中文(文件/JSON 仍英文)")
    print("=" * 72)
    configure_logging(
        LogConfig(
            profile="live",
            level="INFO",
            console=True,
            order_audit_file=str(audit_path),
            order_audit_level="INFO",
            language="zh",
        )
    )
    _emit_sample_events(trace_id="GROUP-2")
    _flush()

    # 3) 证明审计文件可脱机重建(且恒英文, 便于跨团队/工具消费)
    _dump_audit_file(audit_path)

    # 复位为库默认静默, 不影响后续导入方
    configure_logging(LogConfig(console=False, filename=None, reset_handlers=True))
    print(f"\n审计文件路径: {audit_path}")
    print("完成。")


if __name__ == "__main__":
    main()
