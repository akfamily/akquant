"""实盘历史预热: Engine 灌入入口与 runner 接入."""

from akquant import Bar, Engine


def _bar(ts_ns: int, symbol: str = "600000.SH", close: float = 10.0) -> Bar:
    return Bar(ts_ns, close, close, close, close, 100.0, symbol)


def test_engine_preload_history_accepts_bars_without_error() -> None:
    """灌入本身可调用且不抛(内容断言在 Task 3 的端到端里, 见下方说明).

    Engine 未暴露独立的 context 入口, 历史内容只能经策略上下文读取, 因此这一层
    只能验证"调得通、不炸"; 真正证明数据进了缓冲的是
    test_preload_makes_history_available_in_first_on_bar。
    """
    engine = Engine()
    base = 1_000_000_000_000_000_000
    engine.preload_history([_bar(base + i, close=10.0 + i) for i in range(3)])
    engine.preload_history([])  # 空输入是 no-op, 不该抛
