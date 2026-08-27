"""实盘历史预热: Engine 灌入入口与 runner 接入."""

from akquant import Bar, Engine


def _bar(ts_ns: int, symbol: str = "600000.SH", close: float = 10.0) -> Bar:
    return Bar(ts_ns, close, close, close, close, 100.0, symbol)


def test_engine_preload_history_updates_internal_state() -> None:
    """验证灌入的历史数据确实被写进引擎状态.

    历史缓冲通过 get_state_bytes 随引擎状态一起序列化(src/engine/python.rs:622-624),
    因此可以通过检查序列化后是否含特定标的来验证数据已进缓冲,
    无需为测试新增生产接口。
    """
    engine = Engine()
    before = engine.get_state_bytes()
    base = 1_000_000_000_000_000_000
    bars = [_bar(base + i, symbol="600000.SH", close=10.0 + i) for i in range(3)]
    engine.preload_history(bars)
    after = engine.get_state_bytes()

    # 验证数据确实进了缓冲
    assert b"600000.SH" not in before
    assert b"600000.SH" in after

    # 空输入是 no-op, 不该抛
    engine.preload_history([])
