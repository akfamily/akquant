"""akquant.live 公开 API 与版本号回归."""


def test_streaming_indicator_sink_is_public() -> None:
    """StreamingIndicatorSink 可从 akquant.live 直接导入."""
    from akquant.live import StreamingIndicatorSink
    from akquant.live._stream_sink import StreamingIndicatorSink as Private

    assert StreamingIndicatorSink is Private


def test_engine_rule_version_is_2_1_3() -> None:
    """自定义 __init__ 收到策略参数、walk-forward 跳过全失败窗口, 规则版本为 2.1.3."""
    import akquant

    assert akquant.__engine_rule_version__ == "2.1.3"
