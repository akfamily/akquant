"""akquant.live 公开 API 与版本号回归."""


def test_streaming_indicator_sink_is_public() -> None:
    """StreamingIndicatorSink 可从 akquant.live 直接导入."""
    from akquant.live import StreamingIndicatorSink
    from akquant.live._stream_sink import StreamingIndicatorSink as Private

    assert StreamingIndicatorSink is Private


def test_engine_rule_version_is_current() -> None:
    """规则版本为 2.6.1(自定义撮合器了结订单, 当根成交模式下 bracket 子单)."""
    import akquant

    assert akquant.__engine_rule_version__ == "2.6.1"
