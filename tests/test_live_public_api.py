"""akquant.live 公开 API 与版本号回归."""


def test_streaming_indicator_sink_is_public() -> None:
    """StreamingIndicatorSink 可从 akquant.live 直接导入."""
    from akquant.live import StreamingIndicatorSink
    from akquant.live._stream_sink import StreamingIndicatorSink as Private

    assert StreamingIndicatorSink is Private


def test_engine_rule_version_is_2_3_1() -> None:
    """公司行动的浮点比例按十进制取值、目标为 0 时卖出零股, 规则版本为 2.5.2."""
    import akquant

    assert akquant.__engine_rule_version__ == "2.5.2"
