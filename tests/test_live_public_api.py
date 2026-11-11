"""akquant.live 公开 API 与版本号回归."""


def test_streaming_indicator_sink_is_public() -> None:
    """StreamingIndicatorSink 可从 akquant.live 直接导入."""
    from akquant.live import StreamingIndicatorSink
    from akquant.live._stream_sink import StreamingIndicatorSink as Private

    assert StreamingIndicatorSink is Private


def test_engine_rule_version_is_2_3_1() -> None:
    """缺省交易日历改为内置交易所日历并按回测数据修正, 规则版本为 2.4.0."""
    import akquant

    assert akquant.__engine_rule_version__ == "2.4.0"
