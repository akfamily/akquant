"""akquant.live 公开 API 与版本号回归."""


def test_streaming_indicator_sink_is_public() -> None:
    """StreamingIndicatorSink 可从 akquant.live 直接导入."""
    from akquant.live import StreamingIndicatorSink
    from akquant.live._stream_sink import StreamingIndicatorSink as Private

    assert StreamingIndicatorSink is Private


def test_engine_rule_version_is_2_3_1() -> None:
    """股票类日内 K 线改为从交易时段的连续竞价起点起算, 规则版本为 2.5.0."""
    import akquant

    assert akquant.__engine_rule_version__ == "2.5.0"
