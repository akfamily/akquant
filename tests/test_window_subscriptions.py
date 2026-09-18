"""引擎原生多周期: subscribe_bars / on_window_bar / get_history(freq=) / 指标按周期."""

from akquant.akquant import Bar


def test_bar_freq_defaults_to_none_and_is_settable() -> None:
    """Bar.freq 默认 None, 可读写, 构造函数支持 freq 关键字参数."""
    bar = Bar(1_700_000_000_000_000_000, 1.0, 2.0, 0.5, 1.5, 100.0, "X")
    assert bar.freq is None
    bar.freq = "5min"
    assert bar.freq == "5min"
    tagged = Bar(1, 1.0, 1.0, 1.0, 1.0, 1.0, "X", freq="1d")
    assert tagged.freq == "1d"
    assert "freq=1d" in repr(tagged)
