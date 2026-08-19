"""实盘侧把网关声明的数据周期注入策略只读属性 self.freq.

周期从 ``GatewayBundle.metadata['freq']`` 取, 由各 broker 的 builder 回填并统一到
回测侧口径(klinedata 的 ``period="M1"`` → ``"1min"``), 策略代码因此无需按 broker
写兼容分支。声明缺失时保持 ``None`` —— CTP 等逐笔源与 trader-only broker 都属这类,
此处不做推断也不兜默认值。
"""

from types import SimpleNamespace

from akquant.live._runner import LiveRunner
from akquant.strategy import Strategy


class _Reader(Strategy):
    """只用来读 self.freq 的空策略."""


def _runner() -> LiveRunner:
    """构造裸 LiveRunner(不走 __init__), 与 bounded_session 测试同一手法."""
    runner = LiveRunner.__new__(LiveRunner)
    runner.broker = "test"
    return runner


def _bundle(metadata: object) -> SimpleNamespace:
    """伪造只带 metadata 的 bundle."""
    return SimpleNamespace(metadata=metadata)


def test_injects_freq_from_bundle_metadata() -> None:
    """Metadata 声明了 freq 就注入到 self.freq."""
    strategy = _Reader()
    _runner()._inject_data_freq([strategy], _bundle({"freq": "1min"}))
    assert strategy.freq == "1min"


def test_injects_into_every_target() -> None:
    """多槽位下每个实例都要拿到."""
    targets: list[Strategy] = [_Reader(), _Reader(), _Reader()]
    _runner()._inject_data_freq(targets, _bundle({"freq": "5min"}))
    assert [t.freq for t in targets] == ["5min", "5min", "5min"]


def test_freq_none_when_metadata_absent() -> None:
    """没有 metadata(CTP / trader-only broker)时保持 None."""
    strategy = _Reader()
    _runner()._inject_data_freq([strategy], _bundle(None))
    assert strategy.freq is None


def test_freq_none_when_key_missing() -> None:
    """有 metadata 但没声明 freq 时保持 None, 不兜默认值."""
    strategy = _Reader()
    _runner()._inject_data_freq([strategy], _bundle({"broker": "ctp"}))
    assert strategy.freq is None


def test_blank_freq_normalized_to_none() -> None:
    """空白字符串按未声明处理, 不让 self.freq 变成 ''."""
    strategy = _Reader()
    _runner()._inject_data_freq([strategy], _bundle({"freq": "   "}))
    assert strategy.freq is None


def test_explicit_none_freq_stays_none() -> None:
    """Builder 显式回填 None(如 klinedata 周线无对应写法)时保持 None."""
    strategy = _Reader()
    _runner()._inject_data_freq([strategy], _bundle({"freq": None}))
    assert strategy.freq is None
