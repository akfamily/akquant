"""Instrument 构造期把交易所后缀归一化为大写.

平台侧存在 ``600028.sh`` 这类小写后缀写法(8/11 与 8/17 两批测试反馈)。
``instruments`` 是所有下游的唯一源头, 小写会从这一处流向三个地方各自炸开:

1. ``get_instrument("600028.SH")`` -> KeyError(快照字典 key 是小写);
2. 实盘订阅集(``live/_runner.py`` 的 ``[inst.symbol for ...]``)变成小写, 而
   broker 推送反解恒为大写 -> 本任务自己的 order/trade 回报被静默丢弃;
3. Rust 撮合层按小写登记合约, 大写行情下单时 ``Instrument not found``
   -> 该标的**所有**订单被拒。

故在构造期一次性归一化(与既有的 ``symbol.trim()`` 同属输入清洗)。
"""

import logging

from akquant import AssetType, Instrument


def test_lowercase_exchange_suffix_is_normalized() -> None:
    """``600028.sh`` 构造后 symbol 必须是 ``600028.SH``."""
    inst = Instrument("600028.sh", AssetType.Stock, tick_size=0.01, lot_size=100.0)
    assert inst.symbol == "600028.SH"


def test_mixed_case_suffix_is_normalized() -> None:
    """大小写混写的后缀同样归一化(``.Sz`` -> ``.SZ``)."""
    inst = Instrument("000012.Sz", AssetType.Stock)
    assert inst.symbol == "000012.SZ"


def test_futures_code_without_suffix_keeps_its_case() -> None:
    """无后缀的期货合约代码**必须**原样保留大小写.

    上期所/大商所合约代码本身是小写(``ag2612``/``rb2601``)且柜台大小写敏感,
    对整个 symbol 做 upper() 会把它改坏 —— 这里锁住"只动后缀"这条边界。
    """
    inst = Instrument("ag2612", AssetType.Futures, multiplier=15.0, margin_ratio=0.12)
    assert inst.symbol == "ag2612"


def test_code_segment_case_is_preserved_when_suffix_normalized() -> None:
    """带后缀时也只动后缀, 代码段原样保留."""
    inst = Instrument(
        "ag2612.shfe", AssetType.Futures, multiplier=15.0, margin_ratio=0.12
    )
    assert inst.symbol == "ag2612.SHFE"


def test_normalization_emits_warning(caplog: object) -> None:
    """改写过 symbol 必须留下 WARNING, 否则上游永远不知道自己写法不一致."""
    with caplog.at_level(logging.WARNING):  # type: ignore[attr-defined]
        Instrument("600028.sh", AssetType.Stock)
    text = caplog.text  # type: ignore[attr-defined]
    assert "600028.sh" in text and "600028.SH" in text


def test_already_uppercase_suffix_emits_no_warning(caplog: object) -> None:
    """已经是大写后缀时不打告警(避免正常用法刷屏)."""
    with caplog.at_level(logging.WARNING):  # type: ignore[attr-defined]
        Instrument("600028.SH", AssetType.Stock)
    assert "600028" not in caplog.text  # type: ignore[attr-defined]
