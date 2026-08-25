"""标的代码匹配归一化: 只 upper 最后一段后缀, 不碰合约代码本体."""

from akquant.gateway.symbol_match import normalize_symbol_for_match


def test_uppercases_only_the_suffix() -> None:
    """平台用小写后缀登记、柜台推大写后缀, 两者必须归一到同一个键."""
    assert normalize_symbol_for_match("000012.sz") == "000012.SZ"
    assert normalize_symbol_for_match("000012.SZ") == "000012.SZ"
    assert normalize_symbol_for_match("600008.sh") == "600008.SH"


def test_keeps_meaningful_lowercase_in_futures_code() -> None:
    """期货合约代码含有意义的小写, 整体 upper 会改坏它(柜台大小写敏感)."""
    assert normalize_symbol_for_match("ag2612") == "ag2612"
    assert normalize_symbol_for_match("ag2612.shfe") == "ag2612.SHFE"
    assert normalize_symbol_for_match("rb2601.SHFE") == "rb2601.SHFE"


def test_only_last_dot_is_treated_as_suffix() -> None:
    """多个点时只归一化最后一段, 前面原样保留."""
    assert normalize_symbol_for_match("a.b.c") == "a.b.C"


def test_blank_inputs_collapse_to_empty() -> None:
    """空值归一成空串, 由调用方按'放行'处理."""
    assert normalize_symbol_for_match(None) == ""
    assert normalize_symbol_for_match("") == ""
    assert normalize_symbol_for_match("   ") == ""


def test_strips_surrounding_whitespace() -> None:
    """两端空白不应造成匹配失败."""
    assert normalize_symbol_for_match("  000012.sz  ") == "000012.SZ"
