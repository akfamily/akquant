"""品种类型名的归一化(下单接口用, 回测与实盘共用)."""

# 规范集(文档参考): stock/option/future/fund/bond/index/fx/crypto。
# 别名映射到规范名；规范集外的值原样放行(保持对新品种可扩展)。
_ASSET_TYPE_ALIASES = {
    "opt": "option",
    "stk": "stock",
    "fut": "future",
    "futures": "future",
    "etf": "fund",
}


def normalize_asset_type(value: str | None) -> str:
    """Normalize an asset type to its canonical name; pass unknowns through.

    Canonical names: stock, option, future, fund, bond, index, fx, crypto.
    Unknown values are returned lowercased so new products need no core change.
    """
    text = str(value or "stock").strip().lower()
    if not text:
        return "stock"
    return _ASSET_TYPE_ALIASES.get(text, text)
