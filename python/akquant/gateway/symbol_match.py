"""标的代码匹配工具: 后缀大小写归一化.

实盘的委托/成交回报要按"是否属于本会话挂载标的"过滤, 而登记侧与推送侧的后缀
大小写并不保证一致(平台可能用 ``000012.sz`` 登记, 柜台推 ``000012.SZ``)。
精确字符串比较会把**本会话自己的**回报判成别人的并静默丢弃, 表现为"下单成功
却收不到回调"——这个坑 2026-08-17 已经踩过一次。

只归一化最后一个 ``.`` 之后的后缀, 与 Rust ``Instrument::new`` 里的
``normalize_symbol_suffix`` 同规则: 期货合约代码含有意义的小写
(``ag2612`` / ``rb2601``)且柜台大小写敏感, 整体 ``upper()`` 会改坏它。
"""


def normalize_symbol_for_match(symbol: object) -> str:
    """归一化标的代码用于集合比较(只 upper 最后一段后缀).

    :param symbol: 任意标的代码; 非字符串按 ``str()`` 处理。
    :return: 归一化后的代码; 空值或纯空白返回空串。
    """
    text = str(symbol or "").strip()
    if not text:
        return ""
    head, separator, suffix = text.rpartition(".")
    if not separator:
        return text
    return f"{head}.{suffix.upper()}"
