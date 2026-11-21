"""期货交易所期权合约代码解析."""

import re
from dataclasses import dataclass
from typing import Optional

# 标的(品种字母 + 3/4 位年月) [-系列标记] [-] C/P [-] 行权价, 覆盖:
#   大商所/广期所/中金所 m2701-C-3100, 大商所月度期权 m2703-MS-C-3300,
#   郑商所 SR701C5700, 郑商所月度期权 SR703MSC5000, 上期所/能源中心 cu2612P94000
_SYMBOL_RE = re.compile(
    r"^([A-Za-z]+\d{3,4})(?:-?[A-Za-z]+)?-?([CP])-?(\d+(?:\.\d+)?)$"
)


@dataclass(frozen=True)
class OptionSymbol:
    """从合约代码解析出的期权要素."""

    underlying: str
    option_type: str
    strike: float


def parse_option_symbol(symbol: str) -> Optional[OptionSymbol]:
    """
    解析期货交易所期权合约代码.

    代码中不含到期日: 同一标的可能有多个到期(如大商所月度期权), 到期日须取自合约
    信息(CTP 合约查询的 ExpireDate 等)。中金所股指期权解析出的 ``underlying`` 是
    ``IO2612`` 这类系列代码, 不是可交易的标的指数。ETF 期权的数字代码无法解析。

    :param symbol: 合约代码, 如 ``m2701-C-3100``、``SR701C5700``、``cu2612P94000``
    :return: 解析结果; 无法识别时返回 None
    """
    match = _SYMBOL_RE.match(symbol.strip())
    if match is None:
        return None
    underlying, flag, strike = match.groups()
    return OptionSymbol(
        underlying=underlying,
        option_type="CALL" if flag == "C" else "PUT",
        strike=float(strike),
    )
