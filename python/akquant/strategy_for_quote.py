"""策略侧询价(ForQuote): 向做市商请求某合约的双边报价.

实盘(broker_live)经柜台发出询价请求; 回测与 paper 没有做市商, 询价不产生报价。
"""

from typing import Any, Optional


def request_quote(strategy: Any, symbol: str) -> Optional[str]:
    """校验合约已登记后交给执行后端, 返回柜台询价引用(回测与 paper 为 None)."""
    symbol = str(symbol).strip()
    if not symbol:
        raise ValueError("询价合约代码不能为空")
    strategy.get_instrument(symbol)
    execution = strategy.execution
    request = getattr(execution, "request_quote", None)
    if not callable(request):
        raise RuntimeError(
            f"执行后端 {type(execution).__name__} 没有实现 "
            "request_quote(symbol), 不能询价"
        )
    ref = request(symbol)
    return None if ref is None else str(ref)
