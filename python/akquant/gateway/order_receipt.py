"""下单回执的旧导入路径; 定义已挪到 :mod:`akquant.execution.receipt`.

回执是策略下单接口的返回值, 回测与实盘共用, 不属于网关。这里保留转发, 既有的
``from akquant.gateway.order_receipt import ...`` 照常可用。
"""

from ..execution.receipt import OrderLeg, OrderReceipt

__all__ = ["OrderLeg", "OrderReceipt"]
