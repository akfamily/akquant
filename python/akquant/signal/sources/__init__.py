"""信号来源实现集合.

``QueueSignalSource`` 无额外依赖, 直接导出。HTTP / Redis 等需可选依赖的实现放到
P3, 届时按需惰性导入, 不让 ``import akquant`` 依赖 uvicorn/redis。
"""

from .queue import QueueSignalSource

__all__ = ["QueueSignalSource"]
