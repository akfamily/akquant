"""信号来源实现集合.

三者都无强制额外依赖:

- :class:`QueueSignalSource` —— 进程内队列, 参考实现与测试基座;
- :class:`HttpSignalSource` —— 标准库 ``http.server`` 实现的 webhook, 零依赖
  (刻意不用 FastAPI: 这个端点能触发真实下单, 必须在 CI 里被真实测到);
- :class:`RedisSignalSource` —— Redis Stream 消费, ``redis`` 包惰性导入,
  未安装时只在实际使用时报错, 不影响 ``import akquant``。
"""

from .http import HttpSignalSource
from .queue import QueueSignalSource
from .redis_stream import RedisSignalSource

__all__ = ["QueueSignalSource", "HttpSignalSource", "RedisSignalSource"]
