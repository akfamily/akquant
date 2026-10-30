"""MCP 服务层: 纯函数, 入参出参都是可 JSON 序列化的 dict, 不依赖 mcp SDK."""


class ServiceError(Exception):
    """调用方可修正的错误(参数非法、路径越界、id 不存在等).

    server 层把它转成 MCP 工具错误(``isError=true``), 消息原样给模型看。
    """
