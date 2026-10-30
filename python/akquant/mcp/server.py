"""MCP 薄层: 把 service 函数注册成工具和 resource.

只做参数转换与错误包装(ServiceError -> ToolError); 业务逻辑都在
``service`` 子包, 那里不依赖 mcp SDK。工具返回注解必须写成
``dict[str, Any]``(内置泛型): 裸 ``dict`` 不生成 structured_content,
``typing.Dict[str, Any]`` 会被包成 ``{"result": ...}``。
"""

import sys
import traceback
from typing import Any, Callable, Dict, Literal, Optional

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError

from .service import ServiceError, api_index, check, data, guides, runs
from .service.paths import ServiceConfig

INSTRUCTIONS = (
    "AKQuant 策略开发助手。推荐流程: get_guide() 看要点 -> search_api / "
    "get_api 查接口 -> 写策略文件 -> check_strategy -> prepare_data -> "
    "run_backtest -> 按 error.user_frames 与 hint 修改后重跑; 明细用 "
    "get_run_details 分页读取。只支持类式 Strategy 子类; 路径须在项目根"
    "目录内。回测在子进程中执行, 不做沙箱。"
)


def _call(fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    try:
        return fn()
    except ServiceError as exc:
        raise ToolError(str(exc)) from exc
    except Exception as exc:
        # SDK 对非 ToolError 只回 "Error executing tool X", 模型看不到原因
        traceback.print_exc(file=sys.stderr)
        raise ToolError(f"内部错误 {type(exc).__name__}: {exc}") from exc


def build_server(cfg: ServiceConfig) -> MCPServer:
    """构造注册好 7 个工具与 guide resource 的 MCPServer."""
    srv = MCPServer("akquant", instructions=INSTRUCTIONS)

    @srv.tool()
    def search_api(query: str, limit: int = 20) -> dict[str, Any]:
        """按关键词搜索 AKQuant 公开 API.

        范围: akquant 顶层导出、Strategy 的方法与属性、run_backtest 的参数
        (名称形如 run_backtest.initial_cash)。名称命中排在 docstring 命中
        之前。返回名称、类型与一句摘要; 详情用 get_api。
        """
        return _call(lambda: api_index.search_api(query, limit))

    @srv.tool()
    def get_api(name: str) -> dict[str, Any]:
        """返回一个 API 的完整签名与 docstring.

        name 例: run_backtest、Strategy.buy、Strategy.I、
        run_backtest.initial_cash。找不到时错误信息里给出相近名称。
        """
        return _call(lambda: api_index.get_api(name))

    @srv.tool()
    def get_guide(topic: Optional[str] = None) -> dict[str, Any]:
        """读取写策略要点.

        不传 topic 返回主题列表; 传 topic(如 strategy-skeleton、indicators)
        返回该篇 Markdown 全文。写新策略前先读 strategy-skeleton。
        """
        return _call(lambda: guides.get_guide(topic))

    @srv.tool()
    def check_strategy(
        path: str, strategy_attr: Optional[str] = None
    ) -> dict[str, Any]:
        """静态检查策略文件(不执行代码).

        path 相对项目根目录。返回 ok、类信息与 diagnostics(语法错误、
        找不到/多个 Strategy 子类、拼错的 self 方法、Strategy 方法的错误
        关键字参数、on_bar 里的 self.I、重写 __init__ 等)。文件有多个
        Strategy 子类时用 strategy_attr 指定类名。
        """
        return _call(lambda: check.check_strategy(cfg, path, strategy_attr))

    @srv.tool()
    def prepare_data(spec: Dict[str, Any]) -> dict[str, Any]:
        """准备回测数据并缓存, 返回 dataset_id 与摘要(含 issues).

        spec.source 三选一:
        file: {"source": "file", "path": "data/x.csv", "symbol"?, "timezone"?};
        akshare: {"source": "akshare", "symbol": "600000", "start": "20230101",
        "end": "20241231", "adjust"?: "qfq"}(仅 A 股个股日线);
        synthetic: {"source": "synthetic", "symbols"?, "start"?, "end"?,
        "freq"?: "1d"|"1m", "seed"?}(只用于验证能跑通)。
        """
        return _call(lambda: data.prepare_data(cfg, spec))

    @srv.tool()
    def run_backtest(
        path: str,
        dataset_id: str,
        config: Optional[Dict[str, Any]] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
        strategy_attr: Optional[str] = None,
        timeout_s: int = 120,
    ) -> dict[str, Any]:
        """在子进程中回测策略文件, 返回 status(ok/error/timeout)与摘要.

        config 只接受: initial_cash, commission_rate, stamp_tax_rate,
        min_commission, slippage, t_plus_one, lot_size, timezone, symbols,
        start_time, end_time, broker_profile, warmup_period。
        slippage 必须是字典, 如 {"type": "percent", "value": 0.0002};
        {"type": "zero"} 表示关闭滑点(含覆盖 broker_profile 模板的滑点)。
        strategy_params 只能传类体里 IntParam 等声明过的参数。失败时看
        error.phase、error.user_frames(策略文件里的行号)与 error.hint。
        每次调用都重新加载文件, 改完直接重跑即可。
        """
        return _call(
            lambda: runs.run_backtest(
                cfg,
                path,
                dataset_id,
                config=config,
                strategy_params=strategy_params,
                strategy_attr=strategy_attr,
                timeout_s=timeout_s,
            )
        )

    @srv.tool()
    def get_run_details(
        run_id: str,
        section: Literal["trades", "orders", "equity", "logs"],
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """分页读取一次回测的成交、委托(含拒单原因)、日频权益或日志."""
        return _call(lambda: runs.get_run_details(cfg, run_id, section, offset, limit))

    @srv.resource("akquant://guide/{topic}", mime_type="text/markdown")
    def guide_resource(topic: str) -> str:
        """AKQuant 写策略要点(Markdown)."""
        try:
            return str(guides.get_guide(topic)["content"])
        except ServiceError as exc:
            raise ResourceError(str(exc)) from exc

    return srv
