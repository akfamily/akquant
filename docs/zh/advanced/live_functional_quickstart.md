# Live 函数式策略上手手册

本文聚焦 run_live 的函数式策略入口，帮助你在 `paper` 与 `broker_live` 两种模式下快速搭建最小闭环。

## 1. 适用场景

- 想保留 `on_bar(ctx, bar)` 风格，不继承 `Strategy` 类。
- 需要快速把回测中的函数式策略迁移到实时/仿真运行。
- 需要在 `broker_live` 下直接使用 `submit_order(...)`。

## 2. 两种运行模式

### 2.1 paper（撮合模拟）

推荐先用 paper 检查事件链路是否正常：

- 示例脚本：[38_live_functional_strategy_demo.py](https://github.com/akfamily/akquant/blob/main/examples/38_live_functional_strategy_demo.py)
- 典型参数：
  - `trading_mode="paper"`
  - `strategy_cls=on_bar`
  - `initialize/on_order/on_trade/context`

### 2.2 broker_live（网关真实下单）

确认网关连通后切换到 broker_live：

- 示例脚本：[39_live_broker_submit_order_demo.py](https://github.com/akfamily/akquant/blob/main/examples/39_live_broker_submit_order_demo.py)
- 审计示例：[42_live_broker_event_audit_demo.py](https://github.com/akfamily/akquant/blob/main/examples/42_live_broker_event_audit_demo.py)
- 关键点：
  - `trading_mode="broker_live"`
  - `on_bar` 中调用 `ctx.submit_order(...)`
  - 显式传入 `client_order_id` 便于幂等追踪
  - 默认执行语义 `execution_semantics_mode="strict"`（终态仅由柜台订单回报推进）
  - 可选 `on_broker_event` 统一落盘 `event_type/owner_strategy_id/payload`

`execution_semantics_mode` 可通过 `gateway_options` 传入：

- `strict`（默认，推荐生产）：`Cancelled/Rejected/Filled` 等终态仅由 `OnRtnOrder` 推进；错误回报会缓存拒单原因并在后续订单回报补齐。
- `compatible`（兼容模式）：允许在部分错误/撤单场景下立即本地推进终态，便于旧策略平滑迁移。

## 3. 函数式入口模板

```python
def initialize(ctx):
    ctx.sent = False

def on_bar(ctx, bar):
    if not ctx.sent and getattr(ctx, "broker_ready", False):
        ctx.submit_order(
            symbol=bar.symbol,
            side="Buy",
            quantity=1.0,
            client_order_id="demo-1",
            order_type="Market",
        )
        ctx.sent = True

from akquant import run_live

run_live(
    strategy_cls=on_bar,
    initialize=initialize,
    on_order=on_order,
    on_trade=on_trade,
    on_timer=on_timer,
    context={"strategy_name": "demo"},
    instruments=instruments,
    broker="ctp",
    trading_mode="broker_live",
    gateway_options={"execution_semantics_mode": "strict"},
    duration="30s",
    show_progress=False,
)
```

## 4. 常见排查

- `submit_order` 尚未就绪
  - 原因：交易网关尚未完成连接/登录。
  - 处理：在 `on_bar` 中用 `if getattr(ctx, "broker_ready", False):` 门首单
    （broker 就绪由 run_live 的 heartbeat 轮询裁定；就绪前调用会抛清晰错误）。
- `duplicate active client_order_id`
  - 原因：重复提交活跃 client id。
  - 处理：每次下单生成新的 `client_order_id`。
- 有行情但无成交回调
  - 原因：交易网关未连通、风控拒单、最小变动价位/手数不合规。
  - 处理：优先检查 `on_order` 状态与拒单原因。
- 撤单请求发出后状态仍是 `Submitted`
  - 原因：当前为严格语义，状态需等待 `OnRtnOrder(Cancelled)`。
  - 处理：检查交易侧回报链路与订单回报日志，不要用本地请求发送成功替代终态。

建议在 live/paper 排查前先显式打开日志：

```python
import akquant

akquant.configure_logging(
    akquant.LogConfig(
        profile="live",
        level="INFO",
        console=True,
        file_json=True,
        filename="logs/live_runner.log",
    )
)
```

这样 `on_order` / `on_trade` 的策略日志与网关/执行层 warning 会进入同一套输出链路；遇到拒单、未知撤单、严格语义下状态未终态推进等问题时，更容易按 `symbol`、`order_id`、`client_order_id`、`strategy_id` 做排查。

## 5. 建议上线流程

- 第一步：先跑 paper 模式，确认回调顺序与策略状态变更。
- 第二步：切换 broker_live，先用最小下单量做连通性验证。
- 第三步：稳定后再增加复杂逻辑（定时器、风控、分批下单）。

## 离线验证：`broker="replay"`

`replay` 是内置的确定性回放行情源，用于在没有柜台的环境下验证策略的实盘数据
通路是否通畅——策略能否收到 bar/tick、多品种是否都到齐、`current_tick` 是否
正确。

```python
from akquant import AssetType, Instrument, run_live
from akquant.akquant import Bar

run_live(
    strategy_cls=MyStrategy,
    instruments=[Instrument(symbol="DEMO_A", asset_type=AssetType.Stock, ...)],
    broker="replay",
    trading_mode="paper",
    gateway_options={"bars": bars},   # list[Bar] / list[Tick] / DataFrame
)
```

事件按时间戳升序推送，多品种全局交错。数据放完后会话自行结束。

**适用边界**：

- 只提供行情，**不模拟成交**（`trader_gateway=None`），因此不能用于
  `trading_mode="broker_live"`——那会抛 `ValueError`。撮合由 paper 模式的
  模拟执行后端负责。
- **不覆盖 timer 语义**。回放数据带历史时间戳，而实盘引擎按墙钟判定 timer 是否
  到期，两条时间线错位。`on_timer` / `schedule_daily` 在回放会话中的行为不作
  保证；要验证定时任务请用回测。
- 多品种 DataFrame 需提供 `股票代码` 列，否则会退化为单标的。多品种场景建议直接
  用 `list[Bar]`。
