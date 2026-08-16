# 外部信号平台接入

把外部量化信号平台推来的交易指令变成委托。信号**已经是指令**，不需要策略再决策一次，因此这条路径**不经过策略回调**。

适用场景：信号在平台侧生成（因子、择时、组合优化都在那边），AKQuant 只负责接收、鉴权、风控、执行、审计。

设计背景与取舍见 [外部信号接入 RFC](../meta/signal-ingestion-rfc.md)。

## 最小可运行例子

```python
from akquant import run_live
from akquant.signal import QueueSignalSource, Signal

source = QueueSignalSource()

# 任意线程都可投递
source.put(Signal(
    signal_id="platform-0001",   # 幂等键，平台侧唯一
    symbol="000001.SZ",
    action="buy",
    quantity=100,
    price=10.5,
))

run_live(
    instruments=[...],
    broker="ctp",
    trading_mode="paper",
    signal_source=source,
)
```

`run_live` 托管信号源的生命周期：`bind` → `start`（引擎循环启动前）→ `stop`（会话收尾）。

完整可运行示例：[`examples/61_signal_platform_webhook.py`](https://github.com/akfamily/akquant/blob/main/examples/61_signal_platform_webhook.py)。

## ⚠️ 两种模式的风控覆盖面不同

这是**引擎架构决定的事实，不是配置项**，接入前必须知道：

| | `trading_mode="paper"` | `trading_mode="broker_live"` |
|---|---|---|
| 下单路径 | 引擎事件通道注入 | 柜台通道（`BrokerOrderSubmitter`） |
| `max_order_value` / `max_order_size` / `max_position_size` | ✅ | ✅ |
| `max_daily_loss` / `max_drawdown` / `strategy_risk_budget` | ✅ | ❌ **不生效** |
| 成交 | 引擎模拟撮合 | 真实柜台 |

原因：引擎的实盘执行器（`RealtimeExecutionClient`）不向柜台报单，所以 broker_live 的订单必须走 Python 侧的柜台通道，那条路上只有策略级三项限额做了前置校验。

**实践建议**：先在 `paper` 下用真实信号流验证策略行为与限额配置，再切 `broker_live`；并在平台侧或柜台侧补上日亏/回撤这类账户级熔断。

## 信号契约

`Signal` 是 pydantic 模型，字段非法立即报错（不会带着坏数据走到柜台）：

| 字段 | 必填 | 说明 |
|---|---|---|
| `signal_id` | ✅ | **幂等键**。平台重连/重推同一 id 只会下一次单 |
| `symbol` | ✅ | 标的代码 |
| `action` | ✅ | `"buy"` / `"sell"`（字符串或 `SignalAction` 枚举） |
| `quantity` | ✅ | 须为正 |
| `price` | | 省略即市价单 |
| `order_type` | | `"Limit"` / `"Market"`，省略则按 `price` 有无推断 |
| `strategy_id` | | 默认 `_default`，风控限额按它路由 |
| `timestamp` / `tag` | | 平台生成时刻、附加标记 |

## 幂等：为什么必须给 signal_id

信号平台重连、重启、网络重试都会重推。没有幂等键就意味着**每次重推都是一笔新委托**。

`SignalDedup` 按 `signal_id` 去重，两条边界值得记住：

- 投递**抛异常**（网关挂了）→ 放开去重标记，平台重推会被受理；
- 出口**同步返回未下单**（被前置风控拦下）→ 不放开，重推只会再被拒一次。

去重集合有界（默认 10 万条 LRU）。开始淘汰时会打 WARNING —— 极老的 `signal_id` 若被重推会被当成新信号，静默淘汰等于静默重复下单。长跑会话可调大 `capacity`。

## 回执：拒单必须让平台知道

平台推完指令后需要知道它有没有生效。`SignalSource.on_result` 会收到 `SignalResult`：

| `status` | 含义 |
|---|---|
| `accepted` | 已投递（**不代表已成交**） |
| `duplicate` | `signal_id` 重复，已幂等丢弃 |
| `rejected` | 被风控或柜台拒绝 |
| `error` | 处理过程抛异常 |

风控/柜台的**异步**拒单也会回吐：`run_live` 自动把 `SignalDispatcher.handle_reject` 接到策略的 `on_reject` 上（包装，不覆盖你自己的 `on_reject`），靠 `tag` 里的 `signal_id` 反查。

## 三种信号源

### `QueueSignalSource`：进程内队列

无依赖。适合"外部进程收 HTTP、只把规范化后的信号投进来"的形态，也是测试基座。

### `HttpSignalSource`：HTTP webhook

标准库实现，零额外依赖。

```python
from akquant.signal import HttpSignalSource

source = HttpSignalSource(
    token=os.environ["AKQUANT_SIGNAL_TOKEN"],   # 必填
    port=8765,
    secret=os.environ.get("AKQUANT_SIGNAL_SECRET"),  # 跨主机必开
)
```

平台侧 POST `/signal`：

```bash
curl -X POST http://127.0.0.1:8765/signal \
  -H "Authorization: Bearer $AKQUANT_SIGNAL_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"signal_id":"p-1","symbol":"000001.SZ","action":"buy","quantity":100,"price":10.5}'
```

### `RedisSignalSource`：Redis Stream

生产推荐。需要 `pip install 'akquant[signal-redis]'`。

```python
from akquant.signal import RedisSignalSource

source = RedisSignalSource(
    url="redis://127.0.0.1:6379/0",
    stream="akquant:signals",
    group="akquant",
)
```

用 Stream 而非 List：`XREADGROUP` 有消费组与显式 ack，崩溃重启后未 ack 的消息仍可重投；`BLPOP` 取走即丢，崩在处理中途就丢单。

## 🔒 安全：这是能触发真实下单的网络入口

`HttpSignalSource` 的安全约束是**硬性**的，不是建议：

| 约束 | 行为 |
|---|---|
| 强制鉴权 | `token` 必须是非空字符串，否则构造即 `ValueError`。`None` 也会被挡（否则 `str(None)` 会变成字面量 `"None"` 这个"看似有鉴权"的端点） |
| 默认只听本机 | `host` 默认 `127.0.0.1`。绑其他地址**直接报错**，须显式 `allow_remote=True` |
| 防重放 | 传 `secret` 后要求 `HMAC-SHA256(secret, "{ts}.{body}")` + 时间戳窗口（默认 ±30s），配合 `signal_id` 幂等构成三重防护 |
| 不做区分预言机 | 鉴权失败统一回 `401 {"error":"unauthorized"}`，不区分"token 错"还是"签名错"，具体原因只进日志 |

**AKQuant 不承诺传输层安全**：HTTPS 与公网暴露请在反向代理层解决。

签名的客户端实现可直接复用 `akquant.signal.sign`：

```python
import json, time
from akquant.signal import sign

body = json.dumps(payload).encode()
ts = int(time.time())
headers = {
    "Authorization": f"Bearer {token}",
    "X-Signal-Timestamp": str(ts),
    "X-Signal-Signature": sign(secret, body, ts),
}
```

## 部署形态

对标 vn.py 的取舍（WebTrader 是独立进程 + RPC），**生产推荐进程分离**：

```
平台 ──HTTPS──> 反向代理 ──> 收单进程(FastAPI 等) ──Redis Stream──> 交易进程
                                                                  (RedisSignalSource)
```

好处是 HTTP 服务的故障与负载都被进程边界隔开，不会波及交易主循环。

- **生产**：外部进程收 HTTP，只把规范化后的信号投进 Redis；
- **开发/单机**：`HttpSignalSource` 同进程直收，省一个组件；
- **测试**：`QueueSignalSource`，无网络依赖。

## 自定义信号源

实现 `SignalSource` 协议即可，继承 `SignalSourceBase` 可省掉 `bind`/`on_result` 的样板：

```python
from akquant.signal import SignalSourceBase, Signal

class MySignalSource(SignalSourceBase):
    def start(self) -> None:
        running = threading.Event()

        def worker():
            running.set()
            for payload in my_platform_stream():
                self.dispatch(Signal(**payload))

        threading.Thread(target=worker, daemon=True).start()
        running.wait(timeout=5.0)   # 关键，见下

    def stop(self) -> None:
        ...
```

⚠️ **`start()` 必须确认线程已就绪才返回。** `run_live` 在引擎循环启动前同步调用它，一返回主线程就进入 Rust 主循环并长期持有 GIL；若线程此刻尚未被调度，它可能整场会话都拿不到执行机会。这是实测结论，不是理论担忧。

另一个约束：**注入的委托只在后续市场事件到来时成交**。落在最后一根 bar 之后的信号会停在 `New` 状态 —— 这是撮合语义的必然（没有价格无法成交），不是缺陷。

## 相关文档

- [外部信号接入 RFC](../meta/signal-ingestion-rfc.md) —— 设计取舍、被否决的方案及理由
- [自定义 Broker 注册](custom_broker_registry.md) —— 接入自有柜台
- [实盘生产检查清单](custom_broker_production_checklist.md)
- [多策略指南](multi_strategy_guide.md) —— `strategy_id` 路由与分账风控
