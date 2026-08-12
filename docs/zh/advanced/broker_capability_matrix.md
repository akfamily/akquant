# Broker Capability Matrix（模板 + 首批填充）

## 字段定义

- `market_data`: 行情接入能力（Tick/Bar）
- `order_entry`: 下单能力（Market/Limit/Stop/StopLimit）
- `cancel`: 撤单能力
- `execution_report`: 回报能力（订单状态、成交回报）
- `account`: 账户查询（资金、持仓）
- `tif`: 支持的 TimeInForce 集合
- `notes`: 语义差异或限制

## 重要说明

- 本页的能力矩阵用于描述统一接口层的目标形态与当前适配状态，不等同于“对应 broker 已完成生产级实盘接入”。
- 对于内置 `MiniQMT` / `PTrade`，当前仓库中的 trader gateway 仍以占位适配和联调骨架为主；若要用于真实 A 股交易，通常仍需补齐券商专有字段、订单回报映射和账户/持仓同步逻辑。
- 文档中出现的“集合竞价/盘前”更多指框架侧的时序与成交语义钩子，不应直接理解为已支持交易所或券商柜台的“集合竞价专用委托类型”。
- 新股/新债打新、ETF 申赎等券商专有业务，不在当前内置统一下单接口的默认承诺范围内。

## 能力矩阵（v0）

| Broker | market_data | order_entry | cancel | execution_report | account | tif | notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| IB | Tick/Bar | Market/Limit/Stop/StopLimit | Yes | Yes | Yes | DAY/GTC/IOC | 首批优先 |
| Oanda | Tick/Bar | Market/Limit/Stop | Yes | Yes | Yes | GTC/GTD/FOK/IOC | FX/CFD 语义 |
| CCXT | Tick/Bar(交易所相关) | Market/Limit(主) | Yes | Yes | Yes | 交易所相关 | 需按交易所分层 |
| CTP | Tick/Bar | Market/Limit/Stop(经映射) | Yes | Yes | Yes | 交易所相关 | 已有本地适配，当前更适合作为已落地的内置实盘链路参考 |
| MiniQMT | Tick/Bar(占位) | Market/Limit(占位) | 占位 | 占位 | 占位 | DAY/GTC(目标形态) | 当前仓库以内存/联调骨架为主，不应视为已完成生产级 A 股实盘适配 |
| PTrade | Bar(占位) | Market/Limit(占位) | 占位 | 占位 | 占位 | DAY(目标形态) | 当前仓库以内存/联调骨架为主，不应视为已完成生产级 A 股实盘适配 |

## 统一错误规范

- `UNSUPPORTED_ORDER_TYPE`
- `UNSUPPORTED_TIF`
- `BROKER_DISCONNECTED`
- `BROKER_RATE_LIMITED`
- `BROKER_REJECTED`
- `PRICE_TICK_INVALID`（委托价不是最小变动价位的整数倍；见 [tick size 对齐](tick_size_alignment.md)）

## 最小闭环验收

- 行情订阅成功并触发回调。
- 下单后能收到状态更新与成交回报。
- 撤单可追踪到最终状态。
- 账户与持仓查询返回非空结构。

## A 股专有业务边界

- 当前内置统一下单接口主要覆盖通用报单/撤单语义。
- A 股“集合竞价专用价格类型/撤单时窗控制”是否可用，取决于具体券商柜台适配是否已实现，而不是仅由 `on_pre_open` 或 `submit_order(...)` 决定。
- 新股/新债打新不属于当前内置 broker 的现成能力；如有需求，建议通过自定义 broker 注册机制补齐券商专有下单字段与业务路由。

## 关联代码入口

- [factory.py](https://github.com/akfamily/akquant/blob/main/python/akquant/gateway/factory.py)
- [registry.py](https://github.com/akfamily/akquant/blob/main/python/akquant/gateway/registry.py)
- [protocols.py](https://github.com/akfamily/akquant/blob/main/python/akquant/gateway/protocols.py)
- [broker_runtime.py](https://github.com/akfamily/akquant/blob/main/python/akquant/gateway/broker_runtime.py)
- [brokers/ctp/adapter.py](https://github.com/akfamily/akquant/blob/main/python/akquant/gateway/brokers/ctp/adapter.py)
- [brokers/miniqmt/stub.py](https://github.com/akfamily/akquant/blob/main/python/akquant/gateway/brokers/miniqmt/stub.py)
- [brokers/ptrade/stub.py](https://github.com/akfamily/akquant/blob/main/python/akquant/gateway/brokers/ptrade/stub.py)
