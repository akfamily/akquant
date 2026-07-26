# 自定义 Broker 生产接入清单

本清单用于将自定义 broker 从“可跑通”推进到“可上线”。

## 1. 认证与连接

- 明确账号环境区分（模拟/仿真/生产）并隔离凭证
- 启动前完成连接自检并输出可观测状态
- 提供心跳检测与断线重连策略
- 对网络抖动、鉴权失败、柜台限流做明确重试策略

## 2. 下单与幂等

- 统一使用 `client_order_id` 并保证幂等提交
- 重复提交时返回已存在 `broker_order_id`，不重复报单
- 撤单接口支持幂等（重复撤单不抛异常）
- 订单状态机覆盖 `Submitted/PartiallyFilled/Filled/Cancelled/Rejected`
- 若把 `client_order_id` 当幂等键：AKQuant 生成的格式为 `{broker}-{会话标记}-{会话内序号}`，会话标记每次 `run_live` 重新生成，因此跨重启与多进程并行都不会撞号；不要假设该 id 在重启后可复现
- 声明 `supports_short_sell` 要与真实能力一致：声明 `False` 时，`side=Sell` + `position_effect=open` 会被 AKQuant 在下单前本地拒绝，不会下发柜台

## 3. 回报与状态同步

- 实现 `on_order/on_trade/on_execution_report` 三类回调
- 保证回报包含 `client_order_id` 与 `broker_order_id` 关联信息
- 启动恢复阶段实现 `sync_open_orders/sync_today_trades`
- 对乱序回报、重复回报进行去重与顺序兼容

## 4. 风控与保护

- 接入前置风控（单笔数量/金额、策略总敞口、频控）
- 增加价格保护（偏离保护、涨跌停保护、最小变动价位校验）
- 增加交易时段保护（非交易时段拒单）
- 对风险拒单保留统一 `reject_reason` 字段

## 5. 对账与一致性

- 日内定时对账：订单、成交、持仓、资金
- 收盘后生成对账报告并标记差异项
- 异常差异触发告警并支持人工干预流程
- 明确策略状态与柜台状态冲突时的优先级规则

## 6. 监控与告警

- 关键指标：下单成功率、撤单成功率、回报延迟、断线次数
- 关键时延：报单 RTT、回报处理耗时、策略回调耗时
- 告警分级：P0（无法交易）、P1（状态不同步）、P2（性能劣化）
- 日志结构化并可追溯到 `client_order_id`

## 7. 测试与演练

- 单元测试覆盖正常流、异常流、重连流、重复提交流
- 回放测试覆盖历史柜台事件样本
- 压测验证高并发下回报处理与去重正确性
- 故障演练包括：网络中断、回报延迟、部分成交、柜台重启

## 8. 发布与回滚

- 灰度发布：先小资金、小标的、短交易时段
- 设定一键降级到 `paper` 或停机保护开关
- 发布窗口内安排人工值守与实时监控
- 保留快速回滚版本和配置快照

## 9. 最低上线门槛

- 关键链路测试通过率 100%
- 至少一个完整交易日无状态差异
- 核心告警项无未处理告警
- 回滚预案已演练并确认可执行

## 10. 关联文档

- [自定义 Broker 注册](./custom_broker_registry.md)
- [API 参考：gateway 注册接口](../reference/api.md)
- [示例：35_custom_broker_registry_demo.py](https://github.com/akfamily/akquant/blob/main/examples/35_custom_broker_registry_demo.py)
