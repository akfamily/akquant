# 策略风格决策指南

本指南用于在团队内统一选择“类风格策略”与“函数式策略”，降低后期迁移与维护成本。

## 1. 结论先行

- 默认优先：类风格策略（生产默认）。
- 快速验证与迁移：函数式策略（原型默认）。
- 复杂场景（多团队协作、长周期维护、重度风控）：必须类风格。

## 2. 决策矩阵

| 维度 | 类风格策略 | 函数式策略 |
| :--- | :--- | :--- |
| 开发速度 | 中 | 高 |
| 可维护性 | 高 | 中 |
| 生命周期表达 | 强（`on_start/on_stop/...`） | 中（已支持同名回调） |
| 状态组织 | 强（面向对象） | 中（依赖 `ctx` 约定） |
| 复杂风控/多模块协作 | 强 | 中 |
| 回测支持 | 完整 | 完整 |
| run_live 支持 | 完整 | 完整（callable + 生命周期） |
| 多 slot + 风控 | 完整 | 已支持（建议先小规模验证） |

## 3. 选型规则

- 满足任一条件，选类风格：
  - 预计维护周期超过 3 个月。
  - 需要多人并行开发同一策略。
  - 需要 warm-start、复杂调度、分层风控。
  - 需要稳定的测试与回归矩阵。
- 满足全部条件，可选函数式：
  - 目标是快速验证 alpha 假设。
  - 逻辑规模小，回调数量少。
  - 预计短期内可能重写为类风格。

## 4. 推荐落地流程

## 4.1 阶段 A：函数式快速试错

- 用函数式完成信号验证（建议 1~2 周）。
- 必备回调：`initialize/on_start/on_bar/on_stop`。
- 通过后锁定参数与风控阈值。

## 4.2 阶段 B：迁移到类风格

- 把 `ctx` 状态迁移为类属性。
- 把公共逻辑拆成私有方法。
- 保留同名生命周期回调，减少行为漂移。

## 4.3 阶段 C：生产化验收

- 增加多 slot + 风控回归用例。
- 增加 paper 与 broker_live 双路径演练。
- 固化告警、日志、事件落盘规范。

## 5. 最小验收清单

- 生命周期：`on_start/on_stop` 各触发一次。
- 归属字段：`owner_strategy_id` 可追踪。
- 风控触发：拒单原因可解释、可回放。
- 回测与 Live 行为：关键路径一致。

## 6. 参考示例

- 函数式回调基线：[23_functional_callbacks_demo.py](https://github.com/akfamily/akquant/blob/main/examples/23_functional_callbacks_demo.py)
- Live 函数式入口：[38_live_functional_strategy_demo.py](https://github.com/akfamily/akquant/blob/main/examples/38_live_functional_strategy_demo.py)
- broker_live 函数式下单：[39_live_broker_submit_order_demo.py](https://github.com/akfamily/akquant/blob/main/examples/39_live_broker_submit_order_demo.py)
- 函数式多 slot + 风控：[40_functional_multi_slot_risk_demo.py](https://github.com/akfamily/akquant/blob/main/examples/40_functional_multi_slot_risk_demo.py)
