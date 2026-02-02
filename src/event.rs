use crate::model::{Bar, Tick, Order, Trade};

#[derive(Debug, Clone)]
pub enum Event {
    Bar(Bar),
    Tick(Tick),
    /// 策略发出的订单请求
    OrderRequest(Order),
    /// 风控通过的订单 (准备执行)
    OrderValidated(Order),
    /// 执行报告 (订单状态更新/成交)
    /// 包含：更新后的订单快照，以及生成的成交记录(如果有)
    ExecutionReport(Order, Option<Trade>),
}
