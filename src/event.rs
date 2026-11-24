use crate::model::{Bar, Order, Tick, Timer, Trade};
use chrono::NaiveDate;

#[derive(Debug, Clone)]
pub enum Event {
    Bar(Bar),
    Tick(Tick),
    Timer(Timer),
    /// 策略发出的订单请求
    OrderRequest(Order),
    /// 风控通过的订单 (准备执行)
    OrderValidated(Order),
    /// 执行报告 (订单状态更新/成交)
    /// 包含：更新后的订单快照，以及生成的成交记录(如果有)
    ExecutionReport(Order, Option<Trade>),
    /// 行情源报告的交易日（仅 live feed）：时刻 `timestamp`（UTC 纳秒）属于交易日
    /// `trading_day`。与行情同序处理，用于校验/修正引擎的交易日历。
    TradingDayHint {
        timestamp: i64,
        trading_day: NaiveDate,
        source: String,
    },
    /// 行情源推来的未收盘 K 线（仅 live feed）。不是行情：只交给策略做指标试算，
    /// 不推进时钟、不进时间片、不撮合、不更新最新价、不进历史缓冲。
    /// `timestamp` 是这一桶将来收盘 bar 的时间戳（晚于当前时刻）。
    FormingBar(Bar),
}
