use crate::account::{calculate_account_metrics, stock_margin_ratio_override};
use crate::engine::Engine;
use crate::event::Event;
use crate::pipeline::processor::{Processor, ProcessorResult};
use pyo3::prelude::*;
use std::collections::HashMap;

pub struct StatisticsProcessor;

impl Processor for StatisticsProcessor {
    fn process(
        &mut self,
        engine: &mut Engine,
        py: Python<'_>,
        _strategy: &crate::engine::strategy_handler::StrategyHandle,
    ) -> PyResult<ProcessorResult> {
        if let Some(Event::Bar(_) | Event::Tick(_)) = engine.current_event.clone()
            && let Some(timestamp) = engine.clock.timestamp()
            && engine.is_active_timestamp(timestamp)
            // 实盘收盘宽限期内的行情不记权益点: 闭市不撮合, 当日权益以收盘结算为准,
            // 日收益与没有这些行情时一致。
            && !engine.event_in_live_grace(timestamp)
            // 实盘当日已结算后才到的行情(晚到的收盘前行情、晚到的钳制 bar): 当日权益
            // 已由收盘结算定下, 不补记。
            && !engine.current_event_day_settled
        {
            // 通过校验的聚合器收盘钳制 bar(数据阶段 `accept_close_marker`): close 含收盘
            // 后宽限期内的价格, 权益点按收盘时刻的价格(结算价快照, 数据阶段已按标记价
            // 建立)计, 与当日收盘结算一致。
            let uses_close_snapshot = engine.trusted_close_marker.is_some()
                && engine
                    .day_close_prices
                    .as_ref()
                    .is_some_and(|(d, _)| Some(*d) == engine.current_trading_day);
            let (equity, margin) = if uses_close_snapshot {
                let (_, prices) = engine
                    .day_close_prices
                    .as_ref()
                    .expect("uses_close_snapshot 已确认存在收盘价快照");
                let equity = calculate_account_metrics(
                    &engine.state.portfolio,
                    prices,
                    &engine.instruments,
                    &engine.state.order_manager.trade_tracker,
                    &engine.risk_manager.config,
                )
                .equity;
                let margin = engine
                    .state
                    .portfolio
                    .calculate_used_margin(prices, &engine.instruments);
                (equity, margin)
            } else {
                // 按最新价计(绝大多数行情事件): 与回调时取的账户指标同一口径, 输入没变
                // 就复用, 不再遍历一遍持仓(见 Engine::current_account_metrics)。
                let metrics = engine.current_account_metrics();
                // 保证金曲线不套用融资账户的股票保证金比例。没配该比例时, 它与账户指标
                // 里的占用保证金是同一个算法、同一份输入, 直接取用。
                let margin = if stock_margin_ratio_override(&engine.risk_manager.config).is_none() {
                    metrics.used_margin
                } else {
                    let prices = engine.last_prices.read().expect("last_prices 读锁被污染");
                    engine
                        .state
                        .portfolio
                        .calculate_used_margin(&prices, &engine.instruments)
                };
                (metrics.equity, margin)
            };
            engine.statistics_manager.update(
                timestamp,
                equity,
                engine.state.portfolio.cash,
                margin,
            );
            let mut payload = HashMap::new();
            payload.insert("timestamp", timestamp.to_string());
            payload.insert("equity", equity.to_string());
            payload.insert("cash", engine.state.portfolio.cash.to_string());
            payload.insert("margin", margin.to_string());
            engine.emit_stream_event(py, "equity", None, "info", payload);
        }
        Ok(ProcessorResult::Next)
    }
}
