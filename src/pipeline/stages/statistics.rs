use crate::account::calculate_account_metrics;
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
        _strategy: &Bound<'_, PyAny>,
    ) -> PyResult<ProcessorResult> {
        if let Some(Event::Bar(_) | Event::Tick(_)) = engine.current_event.clone()
            && let Some(timestamp) = engine.clock.timestamp()
            && engine.is_active_timestamp(timestamp)
            // 实盘收盘宽限期内的行情不记权益点: 闭市不撮合, 当日权益以收盘结算为准,
            // 日收益与没有这些行情时一致。
            && !engine.event_in_live_grace(timestamp)
        {
            let last_prices = engine.last_prices.read().expect("last_prices 读锁被污染");
            // 聚合器收盘钳制的收盘 bar(实盘): close 含收盘后宽限期内的价格, 权益点按
            // 收盘时刻的价格(结算价快照, 数据阶段已按 bar 携带的收盘价建立)计, 与当日
            // 收盘结算一致。
            let close_prices = match engine.current_event.as_ref() {
                Some(Event::Bar(b))
                    if engine.state.feed.is_live()
                        && b.extra
                            .contains_key(crate::data::aggregator::BAR_CLOSE_PRICE_KEY) =>
                {
                    engine
                        .day_close_prices
                        .as_ref()
                        .filter(|(d, _)| Some(*d) == engine.current_trading_day)
                        .map(|(_, p)| p)
                }
                _ => None,
            };
            let prices = close_prices.unwrap_or(&last_prices);
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
            drop(last_prices);
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
