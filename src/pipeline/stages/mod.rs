mod channel;
mod cleanup;
mod data;
mod execution;
mod shared;
mod statistics;
pub(crate) mod strategy;

pub use channel::ChannelProcessor;
pub use cleanup::CleanupProcessor;
pub use data::DataProcessor;
pub use execution::ExecutionProcessor;
pub use shared::ExecutionPhase;
pub use statistics::StatisticsProcessor;
pub use strategy::StrategyProcessor;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use crate::model::{
        AssetType, Bar, Instrument, InstrumentEnum, Order, OrderSide, OrderStatus, OrderType,
        StockInstrument, TimeInForce,
    };
    use crate::pipeline::processor::Processor;
    use pyo3::prelude::*;
    use rust_decimal_macros::dec;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn create_instrument(symbol: &str) -> Instrument {
        Instrument {
            asset_type: AssetType::Stock,
            inner: InstrumentEnum::Stock(StockInstrument {
                symbol: symbol.to_string(),
                lot_size: dec!(100),
                tick_size: dec!(0.01),
                expiry_date: None,
                sellable_after_days: 1,
            }),
        }
    }

    #[test]
    fn test_data_alignment_late_fill() {
        pyo3::Python::initialize();

        let mut engine = Engine::new();
        engine
            .instruments
            .insert("A".to_string(), create_instrument("A"));
        engine
            .instruments
            .insert("B".to_string(), create_instrument("B"));

        engine
            .last_prices
            .write()
            .unwrap()
            .insert("A".to_string(), dec!(100));
        engine
            .last_prices
            .write()
            .unwrap()
            .insert("B".to_string(), dec!(200));

        let mut processor = DataProcessor::new();

        let bar_t1_a = Bar {
            timestamp: 1000,
            symbol: "A".to_string(),
            open: dec!(100),
            high: dec!(100),
            low: dec!(100),
            close: dec!(100),
            volume: dec!(100),
            extra: HashMap::new(),
            freq: None,
            interval_start: None,
        };

        let bar_t2_b = Bar {
            timestamp: 2000,
            symbol: "B".to_string(),
            open: dec!(205),
            high: dec!(205),
            low: dec!(205),
            close: dec!(205),
            volume: dec!(200),
            extra: HashMap::new(),
            freq: None,
            interval_start: None,
        };

        let bar_t3_a = Bar {
            timestamp: 3000,
            symbol: "A".to_string(),
            open: dec!(102),
            high: dec!(102),
            low: dec!(102),
            close: dec!(102),
            volume: dec!(100),
            extra: HashMap::new(),
            freq: None,
            interval_start: None,
        };

        engine.state.feed.add_bar(bar_t1_a).unwrap();
        engine.state.feed.add_bar(bar_t2_b).unwrap();
        engine.state.feed.add_bar(bar_t3_a).unwrap();

        // Use Python::attach as recommended by PyO3 0.28+
        pyo3::Python::attach(|py| {
            let locals = py.import("builtins").unwrap();
            let strategy = locals.getattr("None").unwrap();

            // Step 1: Process T1 A
            processor.process(&mut engine, py, &strategy).unwrap();

            // Step 2: Process T2 B (Fill T1)
            processor.process(&mut engine, py, &strategy).unwrap();

            // Verify T1 Fill
            {
                let buffer = engine.history_buffer.read().unwrap();
                let hist_b = buffer.data.get("B").unwrap();
                assert_eq!(hist_b.timestamps[0], 1000);
                assert_eq!(hist_b.closes[0], 200.0);
                assert_eq!(hist_b.volumes[0], 0.0);
            }

            engine
                .last_prices
                .write()
                .unwrap()
                .insert("B".to_string(), dec!(205));

            // Step 3: Process T3 A (Fill T2)
            processor.process(&mut engine, py, &strategy).unwrap();

            // Verify T2 Fill
            {
                let buffer = engine.history_buffer.read().unwrap();
                let hist_a = buffer.data.get("A").unwrap();
                assert_eq!(hist_a.timestamps[1], 2000);
                assert_eq!(hist_a.closes[1], 100.0);
                assert_eq!(hist_a.volumes[1], 0.0);
            }
        });
    }

    #[test]
    fn test_missing_symbol_next_open_order_is_rejected_and_removed_from_active_orders() {
        pyo3::Python::initialize();

        let mut engine = Engine::new();
        engine
            .instruments
            .insert("A".to_string(), create_instrument("A"));
        engine
            .instruments
            .insert("B".to_string(), create_instrument("B"));
        engine.state.portfolio.cash = dec!(50000);
        engine
            .last_prices
            .write()
            .unwrap()
            .insert("A".to_string(), dec!(100));
        engine
            .last_prices
            .write()
            .unwrap()
            .insert("B".to_string(), dec!(100));

        let mut pending_order =
            Order::test_new("order-b", "B", OrderSide::Buy, OrderType::Limit, dec!(300));
        pending_order.price = Some(dec!(100));
        pending_order.time_in_force = TimeInForce::GTC;
        pending_order.created_at = 1000;
        pending_order.updated_at = 1000;

        engine.execution_model.on_order(pending_order.clone());
        engine.state.order_manager.add_active_order(pending_order);

        let mut data_processor = DataProcessor::new();
        let mut channel_processor = ChannelProcessor;
        let mut cleanup_processor = CleanupProcessor;

        let bar_t1_a = Bar {
            timestamp: 1000,
            symbol: "A".to_string(),
            open: dec!(100),
            high: dec!(100),
            low: dec!(100),
            close: dec!(100),
            volume: dec!(100),
            extra: HashMap::new(),
            freq: None,
            interval_start: None,
        };
        let bar_t2_a = Bar {
            timestamp: 2000,
            symbol: "A".to_string(),
            open: dec!(101),
            high: dec!(101),
            low: dec!(101),
            close: dec!(101),
            volume: dec!(100),
            extra: HashMap::new(),
            freq: None,
            interval_start: None,
        };

        engine.state.feed.add_bar(bar_t1_a).unwrap();
        engine.state.feed.add_bar(bar_t2_a).unwrap();

        pyo3::Python::attach(|py| {
            let locals = py.import("builtins").unwrap();
            let strategy = locals.getattr("None").unwrap();

            data_processor.process(&mut engine, py, &strategy).unwrap();
            channel_processor
                .process(&mut engine, py, &strategy)
                .unwrap();
            data_processor.process(&mut engine, py, &strategy).unwrap();
            channel_processor
                .process(&mut engine, py, &strategy)
                .unwrap();
            cleanup_processor
                .process(&mut engine, py, &strategy)
                .unwrap();

            let rejected_order = engine
                .state
                .order_manager
                .get_all_orders()
                .into_iter()
                .find(|order| order.id == "order-b")
                .unwrap();
            assert_eq!(rejected_order.status, OrderStatus::Rejected);
            assert!(rejected_order.reject_reason.contains("Missing market data"));
            assert_eq!(
                engine
                    .state
                    .order_manager
                    .current_step_rejected_orders
                    .len(),
                1
            );
            assert!(
                !engine
                    .state
                    .order_manager
                    .active_orders
                    .iter()
                    .any(|order| order.id == "order-b")
            );
        });
    }

    /// 驱动数据阶段直到处理下一根 bar(跳过日终结算定时器与结算后发出的收盘后
    /// 生命周期事件)。
    fn process_until_bar(
        processor: &mut DataProcessor,
        engine: &mut Engine,
        py: Python<'_>,
        strategy: &Bound<'_, PyAny>,
    ) {
        use crate::pipeline::processor::ProcessorResult;
        loop {
            let result = processor.process(engine, py, strategy).unwrap();
            assert_ne!(result, ProcessorResult::Break, "数据在下一根 bar 之前耗尽");
            if result == ProcessorResult::Next
                && matches!(engine.current_event, Some(crate::event::Event::Bar(_)))
            {
                return;
            }
        }
    }

    #[test]
    fn weekend_corporate_actions_apply_at_next_trading_day_open() {
        pyo3::Python::initialize();

        use crate::model::corporate_action::{CorporateAction, CorporateActionType};
        use chrono::NaiveDate;

        // 默认上海时区 + 工作日日历: 周六的拆股、周日的分红都顺延到周一开日处理,
        // 不能因为周末没有交易日而被丢弃。
        let mut engine = Engine::new();
        let symbol = "AAPL".to_string();
        engine
            .instruments
            .insert(symbol.clone(), create_instrument(&symbol));
        {
            let positions = Arc::make_mut(&mut engine.state.portfolio.positions);
            positions.insert(symbol.clone(), dec!(100));
            let available = Arc::make_mut(&mut engine.state.portfolio.available_positions);
            available.insert(symbol.clone(), dec!(100));
        }
        engine.state.portfolio.cash = dec!(0);
        engine.corporate_action_manager.add(CorporateAction {
            symbol: symbol.clone(),
            date: NaiveDate::from_ymd_opt(2024, 1, 6).unwrap(), // 周六
            action_type: CorporateActionType::Split,
            value: dec!(2.0),
        });
        engine.corporate_action_manager.add(CorporateAction {
            symbol: symbol.clone(),
            date: NaiveDate::from_ymd_opt(2024, 1, 7).unwrap(), // 周日
            action_type: CorporateActionType::Dividend,
            value: dec!(0.5),
        });

        let bar = |timestamp: i64| Bar {
            timestamp,
            symbol: symbol.clone(),
            open: dec!(10),
            high: dec!(10),
            low: dec!(10),
            close: dec!(10),
            volume: dec!(100),
            extra: HashMap::new(),
            freq: None,
            interval_start: None,
        };
        engine
            .state
            .feed
            .add_bar(bar(1_704_420_000_000_000_000))
            .unwrap(); // 01-05 周五 10:00
        engine
            .state
            .feed
            .add_bar(bar(1_704_679_200_000_000_000))
            .unwrap(); // 01-08 周一 10:00

        let mut processor = DataProcessor::new();
        pyo3::Python::attach(|py| {
            let strategy = py.import("builtins").unwrap().getattr("None").unwrap();

            processor.process(&mut engine, py, &strategy).unwrap();
            assert_eq!(
                engine.state.portfolio.positions.get(&symbol).unwrap(),
                &dec!(100)
            );
            assert_eq!(engine.state.portfolio.cash, dec!(0));

            // 跳过周五的日终结算与生命周期定时器, 直到处理周一的 bar
            process_until_bar(&mut processor, &mut engine, py, &strategy);
            assert_eq!(
                engine.current_trading_day,
                Some(NaiveDate::from_ymd_opt(2024, 1, 8).unwrap())
            );
            // 先拆股(周六)再分红(周日): 200 股 × 0.5
            assert_eq!(
                engine.state.portfolio.positions.get(&symbol).unwrap(),
                &dec!(200)
            );
            assert_eq!(engine.state.portfolio.cash, dec!(100));
        });
    }

    #[test]
    fn test_corporate_action_processing() {
        pyo3::Python::initialize();

        use crate::model::corporate_action::{CorporateAction, CorporateActionType};
        use chrono::NaiveDate;

        let mut engine = Engine::new();
        // 公司行动按会话交易日处理。2023-01-01 是周日, 按默认工作日日历周日的 bar
        // 会归入下一交易日(01-02), 拆股提前到 T1 发生; 本测试把三天都声明为交易日。
        engine.session_config = crate::engine::core::SessionConfig {
            sessions: None,
            trading_calendar: Some(
                ["2022-12-30", "2023-01-01", "2023-01-02", "2023-01-03"]
                    .iter()
                    .map(|d| NaiveDate::parse_from_str(d, "%Y-%m-%d").unwrap())
                    .collect(),
            ),
        };
        let symbol = "AAPL".to_string();
        engine
            .instruments
            .insert(symbol.clone(), create_instrument(&symbol));

        // Initial Position: 100 shares, Cash 0
        {
            let positions = Arc::make_mut(&mut engine.state.portfolio.positions);
            positions.insert(symbol.clone(), dec!(100));
            let available = Arc::make_mut(&mut engine.state.portfolio.available_positions);
            available.insert(symbol.clone(), dec!(100));
        }
        engine.state.portfolio.cash = dec!(0);

        // Add Split Action: 1-to-2 split on 2023-01-02
        let split_date = NaiveDate::from_ymd_opt(2023, 1, 2).unwrap();
        let split_action = CorporateAction {
            symbol: symbol.clone(),
            date: split_date,
            action_type: CorporateActionType::Split,
            value: dec!(2.0),
        };
        engine.corporate_action_manager.add(split_action);

        // Add Dividend Action: $0.5 per share on 2023-01-03
        let div_date = NaiveDate::from_ymd_opt(2023, 1, 3).unwrap();
        let div_action = CorporateAction {
            symbol: symbol.clone(),
            date: div_date,
            action_type: CorporateActionType::Dividend,
            value: dec!(0.5),
        };
        engine.corporate_action_manager.add(div_action);

        let mut processor = DataProcessor::new();

        // T1: 2023-01-01 (Before Split)
        let bar_t1 = Bar {
            timestamp: 1_672_531_200_000_000_000, // 2023-01-01
            symbol: symbol.clone(),
            open: dec!(100),
            high: dec!(100),
            low: dec!(100),
            close: dec!(100),
            volume: dec!(100),
            extra: HashMap::new(),
            freq: None,
            interval_start: None,
        };
        engine.state.feed.add_bar(bar_t1).unwrap();

        // T2: 2023-01-02 (Split Day)
        let bar_t2 = Bar {
            timestamp: 1_672_617_600_000_000_000, // 2023-01-02
            symbol: symbol.clone(),
            open: dec!(50),
            high: dec!(50),
            low: dec!(50),
            close: dec!(50),
            volume: dec!(100),
            extra: HashMap::new(),
            freq: None,
            interval_start: None,
        };
        engine.state.feed.add_bar(bar_t2).unwrap();

        // T3: 2023-01-03 (Dividend Day)
        let bar_t3 = Bar {
            timestamp: 1_672_704_000_000_000_000, // 2023-01-03
            symbol: symbol.clone(),
            open: dec!(50),
            high: dec!(50),
            low: dec!(50),
            close: dec!(50),
            volume: dec!(100),
            extra: HashMap::new(),
            freq: None,
            interval_start: None,
        };
        engine.state.feed.add_bar(bar_t3).unwrap();

        // Use Python::attach as recommended by PyO3 0.28+
        pyo3::Python::attach(|py| {
            let locals = py.import("builtins").unwrap();
            let strategy = locals.getattr("None").unwrap();

            // Process T1
            processor.process(&mut engine, py, &strategy).unwrap();
            // Verify T1 state: 100 shares
            assert_eq!(
                engine.state.portfolio.positions.get(&symbol).unwrap(),
                &dec!(100)
            );

            // Process T2 (Split happens at day start/end depending on logic, here logic is triggered by date change)
            // Our logic: when processing T2 event, we detect date change T1 -> T2, then trigger end-of-day actions for T2?
            // Wait, let's check DataProcessor logic:
            // if self.last_timestamp != 0 && timestamp > self.last_timestamp {
            //    ...
            //    engine.clock.update(timestamp, ...);
            //    let local_date = local_dt.date_naive();
            //    if engine.current_trading_day != Some(day) {
            //       // New trading day!
            //       engine.current_trading_day = Some(day);
            //       process_date(day) <--- This processes actions for the NEW trading day
            //    }
            // }

            // So when T2 bar comes in, date changes to 2023-01-02.
            // process_date(2023-01-02) is called. Split happens.

            // 跳过前一交易日的日终结算与生命周期定时器, 直到处理下一根 bar。
            process_until_bar(&mut processor, &mut engine, py, &strategy);

            // Verify T2 Split: 100 shares * 2 = 200 shares
            assert_eq!(
                engine.state.portfolio.positions.get(&symbol).unwrap(),
                &dec!(200)
            );

            // Process T3 (Dividend)
            // 跳过前一交易日的日终结算与生命周期定时器, 直到处理下一根 bar。
            process_until_bar(&mut processor, &mut engine, py, &strategy);

            // Verify T3 Dividend: 200 shares * 0.5 = 100 Cash
            assert_eq!(engine.state.portfolio.cash, dec!(100));
        });
    }

    /// 构造一个带 600000.SH 股票、持仓 100 股与周六拆股(1 拆 2)的引擎, 喂入给定日期 10:00 的 bar。
    fn resume_engine(days: &[(i32, u32, u32)]) -> Engine {
        use crate::model::corporate_action::{CorporateAction, CorporateActionType};
        use chrono::NaiveDate;

        let mut engine = Engine::new();
        let symbol = "600000.SH".to_string();
        engine
            .instruments
            .insert(symbol.clone(), create_instrument(&symbol));
        engine.corporate_action_manager.add(CorporateAction {
            symbol: symbol.clone(),
            date: NaiveDate::from_ymd_opt(2024, 1, 6).unwrap(),
            action_type: CorporateActionType::Split,
            value: dec!(2.0),
        });
        for &(y, mo, d) in days {
            // 本地(+08:00) 10:00 = UTC 02:00
            let ts = NaiveDate::from_ymd_opt(y, mo, d)
                .unwrap()
                .and_hms_opt(2, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap();
            engine
                .state
                .feed
                .add_bar(Bar {
                    timestamp: ts,
                    symbol: symbol.clone(),
                    open: dec!(10),
                    high: dec!(10),
                    low: dec!(10),
                    close: dec!(10),
                    volume: dec!(100),
                    extra: HashMap::new(),
                    freq: None,
                    interval_start: None,
                })
                .unwrap();
        }
        engine
    }

    /// 周五处理一根 bar 后存档(未结算周五), 在新引擎上恢复并喂入周一、周二。
    fn friday_checkpoint_then_resume(py: Python<'_>) -> Engine {
        let strategy = py.import("builtins").unwrap().getattr("None").unwrap();
        let symbol = "600000.SH".to_string();

        let mut first = resume_engine(&[(2024, 1, 5)]);
        {
            let positions = Arc::make_mut(&mut first.state.portfolio.positions);
            positions.insert(symbol.clone(), dec!(100));
            let available = Arc::make_mut(&mut first.state.portfolio.available_positions);
            available.insert(symbol.clone(), dec!(100));
        }
        let mut processor = DataProcessor::new();
        processor.process(&mut first, py, &strategy).unwrap();
        assert_eq!(
            first.current_trading_day,
            chrono::NaiveDate::from_ymd_opt(2024, 1, 5)
        );
        assert_eq!(first.last_settled_trading_day, None);
        let bytes = first.get_state_bytes(py).unwrap();

        let mut second = resume_engine(&[(2024, 1, 8), (2024, 1, 9)]);
        second.load_state_bytes(&bytes).unwrap();
        let mut processor = DataProcessor::new();
        // 周一
        while processor.process(&mut second, py, &strategy).unwrap()
            == crate::pipeline::processor::ProcessorResult::Loop
        {}
        second
    }

    #[test]
    fn test_resume_applies_saturday_split_exactly_once() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let symbol = "600000.SH".to_string();
            let strategy = py.import("builtins").unwrap().getattr("None").unwrap();
            let mut engine = friday_checkpoint_then_resume(py);
            // 周一开日: 周六拆股生效一次
            assert_eq!(
                engine.state.portfolio.positions.get(&symbol).unwrap(),
                &dec!(200)
            );
            // 周二不得再次拆股
            let mut processor = DataProcessor::new();
            processor.process(&mut engine, py, &strategy).unwrap();
            while processor.process(&mut engine, py, &strategy).unwrap()
                == crate::pipeline::processor::ProcessorResult::Loop
            {}
            assert_eq!(
                engine.state.portfolio.positions.get(&symbol).unwrap(),
                &dec!(200)
            );
        });
    }

    #[test]
    fn test_stale_day_close_timer_does_not_resettle_older_day() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            use chrono::NaiveDate;
            let strategy = py.import("builtins").unwrap().getattr("None").unwrap();
            let mut engine = resume_engine(&[(2024, 1, 9)]);
            let mon = NaiveDate::from_ymd_opt(2024, 1, 8).unwrap();
            engine.current_trading_day = Some(mon);
            engine.last_settled_trading_day = Some(mon);
            // 早于已结算交易日的残留收盘定时器(周五 15:00 + 1ns, +08:00 = UTC 07:00)
            let fri_close = NaiveDate::from_ymd_opt(2024, 1, 5)
                .unwrap()
                .and_hms_opt(7, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap();
            engine.timers.push(crate::model::Timer {
                timestamp: fri_close + 1,
                payload: format!("{}2024-01-05", crate::model::timer::DAY_CLOSE_TIMER_PREFIX),
            });
            let mut processor = DataProcessor::new();
            assert_eq!(
                processor.process(&mut engine, py, &strategy).unwrap(),
                crate::pipeline::processor::ProcessorResult::Loop
            );
            // 不回头结算更早的交易日, 已结算进度不回退
            assert_eq!(engine.last_settled_trading_day, Some(mon));
            assert!(engine.statistics_manager.snapshots.is_empty());
        });
    }

    #[test]
    fn test_resume_catchup_settlement_uses_session_close() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let engine = friday_checkpoint_then_resume(py);
            // 周五 15:00(+08:00) = UTC 07:00
            let fri_close = chrono::NaiveDate::from_ymd_opt(2024, 1, 5)
                .unwrap()
                .and_hms_opt(7, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap();
            assert_eq!(
                engine.last_settled_trading_day,
                chrono::NaiveDate::from_ymd_opt(2024, 1, 5)
            );
            let stamps: Vec<i64> = engine
                .statistics_manager
                .snapshots
                .iter()
                .map(|(t, _)| *t)
                .collect();
            assert_eq!(stamps, vec![fri_close]);
        });
    }
}
