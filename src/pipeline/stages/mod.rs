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
            custom_sessions: None,
            trading_calendar: Some(
                ["2022-12-30", "2023-01-01", "2023-01-02", "2023-01-03"]
                    .iter()
                    .map(|d| NaiveDate::parse_from_str(d, "%Y-%m-%d").unwrap())
                    .collect(),
            ),
            ..Default::default()
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
                owner: None,
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

    /// 本地(+08:00) 时刻的 UTC 纳秒。
    fn sh_ns(m: u32, d: u32, h: u32, min: u32) -> i64 {
        chrono::NaiveDate::from_ymd_opt(2024, m, d)
            .unwrap()
            .and_hms_opt(h, min, 0)
            .unwrap()
            .and_utc()
            .timestamp_nanos_opt()
            .unwrap()
            - 8 * 3_600_000_000_000
    }

    /// 实盘引擎: 600000.SH, 显式交易日历, 堆中只有引擎按墙钟排的生命周期。
    fn live_calendar_engine(days: &[(u32, u32)]) -> Engine {
        let mut engine = Engine::new();
        engine.state.feed = crate::data::DataFeed::create_live();
        let symbol = "600000.SH".to_string();
        engine
            .instruments
            .insert(symbol.clone(), create_instrument(&symbol));
        engine.session_config.trading_calendar = Some(
            days.iter()
                .map(|&(m, d)| chrono::NaiveDate::from_ymd_opt(2024, m, d).unwrap())
                .collect(),
        );
        engine.invalidate_session_resolver();
        engine
    }

    fn live_bar(ts: i64) -> Bar {
        Bar {
            timestamp: ts,
            symbol: "600000.SH".to_string(),
            open: dec!(10),
            high: dec!(10),
            low: dec!(10),
            close: dec!(10),
            volume: dec!(100),
            extra: HashMap::new(),
            freq: None,
            interval_start: None,
        }
    }

    fn day_close_timer_count(engine: &Engine, day: &str) -> usize {
        let key = format!("{}{day}", crate::model::timer::DAY_CLOSE_TIMER_PREFIX);
        engine.timers.iter().filter(|t| t.payload == key).count()
    }

    fn snapshot_stamps(engine: &Engine) -> Vec<i64> {
        engine
            .statistics_manager
            .snapshots
            .iter()
            .map(|(t, _)| *t)
            .collect()
    }

    #[test]
    fn test_live_replay_close_ends_session_right_after_last_day_settles() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let strategy = py.import("builtins").unwrap().getattr("None").unwrap();
            let mut engine = live_calendar_engine(&[(1, 2), (1, 3), (1, 4)]);
            engine.live_day_close_delay_ns = 0;
            // replay 锚定 01-02 10:00; 最后事件 01-02 15:00 已到收盘
            engine
                .schedule_live_lifecycle_anchored(sh_ns(1, 2, 10, 0) - 1)
                .unwrap();
            assert!(
                engine
                    .set_live_replay_end(Some(sh_ns(1, 2, 15, 0)))
                    .unwrap()
            );
            if let Some(close) = engine.live_replay_close.as_mut() {
                close.last_event_seen = true;
            }
            let day2 = chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
            let mut processor = DataProcessor::new();
            let mut results = Vec::new();
            for _ in 0..10 {
                let result = processor.process(&mut engine, py, &strategy).unwrap();
                let stop = result == crate::pipeline::ProcessorResult::Break;
                results.push(result);
                if stop {
                    break;
                }
            }
            assert_eq!(
                results.last(),
                Some(&crate::pipeline::ProcessorResult::Break),
                "{results:?}"
            );
            assert_eq!(engine.last_settled_trading_day, Some(day2));
            assert!(engine.live_replay_close.is_none());
            // 结束前最后发出的是 01-02 的收盘后事件, 下一交易日的盘前未触发
            match engine.current_event.as_ref() {
                Some(crate::event::Event::Timer(t)) => assert_eq!(
                    t.lifecycle(),
                    Some((crate::model::timer::LIFECYCLE_AFTER_TRADING, day2))
                ),
                other => panic!("未发出收盘后事件: {other:?}"),
            }
            assert!(engine.timers.iter().any(|t| {
                t.lifecycle().is_some_and(|(kind, d)| {
                    kind == crate::model::timer::LIFECYCLE_PRE_OPEN
                        && d == chrono::NaiveDate::from_ymd_opt(2024, 1, 3).unwrap()
                })
            }));
        });
    }

    #[test]
    fn test_live_replay_close_without_day_close_timer_ends_without_waiting() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let strategy = py.import("builtins").unwrap().getattr("None").unwrap();
            let mut engine = live_calendar_engine(&[(1, 2), (1, 3)]);
            let day2 = chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
            let mut state = crate::engine::core::LiveReplayClose::new(day2, sh_ns(1, 2, 15, 0));
            state.last_event_seen = true;
            engine.live_replay_close = Some(state);
            // 末日没有收盘定时器: 等不到结算, 本轮直接结束, 不开始计时
            let mut processor = DataProcessor::new();
            let result = processor.process(&mut engine, py, &strategy).unwrap();
            assert_eq!(result, crate::pipeline::ProcessorResult::Break);
            assert!(engine.live_replay_close.is_none());
            assert_eq!(engine.last_settled_trading_day, None);
        });
    }

    #[test]
    fn test_live_replay_close_wait_times_out_and_ends_session() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let strategy = py.import("builtins").unwrap().getattr("None").unwrap();
            let mut engine = live_calendar_engine(&[(1, 2), (1, 3)]);
            let day2 = chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
            let mut state = crate::engine::core::LiveReplayClose::new(day2, sh_ns(1, 2, 15, 0));
            state.last_event_seen = true;
            engine.live_replay_close = Some(state);
            // 末日收盘定时器已登记但(墙钟上)迟迟不到期
            assert!(engine.push_day_close_timer(day2, i64::MAX / 2));
            // 最后事件已处理、行情已取空: 开始计时(5 秒), 本轮不结束
            let mut processor = DataProcessor::new();
            let before = chrono::Utc::now().timestamp_nanos_opt().unwrap();
            let result = processor.process(&mut engine, py, &strategy).unwrap();
            assert_ne!(result, crate::pipeline::ProcessorResult::Break);
            let deadline = engine
                .live_replay_close
                .and_then(|c| c.deadline_ns)
                .expect("应开始计时");
            assert!(deadline >= before + crate::engine::core::LiveReplayClose::WAIT_NS);
            // 超时未结算: 告警并结束, 不结算
            if let Some(close) = engine.live_replay_close.as_mut() {
                close.deadline_ns = Some(before);
            }
            let result = processor.process(&mut engine, py, &strategy).unwrap();
            assert_eq!(result, crate::pipeline::ProcessorResult::Break);
            assert!(engine.live_replay_close.is_none());
            assert_eq!(engine.last_settled_trading_day, None);
        });
    }

    #[test]
    fn test_live_no_event_day_settles_by_calendar_close_timer() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let strategy = py.import("builtins").unwrap().getattr("None").unwrap();
            let mut engine = live_calendar_engine(&[(1, 2), (1, 3)]);
            // 墙钟在 01-02 盘前: 排当日生命周期时同时登记当日收盘定时器(只一个)
            engine.schedule_live_lifecycle(sh_ns(1, 2, 9, 0)).unwrap();
            assert_eq!(day_close_timer_count(&engine, "2024-01-02"), 1);
            let close = sh_ns(1, 2, 15, 0);
            let grace = engine.live_day_close_delay_ns;
            assert!(
                engine
                    .timers
                    .iter()
                    .any(|t| t.is_day_close() && t.timestamp == close + grace)
            );
            // 当日无任何行情: 墙钟(2024 年的时刻均已过)推进过收盘 + 宽限
            let mut processor = DataProcessor::new();
            while engine.last_settled_trading_day.is_none()
                && engine
                    .timers
                    .peek()
                    .is_some_and(|t| t.timestamp <= close + grace)
            {
                processor.process(&mut engine, py, &strategy).unwrap();
            }
            assert_eq!(day_close_timer_count(&engine, "2024-01-02"), 0);
            assert_eq!(
                engine.last_settled_trading_day,
                chrono::NaiveDate::from_ymd_opt(2024, 1, 2)
            );
            assert_eq!(snapshot_stamps(&engine), vec![close]);
            // 结算了就算开过: 当前交易日推进到 01-02 并记入已开交易日, 日收益按它分组
            let day2 = chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
            assert_eq!(engine.current_trading_day, Some(day2));
            assert_eq!(engine.trading_day_ends, vec![(day2, close)]);
            assert_eq!(
                crate::analysis::BacktestResult::trading_day_key(
                    close,
                    &engine.trading_day_ends,
                    Some("Asia/Shanghai"),
                    28_800,
                ),
                day2
            );
            // 会话在此后、下一交易日开日前结束: 收尾不再判当日未结算(不多结算、不多记快照)
            assert!(!engine.current_day_unsettled());
            // 收盘后事件(on_after_trading)在结算定时器时刻发出
            match engine.current_event.as_ref() {
                Some(crate::event::Event::Timer(t)) => {
                    assert_eq!(
                        t.lifecycle(),
                        Some((
                            crate::model::timer::LIFECYCLE_AFTER_TRADING,
                            chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
                        ))
                    );
                    assert_eq!(t.timestamp, close + grace);
                }
                other => panic!("未发出收盘后事件: {other:?}"),
            }
        });
    }

    #[test]
    fn test_live_open_day_does_not_duplicate_calendar_close_timer() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let strategy = py.import("builtins").unwrap().getattr("None").unwrap();
            let mut engine = live_calendar_engine(&[(1, 2), (1, 3)]);
            engine.schedule_live_lifecycle(sh_ns(1, 2, 9, 0)).unwrap();
            engine
                .state
                .feed
                .add_bar(live_bar(sh_ns(1, 2, 10, 0)))
                .unwrap();
            let mut processor = DataProcessor::new();
            while engine.current_trading_day.is_none() {
                processor.process(&mut engine, py, &strategy).unwrap();
            }
            // 开日路径按键去重: 01-02 收盘定时器仍只有一个
            assert_eq!(day_close_timer_count(&engine, "2024-01-02"), 1);
        });
    }

    #[test]
    fn test_live_no_event_day_between_event_days_settles_each_day_once() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let strategy = py.import("builtins").unwrap().getattr("None").unwrap();
            let mut engine = live_calendar_engine(&[(1, 2), (1, 3), (1, 4)]);
            engine.schedule_live_lifecycle(sh_ns(1, 2, 9, 0)).unwrap();
            engine
                .state
                .feed
                .add_bar(live_bar(sh_ns(1, 2, 10, 0)))
                .unwrap();
            engine
                .state
                .feed
                .add_bar(live_bar(sh_ns(1, 4, 10, 0)))
                .unwrap();
            let day4 = chrono::NaiveDate::from_ymd_opt(2024, 1, 4);
            let mut processor = DataProcessor::new();
            while engine.current_trading_day != day4 {
                processor.process(&mut engine, py, &strategy).unwrap();
            }
            // 01-03 无行情也按日历结算; 01-04 开日不回头重复结算 01-02
            assert_eq!(
                snapshot_stamps(&engine),
                vec![sh_ns(1, 2, 15, 0), sh_ns(1, 3, 15, 0)]
            );
            assert_eq!(
                engine.last_settled_trading_day,
                chrono::NaiveDate::from_ymd_opt(2024, 1, 3)
            );
            // 无行情的 01-03 也记入已开交易日, 其收盘快照按 01-03 分组(不被 01-04 吞掉)
            let d = |day| chrono::NaiveDate::from_ymd_opt(2024, 1, day).unwrap();
            assert_eq!(
                engine.trading_day_ends,
                vec![
                    (d(2), sh_ns(1, 2, 15, 0)),
                    (d(3), sh_ns(1, 3, 15, 0)),
                    (d(4), sh_ns(1, 4, 15, 0)),
                ]
            );
            assert_eq!(
                crate::analysis::BacktestResult::trading_day_key(
                    sh_ns(1, 3, 15, 0),
                    &engine.trading_day_ends,
                    Some("Asia/Shanghai"),
                    28_800,
                ),
                d(3)
            );
        });
    }

    #[test]
    fn test_live_no_event_day_expires_next_open_day_order_at_its_close() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let strategy = py.import("builtins").unwrap().getattr("None").unwrap();
            let mut engine = live_calendar_engine(&[(1, 2), (1, 3), (1, 4)]);
            engine.schedule_live_lifecycle(sh_ns(1, 2, 9, 0)).unwrap();
            engine
                .state
                .feed
                .add_bar(live_bar(sh_ns(1, 2, 10, 0)))
                .unwrap();
            // 01-02 收盘后(on_after_trading 里)下的 NextOpen Day 单, 属交易日 01-03
            let mut order = crate::model::Order::test_new(
                "o-after-close",
                "600000.SH",
                crate::model::OrderSide::Buy,
                crate::model::OrderType::Limit,
                dec!(100),
            );
            order.price = Some(dec!(1));
            order.created_at = sh_ns(1, 2, 15, 1);
            engine.state.order_manager.active_orders.push(order);
            let day = |d| chrono::NaiveDate::from_ymd_opt(2024, 1, d);
            let mut processor = DataProcessor::new();
            while engine.last_settled_trading_day != day(2) {
                processor.process(&mut engine, py, &strategy).unwrap();
            }
            // 01-02 结算: 建于当日最后一个事件之后, 豁免
            assert_eq!(engine.state.order_manager.active_orders.len(), 1);
            while engine.last_settled_trading_day != day(3) {
                processor.process(&mut engine, py, &strategy).unwrap();
            }
            // 01-03 无行情: 与有行情日一样在所属交易日收盘过期, 不活到 01-04
            assert!(engine.state.order_manager.active_orders.is_empty());
            let expired = &engine.state.order_manager.current_step_expired_orders;
            assert_eq!(expired.len(), 1);
            assert_eq!(expired[0].id, "o-after-close");
            assert_eq!(expired[0].updated_at, sh_ns(1, 3, 15, 0));
        });
    }

    fn live_tick(ts: i64, price: rust_decimal::Decimal) -> crate::model::Tick {
        crate::model::Tick {
            timestamp: ts,
            price,
            volume: dec!(100),
            symbol: "600000.SH".to_string(),
        }
    }

    /// 数据阶段逐步驱动时记下的一步: (标签, 时刻, 上下文交易日, 当前交易日)。
    type GraceStep = (
        String,
        i64,
        Option<chrono::NaiveDate>,
        Option<chrono::NaiveDate>,
    );

    fn grace_step(engine: &mut Engine) -> Option<GraceStep> {
        let (label, ts) = match engine.current_event.as_ref()? {
            crate::event::Event::Bar(b) => ("bar".to_string(), b.timestamp),
            crate::event::Event::Tick(t) => ("tick".to_string(), t.timestamp),
            crate::event::Event::Timer(t) => (t.payload.clone(), t.timestamp),
            _ => return None,
        };
        let ctx_day = engine.context_sessions().trading_day;
        Some((label, ts, ctx_day, engine.current_trading_day))
    }

    #[test]
    fn test_live_post_close_tick_inside_grace_belongs_to_same_day() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let strategy = py.import("builtins").unwrap().getattr("None").unwrap();
            let mut engine = live_calendar_engine(&[(1, 2), (1, 3)]);
            engine.live_day_close_delay_ns = 60_000_000_000;
            engine.schedule_live_lifecycle(sh_ns(1, 2, 9, 0)).unwrap();
            let symbol = "600000.SH".to_string();
            Arc::make_mut(&mut engine.state.portfolio.positions).insert(symbol.clone(), dec!(100));
            let sec = 1_000_000_000_i64;
            let close = sh_ns(1, 2, 15, 0);
            let bar_ts = sh_ns(1, 2, 14, 59) + 30 * sec;
            let grace1 = close + sec / 2;
            let grace2 = close + 30 * sec;
            let next_bar = sh_ns(1, 3, 9, 31);
            engine.state.feed.add_bar(live_bar(bar_ts)).unwrap();
            engine
                .state
                .feed
                .add_tick(live_tick(grace1, dec!(11)))
                .unwrap();
            engine
                .state
                .feed
                .add_tick(live_tick(grace2, dec!(12)))
                .unwrap();
            engine.state.feed.add_bar(live_bar(next_bar)).unwrap();
            let d2 = chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
            let d3 = chrono::NaiveDate::from_ymd_opt(2024, 1, 3).unwrap();
            let mut processor = DataProcessor::new();
            let mut steps: Vec<GraceStep> = Vec::new();
            // 结算快照首次出现时的步骤(结算由哪一步触发)
            let mut snapshot_seen_at: Option<GraceStep> = None;
            let mut corp_before_next_bar = None;
            loop {
                let before = engine.last_corp_action_date;
                let result = processor.process(&mut engine, py, &strategy).unwrap();
                assert_ne!(result, crate::pipeline::ProcessorResult::Break);
                if result != crate::pipeline::ProcessorResult::Next {
                    continue;
                }
                let step = grace_step(&mut engine).expect("应有当前事件");
                // 最新价由数据阶段之后的策略阶段写入: 这里模拟
                let price = match engine.current_event.as_ref() {
                    Some(crate::event::Event::Bar(b)) => Some(b.close),
                    Some(crate::event::Event::Tick(t)) => Some(t.price),
                    _ => None,
                };
                if let Some(p) = price {
                    engine
                        .last_prices
                        .write()
                        .unwrap()
                        .insert(symbol.clone(), p);
                }
                if snapshot_seen_at.is_none() && !snapshot_stamps(&engine).is_empty() {
                    snapshot_seen_at = Some(step.clone());
                }
                let done = step.0 == "bar" && step.1 == next_bar;
                if done {
                    corp_before_next_bar = before;
                }
                steps.push(step);
                if done {
                    break;
                }
            }
            let ticks: Vec<&GraceStep> = steps.iter().filter(|s| s.0 == "tick").collect();
            assert_eq!(
                ticks,
                vec![
                    &("tick".to_string(), grace1, Some(d2), Some(d2)),
                    &("tick".to_string(), grace2, Some(d2), Some(d2)),
                ],
                "{steps:?}"
            );
            // 结算由 15:01 的收盘定时器触发(收盘后事件在其时刻发出), 快照记在收盘, 只一次
            let after: Vec<&GraceStep> = steps
                .iter()
                .filter(|s| s.0.contains("after_trading"))
                .collect();
            assert_eq!(after.len(), 1, "{steps:?}");
            assert_eq!(after[0].1, close + 60 * sec);
            let seen = snapshot_seen_at.expect("应有结算快照");
            assert!(seen.0.contains("after_trading"), "{seen:?}");
            assert_eq!(snapshot_stamps(&engine), vec![close]);
            // 结算价是收盘时刻的最新价(不含宽限期内的 tick)
            let snap = &engine.statistics_manager.snapshots[0].1;
            assert_eq!(snap.len(), 1);
            assert_eq!(snap[0].close, 10.0);
            // D+1 的盘前先于其首个 bar, 盘前时 D+1 尚未开日; 公司行动到首个事件才处理
            let pre_open_idx = steps
                .iter()
                .position(|s| s.0.ends_with("pre_open|2024-01-03"))
                .expect("应有 01-03 盘前");
            assert!(pre_open_idx < steps.len() - 1, "{steps:?}");
            assert_eq!(steps[pre_open_idx].3, Some(d2));
            assert_eq!(corp_before_next_bar, Some(d2));
            assert_eq!(steps.last().unwrap().3, Some(d3));
            assert_eq!(engine.last_corp_action_date, Some(d3));
            // 日收益分组: 宽限期内的权益点归 D
            assert_eq!(
                crate::analysis::BacktestResult::trading_day_key(
                    grace2,
                    &engine.trading_day_ends,
                    Some("Asia/Shanghai"),
                    28_800,
                ),
                d2
            );
            assert_eq!(
                crate::analysis::BacktestResult::trading_day_key(
                    next_bar,
                    &engine.trading_day_ends,
                    Some("Asia/Shanghai"),
                    28_800,
                ),
                d3
            );
        });
    }

    #[test]
    fn test_live_always_open_next_midnight_event_inside_grace_opens_next_day() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let strategy = py.import("builtins").unwrap().getattr("None").unwrap();
            let mut engine = live_calendar_engine(&[(1, 2), (1, 3)]);
            engine.session_config.sessions =
                Some(vec![("default".to_string(), "ALWAYS_OPEN".to_string())]);
            engine.invalidate_session_resolver();
            engine.live_day_close_delay_ns = 60_000_000_000;
            let close = sh_ns(1, 3, 0, 0);
            let late = close + 30_000_000_000;
            engine
                .state
                .feed
                .add_bar(live_bar(sh_ns(1, 2, 10, 0)))
                .unwrap();
            engine.state.feed.add_bar(live_bar(late)).unwrap();
            let d2 = chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
            let d3 = chrono::NaiveDate::from_ymd_opt(2024, 1, 3).unwrap();
            let mut processor = DataProcessor::new();
            loop {
                let result = processor.process(&mut engine, py, &strategy).unwrap();
                assert_ne!(result, crate::pipeline::ProcessorResult::Break);
                if result == crate::pipeline::ProcessorResult::Next
                    && matches!(
                        engine.current_event,
                        Some(crate::event::Event::Bar(ref b)) if b.timestamp == late
                    )
                {
                    break;
                }
            }
            // 收盘(次日 0 点)后 30 秒不早于 D+1 首段起点: 归 D+1, 开日时补结算 D
            assert_eq!(engine.current_trading_day, Some(d3));
            assert_eq!(engine.context_sessions().trading_day, Some(d3));
            assert_eq!(engine.last_settled_trading_day, Some(d2));
            assert_eq!(snapshot_stamps(&engine), vec![close]);
        });
    }

    #[test]
    fn test_live_grace_stamped_event_after_settlement_belongs_to_next_day() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let strategy = py.import("builtins").unwrap().getattr("None").unwrap();
            let mut engine = live_calendar_engine(&[(1, 2), (1, 3)]);
            engine.live_day_close_delay_ns = 60_000_000_000;
            engine.schedule_live_lifecycle(sh_ns(1, 2, 9, 0)).unwrap();
            engine
                .state
                .feed
                .add_bar(live_bar(sh_ns(1, 2, 14, 59)))
                .unwrap();
            let d2 = chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
            let d3 = chrono::NaiveDate::from_ymd_opt(2024, 1, 3).unwrap();
            let mut processor = DataProcessor::new();
            // 行情取空后墙钟已过收盘 + 宽限: 收盘定时器结算 D
            while engine.last_settled_trading_day != Some(d2) {
                processor.process(&mut engine, py, &strategy).unwrap();
            }
            let close = sh_ns(1, 2, 15, 0);
            assert_eq!(snapshot_stamps(&engine), vec![close]);
            // 结算之后才到、时间戳仍在宽限内的行情: 归 D+1, 不重复结算 D
            let late = close + 30_000_000_000;
            engine
                .state
                .feed
                .add_tick(live_tick(late, dec!(10)))
                .unwrap();
            loop {
                let result = processor.process(&mut engine, py, &strategy).unwrap();
                if result == crate::pipeline::ProcessorResult::Next
                    && matches!(engine.current_event, Some(crate::event::Event::Tick(_)))
                {
                    break;
                }
            }
            assert_eq!(engine.current_trading_day, Some(d3));
            assert_eq!(engine.context_sessions().trading_day, Some(d3));
            assert_eq!(engine.last_settled_trading_day, Some(d2));
            assert_eq!(snapshot_stamps(&engine), vec![close]);
        });
    }

    #[test]
    fn test_backtest_post_close_tick_still_belongs_to_next_day() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let strategy = py.import("builtins").unwrap().getattr("None").unwrap();
            let mut engine = live_calendar_engine(&[(1, 2), (1, 3)]);
            engine.state.feed = crate::data::DataFeed::new();
            engine.live_day_close_delay_ns = 60_000_000_000;
            let close = sh_ns(1, 2, 15, 0);
            let tick_ts = close + 500_000_000;
            engine
                .state
                .feed
                .add_bar(live_bar(sh_ns(1, 2, 14, 59)))
                .unwrap();
            engine
                .state
                .feed
                .add_tick(live_tick(tick_ts, dec!(10)))
                .unwrap();
            engine
                .state
                .feed
                .add_bar(live_bar(sh_ns(1, 3, 9, 31)))
                .unwrap();
            let d2 = chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
            let d3 = chrono::NaiveDate::from_ymd_opt(2024, 1, 3).unwrap();
            let mut processor = DataProcessor::new();
            loop {
                let result = processor.process(&mut engine, py, &strategy).unwrap();
                assert_ne!(result, crate::pipeline::ProcessorResult::Break);
                if result == crate::pipeline::ProcessorResult::Next
                    && matches!(engine.current_event, Some(crate::event::Event::Tick(_)))
                {
                    break;
                }
            }
            // 回测不变: 收盘 + 1ns 结算, 收盘后的 tick 属下一交易日
            assert_eq!(engine.current_trading_day, Some(d3));
            assert_eq!(engine.context_sessions().trading_day, Some(d3));
            assert_eq!(engine.last_settled_trading_day, Some(d2));
            assert_eq!(snapshot_stamps(&engine), vec![close]);
        });
    }

    #[test]
    fn test_live_replay_end_waits_for_close_when_last_event_inside_grace() {
        let mut engine = live_calendar_engine(&[(1, 2), (1, 3)]);
        engine.live_day_close_delay_ns = 60_000_000_000;
        let close = sh_ns(1, 2, 15, 0);
        // 最后事件在 01-02 收盘宽限内: 归 01-02, 等其收盘定时器结算后再结束
        assert!(
            engine
                .set_live_replay_end(Some(close + 30_000_000_000))
                .unwrap()
        );
        let state = engine.live_replay_close.expect("应设置收尾状态");
        assert_eq!(
            state.day,
            chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap()
        );
        // 宽限之外(收盘 + 61 秒)按日历属 01-03, 未到其收盘, 不等待
        assert!(
            !engine
                .set_live_replay_end(Some(close + 61_000_000_000))
                .unwrap()
        );
    }
}
