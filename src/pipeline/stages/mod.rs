mod channel;
mod cleanup;
mod data;
mod execution;
mod shared;
mod statistics;
mod strategy;

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

        engine.last_prices.insert("A".to_string(), dec!(100));
        engine.last_prices.insert("B".to_string(), dec!(200));

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

            engine.last_prices.insert("B".to_string(), dec!(205));

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
        engine.last_prices.insert("A".to_string(), dec!(100));
        engine.last_prices.insert("B".to_string(), dec!(100));

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

    #[test]
    fn test_corporate_action_processing() {
        pyo3::Python::initialize();

        use crate::model::corporate_action::{CorporateAction, CorporateActionType};
        use chrono::NaiveDate;

        let mut engine = Engine::new();
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
            //    if engine.current_date != Some(local_date) {
            //       // New Day!
            //       engine.current_date = Some(local_date);
            //       process_date(local_date) <--- This processes actions for the NEW date
            //    }
            // }

            // So when T2 bar comes in, date changes to 2023-01-02.
            // process_date(2023-01-02) is called. Split happens.

            processor.process(&mut engine, py, &strategy).unwrap();

            // Verify T2 Split: 100 shares * 2 = 200 shares
            assert_eq!(
                engine.state.portfolio.positions.get(&symbol).unwrap(),
                &dec!(200)
            );

            // Process T3 (Dividend)
            processor.process(&mut engine, py, &strategy).unwrap();

            // Verify T3 Dividend: 200 shares * 0.5 = 100 Cash
            assert_eq!(engine.state.portfolio.cash, dec!(100));
        });
    }
}
