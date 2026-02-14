use crate::event::Event;
use crate::model::{Order, OrderSide, OrderStatus, OrderType, TimeInForce, Trade, TradingSession};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use uuid::Uuid;

/// 滑点模型特征
pub trait SlippageModel: Send + Sync {
    /// 计算滑点后的成交价
    ///
    /// :param price: 理论成交价
    /// :param quantity: 成交数量
    /// :param side: 交易方向
    /// :return: 实际成交价
    fn calculate_price(&self, price: Decimal, quantity: Decimal, side: OrderSide) -> Decimal;
}

/// 零滑点模型 (默认)
#[derive(Debug, Clone, Copy, Default)]
pub struct ZeroSlippage;

impl SlippageModel for ZeroSlippage {
    fn calculate_price(&self, price: Decimal, _quantity: Decimal, _side: OrderSide) -> Decimal {
        price
    }
}

/// 固定值滑点模型
/// 每单位价格增加/减少固定数值
#[derive(Debug, Clone, Copy)]
pub struct FixedSlippage {
    pub delta: Decimal,
}

impl SlippageModel for FixedSlippage {
    fn calculate_price(&self, price: Decimal, _quantity: Decimal, side: OrderSide) -> Decimal {
        match side {
            OrderSide::Buy => price + self.delta,
            OrderSide::Sell => price - self.delta,
        }
    }
}

/// 百分比滑点模型
/// 价格增加/减少固定百分比
#[derive(Debug, Clone, Copy)]
pub struct PercentSlippage {
    pub rate: Decimal,
}

impl SlippageModel for PercentSlippage {
    fn calculate_price(&self, price: Decimal, _quantity: Decimal, side: OrderSide) -> Decimal {
        match side {
            OrderSide::Buy => price * (Decimal::ONE + self.rate),
            OrderSide::Sell => price * (Decimal::ONE - self.rate),
        }
    }
}

/// 交易执行接口 (Execution Client Trait)
pub trait ExecutionClient: Send + Sync {
    /// 接收新订单
    fn on_order(&mut self, order: Order);

    /// 取消订单请求
    fn on_cancel(&mut self, order_id: &str);

    /// 处理市场事件并返回执行报告
    fn on_event(
        &mut self,
        event: &Event,
        instruments: &std::collections::HashMap<String, crate::model::Instrument>,
        execution_mode: crate::model::ExecutionMode,
        bar_index: usize,
        session: TradingSession,
    ) -> Vec<Event>;

    /// 设置滑点模型 (仅回测有效)
    fn set_slippage_model(&mut self, _model: Box<dyn SlippageModel>) {}

    /// 设置成交量限制 (仅回测有效)
    fn set_volume_limit(&mut self, _limit: f64) {}

    /// 是否为实盘模式
    #[allow(dead_code)]
    fn is_live(&self) -> bool {
        false
    }
}

/// 模拟交易所执行器 (Simulated Execution Client)
/// 负责在内存中撮合订单 (回测模式)
pub struct SimulatedExecutionClient {
    slippage_model: Box<dyn SlippageModel>,
    volume_limit_pct: Decimal, // 成交量限制比例 (0.0 = 不限制)
    pending_orders: Vec<Order>,
}

impl SimulatedExecutionClient {
    pub fn new() -> Self {
        SimulatedExecutionClient {
            slippage_model: Box::new(ZeroSlippage),
            volume_limit_pct: Decimal::ZERO,
            pending_orders: Vec::new(),
        }
    }
}

impl ExecutionClient for SimulatedExecutionClient {
    fn set_slippage_model(&mut self, model: Box<dyn SlippageModel>) {
        self.slippage_model = model;
    }

    fn set_volume_limit(&mut self, limit: f64) {
        self.volume_limit_pct = Decimal::from_f64(limit).unwrap_or(Decimal::ZERO);
    }

    fn on_order(&mut self, order: Order) {
        // 模拟交易所接收订单
        let mut order = order;
        if order.status == OrderStatus::New {
            order.status = OrderStatus::Submitted;
        }
        self.pending_orders.push(order);
    }

    fn on_cancel(&mut self, order_id: &str) {
        if let Some(order) = self.pending_orders.iter_mut().find(|o| o.id == order_id) {
            if order.status == OrderStatus::Submitted || order.status == OrderStatus::New {
                order.status = OrderStatus::Cancelled;
                // Note: In a real system, we would generate an ExecutionReport for cancellation here.
                // But on_event loop will pick it up or we should return events from on_cancel too?
                // For simplicity, we just mark it. The loop will clean it up or we handle it in on_event.
                // Better: Keep it as Cancelled, on_event will generate report and remove it?
                // Or just mark it and let the engine see the state?
                // Let's assume on_event handles reporting.
            }
        }
    }

    fn on_event(
        &mut self,
        event: &Event,
        instruments: &std::collections::HashMap<String, crate::model::Instrument>,
        execution_mode: crate::model::ExecutionMode,
        bar_index: usize,
        session: TradingSession,
    ) -> Vec<Event> {
        let mut reports = Vec::new();

        // Skip matching during non-trading sessions
        if session == TradingSession::Break
            || session == TradingSession::Closed
            || session == TradingSession::PreOpen
            || session == TradingSession::PostClose
        {
            return reports;
        }

        // 实际撮合逻辑：遍历所有挂单，看当前 Event 是否满足成交条件
        for order in self.pending_orders.iter_mut() {
            // Check for cancellation first
            if order.status == OrderStatus::Cancelled {
                reports.push(Event::ExecutionReport(order.clone(), None));
                continue;
            }

            if order.status != OrderStatus::New && order.status != OrderStatus::Submitted {
                continue;
            }
            // 0. 检查最小交易单位 (Lot Size)
            // 仅针对买入订单，且必须存在标的定义
            if order.side == OrderSide::Buy {
                if let Some(instrument) = instruments.get(&order.symbol) {
                    if order.quantity % instrument.lot_size != Decimal::ZERO {
                        order.status = OrderStatus::Rejected;
                        order.reject_reason = format!(
                            "Quantity {} is not a multiple of lot size {}",
                            order.quantity, instrument.lot_size
                        );
                        match event {
                            Event::Bar(b) => order.updated_at = b.timestamp,
                            Event::Tick(t) => order.updated_at = t.timestamp,
                            _ => {}
                        }
                        reports.push(Event::ExecutionReport(order.clone(), None));
                        continue;
                    }
                }
            }

            match event {
                Event::Bar(bar) => {
                    if order.symbol != bar.symbol {
                        continue;
                    }

                    // 1. 检查是否触发止损/止盈
                    if let Some(trigger_price) = order.trigger_price {
                        let triggered = match order.side {
                            OrderSide::Buy => bar.high >= trigger_price, // 价格突破触发
                            OrderSide::Sell => bar.low <= trigger_price, // 价格跌破触发
                        };

                        if !triggered {
                            continue; // 未触发，跳过
                        }

                        // 触发后，清除 trigger_price，并根据类型转换为市价或限价单
                        order.trigger_price = None;
                        match order.order_type {
                            OrderType::StopMarket => order.order_type = OrderType::Market,
                            OrderType::StopLimit => order.order_type = OrderType::Limit,
                            _ => {} // Should not happen for Limit/Market unless they have trigger_price (Conditional)
                        }
                    }

                    // 2. 撮合逻辑
                    let mut execute_price: Option<Decimal> = None;

                    match order.order_type {
                        OrderType::Market | OrderType::StopMarket => {
                            // 市价单 / 触发后的止损市价单
                            execute_price = match execution_mode {
                                crate::model::ExecutionMode::NextOpen => Some(bar.open),
                                crate::model::ExecutionMode::CurrentClose => Some(bar.close),
                                crate::model::ExecutionMode::NextAverage => Some(
                                    (bar.open + bar.high + bar.low + bar.close) / Decimal::from(4),
                                ),
                                crate::model::ExecutionMode::NextHighLowMid => {
                                    Some((bar.high + bar.low) / Decimal::from(2))
                                }
                            };
                        }
                        OrderType::Limit | OrderType::StopLimit => {
                            // 限价单 / 触发后的止损限价单
                            if let Some(limit_price) = order.price {
                                let avg_price =
                                    (bar.open + bar.high + bar.low + bar.close) / Decimal::from(4);
                                let mid_price = (bar.high + bar.low) / Decimal::from(2);
                                match order.side {
                                    OrderSide::Buy => {
                                        // 买单：最低价 <= 限价
                                        if bar.low <= limit_price {
                                            match execution_mode {
                                                crate::model::ExecutionMode::NextAverage => {
                                                    // 在均价模式下，如果能成交，尝试以均价成交，但不能超过限价
                                                    execute_price = Some(limit_price.min(avg_price));
                                                }
                                                crate::model::ExecutionMode::NextHighLowMid => {
                                                    // 在中间价模式下，如果能成交，尝试以中间价成交，但不能超过限价
                                                    execute_price = Some(limit_price.min(mid_price));
                                                }
                                                _ => {
                                                    // 默认逻辑 (gap handling)
                                                    execute_price = Some(limit_price.min(bar.open));
                                                    if execute_price.unwrap() > limit_price {
                                                        execute_price = Some(limit_price);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    OrderSide::Sell => {
                                        // 卖单：最高价 >= 限价
                                        if bar.high >= limit_price {
                                            match execution_mode {
                                                crate::model::ExecutionMode::NextAverage => {
                                                    // 在均价模式下，如果能成交，尝试以均价成交，但不能低于限价
                                                    execute_price = Some(limit_price.max(avg_price));
                                                }
                                                crate::model::ExecutionMode::NextHighLowMid => {
                                                    // 在中间价模式下，如果能成交，尝试以中间价成交，但不能低于限价
                                                    execute_price = Some(limit_price.max(mid_price));
                                                }
                                                _ => {
                                                    // 默认逻辑 (gap handling)
                                                    execute_price = Some(limit_price.max(bar.open));
                                                    if execute_price.unwrap() < limit_price {
                                                        execute_price = Some(limit_price);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if let Some(price) = execute_price {
                        // Apply Slippage
                        let final_price =
                            self.slippage_model
                                .calculate_price(price, order.quantity, order.side);

                        // Check Volume Limit
                        let max_qty = if self.volume_limit_pct > Decimal::ZERO {
                            bar.volume * self.volume_limit_pct
                        } else {
                            Decimal::MAX
                        };

                        let trade_qty = (order.quantity - order.filled_quantity).min(max_qty);

                        if trade_qty > Decimal::ZERO {
                            order.status = OrderStatus::Filled;
                            order.updated_at = bar.timestamp;
                            // Check if partial fill
                            if trade_qty < order.quantity - order.filled_quantity {
                                order.status = OrderStatus::Submitted; // Remain submitted/partially filled
                            }

                            order.filled_quantity += trade_qty;

                            // Update weighted average price
                            let current_filled = order.filled_quantity; // Already updated
                            let prev_filled = current_filled - trade_qty;
                            let prev_avg = order.average_filled_price.unwrap_or(Decimal::ZERO);
                            let new_avg =
                                (prev_avg * prev_filled + final_price * trade_qty) / current_filled;
                            order.average_filled_price = Some(new_avg);

                            let trade = Trade {
                                id: Uuid::new_v4().to_string(),
                                order_id: order.id.clone(),
                                symbol: order.symbol.clone(),
                                side: order.side,
                                quantity: trade_qty,
                                price: final_price,
                                commission: Decimal::ZERO, // 佣金稍后计算
                                timestamp: bar.timestamp,
                                bar_index,
                            };
                            reports.push(Event::ExecutionReport(order.clone(), Some(trade)));
                        }
                    } else if order.time_in_force == TimeInForce::IOC
                        || order.time_in_force == TimeInForce::FOK
                    {
                        // IOC/FOK 未能立即成交则取消
                        order.status = OrderStatus::Cancelled;
                        order.updated_at = bar.timestamp;
                        reports.push(Event::ExecutionReport(order.clone(), None));
                    } else if order.time_in_force == TimeInForce::Day && session == TradingSession::Closed {
                         // Day order expired
                         order.status = OrderStatus::Cancelled;
                         order.updated_at = bar.timestamp;
                         reports.push(Event::ExecutionReport(order.clone(), None));
                    }
                }
                Event::Tick(tick) => {
                    if order.symbol != tick.symbol {
                        continue;
                    }

                    // 1. 检查是否触发止损/止盈
                    if let Some(trigger_price) = order.trigger_price {
                        let triggered = match order.side {
                            OrderSide::Buy => tick.price >= trigger_price,
                            OrderSide::Sell => tick.price <= trigger_price,
                        };
                        if !triggered {
                            continue;
                        }

                        // 触发后，清除 trigger_price
                        order.trigger_price = None;
                        match order.order_type {
                            OrderType::StopMarket => order.order_type = OrderType::Market,
                            OrderType::StopLimit => order.order_type = OrderType::Limit,
                            _ => {}
                        }
                    }

                    let mut execute_price: Option<Decimal> = None;
                    match order.order_type {
                        OrderType::Market | OrderType::StopMarket => {
                            execute_price = Some(tick.price);
                        }
                        OrderType::Limit | OrderType::StopLimit => {
                            if let Some(limit_price) = order.price {
                                match order.side {
                                    OrderSide::Buy => {
                                        if tick.price <= limit_price {
                                            execute_price = Some(limit_price);
                                        }
                                    }
                                    OrderSide::Sell => {
                                        if tick.price >= limit_price {
                                            execute_price = Some(limit_price);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if let Some(price) = execute_price {
                        // Apply Slippage
                        let final_price =
                            self.slippage_model
                                .calculate_price(price, order.quantity, order.side);

                        // Check Volume Limit
                        let max_qty = if self.volume_limit_pct > Decimal::ZERO {
                            tick.volume * self.volume_limit_pct
                        } else {
                            Decimal::MAX
                        };

                        let trade_qty = (order.quantity - order.filled_quantity).min(max_qty);

                        if trade_qty > Decimal::ZERO {
                            order.status = OrderStatus::Filled;
                            order.updated_at = tick.timestamp;
                            order.filled_quantity += trade_qty;

                            // Check partial fill
                            if order.filled_quantity < order.quantity {
                                order.status = OrderStatus::Submitted;
                            }

                            // Update average price
                            let current_filled = order.filled_quantity;
                            let prev_filled = current_filled - trade_qty;
                            let prev_avg = order.average_filled_price.unwrap_or(Decimal::ZERO);
                            let new_avg =
                                (prev_avg * prev_filled + final_price * trade_qty) / current_filled;
                            order.average_filled_price = Some(new_avg);

                            let trade = Trade {
                                id: Uuid::new_v4().to_string(),
                                order_id: order.id.clone(),
                                symbol: order.symbol.clone(),
                                side: order.side,
                                quantity: trade_qty,
                                price: final_price,
                                commission: Decimal::ZERO,
                                timestamp: tick.timestamp,
                                bar_index,
                            };
                            reports.push(Event::ExecutionReport(order.clone(), Some(trade)));
                        }
                    } else if order.time_in_force == TimeInForce::IOC
                        || order.time_in_force == TimeInForce::FOK
                    {
                        order.status = OrderStatus::Cancelled;
                        order.updated_at = tick.timestamp;
                        reports.push(Event::ExecutionReport(order.clone(), None));
                    }
                }
                _ => {}
            }
        }

        // Cleanup filled/cancelled/rejected orders from pending list
        self.pending_orders.retain(|o| {
            o.status != OrderStatus::Filled
                && o.status != OrderStatus::Cancelled
                && o.status != OrderStatus::Rejected
                && o.status != OrderStatus::Expired
        });

        reports
    }
}

/// 实盘执行器 (Realtime Execution Client)
/// 对接外部交易接口 (如 CTP/Broker API)
pub struct RealtimeExecutionClient;

impl RealtimeExecutionClient {
    pub fn new() -> Self {
        RealtimeExecutionClient
    }
}

impl ExecutionClient for RealtimeExecutionClient {
    fn is_live(&self) -> bool {
        true
    }

    fn on_order(&mut self, _order: Order) {
        // println!(
        //     "[Realtime] Sending Order to Broker: {} {:?} {}",
        //     order.symbol, order.side, order.quantity
        // );
        // In real impl, send to broker API
    }

    fn on_cancel(&mut self, _order_id: &str) {
        // println!("[Realtime] Cancelling Order: {}", order_id);
    }

    fn on_event(
        &mut self,
        _event: &Event,
        _instruments: &std::collections::HashMap<String, crate::model::Instrument>,
        _execution_mode: crate::model::ExecutionMode,
        _bar_index: usize,
        _session: TradingSession,
    ) -> Vec<Event> {
        // In realtime, this might check for interaction with broker
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AssetType, Bar, Instrument, TimeInForce, ExecutionMode};
    use std::collections::HashMap;

    fn create_test_instruments() -> HashMap<String, Instrument> {
        let mut map = HashMap::new();
        let aapl = Instrument {
            symbol: "AAPL".to_string(),
            asset_type: AssetType::Stock,
            multiplier: Decimal::ONE,
            margin_ratio: Decimal::ONE,
            tick_size: Decimal::new(1, 2),
            option_type: None,
            strike_price: None,
            expiry_date: None,
            lot_size: Decimal::from(100),
            underlying_symbol: None,
            settlement_type: None,
        };
        map.insert("AAPL".to_string(), aapl);
        map
    }

    fn create_test_order(
        symbol: &str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Option<Decimal>,
    ) -> Order {
        Order {
            id: Uuid::new_v4().to_string(),
            symbol: symbol.to_string(),
            side,
            order_type,
            quantity,
            price,
            time_in_force: TimeInForce::Day,
            trigger_price: None,
            status: OrderStatus::New,
            filled_quantity: Decimal::ZERO,
            average_filled_price: None,
            created_at: 0,
            updated_at: 0,
            commission: Decimal::ZERO,
            tag: String::new(),
            reject_reason: String::new(),
        }
    }

    fn create_test_bar(
        symbol: &str,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
    ) -> Bar {
        Bar {
            symbol: symbol.to_string(),
            timestamp: 1000,
            open,
            high,
            low,
            close,
            volume: Decimal::from(1000),
            extra: HashMap::new(),
        }
    }

    #[test]
    fn test_execution_market_order() {
        let mut sim = SimulatedExecutionClient::new();
        let instruments = create_test_instruments();
        let order = create_test_order(
            "AAPL",
            OrderSide::Buy,
            OrderType::Market,
            Decimal::from(100),
            None,
        );
        // Setup sim
        sim.on_order(order);

        let bar = create_test_bar(
            "AAPL",
            Decimal::from(100),
            Decimal::from(105),
            Decimal::from(95),
            Decimal::from(102),
        );
        let event = Event::Bar(bar);

        // Match at Open
        let events = sim.on_event(&event, &instruments, ExecutionMode::NextOpen, 0, TradingSession::Continuous);

        let trades: Vec<Trade> = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(_, Some(trade)) = e {
                    Some(trade.clone())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, Decimal::from(100)); // Open price

        // Check order status from the last report
        let last_order = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(o, _) = e {
                    Some(o)
                } else {
                    None
                }
            })
            .last()
            .unwrap();
        assert_eq!(last_order.status, OrderStatus::Filled);

        // Reset and Match at Close
        let order2 = create_test_order(
            "AAPL",
            OrderSide::Buy,
            OrderType::Market,
            Decimal::from(100),
            None,
        );
        sim.on_order(order2);

        let events2 = sim.on_event(&event, &instruments, ExecutionMode::CurrentClose, 0, TradingSession::Continuous);
        let trades2: Vec<Trade> = events2
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(_, Some(trade)) = e {
                    Some(trade.clone())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(trades2.len(), 1);
        assert_eq!(trades2[0].price, Decimal::from(102)); // Close price
    }

    #[test]
    fn test_execution_limit_buy() {
        let mut sim = SimulatedExecutionClient::new();
        let instruments = create_test_instruments();
        // Limit Buy @ 98. Low is 95. Should fill.
        let order = create_test_order(
            "AAPL",
            OrderSide::Buy,
            OrderType::Limit,
            Decimal::from(100),
            Some(Decimal::from(98)),
        );
        sim.on_order(order);

        let bar = create_test_bar(
            "AAPL",
            Decimal::from(100),
            Decimal::from(105),
            Decimal::from(95),
            Decimal::from(102),
        );
        let event = Event::Bar(bar);

        let events = sim.on_event(&event, &instruments, ExecutionMode::NextOpen, 0, TradingSession::Continuous);
        let trades: Vec<Trade> = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(_, Some(trade)) = e {
                    Some(trade.clone())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(trades.len(), 1);
        // Fill logic in code: min(limit, open) if Low <= limit.
        // Here limit=98, open=100. min(98, 100) = 98.
        assert_eq!(trades[0].price, Decimal::from(98));
    }

    #[test]
    fn test_execution_limit_buy_no_fill() {
        let mut sim = SimulatedExecutionClient::new();
        let instruments = create_test_instruments();
        // Limit Buy @ 90. Low is 95. Should NOT fill.
        let order = create_test_order(
            "AAPL",
            OrderSide::Buy,
            OrderType::Limit,
            Decimal::from(100),
            Some(Decimal::from(90)),
        );
        sim.on_order(order);

        let bar = create_test_bar(
            "AAPL",
            Decimal::from(100),
            Decimal::from(105),
            Decimal::from(95),
            Decimal::from(102),
        );
        let event = Event::Bar(bar);

        let events = sim.on_event(&event, &instruments, ExecutionMode::NextOpen, 0, TradingSession::Continuous);
        let trades: Vec<Trade> = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(_, Some(trade)) = e {
                    Some(trade.clone())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(trades.len(), 0);
        assert_eq!(events.len(), 0);
    }

    #[test]
    fn test_execution_limit_sell() {
        let mut sim = SimulatedExecutionClient::new();
        let instruments = create_test_instruments();
        // Limit Sell @ 103. High is 105. Should fill.
        let order = create_test_order(
            "AAPL",
            OrderSide::Sell,
            OrderType::Limit,
            Decimal::from(100),
            Some(Decimal::from(103)),
        );
        sim.on_order(order);

        let bar = create_test_bar(
            "AAPL",
            Decimal::from(100),
            Decimal::from(105),
            Decimal::from(95),
            Decimal::from(102),
        );
        let event = Event::Bar(bar);

        let events = sim.on_event(&event, &instruments, ExecutionMode::NextOpen, 0, TradingSession::Continuous);
        let trades: Vec<Trade> = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(_, Some(trade)) = e {
                    Some(trade.clone())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(trades.len(), 1);
        // Fill logic: max(limit, open) if High >= limit.
        // limit=103, open=100. max=103.
        assert_eq!(trades[0].price, Decimal::from(103));
    }

    #[test]
    fn test_lot_size_check() {
        let mut sim = SimulatedExecutionClient::new();
        let instruments = create_test_instruments();

        // Odd lot buy (50 shares), should be rejected (lot size 100)
        let order = create_test_order(
            "AAPL",
            OrderSide::Buy,
            OrderType::Market,
            Decimal::from(50),
            None,
        );
        sim.on_order(order);

        let bar = create_test_bar(
            "AAPL",
            Decimal::from(100),
            Decimal::from(105),
            Decimal::from(95),
            Decimal::from(102),
        );
        let event = Event::Bar(bar);

        let events = sim.on_event(&event, &instruments, ExecutionMode::NextOpen, 0, TradingSession::Continuous);

        let last_order = events
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(o, _) = e {
                    Some(o)
                } else {
                    None
                }
            })
            .last()
            .unwrap();
        assert_eq!(last_order.status, OrderStatus::Rejected);

        // Valid lot buy (200 shares), should be accepted
        let order2 = create_test_order(
            "AAPL",
            OrderSide::Buy,
            OrderType::Market,
            Decimal::from(200),
            None,
        );
        sim.on_order(order2);

        let events2 = sim.on_event(&event, &instruments, ExecutionMode::NextOpen, 0, TradingSession::Continuous);
        let trades2: Vec<Trade> = events2
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(_, Some(trade)) = e {
                    Some(trade.clone())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(trades2.len(), 1);
        let last_order2 = events2
            .iter()
            .filter_map(|e| {
                if let Event::ExecutionReport(o, _) = e {
                    Some(o)
                } else {
                    None
                }
            })
            .last()
            .unwrap();
        assert_eq!(last_order2.status, OrderStatus::Filled);
    }
}
