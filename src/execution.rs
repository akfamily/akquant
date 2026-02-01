use crate::data::Event;
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

/// 模拟交易所执行器
/// 负责处理订单撮合逻辑
pub struct ExchangeSimulator {
    slippage_model: Box<dyn SlippageModel>,
    volume_limit_pct: Decimal, // 成交量限制比例 (0.0 = 不限制)
}

impl ExchangeSimulator {
    pub fn new() -> Self {
        ExchangeSimulator {
            slippage_model: Box::new(ZeroSlippage),
            volume_limit_pct: Decimal::ZERO,
        }
    }

    pub fn set_slippage_model(&mut self, model: Box<dyn SlippageModel>) {
        self.slippage_model = model;
    }

    pub fn set_volume_limit(&mut self, limit: f64) {
        self.volume_limit_pct = Decimal::from_f64(limit).unwrap_or(Decimal::ZERO);
    }

    /// 处理事件并撮合订单
    /// 返回生成的成交记录
    pub fn process_event(
        &mut self,
        event: &Event,
        orders: &mut [Order],
        instruments: &std::collections::HashMap<String, crate::model::Instrument>,
        match_at_open: bool,
        bar_index: usize,
        session: TradingSession,
    ) -> Vec<Trade> {
        let mut trades = Vec::new();

        // Skip matching during non-trading sessions
        if session == TradingSession::Break
            || session == TradingSession::Closed
            || session == TradingSession::PreOpen
            || session == TradingSession::PostClose
        {
            return trades;
        }

        // 筛选出未完成的订单
        // 这里需要注意：我们不能在遍历时修改 orders，所以需要两步走或者使用索引
        // 为了简单，我们收集需要更新的索引

        // 实际撮合逻辑：遍历所有挂单，看当前 Event 是否满足成交条件
        for order in orders.iter_mut() {
            if order.status != OrderStatus::New && order.status != OrderStatus::Submitted {
                continue;
            }
            // 0. 检查最小交易单位 (Lot Size)
            // 仅针对买入订单，且必须存在标的定义
            if order.side == OrderSide::Buy {
                if let Some(instrument) = instruments.get(&order.symbol) {
                    if order.quantity % instrument.lot_size != Decimal::ZERO {
                        order.status = OrderStatus::Rejected;
                        // 可以添加日志说明拒绝原因
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
                            if match_at_open {
                                execute_price = Some(bar.open);
                            } else {
                                execute_price = Some(bar.close);
                            }
                        }
                        OrderType::Limit | OrderType::StopLimit => {
                            // 限价单 / 触发后的止损限价单
                            if let Some(limit_price) = order.price {
                                match order.side {
                                    OrderSide::Buy => {
                                        // 买单：最低价 <= 限价
                                        if bar.low <= limit_price {
                                            execute_price = Some(limit_price.min(bar.open)); // 简化撮合价
                                            if execute_price.unwrap() > limit_price {
                                                execute_price = Some(limit_price); // 修正，买入不能高于限价
                                            }
                                        }
                                    }
                                    OrderSide::Sell => {
                                        // 卖单：最高价 >= 限价
                                        if bar.high >= limit_price {
                                            execute_price = Some(limit_price.max(bar.open)); // 简化撮合价
                                            if execute_price.unwrap() < limit_price {
                                                execute_price = Some(limit_price); // 修正，卖出不能低于限价
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if let Some(price) = execute_price {
                        // Apply Slippage
                        let final_price = self.slippage_model.calculate_price(price, order.quantity, order.side);

                        // Check Volume Limit
                        let max_qty = if self.volume_limit_pct > Decimal::ZERO {
                            bar.volume * self.volume_limit_pct
                        } else {
                            Decimal::MAX
                        };

                        let trade_qty = (order.quantity - order.filled_quantity).min(max_qty);

                        if trade_qty > Decimal::ZERO {
                            order.status = OrderStatus::Filled;
                            // Check if partial fill
                            if trade_qty < order.quantity - order.filled_quantity {
                                order.status = OrderStatus::Submitted; // Remain submitted/partially filled
                            }

                            order.filled_quantity += trade_qty;

                            // Update weighted average price
                            let current_filled = order.filled_quantity; // Already updated
                            let prev_filled = current_filled - trade_qty;
                            let prev_avg = order.average_filled_price.unwrap_or(Decimal::ZERO);
                            let new_avg = (prev_avg * prev_filled + final_price * trade_qty) / current_filled;
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
                            trades.push(trade);
                        }
                    } else if order.time_in_force == TimeInForce::IOC
                        || order.time_in_force == TimeInForce::FOK
                    {
                        // IOC/FOK 未能立即成交则取消
                        order.status = OrderStatus::Cancelled;
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
                        let final_price = self.slippage_model.calculate_price(price, order.quantity, order.side);

                        // Check Volume Limit
                        let max_qty = if self.volume_limit_pct > Decimal::ZERO {
                            tick.volume * self.volume_limit_pct
                        } else {
                            Decimal::MAX
                        };

                        let trade_qty = (order.quantity - order.filled_quantity).min(max_qty);

                        if trade_qty > Decimal::ZERO {
                            order.status = OrderStatus::Filled;
                            order.filled_quantity += trade_qty;

                            // Check partial fill
                            if order.filled_quantity < order.quantity {
                                order.status = OrderStatus::Submitted;
                            }

                            // Update average price
                            let current_filled = order.filled_quantity;
                            let prev_filled = current_filled - trade_qty;
                            let prev_avg = order.average_filled_price.unwrap_or(Decimal::ZERO);
                            let new_avg = (prev_avg * prev_filled + final_price * trade_qty) / current_filled;
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
                            trades.push(trade);
                        }
                    } else if order.time_in_force == TimeInForce::IOC
                        || order.time_in_force == TimeInForce::FOK
                    {
                        order.status = OrderStatus::Cancelled;
                    }
                }
            }
        }

        trades
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Bar, TimeInForce, Instrument, AssetType};
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
        }
    }

    fn create_test_bar(symbol: &str, open: Decimal, high: Decimal, low: Decimal, close: Decimal) -> Bar {
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
        let mut sim = ExchangeSimulator::new();
        let instruments = create_test_instruments();
        let order = create_test_order("AAPL", OrderSide::Buy, OrderType::Market, Decimal::from(100), None);
        let mut orders = vec![order.clone()];

        let bar = create_test_bar("AAPL", Decimal::from(100), Decimal::from(105), Decimal::from(95), Decimal::from(102));
        let event = Event::Bar(bar);

        // Match at Open
        let trades = sim.process_event(&event, &mut orders, &instruments, true, 0, TradingSession::Continuous);

        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, Decimal::from(100)); // Open price
        assert_eq!(orders[0].status, OrderStatus::Filled);

        // Reset and Match at Close
        let order2 = create_test_order("AAPL", OrderSide::Buy, OrderType::Market, Decimal::from(100), None);
        let mut orders2 = vec![order2.clone()];
        let trades2 = sim.process_event(&event, &mut orders2, &instruments, false, 0, TradingSession::Continuous);

        assert_eq!(trades2.len(), 1);
        assert_eq!(trades2[0].price, Decimal::from(102)); // Close price
    }

    #[test]
    fn test_execution_limit_buy() {
        let mut sim = ExchangeSimulator::new();
        let instruments = create_test_instruments();
        // Limit Buy @ 98. Low is 95. Should fill.
        let order = create_test_order("AAPL", OrderSide::Buy, OrderType::Limit, Decimal::from(100), Some(Decimal::from(98)));
        let mut orders = vec![order];

        let bar = create_test_bar("AAPL", Decimal::from(100), Decimal::from(105), Decimal::from(95), Decimal::from(102));
        let event = Event::Bar(bar);

        let trades = sim.process_event(&event, &mut orders, &instruments, true, 0, TradingSession::Continuous);

        assert_eq!(trades.len(), 1);
        // Fill logic in code: min(limit, open) if Low <= limit.
        // Here limit=98, open=100. min(98, 100) = 98.
        assert_eq!(trades[0].price, Decimal::from(98));
    }

    #[test]
    fn test_execution_limit_buy_no_fill() {
        let mut sim = ExchangeSimulator::new();
        let instruments = create_test_instruments();
        // Limit Buy @ 90. Low is 95. Should NOT fill.
        let order = create_test_order("AAPL", OrderSide::Buy, OrderType::Limit, Decimal::from(100), Some(Decimal::from(90)));
        let mut orders = vec![order];

        let bar = create_test_bar("AAPL", Decimal::from(100), Decimal::from(105), Decimal::from(95), Decimal::from(102));
        let event = Event::Bar(bar);

        let trades = sim.process_event(&event, &mut orders, &instruments, true, 0, TradingSession::Continuous);

        assert_eq!(trades.len(), 0);
        assert_eq!(orders[0].status, OrderStatus::New);
    }

    #[test]
    fn test_execution_limit_sell() {
        let mut sim = ExchangeSimulator::new();
        let instruments = create_test_instruments();
        // Limit Sell @ 103. High is 105. Should fill.
        let order = create_test_order("AAPL", OrderSide::Sell, OrderType::Limit, Decimal::from(100), Some(Decimal::from(103)));
        let mut orders = vec![order];

        let bar = create_test_bar("AAPL", Decimal::from(100), Decimal::from(105), Decimal::from(95), Decimal::from(102));
        let event = Event::Bar(bar);

        let trades = sim.process_event(&event, &mut orders, &instruments, true, 0, TradingSession::Continuous);

        assert_eq!(trades.len(), 1);
        // Fill logic: max(limit, open) if High >= limit.
        // limit=103, open=100. max=103.
        assert_eq!(trades[0].price, Decimal::from(103));
    }

    #[test]
    fn test_lot_size_check() {
        let mut sim = ExchangeSimulator::new();
        let instruments = create_test_instruments();

        // Odd lot buy (50 shares), should be rejected (lot size 100)
        let order = create_test_order("AAPL", OrderSide::Buy, OrderType::Market, Decimal::from(50), None);
        let mut orders = vec![order];

        let bar = create_test_bar("AAPL", Decimal::from(100), Decimal::from(105), Decimal::from(95), Decimal::from(102));
        let event = Event::Bar(bar);

        let trades = sim.process_event(&event, &mut orders, &instruments, true, 0, TradingSession::Continuous);

        assert_eq!(trades.len(), 0);
        assert_eq!(orders[0].status, OrderStatus::Rejected);

        // Valid lot buy (200 shares), should be accepted
        let order2 = create_test_order("AAPL", OrderSide::Buy, OrderType::Market, Decimal::from(200), None);
        let mut orders2 = vec![order2];
        let trades2 = sim.process_event(&event, &mut orders2, &instruments, true, 0, TradingSession::Continuous);

        assert_eq!(trades2.len(), 1);
        assert_eq!(orders2[0].status, OrderStatus::Filled);
    }
}
