use crate::event::Event;
use crate::execution::matcher::MatchContext;
use crate::log_context::{execution_order_context_from_event, render_log_message};
use crate::model::{
    Order, OrderSide, OrderStatus, OrderType, PriceBasis, TimeInForce, Timer, Trade,
};
use rust_decimal::Decimal;
use uuid::Uuid;

/// 通用撮合逻辑
pub struct CommonMatcher;

impl CommonMatcher {
    fn effective_timer_execution_timestamp(timer: &Timer) -> i64 {
        timer
            .payload
            .strip_prefix("__framework_cross_section__|")
            .and_then(|payload| payload.rsplit('|').next())
            .and_then(|value| value.parse::<i64>().ok())
            .unwrap_or(timer.timestamp)
    }

    fn cancelled_unfilled_ioc_fok_warning(order: &Order, event: &Event) -> String {
        render_log_message(
            format!(
                "Cancelled {:?} order because it was not filled on current event",
                order.time_in_force
            ),
            execution_order_context_from_event(order, event),
        )
    }

    fn deferred_triggered_stop_limit_warning(
        order: &Order,
        event: &Event,
        trigger_price: Decimal,
        limit_price: Decimal,
    ) -> String {
        render_log_message(
            format!(
                "Deferred triggered stop-limit order because trigger price {trigger_price} breached limit price {limit_price} during in-bar activation"
            ),
            execution_order_context_from_event(order, event),
        )
    }

    fn apply_slippage(order: &Order, base_price: Decimal, ctx: &MatchContext) -> Decimal {
        let override_type = order
            .slippage_type_override
            .as_ref()
            .map(|v| v.trim().to_ascii_lowercase());
        let override_value = order.slippage_value_override;
        match (override_type.as_deref(), override_value) {
            (Some("fixed"), Some(delta)) => match order.side {
                OrderSide::Buy => base_price + delta,
                OrderSide::Sell => base_price - delta,
            },
            (Some("percent"), Some(rate)) => match order.side {
                OrderSide::Buy => base_price * (Decimal::ONE + rate),
                OrderSide::Sell => base_price * (Decimal::ONE - rate),
            },
            _ => ctx
                .slippage
                .calculate_price(base_price, order.quantity, order.side),
        }
    }

    fn update_trailing_trigger_with_bar(order: &mut Order, high: Decimal, low: Decimal) {
        let Some(offset) = order.trail_offset else {
            return;
        };
        if offset <= Decimal::ZERO {
            return;
        }
        match order.side {
            OrderSide::Sell => {
                let prev = order.trail_reference_price.unwrap_or(high);
                let next = prev.max(high);
                order.trail_reference_price = Some(next);
                order.trigger_price = Some(next - offset);
            }
            OrderSide::Buy => {
                let prev = order.trail_reference_price.unwrap_or(low);
                let next = prev.min(low);
                order.trail_reference_price = Some(next);
                order.trigger_price = Some(next + offset);
            }
        }
    }

    fn update_trailing_trigger_with_tick(order: &mut Order, price: Decimal) {
        let Some(offset) = order.trail_offset else {
            return;
        };
        if offset <= Decimal::ZERO {
            return;
        }
        match order.side {
            OrderSide::Sell => {
                let prev = order.trail_reference_price.unwrap_or(price);
                let next = prev.max(price);
                order.trail_reference_price = Some(next);
                order.trigger_price = Some(next - offset);
            }
            OrderSide::Buy => {
                let prev = order.trail_reference_price.unwrap_or(price);
                let next = prev.min(price);
                order.trail_reference_price = Some(next);
                order.trigger_price = Some(next + offset);
            }
        }
    }

    fn promote_triggered_order_type(order: &mut Order) {
        match order.order_type {
            OrderType::StopMarket | OrderType::StopTrail => order.order_type = OrderType::Market,
            OrderType::StopLimit | OrderType::StopTrailLimit => order.order_type = OrderType::Limit,
            _ => {}
        }
    }

    /// 核心撮合逻辑 (支持穿透检查、Bar内止损、价格改善)
    ///
    /// :param order: 订单
    /// :param ctx: 撮合上下文
    /// :param check_lot_size: 是否检查最小交易单位 (针对股票买入等场景)
    pub fn match_order(
        order: &mut Order,
        ctx: &MatchContext,
        check_lot_size: bool,
    ) -> Option<Event> {
        let event = ctx.event;
        let instrument = ctx.instrument;
        let execution_policy = ctx.execution_policy_core;
        let volume_limit_pct = ctx.volume_limit_pct;
        let bar_index = ctx.bar_index;

        // 0. 检查最小交易单位 (Lot Size)
        // 仅针对买入订单，且必须存在标的定义
        if check_lot_size
            && order.side == OrderSide::Buy
            && order.quantity % instrument.lot_size() != Decimal::ZERO
        {
            order.status = OrderStatus::Rejected;
            order.reject_reason = format!(
                "Quantity {} is not a multiple of lot size {}",
                order.quantity,
                instrument.lot_size()
            );
            match event {
                Event::Bar(b) => order.updated_at = b.timestamp,
                Event::Tick(t) => order.updated_at = t.timestamp,
                Event::Timer(t) => order.updated_at = Self::effective_timer_execution_timestamp(t),
                _ => {}
            }
            log::warn!(
                "{}",
                render_log_message(
                    format!("Rejected order because {}", order.reject_reason),
                    execution_order_context_from_event(order, event),
                )
            );
            return Some(Event::ExecutionReport(order.clone(), None));
        }

        match event {
            Event::Bar(bar) => {
                if order.symbol != bar.symbol {
                    return None;
                }

                if matches!(
                    order.order_type,
                    OrderType::StopTrail | OrderType::StopTrailLimit
                ) {
                    Self::update_trailing_trigger_with_bar(order, bar.high, bar.low);
                }

                // 1. Volume Check (Suspension)
                if bar.volume <= Decimal::ZERO {
                    return None;
                }

                // 2. Check Stop Trigger
                let mut is_triggered_now = false;
                let mut trigger_price_val = None;

                if let Some(trigger_price) = order.trigger_price {
                    // Check Gap (Open vs Trigger)
                    let gap_triggered = match order.side {
                        OrderSide::Buy => bar.open >= trigger_price,
                        OrderSide::Sell => bar.open <= trigger_price,
                    };

                    // Check In-Bar (High/Low vs Trigger)
                    let in_bar_triggered = if !gap_triggered {
                        match order.side {
                            OrderSide::Buy => bar.high >= trigger_price,
                            OrderSide::Sell => bar.low <= trigger_price,
                        }
                    } else {
                        false
                    };

                    if gap_triggered || in_bar_triggered {
                        is_triggered_now = true;
                        trigger_price_val = Some(trigger_price);
                        order.trigger_price = None; // Clear trigger

                        // Update Order Type
                        Self::promote_triggered_order_type(order);
                    } else {
                        return None; // Not triggered
                    }
                }

                // 3. Execution Logic
                let mut execute_price: Option<Decimal> = None;

                // Determine Market Base Price
                let market_price = match (execution_policy.price_basis, execution_policy.bar_offset)
                {
                    (PriceBasis::Open, 1) => bar.open,
                    (PriceBasis::Close, 0) => bar.close,
                    (PriceBasis::Close, 1) => bar.close,
                    (PriceBasis::Ohlc4, 1) => {
                        (bar.open + bar.high + bar.low + bar.close) / Decimal::from(4)
                    }
                    (PriceBasis::Hl2, 1) => (bar.high + bar.low) / Decimal::from(2),
                    (PriceBasis::Open, _) => bar.open,
                    (PriceBasis::Close, _) => bar.close,
                    (PriceBasis::Ohlc4, _) => {
                        (bar.open + bar.high + bar.low + bar.close) / Decimal::from(4)
                    }
                    (PriceBasis::Hl2, _) => (bar.high + bar.low) / Decimal::from(2),
                };

                match order.order_type {
                    OrderType::Market => {
                        // Special handling for Stop-Market triggered IN-BAR
                        if is_triggered_now {
                            if let Some(tp) = trigger_price_val {
                                // Re-check gap to decide price
                                let is_gap = match order.side {
                                    OrderSide::Buy => bar.open >= tp,
                                    OrderSide::Sell => bar.open <= tp,
                                };

                                if is_gap {
                                    execute_price = Some(bar.open);
                                } else {
                                    // In-bar trigger: execute at trigger price
                                    execute_price = Some(tp);
                                }
                            } else {
                                execute_price = Some(market_price);
                            }
                        } else {
                            // Standard Market Order
                            execute_price = Some(market_price);
                        }
                    }
                    OrderType::Limit => {
                        if let Some(limit_price) = order.price {
                            // 3.1 Check Executability (Low/High Penetration)
                            let can_execute = match order.side {
                                OrderSide::Buy => bar.low <= limit_price,
                                OrderSide::Sell => bar.high >= limit_price,
                            };

                            if can_execute {
                                // 3.2 Determine Fill Price
                                let mut final_fill_price = limit_price;

                                if execution_policy.price_basis == PriceBasis::Open {
                                    match order.side {
                                        OrderSide::Buy => {
                                            if bar.open < limit_price {
                                                final_fill_price = bar.open;
                                            }
                                        }
                                        OrderSide::Sell => {
                                            if bar.open > limit_price {
                                                final_fill_price = bar.open;
                                            }
                                        }
                                    }
                                }

                                // 3.3 Apply Stop-Limit Constraints (In-Bar Trigger)
                                if is_triggered_now && let Some(tp) = trigger_price_val {
                                    let is_gap = match order.side {
                                        OrderSide::Buy => bar.open >= tp,
                                        OrderSide::Sell => bar.open <= tp,
                                    };

                                    if !is_gap {
                                        // Triggered In-Bar.
                                        match order.side {
                                            OrderSide::Buy => {
                                                if final_fill_price < tp {
                                                    final_fill_price = tp;
                                                }
                                                if final_fill_price > limit_price {
                                                    log::warn!(
                                                        "{}",
                                                        Self::deferred_triggered_stop_limit_warning(
                                                            order,
                                                            event,
                                                            tp,
                                                            limit_price,
                                                        )
                                                    );
                                                    return None;
                                                }
                                            }
                                            OrderSide::Sell => {
                                                if final_fill_price > tp {
                                                    final_fill_price = tp;
                                                }
                                                if final_fill_price < limit_price {
                                                    log::warn!(
                                                        "{}",
                                                        Self::deferred_triggered_stop_limit_warning(
                                                            order,
                                                            event,
                                                            tp,
                                                            limit_price,
                                                        )
                                                    );
                                                    return None;
                                                }
                                            }
                                        }
                                    }
                                }
                                execute_price = Some(final_fill_price);
                            }
                        }
                    }
                    _ => {}
                }

                // 4. Create Trade if executed
                if let Some(price) = execute_price {
                    // Apply Slippage
                    let final_price = Self::apply_slippage(order, price, ctx);

                    // Check Volume Limit
                    let max_qty = if volume_limit_pct > Decimal::ZERO {
                        bar.volume * volume_limit_pct
                    } else {
                        Decimal::MAX
                    };

                    let trade_qty = (order.quantity - order.filled_quantity).min(max_qty);

                    if trade_qty > Decimal::ZERO {
                        order.status = OrderStatus::Filled;
                        order.updated_at = bar.timestamp;

                        // Check partial fill
                        if trade_qty < order.quantity - order.filled_quantity {
                            order.status = OrderStatus::PartiallyFilled;
                        }

                        order.filled_quantity += trade_qty;

                        // Update Average Price
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
                            position_effect: order.position_effect,
                            quantity: trade_qty,
                            price: final_price,
                            commission: Decimal::ZERO,
                            timestamp: bar.timestamp,
                            bar_index,
                            owner_strategy_id: order.owner_strategy_id.clone(),
                        };
                        return Some(Event::ExecutionReport(order.clone(), Some(trade)));
                    }
                } else if order.time_in_force == TimeInForce::IOC
                    || order.time_in_force == TimeInForce::FOK
                {
                    // Cancel IOC/FOK if not filled
                    order.status = OrderStatus::Cancelled;
                    order.updated_at = bar.timestamp;
                    log::warn!("{}", Self::cancelled_unfilled_ioc_fok_warning(order, event));
                    return Some(Event::ExecutionReport(order.clone(), None));
                }
            }
            Event::Tick(tick) => {
                if order.symbol != tick.symbol {
                    return None;
                }

                if matches!(
                    order.order_type,
                    OrderType::StopTrail | OrderType::StopTrailLimit
                ) {
                    Self::update_trailing_trigger_with_tick(order, tick.price);
                }

                // 1. Check Trigger
                if let Some(trigger_price) = order.trigger_price {
                    let triggered = match order.side {
                        OrderSide::Buy => tick.price >= trigger_price,
                        OrderSide::Sell => tick.price <= trigger_price,
                    };
                    if !triggered {
                        return None;
                    }
                    order.trigger_price = None;
                    Self::promote_triggered_order_type(order);
                }

                // 2. Execute
                let mut execute_price = None;
                match order.order_type {
                    OrderType::Market => execute_price = Some(tick.price),
                    OrderType::Limit => {
                        if let Some(limit) = order.price {
                            match order.side {
                                OrderSide::Buy => {
                                    if tick.price <= limit {
                                        execute_price = Some(limit)
                                    }
                                }
                                OrderSide::Sell => {
                                    if tick.price >= limit {
                                        execute_price = Some(limit)
                                    }
                                }
                            }
                            if execute_price.is_some() {
                                execute_price = Some(tick.price);
                            }
                        }
                    }
                    _ => {}
                }

                if let Some(price) = execute_price {
                    let final_price = Self::apply_slippage(order, price, ctx);

                    let trade_qty = order.quantity - order.filled_quantity;

                    if trade_qty > Decimal::ZERO {
                        order.status = OrderStatus::Filled;
                        order.updated_at = tick.timestamp;
                        order.filled_quantity += trade_qty;
                        order.average_filled_price = Some(final_price);
                        let trade = Trade {
                            id: Uuid::new_v4().to_string(),
                            order_id: order.id.clone(),
                            symbol: order.symbol.clone(),
                            side: order.side,
                            position_effect: order.position_effect,
                            quantity: trade_qty,
                            price: final_price,
                            commission: Decimal::ZERO,
                            timestamp: tick.timestamp,
                            bar_index,
                            owner_strategy_id: order.owner_strategy_id.clone(),
                        };
                        return Some(Event::ExecutionReport(order.clone(), Some(trade)));
                    }
                } else if order.time_in_force == TimeInForce::IOC
                    || order.time_in_force == TimeInForce::FOK
                {
                    order.status = OrderStatus::Cancelled;
                    order.updated_at = tick.timestamp;
                    log::warn!("{}", Self::cancelled_unfilled_ioc_fok_warning(order, event));
                    return Some(Event::ExecutionReport(order.clone(), None));
                }
            }
            Event::Timer(timer) => {
                let execution_timestamp = Self::effective_timer_execution_timestamp(timer);
                if !matches!(
                    (execution_policy.price_basis, execution_policy.bar_offset),
                    (PriceBasis::Close, 0)
                ) {
                    return None;
                }
                let Some(reference_price) = ctx.last_price else {
                    return None;
                };

                if matches!(
                    order.order_type,
                    OrderType::StopTrail | OrderType::StopTrailLimit
                ) {
                    Self::update_trailing_trigger_with_tick(order, reference_price);
                }

                if let Some(trigger_price) = order.trigger_price {
                    let triggered = match order.side {
                        OrderSide::Buy => reference_price >= trigger_price,
                        OrderSide::Sell => reference_price <= trigger_price,
                    };
                    if !triggered {
                        return None;
                    }
                    order.trigger_price = None;
                    Self::promote_triggered_order_type(order);
                }

                let mut execute_price = None;
                match order.order_type {
                    OrderType::Market => execute_price = Some(reference_price),
                    OrderType::Limit => {
                        if let Some(limit) = order.price {
                            let can_execute = match order.side {
                                OrderSide::Buy => reference_price <= limit,
                                OrderSide::Sell => reference_price >= limit,
                            };
                            if can_execute {
                                execute_price = Some(reference_price);
                            }
                        }
                    }
                    _ => {}
                }

                if let Some(price) = execute_price {
                    let final_price = Self::apply_slippage(order, price, ctx);
                    let trade_qty = order.quantity - order.filled_quantity;
                    if trade_qty > Decimal::ZERO {
                        order.status = OrderStatus::Filled;
                        order.updated_at = execution_timestamp;
                        order.filled_quantity += trade_qty;
                        order.average_filled_price = Some(final_price);
                        let trade = Trade {
                            id: Uuid::new_v4().to_string(),
                            order_id: order.id.clone(),
                            symbol: order.symbol.clone(),
                            side: order.side,
                            position_effect: order.position_effect,
                            quantity: trade_qty,
                            price: final_price,
                            commission: Decimal::ZERO,
                            timestamp: execution_timestamp,
                            bar_index,
                            owner_strategy_id: order.owner_strategy_id.clone(),
                        };
                        return Some(Event::ExecutionReport(order.clone(), Some(trade)));
                    }
                } else if order.time_in_force == TimeInForce::IOC
                    || order.time_in_force == TimeInForce::FOK
                {
                    order.status = OrderStatus::Cancelled;
                    order.updated_at = execution_timestamp;
                    log::warn!("{}", Self::cancelled_unfilled_ioc_fok_warning(order, event));
                    return Some(Event::ExecutionReport(order.clone(), None));
                }
            }
            _ => {}
        }
        None
    }
}
