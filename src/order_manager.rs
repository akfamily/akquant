use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rust_decimal::prelude::*;
use rust_decimal::Decimal;

use crate::analysis::TradeTracker;
use crate::history::HistoryBuffer;
use crate::market::MarketModel;
use crate::model::{Instrument, Order, OrderStatus, Trade};
use crate::portfolio::Portfolio;
use crate::risk::RiskManager;

/// 订单管理器
/// 负责管理订单列表、成交记录及状态流转
pub struct OrderManager {
    /// 历史订单 (已完成)
    pub orders: Vec<Order>,
    /// 当前活跃订单 (未完成)
    pub active_orders: Vec<Order>,
    /// 历史成交记录
    pub trades: Vec<Trade>,
    /// 当前步生成的成交 (用于通知策略)
    pub current_step_trades: Vec<Trade>,
    /// 交易追踪器 (用于计算 PnL 和统计)
    pub trade_tracker: TradeTracker,
}

impl OrderManager {
    pub fn new() -> Self {
        OrderManager {
            orders: Vec::new(),
            active_orders: Vec::new(),
            trades: Vec::new(),
            current_step_trades: Vec::new(),
            trade_tracker: TradeTracker::new(),
        }
    }

    /// 添加新订单 (例如从 OrderValidated 事件)
    pub fn add_active_order(&mut self, order: Order) {
        self.active_orders.push(order);
    }

    /// 处理执行报告 (ExecutionReport)
    /// 更新活跃订单状态，并返回是否生成了成交
    pub fn on_execution_report(&mut self, report: Order) {
        if let Some(existing) = self.active_orders.iter_mut().find(|o| o.id == report.id) {
            existing.status = report.status;
            existing.filled_quantity = report.filled_quantity;
            existing.average_filled_price = report.average_filled_price;
            existing.updated_at = report.updated_at;
            existing.reject_reason = report.reject_reason;
        } else {
            // 如果是新的且状态为 Rejected (可能直接被拒)，加入活跃列表以便后续移入历史
            if report.status == OrderStatus::Rejected {
                self.active_orders.push(report);
            }
        }
    }

    /// 清理已完成的订单 (Filled, Cancelled, Expired, Rejected)
    /// 将其移入历史列表
    pub fn cleanup_finished_orders(&mut self) {
        let (finished, active): (Vec<Order>, Vec<Order>) =
            self.active_orders.drain(..).partition(|o| {
                o.status == OrderStatus::Filled
                    || o.status == OrderStatus::Cancelled
                    || o.status == OrderStatus::Expired
                    || o.status == OrderStatus::Rejected
            });

        self.orders.extend(finished);
        self.active_orders = active;
    }

    /// 获取所有订单 (历史 + 活跃)
    pub fn get_all_orders(&self) -> Vec<Order> {
        let mut all = self.orders.clone();
        all.extend(self.active_orders.clone());
        all
    }

    /// 处理成交列表
    /// 包括资金调整、持仓更新、PnL 计算等
    pub fn process_trades(
        &mut self,
        mut trades: Vec<Trade>,
        portfolio: &mut Portfolio,
        instruments: &HashMap<String, Instrument>,
        market_model: &dyn MarketModel,
        risk_manager: &RiskManager,
        history_buffer: &Arc<RwLock<HistoryBuffer>>,
        last_prices: &HashMap<String, Decimal>,
    ) {
        // 1. Adjust trades for insufficient cash (Dynamic Position Sizing)
        for trade in trades.iter_mut() {
            if trade.side == crate::model::OrderSide::Buy {
                // Calculate estimated commission for the full trade first
                let instr_opt = instruments.get(&trade.symbol);
                let mut commission = Decimal::ZERO;
                if let Some(instr) = instr_opt {
                    commission = market_model.calculate_commission(
                        instr,
                        trade.side,
                        trade.price,
                        trade.quantity,
                    );
                }

                let multiplier = instr_opt.map(|i| i.multiplier).unwrap_or(Decimal::ONE);
                let cost = trade.price * trade.quantity * multiplier;
                let total_required = cost + commission;

                if total_required > portfolio.cash {
                    // Insufficient cash, reduce quantity
                    let mut ratio = if total_required > Decimal::ZERO {
                        portfolio.cash / total_required
                    } else {
                        Decimal::ZERO
                    };

                    if ratio < Decimal::ZERO {
                        ratio = Decimal::ZERO;
                    }

                    // Apply ratio and round down to lot size
                    // Use configurable safety factor to avoid rounding issues causing rejection
                    let safety_margin = risk_manager.config.safety_margin;
                    let safety_factor = Decimal::from_f64(1.0 - safety_margin)
                        .unwrap_or(Decimal::from_f64(0.9999).unwrap());
                    ratio = ratio * safety_factor;

                    let lot_size = instr_opt.map(|i| i.lot_size).unwrap_or(Decimal::ONE);
                    let mut new_qty = (trade.quantity * ratio).floor();
                    if lot_size > Decimal::ZERO {
                        new_qty = new_qty - (new_qty % lot_size);
                    }

                    // Recalculate to ensure it fits (handling min commission etc)
                    if let Some(instr) = instr_opt {
                        let new_comm = market_model.calculate_commission(
                            instr,
                            trade.side,
                            trade.price,
                            new_qty,
                        );
                        let new_cost = trade.price * new_qty * multiplier;
                        if new_cost + new_comm > portfolio.cash {
                            // Still too high, reduce by one lot
                            if new_qty >= lot_size {
                                new_qty -= lot_size;
                            } else {
                                new_qty = Decimal::ZERO;
                            }
                        }
                    }

                    // Update trade quantity
                    trade.quantity = new_qty;
                }
            }
        }

        // Filter out zero quantity trades
        trades.retain(|t| t.quantity > Decimal::ZERO);

        for mut trade in trades {
            // 2. Calculate Final Commission
            let instr_opt = instruments.get(&trade.symbol);
            if let Some(instr) = instr_opt {
                trade.commission = market_model.calculate_commission(
                    instr,
                    trade.side,
                    trade.price,
                    trade.quantity,
                );
            }

            // 3. Update Portfolio
            portfolio.adjust_cash(-trade.commission);

            let multiplier = instr_opt.map(|i| i.multiplier).unwrap_or(Decimal::ONE);
            let cost = trade.price * trade.quantity * multiplier;

            if trade.side == crate::model::OrderSide::Buy {
                portfolio.adjust_cash(-cost);
                portfolio.adjust_position(&trade.symbol, trade.quantity);
            } else {
                portfolio.adjust_cash(cost); // Sell adds cash
                portfolio.adjust_position(&trade.symbol, -trade.quantity);
            }

            // Update available positions (T+1/T+0 rules)
            if let Some(instr) = instr_opt {
                market_model.update_available_position(
                    &mut portfolio.available_positions,
                    instr,
                    trade.quantity,
                    trade.side,
                );
            }

            // 4. Update Order Filled Quantity & Avg Price
            if let Some(order) = self.active_orders.iter_mut().find(|o| o.id == trade.order_id) {
                let old_qty = order.filled_quantity;
                let old_avg = order.average_filled_price.unwrap_or(Decimal::ZERO);
                let old_total = old_qty * old_avg;

                let new_trade_val = trade.quantity * trade.price;
                let new_total = old_total + new_trade_val;
                let new_qty = old_qty + trade.quantity;

                if new_qty > Decimal::ZERO {
                    order.average_filled_price = Some(new_total / new_qty);
                }
                order.filled_quantity = new_qty;
                order.commission += trade.commission;

                // Check if fully filled
                // Note: We don't change status to Filled here immediately because
                // execution report might come later or we want to wait for it?
                // Actually Engine logic relied on ExecutionReport to set status to Filled.
                // But here we are processing trade first?
                // In Engine::run, ExecutionReport updates status, THEN process_trades is called.
                // So status might already be Filled if this is the last trade.
                // But if we generated trade internally (simulated), we might need to update status.
                // However, SimulatedExecutionClient sends ExecutionReport with status Filled.
                // So we should rely on that.
                // But we update filled_qty here just in case?
                // Actually, if we rely on ExecutionReport for status, we should be fine.
                // The order update logic in on_execution_report handles it.
                // We just update commission here maybe?
            }

            // 5. Track Trade (PnL)
            let order_tag = self
                .active_orders
                .iter()
                .find(|o| o.id == trade.order_id)
                .map(|o| o.tag.as_str());

            // Get history for MAE/MFE
            // Need to lock history buffer
            let history_guard = history_buffer.read().unwrap();
            let symbol_history = history_guard.get_history(&trade.symbol);

            // Calculate Portfolio Value for % metrics
            let portfolio_value = portfolio.calculate_equity(last_prices, instruments);

            self.trade_tracker.process_trade(
                &trade,
                order_tag,
                symbol_history,
                portfolio_value,
            );

            // 6. Record Trade
            self.trades.push(trade.clone());
            self.current_step_trades.push(trade);
        }
    }
}

impl Default for OrderManager {
    fn default() -> Self {
        Self::new()
    }
}
