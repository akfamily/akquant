use super::types::{ClosedTrade, TradePnL};
use crate::history::SymbolHistory;
use crate::model::{OrderSide, Trade};
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeEntry {
    pub quantity: Decimal,
    pub price: Decimal,
    pub multiplier: Decimal,
    pub commission: Decimal,
    pub bar_idx: usize,
    pub timestamp: i64,
    pub tag: String,
    pub entry_portfolio_value: Decimal,
}

/// 一笔平仓的汇总更新；两个方向的平仓分支共用（以前是两段相同的代码）。
/// largest_* 的比较规则与 calculate_pnl 的循环相同：初值 0，只有严格更大（更小）才替换。
///
/// 用宏而不是 `&mut self` 方法：调用点持有 `self.long_inventory`/`short_inventory` 的
/// 可变借用，方法会借走整个 `self`；宏直接访问字段，借用检查器能看出字段互不相交。
macro_rules! record_close {
    ($self:ident, $pnl:expr, $ret_pct:expr, $bars:expr) => {{
        let (pnl, ret_pct, bars) = ($pnl, $ret_pct, $bars);
        $self.sum_closed_pnl += pnl;
        if pnl > Decimal::ZERO {
            $self.won_count += 1;
            $self.won_pnl += pnl;
            $self.closed_trades_stats.push((pnl, ret_pct, bars, true));
            if pnl > $self.largest_win {
                $self.largest_win = pnl;
            }
            $self.current_wins += 1;
            $self.current_losses = 0;
            if $self.current_wins > $self.max_wins {
                $self.max_wins = $self.current_wins;
            }
        } else {
            $self.lost_count += 1;
            $self.lost_pnl += pnl;
            $self.closed_trades_stats.push((pnl, ret_pct, bars, false));
            if pnl < $self.largest_loss {
                $self.largest_loss = pnl;
            }
            $self.current_losses += 1;
            $self.current_wins = 0;
            if $self.current_losses > $self.max_losses {
                $self.max_losses = $self.current_losses;
            }
        }
    }};
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeTracker {
    // (qty, price, commission, bar_idx, timestamp, tag, entry_portfolio_value)
    pub long_inventory: HashMap<String, VecDeque<TradeEntry>>,
    pub short_inventory: HashMap<String, VecDeque<TradeEntry>>,
    pub closed_trades: Arc<Vec<ClosedTrade>>,
    pub closed_trades_stats: Vec<(Decimal, Decimal, Decimal, bool)>, // (pnl, return_pct, bars, is_win)

    // Aggregate stats
    pub total_pnl: Decimal,
    pub total_commission: Decimal,
    pub won_count: usize,
    pub lost_count: usize,
    pub won_pnl: Decimal,
    pub lost_pnl: Decimal,
    pub max_wins: usize,
    pub max_losses: usize,
    pub current_wins: usize,
    pub current_losses: usize,

    // 运行时统计用的增量汇总；放在末尾并带 default，旧快照仍可反序列化
    #[serde(default)]
    pub sum_closed_pnl: Decimal,
    #[serde(default)]
    pub largest_win: Decimal,
    #[serde(default)]
    pub largest_loss: Decimal,
}

impl TradeTracker {
    pub fn new() -> Self {
        Self {
            long_inventory: HashMap::new(),
            short_inventory: HashMap::new(),
            closed_trades: Arc::new(Vec::new()),
            closed_trades_stats: Vec::new(),
            total_pnl: Decimal::ZERO,
            total_commission: Decimal::ZERO,
            won_count: 0,
            lost_count: 0,
            won_pnl: Decimal::ZERO,
            lost_pnl: Decimal::ZERO,
            max_wins: 0,
            max_losses: 0,
            current_wins: 0,
            current_losses: 0,
            sum_closed_pnl: Decimal::ZERO,
            largest_win: Decimal::ZERO,
            largest_loss: Decimal::ZERO,
        }
    }

    /// 从 `closed_trades_stats` 重建 sum_closed_pnl / largest_win / largest_loss。
    /// 旧状态快照缺这三个字段（反序列化为 0），恢复后必须调用。
    pub fn rebuild_summary_fields(&mut self) {
        let mut sum = Decimal::ZERO;
        let mut largest_win = Decimal::ZERO;
        let mut largest_loss = Decimal::ZERO;
        for (pnl, _, _, is_win) in &self.closed_trades_stats {
            sum += *pnl;
            if *is_win {
                if *pnl > largest_win {
                    largest_win = *pnl;
                }
            } else if *pnl < largest_loss {
                largest_loss = *pnl;
            }
        }
        self.sum_closed_pnl = sum;
        self.largest_win = largest_win;
        self.largest_loss = largest_loss;
    }

    pub fn summary(&self) -> crate::analysis::TradeSummary {
        crate::analysis::TradeSummary {
            won_count: self.won_count,
            lost_count: self.lost_count,
            won_pnl: self.won_pnl,
            lost_pnl: self.lost_pnl,
            sum_closed_pnl: self.sum_closed_pnl,
            largest_win: self.largest_win,
            largest_loss: self.largest_loss,
            total_commission: self.total_commission,
            max_wins: self.max_wins,
            max_losses: self.max_losses,
        }
    }

    pub fn on_split(&mut self, symbol: &str, ratio: Decimal) {
        let safe_ratio = ratio.abs();
        if safe_ratio <= Decimal::ZERO {
            return;
        }

        if let Some(queue) = self.long_inventory.get_mut(symbol) {
            for entry in queue.iter_mut() {
                entry.quantity = entry
                    .quantity
                    .checked_mul(safe_ratio)
                    .unwrap_or(Decimal::MAX);
                entry.price = entry.price.checked_div(safe_ratio).unwrap_or(Decimal::ZERO);
            }
        }

        if let Some(queue) = self.short_inventory.get_mut(symbol) {
            for entry in queue.iter_mut() {
                entry.quantity = entry
                    .quantity
                    .checked_mul(safe_ratio)
                    .unwrap_or(Decimal::MAX);
                entry.price = entry.price.checked_div(safe_ratio).unwrap_or(Decimal::ZERO);
            }
        }
    }

    pub fn get_unrealized_pnl(&self, symbol: &str, current_price: Decimal) -> Decimal {
        let mut pnl = Decimal::ZERO;

        // Long positions: (Price - Entry) * Qty * Multiplier
        if let Some(queue) = self.long_inventory.get(symbol) {
            for entry in queue {
                pnl += (current_price - entry.price) * entry.quantity * entry.multiplier;
            }
        }

        // Short positions: (Entry - Price) * Qty * Multiplier
        if let Some(queue) = self.short_inventory.get(symbol) {
            for entry in queue {
                pnl += (entry.price - current_price) * entry.quantity * entry.multiplier;
            }
        }

        pnl
    }

    pub fn process_trade(
        &mut self,
        trade: &Trade,
        multiplier: Decimal,
        order_tag: Option<&str>,
        history: Option<&SymbolHistory>,
        portfolio_value: Decimal,
    ) {
        let symbol = trade.symbol.clone();
        let side = trade.side;
        let qty = trade.quantity;
        let price = trade.price;
        let comm = trade.commission;
        let bar_idx = trade.bar_index;
        let timestamp = trade.timestamp;
        let tag = order_tag.unwrap_or("").to_string();

        self.total_commission += comm;

        let mut remaining_qty = qty;

        match side {
            OrderSide::Buy => {
                // Try to cover shorts
                if let Some(inventory) = self.short_inventory.get_mut(&symbol) {
                    while remaining_qty > Decimal::ZERO && !inventory.is_empty() {
                        let match_entry = inventory.front_mut().unwrap();
                        let covered_qty = remaining_qty.min(match_entry.quantity);

                        // Short PnL = (Entry Price - Exit Price) * Qty
                        let pnl =
                            (match_entry.price - price) * covered_qty * match_entry.multiplier;
                        self.total_pnl += pnl;

                        let entry_val = match_entry.price * covered_qty * match_entry.multiplier;
                        let ret_pct = if !entry_val.is_zero() {
                            pnl / entry_val
                        } else {
                            Decimal::ZERO
                        };
                        let bars = if bar_idx >= match_entry.bar_idx {
                            Decimal::from((bar_idx - match_entry.bar_idx) as i64)
                        } else {
                            Decimal::ZERO
                        };

                        // Pro-rate commission for entry and exit
                        // Entry comm (partial)
                        let entry_comm_part = if match_entry.quantity > Decimal::ZERO {
                            match_entry.commission * (covered_qty / match_entry.quantity)
                        } else {
                            Decimal::ZERO
                        };
                        // Exit comm (partial of current trade)
                        let exit_comm_part = if qty > Decimal::ZERO {
                            comm * (covered_qty / qty)
                        } else {
                            Decimal::ZERO
                        };
                        let total_trade_comm = entry_comm_part + exit_comm_part;

                        let to_f64 = |d: Decimal| d.to_f64().unwrap_or_default();

                        // Calculate MAE/MFE and Max Drawdown
                        let mut mae = 0.0;
                        let mut mfe = 0.0;
                        let mut max_dd_pct = 0.0;

                        if let Some(hist) = history {
                            // Short trade:
                            // MAE (Adverse) = Max High during trade
                            // MFE (Favorable) = Min Low during trade
                            let start_ts = match_entry.timestamp;
                            let end_ts = timestamp;

                            // Find indices
                            let start_idx = match hist.timestamps.binary_search(&start_ts) {
                                Ok(i) => i,
                                Err(i) => i,
                            };
                            let end_idx = match hist.timestamps.binary_search(&end_ts) {
                                Ok(i) => i,
                                Err(i) => i, // Include up to end_ts
                            };

                            // Iterate
                            let entry_px_f64 = to_f64(match_entry.price);
                            let mut max_high = entry_px_f64;
                            let mut min_low = entry_px_f64;

                            let mut valley_low = entry_px_f64; // For Short Drawdown: lowest price seen so far (max profit)
                            let mut current_max_dd = 0.0;

                            let search_start = start_idx;
                            let search_end = end_idx + 1; // inclusive of end_idx

                            if search_start < hist.timestamps.len() {
                                let limit = search_end.min(hist.timestamps.len());
                                for i in search_start..limit {
                                    let h = hist.highs[i];
                                    let l = hist.lows[i];

                                    if h > max_high {
                                        max_high = h;
                                    }
                                    if l < min_low {
                                        min_low = l;
                                    }

                                    // Short Drawdown Logic:
                                    // Drawdown is from the lowest price seen so far (valley).
                                    // If price rises from valley, that's drawdown.
                                    if l < valley_low {
                                        valley_low = l;
                                    }
                                    // Calculate drawdown at this bar's high (worst case in this bar)
                                    // DD = (High - Valley) / Valley
                                    if valley_low > 0.0 {
                                        let dd = (h - valley_low) / valley_low;
                                        if dd > current_max_dd {
                                            current_max_dd = dd;
                                        }
                                    }
                                }
                            }

                            // If no bars in between (same bar entry/exit), assume entry/exit prices are bounds
                            if max_high == -f64::INFINITY {
                                let p = to_f64(price);
                                max_high = entry_px_f64.max(p);
                                min_low = entry_px_f64.min(p);

                                // Check DD with exit price
                                if p < valley_low {
                                    valley_low = p;
                                }
                                if valley_low > 0.0 {
                                    let dd = (p - valley_low) / valley_low;
                                    if dd > current_max_dd {
                                        current_max_dd = dd;
                                    }
                                }
                            } else {
                                // Also consider exit price
                                let p = to_f64(price);
                                max_high = max_high.max(p);
                                min_low = min_low.min(p);

                                if p < valley_low {
                                    valley_low = p;
                                }
                                if valley_low > 0.0 {
                                    let dd = (p - valley_low) / valley_low;
                                    if dd > current_max_dd {
                                        current_max_dd = dd;
                                    }
                                }
                            }

                            // Short:
                            // MAE: (Entry - MaxHigh) / Entry. (e.g. 100 - 110 / 100 = -10%)
                            mae = (entry_px_f64 - max_high) / entry_px_f64 * 100.0;
                            // MFE: (Entry - MinLow) / Entry. (e.g. 100 - 90 / 100 = +10%)
                            mfe = (entry_px_f64 - min_low) / entry_px_f64 * 100.0;

                            max_dd_pct = current_max_dd * 100.0;
                        }

                        Arc::make_mut(&mut self.closed_trades).push(ClosedTrade {
                            symbol: symbol.clone(),
                            entry_time: match_entry.timestamp,
                            exit_time: timestamp,
                            entry_price: to_f64(match_entry.price),
                            exit_price: to_f64(price),
                            quantity: to_f64(covered_qty),
                            side: "Short".to_string(),
                            pnl: to_f64(pnl),
                            net_pnl: to_f64(pnl - total_trade_comm),
                            return_pct: to_f64(ret_pct) * 100.0,
                            commission: to_f64(total_trade_comm),
                            duration_bars: bars.to_usize().unwrap_or(0),
                            duration: ((timestamp as i128) - (match_entry.timestamp as i128))
                                as u64,
                            mae,
                            mfe,
                            entry_tag: match_entry.tag.clone(),
                            exit_tag: tag.clone(),
                            entry_portfolio_value: to_f64(match_entry.entry_portfolio_value),
                            max_drawdown_pct: max_dd_pct,
                        });

                        record_close!(self, pnl, ret_pct, bars);

                        remaining_qty -= covered_qty;
                        match_entry.quantity -= covered_qty;
                        // Reduce remaining commission in inventory proportionally
                        match_entry.commission -= entry_comm_part;

                        if match_entry.quantity <= Decimal::new(1, 6) {
                            inventory.pop_front();
                        }
                    }
                }

                if remaining_qty > Decimal::ZERO {
                    // Calculate remaining commission for this part
                    let remaining_comm = if qty > Decimal::ZERO {
                        comm * (remaining_qty / qty)
                    } else {
                        Decimal::ZERO
                    };
                    self.long_inventory
                        .entry(symbol.clone())
                        .or_default()
                        .push_back(TradeEntry {
                            quantity: remaining_qty,
                            price,
                            multiplier,
                            commission: remaining_comm,
                            bar_idx,
                            timestamp,
                            tag: tag.clone(),
                            entry_portfolio_value: portfolio_value,
                        });
                }
            }
            OrderSide::Sell => {
                // Try to close longs
                if let Some(inventory) = self.long_inventory.get_mut(&symbol) {
                    while remaining_qty > Decimal::ZERO && !inventory.is_empty() {
                        let match_entry = inventory.front_mut().unwrap();
                        let covered_qty = remaining_qty.min(match_entry.quantity);

                        // Long PnL = (Exit Price - Entry Price) * Qty
                        let pnl =
                            (price - match_entry.price) * covered_qty * match_entry.multiplier;
                        self.total_pnl += pnl;

                        let entry_val = match_entry.price * covered_qty * match_entry.multiplier;
                        let ret_pct = if !entry_val.is_zero() {
                            pnl / entry_val
                        } else {
                            Decimal::ZERO
                        };
                        let bars = if bar_idx >= match_entry.bar_idx {
                            Decimal::from((bar_idx - match_entry.bar_idx) as i64)
                        } else {
                            Decimal::ZERO
                        };

                        // Pro-rate commission
                        let entry_comm_part = if match_entry.quantity > Decimal::ZERO {
                            match_entry.commission * (covered_qty / match_entry.quantity)
                        } else {
                            Decimal::ZERO
                        };
                        let exit_comm_part = if qty > Decimal::ZERO {
                            comm * (covered_qty / qty)
                        } else {
                            Decimal::ZERO
                        };
                        let total_trade_comm = entry_comm_part + exit_comm_part;

                        let to_f64 = |d: Decimal| d.to_f64().unwrap_or_default();

                        // Calculate MAE/MFE
                        let mut mae = 0.0;
                        let mut mfe = 0.0;
                        let mut max_dd_pct = 0.0;

                        if let Some(hist) = history {
                            // Long trade:
                            // MAE (Adverse) = Min Low during trade
                            // MFE (Favorable) = Max High during trade
                            let start_ts = match_entry.timestamp;
                            let end_ts = timestamp;

                            let start_idx = match hist.timestamps.binary_search(&start_ts) {
                                Ok(i) => i,
                                Err(i) => i,
                            };
                            let end_idx = match hist.timestamps.binary_search(&end_ts) {
                                Ok(i) => i,
                                Err(i) => i,
                            };

                            let entry_px_f64 = to_f64(match_entry.price);
                            let mut max_high = entry_px_f64;
                            let mut min_low = entry_px_f64;

                            let mut peak_high = entry_px_f64; // For Long Drawdown: highest price seen so far (max profit)
                            let mut current_max_dd = 0.0;

                            let search_start = start_idx;
                            let search_end = end_idx + 1;

                            if search_start < hist.timestamps.len() {
                                let limit = search_end.min(hist.timestamps.len());
                                for i in search_start..limit {
                                    let h = hist.highs[i];
                                    let l = hist.lows[i];

                                    if h > max_high {
                                        max_high = h;
                                    }
                                    if l < min_low {
                                        min_low = l;
                                    }

                                    // Long Drawdown Logic:
                                    // Drawdown is from the highest price seen so far (peak).
                                    // If price falls from peak, that's drawdown.
                                    if h > peak_high {
                                        peak_high = h;
                                    }
                                    // Calculate drawdown at this bar's low (worst case in this bar)
                                    // DD = (Peak - Low) / Peak
                                    if peak_high > 0.0 {
                                        let dd = (peak_high - l) / peak_high;
                                        if dd > current_max_dd {
                                            current_max_dd = dd;
                                        }
                                    }
                                }
                            }

                            if max_high == -f64::INFINITY {
                                let p = to_f64(price);
                                max_high = entry_px_f64.max(p);
                                min_low = entry_px_f64.min(p);

                                // Check DD with exit price
                                if p > peak_high {
                                    peak_high = p;
                                }
                                if peak_high > 0.0 {
                                    let dd = (peak_high - p) / peak_high;
                                    if dd > current_max_dd {
                                        current_max_dd = dd;
                                    }
                                }
                            } else {
                                let p = to_f64(price);
                                max_high = max_high.max(p);
                                min_low = min_low.min(p);

                                // Check DD with exit price
                                if p > peak_high {
                                    peak_high = p;
                                }
                                if peak_high > 0.0 {
                                    let dd = (peak_high - p) / peak_high;
                                    if dd > current_max_dd {
                                        current_max_dd = dd;
                                    }
                                }
                            }

                            // Long:
                            // MAE: (MinLow - Entry) / Entry. (e.g. 90 - 100 / 100 = -10%)
                            mae = (min_low - entry_px_f64) / entry_px_f64 * 100.0;
                            // MFE: (MaxHigh - Entry) / Entry. (e.g. 110 - 100 / 100 = +10%)
                            mfe = (max_high - entry_px_f64) / entry_px_f64 * 100.0;

                            max_dd_pct = current_max_dd * 100.0;
                        }

                        Arc::make_mut(&mut self.closed_trades).push(ClosedTrade {
                            symbol: symbol.clone(),
                            entry_time: match_entry.timestamp,
                            exit_time: timestamp,
                            entry_price: to_f64(match_entry.price),
                            exit_price: to_f64(price),
                            quantity: to_f64(covered_qty),
                            side: "Long".to_string(),
                            pnl: to_f64(pnl),
                            net_pnl: to_f64(pnl - total_trade_comm),
                            return_pct: to_f64(ret_pct) * 100.0,
                            commission: to_f64(total_trade_comm),
                            duration_bars: bars.to_usize().unwrap_or(0),
                            duration: ((timestamp as i128) - (match_entry.timestamp as i128))
                                as u64,
                            mae,
                            mfe,
                            entry_tag: match_entry.tag.clone(),
                            exit_tag: tag.clone(),
                            entry_portfolio_value: to_f64(match_entry.entry_portfolio_value),
                            max_drawdown_pct: max_dd_pct,
                        });

                        record_close!(self, pnl, ret_pct, bars);

                        remaining_qty -= covered_qty;
                        match_entry.quantity -= covered_qty;
                        match_entry.commission -= entry_comm_part;

                        if match_entry.quantity <= Decimal::new(1, 6) {
                            inventory.pop_front();
                        }
                    }
                }

                if remaining_qty > Decimal::ZERO {
                    let remaining_comm = if qty > Decimal::ZERO {
                        comm * (remaining_qty / qty)
                    } else {
                        Decimal::ZERO
                    };
                    self.short_inventory
                        .entry(symbol.clone())
                        .or_default()
                        .push_back(TradeEntry {
                            quantity: remaining_qty,
                            price,
                            multiplier,
                            commission: remaining_comm,
                            bar_idx,
                            timestamp,
                            tag: tag.clone(),
                            entry_portfolio_value: portfolio_value,
                        });
                }
            }
        }
    }

    pub fn get_average_price(&self, symbol: &str) -> Decimal {
        let calc_avg = |inventory: &VecDeque<TradeEntry>| {
            if inventory.is_empty() {
                return Decimal::ZERO;
            }
            let mut total_val = Decimal::ZERO;
            let mut total_qty = Decimal::ZERO;
            for entry in inventory {
                total_val += entry.quantity * entry.price;
                total_qty += entry.quantity;
            }
            if total_qty.is_zero() {
                Decimal::ZERO
            } else {
                total_val / total_qty
            }
        };

        if let Some(inventory) = self.long_inventory.get(symbol) {
            calc_avg(inventory)
        } else if let Some(inventory) = self.short_inventory.get(symbol) {
            calc_avg(inventory)
        } else {
            Decimal::ZERO
        }
    }

    pub fn calculate_pnl(&self, current_prices: Option<HashMap<String, Decimal>>) -> TradePnL {
        let mut unrealized_pnl = Decimal::ZERO;

        if let Some(prices) = current_prices {
            for (symbol, inventory) in &self.long_inventory {
                if let Some(price) = prices.get(symbol) {
                    for entry in inventory {
                        unrealized_pnl +=
                            (*price - entry.price) * entry.quantity * entry.multiplier;
                    }
                }
            }
            for (symbol, inventory) in &self.short_inventory {
                if let Some(price) = prices.get(symbol) {
                    for entry in inventory {
                        unrealized_pnl +=
                            (entry.price - *price) * entry.quantity * entry.multiplier;
                    }
                }
            }
        }

        let total_closed_trades = self.closed_trades.len();
        let win_rate = if total_closed_trades > 0 {
            (self.won_count as f64 / total_closed_trades as f64) * 100.0
        } else {
            0.0
        };
        let loss_rate = if total_closed_trades > 0 {
            (self.lost_count as f64 / total_closed_trades as f64) * 100.0
        } else {
            0.0
        };

        // Avg stats
        let mut sum_pnl = Decimal::ZERO;
        let mut sum_ret = Decimal::ZERO;
        let mut sum_bars = Decimal::ZERO;
        let mut sum_win_bars = Decimal::ZERO;
        let mut sum_loss_bars = Decimal::ZERO;

        let mut largest_win = Decimal::ZERO;
        let mut largest_win_pct = Decimal::ZERO;
        let mut largest_win_bars = Decimal::ZERO;
        let mut largest_loss = Decimal::ZERO;
        let mut largest_loss_pct = Decimal::ZERO;
        let mut largest_loss_bars = Decimal::ZERO;

        for (pnl, ret, bars, is_win) in &self.closed_trades_stats {
            sum_pnl += pnl;
            sum_ret += ret;
            sum_bars += bars;

            if *is_win {
                sum_win_bars += bars;
                if *pnl > largest_win {
                    largest_win = *pnl;
                    largest_win_pct = *ret;
                    largest_win_bars = *bars;
                }
            } else {
                sum_loss_bars += bars;
                if *pnl < largest_loss {
                    largest_loss = *pnl;
                    largest_loss_pct = *ret;
                    largest_loss_bars = *bars;
                }
            }
        }

        let to_f64 = |d: Decimal| d.to_f64().unwrap_or_default();

        let avg_pnl = if total_closed_trades > 0 {
            to_f64(sum_pnl) / total_closed_trades as f64
        } else {
            0.0
        };
        let avg_return_pct = if total_closed_trades > 0 {
            to_f64(sum_ret) / total_closed_trades as f64
        } else {
            0.0
        };
        let avg_trade_bars = if total_closed_trades > 0 {
            to_f64(sum_bars) / total_closed_trades as f64
        } else {
            0.0
        };

        let avg_profit = if self.won_count > 0 {
            to_f64(self.won_pnl) / self.won_count as f64
        } else {
            0.0
        };
        let avg_profit_pct = if self.won_count > 0 {
            let sum_win_ret: Decimal = self
                .closed_trades_stats
                .iter()
                .filter(|(_, _, _, w)| *w)
                .map(|(_, r, _, _)| *r)
                .sum();
            to_f64(sum_win_ret) / self.won_count as f64
        } else {
            0.0
        };
        let avg_winning_trade_bars = if self.won_count > 0 {
            to_f64(sum_win_bars) / self.won_count as f64
        } else {
            0.0
        };

        let avg_loss = if self.lost_count > 0 {
            to_f64(self.lost_pnl) / self.lost_count as f64
        } else {
            0.0
        };
        let avg_loss_pct = if self.lost_count > 0 {
            let sum_loss_ret: Decimal = self
                .closed_trades_stats
                .iter()
                .filter(|(_, _, _, w)| !*w)
                .map(|(_, r, _, _)| *r)
                .sum();
            to_f64(sum_loss_ret) / self.lost_count as f64
        } else {
            0.0
        };
        let avg_losing_trade_bars = if self.lost_count > 0 {
            to_f64(sum_loss_bars) / self.lost_count as f64
        } else {
            0.0
        };

        let profit_factor = if self.lost_pnl.abs() > Decimal::ZERO {
            to_f64(self.won_pnl / self.lost_pnl.abs())
        } else if self.won_pnl > Decimal::ZERO {
            f64::INFINITY
        } else {
            0.0
        };

        // SQN
        let sqn = if total_closed_trades > 0 {
            let avg_pnl_val = to_f64(sum_pnl) / total_closed_trades as f64;
            let variance = self
                .closed_trades_stats
                .iter()
                .map(|(pnl, _, _, _)| {
                    let diff = to_f64(*pnl) - avg_pnl_val;
                    diff * diff
                })
                .sum::<f64>()
                / total_closed_trades as f64;
            let std_dev = variance.sqrt();
            if std_dev != 0.0 {
                (avg_pnl_val / std_dev) * (total_closed_trades as f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Kelly Criterion
        let kelly_criterion = if win_rate > 0.0 && avg_loss.abs() > 0.0 {
            let w = win_rate / 100.0;
            let r = avg_profit / avg_loss.abs();
            if r > 0.0 { w - (1.0 - w) / r } else { 0.0 }
        } else {
            0.0
        };

        TradePnL {
            gross_pnl: to_f64(self.total_pnl),
            net_pnl: to_f64(self.total_pnl - self.total_commission),
            total_commission: to_f64(self.total_commission),
            total_closed_trades,
            won_count: self.won_count,
            lost_count: self.lost_count,
            won_pnl: to_f64(self.won_pnl),
            lost_pnl: to_f64(self.lost_pnl),
            win_rate,
            loss_rate,
            unrealized_pnl: to_f64(unrealized_pnl),
            avg_pnl,
            avg_return_pct: avg_return_pct * 100.0,
            avg_trade_bars,
            avg_profit,
            avg_profit_pct: avg_profit_pct * 100.0,
            avg_winning_trade_bars,
            avg_loss,
            avg_loss_pct: avg_loss_pct * 100.0,
            avg_losing_trade_bars,
            largest_win: to_f64(largest_win),
            largest_win_pct: to_f64(largest_win_pct) * 100.0,
            largest_win_bars: to_f64(largest_win_bars),
            largest_loss: to_f64(largest_loss),
            largest_loss_pct: to_f64(largest_loss_pct) * 100.0,
            largest_loss_bars: to_f64(largest_loss_bars),
            max_wins: self.max_wins,
            max_losses: self.max_losses,
            profit_factor,
            total_profit: to_f64(self.won_pnl),
            total_loss: to_f64(self.lost_pnl),
            sqn,
            kelly_criterion,
        }
    }
}
