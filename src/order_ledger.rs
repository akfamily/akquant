//! 策略槽的订单账本: 由上下文里的订单列表推算出要通知策略的 on_order / on_reject /
//! on_trade 事件, 是原 Python `strategy_order_events.check_order_events` 比对算法
//! 的逐步复刻(规格: docs/superpowers/specs/2026-11-08-order-event-ledger-design.md §3)。
//!
//! 纯 Rust, 不依赖 PyO3: Python 只按 [`OrderLedger::stage_a`] / [`OrderLedger::stage_b`]
//! 返回的顺序调用回调。分两段是为了保住时机: 第一段(主动撤单/拒单/过期)的回调里
//! 下的新单, 要在第二段读"本轮新单"时被看到, 同一次检查就通知。

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::model::{Order, OrderSide, OrderStatus, Trade};

/// 订单状态指纹去重的上限(与原 Python `_ORDER_EVENT_DEDUPE_LIMIT` 一致)。
pub const ORDER_EVENT_DEDUPE_LIMIT: usize = 50_000;

/// 终态留档与成交去重的上限, 由 Python 每次按策略属性传入
/// (`finalized_order_cache_size` 默认 10000, `trade_dedupe_cache_size` 默认 50000)。
#[derive(Debug, Clone, Copy)]
pub struct LedgerLimits {
    pub finalized: usize,
    pub trade: usize,
}

/// 多策略槽位的归属规则(原 Python `_slot_filter`): 订单 / 成交的 `owner_strategy_id`
/// 去空白后不在已登记槽位里的, 归默认槽; 只保留归属本槽的。
#[derive(Debug, Clone)]
pub struct SlotOwnership {
    mine: String,
    default_id: String,
    slot_ids: HashSet<String>,
}

impl SlotOwnership {
    /// 槽位数不超过 1 时不过滤, 返回 None(与原实现一致)。
    pub fn new(mine: String, default_id: String, slot_ids: HashSet<String>) -> Option<Self> {
        if slot_ids.len() <= 1 {
            return None;
        }
        let mine = if mine.trim().is_empty() {
            default_id.clone()
        } else {
            mine.trim().to_string()
        };
        Some(Self {
            mine,
            default_id,
            slot_ids,
        })
    }

    fn owns(&self, owner: Option<&str>) -> bool {
        let owner = owner.map(str::trim).unwrap_or("");
        let owner = if self.slot_ids.contains(owner) {
            owner
        } else {
            self.default_id.as_str()
        };
        owner == self.mine
    }
}

fn owned_orders<'a>(orders: &'a [Order], ownership: Option<&SlotOwnership>) -> Vec<&'a Order> {
    orders
        .iter()
        .filter(|o| ownership.is_none_or(|own| own.owns(o.owner_strategy_id.as_deref())))
        .collect()
}

fn owned_trades<'a>(trades: &'a [Trade], ownership: Option<&SlotOwnership>) -> Vec<&'a Trade> {
    trades
        .iter()
        .filter(|t| ownership.is_none_or(|own| own.owns(t.owner_strategy_id.as_deref())))
        .collect()
}

/// 要通知策略的一次订单事件: 先 `on_order`, `fire_reject` 为 true 时紧接 `on_reject`。
#[derive(Debug, Clone)]
pub struct OrderEvent {
    pub order: Order,
    pub fire_reject: bool,
}

pub struct StageAInput<'a> {
    /// 本槽本周期主动撤单(上下文的 `canceled_order_ids`, 不按归属过滤)。
    pub canceled_ids: &'a [String],
    /// 上下文的 `active_orders`(①里找"不在账本但在途"的撤单, 不按归属过滤)。
    pub active: &'a [Order],
    pub rejected: &'a [Order],
    pub expired: &'a [Order],
}

pub struct StageBInput<'a> {
    /// 本周期新提交(上下文的 `orders`), 必须在第一段回调执行完之后再读。
    pub new_orders: &'a [Order],
    pub active: &'a [Order],
    pub trades: &'a [Trade],
    /// 本步离开在途账本的终态快照(不按归属过滤, 只与本槽账本对照)。
    pub finalized: &'a [Order],
}

/// 订单状态指纹(原 Python `order_event_key`); 数值取 normalize 后比较。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OrderEventKey {
    id: String,
    status: OrderStatus,
    filled_quantity: Decimal,
    average_filled_price: Option<Decimal>,
    reject_reason: String,
}

fn order_event_key(order: &Order) -> OrderEventKey {
    OrderEventKey {
        id: order.id.clone(),
        status: order.status,
        filled_quantity: order.filled_quantity.normalize(),
        average_filled_price: order.average_filled_price.map(|d| d.normalize()),
        reject_reason: order.reject_reason.clone(),
    }
}

/// 成交指纹(原 Python `trade_event_key`, 其中 trade_id 恒为 None 故省略)。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TradeKey {
    id: String,
    order_id: String,
    timestamp: i64,
    symbol: String,
    side: OrderSide,
    quantity: Decimal,
    price: Decimal,
}

fn trade_key(trade: &Trade) -> TradeKey {
    TradeKey {
        id: trade.id.clone(),
        order_id: trade.order_id.clone(),
        timestamp: trade.timestamp,
        symbol: trade.symbol.clone(),
        side: trade.side,
        quantity: trade.quantity.normalize(),
        price: trade.price.normalize(),
    }
}

/// 有界 FIFO 去重集合(原 Python 的 set + deque 组合)。
#[derive(Debug, Clone)]
struct BoundedSet<K> {
    set: HashSet<K>,
    order: VecDeque<K>,
}

impl<K> Default for BoundedSet<K> {
    fn default() -> Self {
        Self {
            set: HashSet::new(),
            order: VecDeque::new(),
        }
    }
}

impl<K: Eq + Hash + Clone> BoundedSet<K> {
    /// 首次出现返回 true 并记入; 超过上限时淘汰最早记入的。
    fn insert(&mut self, key: K, limit: usize) -> bool {
        if self.set.contains(&key) {
            return false;
        }
        self.set.insert(key.clone());
        self.order.push_back(key);
        while self.order.len() > limit.max(1) {
            if let Some(oldest) = self.order.pop_front() {
                self.set.remove(&oldest);
            }
        }
        true
    }
}

/// 按插入顺序遍历的在途订单表(原 Python dict 语义: 覆盖已有键不改位置,
/// 删除后再插入排到末尾)。
#[derive(Debug, Clone, Default)]
struct KnownOrders {
    entries: HashMap<String, (u64, Order)>,
    sequence: Vec<(u64, String)>,
    next_seq: u64,
}

impl KnownOrders {
    fn contains(&self, id: &str) -> bool {
        self.entries.contains_key(id)
    }

    fn get(&self, id: &str) -> Option<&Order> {
        self.entries.get(id).map(|(_, order)| order)
    }

    fn insert(&mut self, order: Order) {
        if let Some((_, slot)) = self.entries.get_mut(&order.id) {
            *slot = order;
            return;
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        self.sequence.push((seq, order.id.clone()));
        self.entries.insert(order.id.clone(), (seq, order));
    }

    fn remove(&mut self, id: &str) -> Option<Order> {
        let removed = self.entries.remove(id).map(|(_, order)| order);
        if self.sequence.len() > 2 * self.entries.len() + 16 {
            let entries = &self.entries;
            self.sequence
                .retain(|(seq, id)| entries.get(id).is_some_and(|(s, _)| s == seq));
        }
        removed
    }

    /// 当前在表中的订单号, 按插入顺序。
    fn ids(&self) -> Vec<String> {
        self.sequence
            .iter()
            .filter(|(seq, id)| self.entries.get(id).is_some_and(|(s, _)| s == seq))
            .map(|(_, id)| id.clone())
            .collect()
    }
}

fn is_terminal(status: OrderStatus) -> bool {
    matches!(
        status,
        OrderStatus::Filled | OrderStatus::Cancelled | OrderStatus::Rejected | OrderStatus::Expired
    )
}

/// 一个策略槽的订单账本。
#[derive(Debug, Clone, Default)]
pub struct OrderLedger {
    known: KnownOrders,
    order_event_keys: BoundedSet<OrderEventKey>,
    rejected_notified: HashSet<String>,
    trade_keys: BoundedSet<TradeKey>,
    finalized: HashMap<String, Order>,
    finalized_ids: VecDeque<String>,
}

/// 存档内容: 只保存在途订单与已通知过的成交(与原实现随策略 pickle 的范围一致;
/// 状态指纹、on_reject 记录、终态留档不保存, 续跑时宁可多通知一次)。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrderLedgerSnapshot {
    pub known: Vec<Order>,
    pub trade_keys: Vec<TradeKey>,
}

impl OrderLedger {
    /// 第一段: ① 主动撤单、② 拒单、③ 过期。
    pub fn stage_a(
        &mut self,
        input: StageAInput<'_>,
        ownership: Option<&SlotOwnership>,
        limits: LedgerLimits,
    ) -> Vec<OrderEvent> {
        let mut events = Vec::new();
        // ① 本槽主动撤单(不按归属过滤)
        for oid in input.canceled_ids {
            if let Some(mut order) = self.known.remove(oid) {
                order.status = OrderStatus::Cancelled;
                self.emit(&order, &mut events);
                self.remember_finalized(order, limits.finalized);
                continue;
            }
            // 不在账本但在途: 只按撤单留档, 不通知
            if let Some(active) = input.active.iter().find(|o| &o.id == oid) {
                let mut order = active.clone();
                order.status = OrderStatus::Cancelled;
                self.remember_finalized(order, limits.finalized);
            }
        }
        // ② 拒单
        for order in owned_orders(input.rejected, ownership) {
            if order.id.is_empty() {
                continue;
            }
            self.known.insert(order.clone());
            self.emit(order, &mut events);
        }
        // ③ 过期
        for order in owned_orders(input.expired, ownership) {
            if order.id.is_empty() {
                continue;
            }
            self.known.insert(order.clone());
            self.emit(order, &mut events);
            self.known.remove(&order.id);
            self.remember_finalized(order.clone(), limits.finalized);
        }
        events
    }

    /// 第二段: ④ 新单、⑤ 在途变化、⑥⑦⑧ 离开在途、⑨ 成交。
    pub fn stage_b(
        &mut self,
        input: StageBInput<'_>,
        ownership: Option<&SlotOwnership>,
        limits: LedgerLimits,
    ) -> (Vec<OrderEvent>, Vec<Trade>) {
        let mut events = Vec::new();
        // ④ 本轮新单
        let mut pending: HashSet<String> = HashSet::new();
        for order in owned_orders(input.new_orders, ownership) {
            if order.id.is_empty() || self.known.contains(&order.id) {
                continue;
            }
            pending.insert(order.id.clone());
            self.known.insert(order.clone());
            self.emit(order, &mut events);
        }
        // ⑤ 在途变化
        let mut current_active: HashSet<String> = HashSet::new();
        for order in owned_orders(input.active, ownership) {
            current_active.insert(order.id.clone());
            let changed = match self.known.get(&order.id) {
                None => true,
                Some(known) => {
                    known.status != order.status || known.filled_quantity != order.filled_quantity
                }
            };
            if changed {
                self.known.insert(order.clone());
                self.emit(order, &mut events);
            }
        }
        // ⑥ 本步有成交的订单号
        let trades = owned_trades(input.trades, ownership);
        let traded: HashSet<&str> = trades.iter().map(|t| t.order_id.as_str()).collect();
        // ⑦ 本步终态快照(不按归属过滤)
        let finalized: HashMap<&str, &Order> =
            input.finalized.iter().map(|o| (o.id.as_str(), o)).collect();
        // ⑧ 离开在途的订单, 按插入顺序
        for oid in self.known.ids() {
            if current_active.contains(&oid) {
                continue;
            }
            if traded.contains(oid.as_str()) {
                if let Some(mut order) = self.known.remove(&oid) {
                    order.status = OrderStatus::Filled;
                    self.emit(&order, &mut events);
                    self.remember_finalized(order, limits.finalized);
                }
            } else if pending.contains(&oid) {
                continue;
            } else if let Some(terminal) = finalized.get(oid.as_str()) {
                let order = (*terminal).clone();
                self.known.remove(&oid);
                self.emit(&order, &mut events);
                self.remember_finalized(order, limits.finalized);
            } else if let Some(order) = self.known.remove(&oid) {
                self.remember_finalized(order, limits.finalized);
            }
        }
        // ⑨ 成交去重
        let mut out = Vec::new();
        for t in trades {
            if self.trade_keys.insert(trade_key(t), limits.trade) {
                out.push(t.clone());
            }
        }
        (events, out)
    }

    /// 查订单: 账本在途 → 上下文在途 → 终态留档(原 Python `get_order` 的顺序)。
    pub fn find(&self, order_id: &str, active: &[Order]) -> Option<Order> {
        if let Some(order) = self.known.get(order_id) {
            return Some(order.clone());
        }
        if let Some(order) = active.iter().find(|o| o.id == order_id) {
            return Some(order.clone());
        }
        self.finalized.get(order_id).cloned()
    }

    pub fn snapshot(&self) -> OrderLedgerSnapshot {
        OrderLedgerSnapshot {
            known: self
                .known
                .ids()
                .iter()
                .filter_map(|id| self.known.get(id).cloned())
                .collect(),
            trade_keys: self.trade_keys.order.iter().cloned().collect(),
        }
    }

    pub fn restore(snapshot: OrderLedgerSnapshot) -> Self {
        let mut ledger = Self::default();
        for order in snapshot.known {
            ledger.known.insert(order);
        }
        for key in snapshot.trade_keys {
            ledger.trade_keys.set.insert(key.clone());
            ledger.trade_keys.order.push_back(key);
        }
        ledger
    }

    /// 按状态指纹去重后记一次通知; 拒单首次通知时附带 on_reject。
    fn emit(&mut self, order: &Order, events: &mut Vec<OrderEvent>) {
        if !self
            .order_event_keys
            .insert(order_event_key(order), ORDER_EVENT_DEDUPE_LIMIT)
        {
            return;
        }
        let fire_reject = order.status == OrderStatus::Rejected
            && !order.id.is_empty()
            && self.rejected_notified.insert(order.id.clone());
        events.push(OrderEvent {
            order: order.clone(),
            fire_reject,
        });
    }

    /// 终态留档(原 Python `remember_finalized_order`)。
    fn remember_finalized(&mut self, order: Order, limit: usize) {
        if order.id.is_empty() {
            return;
        }
        match self.finalized.get(&order.id) {
            Some(existing) if is_terminal(existing.status) && !is_terminal(order.status) => return,
            Some(_) => {}
            None => self.finalized_ids.push_back(order.id.clone()),
        }
        self.finalized.insert(order.id.clone(), order);
        while self.finalized_ids.len() > limit.max(1) {
            if let Some(oldest) = self.finalized_ids.pop_front() {
                self.finalized.remove(&oldest);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{OrderType, PositionEffect};
    use rust_decimal_macros::dec;

    const LIMITS: LedgerLimits = LedgerLimits {
        finalized: 10_000,
        trade: 50_000,
    };

    fn order(id: &str, status: OrderStatus, filled: i64) -> Order {
        let mut o = Order::test_new(id, "X", OrderSide::Buy, OrderType::Limit, dec!(100));
        o.status = status;
        o.filled_quantity = Decimal::from(filled);
        o
    }

    fn owned(id: &str, owner: Option<&str>) -> Order {
        let mut o = order(id, OrderStatus::New, 0);
        o.owner_strategy_id = owner.map(str::to_string);
        o
    }

    fn trade(id: &str, order_id: &str) -> Trade {
        Trade {
            id: id.to_string(),
            order_id: order_id.to_string(),
            symbol: "X".to_string(),
            side: OrderSide::Buy,
            position_effect: PositionEffect::Auto,
            quantity: dec!(100),
            price: dec!(10),
            commission: Decimal::ZERO,
            timestamp: 1,
            bar_index: 0,
            owner_strategy_id: None,
        }
    }

    fn a<'a>(
        canceled: &'a [String],
        active: &'a [Order],
        rejected: &'a [Order],
        expired: &'a [Order],
    ) -> StageAInput<'a> {
        StageAInput {
            canceled_ids: canceled,
            active,
            rejected,
            expired,
        }
    }

    fn b<'a>(
        new_orders: &'a [Order],
        active: &'a [Order],
        trades: &'a [Trade],
        finalized: &'a [Order],
    ) -> StageBInput<'a> {
        StageBInput {
            new_orders,
            active,
            trades,
            finalized,
        }
    }

    fn sig(events: &[OrderEvent]) -> Vec<(String, OrderStatus, bool)> {
        events
            .iter()
            .map(|e| (e.order.id.clone(), e.order.status, e.fire_reject))
            .collect()
    }

    fn st(id: &str, status: OrderStatus, reject: bool) -> (String, OrderStatus, bool) {
        (id.to_string(), status, reject)
    }

    /// ④ 新单通知一次; ⑤ 在途且未变不重复; 状态或已成交量变化再通知。
    #[test]
    fn new_order_announced_once_then_changes_announced() {
        let mut l = OrderLedger::default();
        let new = [order("A", OrderStatus::New, 0)];
        let (ev, _) = l.stage_b(b(&new, &[], &[], &[]), None, LIMITS);
        assert_eq!(sig(&ev), vec![st("A", OrderStatus::New, false)]);
        let (ev, _) = l.stage_b(b(&[], &new, &[], &[]), None, LIMITS);
        assert!(ev.is_empty());
        let partial = [order("A", OrderStatus::PartiallyFilled, 50)];
        let (ev, _) = l.stage_b(b(&[], &partial, &[], &[]), None, LIMITS);
        assert_eq!(sig(&ev), vec![st("A", OrderStatus::PartiallyFilled, false)]);
    }

    /// ④ 本轮新单还没进在途时保留在账本, 下一次出现在在途里不重复通知。
    #[test]
    fn new_order_not_yet_active_is_kept() {
        let mut l = OrderLedger::default();
        let new = [order("A", OrderStatus::New, 0)];
        l.stage_b(b(&new, &[], &[], &[]), None, LIMITS);
        let (ev, _) = l.stage_b(b(&[], &new, &[], &[]), None, LIMITS);
        assert!(ev.is_empty());
    }

    /// ⑨ 同一笔成交在同一周期的两次检查里只回调一次(原 test_trade_callback_not_duplicated_on_bar /
    /// test_trade_callback_not_replayed_across_events 的算法部分)。
    #[test]
    fn trade_deduped_across_checks() {
        let mut l = OrderLedger::default();
        let trades = [trade("t1", "o1")];
        let (_, out) = l.stage_b(b(&[], &[], &trades, &[]), None, LIMITS);
        assert_eq!(out.len(), 1);
        let (_, out) = l.stage_b(b(&[], &[], &trades, &[]), None, LIMITS);
        assert!(out.is_empty());
    }

    /// ⑨ 去重缓存满后淘汰最早的指纹(原 test_trade_dedupe_cache_limit_eviction_allows_replay)。
    #[test]
    fn trade_dedupe_eviction_allows_replay() {
        let mut l = OrderLedger::default();
        let limits = LedgerLimits {
            finalized: 10_000,
            trade: 2,
        };
        for id in ["t1", "t2", "t3"] {
            let trades = [trade(id, "o")];
            let (_, out) = l.stage_b(b(&[], &[], &trades, &[]), None, limits);
            assert_eq!(out.len(), 1, "{id}");
        }
        let trades = [trade("t1", "o")];
        let (_, out) = l.stage_b(b(&[], &[], &trades, &[]), None, limits);
        assert_eq!(out.len(), 1, "t1 已被淘汰, 应再次回调");
    }

    /// ② 拒单: on_order 后紧接 on_reject; 同一快照不重复; on_reject 每单只一次
    /// (原 test_framework_hooks_emits_reject_from_recent_rejected_orders 的算法部分)。
    #[test]
    fn rejected_snapshot_fires_reject_once() {
        let mut l = OrderLedger::default();
        let rejected = [order("R", OrderStatus::Rejected, 0)];
        let ev = l.stage_a(a(&[], &[], &rejected, &[]), None, LIMITS);
        assert_eq!(sig(&ev), vec![st("R", OrderStatus::Rejected, true)]);
        let ev = l.stage_a(a(&[], &[], &rejected, &[]), None, LIMITS);
        assert!(ev.is_empty());
        let mut changed = order("R", OrderStatus::Rejected, 0);
        changed.reject_reason = "other".to_string();
        let ev = l.stage_a(a(&[], &[], &[changed], &[]), None, LIMITS);
        assert_eq!(sig(&ev), vec![st("R", OrderStatus::Rejected, false)]);
    }

    /// ⑤ 在途列表里出现的拒单同样触发 on_reject
    /// (原 test_framework_hooks_session_day_reject_and_portfolio 的算法部分)。
    #[test]
    fn rejected_in_active_fires_reject() {
        let mut l = OrderLedger::default();
        let active = [order("R", OrderStatus::Rejected, 0)];
        let (ev, _) = l.stage_b(b(&[], &active, &[], &[]), None, LIMITS);
        assert_eq!(sig(&ev), vec![st("R", OrderStatus::Rejected, true)]);
    }

    /// ① 账本里没有、在途里也没有的撤单不通知(原 test_unknown_canceled_order_does_not_emit_order_callback)。
    #[test]
    fn unknown_cancel_emits_nothing() {
        let mut l = OrderLedger::default();
        let ev = l.stage_a(a(&["ghost".to_string()], &[], &[], &[]), None, LIMITS);
        assert!(ev.is_empty());
        assert!(l.find("ghost", &[]).is_none());
    }

    /// ① 账本里的撤单: 通知 Cancelled、留档、移出账本
    /// (原 test_mixed_events_partial_fill_then_cancel_sequence 的算法部分)。
    #[test]
    fn known_cancel_announced_and_finalized() {
        let mut l = OrderLedger::default();
        let new = [order("A", OrderStatus::New, 0)];
        l.stage_b(b(&new, &[], &[], &[]), None, LIMITS);
        let partial = [order("A", OrderStatus::PartiallyFilled, 30)];
        l.stage_b(b(&[], &partial, &[], &[]), None, LIMITS);
        let ev = l.stage_a(a(&["A".to_string()], &[], &[], &[]), None, LIMITS);
        assert_eq!(sig(&ev), vec![st("A", OrderStatus::Cancelled, false)]);
        assert_eq!(ev[0].order.filled_quantity, Decimal::from(30));
        assert_eq!(l.find("A", &[]).unwrap().status, OrderStatus::Cancelled);
        let (ev, _) = l.stage_b(b(&[], &[], &[], &[]), None, LIMITS);
        assert!(ev.is_empty(), "已移出账本, 不再按消失处理");
    }

    /// ① 不在账本但在途的撤单: 只留档(状态 Cancelled), 不通知。
    #[test]
    fn cancel_of_unknown_active_order_is_finalized_silently() {
        let mut l = OrderLedger::default();
        let active = [order("B", OrderStatus::New, 0)];
        let ev = l.stage_a(a(&["B".to_string()], &active, &[], &[]), None, LIMITS);
        assert!(ev.is_empty());
        assert_eq!(l.find("B", &[]).unwrap().status, OrderStatus::Cancelled);
    }

    /// ③ 过期单: 通知、留档、移出账本。
    #[test]
    fn expired_announced_and_finalized() {
        let mut l = OrderLedger::default();
        let expired = [order("E", OrderStatus::Expired, 0)];
        let ev = l.stage_a(a(&[], &[], &[], &expired), None, LIMITS);
        assert_eq!(sig(&ev), vec![st("E", OrderStatus::Expired, false)]);
        assert_eq!(l.find("E", &[]).unwrap().status, OrderStatus::Expired);
    }

    /// ⑧ 消失且本步有成交: 账本快照改为 Filled 后通知(已成交量取账本里的旧值)。
    #[test]
    fn disappeared_with_trade_announced_filled() {
        let mut l = OrderLedger::default();
        let new = [order("A", OrderStatus::New, 0)];
        l.stage_b(b(&new, &[], &[], &[]), None, LIMITS);
        let trades = [trade("t", "A")];
        let (ev, out) = l.stage_b(b(&[], &[], &trades, &[]), None, LIMITS);
        assert_eq!(sig(&ev), vec![st("A", OrderStatus::Filled, false)]);
        assert_eq!(ev[0].order.filled_quantity, Decimal::ZERO);
        assert_eq!(out.len(), 1);
        assert_eq!(l.find("A", &[]).unwrap().status, OrderStatus::Filled);
    }

    /// ⑧ 消失且有终态快照: 用终态快照通知。
    #[test]
    fn disappeared_with_finalized_snapshot_uses_it() {
        let mut l = OrderLedger::default();
        let new = [order("A", OrderStatus::New, 0)];
        l.stage_b(b(&new, &[], &[], &[]), None, LIMITS);
        let fin = [order("A", OrderStatus::Cancelled, 0)];
        let (ev, _) = l.stage_b(b(&[], &[], &[], &fin), None, LIMITS);
        assert_eq!(sig(&ev), vec![st("A", OrderStatus::Cancelled, false)]);
    }

    /// ⑧ 消失且无从判断: 不通知, 按账本旧快照留档。
    #[test]
    fn disappeared_silently_is_finalized_without_event() {
        let mut l = OrderLedger::default();
        let active = [order("A", OrderStatus::New, 0)];
        l.stage_b(b(&[], &active, &[], &[]), None, LIMITS);
        let (ev, _) = l.stage_b(b(&[], &[], &[], &[]), None, LIMITS);
        assert!(ev.is_empty());
        assert_eq!(l.find("A", &[]).unwrap().status, OrderStatus::New);
    }

    /// ⑧ 遍历按插入顺序; 删除后再插入排到末尾。
    #[test]
    fn disappearance_follows_insertion_order() {
        let mut l = OrderLedger::default();
        let active = [
            order("A", OrderStatus::New, 0),
            order("B", OrderStatus::New, 0),
            order("C", OrderStatus::New, 0),
        ];
        l.stage_b(b(&[], &active, &[], &[]), None, LIMITS);
        let fin_b = [order("B", OrderStatus::Cancelled, 0)];
        l.stage_b(
            b(&[], &[active[0].clone(), active[2].clone()], &[], &fin_b),
            None,
            LIMITS,
        );
        let b_again = [order("B", OrderStatus::New, 0)];
        l.stage_b(
            b(
                &[],
                &[active[0].clone(), active[2].clone(), b_again[0].clone()],
                &[],
                &[],
            ),
            None,
            LIMITS,
        );
        let trades = [trade("t1", "A"), trade("t2", "B"), trade("t3", "C")];
        let (ev, _) = l.stage_b(b(&[], &[], &trades, &[]), None, LIMITS);
        let ids: Vec<String> = ev.iter().map(|e| e.order.id.clone()).collect();
        assert_eq!(ids, vec!["A", "C", "B"]);
    }

    /// ⑧ 覆盖已有订单(状态变化更新快照)不改变它在遍历顺序里的位置(Python dict 语义)。
    #[test]
    fn updating_known_order_keeps_its_position() {
        let mut l = OrderLedger::default();
        let active = [
            order("A", OrderStatus::New, 0),
            order("B", OrderStatus::New, 0),
        ];
        l.stage_b(b(&[], &active, &[], &[]), None, LIMITS);
        let updated = [
            order("A", OrderStatus::PartiallyFilled, 10),
            order("B", OrderStatus::New, 0),
        ];
        l.stage_b(b(&[], &updated, &[], &[]), None, LIMITS);
        let trades = [trade("t1", "A"), trade("t2", "B")];
        let (ev, _) = l.stage_b(b(&[], &[], &trades, &[]), None, LIMITS);
        let ids: Vec<String> = ev.iter().map(|e| e.order.id.clone()).collect();
        assert_eq!(ids, vec!["A", "B"]);
    }

    /// 留档: 已留档的终态快照不被之后的非终态快照覆盖。
    #[test]
    fn finalized_terminal_not_overwritten_by_nonterminal() {
        let mut l = OrderLedger::default();
        let active = [order("A", OrderStatus::New, 0)];
        // 先经①的"在途撤单"留档为 Cancelled, 账本里没有 A
        l.stage_a(a(&["A".to_string()], &active, &[], &[]), None, LIMITS);
        // 再以 New 进入账本并按⑧"无从判断"留档旧快照
        l.stage_b(b(&[], &active, &[], &[]), None, LIMITS);
        l.stage_b(b(&[], &[], &[], &[]), None, LIMITS);
        assert_eq!(l.find("A", &[]).unwrap().status, OrderStatus::Cancelled);
    }

    /// 留档按 FIFO 淘汰到上限。
    #[test]
    fn finalized_cache_is_bounded_fifo() {
        let mut l = OrderLedger::default();
        let limits = LedgerLimits {
            finalized: 2,
            trade: 50_000,
        };
        for id in ["A", "B", "C"] {
            let expired = [order(id, OrderStatus::Expired, 0)];
            l.stage_a(a(&[], &[], &[], &expired), None, limits);
        }
        assert!(l.find("A", &[]).is_none());
        assert!(l.find("B", &[]).is_some());
        assert!(l.find("C", &[]).is_some());
    }

    /// 状态指纹按 FIFO 淘汰到上限后, 同一状态可再次通知。
    #[test]
    fn order_event_keys_are_bounded() {
        let mut keys: BoundedSet<u32> = BoundedSet::default();
        assert!(keys.insert(1, 2));
        assert!(keys.insert(2, 2));
        assert!(!keys.insert(1, 2));
        assert!(keys.insert(3, 2));
        assert!(keys.insert(1, 2), "1 已被淘汰");
    }

    /// 多槽位归属: 不在已登记槽位的 owner 归默认槽。
    #[test]
    fn ownership_filters_other_slots() {
        let slots: HashSet<String> = ["alpha", "beta"].iter().map(|s| s.to_string()).collect();
        let beta = SlotOwnership::new("beta".into(), "alpha".into(), slots.clone()).unwrap();
        let alpha = SlotOwnership::new("alpha".into(), "alpha".into(), slots).unwrap();
        let new = [
            owned("A", Some("beta")),
            owned("B", Some("alpha")),
            owned("C", None),
            owned("D", Some("ghost")),
            owned("E", Some(" beta ")),
            owned("F", Some("")),
        ];
        let mut lb = OrderLedger::default();
        let (ev, _) = lb.stage_b(b(&new, &[], &[], &[]), Some(&beta), LIMITS);
        assert_eq!(
            ev.iter().map(|e| e.order.id.as_str()).collect::<Vec<_>>(),
            vec!["A", "E"]
        );
        let mut la = OrderLedger::default();
        let (ev, _) = la.stage_b(b(&new, &[], &[], &[]), Some(&alpha), LIMITS);
        assert_eq!(
            ev.iter().map(|e| e.order.id.as_str()).collect::<Vec<_>>(),
            vec!["B", "C", "D", "F"]
        );
    }

    /// 多槽位归属同样过滤成交(原 tests/test_multi_slot_ownership.py::
    /// test_orders_without_owner_go_to_default_slot_only 的成交部分)。
    #[test]
    fn ownership_filters_trades() {
        let slots: HashSet<String> = ["main", "b"].iter().map(|s| s.to_string()).collect();
        let main = SlotOwnership::new("main".into(), "main".into(), slots.clone()).unwrap();
        let slot_b = SlotOwnership::new("b".into(), "main".into(), slots).unwrap();
        let mut broker = trade("t-broker", "broker");
        broker.owner_strategy_id = None;
        let mut mine_b = trade("t-b", "mine-b");
        mine_b.owner_strategy_id = Some("b".to_string());
        let trades = [broker, mine_b];
        let mut lm = OrderLedger::default();
        let (_, out) = lm.stage_b(b(&[], &[], &trades, &[]), Some(&main), LIMITS);
        assert_eq!(
            out.iter().map(|t| t.order_id.as_str()).collect::<Vec<_>>(),
            vec!["broker"]
        );
        let mut lb = OrderLedger::default();
        let (_, out) = lb.stage_b(b(&[], &[], &trades, &[]), Some(&slot_b), LIMITS);
        assert_eq!(
            out.iter().map(|t| t.order_id.as_str()).collect::<Vec<_>>(),
            vec!["mine-b"]
        );
    }

    /// 单槽位不过滤(原 tests/test_multi_slot_ownership.py::test_single_slot_receives_all_orders)。
    #[test]
    fn single_slot_has_no_ownership_filter() {
        let slots: HashSet<String> = ["alpha"].iter().map(|s| s.to_string()).collect();
        assert!(SlotOwnership::new("alpha".into(), "alpha".into(), slots).is_none());
        let mut l = OrderLedger::default();
        let new = [owned("x", None), owned("y", Some("other"))];
        let (ev, _) = l.stage_b(b(&new, &[], &[], &[]), None, LIMITS);
        assert_eq!(
            ev.iter().map(|e| e.order.id.as_str()).collect::<Vec<_>>(),
            vec!["x", "y"]
        );
    }

    /// find: 账本在途 → 上下文在途 → 终态留档。
    #[test]
    fn find_looks_up_known_then_active_then_finalized() {
        let mut l = OrderLedger::default();
        let new = [order("A", OrderStatus::New, 0)];
        l.stage_b(b(&new, &[], &[], &[]), None, LIMITS);
        assert_eq!(l.find("A", &[]).unwrap().status, OrderStatus::New);
        let active = [order("Z", OrderStatus::Submitted, 0)];
        assert_eq!(l.find("Z", &active).unwrap().status, OrderStatus::Submitted);
        assert!(l.find("nope", &active).is_none());
    }

    /// 存档只保存在途订单与成交指纹: 往返后不重复通知在途单、不重复回调旧成交,
    /// 但状态指纹与 on_reject 记录不保留。
    #[test]
    fn snapshot_roundtrip_keeps_known_and_trade_keys_only() {
        let mut l = OrderLedger::default();
        let active = [
            order("A", OrderStatus::New, 0),
            order("B", OrderStatus::New, 0),
        ];
        l.stage_b(b(&[], &active, &[trade("t1", "x")], &[]), None, LIMITS);
        let rejected = [order("R", OrderStatus::Rejected, 0)];
        l.stage_a(a(&[], &[], &rejected, &[]), None, LIMITS);
        let bytes = rmp_serde::to_vec(&l.snapshot()).unwrap();
        let mut r = OrderLedger::restore(rmp_serde::from_slice(&bytes).unwrap());
        let (ev, out) = r.stage_b(b(&[], &active, &[trade("t1", "x")], &[]), None, LIMITS);
        assert!(ev.is_empty(), "在途单不重复通知");
        assert!(out.is_empty(), "旧成交不重复回调");
        let ev = r.stage_a(a(&[], &[], &rejected, &[]), None, LIMITS);
        assert_eq!(
            sig(&ev),
            vec![st("R", OrderStatus::Rejected, true)],
            "指纹与 on_reject 记录不随存档"
        );
        assert_eq!(
            r.snapshot()
                .known
                .iter()
                .map(|o| o.id.as_str())
                .collect::<Vec<_>>(),
            vec!["A", "B", "R"]
        );
    }
}
