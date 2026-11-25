use crate::engine::Engine;
use crate::event::Event;
use crate::model::{ExecutionPolicyCore, Order, OrderStatus, PriceBasis, TemporalPolicy, Trade};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

pub(crate) fn effective_execution_policy_for_order(engine: &Engine, order: &Order) -> ExecutionPolicyCore {
    order
        .fill_policy_override
        .unwrap_or(engine.execution_policy_core())
}

pub(crate) fn should_run_phase_for_engine_default(
    phase: &ExecutionPhase,
    engine: &Engine,
    event: Option<&Event>,
) -> bool {
    should_run_phase_for_current_event(phase, engine.execution_policy_core(), event)
}

pub(crate) fn should_run_phase_for_current_event(
    phase: &ExecutionPhase,
    policy: ExecutionPolicyCore,
    event: Option<&Event>,
) -> bool {
    match phase {
        ExecutionPhase::PreStrategy => policy.bar_offset == 1,
        ExecutionPhase::PostStrategy => {
            if !(policy.price_basis == PriceBasis::Close && policy.bar_offset == 0) {
                return false;
            }
            match event {
                Some(Event::Bar(_) | Event::Tick(_)) => true,
                Some(Event::Timer(timer)) => {
                    // 框架与引擎生命周期定时器(截面定时器除外)不撮合同周期单。
                    matches!(policy.temporal, TemporalPolicy::SameCycle)
                        && !timer.is_lifecycle()
                        && (!timer.payload.starts_with("__framework_")
                            || timer
                                .payload
                                .starts_with("__framework_cross_section__|"))
                }
                _ => false,
            }
        }
    }
}

pub(crate) fn has_orders_matching_phase(engine: &Engine, phase: &ExecutionPhase, orders: &[Order]) -> bool {
    let event = engine.current_event.as_ref();
    orders.iter().any(|order| {
        should_run_phase_for_current_event(
            phase,
            effective_execution_policy_for_order(engine, order),
            event,
        )
    })
}
pub(crate) fn flush_accumulated_trades(
    engine: &mut Engine,
    py: Python<'_>,
    trades_to_process: &mut Vec<Trade>,
) {
    if trades_to_process.is_empty() {
        return;
    }
    // bracket 出场子单只减仓: 成交入账后, 持仓已了结的品种上不再有持仓可平的子单
    // 立即撤销(在下一次撮合之前)。没有挂着的子单时(常态)不收集品种。
    let traded: Vec<(String, i64)> = if engine
        .state
        .order_manager
        .active_orders
        .iter()
        .any(Order::is_bracket_exit)
    {
        let mut latest: HashMap<&str, i64> = HashMap::new();
        for trade in trades_to_process.iter() {
            let ts = latest.entry(trade.symbol.as_str()).or_insert(trade.timestamp);
            *ts = (*ts).max(trade.timestamp);
        }
        let mut traded: Vec<(String, i64)> = latest
            .into_iter()
            .map(|(symbol, ts)| (symbol.to_string(), ts))
            .collect();
        traded.sort();
        traded
    } else {
        Vec::new()
    };
    {
        let prices = engine.last_prices.read().expect("last_prices 读锁被污染");
        engine.state.order_manager.process_trades(
            std::mem::take(trades_to_process),
            &mut engine.state.portfolio,
            &engine.instruments,
            &engine.risk_manager.config,
            engine.market_manager.model.as_ref(),
            &engine.history_buffer,
            &prices,
        );
    }
    for (symbol, ts) in traded {
        cancel_bracket_exits_without_position(engine, py, &symbol, ts);
    }
}

/// 回测与 paper: 撤掉 `symbol` 上已无持仓可平的 bracket 出场子单(持仓按账户合计, 见
/// [`crate::order_manager::OrderManager::cancel_bracket_exits_without_position`]):
/// 通知撮合器、改订单状态、发 order 流事件。策略的 `on_order(Cancelled)` 与 OCO
/// 同伴撤单一样由订单账本在下一个回调入口派发, 与使持仓归零的那笔成交同一批:
/// 账本先派发订单状态(按挂单顺序)、再派发成交, 所以它在那笔成交的 `on_trade`
/// 之前。
pub(crate) fn cancel_bracket_exits_without_position(
    engine: &mut Engine,
    py: Python<'_>,
    symbol: &str,
    updated_at: i64,
) {
    // 柜台路径不处理: 那里撤单要等柜台回报, 引擎先行把订单记为已撤会与柜台不一致。
    if engine.execution_model.is_live() {
        return;
    }
    let position = engine
        .state
        .portfolio
        .positions
        .get(symbol)
        .copied()
        .unwrap_or_default();
    let cancelled = engine
        .state
        .order_manager
        .cancel_bracket_exits_without_position(symbol, position, updated_at);
    for order in cancelled {
        engine.execution_model.on_cancel(&order.id);
        emit_cancelled_order_event(engine, py, &order);
    }
}

fn emit_cancelled_order_event(engine: &mut Engine, py: Python<'_>, order: &Order) {
    let mut cancel_payload = HashMap::new();
    cancel_payload.insert("order_id", order.id.clone());
    cancel_payload.insert("status", format!("{:?}", order.status));
    cancel_payload.insert("filled_qty", order.filled_quantity.to_string());
    cancel_payload.insert("symbol", order.symbol.clone());
    cancel_payload.insert(
        "owner_strategy_id",
        order.owner_strategy_id.clone().unwrap_or_default(),
    );
    engine.emit_stream_event(py, "order", Some(order.symbol.as_str()), "info", cancel_payload);
}

pub(crate) fn apply_execution_report(
    engine: &mut Engine,
    py: Python<'_>,
    order: Order,
    trade: Option<Trade>,
    oco_suppressed_fill_order_ids: &mut HashSet<String>,
    trades_to_process: &mut Vec<Trade>,
) {
    if order.status == OrderStatus::Filled && oco_suppressed_fill_order_ids.contains(&order.id) {
        return;
    }

    let report_order_id = order.id.clone();
    let report_status = order.status;
    let report_updated_at = order.updated_at;
    engine.state.order_manager.on_execution_report(order);
    // 性能: 原实现 get_all_orders() 会克隆整张订单表再 find, 每个成交回报做一次 →
    // O(N²) 且是全程头号热点. 改为 find_order(与 get_all_orders 相同的 orders-first
    // 优先级, 历史订单走索引), 只克隆命中的那一个订单.
    let updated_order = engine
        .state
        .order_manager
        .find_order(&report_order_id)
        .cloned();
    if let Some(order_snapshot) = updated_order {
        if order_snapshot.status == OrderStatus::Rejected {
            engine
                .state
                .order_manager
                .current_step_rejected_orders
                .push(order_snapshot.clone());
        }
        let mut order_payload = HashMap::new();
        order_payload.insert("order_id", order_snapshot.id.clone());
        order_payload.insert("status", format!("{:?}", order_snapshot.status));
        order_payload.insert("filled_qty", order_snapshot.filled_quantity.to_string());
        order_payload.insert("symbol", order_snapshot.symbol.clone());
        order_payload.insert(
            "owner_strategy_id",
            order_snapshot.owner_strategy_id.clone().unwrap_or_default(),
        );
        engine.emit_stream_event(
            py,
            "order",
            Some(order_snapshot.symbol.as_str()),
            "info",
            order_payload,
        );
    }

    // 成交完, 或 IOC 成交一部分后撤掉剩余(回报带成交且已撤销): 都按成交处理 OCO 对手单
    // 与括号单。
    let filled_terminal = report_status == OrderStatus::Filled
        || (report_status == OrderStatus::Cancelled && trade.is_some());
    if filled_terminal {
        let peer_ids = engine
            .state
            .order_manager
            .consume_oco_peer_cancels_on_fill(&report_order_id);
        for peer_id in peer_ids {
            oco_suppressed_fill_order_ids.insert(peer_id.clone());
            engine.execution_model.on_cancel(&peer_id);
            let cancelled_order_snapshot = engine
                .state
                .order_manager
                .cancel_active_order(&peer_id, report_updated_at);
            if let Some(cancelled_order_snapshot) = cancelled_order_snapshot {
                emit_cancelled_order_event(engine, py, &cancelled_order_snapshot);
            }
        }

        // 性能: 同上, 避免克隆整张订单表; 只克隆命中项.
        let filled_order_snapshot = engine
            .state
            .order_manager
            .find_order(&report_order_id)
            .cloned();
        if let Some(filled_order_snapshot) = filled_order_snapshot {
            let bracket_exit_orders = engine
                .state
                .order_manager
                .consume_bracket_activation_on_fill(&filled_order_snapshot);
            for bracket_order in bracket_exit_orders {
                let _ = engine
                    .event_manager
                    .send(Event::OrderRequest(bracket_order));
            }
        }
    }

    if let Some(t) = trade {
        engine.maybe_reset_risk_budget_usage(t.timestamp);
        if engine.risk_budget_use_trade_mode() {
            engine.apply_risk_budget_usage_from_trade(&t);
        }
        engine.apply_strategy_trade_position(&t);
        let mut trade_payload = HashMap::new();
        trade_payload.insert("trade_id", t.id.clone());
        trade_payload.insert("order_id", t.order_id.clone());
        trade_payload.insert("price", t.price.to_string());
        trade_payload.insert("quantity", t.quantity.to_string());
        trade_payload.insert("side", format!("{:?}", t.side));
        trade_payload.insert(
            "position_effect",
            t.position_effect.as_canonical_str().to_string(),
        );
        trade_payload.insert(
            "owner_strategy_id",
            t.owner_strategy_id.clone().unwrap_or_default(),
        );
        engine.emit_stream_event(py, "trade", Some(t.symbol.as_str()), "info", trade_payload);
        trades_to_process.push(t);
    }
}
#[derive(Debug)]
pub enum ExecutionPhase {
    PreStrategy,
    PostStrategy,
}
