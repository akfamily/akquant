use super::shared::{
    apply_execution_report, effective_execution_policy_for_order, flush_accumulated_trades,
};
use crate::calendar::trading_calendar::CalendarError;
use crate::context::EngineContext;
use crate::data::FeedAction;
use crate::engine::Engine;
use crate::event::Event;
use crate::model::{Bar, Order, OrderStatus, TradingSession};
use crate::pipeline::processor::{Processor, ProcessorResult};
use chrono::NaiveDate;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::{HashMap, HashSet};

pub struct DataProcessor {
    last_timestamp: i64,
    finalized_timestamp: i64,
    seen_symbols: HashSet<String>,
    // Symbols with a *tradable* bar this slice (volume > 0). `seen_symbols`
    // tracks "did a bar appear at all" (drives synthetic-bar backfill); this
    // tracks "was it matchable". A suspended symbol emits a volume=0 placeholder
    // bar — present in `seen_symbols` but absent here — so it must not look
    // tradable to the terminal-state path (issue #334).
    tradable_symbols: HashSet<String>,
    current_symbol_events: HashMap<String, Event>,
}

impl Default for DataProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl DataProcessor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            last_timestamp: 0,
            finalized_timestamp: 0,
            seen_symbols: HashSet::new(),
            tradable_symbols: HashSet::new(),
            current_symbol_events: HashMap::new(),
        }
    }

    fn fill_missing_bars(&self, engine: &Engine) {
        if self.last_timestamp == 0 {
            return;
        }
        if let Ok(mut buffer) = engine.history_buffer.write() {
            let prices = engine.last_prices.read().expect("last_prices 读锁被污染");
            for symbol in engine.instruments.keys() {
                // 跑 tick 流的 symbol 不补合成 bar: 双流下合成 bar 打区间结束戳,
                // 与 tick 时间戳天然错开, 于是每个 bar 事件都会让其余 symbol 显得
                // "本批次缺失"。此时用 last_prices(成交价)合成退化 bar 写进 bar
                // 序列, 等于用 tick 价冒充 bar, 且完全静默。该 symbol 的 bar 应当
                // 只来自真实聚合, 故此处直接跳过。
                // NOTE: 该排除条件具有永久性: 一旦 symbol 曾经产生过一个 tick 事件,
                // has_tick_history 的记录就再也不会清零, 此后该 symbol 永远不再获得
                // fill_missing_bars 的合成补偿——若行情源后来切换为纯 bar 模式, 其
                // 停牌缺失 bar 将不再被填补。
                if buffer.has_tick_history(symbol) {
                    continue;
                }
                if !self.seen_symbols.contains(symbol)
                    && let Some(&last_price) = prices.get(symbol)
                {
                    // Create synthetic bar
                    let bar = Bar {
                        timestamp: self.last_timestamp,
                        symbol: symbol.clone(),
                        open: last_price,
                        high: last_price,
                        low: last_price,
                        close: last_price,
                        volume: Decimal::ZERO,
                        extra: HashMap::default(),
                        freq: None,
                        interval_start: None,
                    };
                    buffer.update(&bar);
                }
            }
        }
    }

    fn reject_missing_symbol_orders(&self, engine: &mut Engine) {
        if self.last_timestamp == 0 {
            return;
        }

        let rejected_orders: Vec<Order> = engine
            .state
            .order_manager
            .active_orders
            .iter()
            .filter(|order| {
                matches!(order.status, OrderStatus::New | OrderStatus::Submitted)
                    // `<=` (not `==`): a timer-created order carries the timer's
                    // timestamp (e.g. 09:00) while the day's bar timestamp is
                    // 00:00, so `== last_timestamp` never matched timer orders —
                    // a suspended symbol's order then lived forever as New,
                    // piling into free-margin projection until trading froze
                    // (issue #329). Rejecting once its matchable slice has passed
                    // without a bar gives it a terminal state instead.
                    && order.created_at <= self.last_timestamp
                    // Was there a *tradable* (volume>0) bar this slice, not merely
                    // any bar? A suspended symbol emits a volume=0 placeholder bar
                    // that the matcher skips (execution/common.rs volume<=0 => None)
                    // but that still landed in `seen_symbols`, so the old check let
                    // the order live forever until the symbol resumed (issue #334).
                    // Keying off tradability aligns this terminal-state path with
                    // the matcher and generalizes the #329 "no bar at all" case.
                    && !self.tradable_symbols.contains(&order.symbol)
                    && effective_execution_policy_for_order(engine, order).bar_offset == 1
            })
            .cloned()
            .map(|mut order| {
                order.status = OrderStatus::Rejected;
                order.updated_at = self.last_timestamp;
                // Distinguish the two untradable causes so audit/log analysis is
                // not misled: a symbol present in `seen_symbols` did emit a bar
                // this slice — it was just not tradable (volume=0 suspension,
                // issue #334); one absent from it had no market data at all
                // (issue #329).
                order.reject_reason = if self.seen_symbols.contains(&order.symbol) {
                    format!(
                        "Symbol {} not tradable (zero volume, suspension) at execution timestamp {}",
                        order.symbol, self.last_timestamp
                    )
                } else {
                    format!(
                        "Missing market data for symbol {} at execution timestamp {}",
                        order.symbol, self.last_timestamp
                    )
                };
                order
            })
            .collect();

        for order in rejected_orders {
            engine.execution_model.on_cancel(&order.id);
            let _ = engine
                .event_manager
                .send(Event::ExecutionReport(order, None));
        }
    }

    fn finalize_current_timestamp(&mut self, engine: &mut Engine, py: Python<'_>) {
        if self.last_timestamp == 0 || self.finalized_timestamp == self.last_timestamp {
            return;
        }
        let current_events: Vec<Event> = self.current_symbol_events.values().cloned().collect();
        if !current_events.is_empty() {
            // 注意:此读锁会跨越下面的 finalize_timestamp 调用,自定义撮合器
            // (register_custom_matcher)会经此路径调回用户 Python 代码。
            // 安全性同 core.rs Engine.last_prices 字段注释:写锁只有 3 处
            // 且都要求 &mut Engine,run() 期间该借用被占用,再入会
            // PyBorrowMutError 而非阻塞。
            let prices = engine.last_prices.read().expect("last_prices 读锁被污染");
            let ctx = EngineContext {
                instruments: &engine.instruments,
                portfolio: &engine.state.portfolio,
                last_prices: &prices,
                trade_tracker: &engine.state.order_manager.trade_tracker,
                market_model: engine.market_manager.model.as_ref(),
                execution_policy_core: engine.execution_policy_core(),
                bar_index: engine.bar_count,
                current_time: self.last_timestamp,
                session: engine.clock.session,
                active_orders: &engine.state.order_manager.active_orders,
                risk_config: &engine.risk_manager.config,
                timezone_name: engine.timezone_name.as_deref(),
                timezone_offset: engine.timezone_offset,
            };
            let reports = engine
                .execution_model
                .finalize_timestamp(&current_events, &ctx);
            drop(ctx);
            drop(prices);
            let mut trades_to_process = Vec::new();
            let mut oco_suppressed_fill_order_ids: HashSet<String> = HashSet::new();
            for report in reports {
                if let Event::ExecutionReport(order, trade) = report {
                    apply_execution_report(
                        engine,
                        py,
                        order,
                        trade,
                        &mut oco_suppressed_fill_order_ids,
                        &mut trades_to_process,
                    );
                }
            }
            flush_accumulated_trades(engine, &mut trades_to_process);
        }
        self.fill_missing_bars(engine);
        self.reject_missing_symbol_orders(engine);
        self.finalized_timestamp = self.last_timestamp;
        self.current_symbol_events.clear();
    }
}

impl Processor for DataProcessor {
    fn process(
        &mut self,
        engine: &mut Engine,
        py: Python<'_>,
        _strategy: &Bound<'_, PyAny>,
    ) -> PyResult<ProcessorResult> {
        // 墙钟兜底: 到点即结束, 不依赖行情事件到达。放在 feed 等待**之前**,
        // 因为等待本身可能长达 1 秒, 而行情停摆时 `Wait` 会无限循环下去。
        if let Some(deadline) = engine.session_deadline_ns
            && chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0) >= deadline
        {
            self.finalize_current_timestamp(engine, py);
            return Ok(ProcessorResult::Break);
        }

        let next_timer_time = engine.timers.peek().map(|t| t.timestamp);
        // 把内部事件通道的接收端交给 feed 一起等: 外部线程注入的 OrderRequest
        // 可立刻打断行情等待(否则最坏要等满 1 秒 timeout)。ChannelProcessor 是
        // pipeline 首位, Wait→Loop 后即被它排空。
        let wakeup = engine.event_manager.receiver();
        let action = engine.state.feed.next_action(next_timer_time, py, wakeup);

        match action {
            FeedAction::Wait => Ok(ProcessorResult::Loop),
            FeedAction::End => {
                self.finalize_current_timestamp(engine, py);
                Ok(ProcessorResult::Break)
            }
            FeedAction::Timer(_timestamp) => {
                if let Some(timer) = engine.timers.pop() {
                    // 引擎内部的日终结算定时器: 不推进 clock、不设 current_event、
                    // 不转给策略。
                    if timer.is_day_close() {
                        if timer.timestamp > self.last_timestamp {
                            self.finalize_current_timestamp(engine, py);
                        }
                        let day = NaiveDate::parse_from_str(
                            &timer.payload[crate::model::timer::DAY_CLOSE_TIMER_PREFIX.len()..],
                            "%Y-%m-%d",
                        )
                        .expect("内部定时器格式");
                        if engine.last_settled_trading_day != Some(day) {
                            run_day_close_settlement(engine, py, day, timer.timestamp - 1);
                            engine.last_settled_trading_day = Some(day);
                        }
                        return Ok(ProcessorResult::Loop);
                    }
                    // Invariant: finalize the current timestamp before any timer
                    // advances the clock past it. Previously only framework
                    // cross-section timers did this; user timers
                    // (add_daily_timer / schedule) skipped it, so same-cycle
                    // deferred orders were stranded — the timer's on_event wiped
                    // the slice-tracking registry before the slice was ever
                    // finalized, leaving phantom New buys that zeroed out free
                    // margin and froze trading permanently (issue #329). Framework
                    // cross-section timers are scheduled at source_ts + 1, so they
                    // always satisfy this condition and keep their prior behavior.
                    if timer.timestamp > self.last_timestamp {
                        self.finalize_current_timestamp(engine, py);
                    }
                    let local_time = engine.local_time_from_ns(timer.timestamp);
                    let session = engine.market_manager.get_session_status(local_time);
                    engine.clock.update(timer.timestamp, session);
                    if engine.force_session_continuous {
                        engine.clock.session = TradingSession::Continuous;
                    }
                    engine.current_event = Some(Event::Timer(timer));
                }
                Ok(ProcessorResult::Next)
            }
            FeedAction::Event(event) => {
                let event = *event;

                // 标的白名单: 用户显式传了 symbols 时只放行白名单内的标的。
                // 返回 Loop = 丢弃本事件、去取下一个, 与下面
                // `timestamp <= engine.snapshot_time` 的处理方式一致。
                // 放在最前面是有意的: 白名单外的事件不该影响 last_timestamp、
                // seen_symbols、bar_count 等任何时间片状态, 否则它们仍会
                // 参与批次边界与进度计算。
                if let Some(whitelist) = &engine.symbol_whitelist {
                    let event_symbol = match &event {
                        Event::Bar(b) => Some(b.symbol.as_str()),
                        Event::Tick(t) => Some(t.symbol.as_str()),
                        _ => None,
                    };
                    if let Some(symbol) = event_symbol
                        && !whitelist.contains(symbol)
                    {
                        // 只计数, 不逐事件打日志(见 `whitelist_filtered_event_count`
                        // 字段文档) —— 会话结束时 `Engine::run` 汇总报一条 info。
                        engine.whitelist_filtered_event_count += 1;
                        return Ok(ProcessorResult::Loop);
                    }
                }

                let timestamp = match &event {
                    Event::Bar(b) => b.timestamp,
                    Event::Tick(t) => t.timestamp,
                    _ => 0,
                };

                if timestamp <= engine.snapshot_time {
                    return Ok(ProcessorResult::Loop);
                }

                if self.last_timestamp != 0 && timestamp > self.last_timestamp {
                    self.finalize_current_timestamp(engine, py);
                    self.seen_symbols.clear();
                    self.tradable_symbols.clear();
                    self.current_symbol_events.clear();

                    if engine.is_active_timestamp(timestamp) {
                        engine.bar_count += 1;
                        if let Some(pb) = &engine.progress_bar {
                            pb.inc(1);
                        }
                        let mut progress_payload = HashMap::new();
                        progress_payload.insert("processed", engine.bar_count.to_string());
                        progress_payload.insert("total", engine.progress_total_steps.to_string());
                        engine.emit_stream_event(py, "progress", None, "info", progress_payload);
                    }
                }
                self.last_timestamp = timestamp;

                // Update Market Manager (Session)
                let session = engine
                    .market_manager
                    .get_session_status(engine.local_time_from_ns(timestamp));

                engine.clock.update(timestamp, session);
                if engine.force_session_continuous {
                    engine.clock.session = TradingSession::Continuous;
                }

                // 开日: 第一个属于新会话交易日的(活跃)事件。日终结算不在这里做,
                // 而是挂在该交易日收盘时刻 + 1ns 的内部定时器上(见 Timer 分支)。
                if engine.is_active_timestamp(timestamp) {
                    let day = engine.trading_day_for(timestamp)?;
                    if engine.current_trading_day != Some(day) {
                        // 上一交易日未结算(实盘漏触发等): 先补结算
                        if let Some(prev) = engine.current_trading_day
                            && engine.last_settled_trading_day != Some(prev)
                        {
                            let close = engine.current_day_span.map_or(timestamp, |s| s.1);
                            run_day_close_settlement(engine, py, prev, close);
                            engine.last_settled_trading_day = Some(prev);
                        }
                        // 公司行动(拆股/分红): 处理日期落在 (上次处理到的日期, day] 的全部
                        // 行动, 落在非交易日(周末、节假日)的顺延到其后第一个交易日开日。
                        // 首个交易日从首个事件的本地日期起算(不回溯数据开始前的行动)。
                        let corp_from = match engine.last_corp_action_date {
                            Some(last) => last.succ_opt().unwrap_or(last),
                            None => engine.local_date_from_ns(timestamp).min(day),
                        };
                        engine.corporate_action_manager.process_range(
                            corp_from,
                            day,
                            &mut engine.state.portfolio,
                            &mut engine.state.order_manager.trade_tracker,
                        );
                        engine.last_corp_action_date = Some(day);
                        engine.current_trading_day = Some(day);
                        engine.current_day_span =
                            Some(session_day_span_for(engine, day, timestamp)?);
                        engine.last_event_ts_in_day = None;
                    }
                    if engine.scheduled_day_close != Some(day)
                        && engine.last_settled_trading_day != Some(day)
                        && let Some((_, end)) = engine.current_day_span
                    {
                        engine.timers.push(crate::model::Timer {
                            timestamp: end + 1,
                            payload: format!(
                                "{}{}",
                                crate::model::timer::DAY_CLOSE_TIMER_PREFIX,
                                day
                            ),
                        });
                        engine.scheduled_day_close = Some(day);
                    }
                    engine.last_event_ts_in_day = Some(timestamp);
                }

                if let Event::Bar(ref b) = event {
                    self.seen_symbols.insert(b.symbol.clone());
                    if b.volume > Decimal::ZERO {
                        self.tradable_symbols.insert(b.symbol.clone());
                    }
                    self.current_symbol_events
                        .insert(b.symbol.clone(), Event::Bar(b.clone()));
                    // Update History Buffer, 并把基础 bar 喂给窗口聚合器。闭合的窗口 bar
                    // 只写历史 + 暂存待派发: 不进 feed、不成为 Event、不动 last_prices。
                    if let Ok(mut buffer) = engine.history_buffer.write() {
                        buffer.update(b);
                        let closed = engine
                            .window_aggregator
                            .write()
                            .expect("window_aggregator 写锁被污染")
                            .update(b);
                        for window_bar in &closed {
                            buffer.update_window(window_bar);
                        }
                        engine.pending_window_bars.extend(closed);
                    }
                    // println!("DataProcessor: Bar Symbol={}, TS={}", b.symbol, b.timestamp);
                } else if let Event::Tick(ref t) = event {
                    self.current_symbol_events
                        .insert(t.symbol.clone(), Event::Tick(t.clone()));
                    // tick 是本时间戳该 symbol 的真实行情, 必须登记为"已见"——否则
                    // fill_missing_bars 会把它当成缺失并合成一根退化 bar, 导致每个
                    // tick 被双写历史(一次 update_tick, 一次合成 bar)。
                    self.seen_symbols.insert(t.symbol.clone());
                    // tick 也写入历史缓冲区: 否则纯 tick 回测下 get_history 恒为空,
                    // 且是静默的(不报错也不工作)。tick 以退化 bar 写入。
                    if let Ok(mut buffer) = engine.history_buffer.write() {
                        buffer.update_tick(t);
                    }
                }

                engine.current_event = Some(event);
                if engine.is_active_timestamp(timestamp)
                    && let Some(current) = engine.current_event.clone()
                {
                    match current {
                        Event::Bar(b) => {
                            let mut payload = HashMap::new();
                            payload.insert("timestamp", b.timestamp.to_string());
                            payload.insert("close", b.close.to_string());
                            payload.insert("volume", b.volume.to_string());
                            engine.emit_stream_event(
                                py,
                                "bar",
                                Some(b.symbol.as_str()),
                                "info",
                                payload,
                            );
                        }
                        Event::Tick(t) => {
                            let mut payload = HashMap::new();
                            payload.insert("timestamp", t.timestamp.to_string());
                            payload.insert("price", t.price.to_string());
                            payload.insert("volume", t.volume.to_string());
                            engine.emit_stream_event(
                                py,
                                "tick",
                                Some(t.symbol.as_str()),
                                "info",
                                payload,
                            );
                        }
                        _ => {}
                    }
                }
                Ok(ProcessorResult::Next)
            }
        }
    }
}

/// 交易日 `day` 的区间 `(上一交易日收盘, 本交易日收盘]`。本交易日收盘取不到(如日历
/// 越界)时报错; 只有"上一交易日"不在用户日历里(日历恰从首个数据日开始)时, 下界
/// 退化为本事件时间戳 - 1 —— 下界只用于 `trading_day_for` 的区间缓存, 时间单调前进,
/// 不会再有更早的事件需要命中它。
fn session_day_span_for(engine: &Engine, day: NaiveDate, ts: i64) -> PyResult<(i64, i64)> {
    let resolver = engine.session_resolver.as_ref().expect("run() 已构造");
    let end = resolver
        .session_day_end(day)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    match resolver.session_day_span(day) {
        Ok(span) => Ok(span),
        Err(CalendarError::OutOfRange { .. }) => Ok((ts - 1, end)),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

/// 交易日 `day` 的日终结算：到期 → 利息与强平 → T+1 → Day 单过期 → 流事件 → 快照。
/// `close_ts` 为会话收盘时刻（快照与流事件时间戳）。
fn run_day_close_settlement(engine: &mut Engine, py: Python<'_>, day: NaiveDate, close_ts: i64) {
    // Settlement Manager (T+1, Option Expiry, Day Order Expiry)
    let mut expired_orders = Vec::new();
    // NextOpen(bar_offset == 1) 的 Day 单若创建于本交易日最后一个已处理行情事件
    // 当时或之后, 唯一的撮合机会是下一交易日的第一个事件, 本次结算不过期。
    // 同周期(bar_offset == 0)单在创建时的事件上已有撮合机会, 照常过期。
    let exempt_day_orders: HashSet<String> = engine
        .state
        .order_manager
        .active_orders
        .iter()
        .filter(|order| {
            order.time_in_force == crate::model::TimeInForce::Day
                && effective_execution_policy_for_order(engine, order).bar_offset == 1
                && engine
                    .last_event_ts_in_day
                    .is_none_or(|last| order.created_at >= last)
        })
        .map(|order| order.id.clone())
        .collect();
    // 到期结算必须先于日终结算(利息/强平/T+1/挂单过期): 强平按当时的
    // 组合选仓, 而它生成的强平成交要到 apply_execution_report 才落地;
    // 若先强平后到期, 到期会立刻清掉持仓与 tracker 库存, 排队的强平成交
    // 再平一次就留下反向幻影仓位。强平必须看到到期后的组合。
    // 单独调用是因为它要写 TradeTracker(合成平仓成交), 而
    // SettlementContext 只持有只读引用。
    let expiry_prices = engine
        .last_prices
        .read()
        .expect("last_prices 读锁被污染")
        .clone();
    // settle_expiries 的 date 是"正在进入的交易日"(到期日严格小于它才结算; 也是
    // 到期事件的 trading_date): 收盘时传会话日历上的下一交易日, 结算到期日早于它
    // (含 day 当天与其后的非交易日)的持仓。日历越界(用户日历最后一天)时退化为次日。
    let entering_day = engine
        .session_resolver
        .as_ref()
        .and_then(|r| r.next_session_day(day).ok())
        .or_else(|| day.succ_opt())
        .unwrap_or(day);
    let expiry_settlement = engine.settlement_manager.settle_expiries(
        entering_day,
        &mut engine.state.portfolio,
        &mut engine.state.order_manager.trade_tracker,
        &engine.instruments,
        &expiry_prices,
        &engine.market_manager,
        &engine.risk_manager.config,
        close_ts,
        engine.bar_count,
    );
    let settlement_prices = engine.last_prices.read().expect("last_prices 读锁被污染");
    let settlement_ctx = crate::settlement::manager::SettlementContext {
        date: day,
        instruments: &engine.instruments,
        last_prices: &settlement_prices,
        market_manager: &engine.market_manager,
        trade_tracker: &engine.state.order_manager.trade_tracker,
        risk_config: &engine.risk_manager.config,
        timestamp: close_ts,
        bar_index: engine.bar_count,
        default_strategy_id: engine.default_strategy_id.clone(),
        exempt_day_orders: &exempt_day_orders,
        accrual_days: 1,
    };
    let settlement_outcome = engine.settlement_manager.process_daily_settlement(
        &mut engine.state.portfolio,
        &mut engine.state.order_manager.active_orders,
        &mut expired_orders,
        &settlement_ctx,
    );
    drop(settlement_ctx);
    drop(settlement_prices);
    engine.recent_expiry_events = expiry_settlement
        .events
        .iter()
        .map(crate::context::ExpiryEvent::from_executed)
        .collect();
    engine.margin_daily_interest = settlement_outcome.daily_interest;
    engine.margin_accrued_interest += settlement_outcome.daily_interest;
    if settlement_outcome.daily_interest > Decimal::ZERO {
        let mut settlement_payload = HashMap::new();
        settlement_payload.insert("date", day.to_string());
        settlement_payload.insert(
            "daily_interest",
            settlement_outcome.daily_interest.to_string(),
        );
        settlement_payload.insert(
            "accrued_interest",
            engine.margin_accrued_interest.to_string(),
        );
        engine.emit_stream_event(py, "settlement", None, "info", settlement_payload);
    }
    if settlement_outcome.forced_liquidation {
        let liquidated_symbols = settlement_outcome.liquidated_symbols.clone();
        let priority = engine.risk_manager.config.liquidation_priority.clone();
        engine
            .statistics_manager
            .record_liquidation_audit(crate::analysis::LiquidationAudit {
                timestamp: close_ts,
                date: day.to_string(),
                daily_interest: settlement_outcome
                    .daily_interest
                    .to_f64()
                    .unwrap_or_default(),
                liquidated_count: liquidated_symbols.len(),
                liquidated_symbols: liquidated_symbols.clone(),
                priority: priority.clone(),
            });
        let mut risk_payload = HashMap::new();
        risk_payload.insert("date", day.to_string());
        risk_payload.insert("liquidated_count", liquidated_symbols.len().to_string());
        risk_payload.insert("liquidated_symbols", liquidated_symbols.join(","));
        risk_payload.insert("priority", priority);
        if let Some(owner_strategy_id) = engine
            .default_strategy_id
            .clone()
            .filter(|text| !text.trim().is_empty())
        {
            risk_payload.insert("owner_strategy_id", owner_strategy_id);
        }
        engine.emit_stream_event(py, "risk", None, "warn", risk_payload);
    }
    // 强平成交就地落地(与 finalize_current_timestamp 落地撮合回报的方式相同),
    // 而不是投进事件通道等下一轮 ChannelProcessor: 收盘快照要反映强平后的组合。
    // 回调(on_order/on_trade)仍按原路径在下一个回调入口派发。
    let mut liquidation_trades = Vec::new();
    let mut oco_suppressed_fill_order_ids: HashSet<String> = HashSet::new();
    for event in settlement_outcome.forced_liquidation_events {
        if let Event::ExecutionReport(order, trade) = event {
            apply_execution_report(
                engine,
                py,
                order,
                trade,
                &mut oco_suppressed_fill_order_ids,
                &mut liquidation_trades,
            );
        }
    }
    flush_accumulated_trades(engine, &mut liquidation_trades);
    engine.emit_expiry_stream_events(py);

    for o in expired_orders {
        // Tell the execution client to drop the order too. Daily
        // settlement only flips the order-manager copy to Expired;
        // without this the simulated client keeps it New in its own
        // book and a later bar crossing the limit fills the zombie
        // long after it should have died (issue #329 hygiene item).
        // Mirrors the reject path above (reject_missing_symbol_orders).
        engine.execution_model.on_cancel(&o.id);
        engine.state.order_manager.orders.push(o);
    }

    // 当日收盘快照: 记在全部结算步骤之后, 反映结算后的组合(spec §4.2)。
    let prices = engine.last_prices.read().expect("last_prices 读锁被污染");
    engine.statistics_manager.record_snapshot(
        close_ts,
        &engine.state.portfolio,
        &engine.instruments,
        &prices,
        &engine.state.order_manager.trade_tracker,
        &engine.risk_manager.config,
    );
}
