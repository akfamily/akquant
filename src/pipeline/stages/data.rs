use super::shared::{
    apply_execution_report, effective_execution_policy_for_order, flush_accumulated_trades,
};
use crate::calendar::trading_calendar::CalendarError;
use crate::context::{EngineContext, SessionView};
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
            // 无解析器(构造失败)时 SessionView 按连续竞价; 配置错误已在 run() 开始时报出。
            let _ = engine.ensure_session_resolver();
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
                trading_day: engine.current_trading_day,
                sessions: SessionView {
                    slice: Some(&*engine.slice_sessions),
                    resolver: engine.session_resolver.as_deref(),
                    instruments: Some(&engine.instruments),
                },
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
        strategy: &Bound<'_, PyAny>,
    ) -> PyResult<ProcessorResult> {
        // 墙钟兜底: 到点即结束, 不依赖行情事件到达。放在 feed 等待**之前**,
        // 因为等待本身可能长达 1 秒, 而行情停摆时 `Wait` 会无限循环下去。
        if let Some(deadline) = engine.session_deadline_ns
            && chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0) >= deadline
        {
            self.finalize_current_timestamp(engine, py);
            return Ok(ProcessorResult::Break);
        }

        // replay 数据结束收尾(见 `Engine::live_replay_close`): 末日结算并派发
        // on_after_trading 后的下一轮结束会话; 最后事件处理完、行情已取空后开始计时,
        // 超时未结算则告警并结束。
        if let Some(close) = engine.live_replay_close {
            if close.settled {
                engine.live_replay_close = None;
                self.finalize_current_timestamp(engine, py);
                return Ok(ProcessorResult::Break);
            }
            if close.last_event_seen {
                let now = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
                match close.deadline_ns {
                    None if engine.state.feed.peek_timestamp().is_none() => {
                        // 末日没有待触发的收盘定时器(该日未开日、不在交易日历内等)时
                        // 等不到结算, 直接结束, 不空等。
                        if !engine.has_day_close_timer(close.day) {
                            log::warn!(
                                "回放数据已结束，但最后一个交易日 {} 没有待触发的日终结算（该日未开日或不在交易日历内），不等待结算，会话直接结束（该日不触发 on_after_trading）；请检查 trading_calendar= 是否覆盖回放数据的日期",
                                close.day
                            );
                            engine.live_replay_close = None;
                            self.finalize_current_timestamp(engine, py);
                            return Ok(ProcessorResult::Break);
                        }
                        if let Some(state) = engine.live_replay_close.as_mut() {
                            state.deadline_ns = Some(
                                now.saturating_add(crate::engine::core::LiveReplayClose::WAIT_NS),
                            );
                        }
                    }
                    Some(deadline) if now >= deadline => {
                        log::warn!(
                            "回放数据已结束，但等待 {} 秒仍未完成交易日 {} 的日终结算，会话直接结束（该日不触发 on_after_trading）；请检查 trading_calendar= / sessions= 是否覆盖回放数据的最后一个交易日",
                            crate::engine::core::LiveReplayClose::WAIT_NS / 1_000_000_000,
                            close.day
                        );
                        engine.live_replay_close = None;
                        self.finalize_current_timestamp(engine, py);
                        return Ok(ProcessorResult::Break);
                    }
                    _ => {}
                }
            }
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
                    // 引擎内部的日终结算定时器: 本身不转给策略; 真的结算了当日时, 发出
                    // 收盘后生命周期事件(after_trading)交给策略。
                    if timer.is_day_close() {
                        if timer.timestamp > self.last_timestamp {
                            self.finalize_current_timestamp(engine, py);
                        }
                        // payload 格式由引擎自己写入; 用户定时器不能使用该前缀
                        // (add_timer/schedule 会拒绝), 解析失败就忽略。
                        if let Ok(day) = NaiveDate::parse_from_str(
                            &timer.payload[crate::model::timer::DAY_CLOSE_TIMER_PREFIX.len()..],
                            "%Y-%m-%d",
                        ) && engine.last_settled_trading_day.is_none_or(|s| s < day)
                        {
                            // 只结算晚于已结算进度的交易日: 残留的旧交易日定时器不回头
                            // 结算、不让进度回退。
                            //
                            // 实盘按日历登记的收盘定时器所属交易日没有任何行情(未开日)时,
                            // 先按行情开日的同一流程开日(推进当前交易日、记入已开交易日),
                            // 再与有行情日一样结算: 结算了就算开过。
                            if engine.state.feed.is_live()
                                && engine.current_trading_day.is_none_or(|cur| cur < day)
                            {
                                open_trading_day(engine, py, strategy, day, timer.timestamp)?;
                                if let Some((_, end)) = engine.current_day_span
                                    && engine.trading_day_ends.last().map(|(d, _)| *d) != Some(day)
                                {
                                    engine.trading_day_ends.push((day, end));
                                }
                                engine.scheduled_day_close = Some(day);
                            }
                            // 结算时刻是会话收盘, 不是定时器时刻(实盘有宽限)。
                            let close_ts =
                                match (engine.current_trading_day, engine.current_day_span) {
                                    (Some(cur), Some((_, end))) if cur == day => end,
                                    _ => engine
                                        .session_resolver
                                        .as_ref()
                                        .and_then(|r| r.session_day_end(day).ok())
                                        .unwrap_or(timer.timestamp - 1),
                                };
                            // 回测数据在盘中耗尽(没有收盘时刻的行情)时当日并未收盘:
                            // 不结算, 留给收尾路径记期末快照; 同日续跑时由续跑段收盘结算。
                            let feed_exhausted = !engine.state.feed.is_live()
                                && engine.state.feed.peek_timestamp().is_none();
                            if day_close_reached(
                                feed_exhausted,
                                engine.current_trading_day,
                                engine.last_event_ts_in_day,
                                day,
                                close_ts,
                            ) {
                                // 收盘时刻的 bar 缺失时, 当日已结束的窗口在结算前闭合派发
                                // (窗口 → 结算 → on_after_trading)。
                                dispatch_day_close_windows(engine, py, strategy, day, close_ts)?;
                                run_day_close_settlement(engine, py, day, close_ts);
                                engine.last_settled_trading_day = Some(day);
                                // replay 收尾: 末日已结算, on_after_trading 派发后结束会话。
                                if let Some(close) = engine.live_replay_close.as_mut()
                                    && day >= close.day
                                {
                                    close.settled = true;
                                }
                                // 没有策略要收盘后事件(未重写钩子且无过期单/到期事件
                                // 要送达)时不发出, 省一次 Python 派发。
                                let dispatch = engine.should_dispatch_lifecycle(
                                    crate::model::timer::LIFECYCLE_AFTER_TRADING,
                                );
                                if dispatch {
                                    emit_after_trading(engine, day, timer.timestamp)?;
                                }
                                // 回测: 按下一个行情事件所属交易日串联生命周期(数据耗尽
                                // 则不排); 实盘在 before_trading 触发时排下一交易日。
                                if !engine.state.feed.is_live()
                                    && let Some(next_ts) = engine.state.feed.peek_timestamp()
                                {
                                    engine
                                        .schedule_lifecycle_for_event(next_ts, timer.timestamp)?;
                                }
                                return Ok(if dispatch {
                                    ProcessorResult::Next
                                } else {
                                    ProcessorResult::Loop
                                });
                            }
                        }
                        return Ok(ProcessorResult::Loop);
                    }
                    // 回测数据耗尽后不再触发盘前/开始交易(只为有数据的交易日、且在数据
                    // 结束前触发; 收盘后事件只在当日真的结算时发出)。
                    if !engine.state.feed.is_live()
                        && engine.state.feed.peek_timestamp().is_none()
                        && timer.lifecycle().is_some_and(|(kind, _)| {
                            kind != crate::model::timer::LIFECYCLE_AFTER_TRADING
                        })
                    {
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
                    // 属于当前时间片的定时器(不晚于本片时间戳, 或框架截面定时器
                    // source_ts + 1 且 source_ts 即本片)沿用本片各品种按 bar 区间求得的
                    // 时段: ctx.session 取其中优先级最高者, 与撮合(SessionView 先查本片)
                    // 一致。其余定时器(用户 schedule 等)在本片之后, 清空本片后
                    // ctx.session 取定时器时刻的会话级时段, 撮合按订单品种在定时器时刻
                    // 解析(上一片已在上面 finalize)。
                    //
                    // 生命周期定时器(盘前/开始交易)不属于任何时间片, 同样清空后按时刻求。
                    let belongs = timer_belongs_to_slice(&timer, self.last_timestamp);
                    if !belongs {
                        std::sync::Arc::make_mut(&mut engine.slice_sessions).clear();
                    }
                    let session = timer_session(engine, timer.timestamp, timer.is_lifecycle())?;
                    if engine.state.feed.is_live() {
                        engine.chain_live_lifecycle(&timer)?;
                    }
                    // 生命周期事件只在有策略要它时派发(见 should_dispatch_lifecycle);
                    // 否则在引擎内消费掉(上面已完成实盘串联), 不推进时钟。
                    if let Some((kind, _)) = timer.lifecycle()
                        && !engine.should_dispatch_lifecycle(kind)
                    {
                        return Ok(ProcessorResult::Loop);
                    }
                    engine.clock.update(timer.timestamp, session);
                    engine.current_event = Some(Event::Timer(timer));
                }
                Ok(ProcessorResult::Next)
            }
            FeedAction::Event(event) => {
                let event = *event;

                // 行情源交易日: 与前后行情同序在此处理(校验/修正日历), 不是行情,
                // 不进时间片、不派发给策略。
                if let Event::TradingDayHint {
                    timestamp,
                    trading_day,
                    source,
                } = &event
                {
                    engine.apply_source_trading_day(*timestamp, *trading_day, source)?;
                    return Ok(ProcessorResult::Loop);
                }

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

                if let Some(close) = engine.live_replay_close.as_mut()
                    && timestamp >= close.last_event_ns
                {
                    close.last_event_seen = true;
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
                if timestamp > self.last_timestamp {
                    // 新时间片(上一片已 finalize): 上一片各品种的时段作废。
                    std::sync::Arc::make_mut(&mut engine.slice_sessions).clear();
                }
                self.last_timestamp = timestamp;

                // 本事件品种的时段: 按 bar 区间 (interval_start, ts] 求(tick 按时刻)。
                let (event_symbol, interval_start) = match &event {
                    Event::Bar(b) => (b.symbol.as_str(), b.interval_start),
                    Event::Tick(t) => (t.symbol.as_str(), None),
                    _ => ("", None),
                };
                let session = engine.event_session(event_symbol, interval_start, timestamp)?;
                // 每个品种最多告警一次, 全局最多 CLOSED_SESSION_WARN_LIMIT 条; 其余品种
                // 只记入集合, 由 Engine::run 结束时汇总一条(多品种回测不刷屏)。
                if session == TradingSession::Closed {
                    engine.closed_session_event_count += 1;
                    if !engine.closed_session_warned.contains(event_symbol) {
                        engine
                            .closed_session_warned
                            .insert(event_symbol.to_string());
                        if engine.closed_session_warned.len()
                            <= crate::engine::core::CLOSED_SESSION_WARN_LIMIT
                        {
                            // 期货默认表无夜盘、行情在夜盘时间: 换成列出夜盘模板的提示。
                            let tail = engine
                                .futures_night_hint(event_symbol, timestamp)
                                .unwrap_or_else(|| {
                                    "如确为夜盘等时段，请用 sessions= 指定时段模板（如 {\"au\": \"CN_FUTURES_NIGHT_0230\"}）".to_string()
                                });
                            log::warn!(
                                "标的 {} 在 {} 的行情落在闭市时段，期间不会撮合成交。请检查时间戳：DataFrame 中无时区的时间按 timezone= 指定的时区解释（默认 Asia/Shanghai），时区不符时请改 timezone= 或先 tz_localize 到交易所时区；手工构造的 Bar 与整数时间戳是 UTC 纳秒，请由带时区的时间换算（如 pd.Timestamp(\"2024-01-02 15:00\", tz=\"Asia/Shanghai\").value）；日期不在交易日历内、时刻落在夜间或午休也会如此。{}",
                                event_symbol,
                                engine.local_datetime_text(timestamp),
                                tail
                            );
                        }
                    }
                }
                let slice = std::sync::Arc::make_mut(&mut engine.slice_sessions);
                match slice.get_mut(event_symbol) {
                    Some(slot) => *slot = session,
                    None => {
                        slice.insert(event_symbol.to_string(), session);
                    }
                }
                engine.clock.update(timestamp, session);

                // 开日: 第一个属于新会话交易日的(活跃)事件。日终结算不在这里做,
                // 而是挂在该交易日收盘后的内部定时器上(回测 +1ns, 实盘留宽限;
                // 见 Timer 分支与 `day_close_timer_timestamp`)。
                if engine.is_active_timestamp(timestamp) {
                    let day = event_trading_day(
                        engine.current_trading_day,
                        engine.trading_day_for(timestamp)?,
                    );
                    if engine.current_trading_day != Some(day) {
                        open_trading_day(engine, py, strategy, day, timestamp)?;
                    }
                    if engine.current_day_span.is_none() {
                        // 从存档恢复后同一交易日内续跑：重建当日跨度以登记收盘定时器
                        engine.current_day_span =
                            Some(session_day_span_for(engine, day, timestamp)?);
                    }
                    // 记下开过的交易日与收盘时刻，供日收益/日权益按交易日分组
                    if let Some((_, end)) = engine.current_day_span
                        && engine.trading_day_ends.last().map(|(d, _)| *d) != Some(day)
                    {
                        engine.trading_day_ends.push((day, end));
                    }
                    // 实盘已按交易日历为当日登记过收盘定时器时按键去重, 不重复登记。
                    if engine.scheduled_day_close != Some(day)
                        && engine.last_settled_trading_day != Some(day)
                        && let Some((_, end)) = engine.current_day_span
                    {
                        let close_timer = engine.day_close_timer_timestamp(end);
                        engine.push_day_close_timer(day, close_timer);
                        engine.scheduled_day_close = Some(day);
                    }
                    engine.last_event_ts_in_day = Some(timestamp);
                }

                if let Event::Bar(ref b) = event {
                    self.seen_symbols.insert(b.symbol.clone());
                    // 集合竞价时段的零量 bar(竞价期快照、09:25～09:30 静默期)不是停牌:
                    // 本就不连续撮合, 不能据此把排队的单当作停牌拒掉。
                    if b.volume > Decimal::ZERO
                        || matches!(
                            session,
                            TradingSession::OpeningAuction
                                | TradingSession::AuctionHold
                                | TradingSession::ClosingAuction
                        )
                    {
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
                    // tick 撮合不看成交量(execution/common.rs), 有 tick 即本片可成交:
                    // 否则纯 tick 回测里跨片排队的 NextOpen 单(如竞价时段下的单)会在
                    // reject_missing_symbol_orders 被当作停牌拒掉。
                    if !self.tradable_symbols.contains(&t.symbol) {
                        self.tradable_symbols.insert(t.symbol.clone());
                    }
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

/// 定时器事件的会话时段: 本时间片有品种时段时取其中优先级最高者(与撮合先查本片
/// 一致), 否则取定时器时刻的会话级时段。
///
/// 普通定时器按时刻前 1ns 求(与行情同口径); 生命周期事件(`at_instant`)按事件时刻
/// 本身求——收盘后事件在收盘 + 1ns, 应为闭市而不是收盘竞价/连续竞价。
fn timer_session(engine: &mut Engine, ts: i64, at_instant: bool) -> PyResult<TradingSession> {
    if let Some(best) = engine
        .slice_sessions
        .values()
        .copied()
        .max_by_key(|s| crate::calendar::resolver::rank(*s))
    {
        return Ok(best);
    }
    engine.ensure_session_resolver()?;
    Ok(engine
        .session_resolver
        .as_ref()
        .map_or(TradingSession::Continuous, |r| {
            // session_level_instant 取前 1ns, 按事件时刻本身求时传 ts + 1。
            r.session_level_instant(if at_instant { ts.saturating_add(1) } else { ts })
        }))
}

/// 日终结算完成后发出交易日 `day` 的收盘后生命周期事件(策略可见的定时器事件),
/// 时刻为收盘定时器时刻 `ts`。它晚于本时间片, 清空本片时段后按时刻求。
fn emit_after_trading(engine: &mut Engine, day: NaiveDate, ts: i64) -> PyResult<()> {
    std::sync::Arc::make_mut(&mut engine.slice_sessions).clear();
    let session = timer_session(engine, ts, true)?;
    engine.clock.update(ts, session);
    engine.current_event = Some(Event::Timer(crate::model::Timer {
        timestamp: ts,
        payload: crate::model::timer::lifecycle_payload(
            crate::model::timer::LIFECYCLE_AFTER_TRADING,
            day,
        ),
        owner: None,
    }));
    Ok(())
}

/// 定时器是否属于时间戳为 `slice_ts` 的时间片(撮合时沿用该片各品种的时段)。
fn timer_belongs_to_slice(timer: &crate::model::Timer, slice_ts: i64) -> bool {
    if timer.timestamp <= slice_ts {
        return true;
    }
    // 框架截面定时器: `__framework_cross_section__|<day>|<source_ts>`, 在 source_ts + 1
    // 触发, 成交时刻按 source_ts 计(见 execution/common.rs)。
    timer
        .payload
        .strip_prefix("__framework_cross_section__|")
        .and_then(|rest| rest.rsplit('|').next())
        .and_then(|v| v.parse::<i64>().ok())
        == Some(slice_ts)
}

/// 事件所属的交易日。乱序到达(解析出的交易日早于当前交易日, 如实盘迟到的上一
/// 交易日行情)时按当前交易日处理: 不让 `current_trading_day` 倒退, 也不触发结算。
fn event_trading_day(current: Option<NaiveDate>, resolved: NaiveDate) -> NaiveDate {
    match current {
        Some(cur) if resolved < cur => cur,
        _ => resolved,
    }
}

/// 开交易日 `day`(时刻 `timestamp`): 上一交易日未结算时先补结算, 处理公司行动,
/// 推进 `current_trading_day` 与当日跨度。行情事件开日与实盘按日历结算无行情日
/// (见 Timer 分支)共用。
fn open_trading_day(
    engine: &mut Engine,
    py: Python<'_>,
    strategy: &Bound<'_, PyAny>,
    day: NaiveDate,
    timestamp: i64,
) -> PyResult<()> {
    // 上一交易日未结算(实盘漏触发等): 先补结算。已结算进度晚于它
    // (实盘按日历结算了其后无行情的交易日)时不回头重复结算。
    if let Some(prev) = engine.current_trading_day
        && engine.last_settled_trading_day.is_none_or(|s| s < prev)
    {
        // 收盘时刻: 优先用已登记的跨度; 从存档恢复后跨度为空,
        // 取该交易日的会话收盘, 不退化为本事件时间戳。
        let close = match engine.current_day_span {
            Some((_, end)) => end,
            None => engine
                .session_resolver
                .as_ref()
                .expect("run() 已构造")
                .session_day_end(prev)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        };
        dispatch_day_close_windows(engine, py, strategy, prev, close)?;
        run_day_close_settlement(engine, py, prev, close);
        engine.last_settled_trading_day = Some(prev);
        // 收盘后生命周期事件以本事件时刻压入, 在本事件处理之后触发
        // (比正常路径晚一个事件, Ruling S4-1)。
        engine.timers.push(crate::model::Timer {
            timestamp,
            payload: crate::model::timer::lifecycle_payload(
                crate::model::timer::LIFECYCLE_AFTER_TRADING,
                prev,
            ),
            owner: None,
        });
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
    engine.current_day_span = Some(session_day_span_for(engine, day, timestamp)?);
    engine.last_event_ts_in_day = None;
    Ok(())
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

/// 日终结算前闭合交易日 `day` 已结束的在形成窗口(日内窗口标签不晚于收盘、当日
/// 日线窗口)并派发: 收盘时刻的 bar 缺失(如数据止于 14:56)时, 窗口不拖到下一交易日
/// 第一根 bar 才派发。闭合的窗口从聚合器移除, 只派发一次; 写历史先于派发, 与数据
/// 阶段同序。
fn dispatch_day_close_windows(
    engine: &mut Engine,
    py: Python<'_>,
    strategy: &Bound<'_, PyAny>,
    day: NaiveDate,
    close_ts: i64,
) -> PyResult<()> {
    let closed = engine
        .window_aggregator
        .write()
        .expect("window_aggregator 写锁被污染")
        .close_through_day(day, close_ts);
    if closed.is_empty() && engine.pending_window_bars.is_empty() {
        return Ok(());
    }
    if let Ok(mut buffer) = engine.history_buffer.write() {
        for bar in &closed {
            buffer.update_window(bar);
        }
    }
    // 尚未派发的(同一时间步更早闭合的)在前
    let mut bars = std::mem::take(&mut engine.pending_window_bars);
    bars.extend(closed);
    // 派发期间时钟与当前事件置为收盘时刻(时段沿用最后事件的时段): 策略上下文的
    // 当前时刻取自当前事件, 窗口回调里的 now 是窗口标签, 回调中下的单按收盘时刻计
    // 创建时间, 与收盘 bar 存在时的行为一致。派发后还原(原因同
    // run_day_close_settlement), 派发出错时也还原。
    let previous_clock = (engine.clock.timestamp(), engine.clock.session);
    let previous_event = engine.current_event.take();
    engine.clock.update(close_ts, engine.clock.session);
    engine.current_event = Some(Event::Timer(crate::model::Timer {
        timestamp: close_ts,
        payload: "__day_close_windows__".to_string(),
        owner: None,
    }));
    let result = engine.dispatch_window_bars(py, strategy, bars);
    engine.current_event = previous_event;
    if let (Some(prev_ts), prev_session) = previous_clock {
        engine.clock.update(prev_ts, prev_session);
    }
    result
}

/// 交易日 `day` 的日终结算：到期 → 利息与强平 → T+1 → Day 单过期 → 流事件 → 快照。
/// `close_ts` 为会话收盘时刻（快照与流事件时间戳）。
fn run_day_close_settlement(engine: &mut Engine, py: Python<'_>, day: NaiveDate, close_ts: i64) {
    // 流事件信封的 ts 取 flush 时的 clock: 先把结算前已缓冲的事件按原时钟发出,
    // 结算期间把 clock 置为收盘时刻, 结算结束再 flush 一次, 结算产生的
    // settlement/risk/expiry 事件就都带收盘时刻。
    engine.flush_stream_events(py);
    let previous_clock = (engine.clock.timestamp(), engine.clock.session);
    let session = engine.clock.session;
    engine.clock.update(close_ts, session);

    settle_day_close(engine, py, day, close_ts);

    engine.flush_stream_events(py);
    // 结算结束后还原 clock(而不是停在收盘时刻): clock 是"最后处理到的事件时刻",
    // save_checkpoint 把它存为 current_time, 续跑时丢弃不晚于它的行情; 停在收盘会
    // 让同一交易日内续跑的行情被整段丢弃(test_window_checkpoint 等)。补结算路径
    // (开日时补上一交易日)发生在新事件已更新 clock 之后, 不还原则本事件的处理会
    // 倒退到上一交易日收盘。流事件已在上面 flush, 信封时间不受还原影响。
    if let (Some(prev_ts), prev_session) = previous_clock {
        engine.clock.update(prev_ts, prev_session);
    }
}

/// [`run_day_close_settlement`] 的结算步骤本体。
fn settle_day_close(engine: &mut Engine, py: Python<'_>, day: NaiveDate, close_ts: i64) {
    // 收盘竞价收尾(缺收盘时刻行情时按最后一个收盘竞价价统一撮合): 先于到期结算与
    // Day 单过期, 成交就地落地, 使收盘快照包含竞价成交。
    finalize_closing_auctions(engine, py, day, close_ts);
    // Settlement Manager (T+1, Option Expiry, Day Order Expiry)
    let mut expired_orders = Vec::new();
    // NextOpen(bar_offset == 1) 的 Day 单若创建于本交易日最后一个已处理行情事件
    // 当时或之后, 唯一的撮合机会是下一交易日的第一个事件, 本次结算不过期(当日没有
    // 行情事件时不适用)。
    // 同周期(bar_offset == 0)单在创建时的事件上已有撮合机会, 照常过期。
    // 收盘时刻及之后下的 Day 单(如收盘竞价结束的 15:00 bar 上下的单)属于下一交易日,
    // 不论成交模式都不在本次结算过期。
    let exempt_day_orders: HashSet<String> = engine
        .state
        .order_manager
        .active_orders
        .iter()
        .filter(|order| {
            order.time_in_force == crate::model::TimeInForce::Day
                && ((effective_execution_policy_for_order(engine, order).bar_offset == 1
                    // 当日没有任何行情(实盘按日历结算的无行情日)时 NextOpen 单在当日
                    // 也没有撮合机会可等, 不豁免, 与有行情日一样在所属交易日收盘过期。
                    && engine
                        .last_event_ts_in_day
                        .is_some_and(|last| order.created_at >= last))
                    // 收盘时刻及之后、在非连续竞价时段(如收盘竞价结束的 15:00 分钟 bar)
                    // 下的单当日没有撮合机会, 属于下一交易日。日线(整日区间, 连续竞价)
                    // 上下的单当日有撮合机会, 照常过期。
                    || (order.created_at >= close_ts
                        && engine.order_creation_sessions.contains_key(&order.id)))
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
        accrual_days: engine
            .last_settled_trading_day
            .map(|p| (day - p).num_days())
            .unwrap_or(1)
            .max(1),
    };
    let settlement_outcome = engine.settlement_manager.process_daily_settlement(
        &mut engine.state.portfolio,
        &mut engine.state.order_manager.active_orders,
        &mut expired_orders,
        &settlement_ctx,
    );
    drop(settlement_ctx);
    drop(settlement_prices);
    // 只保留仍在途订单的创建时段记录
    if !engine.order_creation_sessions.is_empty() {
        let active: HashSet<&str> = engine
            .state
            .order_manager
            .active_orders
            .iter()
            .map(|o| o.id.as_str())
            .collect();
        engine
            .order_creation_sessions
            .retain(|id, _| active.contains(id.as_str()));
    }
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
        // 日终定时器不转给策略: 过期单留到下一个策略事件(或会话收尾)回调 on_order。
        engine
            .state
            .order_manager
            .current_step_expired_orders
            .push(o.clone());
        // 与拒单/撤单回报对称发 order 流事件, on_event 与指标流消费方也能看到 Expired。
        let mut order_payload = HashMap::new();
        order_payload.insert("order_id", o.id.clone());
        order_payload.insert("status", format!("{:?}", o.status));
        order_payload.insert("filled_qty", o.filled_quantity.to_string());
        order_payload.insert("symbol", o.symbol.clone());
        order_payload.insert(
            "owner_strategy_id",
            o.owner_strategy_id.clone().unwrap_or_default(),
        );
        engine.emit_stream_event(py, "order", Some(o.symbol.as_str()), "info", order_payload);
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

/// 日终结算前收尾 `day` 的收盘集合竞价, 撮合回报按强平成交的方式就地落地。
fn finalize_closing_auctions(engine: &mut Engine, py: Python<'_>, day: NaiveDate, close_ts: i64) {
    // 无解析器(构造失败)时 SessionView 按连续竞价; 配置错误已在 run() 开始时报出。
    let _ = engine.ensure_session_resolver();
    let prices = engine.last_prices.read().expect("last_prices 读锁被污染");
    let ctx = EngineContext {
        instruments: &engine.instruments,
        portfolio: &engine.state.portfolio,
        last_prices: &prices,
        trade_tracker: &engine.state.order_manager.trade_tracker,
        market_model: engine.market_manager.model.as_ref(),
        execution_policy_core: engine.execution_policy_core(),
        bar_index: engine.bar_count,
        current_time: close_ts,
        session: engine.clock.session,
        active_orders: &engine.state.order_manager.active_orders,
        risk_config: &engine.risk_manager.config,
        timezone_name: engine.timezone_name.as_deref(),
        timezone_offset: engine.timezone_offset,
        trading_day: Some(day),
        sessions: SessionView {
            slice: Some(&*engine.slice_sessions),
            resolver: engine.session_resolver.as_deref(),
            instruments: Some(&engine.instruments),
        },
    };
    let reports = engine.execution_model.finalize_closing_auctions(&ctx);
    drop(ctx);
    drop(prices);
    if reports.is_empty() {
        return;
    }
    let mut trades = Vec::new();
    let mut oco_suppressed_fill_order_ids: HashSet<String> = HashSet::new();
    for report in reports {
        if let Event::ExecutionReport(order, trade) = report {
            apply_execution_report(
                engine,
                py,
                order,
                trade,
                &mut oco_suppressed_fill_order_ids,
                &mut trades,
            );
        }
    }
    flush_accumulated_trades(engine, &mut trades);
}

/// 收盘定时器触发时交易日 `day` 是否真的走到了收盘。
///
/// 回测数据耗尽(`feed_exhausted`)且 `day` 最后处理的行情事件早于会话收盘 `close_ts`
/// 时, 数据是在盘中结束的, 当日没有收盘, 返回 `false`; 其余情况(后面还有数据、
/// 实盘、末根事件已到收盘)返回 `true`。
fn day_close_reached(
    feed_exhausted: bool,
    current_trading_day: Option<NaiveDate>,
    last_event_ts_in_day: Option<i64>,
    day: NaiveDate,
    close_ts: i64,
) -> bool {
    if !feed_exhausted || current_trading_day != Some(day) {
        return true;
    }
    // 只有日期的数据在全天连续时段表下改写为收盘前 1ns(保持本地日期), 视同已到收盘。
    last_event_ts_in_day.is_some_and(|ts| ts.saturating_add(1) >= close_ts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_close_not_reached_when_backtest_data_ends_mid_day() {
        let day = NaiveDate::from_ymd_opt(2024, 1, 4).unwrap();
        let close = 1_000;
        // 数据耗尽、末根事件在收盘前: 未收盘
        assert!(!day_close_reached(true, Some(day), Some(998), day, close));
        assert!(!day_close_reached(true, Some(day), None, day, close));
        // 末根事件恰在收盘或之后: 已收盘
        assert!(day_close_reached(true, Some(day), Some(1_000), day, close));
        // 收盘前 1ns(ALWAYS_OPEN 日线改写口径): 已收盘
        assert!(day_close_reached(true, Some(day), Some(999), day, close));
        // 后面还有数据(或实盘): 照常结算
        assert!(day_close_reached(false, Some(day), Some(998), day, close));
        // 定时器不属于当前交易日: 照常结算
        let other = NaiveDate::from_ymd_opt(2024, 1, 3).unwrap();
        assert!(day_close_reached(true, Some(day), Some(999), other, close));
    }

    #[test]
    fn out_of_order_event_does_not_regress_trading_day() {
        let d = |m, day| NaiveDate::from_ymd_opt(2024, m, day).unwrap();
        // 迟到的上一交易日事件归当前交易日
        assert_eq!(event_trading_day(Some(d(1, 3)), d(1, 2)), d(1, 3));
        // 正常推进与首日
        assert_eq!(event_trading_day(Some(d(1, 3)), d(1, 4)), d(1, 4));
        assert_eq!(event_trading_day(Some(d(1, 3)), d(1, 3)), d(1, 3));
        assert_eq!(event_trading_day(None, d(1, 2)), d(1, 2));
    }
}
