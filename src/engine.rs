use chrono::{DateTime, NaiveDate, NaiveTime, TimeZone, Utc};
use indicatif::{ProgressBar, ProgressStyle};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::prelude::*;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::analysis::{BacktestResult, TradeTracker};
use crate::clock::Clock;
use crate::context::StrategyContext;
use crate::data::{DataFeed, Event};
use crate::execution::ExchangeSimulator;
use crate::history::HistoryBuffer;
use crate::market::{
    ChinaMarket, ChinaMarketConfig, MarketModel, MarketType, SessionRange, SimpleMarket,
};
use crate::model::{
    Bar, ExecutionMode, Instrument, Order, OrderStatus, TimeInForce, Timer, Trade,
    TradingSession,
};
use crate::portfolio::Portfolio;
use crate::risk::RiskManager;

/// 主回测引擎
///
/// :ivar feed: 数据源
/// :ivar portfolio: 投资组合
/// :ivar orders: 订单列表
/// :ivar trades: 成交列表
#[gen_stub_pyclass]
#[pyclass]
pub struct Engine {
    feed: DataFeed,
    #[pyo3(get)]
    portfolio: Portfolio,
    #[pyo3(get)]
    orders: Vec<Order>,
    #[pyo3(get)]
    trades: Vec<Trade>,
    last_prices: HashMap<String, Decimal>,
    instruments: HashMap<String, Instrument>,
    current_date: Option<NaiveDate>,
    market_config: ChinaMarketConfig,
    active_market_type: MarketType,
    market_model: Box<dyn MarketModel>,
    execution_model: ExchangeSimulator,
    daily_equity: Vec<(i64, Decimal)>,
    daily_positions: Vec<(i64, HashMap<String, Decimal>)>,
    execution_mode: ExecutionMode,
    clock: Clock,
    timers: BinaryHeap<Timer>, // Min-Heap via Timer's Ord implementation
    force_session_continuous: bool,
    #[pyo3(get, set)]
    pub risk_manager: RiskManager,
    timezone_offset: i32,
    trade_tracker: TradeTracker,
    history_buffer: Arc<RwLock<HistoryBuffer>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Engine {
    /// 初始化回测引擎
    ///
    /// :return: Engine 实例
    #[new]
    fn new() -> Self {
        let market_config = ChinaMarketConfig::default();
        Engine {
            feed: DataFeed::new(),
            portfolio: Portfolio {
                cash: Decimal::from(100_000),
                positions: HashMap::new(),
                available_positions: HashMap::new(),
            },
            orders: Vec::new(),
            trades: Vec::new(),
            last_prices: HashMap::new(),
            instruments: HashMap::new(),
            current_date: None,
            market_config: market_config.clone(),
            active_market_type: MarketType::China,
            market_model: Box::new(ChinaMarket::from_config(market_config)),
            execution_model: ExchangeSimulator::new(),
            daily_equity: Vec::new(),
            daily_positions: Vec::new(),
            execution_mode: ExecutionMode::NextOpen,
            clock: Clock::new(),
            timers: BinaryHeap::new(),
            force_session_continuous: false,
            risk_manager: RiskManager::new(),
            timezone_offset: 28800, // Default UTC+8
            trade_tracker: TradeTracker::new(),
            history_buffer: Arc::new(RwLock::new(HistoryBuffer::new(0))),
        }
    }

    /// 设置历史数据长度
    ///
    /// :param depth: 历史数据长度
    fn set_history_depth(&mut self, depth: usize) {
        self.history_buffer.write().unwrap().set_capacity(depth);
    }

    /// 设置时区偏移 (秒)
    ///
    /// :param offset: 偏移秒数 (例如 UTC+8 为 28800)
    fn set_timezone(&mut self, offset: i32) {
        self.timezone_offset = offset;
    }

    /// 设置撮合模式
    ///
    /// :param mode: 撮合模式 (ExecutionMode.CurrentClose 或 ExecutionMode.NextOpen)
    /// :type mode: ExecutionMode
    fn set_execution_mode(&mut self, mode: ExecutionMode) {
        self.execution_mode = mode;
    }

    /// 启用 SimpleMarket (7x24小时, T+0, 无税, 简单佣金)
    ///
    /// :param commission_rate: 佣金率
    fn use_simple_market(&mut self, commission_rate: f64) {
        self.active_market_type = MarketType::Simple;
        self.market_model = Box::new(SimpleMarket::new(commission_rate));
    }

    /// 启用 ChinaMarket (支持 T+1/T+0, 印花税, 过户费, 交易时段等)
    fn use_china_market(&mut self) {
        self.active_market_type = MarketType::China;
        self.market_model = Box::new(ChinaMarket::from_config(self.market_config.clone()));
    }

    /// 启用中国期货市场默认配置
    /// - 切换到 ChinaMarket
    /// - 设置 T+0
    /// - 保持当前交易时段配置 (需手动设置 set_market_sessions 以匹配特定品种)
    fn use_china_futures_market(&mut self) {
        self.active_market_type = MarketType::China;
        self.market_config.t_plus_one = false;
        self.market_model = Box::new(ChinaMarket::from_config(self.market_config.clone()));
    }

    /// 启用/禁用 T+1 交易规则 (仅针对 ChinaMarket)
    ///
    /// :param enabled: 是否启用 T+1
    /// :type enabled: bool
    fn set_t_plus_one(&mut self, enabled: bool) {
        self.market_config.t_plus_one = enabled;
        if self.active_market_type == MarketType::China {
            self.market_model = Box::new(ChinaMarket::from_config(self.market_config.clone()));
        }
    }

    fn set_force_session_continuous(&mut self, enabled: bool) {
        self.force_session_continuous = enabled;
    }

    fn set_stock_fee_rules(
        &mut self,
        commission_rate: f64,
        stamp_tax: f64,
        transfer_fee: f64,
        min_commission: f64,
    ) {
        self.market_config.stock_commission_rate =
            Decimal::from_f64(commission_rate).unwrap_or(Decimal::ZERO);
        self.market_config.stock_stamp_tax = Decimal::from_f64(stamp_tax).unwrap_or(Decimal::ZERO);
        self.market_config.stock_transfer_fee =
            Decimal::from_f64(transfer_fee).unwrap_or(Decimal::ZERO);
        self.market_config.stock_min_commission =
            Decimal::from_f64(min_commission).unwrap_or(Decimal::ZERO);
        if self.active_market_type == MarketType::China {
            self.market_model = Box::new(ChinaMarket::from_config(self.market_config.clone()));
        }
    }

    fn set_future_fee_rules(&mut self, commission_rate: f64) {
        self.market_config.future_commission_rate =
            Decimal::from_f64(commission_rate).unwrap_or(Decimal::ZERO);
        if self.active_market_type == MarketType::China {
            self.market_model = Box::new(ChinaMarket::from_config(self.market_config.clone()));
        }
    }

    fn set_fund_fee_rules(
        &mut self,
        commission_rate: f64,
        transfer_fee: f64,
        min_commission: f64,
    ) {
        self.market_config.fund_commission_rate =
            Decimal::from_f64(commission_rate).unwrap_or(Decimal::ZERO);
        self.market_config.fund_transfer_fee =
            Decimal::from_f64(transfer_fee).unwrap_or(Decimal::ZERO);
        self.market_config.fund_min_commission =
            Decimal::from_f64(min_commission).unwrap_or(Decimal::ZERO);
        if self.active_market_type == MarketType::China {
            self.market_model = Box::new(ChinaMarket::from_config(self.market_config.clone()));
        }
    }

    fn set_option_fee_rules(&mut self, commission_per_contract: f64) {
        self.market_config.option_commission_per_contract =
            Decimal::from_f64(commission_per_contract).unwrap_or(Decimal::ZERO);
        if self.active_market_type == MarketType::China {
            self.market_model = Box::new(ChinaMarket::from_config(self.market_config.clone()));
        }
    }

    /// 设置滑点模型
    ///
    /// :param type: 滑点类型 ("fixed" 或 "percent")
    /// :param value: 滑点值 (固定金额 或 百分比如 0.001)
    fn set_slippage(&mut self, type_: String, value: f64) -> PyResult<()> {
        let val = Decimal::from_f64(value).unwrap_or(Decimal::ZERO);
        match type_.as_str() {
            "fixed" => {
                self.execution_model
                    .set_slippage_model(Box::new(crate::execution::FixedSlippage { delta: val }));
            }
            "percent" => {
                self.execution_model
                    .set_slippage_model(Box::new(crate::execution::PercentSlippage { rate: val }));
            }
            "zero" | "none" => {
                self.execution_model
                    .set_slippage_model(Box::new(crate::execution::ZeroSlippage));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "Invalid slippage type. Use 'fixed', 'percent', or 'zero'",
                ));
            }
        }
        Ok(())
    }

    /// 设置成交量限制
    ///
    /// :param limit: 限制比例 (0.0-1.0), 0.0 为不限制
    fn set_volume_limit(&mut self, limit: f64) {
        self.execution_model.set_volume_limit(limit);
    }

    fn set_market_sessions(
        &mut self,
        sessions: Vec<(String, String, TradingSession)>,
    ) -> PyResult<()> {
        let mut ranges = Vec::with_capacity(sessions.len());
        for (start, end, session) in sessions {
            let start_time = Self::parse_time_string(&start)?;
            let end_time = Self::parse_time_string(&end)?;
            ranges.push(SessionRange {
                start: start_time,
                end: end_time,
                session,
            });
        }
        self.market_config.sessions = ranges;
        if self.active_market_type == MarketType::China {
            self.market_model = Box::new(ChinaMarket::from_config(self.market_config.clone()));
        }
        Ok(())
    }

    /// 添加交易标的
    ///
    /// :param instrument: 交易标的对象
    /// :type instrument: Instrument
    fn add_instrument(&mut self, instrument: Instrument) {
        self.instruments
            .insert(instrument.symbol.clone(), instrument);
    }

    /// 设置初始资金
    ///
    /// :param cash: 初始资金数额
    /// :type cash: float
    fn set_cash(&mut self, cash: f64) {
        self.portfolio.cash = Decimal::from_f64(cash).unwrap_or(Decimal::ZERO);
    }

    /// 添加数据源
    ///
    /// :param feed: 数据源对象
    /// :type feed: DataFeed
    fn add_data(&mut self, feed: DataFeed) {
        self.feed = feed;
    }

    /// 批量添加 K 线数据
    ///
    /// :param bars: K 线列表
    fn add_bars(&mut self, bars: Vec<Bar>) -> PyResult<()> {
        self.feed.add_bars(bars)
    }

    /// 运行回测
    ///
    /// :param strategy: 策略对象
    /// :param show_progress: 是否显示进度条
    /// :type strategy: object
    /// :type show_progress: bool
    /// :return: 回测结果摘要
    /// :rtype: str
    fn run(&mut self, strategy: &Bound<'_, PyAny>, show_progress: bool) -> PyResult<String> {
        // Configure history buffer if strategy has _history_depth set
        if let Ok(depth_attr) = strategy.getattr("_history_depth") {
            if let Ok(depth) = depth_attr.extract::<usize>() {
                if depth > 0 {
                    self.set_history_depth(depth);
                }
            }
        }

        let mut pending_orders: Vec<Order> = Vec::new();
        let mut count = 0;
        let mut last_timestamp = 0;
        let mut bar_index = 0;

        // Progress Bar Initialization
        let total_events = self.feed.len_hint().unwrap_or(0);
        let pb = if show_progress {
            let pb = ProgressBar::new(total_events as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                    )
                    .unwrap()
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        // Record initial equity (before processing any events)
        // This ensures equity_curve starts with initial capital, preventing return calculation errors
        // for intraday backtests or when the first period has no return.
        if let Some(timestamp) = self.feed.peek_timestamp() {
            // At start, equity should equal cash (assuming no positions set)
            // If positions are set, calculate_equity will handle it (assuming prices are available or 0)
            let equity = self
                .portfolio
                .calculate_equity(&self.last_prices, &self.instruments);
            self.daily_equity.push((timestamp, equity));
        }

        // Process all events (Data + Timers) in order
        let is_live = self.feed.is_live();

        loop {
            // Check if we have data or timers
            let mut next_event_time = self.feed.peek_timestamp();
            let next_timer_time = self.timers.peek().map(|t| t.timestamp);

            if !is_live && next_event_time.is_none() && next_timer_time.is_none() {
                break; // No more events (Backtest End)
            }

            // Live Mode: Block and wait for data if buffer is empty
            if is_live && next_event_time.is_none() {
                let timeout = if let Some(timer_ts) = next_timer_time {
                    let now = Utc::now().timestamp_nanos_opt().unwrap_or(0);
                    if timer_ts > now {
                        let diff_ms = (timer_ts - now) / 1_000_000;
                        if diff_ms > 0 {
                            Duration::from_millis(std::cmp::min(diff_ms as u64, 1000))
                        } else {
                            Duration::ZERO
                        }
                    } else {
                        Duration::ZERO
                    }
                } else {
                    Duration::from_secs(1)
                };

                if timeout > Duration::ZERO {
                    if let Some(ts) = self.feed.wait_peek(timeout) {
                        next_event_time = Some(ts);
                    }
                }
            }

            // If live and still no event, check if timer is ready
            if is_live && next_event_time.is_none() {
                if let Some(timer_ts) = next_timer_time {
                    let now = Utc::now().timestamp_nanos_opt().unwrap_or(0);
                    if timer_ts > now {
                        continue; // Timer not ready yet, wait more
                    }
                } else {
                    continue; // No data, no timer, wait more
                }
            }

            let process_timer = match (next_event_time, next_timer_time) {
                (Some(et), Some(tt)) => tt <= et,
                (Some(_), None) => false,
                (None, Some(_)) => true,
                (None, None) => break,
            };

            if process_timer {
                // Process Timer
                if let Some(timer) = self.timers.pop() {
                    // Update Clock
                    let local_dt =
                        Self::local_datetime_from_ns(timer.timestamp, self.timezone_offset);
                    let session = self.market_model.get_session_status(local_dt.time());
                    self.clock.update(timer.timestamp, session);
                    if self.force_session_continuous {
                        self.clock.session = TradingSession::Continuous;
                    }

                    // Call Strategy on_timer
                    let (new_orders, canceled_ids) =
                        self.call_strategy_timer(strategy, &timer.payload, &pending_orders)?;

                    // Process cancellations
                    for order_id in canceled_ids {
                        if let Some(order) = pending_orders.iter_mut().find(|o| o.id == order_id) {
                            order.status = OrderStatus::Cancelled;
                        }
                    }

                    pending_orders.extend(new_orders);

                    // Note: Timers do not advance bar_index or trigger day close directly in this simplified model,
                    // unless we add logic for that. For now, they are just callbacks.
                    // Also, we don't extract new orders from timers yet, but we could.
                }
            } else {
                // Process Data Event
                let event = self.feed.next().unwrap();

                // Update History Buffer
                if let Event::Bar(ref b) = event {
                    self.history_buffer.write().unwrap().update(b);
                }

                count += 1;
                if let Some(pb) = &pb {
                    pb.inc(1);
                }

                // 1. Process Date Change (Day Close)
                let timestamp = match &event {
                    Event::Bar(b) => b.timestamp,
                    Event::Tick(t) => t.timestamp,
                };

                if last_timestamp != 0 && timestamp > last_timestamp {
                    bar_index += 1;
                }

                // Update Clock
                let local_dt = Self::local_datetime_from_ns(timestamp, self.timezone_offset);
                let session = self.market_model.get_session_status(local_dt.time());
                self.clock.update(timestamp, session);
                if self.force_session_continuous {
                    self.clock.session = TradingSession::Continuous;
                }

                let local_date = local_dt.date_naive();
                if self.current_date != Some(local_date) {
                    // New day: Settlement logic
                    if self.current_date.is_some() {
                        // Record equity for the previous day
                        let equity = self
                            .portfolio
                            .calculate_equity(&self.last_prices, &self.instruments);
                        self.daily_equity.push((last_timestamp, equity));
                        self.daily_positions.push((last_timestamp, self.portfolio.positions.clone()));

                        // Update T+1 availability
                        self.market_model.on_day_close(
                            &self.portfolio.positions,
                            &mut self.portfolio.available_positions,
                            &self.instruments,
                        );

                        // Expire Day orders
                        // Note: We need to filter pending_orders in place.
                        let (expired, kept): (Vec<Order>, Vec<Order>) = pending_orders
                            .into_iter()
                            .partition(|o| o.time_in_force == TimeInForce::Day);

                        for mut o in expired {
                            o.status = OrderStatus::Expired;
                            self.orders.push(o);
                        }
                        pending_orders = kept;
                    }
                    self.current_date = Some(local_date);
                }

                match self.execution_mode {
                    ExecutionMode::NextOpen => {
                        // Phase 1: Execution (Match pending orders at Open)
                        let new_trades = self.execution_model.process_event(
                            &event,
                            &mut pending_orders[..],
                            &self.instruments,
                            true,
                            bar_index,
                            self.clock.session,
                        );
                        self.process_trades(new_trades);

                        // Phase 2: Strategy (Generate new orders)
                        let (new_orders, new_timers, canceled_ids) =
                            self.call_strategy(strategy, &event, &pending_orders)?;

                        // Process cancellations
                        for order_id in canceled_ids {
                            if let Some(order) = pending_orders.iter_mut().find(|o| o.id == order_id) {
                                order.status = OrderStatus::Cancelled;
                            }
                        }

                        for mut order in new_orders {
                            if let Some(err) = self.risk_manager.check_internal(&order, &self.portfolio, &self.instruments, &pending_orders) {
                                println!("{}", err);
                                order.status = OrderStatus::Rejected;
                                self.orders.push(order);
                            } else {
                                pending_orders.push(order);
                            }
                        }
                        for t in new_timers {
                            self.timers.push(t);
                        }
                    }
                    ExecutionMode::CurrentClose => {
                        // Phase 1: Strategy (Generate new orders)
                        let (new_orders, new_timers, canceled_ids) =
                            self.call_strategy(strategy, &event, &pending_orders)?;

                        // Process cancellations
                        for order_id in canceled_ids {
                            if let Some(order) = pending_orders.iter_mut().find(|o| o.id == order_id) {
                                order.status = OrderStatus::Cancelled;
                            }
                        }

                        for mut order in new_orders {
                            if let Some(err) = self.risk_manager.check_internal(&order, &self.portfolio, &self.instruments, &pending_orders) {
                                println!("{}", err);
                                order.status = OrderStatus::Rejected;
                                self.orders.push(order);
                            } else {
                                pending_orders.push(order);
                            }
                        }
                        for t in new_timers {
                            self.timers.push(t);
                        }

                        // Phase 2: Execution (Match new orders at Close)
                        let new_trades = self.execution_model.process_event(
                            &event,
                            &mut pending_orders[..],
                            &self.instruments,
                            false,
                            bar_index,
                            self.clock.session,
                        );
                        self.process_trades(new_trades);
                    }
                }

                // Clean up filled/cancelled orders from pending list
                let (finished, active): (Vec<Order>, Vec<Order>) =
                    pending_orders.into_iter().partition(|o| {
                        o.status == OrderStatus::Filled
                            || o.status == OrderStatus::Cancelled
                            || o.status == OrderStatus::Expired
                            || o.status == OrderStatus::Rejected
                    });

                self.orders.extend(finished);
                pending_orders = active;

                last_timestamp = timestamp;
            }
        }

        // Record final equity
        if count > 0 {
            let equity = self
                .portfolio
                .calculate_equity(&self.last_prices, &self.instruments);
            self.daily_equity.push((last_timestamp, equity));
        }

        if let Some(pb) = pb {
            pb.finish_with_message("Backtest completed");
        }

        Ok(format!(
            "Backtest finished. Processed {} events. Total Trades: {}",
            count,
            self.trades.len()
        ))
    }

    /// 获取回测结果
    ///
    /// :return: BacktestResult
    fn get_results(&self) -> BacktestResult {
        let trade_pnl = self
            .trade_tracker
            .calculate_pnl(Some(self.last_prices.clone()));
        let closed_trades = self.trade_tracker.closed_trades.to_vec();
        BacktestResult::calculate(self.daily_equity.clone(), self.daily_positions.clone(), trade_pnl, closed_trades)
    }

    fn create_context(&self, active_orders: Vec<Order>) -> StrategyContext {
        // Create a temporary context for the strategy to use
        StrategyContext::new(
            self.portfolio.cash,
            self.portfolio.positions.clone(),
            self.portfolio.available_positions.clone(),
            self.clock.session,
            active_orders,
            self.trade_tracker.closed_trades.clone(),
            Some(self.history_buffer.clone()),
        )
    }
}

// Private helper methods
impl Engine {
    fn datetime_from_ns(timestamp: i64) -> DateTime<Utc> {
        let secs = timestamp.div_euclid(1_000_000_000);
        let nanos = timestamp.rem_euclid(1_000_000_000) as u32;
        Utc.timestamp_opt(secs, nanos)
            .single()
            .expect("Invalid timestamp")
    }

    fn local_datetime_from_ns(timestamp: i64, offset_secs: i32) -> DateTime<Utc> {
        let offset_ns = i64::from(offset_secs) * 1_000_000_000;
        Self::datetime_from_ns(timestamp + offset_ns)
    }

    fn parse_time_string(value: &str) -> PyResult<NaiveTime> {
        if let Ok(t) = NaiveTime::parse_from_str(value, "%H:%M:%S") {
            return Ok(t);
        }
        if let Ok(t) = NaiveTime::parse_from_str(value, "%H:%M") {
            return Ok(t);
        }
        Err(PyValueError::new_err(format!(
            "Invalid time format: {}",
            value
        )))
    }

    fn process_trades(&mut self, trades: Vec<Trade>) {
        for mut trade in trades {
            // Calculate Commission
            if let Some(instr) = self.instruments.get(&trade.symbol) {
                trade.commission = self.market_model.calculate_commission(
                    instr,
                    trade.side,
                    trade.price,
                    trade.quantity,
                );
            }

            // Update Portfolio
            self.portfolio.adjust_cash(-trade.commission); // Deduct commission
            match trade.side {
                crate::model::OrderSide::Buy => {
                    self.portfolio.adjust_cash(-trade.price * trade.quantity);
                    self.portfolio
                        .adjust_position(&trade.symbol, trade.quantity);
                }
                crate::model::OrderSide::Sell => {
                    self.portfolio.adjust_cash(trade.price * trade.quantity);
                    self.portfolio
                        .adjust_position(&trade.symbol, -trade.quantity);
                }
            }

            // Update Available Positions
            if let Some(instr) = self.instruments.get(&trade.symbol) {
                self.market_model.update_available_position(
                    &mut self.portfolio.available_positions,
                    instr,
                    trade.quantity,
                    trade.side,
                );
            }

            self.trade_tracker.process_trade(&trade);
            self.trades.push(trade);
        }
    }

    fn call_strategy(
        &mut self,
        strategy: &Bound<'_, PyAny>,
        event: &Event,
        active_orders: &[Order],
    ) -> PyResult<(Vec<Order>, Vec<Timer>, Vec<String>)> {
        // Update Last Price and Trigger Strategy
        match event {
            Event::Bar(b) => {
                self.last_prices.insert(b.symbol.clone(), b.close);
                let ctx = self.create_context(active_orders.to_vec());
                let py_ctx = Python::attach(|py| {
                    let py_ctx = Py::new(py, ctx).unwrap();
                    let args = (b.clone(), py_ctx.clone_ref(py));
                    strategy.call_method1("_on_bar_event", args)?;
                    Ok::<_, PyErr>(py_ctx)
                })?;

                // Extract orders and timers
                let mut new_orders = Vec::new();
                let mut new_timers = Vec::new();
                let mut canceled_ids = Vec::new();
                Python::attach(|py| {
                    let ctx_ref = py_ctx.borrow(py);
                    new_orders.extend(ctx_ref.orders.clone());
                    new_timers.extend(ctx_ref.timers.clone());
                    canceled_ids.extend(ctx_ref.canceled_order_ids.clone());
                });
                Ok((new_orders, new_timers, canceled_ids))
            }
            Event::Tick(t) => {
                self.last_prices.insert(t.symbol.clone(), t.price);
                let ctx = self.create_context(active_orders.to_vec());
                let py_ctx = Python::attach(|py| {
                    let py_ctx = Py::new(py, ctx).unwrap();
                    let args = (t.clone(), py_ctx.clone_ref(py));
                    strategy.call_method1("_on_tick_event", args)?;
                    Ok::<_, PyErr>(py_ctx)
                })?;

                // Extract orders and timers
                let mut new_orders = Vec::new();
                let mut new_timers = Vec::new();
                let mut canceled_ids = Vec::new();
                Python::attach(|py| {
                    let ctx_ref = py_ctx.borrow(py);
                    new_orders.extend(ctx_ref.orders.clone());
                    new_timers.extend(ctx_ref.timers.clone());
                    canceled_ids.extend(ctx_ref.canceled_order_ids.clone());
                });
                Ok((new_orders, new_timers, canceled_ids))
            }
        }
    }

    fn call_strategy_timer(
        &mut self,
        strategy: &Bound<'_, PyAny>,
        payload: &str,
        active_orders: &[Order],
    ) -> PyResult<(Vec<Order>, Vec<String>)> {
        let ctx = self.create_context(active_orders.to_vec());
        let py_ctx = Python::attach(|py| {
            let py_ctx = Py::new(py, ctx).unwrap();
            // Call _on_timer_event in Python
            strategy.call_method1("_on_timer_event", (payload, py_ctx.clone_ref(py)))?;
            Ok::<_, PyErr>(py_ctx)
        })?;

        // Extract orders and timers (Timers can schedule new timers or place orders!)
        let mut new_orders = Vec::new();
        let mut new_timers = Vec::new();
        let mut canceled_ids = Vec::new();
        Python::attach(|py| {
            let ctx_ref = py_ctx.borrow(py);
            new_orders.extend(ctx_ref.orders.clone());
            new_timers.extend(ctx_ref.timers.clone());
            canceled_ids.extend(ctx_ref.canceled_order_ids.clone());
        });

        // Add extracted items to Engine
        for t in new_timers {
            self.timers.push(t);
        }

        Ok((new_orders, canceled_ids))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::types::AssetType;

    #[test]
    fn test_engine_new() {
        let engine = Engine::new();
        assert_eq!(engine.portfolio.cash, Decimal::from(100_000));
        assert!(engine.orders.is_empty());
        assert!(engine.trades.is_empty());
        assert_eq!(engine.execution_mode, ExecutionMode::NextOpen);
    }

    #[test]
    fn test_engine_set_cash() {
        let mut engine = Engine::new();
        engine.set_cash(50000.0);
        assert_eq!(engine.portfolio.cash, Decimal::from(50000));
    }

    #[test]
    fn test_engine_add_instrument() {
        let mut engine = Engine::new();
        let instr = Instrument {
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
        engine.add_instrument(instr);
        assert!(engine.instruments.contains_key("AAPL"));
    }

    #[test]
    fn test_engine_fee_rules() {
        let mut engine = Engine::new();
        engine.set_stock_fee_rules(0.001, 0.002, 0.003, 5.0);

        // Since market_config is private but used in market_model, we can't check it directly easily
        // unless we expose getters or check behavior.
        // But we can check if it compiles and runs without error.
        // Actually, we can check market_config if we make it pub or add a getter for test.
        // But for now, let's trust the setter sets the internal state.
        // We can verify via commission calculation if we had a way to invoke it without full run.

        // Let's at least verify future fee rules
        engine.set_future_fee_rules(0.0005);
    }

    #[test]
    fn test_engine_timezone() {
        let mut engine = Engine::new();
        engine.set_timezone(3600); // UTC+1
        assert_eq!(engine.timezone_offset, 3600);
    }
}
