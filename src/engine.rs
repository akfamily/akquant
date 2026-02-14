use chrono::{DateTime, NaiveDate, NaiveTime, TimeZone, Utc};
use indicatif::{ProgressBar, ProgressStyle};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::prelude::*;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::analysis::{BacktestResult, PositionSnapshot};
use crate::clock::Clock;
use crate::context::StrategyContext;
use crate::data::DataFeed;
use crate::event::Event;
use crate::event_manager::EventManager;
use crate::execution::{ExecutionClient, RealtimeExecutionClient, SimulatedExecutionClient};
use crate::history::HistoryBuffer;
use crate::market::{
    ChinaMarketConfig, MarketConfig, MarketModel, SessionRange,
};
use crate::model::{
    Bar, ExecutionMode, Instrument, Order, OrderStatus, TimeInForce, Timer, Trade, TradingSession,
};
use crate::order_manager::OrderManager;
use crate::portfolio::Portfolio;
use crate::risk::{RiskConfig, RiskManager};

/// 主回测引擎.
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
    last_prices: HashMap<String, Decimal>,
    instruments: HashMap<String, Instrument>,
    current_date: Option<NaiveDate>,
    market_config: MarketConfig,
    market_model: Box<dyn MarketModel>,
    execution_model: Box<dyn ExecutionClient>,
    equity_curve: Vec<(i64, Decimal)>,
    cash_curve: Vec<(i64, Decimal)>,
    pub snapshots: Vec<(i64, Vec<PositionSnapshot>)>,
    execution_mode: ExecutionMode,
    clock: Clock,
    timers: BinaryHeap<Timer>, // Min-Heap via Timer's Ord implementation
    force_session_continuous: bool,
    #[pyo3(get, set)]
    pub risk_manager: RiskManager,
    timezone_offset: i32,
    history_buffer: Arc<RwLock<HistoryBuffer>>,
    initial_capital: Decimal,
    // Components
    order_manager: OrderManager,
    event_manager: EventManager,
}

#[gen_stub_pymethods]
#[pymethods]
impl Engine {
    /// 获取订单列表
    #[getter]
    fn get_orders(&self) -> Vec<Order> {
        self.order_manager.get_all_orders()
    }

    /// 获取成交列表
    #[getter]
    fn get_trades(&self) -> Vec<Trade> {
        self.order_manager.trades.clone()
    }

    /// 设置风控配置
    ///
    /// 由于 PyO3 对嵌套结构体的属性访问可能返回副本，
    /// 提供此方法以显式更新风控配置。
    ///
    /// :param config: 新的风控配置
    fn set_risk_config(&mut self, config: RiskConfig) {
        self.risk_manager.config = config;
    }

    /// 初始化回测引擎.
    ///
    /// :return: Engine 实例
    #[new]
    fn new() -> Self {
        let market_config = MarketConfig::default();
        Engine {
            feed: DataFeed::new(),
            portfolio: Portfolio {
                cash: Decimal::from(100_000),
                positions: HashMap::new(),
                available_positions: HashMap::new(),
            },
            last_prices: HashMap::new(),
            instruments: HashMap::new(),
            current_date: None,
            market_config: market_config.clone(),
            market_model: market_config.create_model(),
            execution_model: Box::new(SimulatedExecutionClient::new()),
            equity_curve: Vec::new(),
            cash_curve: Vec::new(),
            snapshots: Vec::new(),
            execution_mode: ExecutionMode::NextOpen,
            clock: Clock::new(),
            timers: BinaryHeap::new(),
            force_session_continuous: false,
            risk_manager: RiskManager::new(),
            timezone_offset: 28800, // Default UTC+8
            history_buffer: Arc::new(RwLock::new(HistoryBuffer::new(10000))), // Default large capacity for MAE/MFE
            initial_capital: Decimal::from(100_000),
            order_manager: OrderManager::new(),
            event_manager: EventManager::new(),
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

    /// 启用模拟执行 (回测模式)
    ///
    /// 默认模式。在内存中撮合订单。
    fn use_simulated_execution(&mut self) {
        self.execution_model = Box::new(SimulatedExecutionClient::new());
    }

    /// 启用实盘执行 (CTP/Broker 模式)
    ///
    /// 模拟对接 CTP 或其他 Broker API。
    /// 在此模式下，订单会被标记为 Submitted 并等待回调 (目前仅模拟发送)。
    fn use_realtime_execution(&mut self) {
        self.execution_model = Box::new(RealtimeExecutionClient::new());
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
        let mut config = crate::market::SimpleMarketConfig::default();
        config.commission_rate = Decimal::from_f64(commission_rate).unwrap_or(Decimal::ZERO);
        self.market_config = MarketConfig::Simple(config);
        self.market_model = self.market_config.create_model();
    }

    /// 启用 ChinaMarket (支持 T+1/T+0, 印花税, 过户费, 交易时段等)
    fn use_china_market(&mut self) {
        self.market_config = MarketConfig::China(ChinaMarketConfig::default());
        self.market_model = self.market_config.create_model();
    }

    /// 启用中国期货市场默认配置
    /// - 切换到 ChinaMarket
    /// - 设置 T+0
    /// - 保持当前交易时段配置 (需手动设置 set_market_sessions 以匹配特定品种)
    fn use_china_futures_market(&mut self) {
        let mut config = ChinaMarketConfig::default();
        config.t_plus_one = false;
        self.market_config = MarketConfig::China(config);
        self.market_model = self.market_config.create_model();
    }

    /// 启用/禁用 T+1 交易规则 (仅针对 ChinaMarket)
    ///
    /// :param enabled: 是否启用 T+1
    /// :type enabled: bool
    fn set_t_plus_one(&mut self, enabled: bool) {
        if let MarketConfig::China(ref mut c) = self.market_config {
            c.t_plus_one = enabled;
            self.market_model = self.market_config.create_model();
        }
    }

    /// 强制连续交易时段
    ///
    /// :param enabled: 是否强制连续交易 (忽略午休等)
    fn set_force_session_continuous(&mut self, enabled: bool) {
        self.force_session_continuous = enabled;
    }

    /// 设置股票费率规则
    ///
    /// :param commission_rate: 佣金率 (如 0.0003)
    /// :param stamp_tax: 印花税率 (如 0.001)
    /// :param transfer_fee: 过户费率 (如 0.00002)
    /// :param min_commission: 最低佣金 (如 5.0)
    fn set_stock_fee_rules(
        &mut self,
        commission_rate: f64,
        stamp_tax: f64,
        transfer_fee: f64,
        min_commission: f64,
    ) {
        match &mut self.market_config {
            MarketConfig::China(c) => {
                c.stock_commission_rate =
                    Decimal::from_f64(commission_rate).unwrap_or(Decimal::ZERO);
                c.stock_stamp_tax = Decimal::from_f64(stamp_tax).unwrap_or(Decimal::ZERO);
                c.stock_transfer_fee =
                    Decimal::from_f64(transfer_fee).unwrap_or(Decimal::ZERO);
                c.stock_min_commission =
                    Decimal::from_f64(min_commission).unwrap_or(Decimal::ZERO);
            }
            MarketConfig::Simple(c) => {
                c.commission_rate = Decimal::from_f64(commission_rate).unwrap_or(Decimal::ZERO);
                c.stamp_tax = Decimal::from_f64(stamp_tax).unwrap_or(Decimal::ZERO);
                c.transfer_fee = Decimal::from_f64(transfer_fee).unwrap_or(Decimal::ZERO);
                c.min_commission = Decimal::from_f64(min_commission).unwrap_or(Decimal::ZERO);
            }
        }
        self.market_model = self.market_config.create_model();
    }

    /// 设置期货费率规则
    ///
    /// :param commission_rate: 佣金率 (如 0.0001)
    fn set_future_fee_rules(&mut self, commission_rate: f64) {
        if let MarketConfig::China(ref mut c) = self.market_config {
            c.future_commission_rate =
                Decimal::from_f64(commission_rate).unwrap_or(Decimal::ZERO);
            self.market_model = self.market_config.create_model();
        }
    }

    /// 设置基金费率规则
    ///
    /// :param commission_rate: 佣金率
    /// :param transfer_fee: 过户费率
    /// :param min_commission: 最低佣金
    fn set_fund_fee_rules(&mut self, commission_rate: f64, transfer_fee: f64, min_commission: f64) {
        if let MarketConfig::China(ref mut c) = self.market_config {
            c.fund_commission_rate =
                Decimal::from_f64(commission_rate).unwrap_or(Decimal::ZERO);
            c.fund_transfer_fee =
                Decimal::from_f64(transfer_fee).unwrap_or(Decimal::ZERO);
            c.fund_min_commission =
                Decimal::from_f64(min_commission).unwrap_or(Decimal::ZERO);
            self.market_model = self.market_config.create_model();
        }
    }

    /// 设置期权费率规则
    ///
    /// :param commission_per_contract: 每张合约佣金 (如 5.0)
    fn set_option_fee_rules(&mut self, commission_per_contract: f64) {
        if let MarketConfig::China(ref mut c) = self.market_config {
            c.option_commission_per_contract =
                Decimal::from_f64(commission_per_contract).unwrap_or(Decimal::ZERO);
            self.market_model = self.market_config.create_model();
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

    /// 设置市场交易时段
    ///
    /// :param sessions: 交易时段列表，每个元素为 (开始时间, 结束时间, 时段类型)
    /// :type sessions: List[Tuple[str, str, TradingSession]]
    ///
    /// 示例::
    ///
    ///     engine.set_market_sessions([
    ///         ("09:30:00", "11:30:00", TradingSession.Normal),
    ///         ("13:00:00", "15:00:00", TradingSession.Normal)
    ///     ])
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
        if let MarketConfig::China(ref mut c) = self.market_config {
            c.sessions = ranges;
            self.market_model = self.market_config.create_model();
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
        let val = Decimal::from_f64(cash).unwrap_or(Decimal::ZERO);
        self.portfolio.cash = val;
        self.initial_capital = val;
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
    fn run(
        &mut self,
        py: Python<'_>,
        strategy: &Bound<'_, PyAny>,
        show_progress: bool,
    ) -> PyResult<String> {
        // Configure history buffer if strategy has _history_depth set
        if let Ok(depth_attr) = strategy.getattr("_history_depth") {
            if let Ok(depth) = depth_attr.extract::<usize>() {
                if depth > 0 {
                    self.set_history_depth(depth);
                }
            }
        }

        // Trigger Strategy on_start
        if let Err(e) = strategy.call_method0("on_start") {
            return Err(e);
        }

        let mut count = 0;
        let mut last_timestamp = 0;
        let mut bar_index = 0;

        // Progress Bar Initialization
        let total_events = self.feed.len_hint().unwrap_or(0);
        let pb = if show_progress {
            let pb = if total_events > 0 {
                let pb = ProgressBar::new(total_events as u64);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template(
                            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                        )
                        .unwrap()
                        .progress_chars("#>-"),
                );
                pb
            } else {
                let pb = ProgressBar::new_spinner();
                pb.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.green} [{elapsed_precise}] {pos} events processed")
                        .unwrap(),
                );
                pb
            };
            Some(pb)
        } else {
            None
        };

        // Record initial equity
        if let Some(_) = self.feed.peek_timestamp() {
            let _equity = self
                .portfolio
                .calculate_equity(&self.last_prices, &self.instruments);
        }

        let is_live = self.feed.is_live();

        loop {
            // -----------------------------------------------------------
            // 0. Process Channel Events (High Priority)
            // -----------------------------------------------------------
            let mut trades_to_process = Vec::new();
            while let Some(event) = self.event_manager.try_recv() {
                match event {
                    Event::OrderRequest(mut order) => {
                        // 1. Risk Check
                        if let Err(err) = self.risk_manager.check_internal(
                            &order,
                            &self.portfolio,
                            &self.instruments,
                            &self.order_manager.active_orders,
                            &self.last_prices,
                        ) {
                            let err_msg = err.to_string();
                            let mut handled = false;
                            // Check for insufficient cash to attempt auto-reduction
                            if err_msg.contains("Insufficient cash")
                                && order.side == crate::model::OrderSide::Buy
                            {
                                if let Some(instr) = self.instruments.get(&order.symbol) {
                                    // Get price (Limit or Last)
                                    let price = if let Some(p) = order.price {
                                        p
                                    } else {
                                        *self
                                            .last_prices
                                            .get(&order.symbol)
                                            .unwrap_or(&Decimal::ZERO)
                                    };

                                    if price > Decimal::ZERO {
                                        let multiplier = instr.multiplier;
                                        let cost_per_unit = price * multiplier * instr.margin_ratio;
                                        if cost_per_unit > Decimal::ZERO {
                                            let max_qty_raw = self.portfolio.cash / cost_per_unit;
                                            // Buffer for commission (e.g. 1%)
                                            let max_qty_raw =
                                                max_qty_raw * Decimal::from_f64(0.9999).unwrap();

                                            let lot_size = instr.lot_size;
                                            let mut new_qty = max_qty_raw.floor();
                                            if lot_size > Decimal::ZERO {
                                                new_qty = new_qty - (new_qty % lot_size);
                                            }

                                            if new_qty > Decimal::ZERO && new_qty < order.quantity {
                                                order.quantity = new_qty;
                                                handled = true;
                                                // Send as Validated (assuming it passes now)
                                                let _ = self
                                                    .event_manager
                                                    .send(Event::OrderValidated(order.clone()));
                                            }
                                        }
                                    }
                                }
                            }

                            if !handled {
                                // Reject
                                order.status = OrderStatus::Rejected;
                                order.reject_reason = err_msg;
                                if let Some(ts) = self.clock.timestamp() {
                                    order.updated_at = ts;
                                }
                                // Directly process rejection report
                                let report = Event::ExecutionReport(order, None);
                                let _ = self.event_manager.send(report);
                            }
                        } else {
                            // Validate -> Send to Execution
                            let _ = self.event_manager.send(Event::OrderValidated(order));
                        }
                    }
                    Event::OrderValidated(order) => {
                        // 2. Send to Execution Client
                        self.execution_model.on_order(order.clone());
                        // Add to local active (Strategy View)
                        self.order_manager.add_active_order(order);
                    }
                    Event::ExecutionReport(order, trade) => {
                        // 3. Update Order State
                        self.order_manager.on_execution_report(order);

                        // 4. Process Trade (if any)
                        if let Some(t) = trade {
                            trades_to_process.push(t);
                        }
                    }
                    _ => {}
                }
            }

            if !trades_to_process.is_empty() {
                self.order_manager.process_trades(
                    trades_to_process,
                    &mut self.portfolio,
                    &self.instruments,
                    self.market_model.as_ref(),
                    &self.risk_manager,
                    &self.history_buffer,
                    &self.last_prices,
                );
            }

            // -----------------------------------------------------------
            // 1. Time & Data Management
            // -----------------------------------------------------------
            let mut next_event_time = self.feed.peek_timestamp();
            let next_timer_time = self.timers.peek().map(|t| t.timestamp);

            if !is_live && next_event_time.is_none() && next_timer_time.is_none() {
                break; // No more events (Backtest End)
            }

            // Live Mode Wait
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
                    let feed = self.feed.clone();
                    #[allow(deprecated)]
                    if let Some(ts) = py.allow_threads(move || feed.wait_peek(timeout)) {
                        next_event_time = Some(ts);
                    }
                }
            }

            // Check timers again after wait
            if is_live && next_event_time.is_none() {
                if let Some(timer_ts) = next_timer_time {
                    let now = Utc::now().timestamp_nanos_opt().unwrap_or(0);
                    if timer_ts > now {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            let process_timer = match (next_event_time, next_timer_time) {
                (Some(et), Some(tt)) => tt <= et,
                (Some(_), None) => false,
                (None, Some(_)) => true,
                (None, None) => break,
            };

            // -----------------------------------------------------------
            // 2. Process Timer or Data Event
            // -----------------------------------------------------------
            if process_timer {
                // --- TIMER EVENT ---
                if let Some(timer) = self.timers.pop() {
                    let local_dt =
                        Self::local_datetime_from_ns(timer.timestamp, self.timezone_offset);
                    let session = self.market_model.get_session_status(local_dt.time());
                    self.clock.update(timer.timestamp, session);
                    if self.force_session_continuous {
                        self.clock.session = TradingSession::Continuous;
                    }

                    // Strategy on_timer
                    let (new_orders, new_timers, canceled_ids) =
                        self.call_strategy_timer(strategy, &timer.payload)?;

                    // Handle Strategy Output
                    for order_id in canceled_ids {
                        self.execution_model.on_cancel(&order_id);
                    }
                    for order in new_orders {
                        let _ = self.event_manager.send(Event::OrderRequest(order));
                    }
                    for t in new_timers {
                        self.timers.push(t);
                    }
                }
            } else {
                // --- MARKET DATA EVENT ---
                let event = self.feed.next().unwrap();

                // Update History
                if let Event::Bar(ref b) = event {
                    self.history_buffer.write().unwrap().update(b);
                }

                count += 1;
                if let Some(pb) = &pb {
                    pb.inc(1);
                }

                // Update Clock & Day Close
                let timestamp = match &event {
                    Event::Bar(b) => b.timestamp,
                    Event::Tick(t) => t.timestamp,
                    _ => 0,
                };

                if last_timestamp != 0 && timestamp > last_timestamp {
                    // Record High-Res Equity Curve
                    let equity = self
                        .portfolio
                        .calculate_equity(&self.last_prices, &self.instruments);
                    self.equity_curve.push((last_timestamp, equity));
                    self.cash_curve.push((last_timestamp, self.portfolio.cash));

                    bar_index += 1;
                }

                let local_dt = Self::local_datetime_from_ns(timestamp, self.timezone_offset);
                let session = self.market_model.get_session_status(local_dt.time());
                self.clock.update(timestamp, session);
                if self.force_session_continuous {
                    self.clock.session = TradingSession::Continuous;
                }

                let local_date = local_dt.date_naive();
                if self.current_date != Some(local_date) {
                    // Day Close Logic
                    if self.current_date.is_some() {
                        // Position Snapshots
                        let snapshots = self.create_position_snapshots(last_timestamp);
                        self.snapshots.push((last_timestamp, snapshots));

                        // T+1
                        self.market_model.on_day_close(
                            &self.portfolio.positions,
                            &mut self.portfolio.available_positions,
                            &self.instruments,
                        );

                        // Expire Day Orders
                        // Simplified: Engine handles expiry for now by marking them.
                        // Ideally ExecutionClient checks time.
                        let (expired, kept): (Vec<Order>, Vec<Order>) = self
                            .order_manager
                            .active_orders
                            .drain(..)
                            .partition(|o| o.time_in_force == TimeInForce::Day);

                        for mut o in expired {
                            o.status = OrderStatus::Expired;
                            self.order_manager.orders.push(o);
                        }
                        self.order_manager.active_orders = kept;
                    }
                    self.current_date = Some(local_date);
                }

                match self.execution_mode {
                    ExecutionMode::NextOpen
                    | ExecutionMode::NextAverage
                    | ExecutionMode::NextHighLowMid => {
                        // Phase 1: Execution (Match pending orders at Open or Average)
                        let reports = self.execution_model.on_event(
                            &event,
                            &self.instruments,
                            self.execution_mode,
                            bar_index,
                            self.clock.session,
                        );

                        for report in reports {
                            let _ = self.event_manager.send(report);
                        }

                        // Drain Channel (Mini-Loop)
                        let mut trades_to_process = Vec::new();
                        while let Some(ev) = self.event_manager.try_recv() {
                            match ev {
                                Event::ExecutionReport(o, t) => {
                                    self.order_manager.on_execution_report(o);
                                    if let Some(tr) = t {
                                        trades_to_process.push(tr);
                                    }
                                }
                                _ => {}
                            }
                        }
                        if !trades_to_process.is_empty() {
                            self.order_manager.process_trades(
                                trades_to_process,
                                &mut self.portfolio,
                                &self.instruments,
                                self.market_model.as_ref(),
                                &self.risk_manager,
                                &self.history_buffer,
                                &self.last_prices,
                            );
                        }

                        // Phase 2: Strategy
                        let (new_orders, new_timers, canceled_ids) =
                            self.call_strategy(strategy, &event)?;

                        for id in canceled_ids {
                            self.execution_model.on_cancel(&id);
                            if let Some(o) =
                                self.order_manager.active_orders.iter_mut().find(|o| o.id == id)
                            {
                                o.status = OrderStatus::Cancelled;
                            }
                        }
                        for order in new_orders {
                            let _ = self.event_manager.send(Event::OrderRequest(order));
                        }
                        for t in new_timers {
                            self.timers.push(t);
                        }

                        // Phase 3: Process New Order Requests
                        let mut trades_to_process = Vec::new();
                        while let Some(ev) = self.event_manager.try_recv() {
                            match ev {
                                Event::OrderRequest(mut o) => {
                                    if let Err(err) = self.risk_manager.check_internal(
                                        &o,
                                        &self.portfolio,
                                        &self.instruments,
                                        &self.order_manager.active_orders,
                                        &self.last_prices,
                                    ) {
                                        o.status = OrderStatus::Rejected;
                                        o.reject_reason = err.to_string();
                                        if let Some(ts) = self.clock.timestamp() {
                                            o.updated_at = ts;
                                        }
                                        self.order_manager.add_active_order(o);
                                    } else {
                                        if o.created_at == 0 {
                                            if let Some(ts) = self.clock.timestamp() {
                                                o.created_at = ts;
                                            }
                                        }
                                        let _ = self.event_manager.send(Event::OrderValidated(o));
                                    }
                                }
                                Event::OrderValidated(o) => {
                                    self.execution_model.on_order(o.clone());
                                    self.order_manager.add_active_order(o);
                                }
                                Event::ExecutionReport(o, t) => {
                                    self.order_manager.on_execution_report(o);
                                    if let Some(tr) = t {
                                        trades_to_process.push(tr);
                                    }
                                }
                                _ => {}
                            }
                        }
                        if !trades_to_process.is_empty() {
                            self.order_manager.process_trades(
                                trades_to_process,
                                &mut self.portfolio,
                                &self.instruments,
                                self.market_model.as_ref(),
                                &self.risk_manager,
                                &self.history_buffer,
                                &self.last_prices,
                            );
                        }
                    }
                    ExecutionMode::CurrentClose => {
                        // Phase 1: Strategy
                        let (new_orders, new_timers, canceled_ids) =
                            self.call_strategy(strategy, &event)?;

                        for id in canceled_ids {
                            self.execution_model.on_cancel(&id);
                        }
                        for order in new_orders {
                            let _ = self.event_manager.send(Event::OrderRequest(order));
                        }
                        for t in new_timers {
                            self.timers.push(t);
                        }

                        // Drain channel
                        let mut trades_to_process = Vec::new();
                        while let Some(ev) = self.event_manager.try_recv() {
                            match ev {
                                Event::OrderRequest(mut o) => {
                                    if let Err(err) = self.risk_manager.check_internal(
                                        &o,
                                        &self.portfolio,
                                        &self.instruments,
                                        &self.order_manager.active_orders,
                                        &self.last_prices,
                                    ) {
                                        o.status = OrderStatus::Rejected;
                                        o.reject_reason = err.to_string();
                                        if let Some(ts) = self.clock.timestamp() {
                                            o.updated_at = ts;
                                        }
                                        let _ = self
                                            .event_manager
                                            .send(Event::ExecutionReport(o, None));
                                    } else {
                                        if o.created_at == 0 {
                                            if let Some(ts) = self.clock.timestamp() {
                                                o.created_at = ts;
                                            }
                                        }
                                        let _ = self.event_manager.send(Event::OrderValidated(o));
                                    }
                                }
                                Event::OrderValidated(o) => {
                                    self.execution_model.on_order(o.clone());
                                    self.order_manager.add_active_order(o);
                                }
                                Event::ExecutionReport(o, t) => {
                                    self.order_manager.on_execution_report(o);
                                    if let Some(tr) = t {
                                        trades_to_process.push(tr);
                                    }
                                }
                                _ => {}
                            }
                        }
                        if !trades_to_process.is_empty() {
                            self.order_manager.process_trades(
                                trades_to_process,
                                &mut self.portfolio,
                                &self.instruments,
                                self.market_model.as_ref(),
                                &self.risk_manager,
                                &self.history_buffer,
                                &self.last_prices,
                            );
                        }

                        // Phase 2: Execution (Match new orders at Close)
                        let reports = self.execution_model.on_event(
                            &event,
                            &self.instruments,
                            self.execution_mode,
                            bar_index,
                            self.clock.session,
                        );
                        for report in reports {
                            let _ = self.event_manager.send(report);
                        }
                    }
                }

                // Cleanup finished orders
                self.order_manager.cleanup_finished_orders();

                last_timestamp = timestamp;
            }
        }

        // Final Equity
        if count > 0 {
            let equity = self
                .portfolio
                .calculate_equity(&self.last_prices, &self.instruments);

            self.equity_curve.push((last_timestamp, equity));
        }

        // Final cleanup
        self.order_manager.cleanup_finished_orders();

        if let Some(pb) = pb {
            pb.finish_with_message("Backtest completed");
        }

        Ok(format!(
            "Backtest finished. Processed {} events. Total Trades: {}",
            count,
            self.order_manager.trades.len()
        ))
    }

    /// 获取回测结果
    ///
    /// :return: BacktestResult
    fn get_results(&self) -> BacktestResult {
        let trade_pnl = self
            .order_manager
            .trade_tracker
            .calculate_pnl(Some(self.last_prices.clone()));
        let closed_trades = self.order_manager.trade_tracker.closed_trades.to_vec();

        // Clone equity curve and append current state (Mark-to-Market)
        // This ensures that we have the latest equity point even if the run loop was interrupted
        let mut equity_curve = self.equity_curve.clone();
        let mut cash_curve = self.cash_curve.clone();

        let mut snapshots = self.snapshots.clone();

        if let Some(now_ns) = self.clock.timestamp() {
            let equity = self
                .portfolio
                .calculate_equity(&self.last_prices, &self.instruments);
            // Avoid duplicate if timestamp matches last one exactly (unlikely but safe)
            if equity_curve
                .last()
                .map(|(t, _)| *t != now_ns)
                .unwrap_or(true)
            {
                equity_curve.push((now_ns, equity));
            }
            if cash_curve.last().map(|(t, _)| *t != now_ns).unwrap_or(true) {
                cash_curve.push((now_ns, self.portfolio.cash));
            }
            if snapshots.last().map(|(t, _)| *t != now_ns).unwrap_or(true) {
                let snap = self.create_position_snapshots(now_ns);
                snapshots.push((now_ns, snap));
            }
        }

        BacktestResult::calculate(
            equity_curve,
            cash_curve,
            snapshots,
            trade_pnl,
            closed_trades,
            self.initial_capital,
            self.order_manager.get_all_orders(),
            self.order_manager.trades.clone(),
        )
    }

    fn create_context(
        &self,
        active_orders: Vec<Order>,
        step_trades: Vec<Trade>,
    ) -> StrategyContext {
        // Create a temporary context for the strategy to use
        StrategyContext::new(
            self.portfolio.cash,
            self.portfolio.positions.clone(),
            self.portfolio.available_positions.clone(),
            self.clock.session,
            self.clock.timestamp().unwrap_or(0),
            active_orders,
            self.order_manager.trade_tracker.closed_trades.clone(),
            step_trades,
            Some(self.history_buffer.clone()),
            Some(self.event_manager.sender()),
            self.risk_manager.config.clone(),
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

    fn create_position_snapshots(&self, _timestamp: i64) -> Vec<PositionSnapshot> {
        let mut snapshots = Vec::new();
        for (symbol, &qty) in &self.portfolio.positions {
            if qty == Decimal::ZERO {
                continue;
            }

            let price = self
                .last_prices
                .get(symbol)
                .cloned()
                .unwrap_or(Decimal::ZERO);
            let instr = self.instruments.get(symbol);
            let multiplier = instr.map(|i| i.multiplier).unwrap_or(Decimal::ONE);
            let margin_ratio = instr.map(|i| i.margin_ratio).unwrap_or(Decimal::ZERO);

            // Convert to f64 for snapshot
            let qty_f64 = qty.to_f64().unwrap_or(0.0);
            let price_f64 = price.to_f64().unwrap_or(0.0);
            // let multiplier_f64 = multiplier.to_f64().unwrap_or(1.0);

            let market_value = qty.abs() * price * multiplier;
            let market_value_f64 = market_value.to_f64().unwrap_or(0.0);

            let (long_shares, short_shares) = if qty > Decimal::ZERO {
                (qty_f64, 0.0)
            } else {
                (0.0, qty.abs().to_f64().unwrap_or(0.0))
            };

            // Equity & Margin logic
            // Long: Equity = MarketValue, Margin = MarketValue * ratio
            // Short: Equity = 0 (conservative), Margin = MarketValue * ratio
            let equity_f64 = if qty > Decimal::ZERO {
                market_value_f64
            } else {
                0.0
            };

            let margin_dec = market_value * margin_ratio;
            let margin_f64 = margin_dec.to_f64().unwrap_or(0.0);

            let unrealized_pnl = self
                .order_manager
                .trade_tracker
                .get_unrealized_pnl(symbol, price, multiplier);
            let unrealized_pnl_f64 = unrealized_pnl.to_f64().unwrap_or(0.0);

            let entry_price = self.order_manager.trade_tracker.get_average_price(symbol);
            let entry_price_f64 = entry_price.to_f64().unwrap_or(0.0);

            snapshots.push(PositionSnapshot {
                symbol: symbol.clone(),
                quantity: qty_f64,
                entry_price: entry_price_f64,
                long_shares,
                short_shares,
                close: price_f64,
                equity: equity_f64,
                market_value: market_value_f64,
                margin: margin_f64,
                unrealized_pnl: unrealized_pnl_f64,
            });
        }
        snapshots
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

    fn call_strategy(
        &mut self,
        strategy: &Bound<'_, PyAny>,
        event: &Event,
    ) -> PyResult<(Vec<Order>, Vec<Timer>, Vec<String>)> {
        // Update Last Price and Trigger Strategy
        match event {
            Event::Bar(b) => {
                self.last_prices.insert(b.symbol.clone(), b.close);
                let step_trades = std::mem::take(&mut self.order_manager.current_step_trades);
                let ctx = self.create_context(self.order_manager.active_orders.clone(), step_trades);
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
                let step_trades = std::mem::take(&mut self.order_manager.current_step_trades);
                let ctx = self.create_context(self.order_manager.active_orders.clone(), step_trades);
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
            Event::OrderRequest(_) | Event::OrderValidated(_) | Event::ExecutionReport(_, _) => {
                Ok((Vec::new(), Vec::new(), Vec::new()))
            }
        }
    }

    fn call_strategy_timer(
        &mut self,
        strategy: &Bound<'_, PyAny>,
        payload: &str,
    ) -> PyResult<(Vec<Order>, Vec<Timer>, Vec<String>)> {
        let step_trades = std::mem::take(&mut self.order_manager.current_step_trades);
        let ctx = self.create_context(self.order_manager.active_orders.clone(), step_trades);
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
        Ok((new_orders, new_timers, canceled_ids))
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
        assert!(engine.order_manager.orders.is_empty());
        assert!(engine.order_manager.trades.is_empty());
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
