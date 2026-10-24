use crate::account::{AccountMetrics, calculate_account_metrics};
#[cfg(test)]
use chrono::NaiveTime;
use chrono::{DateTime, NaiveDate, TimeZone, Timelike, Utc};
use chrono_tz::Tz;
use indicatif::ProgressBar;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::*;
use rust_decimal::prelude::*;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, RwLock};
use uuid::Uuid;

// use crate::analysis::{BacktestResult, PositionSnapshot};
use crate::clock::Clock;
use crate::context::ExpiryEvent;
use crate::context::StrategyContext;
use crate::event::Event;
use crate::event_manager::EventManager;
use crate::execution::ExecutionClient;
use crate::history::HistoryBuffer;
use crate::market::corporate_action::CorporateActionManager;
use crate::market::manager::MarketManager;
use crate::model::{Bar, ExecutionPolicyCore, Instrument, Order, OrderSide, Timer, Trade};
use crate::pipeline::PipelineRunner;
use crate::pipeline::stages::{
    ChannelProcessor, CleanupProcessor, DataProcessor, ExecutionPhase, ExecutionProcessor,
    StatisticsProcessor, StrategyProcessor,
};
use crate::risk::RiskManager;
use crate::settlement::SettlementManager;
use crate::statistics::StatisticsManager;

use super::state::SharedState;

/// “行情落在闭市时段”逐品种告警的全局上限；超出的品种在回测结束时汇总一条。
pub(crate) const CLOSED_SESSION_WARN_LIMIT: usize = 20;

/// 用户配置的交易时段表与交易日历（`configure_sessions` 写入，随存档保存）。
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SessionConfig {
    /// 模板项：(键, 模板名)。
    pub sessions: Option<Vec<(String, String)>>,
    pub trading_calendar: Option<Vec<NaiveDate>>,
    /// 自定义时段表项：(键, 段序列)。存档字段只能追加在末尾：旧存档缺此项时为 None。
    #[serde(default)]
    pub custom_sessions: Option<Vec<(String, Vec<crate::calendar::schedule::CustomSegment>)>>,
}

impl SessionConfig {
    /// 合并模板项与自定义项；两者都未配置时为 None。
    pub(crate) fn session_specs(
        &self,
    ) -> Option<Vec<(String, crate::calendar::resolver::SessionSpec)>> {
        use crate::calendar::resolver::SessionSpec;
        if self.sessions.is_none() && self.custom_sessions.is_none() {
            return None;
        }
        let mut specs: Vec<(String, SessionSpec)> = Vec::new();
        for (k, name) in self.sessions.iter().flatten() {
            specs.push((k.clone(), SessionSpec::Template(name.clone())));
        }
        for (k, segs) in self.custom_sessions.iter().flatten() {
            specs.push((k.clone(), SessionSpec::Custom(segs.clone())));
        }
        Some(specs)
    }
}

#[derive(Debug, Clone)]
pub struct StrategySlot {
    pub(crate) strategy_id: String,
}

/// 各交易日生命周期钩子是否有策略重写（见 `Engine::lifecycle_hooks_dispatch`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LifecycleHooksDispatch {
    pub(crate) pre_open: bool,
    pub(crate) before_trading: bool,
    pub(crate) after_trading: bool,
}

impl Default for LifecycleHooksDispatch {
    /// 直接驱动引擎（不经框架入口）时全部派发。
    fn default() -> Self {
        Self {
            pre_open: true,
            before_trading: true,
            after_trading: true,
        }
    }
}

/// 主回测引擎.
///
/// :ivar feed: 数据源
/// :ivar portfolio: 投资组合
/// :ivar orders: 订单列表
/// :ivar trades: 成交列表
#[gen_stub_pyclass]
#[pyclass]
pub struct Engine {
    pub(crate) state: SharedState,
    /// 每 bar 交给 StrategyContext 的价格表。用 Arc<RwLock<..>> 而非每 bar
    /// 克隆:该表从不暴露给 Python(无 getter,Python 侧另有 strategy._last_prices),
    /// 消费者仅在回调期间做键查找,故共享活视图与逐 bar 快照取值等价。
    ///
    /// 锁纪律(真实机制,而非"guard 不跨 Python"这种过于乐观的表述):
    /// 读锁**确实会**跨越到 Python 里去——`pipeline/stages/{channel,execution,
    /// data}.rs` 里构造 EngineContext 时持有的读锁,会经
    /// `execution_model.on_event` / `finalize_timestamp` 一路调进
    /// `SimulatedExecutionClient` 再到用户通过 `register_custom_matcher`
    /// 注册的自定义撮合器(`PyExecutionMatcher::match_order` 内部
    /// `Python::attach` + `call_method1`),届时读锁是跨 Python 调用存活的。
    /// 真正兜底的是:写锁只有 3 处(本文件 Bar/Tick 分支的 get-or-insert 快
    /// 路径、python.rs 快照恢复),而这 3 处全部要求 `&mut Engine`——
    /// `run(&mut self, ...)` 执行期间 Engine 的 PyCell 处于可变借用中,任何
    /// 从自定义撮合器的 Python 代码再入 Engine 的 pymethod 都会触发
    /// `PyBorrowMutError` 而不是被挂起,所以今天不会死锁。这条防线依赖
    /// **不存在任何 `&self` 可达的 Python API 会去拿写锁**——`StrategyContext`
    /// 目前也只有读路径(`get_buying_power`),同线程嵌套读在无写者排队时
    /// 是安全的。一旦未来出现"以 `&self` 暴露、内部却写 last_prices"的
    /// pymethod,或 StrategyContext 长出写路径,或撮合器路径上加了
    /// `py.detach()`/`allow_threads`,这条防线就会失效并造成永久死锁
    /// (std 的 RwLock 写锁请求会排队等待,不可重入)。
    pub(crate) last_prices: Arc<RwLock<HashMap<String, Decimal>>>,
    pub(crate) instruments: HashMap<String, Instrument>,
    /// 会话级时段解析器；`run()` 开始时按 `session_config` 与时区构造（见
    /// [`Engine::ensure_session_resolver`]），配置或时区变化时置 `None` 以便重建。
    /// 包在 `Arc` 里供 `StrategyContext` 廉价共享（解析器内部缓存用 `Mutex`）。
    pub(crate) session_resolver: Option<Arc<crate::calendar::resolver::SessionResolver>>,
    /// 当前所处的会话交易日（最近一个已开日的交易日）。
    pub(crate) current_trading_day: Option<NaiveDate>,
    /// `current_trading_day` 覆盖的时间区间 `(上一交易日收盘, 本交易日收盘]`（纳秒）。
    pub(crate) current_day_span: Option<(i64, i64)>,
    /// 本次运行实际开过的交易日及其收盘时刻 `(交易日, 收盘纳秒)`，按开日顺序追加。
    /// 日收益/日权益按它把权益点归入交易日（夜盘点属下一交易日）；不入存档，
    /// 续跑结果只覆盖续跑段开过（含续跑时重建跨度）的交易日。
    pub(crate) trading_day_ends: Vec<(NaiveDate, i64)>,
    /// 最近一个已完成日终结算的交易日。
    pub(crate) last_settled_trading_day: Option<NaiveDate>,
    /// 当前交易日最后一个已处理行情事件的时间戳（Day 单豁免判据）。
    pub(crate) last_event_ts_in_day: Option<i64>,
    /// 已登记收盘定时器的交易日（避免重复登记）。
    pub(crate) scheduled_day_close: Option<NaiveDate>,
    /// 已排程生命周期定时器（pre_open / before_trading）的最晚交易日（避免重复排程；
    /// 不入存档，续跑时 `run()` 按存档时刻重新排程并跳过已过时刻）。
    pub(crate) lifecycle_scheduled_for: Option<NaiveDate>,
    /// 实盘生命周期串联因交易日历到尽头失败时是否已告警（只告警一次）。
    pub(crate) lifecycle_chain_warned: bool,
    /// 各生命周期钩子（on_pre_open / on_before_trading / on_after_trading）是否有策略
    /// （含槽位）重写，由 Python 在 run 开始时经 `set_lifecycle_hooks_dispatch` 告知
    /// （直接驱动引擎时缺省全为真）。未重写的盘前/开始交易事件只在引擎内消费（实盘
    /// 串联仍依赖开始交易），不派发给策略；收盘后事件未重写时只在有过期单或到期事件
    /// 要送达时派发。
    pub(crate) lifecycle_hooks_dispatch: LifecycleHooksDispatch,
    /// 公司行动已处理到的日期(含)。开日时处理 (该日期, 当日交易日] 内的全部行动。
    pub(crate) last_corp_action_date: Option<NaiveDate>,
    pub(crate) market_manager: MarketManager,
    pub(crate) corporate_action_manager: CorporateActionManager,
    pub(crate) execution_model: Box<dyn ExecutionClient>,
    pub(crate) execution_policy_core_state: ExecutionPolicyCore,
    pub(crate) clock: Clock,
    pub(crate) timers: BinaryHeap<Timer>, // Min-Heap via Timer's Ord implementation
    pub(crate) force_session_continuous: bool,
    /// 当前时间片各品种的时段（行情事件按 bar 区间解析）；时间戳推进时清空。
    /// 修改走 `Arc::make_mut`；`StrategyContext` 只持 `Weak`，不会迫使逐事件复制整表。
    pub(crate) slice_sessions: Arc<HashMap<String, crate::model::TradingSession>>,
    /// 各品种最近一次解析所在的时段区间 `[start, end)` 与时段（`event_session` 缓存）；
    /// 解析器重建时清空。
    pub(crate) session_span_cache: HashMap<String, (i64, i64, crate::model::TradingSession)>,
    /// 跨多段区间的解析缓存：最近一次 ((时段表指针, interval_start, ts), 时段)；
    /// 时段表指针来自当前解析器，解析器重建时清空。
    pub(crate) last_interval_session:
        Option<((usize, Option<i64>, i64), crate::model::TradingSession)>,
    /// 在非连续竞价时段创建的在途订单 → 创建时间片的品种时段（日终 Day 单豁免用；
    /// 日终结算后只保留仍在途的订单）。
    pub(crate) order_creation_sessions: HashMap<String, crate::model::TradingSession>,
    /// 出现过“行情落在闭市时段”的品种。每个品种最多告警一次，全局最多
    /// [`CLOSED_SESSION_WARN_LIMIT`] 条，其余在回测结束时汇总一条。
    pub(crate) closed_session_warned: HashSet<String>,
    /// 落在闭市时段的行情事件总数（回测结束汇总用）。
    pub(crate) closed_session_event_count: u64,
    #[pyo3(get, set)]
    pub risk_manager: RiskManager,
    pub(crate) timezone_offset: i32,
    pub(crate) timezone_name: Option<String>,
    /// 年化天数因子。传统市场 252，数字货币 24/7 用 365。
    pub(crate) days_per_year: f64,
    /// 年化无风险利率，默认 0.0。
    pub(crate) risk_free_rate: f64,
    pub(crate) history_buffer: Arc<RwLock<HistoryBuffer>>,
    /// 多周期窗口聚合器(策略 subscribe_bars 声明的订阅), 见 src/data/window.rs。
    /// 与 history_buffer 同样以 Arc<RwLock> 共享给 StrategyContext(供 current_window
    /// 在回调里读), 引擎自身只在数据阶段/预加载/收尾/时区 setter 拿写锁。
    pub(crate) window_aggregator: Arc<RwLock<crate::data::WindowAggregator>>,
    /// 本时间步闭合、尚未派发给策略的窗口 bar。DataProcessor 写入, StrategyProcessor
    /// 在基础事件回调之后取走派发。
    pub(crate) pending_window_bars: Vec<Bar>,
    pub(crate) initial_cash: Decimal,
    #[pyo3(get, set)]
    pub active_start_time_ns: Option<i64>,
    // Components
    pub(crate) event_manager: EventManager,
    pub(crate) statistics_manager: StatisticsManager,
    pub(crate) settlement_manager: SettlementManager,
    // Pipeline state
    pub(crate) current_event: Option<Event>,
    pub(crate) bar_count: usize,
    /// 用户显式传入 `symbols` 时的标的白名单。
    ///
    /// 三态语义（不能用空集合兼表「不过滤」）:
    /// - `None` = 未设置 ⇒ 放行全部（不传 `symbols` 时的现有行为）
    /// - `Some(非空)` = 只放行集合内的标的
    /// - `Some(空)` 不应出现: Python 侧在参数解析阶段即拒绝空 `symbols`,
    ///   FFI setter 亦把空列表折叠成 `None`, 避免静默跑出零事件回测。
    pub(crate) symbol_whitelist: Option<std::collections::HashSet<String>>,
    /// 因不在 `symbol_whitelist` 内而被 `pipeline/stages/data.rs` 丢弃的事件数.
    ///
    /// 只计数、不逐事件打日志(那会淹没输出) —— 由 `Engine::run` 在会话结束时
    /// 汇总报一条 `log::info!`, 让白名单丢事件不再是完全静默的。
    pub(crate) whitelist_filtered_event_count: u64,
    pub(crate) progress_total_steps: usize,
    pub(crate) progress_bar: Option<ProgressBar>,
    pub(crate) strategy_contexts: Vec<Option<Py<StrategyContext>>>,
    pub(crate) strategy_slot_strategies: Vec<Option<Py<PyAny>>>,
    pub(crate) strategy_slots: Vec<StrategySlot>,
    pub(crate) active_strategy_slot: usize,
    pub(crate) default_strategy_id: Option<String>,
    /// 已告警过的“所属策略不在任何槽位”的定时器 owner(每个 owner 只告警一次)。
    pub(crate) warned_orphan_timer_owners: HashSet<String>,
    pub(crate) strategy_priorities: HashMap<String, i32>,
    pub(crate) strategy_risk_budget_limits: HashMap<String, Decimal>,
    pub(crate) portfolio_risk_budget_limit: Option<Decimal>,
    pub(crate) strategy_risk_budget_used: HashMap<String, Decimal>,
    pub(crate) portfolio_risk_budget_used: Decimal,
    pub(crate) risk_budget_mode: String,
    pub(crate) risk_budget_reset_daily: bool,
    /// 实时会话的墙钟截止时刻(纳秒, UTC epoch)。到点则主循环自行结束。
    ///
    /// 存在的原因: live 循环在行情通道空时会一直 `FeedAction::Wait`, 而
    /// `duration` 此前是靠 patch `on_bar`/`on_tick` 抛 `KeyboardInterrupt` 实现的
    /// —— 行情一停就再也不被调用, 于是会话挂死。把时限下沉到等待循环本身才是
    /// 真正的墙钟兜底(见 docs/zh/meta/signal-ingestion-rfc.md 4.6)。
    pub(crate) session_deadline_ns: Option<i64>,
    pub(crate) risk_budget_usage_day: Option<NaiveDate>,
    pub(crate) strategy_max_order_value_limits: HashMap<String, Decimal>,
    pub(crate) strategy_max_order_size_limits: HashMap<String, Decimal>,
    pub(crate) strategy_max_position_size_limits: HashMap<String, Decimal>,
    pub(crate) strategy_max_daily_loss_limits: HashMap<String, Decimal>,
    pub(crate) strategy_max_drawdown_limits: HashMap<String, Decimal>,
    pub(crate) strategy_risk_cooldown_bars: HashMap<String, usize>,
    pub(crate) strategy_risk_cooldown_until_bar: HashMap<String, usize>,
    pub(crate) strategy_reduce_only_after_risk: HashSet<String>,
    pub(crate) strategy_positions: HashMap<String, HashMap<String, Decimal>>,
    pub(crate) strategy_cashflows: HashMap<String, Decimal>,
    pub(crate) strategy_daily_loss_day: HashMap<String, NaiveDate>,
    pub(crate) strategy_daily_loss_baseline_pnl: HashMap<String, Decimal>,
    pub(crate) strategy_last_pnl: HashMap<String, Decimal>,
    pub(crate) strategy_peak_pnl: HashMap<String, Decimal>,
    pub(crate) strategy_reduce_only_active: HashSet<String>,
    pub(crate) margin_accrued_interest: Decimal,
    pub(crate) margin_daily_interest: Decimal,
    pub(crate) recent_expiry_events: Vec<ExpiryEvent>,
    pub(crate) snapshot_time: i64,
    pub(crate) stream_callback: Option<Py<PyAny>>,
    pub(crate) stream_run_id: Option<String>,
    pub(crate) stream_seq: u64,
    pub(crate) stream_progress_interval: usize,
    pub(crate) stream_equity_interval: usize,
    pub(crate) stream_batch_size: usize,
    pub(crate) stream_max_buffer: usize,
    pub(crate) stream_buffer: Vec<PendingStreamEvent>,
    pub(crate) stream_fail_fast_on_callback_error: bool,
    pub(crate) stream_callback_error_count: u64,
    pub(crate) stream_callback_last_error: Option<String>,
    pub(crate) stream_fatal_error: Option<String>,
    pub(crate) stream_dropped_event_count: u64,
    pub(crate) stream_dropped_event_count_by_type: HashMap<String, u64>,
    pub(crate) stream_mode: String,
    pub(crate) stream_sampling_enabled: bool,
    pub(crate) stream_drop_non_critical: bool,
    pub(crate) session_config: SessionConfig,
}

pub(crate) struct PendingStreamEvent {
    pub(crate) event_type: String,
    pub(crate) symbol: Option<String>,
    pub(crate) level: String,
    pub(crate) payload: Vec<(String, String)>,
}

/// 策略回调返回的待注册引擎级计划(OCO 组 / bracket 计划)。
/// 常态为 None,避免每 bar 对策略对象做两次 getattr。
pub(crate) struct PendingEnginePlans {
    pub oco_groups: Vec<(String, String, String)>,
    pub bracket_plans: Vec<(
        String,
        Option<f64>,
        Option<f64>,
        Option<crate::model::TimeInForce>,
        Option<String>,
        Option<String>,
    )>,
}

// Internal implementation of Engine (not exposed to Python)
impl Engine {
    fn current_account_metrics(&self) -> AccountMetrics {
        let prices = self.last_prices.read().expect("last_prices 读锁被污染");
        calculate_account_metrics(
            &self.state.portfolio,
            &prices,
            &self.instruments,
            &self.state.order_manager.trade_tracker,
            &self.risk_manager.config,
        )
    }

    /// Cash/margin reserved by open (non-terminal) orders: the incremental used
    /// margin they would consume beyond the current portfolio. Reported as the
    /// account's `frozen_cash`, computed in Rust so it cannot drift from the
    /// engine's own margin accounting (the Python re-implementation was removed).
    fn current_frozen_cash(&self, active_orders: &[Order]) -> Decimal {
        let stock_ratio_override = if self.risk_manager.config.is_margin_account() {
            Some(self.risk_manager.config.stock_initial_margin_ratio())
        } else {
            None
        };
        let prices = self.last_prices.read().expect("last_prices 读锁被污染");
        let base_used = self.state.portfolio.calculate_used_margin_with_stock_ratio(
            &prices,
            &self.instruments,
            stock_ratio_override,
        );

        // Project every open order's position onto a clone and diff used margin.
        let mut projected = self.state.portfolio.clone();
        for order in active_orders {
            if !matches!(
                order.status,
                crate::model::OrderStatus::New
                    | crate::model::OrderStatus::Submitted
                    | crate::model::OrderStatus::PartiallyFilled
            ) {
                continue;
            }
            let remaining = order.quantity - order.filled_quantity;
            if remaining <= Decimal::ZERO {
                continue;
            }
            let current_pos = projected
                .positions
                .get(&order.symbol)
                .copied()
                .unwrap_or(Decimal::ZERO);
            let next_pos = crate::model::project_position_after(
                order.side,
                order.position_effect,
                current_pos,
                remaining,
            );
            projected.adjust_position(&order.symbol, next_pos - current_pos);
        }
        let projected_used = projected.calculate_used_margin_with_stock_ratio(
            &prices,
            &self.instruments,
            stock_ratio_override,
        );
        (projected_used - base_used).max(Decimal::ZERO)
    }

    fn build_position_entry_prices(&self) -> Arc<HashMap<String, Decimal>> {
        let mut entry_prices = HashMap::new();
        for (symbol, quantity) in self.state.portfolio.positions.iter() {
            if quantity.is_zero() {
                continue;
            }
            let average_price = self
                .state
                .order_manager
                .trade_tracker
                .get_average_price(symbol);
            entry_prices.insert(symbol.clone(), average_price);
        }
        Arc::new(entry_prices)
    }

    /// 作废会话解析器（时区、时段表、标的变化或载入存档后），窗口聚合器同步退回
    /// 无解析器状态，直到下次 [`Self::ensure_session_resolver`] 重建。
    pub(crate) fn invalidate_session_resolver(&mut self) {
        self.session_resolver = None;
        self.window_aggregator
            .write()
            .expect("window_aggregator 写锁被污染")
            .set_resolver(None, HashMap::new());
    }

    /// 首次调用时按 `session_config`、引擎时区与规则 B 构造会话解析器，并把全部
    /// 已登记标的设为在用。
    pub(crate) fn ensure_session_resolver(&mut self) -> PyResult<()> {
        if self.session_resolver.is_some() {
            return Ok(());
        }
        let clock = crate::calendar::clock::LocalClock::new(
            self.timezone_name.as_deref(),
            self.timezone_offset,
        );
        let user = match self.session_config.session_specs() {
            Some(specs) => Some(
                crate::calendar::resolver::build_sessions(specs).map_err(PyValueError::new_err)?,
            ),
            None => None,
        };
        let sessions = crate::calendar::resolver::effective_sessions(
            user,
            self.timezone_name.as_deref(),
            self.timezone_offset,
        );
        let calendar = match &self.session_config.trading_calendar {
            Some(days) => Some(
                crate::calendar::trading_calendar::TradingCalendar::explicit(days.clone())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
            None => None,
        };
        let mut resolver =
            crate::calendar::resolver::SessionResolver::new(clock, calendar, sessions)
                .map_err(PyValueError::new_err)?;
        let active: Vec<(String, Option<crate::model::AssetType>)> = self
            .instruments
            .values()
            .map(|i| (i.symbol().to_string(), Some(i.asset_type)))
            .collect();
        // 期权 → 标的：标的为已登记期货的商品期权跟随期货时段表（由 instruments 推导，不入存档）。
        let mut underlyings: Vec<(String, String)> = self
            .instruments
            .values()
            .filter(|i| i.asset_type == crate::model::AssetType::Option)
            .filter_map(|i| {
                i.underlying_symbol()
                    .filter(|u| !u.trim().is_empty())
                    .map(|u| (i.symbol().to_string(), u.clone()))
            })
            .collect();
        underlyings.sort();
        resolver.set_option_underlyings(&underlyings);
        resolver.set_active(&active);
        let resolver = Arc::new(resolver);
        // 多周期窗口按同一解析器切分交易日与交易块。
        let asset_types = self
            .instruments
            .values()
            .map(|i| (i.symbol().to_string(), i.asset_type))
            .collect();
        self.window_aggregator
            .write()
            .expect("window_aggregator 写锁被污染")
            .set_resolver(Some(resolver.clone()), asset_types);
        self.session_resolver = Some(resolver);
        // 缓存的时段区间来自旧解析器（时区、时段表或标的变化后失效）。
        self.session_span_cache.clear();
        self.last_interval_session = None;
        Ok(())
    }

    /// 行情事件的时段：bar 区间 `(interval_start, ts]`（无区间按 `ts` 前 1ns）落在
    /// 缓存的同一时段区间内时直接复用，否则查解析器并刷新缓存。
    pub(crate) fn event_session(
        &mut self,
        symbol: &str,
        interval_start: Option<i64>,
        ts: i64,
    ) -> PyResult<crate::model::TradingSession> {
        use crate::model::TradingSession;
        if self.force_session_continuous {
            return Ok(TradingSession::Continuous);
        }
        self.ensure_session_resolver()?;
        let point = ts.saturating_sub(1);
        if let Some(&span) = self.session_span_cache.get(symbol)
            && span_covers_event(span, interval_start, ts)
        {
            return Ok(span.2);
        }
        let resolver = self.session_resolver.as_ref().expect("已构造");
        let asset_type = self.instruments.get(symbol).map(|i| i.asset_type);
        // 跨多段的区间（如只有日期的日线覆盖整个交易日）不落在单个时段区间内，上面的
        // 品种缓存永不命中。解析结果只取决于（时段表, 区间），同一时间戳上同一时段表
        // 的各品种共享：缓存最近一次的 (时段表指针, interval_start, ts) → 时段。
        let schedule_key = Arc::as_ptr(&resolver.schedule_for(symbol, asset_type)) as usize;
        let key = (schedule_key, interval_start, ts);
        if let Some((k, session)) = self.last_interval_session
            && k == key
        {
            return Ok(session);
        }
        let session = resolver
            .resolve_bar(symbol, asset_type, interval_start, ts)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .session;
        // 只缓存与本次解析结果一致、且确实覆盖本事件的区间；多段 bar、恰在段结束的
        // 时刻等情形改记到按区间的缓存。
        match resolver.segment_span_at(symbol, asset_type, point) {
            Ok(span) if span.2 == session && span_covers_event(span, interval_start, ts) => {
                self.session_span_cache.insert(symbol.to_string(), span);
            }
            _ => self.last_interval_session = Some((key, session)),
        }
        Ok(session)
    }

    /// `ts` 所属的会话交易日。命中当前交易日区间时直接返回，否则交给解析器。
    pub(crate) fn trading_day_for(&mut self, ts: i64) -> PyResult<NaiveDate> {
        if let (Some(day), Some((prev_end, end))) =
            (self.current_trading_day, self.current_day_span)
            && prev_end < ts
            && ts <= end
        {
            return Ok(day);
        }
        // run() 开始时已构造; 直接驱动 pipeline 的调用方(单测)在这里补建。
        self.ensure_session_resolver()?;
        let resolver = self.session_resolver.as_ref().expect("刚构造");
        let day = resolver
            .session_trading_day(ts)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(day)
    }

    /// 风控（每日亏损、风险预算）使用的交易日：`now` 所属的会话交易日。
    ///
    /// `now` 落在当前交易日区间内时直接返回；超出时（如次日开盘前的定时器，当日
    /// 尚未开日）用解析器求，但不开日。尚未开过日（`current_trading_day` 为空）或
    /// 解析失败时退回 `current_trading_day`，由调用方按本地日期兜底。
    pub(crate) fn risk_trading_day(&mut self, now: i64) -> Option<NaiveDate> {
        let current = self.current_trading_day?;
        if let Some((prev_end, end)) = self.current_day_span
            && prev_end < now
            && now <= end
        {
            return Some(current);
        }
        if self.ensure_session_resolver().is_err() {
            return Some(current);
        }
        self.session_resolver
            .as_ref()
            .and_then(|r| r.session_trading_day(now).ok())
            .or(Some(current))
    }

    pub(crate) fn is_active_timestamp(&self, timestamp: i64) -> bool {
        self.active_start_time_ns
            .is_none_or(|start_ns| timestamp >= start_ns)
    }

    pub(crate) fn current_event_timestamp(&self) -> Option<i64> {
        match self.current_event.as_ref() {
            Some(Event::Bar(bar)) => Some(bar.timestamp),
            Some(Event::Tick(tick)) => Some(tick.timestamp),
            Some(Event::Timer(timer)) => Some(timer.timestamp),
            _ => None,
        }
    }

    fn effective_terminal_timestamp_for_timer(&self, timer: &Timer) -> Option<i64> {
        if let Some(source_ts) = timer
            .payload
            .strip_prefix("__framework_cross_section__|")
            .and_then(|payload| payload.rsplit('|').next())
            .and_then(|value| value.parse::<i64>().ok())
        {
            return Some(source_ts);
        }
        // 收盘后生命周期事件不是行情：期末时刻仍取当日最后一个行情事件，不在收盘之后
        // 多补一个权益点/持仓快照。
        if let Some((crate::model::timer::LIFECYCLE_AFTER_TRADING, _)) = timer.lifecycle() {
            return Some(
                self.last_event_ts_in_day
                    .unwrap_or_else(|| timer.timestamp.saturating_sub(1)),
            );
        }
        Some(timer.timestamp)
    }

    pub(crate) fn terminal_result_timestamp(&self) -> Option<i64> {
        match self.current_event.as_ref() {
            Some(Event::Bar(bar)) => Some(bar.timestamp),
            Some(Event::Tick(tick)) => Some(tick.timestamp),
            Some(Event::Timer(timer)) => self.effective_terminal_timestamp_for_timer(timer),
            _ => self.clock.timestamp(),
        }
    }

    /// 交给策略上下文、风控与撮合的“当前时刻”。生命周期事件取其触发时刻（收盘后
    /// 事件在收盘之后，其中下的单属于下一交易日）。
    pub(crate) fn context_timestamp(&self) -> i64 {
        if let Some(Event::Timer(timer)) = self.current_event.as_ref()
            && timer.is_lifecycle()
        {
            return timer.timestamp;
        }
        self.terminal_result_timestamp()
            .or_else(|| self.clock.timestamp())
            .unwrap_or(0)
    }

    /// 交易日 `day` 收盘定时器的触发时刻：回测为会话收盘 + 1ns，实盘为收盘 + 宽限。
    pub(crate) fn close_timer_ts(&mut self, day: NaiveDate) -> PyResult<i64> {
        self.ensure_session_resolver()?;
        let end = self
            .session_resolver
            .as_ref()
            .expect("已构造")
            .session_day_end(day)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(crate::model::timer::day_close_timer_timestamp(
            end,
            self.state.feed.is_live(),
        ))
    }

    /// 为交易日 `day` 排程盘前（pre_open）与开始交易（before_trading）生命周期定时器，
    /// 时刻不晚于 `not_before` 的跳过；已为 `day` 或更晚的交易日排程过时不重复。
    ///
    /// - `pre_open(D) = max(D 最早时段起点 - 1, 上一交易日收盘定时器 + 1)`
    /// - `before_trading(D) = max(D 最早连续竞价起点 + 1, pre_open(D) + 1)`；D 没有
    ///   连续竞价段时不排。
    ///
    /// 日历首日的夜盘未展开（缺前一交易日）时最早时段只看锚定当日的时段。
    pub(crate) fn schedule_lifecycle(&mut self, day: NaiveDate, not_before: i64) -> PyResult<()> {
        use crate::model::timer::{
            LIFECYCLE_BEFORE_TRADING, LIFECYCLE_PRE_OPEN, lifecycle_payload,
        };
        if self.lifecycle_scheduled_for.is_some_and(|d| d >= day) {
            return Ok(());
        }
        self.ensure_session_resolver()?;
        let resolver = self.session_resolver.clone().expect("已构造");
        let err = |e: crate::calendar::trading_calendar::CalendarError| {
            PyValueError::new_err(e.to_string())
        };
        let first = resolver.session_day_first_start(day).map_err(err)?;
        let prev_close = match resolver.session_calendar_before(day) {
            Ok(prev) => self.close_timer_ts(prev)?,
            Err(_) => i64::MIN,
        };
        let pre_open = first.saturating_sub(1).max(prev_close.saturating_add(1));
        let mut planned = vec![(pre_open, LIFECYCLE_PRE_OPEN)];
        if let Some(cont) = resolver
            .session_day_first_continuous_start(day)
            .map_err(err)?
        {
            planned.push((
                cont.saturating_add(1).max(pre_open.saturating_add(1)),
                LIFECYCLE_BEFORE_TRADING,
            ));
        }
        for (ts, kind) in planned {
            if ts > not_before {
                self.timers.push(Timer {
                    timestamp: ts,
                    payload: lifecycle_payload(kind, day),
                    owner: None,
                });
            }
        }
        self.lifecycle_scheduled_for = Some(day);
        Ok(())
    }

    /// 回测：为时刻 `ts` 所属的交易日排程生命周期定时器。`ts` 不在交易日历内时
    /// 不排（行情事件本身会在数据阶段报出日历错误）。
    pub(crate) fn schedule_lifecycle_for_event(
        &mut self,
        ts: i64,
        not_before: i64,
    ) -> PyResult<()> {
        self.ensure_session_resolver()?;
        let day = self
            .session_resolver
            .as_ref()
            .expect("已构造")
            .session_trading_day(ts);
        match day {
            Ok(day) => self.schedule_lifecycle(day, not_before),
            Err(_) => Ok(()),
        }
    }

    /// 实盘：会话开始时按墙钟 `now` 排程——当日时刻已过的跳过；当日开始交易
    /// （before_trading）已过时改排下一交易日。`now` 不在交易日历内时不排并告警。
    pub(crate) fn schedule_live_lifecycle(&mut self, now: i64) -> PyResult<()> {
        self.ensure_session_resolver()?;
        let resolver = self.session_resolver.clone().expect("已构造");
        let day = match resolver.session_trading_day(now) {
            Ok(day) => day,
            Err(e) => {
                log::warn!(
                    "实盘当前时刻不在交易日历内（{e}），本次会话不会触发 on_pre_open / on_before_trading；请检查 trading_calendar= 是否覆盖当前日期"
                );
                return Ok(());
            }
        };
        self.schedule_lifecycle(day, now)?;
        let has_before_trading = self.timers.iter().any(|t| {
            matches!(
                t.lifecycle(),
                Some((crate::model::timer::LIFECYCLE_BEFORE_TRADING, d)) if d == day
            )
        });
        if !has_before_trading {
            self.schedule_next_session_lifecycle(day, now)?;
        }
        Ok(())
    }

    /// 实盘按日历串联：before_trading(D) 触发时排下一交易日；D 没有连续竞价段
    /// （不排 before_trading）时改在 pre_open(D) 触发时排。
    pub(crate) fn chain_live_lifecycle(&mut self, timer: &Timer) -> PyResult<()> {
        use crate::model::timer::{LIFECYCLE_BEFORE_TRADING, LIFECYCLE_PRE_OPEN};
        let Some((kind, day)) = timer.lifecycle() else {
            return Ok(());
        };
        let chain = match kind {
            LIFECYCLE_BEFORE_TRADING => true,
            LIFECYCLE_PRE_OPEN => {
                self.ensure_session_resolver()?;
                self.session_resolver
                    .as_ref()
                    .is_some_and(|r| matches!(r.session_day_first_continuous_start(day), Ok(None)))
            }
            _ => false,
        };
        if chain {
            self.schedule_next_session_lifecycle(day, timer.timestamp)?;
        }
        Ok(())
    }

    /// 为 `day` 的下一个会话交易日排程生命周期；交易日历已到尽头时告警一次。
    fn schedule_next_session_lifecycle(&mut self, day: NaiveDate, not_before: i64) -> PyResult<()> {
        self.ensure_session_resolver()?;
        let next = self
            .session_resolver
            .as_ref()
            .expect("已构造")
            .next_session_day(day);
        match next {
            Ok(next) => self.schedule_lifecycle(next, not_before),
            Err(e) => {
                if !self.lifecycle_chain_warned {
                    self.lifecycle_chain_warned = true;
                    log::warn!(
                        "实盘交易日历在 {day} 之后没有交易日（{e}），之后不会再触发 on_pre_open / on_before_trading；请检查 trading_calendar= 是否覆盖会话期间"
                    );
                }
                Ok(())
            }
        }
    }

    pub(crate) fn execution_policy_core(&self) -> ExecutionPolicyCore {
        self.execution_policy_core_state
    }

    pub(crate) fn set_execution_policy_core(&mut self, policy: ExecutionPolicyCore) {
        self.execution_policy_core_state = policy;
    }

    pub(crate) fn normalized_order_strategy_id(order: &Order) -> Option<String> {
        order
            .owner_strategy_id
            .as_ref()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    }

    pub(crate) fn ensure_strategy_slot_exists(&mut self) {
        if self.strategy_slots.is_empty() {
            let strategy_id = self
                .default_strategy_id
                .clone()
                .unwrap_or_else(|| "_default".to_string());
            self.strategy_slots.push(StrategySlot { strategy_id });
        }
        if self.active_strategy_slot >= self.strategy_slots.len() {
            self.active_strategy_slot = 0;
        }
    }

    pub(crate) fn strategy_priority_for_order(&self, order: &Order) -> i32 {
        let strategy_id =
            Self::normalized_order_strategy_id(order).unwrap_or_else(|| "_default".to_string());
        self.strategy_priorities
            .get(&strategy_id)
            .copied()
            .unwrap_or(0)
    }

    pub(crate) fn order_estimated_notional(&self, order: &Order) -> Option<Decimal> {
        let price = if let Some(p) = order.price {
            Some(p)
        } else {
            self.last_prices
                .read()
                .expect("last_prices 读锁被污染")
                .get(&order.symbol)
                .copied()
        }?;
        Some((price * order.quantity).abs())
    }

    pub(crate) fn trade_notional(&self, trade: &Trade) -> Decimal {
        (trade.price * trade.quantity).abs()
    }

    pub(crate) fn maybe_reset_risk_budget_usage(&mut self, current_time: i64) {
        if !self.risk_budget_reset_daily {
            return;
        }
        let current_day = self
            .risk_trading_day(current_time)
            .unwrap_or_else(|| self.local_date_from_ns(current_time));
        if self.risk_budget_usage_day == Some(current_day) {
            return;
        }
        self.strategy_risk_budget_used.clear();
        self.portfolio_risk_budget_used = Decimal::ZERO;
        self.risk_budget_usage_day = Some(current_day);
    }

    pub(crate) fn risk_budget_use_trade_mode(&self) -> bool {
        self.risk_budget_mode == "trade_notional"
    }

    pub(crate) fn check_strategy_risk_budget_limit(&self, order: &Order) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        let budget = self
            .strategy_risk_budget_limits
            .get(&strategy_id)
            .copied()?;
        let used = self
            .strategy_risk_budget_used
            .get(&strategy_id)
            .copied()
            .unwrap_or(Decimal::ZERO);
        let projected = used + self.order_estimated_notional(order)?;
        if projected > budget {
            return Some(format!(
                "Risk: Strategy {} risk budget {} exceeds strategy budget {}",
                strategy_id, projected, budget
            ));
        }
        None
    }

    pub(crate) fn check_portfolio_risk_budget_limit(&self, order: &Order) -> Option<String> {
        let budget = self.portfolio_risk_budget_limit?;
        let projected = self.portfolio_risk_budget_used + self.order_estimated_notional(order)?;
        if projected > budget {
            return Some(format!(
                "Risk: Portfolio risk budget {} exceeds portfolio budget {}",
                projected, budget
            ));
        }
        None
    }

    pub(crate) fn apply_risk_budget_usage(&mut self, order: &Order) {
        let Some(notional) = self.order_estimated_notional(order) else {
            return;
        };
        if let Some(strategy_id) = Self::normalized_order_strategy_id(order)
            && self.strategy_risk_budget_limits.contains_key(&strategy_id)
        {
            let entry = self
                .strategy_risk_budget_used
                .entry(strategy_id)
                .or_insert(Decimal::ZERO);
            *entry += notional;
        }
        if self.portfolio_risk_budget_limit.is_some() {
            self.portfolio_risk_budget_used += notional;
        }
    }

    pub(crate) fn apply_risk_budget_usage_from_trade(&mut self, trade: &Trade) {
        let notional = self.trade_notional(trade);
        if let Some(strategy_id) = trade
            .owner_strategy_id
            .as_ref()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            && self.strategy_risk_budget_limits.contains_key(&strategy_id)
        {
            let entry = self
                .strategy_risk_budget_used
                .entry(strategy_id)
                .or_insert(Decimal::ZERO);
            *entry += notional;
        }
        if self.portfolio_risk_budget_limit.is_some() {
            self.portfolio_risk_budget_used += notional;
        }
    }

    pub(crate) fn check_strategy_order_value_limit(&self, order: &Order) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        let max_value = self.strategy_max_order_value_limits.get(&strategy_id)?;
        let price = if let Some(p) = order.price {
            Some(p)
        } else {
            self.last_prices
                .read()
                .expect("last_prices 读锁被污染")
                .get(&order.symbol)
                .copied()
        };
        crate::risk::strategy_limits::exceeds_order_value(
            &strategy_id,
            order.quantity,
            price,
            *max_value,
        )
    }

    pub(crate) fn check_strategy_order_size_limit(&self, order: &Order) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        let max_size = self.strategy_max_order_size_limits.get(&strategy_id)?;
        crate::risk::strategy_limits::exceeds_order_size(&strategy_id, order.quantity, *max_size)
    }

    pub(crate) fn check_strategy_position_size_limit(&self, order: &Order) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        let max_size = self.strategy_max_position_size_limits.get(&strategy_id)?;
        let strategy_positions = self.strategy_positions.get(&strategy_id);
        let current_qty = strategy_positions
            .and_then(|positions| positions.get(&order.symbol).copied())
            .unwrap_or(Decimal::ZERO);
        let delta = match order.side {
            OrderSide::Buy => order.quantity,
            OrderSide::Sell => -order.quantity,
        };
        crate::risk::strategy_limits::exceeds_position_size(
            &strategy_id,
            current_qty,
            delta,
            *max_size,
        )
    }

    pub(crate) fn apply_strategy_trade_position(&mut self, trade: &Trade) {
        let strategy_id = trade
            .owner_strategy_id
            .as_ref()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        let Some(strategy_key) = strategy_id else {
            return;
        };
        let strategy_key_for_position = strategy_key.clone();
        let entry = self
            .strategy_positions
            .entry(strategy_key_for_position)
            .or_default()
            .entry(trade.symbol.clone())
            .or_insert(Decimal::ZERO);
        match trade.side {
            OrderSide::Buy => *entry += trade.quantity,
            OrderSide::Sell => *entry -= trade.quantity,
        }
        let cashflow_entry = self
            .strategy_cashflows
            .entry(strategy_key)
            .or_insert(Decimal::ZERO);
        match trade.side {
            OrderSide::Buy => *cashflow_entry -= trade.price * trade.quantity,
            OrderSide::Sell => *cashflow_entry += trade.price * trade.quantity,
        }
        *cashflow_entry -= trade.commission;
    }

    pub(crate) fn current_strategy_pnl(&self, strategy_id: &str) -> Decimal {
        let cashflow = self
            .strategy_cashflows
            .get(strategy_id)
            .copied()
            .unwrap_or(Decimal::ZERO);
        let prices = self.last_prices.read().expect("last_prices 读锁被污染");
        let mark_to_market = self
            .strategy_positions
            .get(strategy_id)
            .map(|positions| {
                positions
                    .iter()
                    .map(|(symbol, qty)| {
                        let price = prices.get(symbol).copied().unwrap_or(Decimal::ZERO);
                        *qty * price
                    })
                    .sum::<Decimal>()
            })
            .unwrap_or(Decimal::ZERO);
        cashflow + mark_to_market
    }

    pub(crate) fn check_strategy_daily_loss_limit(
        &mut self,
        order: &Order,
        current_time: i64,
    ) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        let max_daily_loss = self
            .strategy_max_daily_loss_limits
            .get(&strategy_id)
            .copied()?;
        let current_day = self
            .risk_trading_day(current_time)
            .unwrap_or_else(|| self.local_date_from_ns(current_time));
        let current_pnl = self.current_strategy_pnl(&strategy_id);
        let needs_reset = self
            .strategy_daily_loss_day
            .get(&strategy_id)
            .map(|day| *day != current_day)
            .unwrap_or(true);
        if needs_reset {
            let reset_baseline = if self.strategy_daily_loss_day.contains_key(&strategy_id) {
                self.strategy_last_pnl
                    .get(&strategy_id)
                    .copied()
                    .unwrap_or(current_pnl)
            } else {
                current_pnl
            };
            self.strategy_daily_loss_day
                .insert(strategy_id.clone(), current_day);
            self.strategy_daily_loss_baseline_pnl
                .insert(strategy_id.clone(), reset_baseline);
        }
        let baseline = self
            .strategy_daily_loss_baseline_pnl
            .get(&strategy_id)
            .copied()
            .unwrap_or(current_pnl);
        self.strategy_last_pnl
            .insert(strategy_id.clone(), current_pnl);
        let loss = baseline - current_pnl;
        if loss > max_daily_loss {
            return Some(format!(
                "Risk: Strategy {} daily loss {} exceeds strategy limit {}",
                strategy_id, loss, max_daily_loss
            ));
        }
        None
    }

    pub(crate) fn check_strategy_drawdown_limit(&mut self, order: &Order) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        let max_drawdown = self
            .strategy_max_drawdown_limits
            .get(&strategy_id)
            .copied()?;
        let current_pnl = self.current_strategy_pnl(&strategy_id);
        let peak = self
            .strategy_peak_pnl
            .get(&strategy_id)
            .copied()
            .unwrap_or(current_pnl);
        let updated_peak = if current_pnl > peak {
            current_pnl
        } else {
            peak
        };
        self.strategy_peak_pnl
            .insert(strategy_id.clone(), updated_peak);
        let drawdown = updated_peak - current_pnl;
        if drawdown > max_drawdown {
            return Some(format!(
                "Risk: Strategy {} drawdown {} exceeds strategy limit {}",
                strategy_id, drawdown, max_drawdown
            ));
        }
        None
    }

    pub(crate) fn activate_strategy_reduce_only_if_configured(&mut self, order: &Order) {
        if let Some(strategy_id) = Self::normalized_order_strategy_id(order)
            && self.strategy_reduce_only_after_risk.contains(&strategy_id)
        {
            self.strategy_reduce_only_active.insert(strategy_id);
        }
    }

    pub(crate) fn check_strategy_reduce_only_mode(&self, order: &Order) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        if !self.strategy_reduce_only_active.contains(&strategy_id) {
            return None;
        }
        let current_qty = self
            .strategy_positions
            .get(&strategy_id)
            .and_then(|positions| positions.get(&order.symbol).copied())
            .unwrap_or(Decimal::ZERO);
        let allowed = if current_qty > Decimal::ZERO {
            order.side == OrderSide::Sell && order.quantity <= current_qty
        } else if current_qty < Decimal::ZERO {
            order.side == OrderSide::Buy && order.quantity <= current_qty.abs()
        } else {
            false
        };
        if !allowed {
            return Some(format!(
                "Risk: Strategy {} is in reduce_only mode and cannot open/increase position",
                strategy_id
            ));
        }
        None
    }

    pub(crate) fn activate_strategy_risk_cooldown_if_configured(&mut self, order: &Order) {
        let Some(strategy_id) = Self::normalized_order_strategy_id(order) else {
            return;
        };
        let Some(cooldown_bars) = self.strategy_risk_cooldown_bars.get(&strategy_id).copied()
        else {
            return;
        };
        if cooldown_bars == 0 {
            return;
        }
        let until_bar = self.bar_count.saturating_add(cooldown_bars);
        self.strategy_risk_cooldown_until_bar
            .insert(strategy_id, until_bar);
    }

    pub(crate) fn check_strategy_risk_cooldown_mode(&mut self, order: &Order) -> Option<String> {
        let strategy_id = Self::normalized_order_strategy_id(order)?;
        let until_bar = self
            .strategy_risk_cooldown_until_bar
            .get(&strategy_id)
            .copied()?;
        if self.bar_count > until_bar {
            self.strategy_risk_cooldown_until_bar.remove(&strategy_id);
            return None;
        }
        let bars_remaining = until_bar.saturating_sub(self.bar_count);
        Some(format!(
            "Risk: Strategy {} is in cooldown for {} more bar(s)",
            strategy_id, bars_remaining
        ))
    }

    pub(crate) fn ensure_strategy_context_capacity(&mut self) {
        self.ensure_strategy_slot_exists();
        while self.strategy_contexts.len() < self.strategy_slots.len() {
            self.strategy_contexts.push(None);
        }
        while self.strategy_slot_strategies.len() < self.strategy_slots.len() {
            self.strategy_slot_strategies.push(None);
        }
    }

    pub(crate) fn get_or_create_strategy_context<'py>(
        &mut self,
        py: Python<'py>,
        slot_index: usize,
        active_orders: Arc<Vec<Order>>,
        step_trades: Vec<Trade>,
        step_rejected_orders: Vec<Order>,
        previous_cash: Decimal,
        previous_account_metrics: AccountMetrics,
    ) -> PyResult<Py<StrategyContext>> {
        self.ensure_strategy_context_capacity();
        let sessions = self.context_sessions();
        let account_metrics = self.current_account_metrics();
        let position_entry_prices = self.build_position_entry_prices();
        let frozen_cash = self
            .current_frozen_cash(&active_orders)
            .to_f64()
            .unwrap_or_default();
        if let Some(existing_ctx) = self
            .strategy_contexts
            .get(slot_index)
            .and_then(|ctx| ctx.as_ref())
        {
            let py_ctx = existing_ctx.clone_ref(py);
            {
                let mut ctx_mut = py_ctx.borrow_mut(py);
                ctx_mut.update_state(crate::context::ContextUpdate {
                    cash: self.state.portfolio.cash,
                    previous_cash,
                    positions: self.state.portfolio.positions.clone(),
                    available_positions: self.state.portfolio.available_positions.clone(),
                    position_entry_prices,
                    session: self.clock.session,
                    current_time: self.context_timestamp(),
                    active_orders,
                    closed_trades: self.state.order_manager.trade_tracker.closed_trades.clone(),
                    recent_trades: step_trades,
                    recent_rejected_orders: step_rejected_orders,
                    recent_expired_orders: self
                        .state
                        .order_manager
                        .current_step_expired_orders
                        .clone(),
                    recent_finalized_orders: self
                        .state
                        .order_manager
                        .current_step_finalized_orders
                        .clone(),
                    recent_expiry_events: self.recent_expiry_events.clone(),
                    account_equity: account_metrics.equity.to_f64().unwrap_or_default(),
                    account_market_value: account_metrics.market_value.to_f64().unwrap_or_default(),
                    account_notional_value: account_metrics
                        .notional_value
                        .to_f64()
                        .unwrap_or_default(),
                    account_used_margin: account_metrics.used_margin.to_f64().unwrap_or_default(),
                    account_unrealized_pnl: account_metrics
                        .unrealized_pnl
                        .to_f64()
                        .unwrap_or_default(),
                    account_maintenance_ratio: account_metrics
                        .maintenance_ratio
                        .to_f64()
                        .unwrap_or_default(),
                    account_short_market_value: account_metrics
                        .short_market_value
                        .to_f64()
                        .unwrap_or_default(),
                    account_frozen_cash: frozen_cash,
                    previous_account_equity: previous_account_metrics
                        .equity
                        .to_f64()
                        .unwrap_or_default(),
                    previous_account_market_value: previous_account_metrics
                        .market_value
                        .to_f64()
                        .unwrap_or_default(),
                    previous_account_notional_value: previous_account_metrics
                        .notional_value
                        .to_f64()
                        .unwrap_or_default(),
                    previous_account_used_margin: previous_account_metrics
                        .used_margin
                        .to_f64()
                        .unwrap_or_default(),
                    previous_account_unrealized_pnl: previous_account_metrics
                        .unrealized_pnl
                        .to_f64()
                        .unwrap_or_default(),
                    previous_account_maintenance_ratio: previous_account_metrics
                        .maintenance_ratio
                        .to_f64()
                        .unwrap_or_default(),
                    margin_accrued_interest: self
                        .margin_accrued_interest
                        .to_f64()
                        .unwrap_or_default(),
                    margin_daily_interest: self.margin_daily_interest.to_f64().unwrap_or_default(),
                    last_prices: Arc::clone(&self.last_prices),
                    sessions,
                });
            }
            return Ok(py_ctx);
        }

        let strategy_id = self
            .strategy_slots
            .get(slot_index)
            .map(|slot| slot.strategy_id.clone());
        let ctx = self.create_context(
            active_orders,
            step_trades,
            step_rejected_orders,
            strategy_id,
            previous_cash,
            previous_account_metrics,
            sessions,
        );
        let py_ctx = Py::new(py, ctx).unwrap();
        let persistent_ref = py_ctx.clone_ref(py);
        if let Some(slot_ctx) = self.strategy_contexts.get_mut(slot_index) {
            *slot_ctx = Some(persistent_ref);
        }
        Ok(py_ctx)
    }

    pub(crate) fn set_stream_callback_internal(&mut self, callback: Option<Py<PyAny>>) {
        self.stream_callback = callback;
        self.stream_run_id = None;
        self.stream_seq = 0;
        self.stream_buffer.clear();
        self.stream_callback_error_count = 0;
        self.stream_callback_last_error = None;
        self.stream_fatal_error = None;
        self.stream_dropped_event_count = 0;
        self.stream_dropped_event_count_by_type.clear();
    }

    pub(crate) fn set_stream_options_internal(
        &mut self,
        progress_interval: usize,
        equity_interval: usize,
        batch_size: usize,
        max_buffer: usize,
        error_mode: &str,
        stream_mode: &str,
    ) {
        self.stream_progress_interval = progress_interval.max(1);
        self.stream_equity_interval = equity_interval.max(1);
        self.stream_batch_size = batch_size.max(1);
        self.stream_max_buffer = max_buffer.max(1);
        self.stream_fail_fast_on_callback_error = error_mode == "fail_fast";
        if stream_mode == "audit" {
            self.stream_mode = "audit".to_string();
            self.stream_sampling_enabled = false;
            self.stream_drop_non_critical = false;
        } else {
            self.stream_mode = "observability".to_string();
            self.stream_sampling_enabled = true;
            self.stream_drop_non_critical = true;
        }
    }

    pub(crate) fn start_stream_run(&mut self, py: Python<'_>, total_events: Option<usize>) {
        if self.stream_callback.is_none() {
            self.stream_run_id = None;
            self.stream_seq = 0;
            self.stream_buffer.clear();
            self.stream_callback_error_count = 0;
            self.stream_callback_last_error = None;
            self.stream_fatal_error = None;
            self.stream_dropped_event_count = 0;
            self.stream_dropped_event_count_by_type.clear();
            return;
        }
        self.stream_seq = 0;
        self.stream_run_id = Some(Uuid::new_v4().to_string());
        self.stream_callback_error_count = 0;
        self.stream_callback_last_error = None;
        self.stream_fatal_error = None;
        self.stream_dropped_event_count = 0;
        self.stream_dropped_event_count_by_type.clear();

        let mut payload = HashMap::new();
        payload.insert("total_events", total_events.unwrap_or(0).to_string());
        payload.insert(
            "fill_policy_price_basis",
            format!("{:?}", self.execution_policy_core_state.price_basis),
        );
        payload.insert(
            "fill_policy_bar_offset",
            self.execution_policy_core_state.bar_offset.to_string(),
        );
        payload.insert(
            "fill_policy_temporal",
            self.execution_policy_core_state
                .temporal_as_str()
                .to_string(),
        );
        self.emit_stream_event(py, "started", None, "info", payload);
    }

    pub(crate) fn emit_stream_event(
        &mut self,
        py: Python<'_>,
        event_type: &str,
        symbol: Option<&str>,
        level: &str,
        payload: HashMap<&str, String>,
    ) {
        let (Some(_), Some(_)) = (&self.stream_callback, &self.stream_run_id) else {
            return;
        };

        if self.stream_sampling_enabled
            && event_type == "progress"
            && !self.bar_count.is_multiple_of(self.stream_progress_interval)
        {
            return;
        }
        if self.stream_sampling_enabled
            && event_type == "equity"
            && !self.bar_count.is_multiple_of(self.stream_equity_interval)
        {
            return;
        }

        let is_critical = matches!(
            event_type,
            "started" | "order" | "trade" | "risk" | "error" | "finished"
        );
        if self.stream_buffer.len() >= self.stream_max_buffer {
            if is_critical || !self.stream_drop_non_critical {
                self.flush_stream_events(py);
            } else {
                self.record_dropped_stream_event(event_type);
                return;
            }
        }

        let mut payload_vec = Vec::new();
        for (k, v) in payload {
            payload_vec.push((k.to_string(), v));
        }
        self.stream_buffer.push(PendingStreamEvent {
            event_type: event_type.to_string(),
            symbol: symbol.map(str::to_string),
            level: level.to_string(),
            payload: payload_vec,
        });

        if is_critical || self.stream_buffer.len() >= self.stream_batch_size {
            self.flush_stream_events(py);
        }
    }

    pub(crate) fn emit_stream_event_owned(
        &mut self,
        py: Python<'_>,
        event_type: &str,
        symbol: Option<String>,
        level: &str,
        payload: HashMap<String, String>,
    ) {
        let payload_ref: HashMap<&str, String> = payload
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();
        self.emit_stream_event(py, event_type, symbol.as_deref(), level, payload_ref);
    }

    pub(crate) fn flush_stream_events(&mut self, py: Python<'_>) {
        let (Some(callback_ref), Some(run_id_ref)) = (&self.stream_callback, &self.stream_run_id)
        else {
            self.stream_buffer.clear();
            return;
        };
        let callback = callback_ref.clone_ref(py);
        let run_id = run_id_ref.clone();
        if self.stream_buffer.is_empty() {
            return;
        }

        for pending in self.stream_buffer.drain(..) {
            self.stream_seq = self.stream_seq.saturating_add(1);
            let event_dict = PyDict::new(py);
            if event_dict.set_item("run_id", run_id.as_str()).is_err()
                || event_dict.set_item("seq", self.stream_seq).is_err()
                || event_dict
                    .set_item("ts", self.clock.timestamp().unwrap_or(0))
                    .is_err()
                || event_dict
                    .set_item("event_type", pending.event_type.as_str())
                    .is_err()
                || event_dict.set_item("symbol", pending.symbol).is_err()
                || event_dict
                    .set_item("level", pending.level.as_str())
                    .is_err()
            {
                continue;
            }

            let payload_dict = PyDict::new(py);
            for (k, v) in pending.payload {
                if payload_dict.set_item(k, v).is_err() {
                    continue;
                }
            }
            if event_dict.set_item("payload", payload_dict).is_err() {
                continue;
            }
            if let Err(err) = callback.bind(py).call1((event_dict,)) {
                self.stream_callback_error_count =
                    self.stream_callback_error_count.saturating_add(1);
                let message = err.to_string();
                self.stream_callback_last_error = Some(message.clone());
                if self.stream_fail_fast_on_callback_error {
                    self.stream_fatal_error = Some(format!(
                        "stream callback failed in fail_fast mode: {}",
                        message
                    ));
                    break;
                }
            }
        }
    }

    pub(crate) fn finish_stream_run(&mut self, py: Python<'_>, status: &str, reason: Option<&str>) {
        if self.stream_callback.is_none() {
            self.stream_run_id = None;
            self.stream_seq = 0;
            self.stream_buffer.clear();
            self.stream_callback_error_count = 0;
            self.stream_callback_last_error = None;
            self.stream_fatal_error = None;
            self.stream_dropped_event_count = 0;
            self.stream_dropped_event_count_by_type.clear();
            return;
        }

        let mut payload = HashMap::new();
        payload.insert("status", status.to_string());
        payload.insert("processed_events", self.bar_count.to_string());
        payload.insert(
            "total_trades",
            self.state.order_manager.trades.len().to_string(),
        );
        payload.insert(
            "callback_error_count",
            self.stream_callback_error_count.to_string(),
        );
        payload.insert(
            "dropped_event_count",
            self.stream_dropped_event_count.to_string(),
        );
        payload.insert(
            "dropped_event_count_by_type",
            self.format_stream_dropped_event_counts(),
        );
        payload.insert("stream_mode", self.stream_mode.clone());
        payload.insert("sampling_enabled", self.stream_sampling_enabled.to_string());
        payload.insert(
            "backpressure_policy",
            if self.stream_drop_non_critical {
                "drop_non_critical".to_string()
            } else {
                "block".to_string()
            },
        );
        if let Some(message) = reason {
            payload.insert("reason", message.to_string());
        }
        if let Some(message) = &self.stream_callback_last_error {
            payload.insert("last_callback_error", message.clone());
        }
        self.emit_stream_event(py, "finished", None, "info", payload);
        self.flush_stream_events(py);
        self.stream_run_id = None;
        self.stream_buffer.clear();
    }

    pub(crate) fn record_dropped_stream_event(&mut self, event_type: &str) {
        self.stream_dropped_event_count = self.stream_dropped_event_count.saturating_add(1);
        let counter = self
            .stream_dropped_event_count_by_type
            .entry(event_type.to_string())
            .or_insert(0);
        *counter = counter.saturating_add(1);
    }

    pub(crate) fn format_stream_dropped_event_counts(&self) -> String {
        if self.stream_dropped_event_count_by_type.is_empty() {
            return String::new();
        }
        let mut entries: Vec<(&String, &u64)> =
            self.stream_dropped_event_count_by_type.iter().collect();
        entries.sort_by(|(left, _), (right, _)| left.cmp(right));
        entries
            .into_iter()
            .map(|(event_type, count)| format!("{}={}", event_type, count))
            .collect::<Vec<String>>()
            .join(",")
    }

    pub(crate) fn raise_stream_fatal_error_if_any(&mut self) -> PyResult<()> {
        if let Some(message) = self.stream_fatal_error.take() {
            return Err(PyRuntimeError::new_err(message));
        }
        Ok(())
    }

    pub(crate) fn create_context(
        &self,
        active_orders: Arc<Vec<Order>>,
        step_trades: Vec<Trade>,
        step_rejected_orders: Vec<Order>,
        strategy_id: Option<String>,
        previous_cash: Decimal,
        previous_account_metrics: AccountMetrics,
        sessions: crate::context::ContextSessions,
    ) -> StrategyContext {
        let account_metrics = self.current_account_metrics();
        let position_entry_prices = self.build_position_entry_prices();
        let frozen_cash = self
            .current_frozen_cash(&active_orders)
            .to_f64()
            .unwrap_or_default();
        // Create a temporary context for the strategy to use
        StrategyContext::new(crate::context::ContextInit {
            cash: self.state.portfolio.cash,
            previous_cash,
            positions: self.state.portfolio.positions.clone(),
            available_positions: self.state.portfolio.available_positions.clone(),
            position_entry_prices,
            session: self.clock.session,
            current_time: self.context_timestamp(),
            active_orders,
            closed_trades: self.state.order_manager.trade_tracker.closed_trades.clone(),
            recent_trades: step_trades,
            recent_rejected_orders: step_rejected_orders,
            recent_expired_orders: self.state.order_manager.current_step_expired_orders.clone(),
            recent_finalized_orders: self
                .state
                .order_manager
                .current_step_finalized_orders
                .clone(),
            recent_expiry_events: self.recent_expiry_events.clone(),
            history_buffer: Some(self.history_buffer.clone()),
            window_aggregator: Some(self.window_aggregator.clone()),
            event_tx: Some(self.event_manager.sender()),
            risk_config: self.risk_manager.config.clone(),
            strategy_id,
            account_equity: account_metrics.equity.to_f64().unwrap_or_default(),
            account_market_value: account_metrics.market_value.to_f64().unwrap_or_default(),
            account_notional_value: account_metrics.notional_value.to_f64().unwrap_or_default(),
            account_used_margin: account_metrics.used_margin.to_f64().unwrap_or_default(),
            account_unrealized_pnl: account_metrics.unrealized_pnl.to_f64().unwrap_or_default(),
            account_maintenance_ratio: account_metrics
                .maintenance_ratio
                .to_f64()
                .unwrap_or_default(),
            account_short_market_value: account_metrics
                .short_market_value
                .to_f64()
                .unwrap_or_default(),
            account_frozen_cash: frozen_cash,
            previous_account_equity: previous_account_metrics.equity.to_f64().unwrap_or_default(),
            previous_account_market_value: previous_account_metrics
                .market_value
                .to_f64()
                .unwrap_or_default(),
            previous_account_notional_value: previous_account_metrics
                .notional_value
                .to_f64()
                .unwrap_or_default(),
            previous_account_used_margin: previous_account_metrics
                .used_margin
                .to_f64()
                .unwrap_or_default(),
            previous_account_unrealized_pnl: previous_account_metrics
                .unrealized_pnl
                .to_f64()
                .unwrap_or_default(),
            previous_account_maintenance_ratio: previous_account_metrics
                .maintenance_ratio
                .to_f64()
                .unwrap_or_default(),
            margin_accrued_interest: self.margin_accrued_interest.to_f64().unwrap_or_default(),
            margin_daily_interest: self.margin_daily_interest.to_f64().unwrap_or_default(),
            instruments: Arc::new(self.instruments.clone()),
            last_prices: Arc::clone(&self.last_prices),
            sessions,
        })
    }

    /// 交给策略上下文的交易日与时段状态：交易日按上下文时刻取风控口径（不开日），
    /// 解析器与本时间片时段表与撮合共用同一份（`Arc` 克隆 / `Weak` 引用）。
    pub(crate) fn context_sessions(&mut self) -> crate::context::ContextSessions {
        let now = self.context_timestamp();
        let trading_day = if self.current_trading_day.is_some() {
            // 已开过日：风控口径（收盘后至下一交易日开日前属下一交易日）。
            self.risk_trading_day(now)
        } else if self
            .terminal_result_timestamp()
            .or_else(|| self.clock.timestamp())
            .is_some()
        {
            // 首个交易日开日前（首日盘前定时器等）：按解析器求当前时刻所属交易日。
            // run() 开始时已构造解析器，这里只为直接驱动的调用方补建。
            let _ = self.ensure_session_resolver();
            self.session_resolver
                .as_ref()
                .and_then(|r| r.session_trading_day(now).ok())
        } else {
            None
        };
        let event_interval_start = match self.current_event.as_ref() {
            Some(Event::Bar(bar)) => bar.interval_start,
            _ => None,
        };
        // 无解析器时 session_of 按连续竞价，与撮合 SessionView 一致。
        crate::context::ContextSessions {
            trading_day,
            session_resolver: self.session_resolver.clone(),
            slice_sessions: Arc::downgrade(&self.slice_sessions),
            force_session_continuous: self.force_session_continuous,
            event_interval_start,
            event_at_instant: self.lifecycle_event_at_instant(),
        }
    }

    /// 当前事件是否为交易日生命周期事件（盘前/开始交易/收盘后）。它们本身就是一个
    /// 时点，时段按事件时刻本身求（收盘后事件为闭市），而不是像行情与普通定时器那样
    /// 取前 1ns（`ctx.session`、`session_of`、订单创建时段同此口径）。
    pub(crate) fn lifecycle_event_at_instant(&self) -> bool {
        matches!(self.current_event.as_ref(), Some(Event::Timer(t)) if t.is_lifecycle())
    }

    /// 生命周期事件 `kind` 是否要派发给策略：有策略重写了对应钩子时派发；收盘后
    /// 事件未重写时，有待送达的过期单或到期事件也派发（它们在日终结算后送达，不必
    /// 等到下一交易日的首个行情事件）。
    pub(crate) fn should_dispatch_lifecycle(&self, kind: &str) -> bool {
        use crate::model::timer::{
            LIFECYCLE_AFTER_TRADING, LIFECYCLE_BEFORE_TRADING, LIFECYCLE_PRE_OPEN,
        };
        let hooks = self.lifecycle_hooks_dispatch;
        match kind {
            LIFECYCLE_PRE_OPEN => return hooks.pre_open,
            LIFECYCLE_BEFORE_TRADING => return hooks.before_trading,
            LIFECYCLE_AFTER_TRADING if hooks.after_trading => return true,
            LIFECYCLE_AFTER_TRADING => {}
            _ => return true,
        }
        !self
            .state
            .order_manager
            .current_step_expired_orders
            .is_empty()
            || !self.recent_expiry_events.is_empty()
    }

    pub(crate) fn datetime_from_ns(timestamp: i64) -> DateTime<Utc> {
        let secs = timestamp.div_euclid(1_000_000_000);
        let nanos = timestamp.rem_euclid(1_000_000_000) as u32;
        Utc.timestamp_opt(secs, nanos)
            .single()
            .expect("Invalid timestamp")
    }

    /// 闭市告警的夜盘模板提示：仅当 `symbol` 是期货、所用时段表没有夜盘、且行情本地时刻
    /// 落在 20:55～02:30（夜盘可能的时间）时返回提示文案。
    pub(crate) fn futures_night_hint(&self, symbol: &str, timestamp: i64) -> Option<String> {
        let instrument = self.instruments.get(symbol)?;
        if instrument.asset_type != crate::model::AssetType::Futures {
            return None;
        }
        let resolver = self.session_resolver.as_ref()?;
        if resolver
            .schedule_for(symbol, Some(crate::model::AssetType::Futures))
            .has_night()
        {
            return None;
        }
        let local = resolver.clock().to_local(timestamp).time();
        let secs = local.num_seconds_from_midnight();
        let in_night = secs >= 20 * 3600 + 55 * 60 || secs <= 2 * 3600 + 30 * 60;
        if !in_night {
            return None;
        }
        // 键用品种代码（代码开头的字母，如 au2406 → au）；取不到时用完整代码。
        let product: String = symbol
            .trim()
            .chars()
            .take_while(|c| c.is_ascii_alphabetic())
            .collect();
        let key = if product.is_empty() {
            symbol.trim().to_string()
        } else {
            product.to_ascii_lowercase()
        };
        Some(format!(
            "该期货所用时段表没有夜盘，而这条行情的本地时刻 {} 在夜盘时间（20:55～02:30）内。如该品种有夜盘，请按夜盘收盘时刻选模板：CN_FUTURES_NIGHT_23（23:00 收）、CN_FUTURES_NIGHT_01（01:00 收）或CN_FUTURES_NIGHT_0230（02:30 收），例如 run_backtest(sessions={{\"{key}\": \"CN_FUTURES_NIGHT_0230\"}})（键是代码前缀，实盘 run_live 同名参数）",
            local.format("%H:%M")
        ))
    }

    fn parsed_timezone(&self) -> Option<Tz> {
        self.timezone_name
            .as_deref()
            .and_then(|name| name.parse::<Tz>().ok())
    }

    /// 引擎时区下的本地时刻(时段改由解析器求后只剩单测使用)。
    #[cfg(test)]
    pub(crate) fn local_time_from_ns(&self, timestamp: i64) -> NaiveTime {
        let utc_dt = Self::datetime_from_ns(timestamp);
        if let Some(tz) = self.parsed_timezone() {
            return utc_dt.with_timezone(&tz).time();
        }
        Self::local_datetime_from_ns(timestamp, self.timezone_offset).time()
    }

    /// 引擎时区下的本地日期时间文本(告警文案用)。
    pub(crate) fn local_datetime_text(&self, timestamp: i64) -> String {
        let utc_dt = Self::datetime_from_ns(timestamp);
        if let Some(tz) = self.parsed_timezone() {
            return utc_dt
                .with_timezone(&tz)
                .format("%Y-%m-%d %H:%M:%S")
                .to_string();
        }
        Self::local_datetime_from_ns(timestamp, self.timezone_offset)
            .format("%Y-%m-%d %H:%M:%S")
            .to_string()
    }

    pub(crate) fn local_date_from_ns(&self, timestamp: i64) -> NaiveDate {
        let utc_dt = Self::datetime_from_ns(timestamp);
        if let Some(tz) = self.parsed_timezone() {
            return utc_dt.with_timezone(&tz).date_naive();
        }
        Self::local_datetime_from_ns(timestamp, self.timezone_offset).date_naive()
    }

    pub(crate) fn local_datetime_from_ns(timestamp: i64, offset_secs: i32) -> DateTime<Utc> {
        let offset_ns = i64::from(offset_secs) * 1_000_000_000;
        Self::datetime_from_ns(timestamp + offset_ns)
    }

    pub(crate) fn call_strategy_for_slot<'py>(
        &mut self,
        py: Python<'py>,
        strategy: &Bound<'py, PyAny>,
        event: &Event,
        slot_index: usize,
        active_orders: Arc<Vec<Order>>,
        step_trades: Vec<Trade>,
        step_rejected_orders: Vec<Order>,
        deliver_timer: bool,
    ) -> PyResult<(
        Vec<Order>,
        Vec<Timer>,
        Vec<String>,
        Option<PendingEnginePlans>,
    )> {
        self.active_strategy_slot = slot_index;
        match event {
            Event::Bar(b) => {
                let previous_cash = self.state.portfolio.cash;
                let previous_account_metrics = self.current_account_metrics();
                // 快路径:symbol 通常已存在于 last_prices,get_mut 命中即可原地更新,
                // 避免每 bar 都为 insert 克隆一次 String。写锁显式限定在本块内,
                // 在调用 get_or_create_strategy_context / Python 之前释放。
                {
                    let mut prices = self.last_prices.write().expect("last_prices 写锁被污染");
                    match prices.get_mut(&b.symbol) {
                        Some(slot) => *slot = b.close,
                        None => {
                            prices.insert(b.symbol.clone(), b.close);
                        }
                    }
                }
                let py_ctx = self.get_or_create_strategy_context(
                    py,
                    slot_index,
                    active_orders,
                    step_trades,
                    step_rejected_orders,
                    previous_cash,
                    previous_account_metrics,
                )?;

                let args = (b.clone(), py_ctx.clone_ref(py));

                let ret = strategy.call_method1("_on_bar_event_and_flush", args)?;
                let pending_plans: Option<PendingEnginePlans> = if ret.is_none() {
                    None
                } else {
                    let (oco, bracket) = ret.extract()?;
                    Some(PendingEnginePlans {
                        oco_groups: oco,
                        bracket_plans: bracket,
                    })
                };

                // Extract orders and timers
                let mut new_orders = Vec::new();
                let mut new_timers = Vec::new();
                let mut canceled_ids = Vec::new();
                {
                    let ctx_ref = py_ctx.borrow(py);
                    // Read from RwLock
                    if let Ok(orders) = ctx_ref.orders_arc.read() {
                        new_orders.extend(orders.clone());
                    }
                    if let Ok(timers) = ctx_ref.timers_arc.read() {
                        new_timers.extend(timers.clone());
                    }
                    if let Ok(canceled) = ctx_ref.canceled_order_ids_arc.read() {
                        canceled_ids.extend(canceled.clone());
                    }
                }
                Ok((new_orders, new_timers, canceled_ids, pending_plans))
            }
            Event::Tick(t) => {
                let previous_cash = self.state.portfolio.cash;
                let previous_account_metrics = self.current_account_metrics();
                {
                    let mut prices = self.last_prices.write().expect("last_prices 写锁被污染");
                    match prices.get_mut(&t.symbol) {
                        Some(slot) => *slot = t.price,
                        None => {
                            prices.insert(t.symbol.clone(), t.price);
                        }
                    }
                }
                let py_ctx = self.get_or_create_strategy_context(
                    py,
                    slot_index,
                    active_orders,
                    step_trades,
                    step_rejected_orders,
                    previous_cash,
                    previous_account_metrics,
                )?;

                let args = (t.clone(), py_ctx.clone_ref(py));

                let ret = strategy.call_method1("_on_tick_event_and_flush", args)?;
                let pending_plans: Option<PendingEnginePlans> = if ret.is_none() {
                    None
                } else {
                    let (oco, bracket) = ret.extract()?;
                    Some(PendingEnginePlans {
                        oco_groups: oco,
                        bracket_plans: bracket,
                    })
                };

                // Extract orders and timers
                let mut new_orders = Vec::new();
                let mut new_timers = Vec::new();
                let mut canceled_ids = Vec::new();
                {
                    let ctx_ref = py_ctx.borrow(py);
                    if let Ok(orders) = ctx_ref.orders_arc.read() {
                        new_orders.extend(orders.clone());
                    }
                    if let Ok(timers) = ctx_ref.timers_arc.read() {
                        new_timers.extend(timers.clone());
                    }
                    if let Ok(canceled) = ctx_ref.canceled_order_ids_arc.read() {
                        canceled_ids.extend(canceled.clone());
                    }
                }
                Ok((new_orders, new_timers, canceled_ids, pending_plans))
            }
            Event::Timer(timer) => {
                let previous_cash = self.state.portfolio.cash;
                let previous_account_metrics = self.current_account_metrics();
                let py_ctx = self.get_or_create_strategy_context(
                    py,
                    slot_index,
                    active_orders,
                    step_trades,
                    step_rejected_orders,
                    previous_cash,
                    previous_account_metrics,
                )?;

                // 属于别的槽位的定时器: 不调本槽位的 on_timer, 只刷本步的订单/成交
                // 等回调(本步撮合产生的成交在 strategy 阶段末清空, 不刷就丢)。
                let method = if deliver_timer {
                    "_on_timer_event_and_flush"
                } else {
                    "_on_foreign_timer_event_and_flush"
                };
                let args = (timer.payload.as_str(), py_ctx.clone_ref(py));
                let ret = strategy.call_method1(method, args)?;
                let pending_plans: Option<PendingEnginePlans> = if ret.is_none() {
                    None
                } else {
                    let (oco, bracket) = ret.extract()?;
                    Some(PendingEnginePlans {
                        oco_groups: oco,
                        bracket_plans: bracket,
                    })
                };

                // Extract orders and timers
                let mut new_orders = Vec::new();
                let mut new_timers = Vec::new();
                let mut canceled_ids = Vec::new();
                {
                    let ctx_ref = py_ctx.borrow(py);
                    if let Ok(orders) = ctx_ref.orders_arc.read() {
                        new_orders.extend(orders.clone());
                    }
                    if let Ok(timers) = ctx_ref.timers_arc.read() {
                        new_timers.extend(timers.clone());
                    }
                    if let Ok(canceled) = ctx_ref.canceled_order_ids_arc.read() {
                        canceled_ids.extend(canceled.clone());
                    }
                }
                Ok((new_orders, new_timers, canceled_ids, pending_plans))
            }
            Event::OrderRequest(_) | Event::OrderValidated(_) | Event::ExecutionReport(_, _) => {
                Ok((Vec::new(), Vec::new(), Vec::new(), None))
            }
        }
    }

    /// 把闭合的窗口 bar 逐个派发给各策略槽的 `_on_window_bar_event_and_flush`。
    ///
    /// 与 `StrategyProcessor` 对基础事件的处理同构(取 ctx → 调 Python → 收订单/
    /// timer/撤单/OCO 计划 → 投递), 但**不更新 last_prices**: 窗口 bar 不是行情事件。
    /// 每个槽是否真的订阅了该 (symbol, freq) 由 Python 侧过滤。
    pub(crate) fn dispatch_window_bars(
        &mut self,
        py: Python<'_>,
        strategy: &Bound<'_, PyAny>,
        bars: Vec<Bar>,
    ) -> PyResult<()> {
        if bars.is_empty() {
            return Ok(());
        }
        self.ensure_strategy_slot_exists();
        self.ensure_strategy_context_capacity();
        let slot_count = self.strategy_slots.len();
        for bar in bars {
            // 与 StrategyProcessor 对基础事件的处理同构(见 strategy.rs 顶部注释):
            // 本步内已知的挂单要在 slot 间累加, 否则多策略下 slot 1 看不到 slot 0
            // 在同一根窗口 bar 上刚提交的单。
            let mut cycle_orders = self.state.order_manager.active_orders.clone();
            let mut active_orders = Arc::new(cycle_orders.clone());
            let step_trades = self.state.order_manager.current_step_trades.clone();
            let step_rejected_orders = self
                .state
                .order_manager
                .current_step_rejected_orders
                .clone();
            for slot_index in 0..slot_count {
                self.active_strategy_slot = slot_index;
                let previous_cash = self.state.portfolio.cash;
                let previous_account_metrics = self.current_account_metrics();
                let py_ctx = self.get_or_create_strategy_context(
                    py,
                    slot_index,
                    active_orders.clone(),
                    step_trades.clone(),
                    step_rejected_orders.clone(),
                    previous_cash,
                    previous_account_metrics,
                )?;
                let slot_strategy = self
                    .strategy_slot_strategies
                    .get(slot_index)
                    .and_then(|slot| slot.as_ref())
                    .map(|slot| slot.clone_ref(py));
                let target = match slot_strategy {
                    Some(ref slot_py) => slot_py.bind(py).clone(),
                    None => strategy.clone(),
                };
                if !target.hasattr("_on_window_bar_event_and_flush")? {
                    continue;
                }
                let ret = target.call_method1(
                    "_on_window_bar_event_and_flush",
                    (bar.clone(), py_ctx.clone_ref(py)),
                )?;
                let pending_plans: Option<PendingEnginePlans> = if ret.is_none() {
                    None
                } else {
                    let (oco, bracket) = ret.extract()?;
                    Some(PendingEnginePlans {
                        oco_groups: oco,
                        bracket_plans: bracket,
                    })
                };
                let (new_orders, new_timers, canceled_ids) = {
                    let ctx_ref = py_ctx.borrow(py);
                    let mut orders = Vec::new();
                    let mut timers = Vec::new();
                    let mut canceled = Vec::new();
                    if let Ok(o) = ctx_ref.orders_arc.read() {
                        orders.extend(o.clone());
                    }
                    if let Ok(t) = ctx_ref.timers_arc.read() {
                        timers.extend(t.clone());
                    }
                    if let Ok(c) = ctx_ref.canceled_order_ids_arc.read() {
                        canceled.extend(c.clone());
                    }
                    (orders, timers, canceled)
                };
                crate::pipeline::stages::strategy::apply_pending_engine_plans(self, pending_plans);
                for id in canceled_ids {
                    self.execution_model.on_cancel(&id);
                    if let Some(cancelled) = self
                        .state
                        .order_manager
                        .cancel_active_order(&id, self.clock.timestamp().unwrap_or(0))
                    {
                        let _ = self
                            .event_manager
                            .send(Event::ExecutionReport(cancelled, None));
                    }
                }
                // 只有多策略才需要在 slot 间累加(单策略下无后续 slot 会读它)。
                // 取本 slot Context 的 `orders` 而非上面的 `new_orders`: 原因同
                // strategy.rs 的同名累加块——`orders_arc` 在带 `event_tx` 的场景下
                // 恒空, 而 `StrategyContext::buy/sell` 总会先 push 进 `orders`。
                if slot_count > 1 {
                    if let Some(Some(slot_ctx)) = self.strategy_contexts.get(slot_index) {
                        let ctx_ref = slot_ctx.borrow(py);
                        cycle_orders.extend(ctx_ref.orders.iter().cloned());
                    }
                    active_orders = Arc::new(cycle_orders.clone());
                }
                for order in new_orders {
                    let _ = self.event_manager.send(Event::OrderRequest(order));
                }
                for t in new_timers {
                    self.timers.push(t);
                }
            }
        }
        Ok(())
    }

    /// 会话结束时闭合全部尾部未满窗口并派发(规格 5.3)。
    ///
    /// 供 `python.rs::run()` 的正常收尾路径与 `KeyboardInterrupt` 收尾路径共用,
    /// 避免两处各写一份同样的 flush 逻辑。写历史与派发顺序同数据阶段
    /// (`update_window` 先于 `dispatch_window_bars`)。
    pub(crate) fn flush_window_tail(
        &mut self,
        py: Python<'_>,
        strategy: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        // `pending_window_bars` 只在"同一基础事件里某窗口已即时闭合、写入
        // pending_window_bars, 但 StrategyProcessor 派发那一段还没来得及跑"时
        // 非空——典型场景是 KeyboardInterrupt 在同一步的 on_bar 回调之后、
        // dispatch_window_bars 之前把 Rust 调用栈中断展开(见
        // pipeline/stages/strategy.rs)。正常的 FeedAction::End 收尾路径里,
        // 每个事件的 dispatch_window_bars 都已经在原地跑完, 这里恒为空, 该路径
        // 行为不变。这些 bar 已经被 DataProcessor 写入过 history(见
        // pipeline/stages/data.rs), 不能再 update_window 一次, 否则会重复写入。
        let mut bars = std::mem::take(&mut self.pending_window_bars);

        let tail = self
            .window_aggregator
            .write()
            .expect("window_aggregator 写锁被污染")
            .flush();
        if !tail.is_empty() {
            if let Ok(mut buffer) = self.history_buffer.write() {
                for bar in &tail {
                    buffer.update_window(bar);
                }
            }
        }
        // 派发顺序 = pending 在前(它们更早闭合), tail 在后, 且只派发一次。
        bars.extend(tail);
        if bars.is_empty() {
            return Ok(());
        }
        self.dispatch_window_bars(py, strategy, bars)
    }

    /// 把 recent_expiry_events 逐条发成 `expiry` 流事件(日常路径与收尾补结算共用)。
    pub(crate) fn emit_expiry_stream_events(&mut self, py: Python<'_>) {
        let recent_expiry_events = self.recent_expiry_events.clone();
        for expiry_event in recent_expiry_events {
            let expiry_symbol = expiry_event.symbol.clone();
            let mut expiry_payload = HashMap::new();
            expiry_payload.insert("symbol", expiry_symbol.clone());
            expiry_payload.insert(
                "asset_type",
                format!("{:?}", expiry_event.asset_type).to_ascii_uppercase(),
            );
            expiry_payload.insert("trading_date", expiry_event.trading_date.clone());
            if let Some(expiry_date) = expiry_event.expiry_date {
                expiry_payload.insert("expiry_date", expiry_date.to_string());
            }
            expiry_payload.insert("quantity_before", expiry_event.quantity_before.to_string());
            expiry_payload.insert("quantity_closed", expiry_event.quantity_closed.to_string());
            expiry_payload.insert("cash_flow", expiry_event.cash_flow.to_string());
            if let Some(ref settlement_type) = expiry_event.settlement_type {
                expiry_payload.insert("settlement_type", settlement_type.clone());
            }
            if let Some(settlement_price) = expiry_event.settlement_price {
                expiry_payload.insert("settlement_price", settlement_price.to_string());
            }
            expiry_payload.insert("reason", expiry_event.reason.clone());
            expiry_payload.insert("description", expiry_event.description.clone());
            if let Some(owner_strategy_id) = self
                .default_strategy_id
                .clone()
                .filter(|text| !text.trim().is_empty())
            {
                expiry_payload.insert("owner_strategy_id", owner_strategy_id);
            }
            self.emit_stream_event(
                py,
                "expiry",
                Some(expiry_symbol.as_str()),
                "info",
                expiry_payload,
            );
        }
    }

    /// 收尾补结算之后, 逐个策略槽位派发 on_expiry(日常路径靠下一根 bar 的回调入口派发,
    /// 收尾时已没有下一根 bar)。
    pub(crate) fn flush_terminal_expiry_events(
        &mut self,
        py: Python<'_>,
        strategy: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if self.recent_expiry_events.is_empty() {
            return Ok(());
        }

        self.ensure_strategy_slot_exists();
        self.ensure_strategy_context_capacity();
        let slot_count = self.strategy_slots.len();
        let active_orders = Arc::new(self.state.order_manager.active_orders.clone());
        // 不清空 step 列表: 这里只派发到期回调, 订单/成交回调归 flush_terminal_pending_order_events。
        let step_trades = self.state.order_manager.current_step_trades.clone();
        let step_rejected_orders = self
            .state
            .order_manager
            .current_step_rejected_orders
            .clone();
        let previous_cash = self.state.portfolio.cash;
        let previous_account_metrics = self.current_account_metrics();

        for slot_index in 0..slot_count {
            self.active_strategy_slot = slot_index;
            let py_ctx = self.get_or_create_strategy_context(
                py,
                slot_index,
                active_orders.clone(),
                step_trades.clone(),
                step_rejected_orders.clone(),
                previous_cash,
                previous_account_metrics,
            )?;
            let slot_strategy = self
                .strategy_slot_strategies
                .get(slot_index)
                .and_then(|slot| slot.as_ref())
                .map(|slot| slot.clone_ref(py));
            if let Some(ref slot_py) = slot_strategy {
                slot_py
                    .bind(py)
                    .call_method1("_flush_terminal_expiry_events", (py_ctx.clone_ref(py),))?;
            } else {
                strategy.call_method1("_flush_terminal_expiry_events", (py_ctx,))?;
            }
        }
        Ok(())
    }

    pub(crate) fn flush_terminal_pending_order_events(
        &mut self,
        py: Python<'_>,
        strategy: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        // 日终过期的 Day 单也在这里派发: 最后一个交易日收盘结算之后已没有下一个
        // 策略事件来触发常规回调入口。
        if self.state.order_manager.current_step_trades.is_empty()
            && self
                .state
                .order_manager
                .current_step_rejected_orders
                .is_empty()
            && self
                .state
                .order_manager
                .current_step_expired_orders
                .is_empty()
            && self
                .state
                .order_manager
                .current_step_finalized_orders
                .is_empty()
        {
            return Ok(());
        }

        self.ensure_strategy_slot_exists();
        self.ensure_strategy_context_capacity();
        let slot_count = self.strategy_slots.len();
        let active_orders = Arc::new(self.state.order_manager.active_orders.clone());
        let step_trades = self.state.order_manager.current_step_trades.clone();
        let step_rejected_orders = self
            .state
            .order_manager
            .current_step_rejected_orders
            .clone();
        let previous_cash = self.state.portfolio.cash;
        let previous_account_metrics = self.current_account_metrics();

        for slot_index in 0..slot_count {
            self.active_strategy_slot = slot_index;
            let py_ctx = self.get_or_create_strategy_context(
                py,
                slot_index,
                active_orders.clone(),
                step_trades.clone(),
                step_rejected_orders.clone(),
                previous_cash,
                previous_account_metrics,
            )?;
            let slot_strategy = self
                .strategy_slot_strategies
                .get(slot_index)
                .and_then(|slot| slot.as_ref())
                .map(|slot| slot.clone_ref(py));
            if let Some(ref slot_py) = slot_strategy {
                slot_py
                    .bind(py)
                    .call_method1("_flush_pending_order_events", (py_ctx.clone_ref(py),))?;
            } else {
                strategy.call_method1("_flush_pending_order_events", (py_ctx,))?;
            }
        }

        self.state.order_manager.current_step_trades.clear();
        self.state
            .order_manager
            .current_step_rejected_orders
            .clear();
        self.state.order_manager.current_step_expired_orders.clear();
        self.state
            .order_manager
            .current_step_finalized_orders
            .clear();
        Ok(())
    }

    pub(crate) fn build_pipeline(&self) -> PipelineRunner {
        let mut pipeline = PipelineRunner::new();
        // 1. Process events from previous iteration (or init)
        pipeline.add_processor(Box::new(ChannelProcessor));

        // 2. Fetch new Data Event
        pipeline.add_processor(Box::new(DataProcessor::new()));

        // 3. Pre-Strategy Execution (Match Pending Orders)
        // For NextOpen/NextClose/NextAverage: Matches orders generated in previous bar against current bar.
        pipeline.add_processor(Box::new(ExecutionProcessor::new(
            ExecutionPhase::PreStrategy,
        )));

        // 4. Process Fills from Pre-Execution immediately (Update Portfolio before Strategy)
        pipeline.add_processor(Box::new(ChannelProcessor));

        // 5. Run Strategy
        pipeline.add_processor(Box::new(StrategyProcessor));

        // 6. Process Order Requests from Strategy immediately (Validate -> Pending)
        pipeline.add_processor(Box::new(ChannelProcessor));

        // 7. Post-Strategy Execution
        // For CurrentClose: Matches orders generated in current bar against current bar.
        pipeline.add_processor(Box::new(ExecutionProcessor::new(
            ExecutionPhase::PostStrategy,
        )));

        // 8. Process Fills from Post-Execution
        pipeline.add_processor(Box::new(ChannelProcessor));

        // 9. Statistics & Cleanup
        pipeline.add_processor(Box::new(StatisticsProcessor));
        pipeline.add_processor(Box::new(CleanupProcessor));

        pipeline
    }
}

/// 缓存的时段区间 `span = [start, end)` 是否覆盖行情事件：bar 区间 `(interval_start, ts]`
/// 整个落在其中；无区间时按 `ts - 1` 这一时刻落在其中。间隙（`Closed`）区间的起点
/// 恰是上一段的结束，`resolve_instant` 把该时刻归入上一段（收盘那一笔），不算覆盖。
fn span_covers_event(
    span: (i64, i64, crate::model::TradingSession),
    interval_start: Option<i64>,
    ts: i64,
) -> bool {
    let (start, end, session) = span;
    match interval_start {
        Some(lo) => lo < ts && start <= lo && ts <= end,
        None => {
            let point = ts.saturating_sub(1);
            point < end
                && (start < point
                    || (start == point && session != crate::model::TradingSession::Closed))
        }
    }
}

#[cfg(test)]
mod span_cache_tests {
    use super::span_covers_event;
    use crate::model::TradingSession;

    #[test]
    fn span_covers_whole_bar_interval_only() {
        let seg = (100, 200, TradingSession::Continuous);
        // (100, 200] 整个落在 [100, 200) 的时段内（终点戳 bar 的最后一笔在 199）
        assert!(span_covers_event(seg, Some(100), 200));
        assert!(span_covers_event(seg, Some(150), 160));
        // 区间起点早于时段开始：与前一段有交集
        assert!(!span_covers_event(seg, Some(99), 150));
        // 区间终点越过时段结束
        assert!(!span_covers_event(seg, Some(150), 201));
        // 非法区间不命中缓存（交给解析器报错）
        assert!(!span_covers_event(seg, Some(160), 160));
    }

    #[test]
    fn span_covers_point_but_not_closing_print_of_gap() {
        let seg = (100, 200, TradingSession::ClosingAuction);
        assert!(span_covers_event(seg, None, 101));
        assert!(span_covers_event(seg, None, 200));
        assert!(!span_covers_event(seg, None, 201));
        let gap = (200, 300, TradingSession::Closed);
        // ts - 1 = 200 恰是上一段结束：解析器归入上一段，缓存的间隙不算覆盖
        assert!(!span_covers_event(gap, None, 201));
        assert!(span_covers_event(gap, None, 202));
        assert!(span_covers_event(gap, None, 300));
        assert!(!span_covers_event(gap, None, 301));
    }
}

#[cfg(test)]
mod context_sessions_tests {
    use super::Engine;
    use crate::model::TradingSession;
    use chrono::NaiveDate;

    fn local_ns(y: i32, m: u32, d: u32, h: u32, mi: u32) -> i64 {
        crate::calendar::clock::LocalClock::new(Some("Asia/Shanghai"), 8 * 3600).from_local(
            NaiveDate::from_ymd_opt(y, m, d)
                .unwrap()
                .and_hms_opt(h, mi, 0)
                .unwrap(),
        )
    }

    #[test]
    fn trading_day_before_first_open_resolves_instant_and_none_without_time() {
        let mut engine = Engine::new();
        // 无上下文时刻：None
        assert_eq!(engine.context_sessions().trading_day, None);
        // 首个交易日开日前（current_trading_day 为空）的 09:00：属当日
        engine
            .clock
            .update(local_ns(2024, 1, 2, 9, 0), TradingSession::Closed);
        assert!(engine.current_trading_day.is_none());
        assert_eq!(
            engine.context_sessions().trading_day,
            NaiveDate::from_ymd_opt(2024, 1, 2)
        );
        // 首日收盘后 15:30（仍未开日）：属下一交易日
        engine
            .clock
            .update(local_ns(2024, 1, 2, 15, 30), TradingSession::Closed);
        assert_eq!(
            engine.context_sessions().trading_day,
            NaiveDate::from_ymd_opt(2024, 1, 3)
        );
        // 查询不开日
        assert!(engine.current_trading_day.is_none());
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::Engine;
    use crate::model::Timer;
    use crate::model::timer::{LIFECYCLE_PRE_OPEN, LIFECYCLE_TIMER_PREFIX};
    use chrono::NaiveDate;

    fn d(m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2024, m, day).unwrap()
    }

    /// 上海本地时间 → UTC 纳秒。
    fn sh(m: u32, day: u32, h: u32, mi: u32) -> i64 {
        crate::calendar::clock::LocalClock::new(Some("Asia/Shanghai"), 8 * 3600)
            .from_local(d(m, day).and_hms_opt(h, mi, 0).unwrap())
    }

    fn lifecycle_timers(engine: &Engine) -> Vec<(i64, String)> {
        let mut out: Vec<(i64, String)> = engine
            .timers
            .iter()
            .filter(|t| t.is_lifecycle())
            .map(|t: &Timer| {
                (
                    t.timestamp,
                    t.payload[LIFECYCLE_TIMER_PREFIX.len()..].to_string(),
                )
            })
            .collect();
        out.sort();
        out
    }

    fn always_open(engine: &mut Engine) {
        engine.session_config.sessions =
            Some(vec![("default".to_string(), "ALWAYS_OPEN".to_string())]);
        engine.invalidate_session_resolver();
    }

    #[test]
    fn backtest_stock_schedule_follows_formula_and_dedupes() {
        let mut engine = Engine::new();
        engine.schedule_lifecycle(d(1, 3), i64::MIN).unwrap();
        assert_eq!(
            lifecycle_timers(&engine),
            vec![
                (sh(1, 3, 9, 15) - 1, "pre_open|2024-01-03".to_string()),
                (sh(1, 3, 9, 30) + 1, "before_trading|2024-01-03".to_string()),
            ]
        );
        // 同一交易日或更早的交易日不重复排程
        engine.schedule_lifecycle(d(1, 3), i64::MIN).unwrap();
        engine.schedule_lifecycle(d(1, 2), i64::MIN).unwrap();
        assert_eq!(lifecycle_timers(&engine).len(), 2);
        // 不晚于 not_before 的时刻跳过（续跑同日）
        engine.schedule_lifecycle(d(1, 4), sh(1, 4, 9, 20)).unwrap();
        let timers = lifecycle_timers(&engine);
        assert_eq!(timers.len(), 3);
        assert_eq!(
            timers[2],
            (sh(1, 4, 9, 30) + 1, "before_trading|2024-01-04".to_string())
        );
    }

    #[test]
    fn always_open_pre_open_waits_for_previous_close_timer() {
        // 全天连续：D 的首段起点即 D-1 收盘；盘前取上一交易日收盘定时器 + 1ns
        let mut backtest = Engine::new();
        always_open(&mut backtest);
        backtest.schedule_lifecycle(d(1, 3), i64::MIN).unwrap();
        let midnight = sh(1, 3, 0, 0);
        assert_eq!(
            lifecycle_timers(&backtest),
            vec![
                (midnight + 2, "pre_open|2024-01-03".to_string()),
                (midnight + 3, "before_trading|2024-01-03".to_string()),
            ]
        );
        // 实盘收盘定时器有宽限：盘前在宽限之后
        let mut live = Engine::new();
        live.state.feed = crate::data::DataFeed::create_live();
        always_open(&mut live);
        live.schedule_lifecycle(d(1, 3), i64::MIN).unwrap();
        let grace = crate::model::timer::LIVE_DAY_CLOSE_GRACE_NS;
        assert_eq!(
            lifecycle_timers(&live),
            vec![
                (midnight + grace + 1, "pre_open|2024-01-03".to_string()),
                (
                    midnight + grace + 2,
                    "before_trading|2024-01-03".to_string()
                ),
            ]
        );
    }

    #[test]
    fn lifecycle_dispatch_gated_per_hook_and_by_pending_deliveries() {
        use crate::model::timer::{
            LIFECYCLE_AFTER_TRADING, LIFECYCLE_BEFORE_TRADING, LIFECYCLE_PRE_OPEN,
        };
        let mut engine = Engine::new();
        // 缺省（直接驱动引擎）全部派发
        for kind in [
            LIFECYCLE_PRE_OPEN,
            LIFECYCLE_BEFORE_TRADING,
            LIFECYCLE_AFTER_TRADING,
        ] {
            assert!(engine.should_dispatch_lifecycle(kind));
        }
        engine.lifecycle_hooks_dispatch = super::LifecycleHooksDispatch {
            pre_open: false,
            before_trading: true,
            after_trading: false,
        };
        assert!(!engine.should_dispatch_lifecycle(LIFECYCLE_PRE_OPEN));
        assert!(engine.should_dispatch_lifecycle(LIFECYCLE_BEFORE_TRADING));
        assert!(!engine.should_dispatch_lifecycle(LIFECYCLE_AFTER_TRADING));
        // 有过期单要送达时收盘后事件照常派发
        engine
            .state
            .order_manager
            .current_step_expired_orders
            .push(crate::model::Order::test_new(
                "o1",
                "600000.SH",
                crate::model::OrderSide::Buy,
                crate::model::OrderType::Market,
                rust_decimal::Decimal::ONE,
            ));
        assert!(engine.should_dispatch_lifecycle(LIFECYCLE_AFTER_TRADING));
        assert!(!engine.should_dispatch_lifecycle(LIFECYCLE_PRE_OPEN));
    }

    #[test]
    fn live_chain_schedules_next_trading_day_on_before_trading() {
        let mut engine = Engine::new();
        engine.state.feed = crate::data::DataFeed::create_live();
        engine.schedule_lifecycle(d(1, 5), i64::MIN).unwrap();
        // 弹出周五的 before_trading 后，堆中出现下周一的 pre_open / before_trading
        let timers = lifecycle_timers(&engine);
        let before = Timer {
            timestamp: timers[1].0,
            payload: format!("{LIFECYCLE_TIMER_PREFIX}{}", timers[1].1),
            owner: None,
        };
        engine.timers.clear();
        engine.chain_live_lifecycle(&before).unwrap();
        assert_eq!(
            lifecycle_timers(&engine),
            vec![
                (sh(1, 8, 9, 15) - 1, "pre_open|2024-01-08".to_string()),
                (sh(1, 8, 9, 30) + 1, "before_trading|2024-01-08".to_string()),
            ]
        );
        // pre_open（当日有连续竞价段）不串联
        let mut engine = Engine::new();
        engine.state.feed = crate::data::DataFeed::create_live();
        let pre_open = Timer {
            timestamp: sh(1, 5, 9, 15) - 1,
            payload: format!("{LIFECYCLE_TIMER_PREFIX}pre_open|2024-01-05"),
            owner: None,
        };
        engine.chain_live_lifecycle(&pre_open).unwrap();
        assert!(lifecycle_timers(&engine).is_empty());
    }

    #[test]
    fn live_chain_at_calendar_end_warns_instead_of_failing() {
        // 显式交易日历到 01-04 为止：01-04 的 before_trading 串联不到下一交易日，
        // 只告警（一次）不报错。内置时段表都有连续竞价段，pre_open 串联分支无法用
        // 内置模板构造，由 before_trading 分支覆盖同一排程函数。
        let mut engine = Engine::new();
        engine.state.feed = crate::data::DataFeed::create_live();
        engine.session_config.trading_calendar = Some(vec![d(1, 2), d(1, 3), d(1, 4)]);
        engine.invalidate_session_resolver();
        let before = Timer {
            timestamp: sh(1, 4, 9, 30) + 1,
            payload: format!("{LIFECYCLE_TIMER_PREFIX}before_trading|2024-01-04"),
            owner: None,
        };
        engine.chain_live_lifecycle(&before).unwrap();
        assert!(lifecycle_timers(&engine).is_empty());
        assert!(engine.lifecycle_chain_warned);
        // 01-03 的 before_trading 串联到 01-04
        let before = Timer {
            timestamp: sh(1, 3, 9, 30) + 1,
            payload: format!("{LIFECYCLE_TIMER_PREFIX}before_trading|2024-01-03"),
            owner: None,
        };
        engine.chain_live_lifecycle(&before).unwrap();
        assert_eq!(
            lifecycle_timers(&engine)[0],
            (sh(1, 4, 9, 15) - 1, "pre_open|2024-01-04".to_string())
        );
    }

    #[test]
    fn live_session_start_schedules_today_or_next_trading_day() {
        // 09:20 开始：当日盘前已过，只排当日开始交易
        let mut engine = Engine::new();
        engine.state.feed = crate::data::DataFeed::create_live();
        engine.schedule_live_lifecycle(sh(1, 2, 9, 20)).unwrap();
        assert_eq!(
            lifecycle_timers(&engine),
            vec![(sh(1, 2, 9, 30) + 1, "before_trading|2024-01-02".to_string())]
        );
        // 10:00 开始：当日开始交易已过，改排下一交易日
        let mut engine = Engine::new();
        engine.state.feed = crate::data::DataFeed::create_live();
        engine.schedule_live_lifecycle(sh(1, 2, 10, 0)).unwrap();
        assert_eq!(
            lifecycle_timers(&engine),
            vec![
                (sh(1, 3, 9, 15) - 1, "pre_open|2024-01-03".to_string()),
                (sh(1, 3, 9, 30) + 1, "before_trading|2024-01-03".to_string()),
            ]
        );
        // 周六：属下周一交易日
        let mut engine = Engine::new();
        engine.state.feed = crate::data::DataFeed::create_live();
        engine.schedule_live_lifecycle(sh(1, 6, 12, 0)).unwrap();
        let timers = lifecycle_timers(&engine);
        assert_eq!(
            timers[0],
            (
                sh(1, 8, 9, 15) - 1,
                format!("{LIFECYCLE_PRE_OPEN}|2024-01-08")
            )
        );
    }
}
