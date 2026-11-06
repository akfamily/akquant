//! 引擎与策略之间的派发边界。
//!
//! 引擎只通过 [`StrategyHandler`] 调用策略, 不直接知道 Python 方法名;
//! Python 策略([`PyStrategy`])只是它的一种实现。这是核心与 Python 解耦的
//! 第一步: 派发点从散落在引擎各处的 `call_method1("...")` 收敛到本文件。
//!
//! 现阶段 trait 仍带 `Python<'_>` 令牌, 上下文仍是 `Py<StrategyContext>`:
//! `StrategyContext` 是 pyclass, Python 策略要跨回调持有它(`self.ctx`)。
//! 去掉这两者要等上下文拆成「纯 Rust 核心 + Python 薄包装」之后再做。

use std::sync::Arc;

use pyo3::prelude::*;

use super::core::PendingEnginePlans;
use crate::context::StrategyContext;
use crate::model::{Bar, Tick};

/// 引擎派发给策略的一次回调。
pub enum StrategyCallback<'a> {
    /// 基础周期 bar。
    Bar(&'a Bar),
    /// 逐笔行情。
    Tick(&'a Tick),
    /// 定时器。`deliver` 为 false 表示定时器属于别的槽位: 不调本槽位的
    /// `on_timer`, 只刷本步的订单/成交等回调。
    Timer { payload: &'a str, deliver: bool },
    /// 闭合的多周期窗口 bar。
    WindowBar(&'a Bar),
    /// 会话收尾: 派发收尾结算产生的到期事件。
    TerminalExpiry,
    /// 会话收尾: 派发最后一步的订单/成交/过期回调。
    TerminalOrderEvents,
}

impl StrategyCallback<'_> {
    /// 收尾回调不产生待注册计划, 返回值被忽略。
    fn is_terminal(&self) -> bool {
        matches!(self, Self::TerminalExpiry | Self::TerminalOrderEvents)
    }
}

/// 引擎眼中的策略。
///
/// 回调内下的单、注册的定时器、撤单都写进 `ctx`, 由引擎在回调返回后收走;
/// 返回值只承载需要引擎登记的 OCO 组 / bracket 计划(常态为 `None`)。
pub trait StrategyHandler: Send + Sync {
    /// 会话开始(数据循环之前)。
    fn on_start(&self, py: Python<'_>) -> PyResult<()>;

    /// 策略要求的历史缓冲深度; `None` 表示不要求。
    fn history_depth(&self, py: Python<'_>) -> Option<usize>;

    /// 是否接收窗口 bar。不接收时引擎跳过 [`StrategyCallback::WindowBar`]。
    fn handles_window_bars(&self, py: Python<'_>) -> PyResult<bool>;

    /// 派发一次回调。
    fn invoke(
        &self,
        py: Python<'_>,
        callback: StrategyCallback<'_>,
        ctx: &Py<StrategyContext>,
    ) -> PyResult<Option<PendingEnginePlans>>;
}

/// 引擎持有的策略句柄(多个槽位、流水线各阶段共享)。
pub type StrategyHandle = Arc<dyn StrategyHandler>;

/// Python 策略对象(`akquant.Strategy` 子类实例)。
///
/// 回调方法名是 Python 侧 `Strategy` 的内部入口, 每个入口把用户回调与订单事件
/// 收尾合并成一次跨界调用, 并返回待注册计划。
pub struct PyStrategy {
    obj: Py<PyAny>,
}

impl PyStrategy {
    pub fn new(obj: Py<PyAny>) -> Self {
        Self { obj }
    }

    pub fn handle(obj: Py<PyAny>) -> StrategyHandle {
        Arc::new(Self::new(obj))
    }
}

impl StrategyHandler for PyStrategy {
    fn on_start(&self, py: Python<'_>) -> PyResult<()> {
        let strategy = self.obj.bind(py);
        if strategy.hasattr("_on_start_internal")? {
            strategy.call_method0("_on_start_internal")?;
        } else {
            strategy.call_method0("on_start")?;
        }
        Ok(())
    }

    fn history_depth(&self, py: Python<'_>) -> Option<usize> {
        let depth = self
            .obj
            .bind(py)
            .getattr("_history_depth")
            .ok()?
            .extract::<usize>()
            .ok()?;
        (depth > 0).then_some(depth)
    }

    fn handles_window_bars(&self, py: Python<'_>) -> PyResult<bool> {
        self.obj.bind(py).hasattr("_on_window_bar_event_and_flush")
    }

    fn invoke(
        &self,
        py: Python<'_>,
        callback: StrategyCallback<'_>,
        ctx: &Py<StrategyContext>,
    ) -> PyResult<Option<PendingEnginePlans>> {
        let strategy = self.obj.bind(py);
        let ctx = ctx.clone_ref(py);
        let terminal = callback.is_terminal();
        let ret = match callback {
            StrategyCallback::Bar(bar) => {
                strategy.call_method1("_on_bar_event_and_flush", (bar.clone(), ctx))?
            }
            StrategyCallback::Tick(tick) => {
                strategy.call_method1("_on_tick_event_and_flush", (tick.clone(), ctx))?
            }
            StrategyCallback::Timer { payload, deliver } => {
                let method = if deliver {
                    "_on_timer_event_and_flush"
                } else {
                    "_on_foreign_timer_event_and_flush"
                };
                strategy.call_method1(method, (payload, ctx))?
            }
            StrategyCallback::WindowBar(bar) => {
                strategy.call_method1("_on_window_bar_event_and_flush", (bar.clone(), ctx))?
            }
            StrategyCallback::TerminalExpiry => {
                strategy.call_method1("_flush_terminal_expiry_events", (ctx,))?
            }
            StrategyCallback::TerminalOrderEvents => {
                strategy.call_method1("_flush_pending_order_events", (ctx,))?
            }
        };
        if terminal || ret.is_none() {
            return Ok(None);
        }
        let (oco_groups, bracket_plans) = ret.extract()?;
        Ok(Some(PendingEnginePlans {
            oco_groups,
            bracket_plans,
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;

    use super::*;
    use crate::engine::Engine;
    use crate::model::instrument::{InstrumentEnum, StockInstrument};
    use crate::model::{AssetType, Instrument, Order, OrderSide, OrderType};

    /// 纯 Rust 策略: 首根 bar 市价买入 100 股, 记录收到的回调。
    #[derive(Default)]
    struct BuyOnFirstBar {
        log: Mutex<Vec<String>>,
    }

    impl StrategyHandler for BuyOnFirstBar {
        fn on_start(&self, _py: Python<'_>) -> PyResult<()> {
            self.log.lock().unwrap().push("start".to_string());
            Ok(())
        }

        fn history_depth(&self, _py: Python<'_>) -> Option<usize> {
            None
        }

        fn handles_window_bars(&self, _py: Python<'_>) -> PyResult<bool> {
            Ok(false)
        }

        fn invoke(
            &self,
            py: Python<'_>,
            callback: StrategyCallback<'_>,
            ctx: &Py<StrategyContext>,
        ) -> PyResult<Option<PendingEnginePlans>> {
            let mut log = self.log.lock().unwrap();
            match callback {
                StrategyCallback::Bar(bar) => {
                    let first = !log.iter().any(|entry| entry.starts_with("bar"));
                    log.push(format!("bar {}", bar.close));
                    if first {
                        let mut ctx = ctx.borrow_mut(py);
                        let mut order = Order::test_new(
                            "native-1",
                            &bar.symbol,
                            OrderSide::Buy,
                            OrderType::Market,
                            dec!(100),
                        );
                        order.created_at = bar.timestamp;
                        order.updated_at = bar.timestamp;
                        order.owner_strategy_id = ctx.strategy_id.clone();
                        ctx.submit_order(order);
                    }
                }
                StrategyCallback::TerminalOrderEvents => log.push("terminal".to_string()),
                _ => {}
            }
            Ok(None)
        }
    }

    fn stock(symbol: &str) -> Instrument {
        Instrument {
            asset_type: AssetType::Stock,
            inner: InstrumentEnum::Stock(StockInstrument {
                symbol: symbol.to_string(),
                lot_size: dec!(100),
                tick_size: dec!(0.01),
                expiry_date: None,
                sellable_after_days: 1,
            }),
        }
    }

    /// 不经任何 Python 策略对象, 用纯 Rust 策略跑完整个引擎会话。
    #[test]
    fn native_strategy_runs_full_session_without_python_strategy() {
        Python::initialize();
        let mut engine = Engine::new();
        engine.add_instrument(stock("A"));
        // 2024-01-02 ~ 04 每天 10:00 (+08:00) 一根 bar, 均在连续竞价时段内。
        let day0 = 1_704_160_800_000_000_000_i64;
        let day_ns = 86_400_000_000_000_i64;
        for (i, px) in [dec!(10), dec!(11), dec!(12)].into_iter().enumerate() {
            engine
                .state
                .feed
                .add_bar(Bar {
                    timestamp: day0 + i as i64 * day_ns,
                    symbol: "A".to_string(),
                    open: px,
                    high: px,
                    low: px,
                    close: px,
                    volume: dec!(100000),
                    extra: HashMap::new(),
                    freq: None,
                    interval_start: None,
                })
                .unwrap();
        }

        let native = Arc::new(BuyOnFirstBar::default());
        let handle: StrategyHandle = native.clone();
        Python::attach(|py| engine.run_with_strategy(py, &handle, false)).unwrap();

        let log = native.log.lock().unwrap().clone();
        assert_eq!(log.first().map(String::as_str), Some("start"));
        assert_eq!(
            log.iter().filter(|e| e.starts_with("bar")).count(),
            3,
            "{log:?}"
        );
        // 默认撮合: 次根 bar 开盘价成交。
        let trades = &engine.state.order_manager.trades;
        assert_eq!(trades.len(), 1, "{log:?}");
        assert_eq!(trades[0].price, dec!(11));
        assert_eq!(
            engine.state.portfolio.positions.get("A").copied(),
            Some(Decimal::from(100))
        );
    }
}
