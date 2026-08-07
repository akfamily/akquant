//! 外部信号注入端口(paper 模式后端)。
//!
//! 让**引擎线程之外**的线程(HTTP webhook / MQ 消费者 / 队列监听)把交易指令送进
//! 引擎:构造 `Order` 后经 `EventManager` 的 crossbeam sender 发 `Event::OrderRequest`。
//! `ChannelProcessor` 是 pipeline 首位, 会在下一轮循环立刻排空它 —— 加上
//! `DataProcessor` 的 `select!` 唤醒(见 `data/client.rs` 的 `wait_peek_with_wakeup`),
//! 注入到落地的延迟是微秒级, 不受行情到达节奏影响。
//!
//! **为何必须在引擎线程之外**:`Engine::run(&mut self)` 在整个会话期间独占可变借用
//! 引擎对象, 任何在策略回调内经 Python 侧触达 `Engine` 的调用都会
//! `RuntimeError: Already borrowed`。本端口持有的是 channel 的 sender 克隆
//! (crossbeam `Sender` 是 `Send + Sync + Clone`), 与 pyclass 借用完全无关。
//!
//! **适用范围**:仅 `trading_mode='paper'`。`broker_live` 下引擎的
//! `RealtimeExecutionClient::on_order` 是空实现, 经此注入的订单会通过风控、进入
//! `active_orders`, 但既不会被撮合也不会报到柜台。broker_live 的外部信号须走
//! Python 侧的 `BrokerOrderSubmitter`(见 `akquant.signal` 的 broker 后端)。

use crossbeam_channel::Sender;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use uuid::Uuid;

use crate::event::Event;
use crate::model::{Order, OrderSide, OrderStatus, OrderType, PositionEffect, TimeInForce};

fn to_decimal(value: f64, field: &str) -> PyResult<Decimal> {
    Decimal::from_f64(value)
        .ok_or_else(|| PyValueError::new_err(format!("{field} 不是可用的数值: {value}")))
}

fn parse_side(side: &str) -> PyResult<OrderSide> {
    match side.trim().to_ascii_lowercase().as_str() {
        "buy" => Ok(OrderSide::Buy),
        "sell" => Ok(OrderSide::Sell),
        other => Err(PyValueError::new_err(format!(
            "side 必须是 'Buy' 或 'Sell', 收到 {other:?}"
        ))),
    }
}

fn parse_order_type(order_type: Option<&str>, has_price: bool) -> PyResult<OrderType> {
    match order_type.map(|value| value.trim().to_ascii_lowercase()) {
        None => Ok(if has_price {
            OrderType::Limit
        } else {
            OrderType::Market
        }),
        Some(value) => match value.as_str() {
            "limit" => Ok(OrderType::Limit),
            "market" => Ok(OrderType::Market),
            other => Err(PyValueError::new_err(format!(
                "order_type 仅支持 'Limit' / 'Market', 收到 {other:?}"
            ))),
        },
    }
}

/// 外部信号注入端口(paper 模式)。
///
/// 由 `Engine.signal_port(strategy_id)` 取得, **必须在 `run()` 之前取**(取的是
/// channel sender 的克隆, 取到后即与引擎对象解耦, 可安全交给任意线程)。
#[gen_stub_pyclass]
#[pyclass]
pub struct SignalPort {
    tx: Sender<Event>,
    owner_strategy_id: String,
}

impl SignalPort {
    pub(crate) fn new(tx: Sender<Event>, owner_strategy_id: String) -> Self {
        Self {
            tx,
            owner_strategy_id,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl SignalPort {
    /// 归属策略 id(风控限额按它路由)
    #[getter]
    fn owner_strategy_id(&self) -> &str {
        &self.owner_strategy_id
    }

    /// 注入一笔委托, 返回本地订单 id。
    ///
    /// 线程安全:可从任意线程调用。订单进入引擎后仍要过完整风控
    /// (`ChannelProcessor` → `risk_manager.check_and_adjust`), 被拒则触发
    /// 策略的 `on_reject`。
    ///
    /// :param symbol: 标的代码
    /// :param side: ``"Buy"`` / ``"Sell"``
    /// :param quantity: 委托数量(须为正)
    /// :param price: 委托价; 省略则为市价单
    /// :param order_type: ``"Limit"`` / ``"Market"``; 省略则按 price 有无推断
    /// :param tag: 自定义标记(建议放信号平台的 signal_id, 便于回溯)
    /// :return: 本地订单 id (UUID)
    #[pyo3(signature = (symbol, side, quantity, price=None, order_type=None, tag=None))]
    fn submit(
        &self,
        symbol: &str,
        side: &str,
        quantity: f64,
        price: Option<f64>,
        order_type: Option<&str>,
        tag: Option<&str>,
    ) -> PyResult<String> {
        let symbol = symbol.trim();
        if symbol.is_empty() {
            return Err(PyValueError::new_err("symbol 不能为空"));
        }
        if !(quantity > 0.0) {
            return Err(PyValueError::new_err(format!(
                "quantity 必须为正, 收到 {quantity}"
            )));
        }
        let side = parse_side(side)?;
        let resolved_type = parse_order_type(order_type, price.is_some())?;
        let quantity = to_decimal(quantity, "quantity")?;
        let price = match price {
            Some(value) => Some(to_decimal(value, "price")?),
            None => None,
        };

        let id = Uuid::new_v4().to_string();
        // created_at 留 0: 外部注入时无法取引擎时钟(那需要借用 Engine)。
        // ChannelProcessor 消费时会以引擎当前时间改写 updated_at; created_at 为 0
        // 使 `reject_missing_symbol_orders` 的 `created_at <= last_timestamp` 判定
        // 恒成立 —— 停牌/无行情标的的注入单因此能在首个可撮合切片终态化,
        // 不会累积挤占保证金(issue #329 家族)。
        let order = Order {
            id: id.clone(),
            symbol: symbol.to_string(),
            side,
            order_type: resolved_type,
            quantity,
            price,
            time_in_force: TimeInForce::GTC,
            trigger_price: None,
            trail_offset: None,
            trail_reference_price: None,
            fill_policy_override: None,
            slippage_type_override: None,
            slippage_value_override: None,
            commission_type_override: None,
            commission_value_override: None,
            graph_id: None,
            parent_order_id: None,
            order_role: crate::model::OrderRole::Standalone,
            position_effect: PositionEffect::default(),
            status: OrderStatus::New,
            filled_quantity: Decimal::ZERO,
            average_filled_price: None,
            created_at: 0,
            updated_at: 0,
            commission: Decimal::ZERO,
            tag: tag.unwrap_or_default().to_string(),
            reject_reason: String::new(),
            owner_strategy_id: Some(self.owner_strategy_id.clone()),
            allow_quantity_auto_resize: false,
            reduce_only: false,
        };
        self.tx
            .send(Event::OrderRequest(order))
            .map_err(|err| PyValueError::new_err(format!("引擎已停止, 无法注入指令: {err}")))?;
        Ok(id)
    }
}
