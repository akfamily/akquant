pub mod common;
pub mod crypto;
pub mod forex;
pub mod futures;
pub mod matcher;
pub mod option;
pub mod python;
pub mod realtime;
pub mod simulated;
pub mod slippage;
pub mod stock;
pub mod validation;

pub use common::CommonMatcher;
pub use matcher::ExecutionMatcher;
pub use python::PyExecutionMatcher;
pub use realtime::RealtimeExecutionClient;
pub use simulated::SimulatedExecutionClient;
pub use slippage::{FixedSlippage, PercentSlippage, SlippageModel, ZeroSlippage};

use crate::context::EngineContext;
use crate::event::Event;
use crate::model::Order;

/// 交易执行接口 (Execution Client Trait)
pub trait ExecutionClient: Send + Sync {
    /// 接收新订单
    fn on_order(&mut self, order: Order);

    /// 取消订单请求
    fn on_cancel(&mut self, order_id: &str);

    /// 处理市场事件并返回执行报告
    fn on_event(&mut self, event: &Event, ctx: &EngineContext) -> Vec<Event>;

    /// 在一个时间片结束时补充处理延迟的 same-cycle 订单。
    fn finalize_timestamp(&mut self, _events: &[Event], _ctx: &EngineContext) -> Vec<Event> {
        Vec::new()
    }

    /// 设置滑点模型 (仅回测有效)
    fn set_slippage_model(&mut self, _model: Box<dyn SlippageModel>) {}

    /// 设置成交量限制 (仅回测有效)
    fn set_volume_limit(&mut self, _limit: f64) {}

    /// 注册自定义撮合器 (仅回测有效)
    fn register_matcher(
        &mut self,
        _asset_type: crate::model::AssetType,
        _matcher: Box<dyn ExecutionMatcher>,
    ) {
    }

    fn set_futures_validation_options(
        &mut self,
        _enforce_tick_size: bool,
        _enforce_lot_size: bool,
    ) {
    }

    fn set_futures_validation_options_by_prefix(
        &mut self,
        _symbol_prefix: String,
        _enforce_tick_size: Option<bool>,
        _enforce_lot_size: Option<bool>,
    ) {
    }

    /// 是否为实盘模式
    #[allow(dead_code)]
    fn is_live(&self) -> bool {
        false
    }
}
