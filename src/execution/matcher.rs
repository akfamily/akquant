use crate::event::Event;
use crate::execution::slippage::SlippageModel;
use crate::model::{ExecutionPolicyCore, Instrument, Order};
use rust_decimal::Decimal;

/// 撮合上下文
pub struct MatchContext<'a> {
    pub event: &'a Event,
    pub instrument: &'a Instrument,
    pub execution_policy_core: ExecutionPolicyCore,
    pub slippage: &'a dyn SlippageModel,
    pub volume_limit_pct: Decimal,
    /// 这根 bar 上该品种已被先撮合的委托用掉的成交量。量限是整根 bar 共用的:
    /// 本笔最多成交 `成交量 × 比例 − 已用量`。
    pub volume_used: Decimal,
    /// 不做成交量检查(停牌判断与成交量限制都跳过)。集合竞价统一撮合的合成事件:
    /// 竞价价来自 tick(tick 撮合本就不看成交量), 或竞价期行情成交量全为 0。
    pub ignore_volume: bool,
    pub bar_index: usize,
    pub last_price: Option<Decimal>,
    /// 当前事件是模拟 tick(simulate_ticks): 挂着的限价单被越过时按限价成交,
    /// 不按 tick 价给价格改善(相邻模拟 tick 之间的价格路径视为连续)。
    pub simulated_ticks: bool,
}

/// 撮合器接口
pub trait ExecutionMatcher: Send + Sync {
    fn match_order(&self, order: &mut Order, ctx: &MatchContext) -> Option<Event>;
}
