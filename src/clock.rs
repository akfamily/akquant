use crate::model::TradingSession;
use chrono::{DateTime, TimeZone, Utc};

/// 交易时钟
/// 负责维护当前时间、交易时段状态
/// 参考 NautilusTrader 的设计，将时间管理从引擎中分离
#[allow(dead_code)]
pub struct Clock {
    /// 当前时间 (UTC)
    pub now: Option<DateTime<Utc>>,
    /// 当前交易时段
    pub session: TradingSession,
}

#[allow(dead_code)]
impl Clock {
    pub fn new() -> Self {
        Self {
            now: None,
            session: TradingSession::Closed,
        }
    }

    /// 更新时间并同步交易时段状态
    ///
    /// # Arguments
    /// * `timestamp` - Unix 时间戳 (纳秒)
    /// * `session` - 当前交易时段
    pub fn update(&mut self, timestamp: i64, session: TradingSession) {
        let secs = timestamp.div_euclid(1_000_000_000);
        let nanos = timestamp.rem_euclid(1_000_000_000) as u32;
        if let Some(dt) = Utc.timestamp_opt(secs, nanos).single() {
            self.now = Some(dt);
            self.session = session;
        }
    }

    /// 获取当前时间戳 (纳秒)
    pub fn timestamp(&self) -> Option<i64> {
        self.now.and_then(|dt| dt.timestamp_nanos_opt())
    }

    /// 是否处于交易时间 (Continuous or CallAuction)
    pub fn is_trading(&self) -> bool {
        matches!(
            self.session,
            TradingSession::Continuous | TradingSession::CallAuction
        )
    }
}
