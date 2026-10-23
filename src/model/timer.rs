use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::cmp::Ordering;

/// 引擎内部定时器的保留前缀：用户定时器 payload 不得以它开头。
pub(crate) const RESERVED_TIMER_PREFIX: &str = "__akq_";

/// 引擎内部日终结算定时器的 payload 前缀；不转发给策略。
pub(crate) const DAY_CLOSE_TIMER_PREFIX: &str = "__akq_day_close__|";

/// 引擎生成的交易日生命周期定时器 payload 前缀，转给策略触发交易日钩子：
/// `__akq_lifecycle__|<pre_open|before_trading|after_trading>|<YYYY-MM-DD>`。
pub(crate) const LIFECYCLE_TIMER_PREFIX: &str = "__akq_lifecycle__|";

/// 生命周期事件种类。
pub(crate) const LIFECYCLE_PRE_OPEN: &str = "pre_open";
pub(crate) const LIFECYCLE_BEFORE_TRADING: &str = "before_trading";
pub(crate) const LIFECYCLE_AFTER_TRADING: &str = "after_trading";

/// 生命周期定时器的 payload。
pub(crate) fn lifecycle_payload(kind: &str, day: chrono::NaiveDate) -> String {
    format!("{LIFECYCLE_TIMER_PREFIX}{kind}|{day}")
}

/// 实盘日终结算相对会话收盘的宽限(纳秒)。行情源在收盘后才推送最后一根已收盘
/// bar(如 klinedata 15:00 之后推 14:59~15:00 的 bar), 收盘即结算会让收盘快照用
/// 旧价、且那根 bar 的 on_bar 里 T+1 已解冻。阶段 7 改为可配置。
pub(crate) const LIVE_DAY_CLOSE_GRACE_NS: i64 = 60_000_000_000;

/// 交易日收盘定时器的触发时刻: 回测为收盘 + 1ns, 实盘为收盘 + 宽限。
pub(crate) fn day_close_timer_timestamp(session_day_end: i64, live: bool) -> i64 {
    if live {
        session_day_end + LIVE_DAY_CLOSE_GRACE_NS
    } else {
        session_day_end + 1
    }
}

/// 用户定时器 payload 不能占用引擎内部保留前缀 `__akq_`。
pub(crate) fn check_user_timer_payload(payload: &str) -> pyo3::PyResult<()> {
    if payload.starts_with(RESERVED_TIMER_PREFIX) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "定时器 payload 不能以 {RESERVED_TIMER_PREFIX:?} 开头（引擎内部保留，如日终结算专用前缀 {DAY_CLOSE_TIMER_PREFIX:?}），请换一个 payload 名称"
        )));
    }
    Ok(())
}

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct Timer {
    #[pyo3(get, set)]
    pub timestamp: i64,
    #[pyo3(get, set)]
    pub payload: String,
    /// 所属策略 id。`ctx.schedule`/`schedule_*` 登记的定时器带登记它的槽位 id，
    /// 只派给该槽位；引擎生成的生命周期/日终定时器与 `Engine.add_timer` 登记的
    /// 定时器为 `None`，派给所有槽位。不参与相等与排序比较。
    #[pyo3(get)]
    pub owner: Option<String>,
}

impl PartialEq for Timer {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp && self.payload == other.payload
    }
}

impl Eq for Timer {}

impl Timer {
    pub(crate) fn is_day_close(&self) -> bool {
        self.payload.starts_with(DAY_CLOSE_TIMER_PREFIX)
    }

    pub(crate) fn is_lifecycle(&self) -> bool {
        self.payload.starts_with(LIFECYCLE_TIMER_PREFIX)
    }

    /// 生命周期定时器的（种类, 交易日）；不是生命周期定时器或格式不符时为 `None`。
    pub(crate) fn lifecycle(&self) -> Option<(&str, chrono::NaiveDate)> {
        let rest = self.payload.strip_prefix(LIFECYCLE_TIMER_PREFIX)?;
        let (kind, day) = rest.split_once('|')?;
        let day = chrono::NaiveDate::parse_from_str(day, "%Y-%m-%d").ok()?;
        Some((kind, day))
    }
}

impl Timer {
    /// 同时间戳定时器的出堆次序：生命周期事件（盘前/开始交易）最先，日终结算最后。
    fn tie_rank(&self) -> u8 {
        if self.is_lifecycle() {
            0
        } else if self.is_day_close() {
            2
        } else {
            1
        }
    }
}

impl Ord for Timer {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap 是大顶堆：反转后最小时间戳先出；同时间戳时生命周期事件先出、
        // 日终结算最后出。
        other
            .timestamp
            .cmp(&self.timestamp)
            .then_with(|| other.tie_rank().cmp(&self.tie_rank()))
    }
}

impl PartialOrd for Timer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_close_timer_grace_applies_only_in_live_mode() {
        let close = 1_000_000_000_000;
        assert_eq!(day_close_timer_timestamp(close, false), close + 1);
        assert_eq!(
            day_close_timer_timestamp(close, true),
            close + LIVE_DAY_CLOSE_GRACE_NS
        );
        assert_eq!(LIVE_DAY_CLOSE_GRACE_NS, 60_000_000_000);
    }

    #[test]
    fn user_timer_payload_rejects_reserved_prefix() {
        assert!(check_user_timer_payload("rebalance").is_ok());
        assert!(check_user_timer_payload("__akq_day_close__|2024-01-02").is_err());
        assert!(check_user_timer_payload("__akq_lifecycle__|x").is_err());
        // 框架自用的定时器 payload 不受影响
        assert!(check_user_timer_payload("__daily__|09:30|x").is_ok());
        assert!(check_user_timer_payload("__framework_cross_section__").is_ok());
        assert!(DAY_CLOSE_TIMER_PREFIX.starts_with(RESERVED_TIMER_PREFIX));
        assert!(LIFECYCLE_TIMER_PREFIX.starts_with(RESERVED_TIMER_PREFIX));
    }

    #[test]
    fn lifecycle_payload_roundtrips() {
        let day = chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
        let timer = Timer {
            timestamp: 1,
            payload: lifecycle_payload(LIFECYCLE_BEFORE_TRADING, day),
            owner: None,
        };
        assert_eq!(timer.payload, "__akq_lifecycle__|before_trading|2024-01-02");
        assert!(timer.is_lifecycle());
        assert_eq!(timer.lifecycle(), Some((LIFECYCLE_BEFORE_TRADING, day)));
        let user = Timer {
            timestamp: 1,
            payload: "rebalance".to_string(),
            owner: None,
        };
        assert!(!user.is_lifecycle());
        assert_eq!(user.lifecycle(), None);
    }

    #[test]
    fn day_close_timer_pops_last_among_equal_timestamps() {
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(Timer {
            timestamp: 10,
            payload: format!("{DAY_CLOSE_TIMER_PREFIX}2024-01-02"),
            owner: None,
        });
        heap.push(Timer {
            timestamp: 10,
            payload: "user".to_string(),
            owner: None,
        });
        heap.push(Timer {
            timestamp: 5,
            payload: "early".to_string(),
            owner: None,
        });
        let order: Vec<String> = std::iter::from_fn(|| heap.pop().map(|t| t.payload)).collect();
        assert_eq!(order[0], "early");
        assert_eq!(order[1], "user");
        assert!(order[2].starts_with(DAY_CLOSE_TIMER_PREFIX));
    }

    #[test]
    fn lifecycle_timer_pops_first_among_equal_timestamps() {
        for lifecycle_first in [true, false] {
            let lifecycle = Timer {
                timestamp: 10,
                payload: "__akq_lifecycle__|before_trading|2024-01-02".to_string(),
                owner: None,
            };
            let user = Timer {
                timestamp: 10,
                payload: "__framework_cross_section__|2024-01-02|9".to_string(),
                owner: None,
            };
            let mut heap = std::collections::BinaryHeap::new();
            if lifecycle_first {
                heap.push(lifecycle.clone());
                heap.push(user.clone());
            } else {
                heap.push(user.clone());
                heap.push(lifecycle.clone());
            }
            assert_eq!(heap.pop(), Some(lifecycle));
            assert_eq!(heap.pop(), Some(user));
        }
    }

    #[test]
    fn day_close_timer_pops_last_regardless_of_push_order() {
        // BinaryHeap 对相等元素的出堆顺序取决于插入顺序；两种插入顺序都要验证。
        for close_first in [true, false] {
            let close = Timer {
                timestamp: 10,
                payload: format!("{DAY_CLOSE_TIMER_PREFIX}2024-01-02"),
                owner: None,
            };
            let users: Vec<Timer> = (0..3)
                .map(|i| Timer {
                    timestamp: 10,
                    payload: format!("user{i}"),
                    owner: None,
                })
                .collect();
            let mut heap = std::collections::BinaryHeap::new();
            if close_first {
                heap.push(close.clone());
                heap.extend(users);
            } else {
                heap.extend(users);
                heap.push(close.clone());
            }
            let order: Vec<Timer> = std::iter::from_fn(|| heap.pop()).collect();
            assert_eq!(order.len(), 4);
            assert!(order[..3].iter().all(|t| !t.is_day_close()), "{order:?}");
            assert!(order[3].is_day_close(), "{order:?}");
        }
    }
}
