use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::cmp::Ordering;

/// 引擎内部日终结算定时器的 payload 前缀；不转发给策略。
pub(crate) const DAY_CLOSE_TIMER_PREFIX: &str = "__akq_day_close__|";

#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timer {
    #[pyo3(get, set)]
    pub timestamp: i64,
    #[pyo3(get, set)]
    pub payload: String,
}

impl Timer {
    pub(crate) fn is_day_close(&self) -> bool {
        self.payload.starts_with(DAY_CLOSE_TIMER_PREFIX)
    }
}

impl Ord for Timer {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap 是大顶堆：反转后最小时间戳先出；同时间戳时日终结算最后出。
        other
            .timestamp
            .cmp(&self.timestamp)
            .then_with(|| other.is_day_close().cmp(&self.is_day_close()))
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
    fn day_close_timer_pops_last_among_equal_timestamps() {
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(Timer {
            timestamp: 10,
            payload: format!("{DAY_CLOSE_TIMER_PREFIX}2024-01-02"),
        });
        heap.push(Timer {
            timestamp: 10,
            payload: "user".to_string(),
        });
        heap.push(Timer {
            timestamp: 5,
            payload: "early".to_string(),
        });
        let order: Vec<String> = std::iter::from_fn(|| heap.pop().map(|t| t.payload)).collect();
        assert_eq!(order[0], "early");
        assert_eq!(order[1], "user");
        assert!(order[2].starts_with(DAY_CLOSE_TIMER_PREFIX));
    }

    #[test]
    fn day_close_timer_pops_last_regardless_of_push_order() {
        // BinaryHeap 对相等元素的出堆顺序取决于插入顺序；两种插入顺序都要验证。
        for close_first in [true, false] {
            let close = Timer {
                timestamp: 10,
                payload: format!("{DAY_CLOSE_TIMER_PREFIX}2024-01-02"),
            };
            let users: Vec<Timer> = (0..3)
                .map(|i| Timer {
                    timestamp: 10,
                    payload: format!("user{i}"),
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
