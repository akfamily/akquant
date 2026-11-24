//! 回测用 K 线模拟 tick(`Engine.simulate_ticks`): 一根 K 线 → 四个模拟 tick。
//!
//! 只做换算, 不碰引擎状态; 谁有资格生成、何时排队由数据阶段决定。

use rust_decimal::Decimal;

use crate::model::{Bar, Tick};

/// 相邻模拟 tick 的最大间隔: 1 秒。
const MAX_STEP_NS: i64 = 1_000_000_000;

/// 一根 K 线的四个模拟 tick, 按时间升序。无法生成时返回空。
///
/// 价格依次为开盘价、两个极值、收盘价: 开盘价离最低价不远于离最高价时先低后高,
/// 否则先高后低。成交量各为 K 线的四分之一。时间戳为 `E − 4δ … E − δ`,
/// `δ = min(1 秒, (E − S) / 5, (E − prev_event_ts) / 5)`(没有区间起点、
/// `prev_event_ts <= 0` 时略去对应项), 严格早于 K 线、严格晚于上一个行情事件。
/// 零宽(`S == E`)或 δ 不为正时不生成。
pub(crate) fn simulated_ticks(bar: &Bar, prev_event_ts: i64) -> Vec<Tick> {
    let end = bar.timestamp;
    let mut step = MAX_STEP_NS;
    if let Some(start) = bar.interval_start {
        if start >= end {
            return Vec::new();
        }
        step = step.min((end - start) / 5);
    }
    if prev_event_ts > 0 {
        step = step.min((end - prev_event_ts) / 5);
    }
    if step <= 0 {
        return Vec::new();
    }
    let low_first = (bar.open - bar.low).abs() <= (bar.open - bar.high).abs();
    let (second, third) = if low_first {
        (bar.low, bar.high)
    } else {
        (bar.high, bar.low)
    };
    let volume = bar.volume / Decimal::from(4);
    [bar.open, second, third, bar.close]
        .into_iter()
        .zip([4_i64, 3, 2, 1])
        .map(|(price, back)| Tick {
            timestamp: end - back * step,
            price,
            volume,
            symbol: bar.symbol.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rust_decimal_macros::dec;

    use super::*;

    const MIN: i64 = 60_000_000_000;
    const END: i64 = 1_704_160_860_000_000_000;

    fn bar(open: Decimal, high: Decimal, low: Decimal, close: Decimal) -> Bar {
        Bar {
            timestamp: END,
            symbol: "A".to_string(),
            open,
            high,
            low,
            close,
            volume: dec!(1000),
            extra: HashMap::new(),
            freq: None,
            interval_start: Some(END - MIN),
        }
    }

    fn prices(ticks: &[Tick]) -> Vec<Decimal> {
        ticks.iter().map(|t| t.price).collect()
    }

    #[test]
    fn low_first_when_open_is_nearer_the_low() {
        let ticks = simulated_ticks(&bar(dec!(10), dec!(10.6), dec!(9.8), dec!(10.4)), 0);
        assert_eq!(
            prices(&ticks),
            vec![dec!(10), dec!(9.8), dec!(10.6), dec!(10.4)]
        );
        assert!(ticks.iter().all(|t| t.volume == dec!(250)));
    }

    #[test]
    fn high_first_when_open_is_nearer_the_high() {
        let ticks = simulated_ticks(&bar(dec!(10), dec!(10.1), dec!(9.5), dec!(9.7)), 0);
        assert_eq!(
            prices(&ticks),
            vec![dec!(10), dec!(10.1), dec!(9.5), dec!(9.7)]
        );
    }

    #[test]
    fn timestamps_are_the_last_four_seconds_before_the_bar() {
        let ticks = simulated_ticks(&bar(dec!(10), dec!(11), dec!(9), dec!(10)), END - MIN);
        let stamps: Vec<i64> = ticks.iter().map(|t| t.timestamp).collect();
        let sec = 1_000_000_000;
        assert_eq!(
            stamps,
            vec![END - 4 * sec, END - 3 * sec, END - 2 * sec, END - sec]
        );
    }

    #[test]
    fn step_shrinks_to_stay_after_the_previous_event() {
        // 上一个行情事件在 2 秒前: δ = 2s / 5 = 0.4s, 四个 tick 都晚于它。
        let prev = END - 2_000_000_000;
        let ticks = simulated_ticks(&bar(dec!(10), dec!(11), dec!(9), dec!(10)), prev);
        assert_eq!(ticks.len(), 4);
        assert_eq!(ticks[0].timestamp, END - 4 * 400_000_000);
        assert!(
            ticks
                .iter()
                .all(|t| t.timestamp > prev && t.timestamp < END)
        );
    }

    #[test]
    fn missing_interval_start_still_generates() {
        let mut b = bar(dec!(10), dec!(11), dec!(9), dec!(10));
        b.interval_start = None;
        assert_eq!(simulated_ticks(&b, 0).len(), 4);
    }

    #[test]
    fn zero_width_bar_generates_nothing() {
        let mut b = bar(dec!(10), dec!(11), dec!(9), dec!(10));
        b.interval_start = Some(END);
        assert!(simulated_ticks(&b, 0).is_empty());
    }
}
