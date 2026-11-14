//! 持仓期间极值(MAE / MFE / 持仓内最大回撤)的边走边算, 与"平仓时扫描完整历史"等价.
//!
//! 参照实现 [`reference_scan`] 是改造前 `TradeTracker::process_trade` 里的扫描算法
//! 原样搬过来的: 平仓时在历史序列里取开仓时刻起(含)到平仓时刻(含)的所有行。参照
//! 用不淘汰的完整历史; 被测的 tracker 只拿到容量为 1 的历史(默认容量改小后的实际
//! 情形), 靠持仓记录自己累计。

use super::tracker::{ExcursionHistories, HistorySeries, TradeTracker};
use super::types::ClosedTrade;
use crate::history::SymbolHistory;
use crate::model::{Bar, OrderSide, PositionEffect, Tick, Trade};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;

fn bar(timestamp: i64, high: f64, low: f64) -> Bar {
    let mid = Decimal::from_f64((high + low) / 2.0).unwrap();
    Bar {
        timestamp,
        open: mid,
        high: Decimal::from_f64(high).unwrap(),
        low: Decimal::from_f64(low).unwrap(),
        close: mid,
        volume: Decimal::from(1000),
        symbol: "X".to_string(),
        extra: HashMap::new(),
        freq: None,
        interval_start: None,
    }
}

fn tick(timestamp: i64, price: f64) -> Tick {
    Tick {
        timestamp,
        price: Decimal::from_f64(price).unwrap(),
        volume: Decimal::from(10),
        symbol: "X".to_string(),
    }
}

fn trade(side: OrderSide, quantity: i64, price: f64, timestamp: i64) -> Trade {
    Trade {
        id: format!("t{timestamp}"),
        order_id: format!("o{timestamp}"),
        symbol: "X".to_string(),
        side,
        quantity: Decimal::from(quantity),
        price: Decimal::from_f64(price).unwrap(),
        commission: Decimal::ZERO,
        timestamp,
        bar_index: timestamp as usize,
        position_effect: PositionEffect::Auto,
        owner_strategy_id: None,
    }
}

/// 改造前的扫描算法(原样): 返回 (mae, mfe, max_drawdown_pct)。
fn reference_scan(hist: &SymbolHistory, closed: &ClosedTrade) -> (f64, f64, f64) {
    let long = closed.side == "Long";
    let start_idx = match hist.timestamps.binary_search(&closed.entry_time) {
        Ok(i) | Err(i) => i,
    };
    let end_idx = match hist.timestamps.binary_search(&closed.exit_time) {
        Ok(i) | Err(i) => i,
    };
    let entry = closed.entry_price;
    let mut max_high = entry;
    let mut min_low = entry;
    let mut extreme = entry;
    let mut max_dd = 0.0;
    if start_idx < hist.timestamps.len() {
        let limit = (end_idx + 1).min(hist.timestamps.len());
        for i in start_idx..limit {
            let h = hist.highs[i];
            let l = hist.lows[i];
            if h > max_high {
                max_high = h;
            }
            if l < min_low {
                min_low = l;
            }
            if long {
                if h > extreme {
                    extreme = h;
                }
                if extreme > 0.0 {
                    let dd = (extreme - l) / extreme;
                    if dd > max_dd {
                        max_dd = dd;
                    }
                }
            } else {
                if l < extreme {
                    extreme = l;
                }
                if extreme > 0.0 {
                    let dd = (h - extreme) / extreme;
                    if dd > max_dd {
                        max_dd = dd;
                    }
                }
            }
        }
    }
    let p = closed.exit_price;
    max_high = max_high.max(p);
    min_low = min_low.min(p);
    if long {
        if p > extreme {
            extreme = p;
        }
        if extreme > 0.0 {
            let dd = (extreme - p) / extreme;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        (
            (min_low - entry) / entry * 100.0,
            (max_high - entry) / entry * 100.0,
            max_dd * 100.0,
        )
    } else {
        if p < extreme {
            extreme = p;
        }
        if extreme > 0.0 {
            let dd = (p - extreme) / extreme;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        (
            (entry - max_high) / entry * 100.0,
            (entry - min_low) / entry * 100.0,
            max_dd * 100.0,
        )
    }
}

fn assert_matches_reference(closed: &[ClosedTrade], full: &SymbolHistory) {
    assert!(!closed.is_empty(), "没有产生平仓, 测试没有覆盖到");
    for (i, c) in closed.iter().enumerate() {
        let (mae, mfe, dd) = reference_scan(full, c);
        assert_eq!(c.mae.to_bits(), mae.to_bits(), "第 {i} 笔 {c:?} mae");
        assert_eq!(c.mfe.to_bits(), mfe.to_bits(), "第 {i} 笔 {c:?} mfe");
        assert_eq!(
            c.max_drawdown_pct.to_bits(),
            dd.to_bits(),
            "第 {i} 笔 {c:?} dd"
        );
    }
}

/// 随机行情与随机买卖(含分批平仓、反手、多笔 FIFO 持仓): tracker 只看容量为 1 的
/// 历史, 结果与参照算法在完整历史上的扫描逐位相同。
#[test]
fn excursion_matches_full_history_scan_with_capacity_one() {
    let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move |modulo: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % modulo
    };
    let mut tracker = TradeTracker::new();
    let mut small = SymbolHistory::new(1);
    let mut full = SymbolHistory::new(1_000_000);
    let mut price = 100.0_f64;
    for step in 1..=20_000_i64 {
        price = (price + (next(201) as f64 - 100.0) / 50.0).max(1.0);
        let spread = next(50) as f64 / 25.0;
        let row = bar(step * 60, price + spread, (price - spread).max(0.5));
        small.push(&row);
        full.push(&row);
        tracker.observe_history_row(
            "X",
            HistorySeries::Bar,
            row.timestamp,
            row.high.to_f64().unwrap(),
            row.low.to_f64().unwrap(),
        );
        if next(7) == 0 {
            let side = if next(2) == 0 {
                OrderSide::Buy
            } else {
                OrderSide::Sell
            };
            let fill = price + (next(21) as f64 - 10.0) / 20.0;
            let histories = ExcursionHistories {
                bar: Some(&small),
                tick: None,
            };
            tracker.process_trade(
                &trade(side, 1 + next(3) as i64, fill, step * 60),
                Decimal::ONE,
                None,
                Some(histories),
                Decimal::ZERO,
            );
        }
    }
    assert!(tracker.closed_trades.len() > 1000);
    assert_matches_reference(&tracker.closed_trades.to_vec(), &full);
}

/// 双流: 平仓时有 bar 序列就按 bar 算, 没有才按 tick(改造前的取法)。tick 先于 bar
/// 出现时, bar 序列是开仓后才建立的, 只含开仓之后的 bar。
#[test]
fn excursion_prefers_bar_series_like_before() {
    let mut tracker = TradeTracker::new();
    let mut ticks_small = SymbolHistory::new(1);
    let mut ticks_full = SymbolHistory::new(1_000);
    let mut bars_small = SymbolHistory::new(1);
    let mut bars_full = SymbolHistory::new(1_000);

    // 只有 tick: 开仓, 价格下探到 90, 平仓 → 按 tick 算
    for (ts, px) in [(1, 100.0), (2, 95.0), (3, 90.0), (4, 98.0)] {
        let t = tick(ts, px);
        ticks_small.push_tick(&t);
        ticks_full.push_tick(&t);
        tracker.observe_history_row("X", HistorySeries::Tick, ts, px, px);
        if ts == 1 || ts == 4 {
            let side = if ts == 1 {
                OrderSide::Buy
            } else {
                OrderSide::Sell
            };
            let histories = ExcursionHistories {
                bar: None,
                tick: Some(&ticks_small),
            };
            tracker.process_trade(
                &trade(side, 1, px, ts),
                Decimal::ONE,
                None,
                Some(histories),
                Decimal::ZERO,
            );
        }
    }
    assert_matches_reference(&tracker.closed_trades.to_vec()[..1], &ticks_full);
    assert_eq!(tracker.closed_trades[0].mae, -10.0);

    // tick 上开仓, 之后才有 bar; 平仓时 bar 序列已存在 → 按 bar 算(只含开仓后的 bar)
    let t = tick(10, 100.0);
    ticks_small.push_tick(&t);
    ticks_full.push_tick(&t);
    tracker.observe_history_row("X", HistorySeries::Tick, 10, 100.0, 100.0);
    tracker.process_trade(
        &trade(OrderSide::Buy, 1, 100.0, 10),
        Decimal::ONE,
        None,
        Some(ExcursionHistories {
            bar: None,
            tick: Some(&ticks_small),
        }),
        Decimal::ZERO,
    );
    for (ts, h, l) in [(20, 104.0, 97.0), (30, 103.0, 95.0)] {
        let b = bar(ts, h, l);
        bars_small.push(&b);
        bars_full.push(&b);
        tracker.observe_history_row("X", HistorySeries::Bar, ts, h, l);
    }
    tracker.process_trade(
        &trade(OrderSide::Sell, 1, 99.0, 30),
        Decimal::ONE,
        None,
        Some(ExcursionHistories {
            bar: Some(&bars_small),
            tick: Some(&ticks_small),
        }),
        Decimal::ZERO,
    );
    assert_matches_reference(&tracker.closed_trades.to_vec()[1..], &bars_full);
    assert_eq!(tracker.closed_trades[1].mae, -5.0);
    assert_eq!(tracker.closed_trades[1].mfe, 4.0);
}

/// 没有传历史(到期结算、强制平仓的成交)时 MAE / MFE / 回撤仍记 0, 与改造前一致。
#[test]
fn excursion_without_histories_stays_zero() {
    let mut tracker = TradeTracker::new();
    tracker.observe_history_row("X", HistorySeries::Bar, 1, 120.0, 80.0);
    tracker.process_trade(
        &trade(OrderSide::Buy, 1, 100.0, 1),
        Decimal::ONE,
        None,
        None,
        Decimal::ZERO,
    );
    tracker.observe_history_row("X", HistorySeries::Bar, 2, 130.0, 70.0);
    tracker.process_trade(
        &trade(OrderSide::Sell, 1, 100.0, 2),
        Decimal::ONE,
        None,
        None,
        Decimal::ZERO,
    );
    let c = &tracker.closed_trades[0];
    assert_eq!((c.mae, c.mfe, c.max_drawdown_pct), (0.0, 0.0, 0.0));
}

/// 未平持仓的累计状态随存档保存, 恢复后接着累计, 结果与不中断一致。
#[test]
fn excursion_survives_snapshot_roundtrip() {
    let run = |roundtrip: bool| {
        let mut tracker = TradeTracker::new();
        let mut hist = SymbolHistory::new(1);
        let mut feed = |tracker: &mut TradeTracker, ts: i64, h: f64, l: f64| {
            let b = bar(ts, h, l);
            hist.push(&b);
            tracker.observe_history_row("X", HistorySeries::Bar, ts, h, l);
            hist.clone()
        };
        let h = feed(&mut tracker, 1, 101.0, 99.0);
        tracker.process_trade(
            &trade(OrderSide::Buy, 1, 100.0, 1),
            Decimal::ONE,
            None,
            Some(ExcursionHistories {
                bar: Some(&h),
                tick: None,
            }),
            Decimal::ZERO,
        );
        feed(&mut tracker, 2, 100.0, 60.0);
        if roundtrip {
            let bytes = rmp_serde::to_vec(&tracker).unwrap();
            tracker = rmp_serde::from_slice(&bytes).unwrap();
        }
        let h = feed(&mut tracker, 3, 110.0, 105.0);
        tracker.process_trade(
            &trade(OrderSide::Sell, 1, 108.0, 3),
            Decimal::ONE,
            None,
            Some(ExcursionHistories {
                bar: Some(&h),
                tick: None,
            }),
            Decimal::ZERO,
        );
        let c = tracker.closed_trades[0].clone();
        (c.mae, c.mfe, c.max_drawdown_pct)
    };
    assert_eq!(run(true), run(false));
    assert_eq!(run(false).0, -40.0);
}

/// 开仓之后才到达、时间戳却早于开仓时刻的行(乱序行情)不计入, 与改造前按开仓
/// 时刻截取历史一致。
#[test]
fn excursion_ignores_rows_older_than_entry() {
    let mut tracker = TradeTracker::new();
    let mut hist = SymbolHistory::new(1);
    let b = bar(100, 101.0, 99.0);
    hist.push(&b);
    tracker.observe_history_row("X", HistorySeries::Bar, 100, 101.0, 99.0);
    tracker.process_trade(
        &trade(OrderSide::Buy, 1, 100.0, 100),
        Decimal::ONE,
        None,
        Some(ExcursionHistories {
            bar: Some(&hist),
            tick: None,
        }),
        Decimal::ZERO,
    );
    // 晚到的旧行情: 时间戳 90 早于开仓时刻 100, 不应把 MAE 拉到 -50%
    tracker.observe_history_row("X", HistorySeries::Bar, 90, 100.0, 50.0);
    tracker.observe_history_row("X", HistorySeries::Bar, 110, 102.0, 98.0);
    let b = bar(110, 102.0, 98.0);
    hist.push(&b);
    tracker.process_trade(
        &trade(OrderSide::Sell, 1, 100.0, 110),
        Decimal::ONE,
        None,
        Some(ExcursionHistories {
            bar: Some(&hist),
            tick: None,
        }),
        Decimal::ZERO,
    );
    assert_eq!(tracker.closed_trades[0].mae, -2.0);
}
