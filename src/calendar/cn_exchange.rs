//! 内置的中国交易所交易日历：覆盖范围内休市的工作日表。
//!
//! 数据文件 `cn_exchange_closed.txt` 由 `scripts/update_cn_exchange_calendar.py` 生成
//! （取自上交所交易日历）。覆盖范围内周末不开市，工作日除表中日期外都是交易日；
//! 覆盖范围之外没有数据，调用方按“周一至周五”推算。

use chrono::NaiveDate;
use std::sync::OnceLock;

static RAW: &str = include_str!("cn_exchange_closed.txt");

struct Table {
    first: NaiveDate,
    last: NaiveDate,
    /// 升序。
    closed: Vec<NaiveDate>,
}

fn parse(raw: &str) -> Table {
    let date = |s: &str| NaiveDate::parse_from_str(s, "%Y-%m-%d").expect("内置休市日表日期格式");
    let mut coverage = None;
    let mut closed = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("coverage ") {
            let mut parts = rest.split_whitespace();
            let first = date(parts.next().expect("coverage 起点"));
            let last = date(parts.next().expect("coverage 终点"));
            coverage = Some((first, last));
        } else {
            closed.push(date(line));
        }
    }
    let (first, last) = coverage.expect("内置休市日表缺少 coverage 行");
    closed.sort();
    closed.dedup();
    Table {
        first,
        last,
        closed,
    }
}

fn table() -> &'static Table {
    static TABLE: OnceLock<Table> = OnceLock::new();
    TABLE.get_or_init(|| parse(RAW))
}

/// 内置日历的覆盖范围 `(首日, 末日)`（含两端）。
pub fn coverage() -> (NaiveDate, NaiveDate) {
    let t = table();
    (t.first, t.last)
}

/// `day` 是否在内置日历的覆盖范围内。
pub fn covers(day: NaiveDate) -> bool {
    let t = table();
    t.first <= day && day <= t.last
}

/// `day` 是否为覆盖范围内休市的工作日（节假日）。覆盖范围之外恒为 false。
pub fn is_closed_weekday(day: NaiveDate) -> bool {
    table().closed.binary_search(&day).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, Weekday};

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn table_is_well_formed() {
        let t = table();
        assert!(t.first < t.last);
        assert!(t.closed.len() > 500);
        assert!(t.closed.windows(2).all(|w| w[0] < w[1]));
        // 表里只有覆盖范围内的工作日
        for day in &t.closed {
            assert!(covers(*day), "{day}");
            assert!(
                !matches!(day.weekday(), Weekday::Sat | Weekday::Sun),
                "{day}"
            );
        }
    }

    #[test]
    fn known_holidays_and_trading_days() {
        // 2024 国庆 10-01～10-07 休市，10-08 复市；春节 02-09（除夕，周五）休市
        for day in [1, 2, 3, 4, 7] {
            assert!(is_closed_weekday(d(2024, 10, day)));
        }
        assert!(!is_closed_weekday(d(2024, 10, 8)));
        assert!(is_closed_weekday(d(2024, 2, 9)));
        assert!(!is_closed_weekday(d(2024, 2, 8)));
        // 覆盖范围之外不判休市
        let (first, last) = coverage();
        assert!(!covers(first.pred_opt().unwrap()));
        assert!(!covers(last.succ_opt().unwrap()));
        assert!(!is_closed_weekday(last.succ_opt().unwrap()));
    }

    #[test]
    fn parse_reads_coverage_and_skips_comments() {
        let t = parse("# 注释\n\ncoverage 2024-01-01 2024-12-31\n2024-10-02\n2024-10-01\n");
        assert_eq!((t.first, t.last), (d(2024, 1, 1), d(2024, 12, 31)));
        assert_eq!(t.closed, vec![d(2024, 10, 1), d(2024, 10, 2)]);
    }
}
