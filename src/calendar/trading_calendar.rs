//! 交易日历：显式交易日集合，或“工作日 / 每天”的缺省规则。

use chrono::{Datelike, Duration, NaiveDate, Weekday};
use std::collections::BTreeSet;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TradingCalendar {
    /// 周一至周五为交易日（用户未传日历时的缺省，节假日会算错）。
    Weekdays,
    /// 每个自然日都是交易日。
    AllDays,
    /// 用户传入的交易日集合。
    Explicit(BTreeSet<NaiveDate>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalendarError {
    Empty,
    OutOfRange {
        requested: NaiveDate,
        first: NaiveDate,
        last: NaiveDate,
    },
}

impl fmt::Display for CalendarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalendarError::Empty => write!(f, "trading_calendar 不能为空"),
            CalendarError::OutOfRange {
                requested,
                first,
                last,
            } => write!(
                f,
                "日期 {requested} 超出 trading_calendar 覆盖范围 {first} ~ {last}；\
                 请传入覆盖全部数据日期的交易日历（起点至少比数据早一个交易日，以便确定夜盘归属）"
            ),
        }
    }
}

fn is_weekday(d: NaiveDate) -> bool {
    !matches!(d.weekday(), Weekday::Sat | Weekday::Sun)
}

impl TradingCalendar {
    pub fn explicit<I: IntoIterator<Item = NaiveDate>>(days: I) -> Result<Self, CalendarError> {
        let set: BTreeSet<NaiveDate> = days.into_iter().collect();
        if set.is_empty() {
            return Err(CalendarError::Empty);
        }
        Ok(TradingCalendar::Explicit(set))
    }

    pub fn first_day(&self) -> Option<NaiveDate> {
        match self {
            TradingCalendar::Explicit(set) => set.first().copied(),
            _ => None,
        }
    }

    fn bounds_error(set: &BTreeSet<NaiveDate>, requested: NaiveDate) -> CalendarError {
        CalendarError::OutOfRange {
            requested,
            first: *set.first().expect("非空"),
            last: *set.last().expect("非空"),
        }
    }

    pub fn is_trading_day(&self, d: NaiveDate) -> bool {
        match self {
            TradingCalendar::Weekdays => is_weekday(d),
            TradingCalendar::AllDays => true,
            TradingCalendar::Explicit(set) => set.contains(&d),
        }
    }

    pub fn on_or_after(&self, d: NaiveDate) -> Result<NaiveDate, CalendarError> {
        match self {
            TradingCalendar::Weekdays => {
                let mut x = d;
                while !is_weekday(x) {
                    x += Duration::days(1);
                }
                Ok(x)
            }
            TradingCalendar::AllDays => Ok(d),
            TradingCalendar::Explicit(set) => {
                if d < *set.first().expect("非空") {
                    return Err(Self::bounds_error(set, d));
                }
                set.range(d..)
                    .next()
                    .copied()
                    .ok_or_else(|| Self::bounds_error(set, d))
            }
        }
    }

    pub fn after(&self, d: NaiveDate) -> Result<NaiveDate, CalendarError> {
        self.on_or_after(d + Duration::days(1))
    }

    pub fn before(&self, d: NaiveDate) -> Result<NaiveDate, CalendarError> {
        match self {
            TradingCalendar::Weekdays => {
                let mut x = d - Duration::days(1);
                while !is_weekday(x) {
                    x -= Duration::days(1);
                }
                Ok(x)
            }
            TradingCalendar::AllDays => Ok(d - Duration::days(1)),
            TradingCalendar::Explicit(set) => {
                if d > *set.last().expect("非空") {
                    return Err(Self::bounds_error(set, d));
                }
                set.range(..d)
                    .next_back()
                    .copied()
                    .ok_or_else(|| Self::bounds_error(set, d))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn weekdays_skip_weekend() {
        let c = TradingCalendar::Weekdays;
        // 2024-01-05 周五, 2024-01-08 周一
        assert_eq!(c.after(d(2024, 1, 5)).unwrap(), d(2024, 1, 8));
        assert_eq!(c.before(d(2024, 1, 8)).unwrap(), d(2024, 1, 5));
        assert_eq!(c.on_or_after(d(2024, 1, 6)).unwrap(), d(2024, 1, 8));
        assert!(!c.is_trading_day(d(2024, 1, 7)));
    }

    #[test]
    fn explicit_skips_holiday_and_errors_out_of_range() {
        // 2024-02-08 周四后春节休市, 2024-02-19 周一复市
        let c = TradingCalendar::explicit([d(2024, 2, 7), d(2024, 2, 8), d(2024, 2, 19)]).unwrap();
        assert_eq!(c.after(d(2024, 2, 8)).unwrap(), d(2024, 2, 19));
        assert_eq!(c.before(d(2024, 2, 19)).unwrap(), d(2024, 2, 8));
        assert!(matches!(
            c.on_or_after(d(2024, 2, 1)),
            Err(CalendarError::OutOfRange { .. })
        ));
        assert!(matches!(
            c.after(d(2024, 2, 19)),
            Err(CalendarError::OutOfRange { .. })
        ));
        assert!(matches!(
            c.before(d(2024, 2, 7)),
            Err(CalendarError::OutOfRange { .. })
        ));
        assert_eq!(c.first_day(), Some(d(2024, 2, 7)));
    }

    #[test]
    fn explicit_rejects_empty_and_message_mentions_range() {
        assert_eq!(TradingCalendar::explicit([]), Err(CalendarError::Empty));
        let c = TradingCalendar::explicit([d(2024, 1, 2)]).unwrap();
        let msg = c.after(d(2024, 1, 2)).unwrap_err().to_string();
        assert!(msg.contains("2024-01-02 ~ 2024-01-02"), "{msg}");
    }

    #[test]
    fn all_days_is_every_day() {
        let c = TradingCalendar::AllDays;
        assert_eq!(c.after(d(2024, 1, 5)).unwrap(), d(2024, 1, 6));
        assert!(c.is_trading_day(d(2024, 1, 7)));
    }
}
