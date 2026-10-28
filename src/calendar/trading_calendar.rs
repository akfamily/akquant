//! 交易日历：显式交易日集合，或“工作日 / 每天”的缺省规则。

use chrono::{Datelike, Duration, NaiveDate, Weekday};
use std::collections::BTreeSet;
use std::fmt;

/// 缺省日历（未传 `trading_calendar`）的修正量：按行情源交易日补入的交易日与剔除的
/// 推算交易日。随存档保存（`SessionConfig::calendar_overrides`）；用户显式日历不受影响。
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CalendarOverrides {
    pub added: Vec<NaiveDate>,
    pub removed: Vec<NaiveDate>,
}

impl CalendarOverrides {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }

    /// 补入交易日 `day`（同时从剔除集合中移除）。返回是否有变化。
    pub fn add(&mut self, day: NaiveDate) -> bool {
        let was_removed = self.removed.contains(&day);
        self.removed.retain(|d| *d != day);
        if self.added.contains(&day) {
            return was_removed;
        }
        self.added.push(day);
        self.added.sort();
        true
    }

    /// 剔除交易日 `day`（同时从补入集合中移除）。返回是否有变化。
    pub fn remove(&mut self, day: NaiveDate) -> bool {
        let was_added = self.added.contains(&day);
        self.added.retain(|d| *d != day);
        if self.removed.contains(&day) {
            return was_added;
        }
        self.removed.push(day);
        self.removed.sort();
        true
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TradingCalendar {
    /// 周一至周五为交易日（用户未传日历时的缺省，节假日会算错）。
    Weekdays,
    /// 每个自然日都是交易日。
    AllDays,
    /// 用户传入的交易日集合。
    Explicit(ExplicitDays),
    /// 缺省规则（工作日 / 每天）按行情源交易日修正后的日历。
    Adjusted(AdjustedDays),
}

/// 缺省规则加修正量：`added` 中的日期是交易日，`removed` 中的不是，其余按 `base`。
/// 只能经 [`TradingCalendar::with_overrides`] 构造（`base` 不会是显式日历或嵌套修正）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdjustedDays {
    base: Box<TradingCalendar>,
    added: BTreeSet<NaiveDate>,
    removed: BTreeSet<NaiveDate>,
}

/// 非空的显式交易日集合。字段私有：只能经 `TradingCalendar::explicit()` 构造，
/// 从而保证集合非空（日历内部的 `expect("非空")` 不可达）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplicitDays(BTreeSet<NaiveDate>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalendarError {
    Empty,
    OutOfRange {
        requested: NaiveDate,
        first: NaiveDate,
        last: NaiveDate,
    },
    /// 日历首日的夜盘归属需要首日之前的一个交易日，而日历里没有。
    NightNeedsPreviousDay {
        first: NaiveDate,
    },
    /// bar 区间起点不早于终点。
    InvalidInterval {
        start: i64,
        end: i64,
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
            CalendarError::NightNeedsPreviousDay { first } => write!(
                f,
                "trading_calendar 首日 {first} 的夜盘时段需要其前一个交易日（{}）才能确定归属，\
                 日历里没有；请把 trading_calendar 的起点提前到比数据至少早一个交易日",
                *first - Duration::days(1)
            ),
            CalendarError::InvalidInterval { start, end } => write!(
                f,
                "bar 区间起点 {start} 必须早于终点 {end}（区间为 (起点, 终点]）"
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
        Ok(TradingCalendar::Explicit(ExplicitDays(set)))
    }

    /// 缺省规则（`Weekdays` / `AllDays`）应用修正量；修正量为空或日历为显式日历时
    /// 原样返回（显式日历以用户为准，不修正）。
    pub fn with_overrides(self, overrides: &CalendarOverrides) -> Self {
        if overrides.is_empty() {
            return self;
        }
        match self {
            TradingCalendar::Weekdays | TradingCalendar::AllDays => {
                let added: BTreeSet<NaiveDate> = overrides.added.iter().copied().collect();
                let removed: BTreeSet<NaiveDate> = overrides
                    .removed
                    .iter()
                    .copied()
                    .filter(|d| !added.contains(d))
                    .collect();
                TradingCalendar::Adjusted(AdjustedDays {
                    base: Box::new(self),
                    added,
                    removed,
                })
            }
            other => other,
        }
    }

    pub fn first_day(&self) -> Option<NaiveDate> {
        match self {
            TradingCalendar::Explicit(ExplicitDays(set)) => set.first().copied(),
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
            TradingCalendar::Explicit(ExplicitDays(set)) => set.contains(&d),
            TradingCalendar::Adjusted(adj) => {
                adj.added.contains(&d) || (!adj.removed.contains(&d) && adj.base.is_trading_day(d))
            }
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
            TradingCalendar::Explicit(ExplicitDays(set)) => {
                if d < *set.first().expect("非空") {
                    return Err(Self::bounds_error(set, d));
                }
                set.range(d..)
                    .next()
                    .copied()
                    .ok_or_else(|| Self::bounds_error(set, d))
            }
            // 基础规则每周都有交易日，剔除集合有限，循环必然终止。
            TradingCalendar::Adjusted(_) => {
                let mut x = d;
                while !self.is_trading_day(x) {
                    x += Duration::days(1);
                }
                Ok(x)
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
            TradingCalendar::Explicit(ExplicitDays(set)) => {
                if d > *set.last().expect("非空") {
                    return Err(Self::bounds_error(set, d));
                }
                set.range(..d)
                    .next_back()
                    .copied()
                    .ok_or_else(|| Self::bounds_error(set, d))
            }
            TradingCalendar::Adjusted(_) => {
                let mut x = d - Duration::days(1);
                while !self.is_trading_day(x) {
                    x -= Duration::days(1);
                }
                Ok(x)
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
    fn explicit_before_after_last_is_out_of_range() {
        let c = TradingCalendar::explicit([d(2024, 1, 2), d(2024, 1, 3)]).unwrap();
        // last + 1 天：before 也必须报越界，而不是返回 last
        assert!(matches!(
            c.before(d(2024, 1, 4)),
            Err(CalendarError::OutOfRange { .. })
        ));
        assert_eq!(c.before(d(2024, 1, 3)).unwrap(), d(2024, 1, 2));
    }

    #[test]
    fn explicit_rejects_empty_and_message_mentions_range() {
        assert_eq!(TradingCalendar::explicit([]), Err(CalendarError::Empty));
        let c = TradingCalendar::explicit([d(2024, 1, 2)]).unwrap();
        let msg = c.after(d(2024, 1, 2)).unwrap_err().to_string();
        assert!(msg.contains("2024-01-02 ~ 2024-01-02"), "{msg}");
    }

    #[test]
    fn weekdays_with_overrides_skip_removed_and_include_added() {
        // 2024 国庆：10-01～10-07 的工作日剔除；10-12（周六）调休补班
        let removed: Vec<NaiveDate> = (1..=7)
            .map(|day| d(2024, 10, day))
            .filter(|x| is_weekday(*x))
            .collect();
        let overrides = CalendarOverrides {
            added: vec![d(2024, 10, 12)],
            removed,
        };
        let c = TradingCalendar::Weekdays.with_overrides(&overrides);
        assert_eq!(c.after(d(2024, 9, 30)).unwrap(), d(2024, 10, 8));
        assert_eq!(c.before(d(2024, 10, 8)).unwrap(), d(2024, 9, 30));
        assert_eq!(c.on_or_after(d(2024, 10, 3)).unwrap(), d(2024, 10, 8));
        assert!(!c.is_trading_day(d(2024, 10, 1)));
        assert!(c.is_trading_day(d(2024, 10, 12)));
        assert_eq!(c.after(d(2024, 10, 11)).unwrap(), d(2024, 10, 12));
        assert_eq!(c.before(d(2024, 10, 14)).unwrap(), d(2024, 10, 12));
        // 显式日历不受修正量影响；空修正量原样返回
        let explicit = TradingCalendar::explicit([d(2024, 10, 1)]).unwrap();
        assert_eq!(explicit.clone().with_overrides(&overrides), explicit);
        assert_eq!(
            TradingCalendar::Weekdays.with_overrides(&CalendarOverrides::default()),
            TradingCalendar::Weekdays
        );
    }

    #[test]
    fn calendar_overrides_add_and_remove_stay_disjoint() {
        let mut o = CalendarOverrides::default();
        assert!(o.remove(d(2024, 10, 1)));
        assert!(!o.remove(d(2024, 10, 1)));
        assert!(o.add(d(2024, 10, 1)));
        assert_eq!(o.added, vec![d(2024, 10, 1)]);
        assert!(o.removed.is_empty());
        assert!(!o.add(d(2024, 10, 1)));
    }

    #[test]
    fn all_days_is_every_day() {
        let c = TradingCalendar::AllDays;
        assert_eq!(c.after(d(2024, 1, 5)).unwrap(), d(2024, 1, 6));
        assert!(c.is_trading_day(d(2024, 1, 7)));
    }
}
