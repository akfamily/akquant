//! 把时间点或 bar 区间解析为 (交易日, 时段)。

use super::clock::LocalClock;
use super::schedule::{Anchor, FallbackCalendar, SessionSchedule, template};
use super::trading_calendar::{CalendarError, TradingCalendar};
use crate::model::{AssetType, TradingSession};
use chrono::{Duration, NaiveDate};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// 解析结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    pub trading_day: NaiveDate,
    pub session: TradingSession,
}

/// 一个交易日展开成绝对时间（UTC 纳秒）后的时段。
#[derive(Debug)]
struct DayWindow {
    start: i64,
    end: i64,
    segments: Vec<(i64, i64, TradingSession)>,
}

/// 交易日查找的最多迭代次数（按交易日计；正常 1-3 次即可命中）。
const MAX_DAY_STEPS: usize = 8;
/// 跨交易日的 bar（周线等）最多覆盖的交易日数。
const MAX_SPAN_DAYS: usize = 400;

fn rank(s: TradingSession) -> u8 {
    match s {
        TradingSession::Continuous => 7,
        TradingSession::ClosingAuction => 6,
        TradingSession::OpeningAuction => 5,
        TradingSession::AuctionHold => 4,
        TradingSession::Break => 3,
        TradingSession::PostClose => 2,
        // 旧值阶段 6 删除；解析器不会产出它们
        TradingSession::CallAuction | TradingSession::PreOpen => 1,
        TradingSession::Closed => 0,
    }
}

pub struct SessionResolver {
    clock: LocalClock,
    calendar: Option<TradingCalendar>,
    weekdays: TradingCalendar,
    all_days: TradingCalendar,
    explicit_default: Option<Arc<SessionSchedule>>,
    /// (小写前缀, 时段表)，按前缀长度降序。
    by_prefix: Vec<(String, Arc<SessionSchedule>)>,
    stock_default: Arc<SessionSchedule>,
    futures_default: Arc<SessionSchedule>,
    option_default: Arc<SessionSchedule>,
    always_open: Arc<SessionSchedule>,
    windows: Mutex<HashMap<(usize, NaiveDate), Arc<DayWindow>>>,
}

impl SessionResolver {
    pub fn new(
        clock: LocalClock,
        calendar: Option<TradingCalendar>,
        sessions: Vec<(String, SessionSchedule)>,
    ) -> Result<Self, String> {
        let builtin = |n: &str| Arc::new(template(n).expect("内置模板"));
        let mut explicit_default = None;
        let mut by_prefix = Vec::new();
        for (key, schedule) in sessions {
            let k = key.trim().to_ascii_lowercase();
            if k.is_empty() {
                return Err("sessions 的键不能为空字符串".to_string());
            }
            if k == "default" {
                explicit_default = Some(Arc::new(schedule));
            } else {
                by_prefix.push((k, Arc::new(schedule)));
            }
        }
        by_prefix.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
        Ok(Self {
            clock,
            calendar,
            weekdays: TradingCalendar::Weekdays,
            all_days: TradingCalendar::AllDays,
            explicit_default,
            by_prefix,
            stock_default: builtin("CN_STOCK"),
            futures_default: builtin("CN_FUTURES_DAY"),
            option_default: builtin("CN_OPTION"),
            always_open: builtin("ALWAYS_OPEN"),
            windows: Mutex::new(HashMap::new()),
        })
    }

    pub fn uses_fallback_calendar(&self) -> bool {
        self.calendar.is_none()
    }

    pub fn schedule_for(
        &self,
        symbol: &str,
        asset_type: Option<AssetType>,
    ) -> Arc<SessionSchedule> {
        let lower = symbol.trim().to_ascii_lowercase();
        if let Some((_, s)) = self
            .by_prefix
            .iter()
            .find(|(p, _)| lower.starts_with(p.as_str()))
        {
            return s.clone();
        }
        if let Some(s) = &self.explicit_default {
            return s.clone();
        }
        match asset_type {
            Some(AssetType::Stock) | Some(AssetType::Fund) => self.stock_default.clone(),
            Some(AssetType::Futures) => self.futures_default.clone(),
            Some(AssetType::Option) => self.option_default.clone(),
            _ => self.always_open.clone(),
        }
    }

    fn calendar_for(&self, schedule: &SessionSchedule) -> &TradingCalendar {
        match &self.calendar {
            Some(c) => c,
            None => match schedule.fallback_calendar {
                FallbackCalendar::Weekdays => &self.weekdays,
                FallbackCalendar::AllDays => &self.all_days,
            },
        }
    }

    fn local_ns(&self, date: NaiveDate, secs: u32) -> i64 {
        let midnight = date.and_hms_opt(0, 0, 0).expect("0 点合法");
        self.clock
            .from_local(midnight + Duration::seconds(i64::from(secs)))
    }

    fn day_window(
        &self,
        schedule: &Arc<SessionSchedule>,
        day: NaiveDate,
    ) -> Result<Arc<DayWindow>, CalendarError> {
        // 按时段表实例（Arc 指针）缓存：name 由调用方指定，可能重名。
        let key = (Arc::as_ptr(schedule) as usize, day);
        if let Some(w) = self.windows.lock().expect("缓存锁").get(&key) {
            return Ok(w.clone());
        }
        let cal = self.calendar_for(schedule);
        let prev = if schedule.has_night() {
            Some(cal.before(day)?)
        } else {
            None
        };
        let mut segments = Vec::with_capacity(schedule.segments.len());
        for seg in &schedule.segments {
            let anchor_date = match seg.anchor {
                Anchor::PrevTradingDay => prev.expect("has_night 时已求出"),
                Anchor::TradingDay => day,
            };
            segments.push((
                self.local_ns(anchor_date, seg.start_sec),
                self.local_ns(anchor_date, seg.end_sec),
                seg.session,
            ));
        }
        segments.sort_by_key(|s| s.0);
        let window = Arc::new(DayWindow {
            start: segments.first().expect("非空").0,
            end: segments.iter().map(|s| s.1).max().expect("非空"),
            segments,
        });
        self.windows
            .lock()
            .expect("缓存锁")
            .insert(key, window.clone());
        Ok(window)
    }

    /// 交易日 T 的结束时刻。只用锚定在 T 当天的时段计算，不需要前一交易日：
    /// 交易日总以日盘收盘结束（夜盘段锚定前一交易日，结束得更早）。这样查找
    /// 时经过日历首日也不会因为“首日的前一交易日不在日历里”而报错。
    /// 全部时段都锚定前一交易日（不现实的自定义表）时退回完整窗口。
    fn day_end(
        &self,
        schedule: &Arc<SessionSchedule>,
        day: NaiveDate,
    ) -> Result<i64, CalendarError> {
        let same_day_end = schedule
            .segments
            .iter()
            .filter(|s| s.anchor == Anchor::TradingDay)
            .map(|s| s.end_sec)
            .max();
        match same_day_end {
            Some(end_sec) => Ok(self.local_ns(day, end_sec)),
            None => Ok(self.day_window(schedule, day)?.end),
        }
    }

    /// 找出时间点 ts 所属交易日：第一个满足 ts <= day_end(T) 的交易日。
    fn trading_day_of(
        &self,
        schedule: &Arc<SessionSchedule>,
        ts: i64,
    ) -> Result<(NaiveDate, Arc<DayWindow>), CalendarError> {
        let cal = self.calendar_for(schedule);
        let local_date = self.clock.to_local(ts).date();
        let mut search_from = local_date - Duration::days(1);
        if let Some(first) = cal.first_day() {
            if search_from < first && local_date >= first {
                search_from = first;
            }
        }
        let mut day = cal.on_or_after(search_from)?;
        for _ in 0..MAX_DAY_STEPS {
            if ts <= self.day_end(schedule, day)? {
                return Ok((day, self.day_window(schedule, day)?));
            }
            day = cal.after(day)?;
        }
        // 不变量：搜索从 on_or_after(local_date - 1) 起；local_date 之后的任何交易日，
        // 其 day_end 不小于当天本地 0 点，必大于 ts，故最多 3 步命中（全部时段都锚定
        // 前一交易日时 4 步）。
        unreachable!(
            "交易日查找超过 {MAX_DAY_STEPS} 步：时段表 {}",
            schedule.name
        )
    }

    fn session_at(w: &DayWindow, ts: i64) -> TradingSession {
        if ts < w.start {
            return TradingSession::Closed;
        }
        if let Some(s) = w.segments.iter().find(|s| s.0 <= ts && ts < s.1) {
            return s.2;
        }
        // 恰好等于某段结束、且没有段从该时刻开始：归入该段（收盘那一笔）。
        if !w.segments.iter().any(|s| s.0 == ts) {
            if let Some(s) = w.segments.iter().find(|s| s.1 == ts) {
                return s.2;
            }
        }
        TradingSession::Closed
    }

    pub fn resolve_instant(
        &self,
        symbol: &str,
        asset_type: Option<AssetType>,
        ts: i64,
    ) -> Result<Resolution, CalendarError> {
        let schedule = self.schedule_for(symbol, asset_type);
        let (trading_day, w) = self.trading_day_of(&schedule, ts)?;
        Ok(Resolution {
            trading_day,
            session: Self::session_at(&w, ts),
        })
    }

    pub fn resolve_bar(
        &self,
        symbol: &str,
        asset_type: Option<AssetType>,
        interval_start: Option<i64>,
        end: i64,
    ) -> Result<Resolution, CalendarError> {
        let point = end - 1;
        let start = match interval_start {
            Some(s) if s < end => s,
            _ => return self.resolve_instant(symbol, asset_type, point),
        };
        let schedule = self.schedule_for(symbol, asset_type);
        let (trading_day, _) = self.trading_day_of(&schedule, point)?;
        // 区间左开：从 start + 1 所属交易日扫起；它落在日历外时只扫 bar 自己的交易日。
        let mut day = match self.trading_day_of(&schedule, start + 1) {
            Ok((d, _)) => d,
            Err(_) => trading_day,
        };
        let cal = self.calendar_for(&schedule);
        let mut best = TradingSession::Closed;
        for _ in 0..MAX_SPAN_DAYS {
            let w = self.day_window(&schedule, day)?;
            for &(s, e, sess) in &w.segments {
                // 时段 [s, e) 与 bar 区间 (start, end] 有正长度交集
                if s.max(start) < e.min(end) && rank(sess) > rank(best) {
                    best = sess;
                }
            }
            if day >= trading_day {
                break;
            }
            day = cal.after(day)?;
        }
        Ok(Resolution {
            trading_day,
            session: best,
        })
    }

    pub fn day_bounds(
        &self,
        symbol: &str,
        asset_type: Option<AssetType>,
        trading_day: NaiveDate,
    ) -> Result<(i64, i64), CalendarError> {
        let schedule = self.schedule_for(symbol, asset_type);
        let w = self.day_window(&schedule, trading_day)?;
        Ok((w.start, w.end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendar::schedule::template;

    const NS: i64 = 1_000_000_000;

    fn sh() -> LocalClock {
        LocalClock::new(Some("Asia/Shanghai"), 8 * 3600)
    }

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    /// 上海本地时间 → UTC 纳秒。
    fn at(y: i32, m: u32, day: u32, h: u32, mi: u32, s: u32) -> i64 {
        sh().from_local(d(y, m, day).and_hms_opt(h, mi, s).unwrap())
    }

    fn resolver(sessions: Vec<(&str, &str)>, calendar: Option<TradingCalendar>) -> SessionResolver {
        let sessions = sessions
            .into_iter()
            .map(|(k, t)| (k.to_string(), template(t).unwrap()))
            .collect();
        SessionResolver::new(sh(), calendar, sessions).unwrap()
    }

    fn stock() -> Option<AssetType> {
        Some(AssetType::Stock)
    }

    #[test]
    fn stock_intraday_instants() {
        let r = resolver(vec![], None);
        let cases = [
            ((9, 14, 0), TradingSession::Closed),
            ((9, 15, 0), TradingSession::OpeningAuction),
            ((9, 25, 0), TradingSession::AuctionHold),
            ((9, 30, 0), TradingSession::Continuous),
            ((11, 30, 0), TradingSession::Break),
            ((13, 0, 0), TradingSession::Continuous),
            ((14, 57, 0), TradingSession::ClosingAuction),
            ((15, 0, 0), TradingSession::ClosingAuction), // 收盘那一笔归入最后一段
        ];
        for ((h, m, s), want) in cases {
            let got = r
                .resolve_instant("600000.SH", stock(), at(2024, 1, 2, h, m, s))
                .unwrap();
            assert_eq!(got.session, want, "{h}:{m}:{s}");
            assert_eq!(got.trading_day, d(2024, 1, 2));
        }
        // 15:00:01 属于下一个交易日的闭市间隙
        let after = r
            .resolve_instant("600000.SH", stock(), at(2024, 1, 2, 15, 0, 1))
            .unwrap();
        assert_eq!(
            after,
            Resolution {
                trading_day: d(2024, 1, 3),
                session: TradingSession::Closed
            }
        );
    }

    #[test]
    fn end_stamped_minute_bars_without_interval() {
        let r = resolver(vec![], None);
        // 时间戳=区间终点，未知区间时按终点前 1ns 判断
        let bar = |h, m| {
            r.resolve_bar("600000.SH", stock(), None, at(2024, 1, 2, h, m, 0))
                .unwrap()
                .session
        };
        assert_eq!(bar(9, 31), TradingSession::Continuous);
        assert_eq!(bar(9, 30), TradingSession::AuctionHold);
        assert_eq!(bar(11, 30), TradingSession::Continuous);
        assert_eq!(bar(15, 0), TradingSession::ClosingAuction);
    }

    #[test]
    fn daily_bar_interval_is_continuous() {
        let r = resolver(vec![], None);
        let (start, end) = r.day_bounds("600000.SH", stock(), d(2024, 1, 2)).unwrap();
        assert_eq!(start, at(2024, 1, 2, 9, 15, 0));
        assert_eq!(end, at(2024, 1, 2, 15, 0, 0));
        let got = r
            .resolve_bar("600000.SH", stock(), Some(start), end)
            .unwrap();
        assert_eq!(
            got,
            Resolution {
                trading_day: d(2024, 1, 2),
                session: TradingSession::Continuous
            }
        );
    }

    #[test]
    fn auction_only_bar_reports_auction() {
        let r = resolver(vec![], None);
        // 9:15-9:25 的竞价 bar
        let got = r
            .resolve_bar(
                "600000.SH",
                stock(),
                Some(at(2024, 1, 2, 9, 15, 0)),
                at(2024, 1, 2, 9, 25, 0),
            )
            .unwrap();
        assert_eq!(got.session, TradingSession::OpeningAuction);
    }

    #[test]
    fn friday_night_belongs_to_monday() {
        let r = resolver(vec![("rb", "CN_FUTURES_NIGHT_23")], None);
        let fut = Some(AssetType::Futures);
        // 2024-01-05 周五 22:00 → 交易日 2024-01-08 周一
        let got = r
            .resolve_instant("rb2405", fut, at(2024, 1, 5, 22, 0, 0))
            .unwrap();
        assert_eq!(
            got,
            Resolution {
                trading_day: d(2024, 1, 8),
                session: TradingSession::Continuous
            }
        );
        // 周五 15:00:01（日盘收盘后、夜盘前）→ 周一的闭市间隙
        let gap = r
            .resolve_instant("rb2405", fut, at(2024, 1, 5, 15, 0, 1))
            .unwrap();
        assert_eq!(
            gap,
            Resolution {
                trading_day: d(2024, 1, 8),
                session: TradingSession::Closed
            }
        );
        // 夜盘集合竞价
        let auction = r
            .resolve_instant("rb2405", fut, at(2024, 1, 5, 20, 56, 0))
            .unwrap();
        assert_eq!(auction.session, TradingSession::OpeningAuction);
        // 周六 01:00 → 夜盘已结束，仍属周一的闭市间隙
        let sat = r
            .resolve_instant("rb2405", fut, at(2024, 1, 6, 1, 0, 0))
            .unwrap();
        assert_eq!(
            sat,
            Resolution {
                trading_day: d(2024, 1, 8),
                session: TradingSession::Closed
            }
        );
    }

    #[test]
    fn cross_midnight_night_session() {
        let r = resolver(vec![("au", "CN_FUTURES_NIGHT_0230")], None);
        let fut = Some(AssetType::Futures);
        // 周二 01:30 属于周二交易日的夜盘（锚定周一晚）
        let got = r
            .resolve_instant("au2406", fut, at(2024, 1, 2, 1, 30, 0))
            .unwrap();
        assert_eq!(
            got,
            Resolution {
                trading_day: d(2024, 1, 2),
                session: TradingSession::Continuous
            }
        );
        // 23:59:30 不再有缺口
        let late = r
            .resolve_instant("au2406", fut, at(2024, 1, 1, 23, 59, 30))
            .unwrap();
        assert_eq!(late.session, TradingSession::Continuous);
    }

    #[test]
    fn holiday_calendar_moves_night_to_next_session() {
        // 春节：2024-02-08 周四后休市，2024-02-19 周一复市
        let cal = TradingCalendar::explicit([
            d(2024, 2, 7),
            d(2024, 2, 8),
            d(2024, 2, 19),
            d(2024, 2, 20),
        ])
        .unwrap();
        let r = resolver(vec![("rb", "CN_FUTURES_NIGHT_23")], Some(cal));
        let got = r
            .resolve_instant(
                "rb2405",
                Some(AssetType::Futures),
                at(2024, 2, 8, 21, 30, 0),
            )
            .unwrap();
        assert_eq!(got.trading_day, d(2024, 2, 19));
    }

    #[test]
    fn explicit_calendar_out_of_range_errors() {
        let cal = TradingCalendar::explicit([d(2024, 1, 2), d(2024, 1, 3)]).unwrap();
        let r = resolver(vec![], Some(cal));
        assert!(
            r.resolve_instant("600000.SH", stock(), at(2024, 1, 10, 10, 0, 0))
                .is_err()
        );
    }

    #[test]
    fn prefix_and_asset_defaults() {
        let r = resolver(
            vec![
                ("IF", "CN_FUTURES_CFFEX_STOCK_INDEX_DAY"),
                ("i", "CN_FUTURES_NIGHT_23"),
            ],
            None,
        );
        assert_eq!(
            r.schedule_for("IF2401", Some(AssetType::Futures)).name,
            "CN_FUTURES_CFFEX_STOCK_INDEX_DAY"
        );
        assert_eq!(
            r.schedule_for("i2405", Some(AssetType::Futures)).name,
            "CN_FUTURES_NIGHT_23"
        );
        assert_eq!(r.schedule_for("600000.SH", stock()).name, "CN_STOCK");
        assert_eq!(
            r.schedule_for("ag2406", Some(AssetType::Futures)).name,
            "CN_FUTURES_DAY"
        );
        assert_eq!(r.schedule_for("BTCUSDT", None).name, "ALWAYS_OPEN");
        let with_default = resolver(vec![("default", "ALWAYS_OPEN")], None);
        assert_eq!(
            with_default.schedule_for("600000.SH", stock()).name,
            "ALWAYS_OPEN"
        );
    }

    #[test]
    fn always_open_counts_weekends() {
        let r = resolver(vec![], None);
        let got = r
            .resolve_instant("BTCUSDT", None, at(2024, 1, 6, 12, 0, 0))
            .unwrap();
        assert_eq!(
            got,
            Resolution {
                trading_day: d(2024, 1, 6),
                session: TradingSession::Continuous
            }
        );
    }

    #[test]
    fn same_name_schedules_do_not_share_cache() {
        let stock_segs = template("CN_STOCK").unwrap().segments;
        let open_segs = template("ALWAYS_OPEN").unwrap().segments;
        let a = SessionSchedule::new("X", stock_segs, FallbackCalendar::Weekdays).unwrap();
        let b = SessionSchedule::new("X", open_segs, FallbackCalendar::AllDays).unwrap();
        let r = SessionResolver::new(
            sh(),
            None,
            vec![("aaa".to_string(), a), ("bbb".to_string(), b)],
        )
        .unwrap();
        // 周一凌晨 3 点：两张表都归交易日 2024-01-08，缓存键只差表实例
        let ts = at(2024, 1, 8, 3, 0, 0);
        let ra = r.resolve_instant("aaa1", None, ts).unwrap();
        let rb = r.resolve_instant("bbb1", None, ts).unwrap();
        assert_eq!(ra.trading_day, d(2024, 1, 8));
        assert_eq!(ra.session, TradingSession::Closed);
        assert_eq!(rb.trading_day, d(2024, 1, 8));
        assert_eq!(rb.session, TradingSession::Continuous);
    }

    #[test]
    fn bar_interval_starting_before_calendar_night() {
        let cal = TradingCalendar::explicit([d(2024, 1, 2), d(2024, 1, 3), d(2024, 1, 4)]).unwrap();
        let r = resolver(vec![("rb", "CN_FUTURES_NIGHT_23")], Some(cal));
        let got = r
            .resolve_bar(
                "rb2405",
                Some(AssetType::Futures),
                Some(at(2024, 1, 2, 15, 0, 0)),
                at(2024, 1, 3, 15, 0, 0),
            )
            .unwrap();
        assert_eq!(
            got,
            Resolution {
                trading_day: d(2024, 1, 3),
                session: TradingSession::Continuous
            }
        );
    }

    #[test]
    fn bar_interval_starting_before_calendar_stock() {
        let cal = TradingCalendar::explicit([d(2024, 1, 2), d(2024, 1, 3)]).unwrap();
        let r = resolver(vec![], Some(cal));
        let got = r
            .resolve_bar(
                "600000.SH",
                stock(),
                Some(at(2024, 1, 1, 15, 0, 0)),
                at(2024, 1, 2, 15, 0, 0),
            )
            .unwrap();
        assert_eq!(
            got,
            Resolution {
                trading_day: d(2024, 1, 2),
                session: TradingSession::Continuous
            }
        );
    }

    #[test]
    fn many_lookups_hit_cache() {
        let r = resolver(vec![], None);
        let base = at(2024, 1, 2, 9, 30, 0);
        for i in 0..200_000_i64 {
            let ts = base + (i % 14_400) * NS;
            assert!(r.resolve_instant("600000.SH", stock(), ts).is_ok());
        }
        assert!(r.windows.lock().unwrap().len() <= 3);
    }
}
