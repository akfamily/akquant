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
    /// 锚定前一交易日的（夜盘）时段是否已展开。日历首日没有前一交易日时为 false，
    /// 此时 segments 只含锚定当日的时段。
    night_known: bool,
}

/// 交易日查找的最多迭代次数（按交易日计；正常 1-3 次即可命中）。
const MAX_DAY_STEPS: usize = 8;
/// 跨交易日的 bar（周线等）最多覆盖的交易日数。
const MAX_SPAN_DAYS: usize = 400;

/// 中国时区：按资产类型取默认时段表；其他时区默认全天连续（用户决定的规则 B）。
const CHINA_TZ_NAMES: &[&str] = &[
    "Asia/Shanghai",
    "Asia/Chongqing",
    "Asia/Chungking",
    "Asia/Harbin",
    "PRC",
];

pub fn is_china_timezone(tz_name: Option<&str>, offset_secs: i32) -> bool {
    match tz_name {
        Some(name) => CHINA_TZ_NAMES
            .iter()
            .any(|n| n.eq_ignore_ascii_case(name.trim())),
        None => offset_secs == 8 * 3600,
    }
}

/// 规则 B：非中国时区且未给 "default" 时补 `default = ALWAYS_OPEN`。
pub fn effective_sessions(
    user: Option<Vec<(String, SessionSchedule)>>,
    tz_name: Option<&str>,
    offset_secs: i32,
) -> Vec<(String, SessionSchedule)> {
    let mut sessions = user.unwrap_or_default();
    let has_default = sessions
        .iter()
        .any(|(k, _)| k.trim().eq_ignore_ascii_case("default"));
    if !has_default && !is_china_timezone(tz_name, offset_secs) {
        sessions.push((
            "default".to_string(),
            template("ALWAYS_OPEN").expect("内置模板"),
        ));
    }
    sessions
}

/// 时段优先级（多个时段并存时取最高者）。
pub(crate) fn rank(s: TradingSession) -> u8 {
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
    /// 在用时段表（按 Arc 指针去重）；空表示未调用 set_active。
    active: Vec<Arc<SessionSchedule>>,
    /// 按时刻解析的缓存（见 [`Self::session_at_instant`]）：(时刻, [(时段表指针, 结果)])，
    /// 时刻变化时整体作废。
    instant_cache: Mutex<(i64, Vec<(usize, Option<TradingSession>)>)>,
    /// 按时刻解析实际计算的次数（缓存未命中），供单测验证缓存。
    #[cfg(test)]
    instant_resolves: std::sync::atomic::AtomicUsize,
}

impl SessionResolver {
    pub(crate) fn new(
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
            active: Vec::new(),
            instant_cache: Mutex::new((i64::MIN, Vec::new())),
            #[cfg(test)]
            instant_resolves: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub fn uses_fallback_calendar(&self) -> bool {
        self.calendar.is_none()
    }

    /// 会话日历是否为全日历兜底：未传交易日历，且在用时段表的缺省日历全是 AllDays
    /// （如非中国时区的 ALWAYS_OPEN）。此时每个自然日（含周末）都算交易日。
    pub fn uses_all_days_fallback(&self) -> bool {
        self.calendar.is_none()
            && self
                .active_schedules()
                .iter()
                .all(|s| s.fallback_calendar == FallbackCalendar::AllDays)
    }

    pub fn set_active(&mut self, symbols: &[(String, Option<AssetType>)]) {
        let mut active: Vec<Arc<SessionSchedule>> = Vec::new();
        for (symbol, asset_type) in symbols {
            let s = self.schedule_for(symbol, *asset_type);
            if !active.iter().any(|a| Arc::ptr_eq(a, &s)) {
                active.push(s);
            }
        }
        self.active = active;
    }

    fn active_schedules(&self) -> Vec<Arc<SessionSchedule>> {
        if !self.active.is_empty() {
            return self.active.clone();
        }
        vec![
            self.explicit_default
                .clone()
                .unwrap_or_else(|| self.stock_default.clone()),
        ]
    }

    /// 会话日历：用户日历；否则在用表全为 AllDays 缺省时用 AllDays，否则 Weekdays。
    fn session_calendar(&self) -> &TradingCalendar {
        if let Some(c) = &self.calendar {
            return c;
        }
        if self.uses_all_days_fallback() {
            &self.all_days
        } else {
            &self.weekdays
        }
    }

    pub fn session_day_end(&self, day: NaiveDate) -> Result<i64, CalendarError> {
        let mut end = i64::MIN;
        for s in self.active_schedules() {
            end = end.max(self.day_end(&s, day)?);
        }
        Ok(end)
    }

    /// `day` 是否为会话日历上的交易日。
    pub fn is_session_day(&self, day: NaiveDate) -> bool {
        self.session_calendar().is_trading_day(day)
    }

    /// 会话日历上 `day` 之后的下一个交易日。
    pub fn next_session_day(&self, day: NaiveDate) -> Result<NaiveDate, CalendarError> {
        self.session_calendar().after(day)
    }

    pub fn session_day_span(&self, day: NaiveDate) -> Result<(i64, i64), CalendarError> {
        let prev = self.session_calendar().before(day)?;
        Ok((self.session_day_end(prev)?, self.session_day_end(day)?))
    }

    /// 会话日历上 `day` 之前的上一个交易日。
    pub fn session_calendar_before(&self, day: NaiveDate) -> Result<NaiveDate, CalendarError> {
        self.session_calendar().before(day)
    }

    /// 交易日 `day` 在用时段表中的最早时段起点（UTC 纳秒，含已展开的夜盘段）。
    ///
    /// 日历首日的夜盘未展开（缺前一交易日）时只看锚定当日的时段。
    pub fn session_day_first_start(&self, day: NaiveDate) -> Result<i64, CalendarError> {
        let mut best = i64::MAX;
        for s in self.active_schedules() {
            let w = self.day_window(&s, day)?;
            if let Some(start) = w.segments.iter().map(|x| x.0).min() {
                best = best.min(start);
            }
        }
        Ok(best)
    }

    /// 交易日 `day` 在用时段表中最早的连续竞价段起点；没有连续竞价段时为 `None`。
    pub fn session_day_first_continuous_start(
        &self,
        day: NaiveDate,
    ) -> Result<Option<i64>, CalendarError> {
        let mut best: Option<i64> = None;
        for s in self.active_schedules() {
            let w = self.day_window(&s, day)?;
            if let Some(start) = w
                .segments
                .iter()
                .filter(|x| x.2 == TradingSession::Continuous)
                .map(|x| x.0)
                .min()
            {
                best = Some(best.map_or(start, |b| b.min(start)));
            }
        }
        Ok(best)
    }

    /// ts 所属的会话交易日：第一个满足 ts <= session_day_end(T) 的交易日。
    pub fn session_trading_day(&self, ts: i64) -> Result<NaiveDate, CalendarError> {
        let cal = self.session_calendar();
        let local_date = self.clock.to_local(ts).date();
        let mut search_from = local_date - Duration::days(1);
        if let Some(first) = cal.first_day()
            && search_from < first
            && local_date >= first
        {
            search_from = first;
        }
        let mut day = cal.on_or_after(search_from)?;
        // 不变量同 trading_day_of：最多 3~4 步。
        for _ in 0..MAX_DAY_STEPS {
            if ts <= self.session_day_end(day)? {
                return Ok(day);
            }
            day = cal.after(day)?;
        }
        unreachable!("会话交易日查找超过 {MAX_DAY_STEPS} 步")
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
            match cal.before(day) {
                Ok(p) => Some(p),
                // 日历首日没有前一交易日：只展开日盘段，夜盘归属未知
                Err(_) if cal.first_day() == Some(day) => None,
                Err(e) => return Err(e),
            }
        } else {
            None
        };
        let night_known = prev.is_some() || !schedule.has_night();
        let mut segments = Vec::with_capacity(schedule.segments.len());
        for seg in &schedule.segments {
            let anchor_date = match seg.anchor {
                Anchor::PrevTradingDay => match prev {
                    Some(p) => p,
                    None => continue,
                },
                Anchor::TradingDay => day,
            };
            segments.push((
                self.local_ns(anchor_date, seg.start_sec),
                self.local_ns(anchor_date, seg.end_sec),
                seg.session,
            ));
        }
        if segments.is_empty() {
            return Err(CalendarError::NightNeedsPreviousDay { first: day });
        }
        segments.sort_by_key(|s| s.0);
        let window = Arc::new(DayWindow {
            start: segments.first().expect("非空").0,
            end: segments.iter().map(|s| s.1).max().expect("非空"),
            segments,
            night_known,
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
        let mut day = match cal.on_or_after(search_from) {
            Ok(d) => d,
            // 夜盘表在日历首日前一天的数据：落在首日的夜盘里，缺前一交易日无法判断
            Err(_)
                if schedule.has_night()
                    && cal.first_day() == Some(local_date + Duration::days(1)) =>
            {
                return Err(CalendarError::NightNeedsPreviousDay {
                    first: local_date + Duration::days(1),
                });
            }
            Err(e) => return Err(e),
        };
        for _ in 0..MAX_DAY_STEPS {
            if ts <= self.day_end(schedule, day)? {
                let w = self.day_window(schedule, day)?;
                // 首日窗口缺夜盘段：早于日盘首段的时间点可能属于未知的夜盘
                if !w.night_known && ts < w.start {
                    return Err(CalendarError::NightNeedsPreviousDay { first: day });
                }
                return Ok((day, w));
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

    /// 品种在时间点 `ts` 的时段（同 [`Self::resolve_instant`]，只取时段），解析出错
    /// 返回 `None`。结果只取决于（时段表, 时刻），按此缓存：定时器清空时间片后，
    /// 同一时刻同一时段表上的多笔订单只解析一次。
    pub fn session_at_instant(
        &self,
        symbol: &str,
        asset_type: Option<AssetType>,
        ts: i64,
    ) -> Option<TradingSession> {
        let schedule = self.schedule_for(symbol, asset_type);
        let key = Arc::as_ptr(&schedule) as usize;
        let mut cache = self.instant_cache.lock().expect("缓存锁");
        if cache.0 != ts {
            cache.0 = ts;
            cache.1.clear();
        } else if let Some((_, hit)) = cache.1.iter().find(|(k, _)| *k == key) {
            return *hit;
        }
        #[cfg(test)]
        self.instant_resolves
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let session = self
            .trading_day_of(&schedule, ts)
            .ok()
            .map(|(_, w)| Self::session_at(&w, ts));
        cache.1.push((key, session));
        session
    }

    pub fn resolve_bar(
        &self,
        symbol: &str,
        asset_type: Option<AssetType>,
        interval_start: Option<i64>,
        end: i64,
    ) -> Result<Resolution, CalendarError> {
        let point = end.saturating_sub(1);
        let start = match interval_start {
            Some(s) if s >= end => return Err(CalendarError::InvalidInterval { start: s, end }),
            Some(s) => s,
            None => return self.resolve_instant(symbol, asset_type, point),
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

    /// 包含时刻 `instant` 的时段区间 `[start, end)` 与其时段；落在两段之间
    /// （午休以外的间隙、收盘后到下一交易日开盘前）时返回间隙与 `Closed`。
    ///
    /// 区间按半开处理：`instant` 恰为某段结束时返回其后的间隙（与 `session_at`
    /// 把收盘那一笔归入最后一段不同）。调用方须传严格落在所求区间内的时刻——
    /// 终点戳事件（bar、收盘 tick）传 `ts - 1`。
    pub fn segment_span_at(
        &self,
        symbol: &str,
        asset_type: Option<AssetType>,
        instant: i64,
    ) -> Result<(i64, i64, TradingSession), CalendarError> {
        let schedule = self.schedule_for(symbol, asset_type);
        let (day, w) = self.trading_day_of(&schedule, instant)?;
        if let Some(&(s, e, sess)) = w.segments.iter().find(|s| s.0 <= instant && instant < s.1) {
            return Ok((s, e, sess));
        }
        // 间隙：左界 = 不晚于 instant 的最近段结束（没有则取前一交易日结束），
        // 右界 = 晚于 instant 的最近段开始（没有则取下一交易日的第一段开始）。
        let cal = self.calendar_for(&schedule);
        let left = match w
            .segments
            .iter()
            .map(|s| s.1)
            .filter(|&e| e <= instant)
            .max()
        {
            Some(e) => e,
            None => match cal.before(day) {
                Ok(prev) => self.day_end(&schedule, prev)?,
                // 日历首日之前没有交易日：间隙左界无界
                Err(_) => i64::MIN,
            },
        };
        let right = match w
            .segments
            .iter()
            .map(|s| s.0)
            .filter(|&s| s > instant)
            .min()
        {
            Some(s) => s,
            None => {
                let next = cal.after(day)?;
                self.day_window(&schedule, next)?.start
            }
        };
        Ok((left, right, TradingSession::Closed))
    }

    /// 会话级时段：在用时段表中优先级最高者（见 `rank`），按“`ts` 前 1ns”判断。
    /// 用于不属于任何品种的事件（定时器）；任一表解析出错按 `Closed` 计。
    pub fn session_level_instant(&self, ts: i64) -> TradingSession {
        let point = ts.saturating_sub(1);
        let mut best = TradingSession::Closed;
        for s in self.active_schedules() {
            // 解析出错（时刻超出交易日历等）按连续竞价，与撮合的回退
            // （SessionView::session_for）一致：日历缺口不应阻断撮合，数据层的
            // 闭市/日历告警已提示用户。
            let sess = match self.trading_day_of(&s, point) {
                Ok((_, w)) => Self::session_at(&w, point),
                Err(_) => TradingSession::Continuous,
            };
            if rank(sess) > rank(best) {
                best = sess;
            }
        }
        best
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

    fn first_day_night_resolver() -> SessionResolver {
        let cal = TradingCalendar::explicit([d(2024, 1, 2), d(2024, 1, 3), d(2024, 1, 4)]).unwrap();
        resolver(vec![("rb", "CN_FUTURES_NIGHT_23")], Some(cal))
    }

    #[test]
    fn night_schedule_first_day_daytime_resolves() {
        let r = first_day_night_resolver();
        let fut = Some(AssetType::Futures);
        let got = r
            .resolve_instant("rb2405", fut, at(2024, 1, 2, 10, 0, 0))
            .unwrap();
        assert_eq!(
            got,
            Resolution {
                trading_day: d(2024, 1, 2),
                session: TradingSession::Continuous
            }
        );
        // 首日 day_bounds：只含日盘段（模板里第一个锚定当日的段起点 ~ 15:00）
        let first_same_day = template("CN_FUTURES_NIGHT_23")
            .unwrap()
            .segments
            .iter()
            .filter(|s| s.anchor == Anchor::TradingDay)
            .map(|s| s.start_sec)
            .min()
            .unwrap();
        let (start, end) = r.day_bounds("rb2405", fut, d(2024, 1, 2)).unwrap();
        assert_eq!(start, r.local_ns(d(2024, 1, 2), first_same_day));
        assert_eq!(end, at(2024, 1, 2, 15, 0, 0));
        // 第二天夜盘有前一交易日（首日），正常解析
        let night = r
            .resolve_instant("rb2405", fut, at(2024, 1, 2, 22, 0, 0))
            .unwrap();
        assert_eq!(night.trading_day, d(2024, 1, 3));
    }

    #[test]
    fn night_schedule_first_day_night_needs_previous_day() {
        let r = first_day_night_resolver();
        let fut = Some(AssetType::Futures);
        // 日历首日前一晚（属首日交易日的夜盘）：缺前一交易日
        let err = r
            .resolve_instant("rb2405", fut, at(2024, 1, 1, 22, 0, 0))
            .unwrap_err();
        assert_eq!(
            err,
            CalendarError::NightNeedsPreviousDay {
                first: d(2024, 1, 2)
            }
        );
        assert!(err.to_string().contains("2024-01-01"), "{err}");
        // 首日凌晨早于日盘首段，同样无法确定
        let err = r
            .resolve_instant("rb2405", fut, at(2024, 1, 2, 3, 0, 0))
            .unwrap_err();
        assert!(matches!(err, CalendarError::NightNeedsPreviousDay { .. }));
    }

    #[test]
    fn resolve_bar_rejects_non_positive_interval() {
        let r = resolver(vec![], None);
        let end = at(2024, 1, 2, 10, 0, 0);
        for s in [end, end + 1] {
            assert_eq!(
                r.resolve_bar("600000.SH", stock(), Some(s), end),
                Err(CalendarError::InvalidInterval { start: s, end })
            );
        }
        // i64::MIN 终点不得溢出 panic
        let _ = r.resolve_bar("BTCUSDT", None, None, i64::MIN);
    }

    #[test]
    fn always_open_midnight_belongs_to_previous_day() {
        let r = resolver(vec![], None);
        let got = |ts| r.resolve_instant("BTCUSDT", None, ts).unwrap();
        // 0 点整 = 前一日的区间终点
        let exact = got(at(2024, 1, 3, 0, 0, 0));
        assert_eq!(
            exact,
            Resolution {
                trading_day: d(2024, 1, 2),
                session: TradingSession::Continuous
            }
        );
        // 0 点后 1ns 起属于当日
        let after = got(at(2024, 1, 3, 0, 0, 0) + 1);
        assert_eq!(
            after,
            Resolution {
                trading_day: d(2024, 1, 3),
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

    #[test]
    fn china_timezone_detection() {
        assert!(is_china_timezone(Some("Asia/Shanghai"), 0));
        assert!(is_china_timezone(Some("PRC"), 0));
        assert!(is_china_timezone(None, 8 * 3600));
        assert!(!is_china_timezone(Some("America/New_York"), 8 * 3600));
        assert!(!is_china_timezone(None, 0));
    }

    #[test]
    fn effective_sessions_rule_b() {
        // 中国时区、未传：空（按资产类型取默认）
        assert!(effective_sessions(None, Some("Asia/Shanghai"), 8 * 3600).is_empty());
        // 非中国时区、未传：default = ALWAYS_OPEN
        let ny = effective_sessions(None, Some("America/New_York"), 0);
        assert_eq!(ny.len(), 1);
        assert_eq!(ny[0].0, "default");
        assert_eq!(ny[0].1.name, "ALWAYS_OPEN");
        // 非中国时区、传了但没有 default：补 default
        let user = vec![("rb".to_string(), template("CN_FUTURES_NIGHT_23").unwrap())];
        let got = effective_sessions(Some(user), Some("Europe/London"), 0);
        assert!(
            got.iter()
                .any(|(k, s)| k == "default" && s.name == "ALWAYS_OPEN")
        );
        assert!(got.iter().any(|(k, _)| k == "rb"));
        // 非中国时区、传了 default（大小写不敏感）：不覆盖
        let user = vec![("DEFAULT".to_string(), template("CN_STOCK").unwrap())];
        let got = effective_sessions(Some(user), Some("Europe/London"), 0);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].1.name, "CN_STOCK");
    }

    #[test]
    fn session_day_uses_latest_close_across_active_schedules() {
        // 股票 15:00 收盘、国债期货 15:15 收盘：会话日结束取 15:15
        let mut r = resolver(vec![("T", "CN_FUTURES_CFFEX_BOND_DAY")], None);
        r.set_active(&[
            ("600000.SH".to_string(), Some(AssetType::Stock)),
            ("T2412".to_string(), Some(AssetType::Futures)),
        ]);
        assert_eq!(
            r.session_day_end(d(2024, 1, 2)).unwrap(),
            at(2024, 1, 2, 15, 15, 0)
        );
        // 15:10 的股票迟到 tick 仍属当日会话（单品种解析会归下一交易日）
        assert_eq!(
            r.session_trading_day(at(2024, 1, 2, 15, 10, 0)).unwrap(),
            d(2024, 1, 2)
        );
        assert_eq!(
            r.session_trading_day(at(2024, 1, 2, 15, 15, 1)).unwrap(),
            d(2024, 1, 3)
        );
        let (prev_end, end) = r.session_day_span(d(2024, 1, 3)).unwrap();
        assert_eq!(prev_end, at(2024, 1, 2, 15, 15, 0));
        assert_eq!(end, at(2024, 1, 3, 15, 15, 0));
    }

    #[test]
    fn session_day_with_night_futures_and_weekend() {
        let mut r = resolver(vec![("rb", "CN_FUTURES_NIGHT_23")], None);
        r.set_active(&[("rb2405".to_string(), Some(AssetType::Futures))]);
        // 周五 21:30 → 下周一
        assert_eq!(
            r.session_trading_day(at(2024, 1, 5, 21, 30, 0)).unwrap(),
            d(2024, 1, 8)
        );
        // 下周一的会话区间从上周五 15:00 开始
        let (prev_end, end) = r.session_day_span(d(2024, 1, 8)).unwrap();
        assert_eq!(prev_end, at(2024, 1, 5, 15, 0, 0));
        assert_eq!(end, at(2024, 1, 8, 15, 0, 0));
        // 周五之后的下一会话交易日是下周一
        assert_eq!(r.next_session_day(d(2024, 1, 5)).unwrap(), d(2024, 1, 8));
    }

    #[test]
    fn session_day_first_start_and_first_continuous_start() {
        // 股票：首段 09:15 开盘集合竞价，首个连续竞价段 09:30
        let mut stock = resolver(vec![], None);
        stock.set_active(&[("600000.SH".to_string(), Some(AssetType::Stock))]);
        assert_eq!(
            stock.session_day_first_start(d(2024, 1, 2)).unwrap(),
            at(2024, 1, 2, 9, 15, 0)
        );
        assert_eq!(
            stock
                .session_day_first_continuous_start(d(2024, 1, 2))
                .unwrap(),
            Some(at(2024, 1, 2, 9, 30, 0))
        );
        // 夜盘期货：周一交易日从上周五 20:55 夜盘集合竞价开始、21:00 连续竞价
        let mut night = resolver(vec![("au", "CN_FUTURES_NIGHT_0230")], None);
        night.set_active(&[("au2406".to_string(), Some(AssetType::Futures))]);
        assert_eq!(
            night.session_day_first_start(d(2024, 1, 8)).unwrap(),
            at(2024, 1, 5, 20, 55, 0)
        );
        assert_eq!(
            night
                .session_day_first_continuous_start(d(2024, 1, 8))
                .unwrap(),
            Some(at(2024, 1, 5, 21, 0, 0))
        );
        // 混合在用表取最小值
        let mut mixed = resolver(vec![("au", "CN_FUTURES_NIGHT_0230")], None);
        mixed.set_active(&[
            ("600000.SH".to_string(), Some(AssetType::Stock)),
            ("au2406".to_string(), Some(AssetType::Futures)),
        ]);
        assert_eq!(
            mixed.session_day_first_start(d(2024, 1, 8)).unwrap(),
            at(2024, 1, 5, 20, 55, 0)
        );
        assert_eq!(
            mixed.session_calendar_before(d(2024, 1, 8)).unwrap(),
            d(2024, 1, 5)
        );
    }

    #[test]
    fn session_calendar_all_days_only_when_every_active_schedule_is_always_open() {
        let mut r = resolver(vec![("default", "ALWAYS_OPEN")], None);
        r.set_active(&[("BTCUSDT".to_string(), None)]);
        // 周六 12:00 属于周六
        assert_eq!(
            r.session_trading_day(at(2024, 1, 6, 12, 0, 0)).unwrap(),
            d(2024, 1, 6)
        );
        let mut mixed = resolver(vec![("BTC", "ALWAYS_OPEN")], None);
        mixed.set_active(&[
            ("BTCUSDT".to_string(), None),
            ("600000.SH".to_string(), Some(AssetType::Stock)),
        ]);
        // 混合时会话日历为工作日：周六 12:00 归下周一
        assert_eq!(
            mixed.session_trading_day(at(2024, 1, 6, 12, 0, 0)).unwrap(),
            d(2024, 1, 8)
        );
        assert!(!mixed.is_session_day(d(2024, 1, 6)));
        assert!(mixed.is_session_day(d(2024, 1, 8)));
    }

    #[test]
    fn is_session_day_follows_session_calendar() {
        let mut crypto = resolver(vec![("default", "ALWAYS_OPEN")], None);
        crypto.set_active(&[("BTCUSDT".to_string(), None)]);
        assert!(crypto.is_session_day(d(2024, 1, 6)));
        let explicit = TradingCalendar::explicit([d(2024, 1, 5), d(2024, 1, 8)]).unwrap();
        let r = resolver(vec![], Some(explicit));
        assert!(r.is_session_day(d(2024, 1, 5)));
        assert!(!r.is_session_day(d(2024, 1, 4)));
    }

    #[test]
    fn segment_span_at_returns_segment_and_gap() {
        let r = resolver(vec![], None);
        let stock = Some(AssetType::Stock);
        // 10:00 在上午连续竞价 [09:30, 11:30)
        let (s, e, sess) = r
            .segment_span_at("600000.SH", stock, at(2024, 1, 2, 10, 0, 0))
            .unwrap();
        assert_eq!(
            (s, e, sess),
            (
                at(2024, 1, 2, 9, 30, 0),
                at(2024, 1, 2, 11, 30, 0),
                TradingSession::Continuous
            )
        );
        // 14:58 在收盘竞价 [14:57, 15:00)
        let (s, e, sess) = r
            .segment_span_at("600000.SH", stock, at(2024, 1, 2, 14, 58, 0))
            .unwrap();
        assert_eq!(
            (s, e, sess),
            (
                at(2024, 1, 2, 14, 57, 0),
                at(2024, 1, 2, 15, 0, 0),
                TradingSession::ClosingAuction
            )
        );
        // 20:00 在收盘后到次日开盘前的间隙：Closed，区间 [15:00, 次日 09:15)
        let (s, e, sess) = r
            .segment_span_at("600000.SH", stock, at(2024, 1, 2, 20, 0, 0))
            .unwrap();
        assert_eq!(sess, TradingSession::Closed);
        assert_eq!(s, at(2024, 1, 2, 15, 0, 0));
        assert_eq!(e, at(2024, 1, 3, 9, 15, 0));
    }

    #[test]
    fn uses_all_days_fallback_only_without_calendar_and_all_always_open() {
        let mut crypto = resolver(vec![("default", "ALWAYS_OPEN")], None);
        crypto.set_active(&[("BTCUSDT".to_string(), None)]);
        assert!(crypto.uses_all_days_fallback());
        // 混有中国股票（工作日缺省日历）时不是全日历兜底
        let mut mixed = resolver(vec![("BTC", "ALWAYS_OPEN")], None);
        mixed.set_active(&[
            ("BTCUSDT".to_string(), None),
            ("600000.SH".to_string(), Some(AssetType::Stock)),
        ]);
        assert!(!mixed.uses_all_days_fallback());
        // 显式交易日历：不是兜底
        let cal = TradingCalendar::explicit([d(2024, 1, 2)]).unwrap();
        let mut explicit = resolver(vec![("default", "ALWAYS_OPEN")], Some(cal));
        explicit.set_active(&[("BTCUSDT".to_string(), None)]);
        assert!(!explicit.uses_all_days_fallback());
    }

    #[test]
    fn session_level_instant_takes_highest_rank_across_active_schedules() {
        let mut r = resolver(vec![], None);
        r.set_active(&[
            ("600000.SH".to_string(), Some(AssetType::Stock)),
            ("rb2405".to_string(), Some(AssetType::Futures)),
        ]);
        // 08:57：期货开盘竞价、股票闭市 → OpeningAuction
        assert_eq!(
            r.session_level_instant(at(2024, 1, 2, 8, 57, 0)),
            TradingSession::OpeningAuction
        );
        // 10:20：期货小节休息、股票连续 → Continuous
        assert_eq!(
            r.session_level_instant(at(2024, 1, 2, 10, 20, 0)),
            TradingSession::Continuous
        );
    }

    #[test]
    fn session_level_instant_out_of_calendar_falls_back_to_continuous() {
        let cal = TradingCalendar::explicit([d(2024, 1, 2), d(2024, 1, 3)]).unwrap();
        let r = resolver(vec![], Some(cal));
        // 01-04 超出日历：与撮合的回退一致，按连续竞价
        assert_eq!(
            r.session_level_instant(at(2024, 1, 4, 10, 0, 0)),
            TradingSession::Continuous
        );
        // 日历内的闭市时刻照常是 Closed
        assert_eq!(
            r.session_level_instant(at(2024, 1, 2, 20, 0, 0)),
            TradingSession::Closed
        );
    }

    #[test]
    fn session_at_instant_caches_per_schedule_and_instant() {
        use std::sync::atomic::Ordering;
        let r = resolver(vec![], None);
        let ts = at(2024, 1, 2, 10, 0, 0);
        // 两只股票共用 CN_STOCK 时段表：同一时刻只解析一次
        let a = r.session_at_instant("600000.SH", stock(), ts);
        let b = r.session_at_instant("000001.SZ", stock(), ts);
        assert_eq!(a, Some(TradingSession::Continuous));
        assert_eq!(b, a);
        assert_eq!(r.instant_resolves.load(Ordering::Relaxed), 1);
        // 不同时段表、不同时刻各解析一次；结果与 resolve_instant 一致
        let fut = r.session_at_instant("rb2405", Some(AssetType::Futures), ts);
        assert_eq!(r.instant_resolves.load(Ordering::Relaxed), 2);
        assert_eq!(
            fut,
            Some(
                r.resolve_instant("rb2405", Some(AssetType::Futures), ts)
                    .unwrap()
                    .session
            )
        );
        let noon = at(2024, 1, 2, 12, 0, 0);
        assert_eq!(
            r.session_at_instant("600000.SH", stock(), noon),
            Some(TradingSession::Break)
        );
        assert_eq!(r.instant_resolves.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn session_at_instant_caches_errors_as_none() {
        use std::sync::atomic::Ordering;
        let cal = TradingCalendar::explicit([d(2024, 1, 2), d(2024, 1, 3)]).unwrap();
        let r = resolver(vec![], Some(cal));
        let ts = at(2024, 1, 4, 10, 0, 0);
        assert_eq!(r.session_at_instant("600000.SH", stock(), ts), None);
        assert_eq!(r.session_at_instant("000001.SZ", stock(), ts), None);
        assert_eq!(r.instant_resolves.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn segment_span_at_weekend_gap() {
        let r = resolver(vec![], None);
        // 周五 20:00：间隙跨周末，[周五 15:00, 下周一 09:15)
        let got = r
            .segment_span_at("600000.SH", stock(), at(2024, 1, 5, 20, 0, 0))
            .unwrap();
        assert_eq!(
            got,
            (
                at(2024, 1, 5, 15, 0, 0),
                at(2024, 1, 8, 9, 15, 0),
                TradingSession::Closed
            )
        );
    }

    #[test]
    fn segment_span_at_night_futures_gap_ends_at_night_auction() {
        let r = resolver(vec![("rb", "CN_FUTURES_NIGHT_23")], None);
        // 周五 20:00：日盘收盘后、夜盘集合竞价前，右界为周五 20:55
        let got = r
            .segment_span_at("rb2405", Some(AssetType::Futures), at(2024, 1, 5, 20, 0, 0))
            .unwrap();
        assert_eq!(
            got,
            (
                at(2024, 1, 5, 15, 0, 0),
                at(2024, 1, 5, 20, 55, 0),
                TradingSession::Closed
            )
        );
    }

    #[test]
    fn segment_span_at_explicit_calendar_edges() {
        let cal = TradingCalendar::explicit([d(2024, 1, 2), d(2024, 1, 3)]).unwrap();
        let r = resolver(vec![], Some(cal));
        // 日历首日开盘前：没有前一交易日，左界无界
        let got = r
            .segment_span_at("600000.SH", stock(), at(2024, 1, 2, 8, 0, 0))
            .unwrap();
        assert_eq!(
            got,
            (i64::MIN, at(2024, 1, 2, 9, 15, 0), TradingSession::Closed)
        );
        // 早于日历首日当天：与 resolve_instant 一致，超出日历范围报错
        assert!(
            r.segment_span_at("600000.SH", stock(), at(2024, 1, 1, 20, 0, 0))
                .is_err()
        );
        // 日历末日收盘之后：找不到下一交易日，报错
        assert!(
            r.segment_span_at("600000.SH", stock(), at(2024, 1, 3, 20, 0, 0))
                .is_err()
        );
    }
}
