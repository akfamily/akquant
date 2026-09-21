//! 运行时多周期窗口聚合器(引擎内).
//!
//! 把基础 bar 流按 (symbol, freq) 聚合成更高周期的窗口 bar。与 Python 侧
//! `BarGenerator` / `feed.resample(label="right", closed="right")` 同一时钟对齐
//! 语义: 分钟/小时窗口标签 = 本地时间 ceil 到周期; 日线标签 = 当日最后一根基础
//! bar 的时间戳(不指向未来)。窗口 bar **不进 feed、不成为 Event**, 只写历史
//! 缓冲并交给策略阶段派发。
use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Timelike, Utc};
use chrono_tz::Tz;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::model::Bar;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFreq {
    Minutes(u32),
    Hours(u32),
    Day,
}

impl WindowFreq {
    /// 解析 `"5min"` / `"1h"` / `"1d"`(大小写不敏感)。
    pub fn parse(text: &str) -> Result<Self, String> {
        let s = text.trim().to_ascii_lowercase();
        let split = s
            .find(|c: char| !c.is_ascii_digit())
            .ok_or_else(|| format!("freq={text:?} 缺少单位, 仅支持 '5min' / '1h' / '1d' 形式"))?;
        let (num, unit) = s.split_at(split);
        let n: u32 = num.parse().map_err(|_| format!("freq={text:?} 数值无效"))?;
        if n == 0 {
            return Err(format!("freq={text:?} 数值必须为正"));
        }
        match unit {
            "min" => Ok(Self::Minutes(n)),
            "h" => Ok(Self::Hours(n)),
            "d" if n == 1 => Ok(Self::Day),
            "d" => Err(format!("freq={text:?}: 日周期仅支持 '1d'")),
            _ => Err(format!(
                "freq={text:?} 单位无效: 仅支持整数分钟 'Nmin'、整数小时 'Nh' 与 '1d'; \
                 秒级/周月请用 akquant.feed_adapter 的 resample()"
            )),
        }
    }

    pub fn total_minutes(self) -> u32 {
        match self {
            Self::Minutes(n) => n,
            Self::Hours(n) => n * 60,
            Self::Day => 1440,
        }
    }

    fn period_seconds(self) -> Option<i64> {
        match self {
            Self::Minutes(n) => Some(i64::from(n) * 60),
            Self::Hours(n) => Some(i64::from(n) * 3600),
            Self::Day => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WindowSubscription {
    /// `None` = 覆盖全部标的。
    pub symbol: Option<String>,
    pub freq: WindowFreq,
    /// 规范化标签(小写), 写进 `Bar.freq` 与历史桶键。
    pub freq_label: String,
    /// 本地交易时段; 给定则窗口不跨时段拼接。
    pub sessions: Vec<(NaiveTime, NaiveTime)>,
}

impl WindowSubscription {
    pub fn parse(
        symbol: Option<String>,
        freq: &str,
        sessions: Option<Vec<(String, String)>>,
    ) -> Result<Self, String> {
        let parsed = WindowFreq::parse(freq)?;
        let mut parsed_sessions = Vec::new();
        for (start, end) in sessions.unwrap_or_default() {
            let s = NaiveTime::parse_from_str(start.trim(), "%H:%M")
                .map_err(|_| format!("session_windows 起点 {start:?} 不是 HH:MM"))?;
            let e = NaiveTime::parse_from_str(end.trim(), "%H:%M")
                .map_err(|_| format!("session_windows 终点 {end:?} 不是 HH:MM"))?;
            if e <= s {
                return Err(format!("session_windows ({start}, {end}) 终点必须晚于起点"));
            }
            parsed_sessions.push((s, e));
        }
        Ok(Self {
            symbol,
            freq: parsed,
            freq_label: freq.trim().to_ascii_lowercase(),
            sessions: parsed_sessions,
        })
    }

    fn matches(&self, symbol: &str) -> bool {
        self.symbol.as_deref().is_none_or(|s| s == symbol)
    }

    /// 返回 (本地日, 时段序号); 段外归 -1。无时段配置返回 None。
    fn session_key(&self, local: NaiveDateTime) -> Option<(NaiveDate, i32)> {
        if self.sessions.is_empty() {
            return None;
        }
        let t = local.time();
        let idx = self
            .sessions
            .iter()
            .position(|(s, e)| *s <= t && t <= *e)
            .map_or(-1, |i| i as i32);
        Some((local.date(), idx))
    }

    fn is_last_session_end(&self, local: NaiveDateTime) -> bool {
        self.sessions
            .last()
            .is_some_and(|(_, e)| *e == local.time())
    }
}

/// 探测夏令时"跳变缺口"时, 向前步进的分钟数上限(6 小时足够覆盖现实世界的
/// 任何春季跳变缺口, 通常缺口只有 1 小时, 极少数地区 30 分钟)。
const MAX_DST_GAP_PROBE_MINUTES: i64 = 360;

/// 引擎时区: 有 IANA 名称用名称(随 DST), 否则用固定偏移秒数。
#[derive(Debug, Clone, Default)]
struct LocalClock {
    tz: Option<Tz>,
    offset_secs: i32,
}

impl LocalClock {
    fn new(tz_name: Option<&str>, offset_secs: i32) -> Self {
        Self {
            tz: tz_name.and_then(|n| n.parse::<Tz>().ok()),
            offset_secs,
        }
    }

    fn to_local(&self, ts_ns: i64) -> NaiveDateTime {
        let secs = ts_ns.div_euclid(1_000_000_000);
        let nanos = ts_ns.rem_euclid(1_000_000_000) as u32;
        // `ts_ns` 总是来自一个已经存在的合法 `Bar.timestamp`(UTC 纳秒); UTC 本身
        // 没有本地时间缺口/重叠这类问题, `single()` 只有在 `secs` 超出 chrono
        // 可表示的范围(约公元前 26 万年~公元后 26 万年)时才会是 `None` —— 真实
        // 行情时间戳不可能触发, 故此处 `expect` 视为不可达分支。
        let utc: DateTime<Utc> = Utc
            .timestamp_opt(secs, nanos)
            .single()
            .expect("invalid ts: secs out of chrono's representable range");
        match &self.tz {
            Some(tz) => utc.with_timezone(tz).naive_local(),
            None => (utc + chrono::Duration::seconds(i64::from(self.offset_secs))).naive_utc(),
        }
    }

    fn from_local(&self, local: NaiveDateTime) -> i64 {
        match &self.tz {
            Some(tz) => Self::nanos_or_clamp(&Self::resolve_local(tz, local)),
            None => Self::nanos_or_clamp(
                &(local - chrono::Duration::seconds(i64::from(self.offset_secs))).and_utc(),
            ),
        }
    }

    /// 把可能落入夏令时"春季跳变缺口"的本地时间解析成合法瞬间。
    ///
    /// 缺口内的本地时间不对应任何真实瞬间(例如美东 2024-03-10 当天 02:00-02:59
    /// 因夏令时整体跳过而不存在); `from_local_datetime(..).earliest()` 对此返回
    /// `None`。为了不让引擎因为一次窗口标签落在缺口里就 panic, 向前逐分钟探测,
    /// 取缺口结束后第一个合法瞬间(与 pandas/pytz `nonexistent="shift_forward"`
    /// 语义一致)。`Ambiguous`(秋季回拨的重叠时段)则已经由 `earliest()` 正确
    /// 处理为取较早的那个瞬间, 无需特殊分支。
    fn resolve_local(tz: &Tz, local: NaiveDateTime) -> DateTime<Tz> {
        if let Some(dt) = tz.from_local_datetime(&local).earliest() {
            return dt;
        }
        let mut probe = local;
        for _ in 0..MAX_DST_GAP_PROBE_MINUTES {
            probe += chrono::Duration::minutes(1);
            if let Some(dt) = tz.from_local_datetime(&probe).earliest() {
                return dt;
            }
        }
        // 现实世界的夏令时缺口不会超过几个小时; 走到这里说明探测窗口给的不够
        // (可能是异常时区配置), 而不是真的存在这么长的缺口——保留 panic 以便
        // 第一时间暴露问题, 而不是悄悄返回一个误导性的时间。
        panic!(
            "no valid local time found after {local} within {MAX_DST_GAP_PROBE_MINUTES} minutes (tz={tz:?})"
        );
    }

    /// `timestamp_nanos_opt()` 只有在瞬间超出 chrono 纳秒可表示范围(约
    /// 1677~2262 年)时才会是 `None`, 真实行情时间戳不会触达; 但这里不用
    /// `expect` panic, 而是 clamp 到边界值, 避免引擎因为一次离群配置崩溃。
    fn nanos_or_clamp<T: TimeZone>(dt: &DateTime<T>) -> i64 {
        dt.timestamp_nanos_opt().unwrap_or(if dt.timestamp() < 0 {
            i64::MIN
        } else {
            i64::MAX
        })
    }

    /// 本地时间 ceil 到周期(以本地零点为锚), 返回 UTC 纳秒标签。
    fn ceil_label(&self, local: NaiveDateTime, period_secs: i64) -> i64 {
        let secs_of_day = i64::from(local.num_seconds_from_midnight());
        // `i64::div_ceil` (signed) is unstable on this toolchain; compute manually.
        let ceiled = ((secs_of_day + period_secs - 1) / period_secs) * period_secs;
        let midnight = local.date().and_hms_opt(0, 0, 0).expect("midnight");
        self.from_local(midnight + chrono::Duration::seconds(ceiled))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowStateSnapshot {
    pub symbol: String,
    pub freq_label: String,
    pub label_ns: i64,
    pub session_key: Option<(NaiveDate, i32)>,
    pub local_date: NaiveDate,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WindowAggregatorSnapshot {
    pub states: Vec<WindowStateSnapshot>,
}

#[derive(Debug, Clone)]
struct WindowState {
    /// 缓存周期(避免 `flush` 每次都重新 `WindowFreq::parse(&freq_label)`);
    /// 从 `restore()` 恢复时若 snapshot 里的 `freq_label` 无法解析(数据损坏/
    /// 来自未来版本), 退化为 `Minutes(0)` —— 只影响 `flush()` 的排序, 不影响
    /// 已缓存的 OHLCV 数据本身。
    freq: WindowFreq,
    label_ns: i64,
    session_key: Option<(NaiveDate, i32)>,
    local_date: NaiveDate,
    open: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    volume: Decimal,
}

impl WindowState {
    fn open_from(
        bar: &Bar,
        freq: WindowFreq,
        label_ns: i64,
        session_key: Option<(NaiveDate, i32)>,
        local_date: NaiveDate,
    ) -> Self {
        Self {
            freq,
            label_ns,
            session_key,
            local_date,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
        }
    }

    fn merge(&mut self, bar: &Bar) {
        if bar.high > self.high {
            self.high = bar.high;
        }
        if bar.low < self.low {
            self.low = bar.low;
        }
        self.close = bar.close;
        self.volume += bar.volume;
    }

    fn to_bar(&self, symbol: &str, freq_label: &str) -> Bar {
        Bar {
            timestamp: self.label_ns,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            symbol: symbol.to_string(),
            extra: HashMap::new(),
            freq: Some(freq_label.to_string()),
        }
    }
}

#[derive(Debug, Default)]
pub struct WindowAggregator {
    subscriptions: Vec<WindowSubscription>,
    base_interval_min: Option<u32>,
    clock: LocalClock,
    /// (symbol, freq_label) -> 在形成窗口
    states: HashMap<(String, String), WindowState>,
}

impl WindowAggregator {
    pub fn with_clock(tz_name: Option<&str>, offset_secs: i32) -> Self {
        Self {
            clock: LocalClock::new(tz_name, offset_secs),
            ..Self::default()
        }
    }

    pub fn set_clock(&mut self, tz_name: Option<&str>, offset_secs: i32) {
        self.clock = LocalClock::new(tz_name, offset_secs);
    }

    /// 替换订阅表。**不清空**仍被订阅的在形成窗口(checkpoint 恢复后再配置时要保留),
    /// 只丢掉不再订阅的周期。
    ///
    /// 同一 `freq_label` 下, 两条订阅的 symbol 范围若有重叠(`None` 与任何范围
    /// 重叠; 两个 `Some` 仅在相等时重叠), 会共享同一个 `(symbol, freq_label)`
    /// state key: `update()` 会对匹配到的每条订阅各跑一次 merge/session 判断,
    /// 重叠但配置不同(sessions 或 symbol 范围不完全一致)时会重复 merge 或用
    /// 不同的 `session_key` 互相打架, 产生错误数据 —— 因此完全相同(scope +
    /// sessions 都一致)的订阅静默去重, 其余重叠一律拒绝。
    pub fn configure(
        &mut self,
        subs: Vec<WindowSubscription>,
        base_interval_min: Option<u32>,
    ) -> Result<(), String> {
        let deduped = Self::dedup_and_validate(subs)?;
        self.subscriptions = deduped;
        self.base_interval_min = base_interval_min;
        let keep: Vec<String> = self
            .subscriptions
            .iter()
            .map(|s| s.freq_label.clone())
            .collect();
        self.states.retain(|(_, f), _| keep.contains(f));
        Ok(())
    }

    /// 见 [`Self::configure`] 的重叠规则说明。
    fn dedup_and_validate(
        subs: Vec<WindowSubscription>,
    ) -> Result<Vec<WindowSubscription>, String> {
        let mut kept: Vec<WindowSubscription> = Vec::with_capacity(subs.len());
        for sub in subs {
            let mut is_duplicate = false;
            for existing in &kept {
                if existing.freq_label != sub.freq_label {
                    continue;
                }
                if existing.symbol == sub.symbol && existing.sessions == sub.sessions {
                    is_duplicate = true;
                    break;
                }
                if Self::scopes_overlap(&existing.symbol, &sub.symbol) {
                    return Err(format!(
                        "订阅冲突: freq={:?} 的 symbol 范围重叠(已有 symbol={:?}, 新增 \
                         symbol={:?})但配置不一致(sessions 或 symbol 范围不完全相同); \
                         请让重叠的订阅保持完全一致, 或改成互斥的 symbol 范围",
                        sub.freq_label, existing.symbol, sub.symbol
                    ));
                }
            }
            if !is_duplicate {
                kept.push(sub);
            }
        }
        Ok(kept)
    }

    /// `None` 覆盖全部标的, 与任何范围都重叠; 两个 `Some` 仅在标的相等时重叠。
    fn scopes_overlap(a: &Option<String>, b: &Option<String>) -> bool {
        match (a, b) {
            (None, _) | (_, None) => true,
            (Some(x), Some(y)) => x == y,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.subscriptions.is_empty()
    }

    pub fn subscribed_freqs(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .subscriptions
            .iter()
            .map(|s| s.freq_label.clone())
            .collect();
        v.sort();
        v.dedup();
        v
    }

    /// 喂入一根基础 bar, 返回本步闭合的窗口 bar(按周期从小到大)。
    pub fn update(&mut self, bar: &Bar) -> Vec<Bar> {
        if self.subscriptions.is_empty() {
            return Vec::new();
        }
        let local = self.clock.to_local(bar.timestamp);
        let base_known = self.base_interval_min.is_some();
        // (总分钟数, freq_label, bar); label 作 tie-break, 保证 "1h" 与 "60min"
        // 这类分钟数相同但标签不同的周期有确定顺序(否则顺序依赖 HashMap 迭代,
        // 对 golden baseline 是隐患)。
        let mut closed: Vec<(u32, String, Bar)> = Vec::new();

        for sub in self.subscriptions.iter().filter(|s| s.matches(&bar.symbol)) {
            let key = (bar.symbol.clone(), sub.freq_label.clone());
            match sub.freq.period_seconds() {
                Some(period) => {
                    let label = self.clock.ceil_label(local, period);
                    let skey = sub.session_key(local);
                    if let Some(st) = self.states.get(&key)
                        && (st.label_ns != label || st.session_key != skey)
                    {
                        closed.push((
                            sub.freq.total_minutes(),
                            sub.freq_label.clone(),
                            st.to_bar(&bar.symbol, &sub.freq_label),
                        ));
                        self.states.remove(&key);
                    }
                    let st = self
                        .states
                        .entry(key.clone())
                        .and_modify(|s| s.merge(bar))
                        .or_insert_with(|| {
                            WindowState::open_from(bar, sub.freq, label, skey, local.date())
                        });
                    if base_known && bar.timestamp == label {
                        closed.push((
                            sub.freq.total_minutes(),
                            sub.freq_label.clone(),
                            st.to_bar(&bar.symbol, &sub.freq_label),
                        ));
                        self.states.remove(&key);
                    }
                }
                None => {
                    let date = local.date();
                    if let Some(st) = self.states.get(&key)
                        && st.local_date != date
                    {
                        closed.push((
                            sub.freq.total_minutes(),
                            sub.freq_label.clone(),
                            st.to_bar(&bar.symbol, &sub.freq_label),
                        ));
                        self.states.remove(&key);
                    }
                    let st = self
                        .states
                        .entry(key.clone())
                        .and_modify(|s| {
                            s.merge(bar);
                            s.label_ns = bar.timestamp;
                        })
                        .or_insert_with(|| {
                            WindowState::open_from(bar, sub.freq, bar.timestamp, None, date)
                        });
                    if base_known && sub.is_last_session_end(local) {
                        closed.push((
                            sub.freq.total_minutes(),
                            sub.freq_label.clone(),
                            st.to_bar(&bar.symbol, &sub.freq_label),
                        ));
                        self.states.remove(&key);
                    }
                }
            }
        }
        closed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        closed.into_iter().map(|(_, _, b)| b).collect()
    }

    /// 闭合全部在形成窗口(会话结束)。按 (symbol, 周期, freq_label) 排序;
    /// freq_label 作 tie-break 原因同 [`Self::update`]。
    pub fn flush(&mut self) -> Vec<Bar> {
        let mut out: Vec<(String, u32, String, Bar)> = Vec::new();
        for ((symbol, label), st) in self.states.drain() {
            out.push((
                symbol.clone(),
                st.freq.total_minutes(),
                label.clone(),
                st.to_bar(&symbol, &label),
            ));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        out.into_iter().map(|(_, _, _, b)| b).collect()
    }

    pub fn current(&self, symbol: &str, freq: &str) -> Option<Bar> {
        let label = freq.trim().to_ascii_lowercase();
        self.states
            .get(&(symbol.to_string(), label.clone()))
            .map(|st| st.to_bar(symbol, &label))
    }

    pub fn snapshot(&self) -> WindowAggregatorSnapshot {
        let mut states: Vec<WindowStateSnapshot> = self
            .states
            .iter()
            .map(|((symbol, freq_label), st)| WindowStateSnapshot {
                symbol: symbol.clone(),
                freq_label: freq_label.clone(),
                label_ns: st.label_ns,
                session_key: st.session_key,
                local_date: st.local_date,
                open: st.open,
                high: st.high,
                low: st.low,
                close: st.close,
                volume: st.volume,
            })
            .collect();
        states.sort_by(|a, b| {
            a.symbol
                .cmp(&b.symbol)
                .then(a.freq_label.cmp(&b.freq_label))
        });
        WindowAggregatorSnapshot { states }
    }

    pub fn restore(&mut self, snapshot: WindowAggregatorSnapshot) {
        self.states.clear();
        for s in snapshot.states {
            // 正常情况下 freq_label 一定是我们自己 `snapshot()` 时写入的合法值,
            // 解析失败(数据损坏/来自未来版本)时退化为 Minutes(0), 只影响
            // `flush()` 的排序, 不影响已恢复的 OHLCV 数据。
            let freq = WindowFreq::parse(&s.freq_label).unwrap_or(WindowFreq::Minutes(0));
            self.states.insert(
                (s.symbol, s.freq_label),
                WindowState {
                    freq,
                    label_ns: s.label_ns,
                    session_key: s.session_key,
                    local_date: s.local_date,
                    open: s.open,
                    high: s.high,
                    low: s.low,
                    close: s.close,
                    volume: s.volume,
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use rust_decimal::Decimal;
    use std::collections::HashMap;

    fn ns(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> i64 {
        chrono_tz::Asia::Shanghai
            .with_ymd_and_hms(y, mo, d, h, mi, 0)
            .unwrap()
            .timestamp_nanos_opt()
            .unwrap()
    }

    fn bar(ts: i64, close: i64, symbol: &str) -> Bar {
        let mut extra = HashMap::new();
        extra.insert("factor".to_string(), 1.0);
        Bar {
            timestamp: ts,
            open: Decimal::from(close),
            high: Decimal::from(close + 1),
            low: Decimal::from(close - 1),
            close: Decimal::from(close),
            volume: Decimal::from(10),
            symbol: symbol.to_string(),
            extra,
            freq: None,
        }
    }

    fn agg(subs: &[(&str, Option<Vec<(String, String)>>)], base: Option<u32>) -> WindowAggregator {
        let mut a = WindowAggregator::with_clock(Some("Asia/Shanghai"), 28800);
        let parsed = subs
            .iter()
            .map(|(f, s)| WindowSubscription::parse(None, f, s.clone()).unwrap())
            .collect();
        a.configure(parsed, base).unwrap();
        a
    }

    #[test]
    fn parse_accepts_min_hour_day_and_rejects_others() {
        assert_eq!(WindowFreq::parse("5min").unwrap(), WindowFreq::Minutes(5));
        assert_eq!(WindowFreq::parse("1H").unwrap(), WindowFreq::Hours(1));
        assert_eq!(WindowFreq::parse("1d").unwrap(), WindowFreq::Day);
        assert!(WindowFreq::parse("30s").is_err());
        assert!(WindowFreq::parse("2d").is_err());
        assert!(WindowFreq::parse("1w").is_err());
        assert!(WindowFreq::parse("0min").is_err());
    }

    #[test]
    fn five_minute_window_closes_immediately_when_base_known() {
        // 1min bar 打右缘: 09:31..09:35 五根 → 一根 5min, 标签 09:35, 在 09:35 那步闭合
        let mut a = agg(&[("5min", None)], Some(1));
        let mut out = Vec::new();
        for (i, m) in (31..=35).enumerate() {
            let closed = a.update(&bar(ns(2024, 1, 2, 9, m), 10 + i as i64, "X"));
            if m < 35 {
                assert!(closed.is_empty(), "minute {m} must not close");
            } else {
                out = closed;
            }
        }
        assert_eq!(out.len(), 1);
        let w = &out[0];
        assert_eq!(w.timestamp, ns(2024, 1, 2, 9, 35));
        assert_eq!(w.freq.as_deref(), Some("5min"));
        assert_eq!(w.open, Decimal::from(10));
        assert_eq!(w.high, Decimal::from(15));
        assert_eq!(w.low, Decimal::from(9));
        assert_eq!(w.close, Decimal::from(14));
        assert_eq!(w.volume, Decimal::from(50));
        assert!(w.extra.is_empty(), "extra 不聚合");
        assert!(a.current("X", "5min").is_none(), "闭合后不应残留在形成窗口");
    }

    #[test]
    fn deferred_close_when_base_unknown() {
        let mut a = agg(&[("5min", None)], None);
        for m in 31..=35 {
            assert!(a.update(&bar(ns(2024, 1, 2, 9, m), 10, "X")).is_empty());
        }
        // 09:36 落入下一窗口才闭合上一窗口
        let closed = a.update(&bar(ns(2024, 1, 2, 9, 36), 10, "X"));
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].timestamp, ns(2024, 1, 2, 9, 35));
    }

    #[test]
    fn hour_and_day_labels() {
        let mut a = agg(&[("1h", None), ("1d", None)], Some(1));
        let mut hours = Vec::new();
        let mut days = Vec::new();
        for (h, m) in [(9, 31), (9, 59), (10, 0), (10, 30), (11, 0)] {
            for b in a.update(&bar(ns(2024, 1, 2, h, m), 10, "X")) {
                match b.freq.as_deref() {
                    Some("1h") => hours.push(b.timestamp),
                    Some("1d") => days.push(b.timestamp),
                    _ => panic!(),
                }
            }
        }
        assert_eq!(hours, vec![ns(2024, 1, 2, 10, 0), ns(2024, 1, 2, 11, 0)]);
        assert!(days.is_empty(), "日线未跨日不闭合");
        // 跨日: 日线标签 = 当日最后一根基础 bar 时间戳(11:00), 不是次日零点
        let closed = a.update(&bar(ns(2024, 1, 3, 9, 31), 10, "X"));
        let day = closed
            .iter()
            .find(|b| b.freq.as_deref() == Some("1d"))
            .unwrap();
        assert_eq!(day.timestamp, ns(2024, 1, 2, 11, 0));
    }

    #[test]
    fn day_closes_immediately_at_last_session_end_when_sessions_given() {
        let sessions = Some(vec![
            ("09:30".to_string(), "11:30".to_string()),
            ("13:00".to_string(), "15:00".to_string()),
        ]);
        let mut a = agg(&[("1d", sessions)], Some(1));
        assert!(a.update(&bar(ns(2024, 1, 2, 14, 59), 10, "X")).is_empty());
        let closed = a.update(&bar(ns(2024, 1, 2, 15, 0), 11, "X"));
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].timestamp, ns(2024, 1, 2, 15, 0));
        assert_eq!(closed[0].close, Decimal::from(11));
    }

    #[test]
    fn session_windows_split_across_lunch_but_clock_only_merges() {
        let sessions = Some(vec![
            ("09:30".to_string(), "11:30".to_string()),
            ("13:00".to_string(), "15:00".to_string()),
        ]);
        // 11:29, 11:30 在上午段; 13:01 在下午段。用 4h 窗口(以本地零点为锚: 边界
        // 0/4/8/12/16/20)不行——12:00 本就是 4h 网格线, 11:29(→12:00)与 13:01
        // (→16:00)天然落在不同桶, 不管有没有 sessions 都会闭合, 测不出 sessions
        // 单独的分裂效果(已用脚本核实: ((41340+14399)//14400)*14400=43200=12:00,
        // ((46860+14399)//14400)*14400=57600=16:00, 二者不同)。改用 5h 窗口:
        // 边界 0/5/10/15/20 不含 12:00, 11:29 与 13:01 同属 ceil(5h)=15:00 桶
        // (((41340+17999)//18000)*18000=54000=15:00, ((46860+17999)//18000)*18000
        // =54000=15:00, 二者相同), 无 sessions 会合并; 有 sessions 则因跨时段
        // (上午段 idx0 → 下午段 idx1)在 13:01 处闭合上午窗口。
        let mut with = agg(&[("5h", sessions)], Some(1));
        with.update(&bar(ns(2024, 1, 2, 11, 29), 10, "X"));
        let closed = with.update(&bar(ns(2024, 1, 2, 13, 1), 20, "X"));
        assert_eq!(closed.len(), 1, "跨时段必须闭合");
        assert_eq!(closed[0].close, Decimal::from(10));

        let mut without = agg(&[("5h", None)], Some(1));
        without.update(&bar(ns(2024, 1, 2, 11, 29), 10, "X"));
        assert!(
            without
                .update(&bar(ns(2024, 1, 2, 13, 1), 20, "X"))
                .is_empty()
        );
    }

    #[test]
    fn flush_emits_all_partial_windows_sorted_by_period() {
        let mut a = agg(&[("1d", None), ("5min", None)], Some(1));
        a.update(&bar(ns(2024, 1, 2, 9, 31), 10, "X"));
        let out = a.flush();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].freq.as_deref(), Some("5min"));
        assert_eq!(out[1].freq.as_deref(), Some("1d"));
        assert!(a.flush().is_empty());
    }

    #[test]
    fn symbols_are_independent_and_symbol_filter_applies() {
        let mut a = WindowAggregator::with_clock(Some("Asia/Shanghai"), 28800);
        a.configure(
            vec![WindowSubscription::parse(Some("X".to_string()), "5min", None).unwrap()],
            Some(1),
        )
        .unwrap();
        for m in 31..=35 {
            a.update(&bar(ns(2024, 1, 2, 9, m), 10, "Y")); // 未订阅, 应忽略
            let closed = a.update(&bar(ns(2024, 1, 2, 9, m), 20, "X"));
            if m == 35 {
                assert_eq!(closed.len(), 1);
                assert_eq!(closed[0].symbol, "X");
            }
        }
        assert!(a.current("Y", "5min").is_none());
    }

    #[test]
    fn current_returns_partial_without_side_effects() {
        let mut a = agg(&[("5min", None)], Some(1));
        a.update(&bar(ns(2024, 1, 2, 9, 31), 10, "X"));
        a.update(&bar(ns(2024, 1, 2, 9, 32), 12, "X"));
        let cur = a.current("X", "5min").unwrap();
        assert_eq!(cur.timestamp, ns(2024, 1, 2, 9, 35));
        assert_eq!(cur.close, Decimal::from(12));
        assert_eq!(cur.freq.as_deref(), Some("5min"));
        assert!(a.current("X", "5min").is_some(), "current 不触发闭合");
    }

    #[test]
    fn snapshot_roundtrip_restores_partial_window() {
        let mut a = agg(&[("5min", None)], Some(1));
        a.update(&bar(ns(2024, 1, 2, 9, 31), 10, "X"));
        let snap = a.snapshot();
        let bytes = rmp_serde::to_vec(&snap).unwrap();
        let decoded: WindowAggregatorSnapshot = rmp_serde::from_slice(&bytes).unwrap();
        let mut b = agg(&[("5min", None)], Some(1));
        b.restore(decoded);
        let closed = b.update(&bar(ns(2024, 1, 2, 9, 35), 14, "X"));
        assert_eq!(closed.len(), 1);
        assert_eq!(
            closed[0].open,
            Decimal::from(10),
            "恢复的窗口应保留 09:31 的 open"
        );
    }

    #[test]
    fn configure_drops_states_of_unsubscribed_freqs_but_keeps_others() {
        let mut a = agg(&[("5min", None), ("1h", None)], Some(1));
        a.update(&bar(ns(2024, 1, 2, 9, 31), 10, "X"));
        a.configure(
            vec![WindowSubscription::parse(None, "1h", None).unwrap()],
            Some(1),
        )
        .unwrap();
        assert!(a.current("X", "5min").is_none());
        assert!(a.current("X", "1h").is_some());
    }

    #[test]
    fn utc_clock_changes_day_boundary() {
        // 北京 07:00 = UTC 前一日 23:00: 用 UTC 时钟时, 08:00 已跨 UTC 日
        let mut a = WindowAggregator::with_clock(Some("UTC"), 0);
        a.configure(
            vec![WindowSubscription::parse(None, "1d", None).unwrap()],
            Some(60),
        )
        .unwrap();
        assert!(a.update(&bar(ns(2024, 1, 2, 7, 0), 10, "X")).is_empty());
        let closed = a.update(&bar(ns(2024, 1, 2, 8, 0), 11, "X"));
        assert_eq!(closed.len(), 1, "UTC 日界在北京 08:00");
    }

    #[test]
    fn dst_gap_label_does_not_panic_and_lands_after_gap() {
        // 美东 2024-03-10 因夏令时"春季跳变", 当地 02:00-02:59:59 整体不存在
        // (01:59:59 EST 直接跳到 03:00:00 EDT)。1h 窗口在本地 01:59 输入后,
        // ceil 到的标签正是不存在的 02:00 —— 曾经会在这里 panic。
        let mut a = WindowAggregator::with_clock(Some("America/New_York"), 0);
        a.configure(
            vec![WindowSubscription::parse(None, "1h", None).unwrap()],
            Some(1),
        )
        .unwrap();
        let ts = chrono_tz::America::New_York
            .with_ymd_and_hms(2024, 3, 10, 1, 59, 0)
            .unwrap()
            .timestamp_nanos_opt()
            .unwrap();
        let closed = a.update(&bar(ts, 10, "X"));
        assert!(closed.is_empty(), "不应 panic, 且尚未闭合");
        let expected_label = chrono_tz::America::New_York
            .with_ymd_and_hms(2024, 3, 10, 3, 0, 0)
            .unwrap()
            .timestamp_nanos_opt()
            .unwrap();
        assert_eq!(
            a.current("X", "1h").unwrap().timestamp,
            expected_label,
            "缺口后应落在缺口结束的第一个合法瞬间(本地 03:00)"
        );
    }

    #[test]
    fn configure_rejects_overlapping_same_label_with_different_sessions() {
        let sessions = Some(vec![("09:30".to_string(), "11:30".to_string())]);
        let mut a = WindowAggregator::with_clock(Some("Asia/Shanghai"), 28800);
        let err = a
            .configure(
                vec![
                    WindowSubscription::parse(None, "5min", None).unwrap(),
                    WindowSubscription::parse(Some("X".to_string()), "5min", sessions).unwrap(),
                ],
                Some(1),
            )
            .unwrap_err();
        assert!(err.contains("5min"), "错误信息应指出冲突的 freq: {err}");
    }

    #[test]
    fn configure_collapses_identical_duplicates() {
        let mut a = agg(&[("5min", None), ("5min", None)], Some(1));
        assert_eq!(a.subscribed_freqs(), vec!["5min".to_string()]);
        let mut out = Vec::new();
        for m in 31..=35 {
            out = a.update(&bar(ns(2024, 1, 2, 9, m), 10, "X"));
        }
        assert_eq!(out.len(), 1, "重复订阅去重后只应闭合一根窗口 bar");
    }

    #[test]
    fn flush_order_is_deterministic_for_equal_periods() {
        let mut a = agg(&[("1h", None), ("60min", None)], Some(1));
        a.update(&bar(ns(2024, 1, 2, 9, 31), 10, "X"));
        let out_a = a.flush();
        let mut b = agg(&[("1h", None), ("60min", None)], Some(1));
        b.update(&bar(ns(2024, 1, 2, 9, 31), 10, "X"));
        let out_b = b.flush();
        let labels_a: Vec<&str> = out_a
            .iter()
            .map(|bar| bar.freq.as_deref().unwrap())
            .collect();
        let labels_b: Vec<&str> = out_b
            .iter()
            .map(|bar| bar.freq.as_deref().unwrap())
            .collect();
        assert_eq!(labels_a, labels_b, "相同周期数量下顺序必须确定");
        assert_eq!(
            labels_a,
            vec!["1h", "60min"],
            "按 freq_label 字典序 tie-break"
        );
    }
}
