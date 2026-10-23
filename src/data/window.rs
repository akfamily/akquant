//! 运行时多周期窗口聚合器(引擎内).
//!
//! 把基础 bar 流按 (symbol, freq) 聚合成更高周期的窗口 bar。窗口 bar **不进 feed、
//! 不成为 Event**, 只写历史缓冲并交给策略阶段派发。
//!
//! 有会话解析器([`WindowAggregator::set_resolver`])时:
//! - 日内窗口(`Nmin`/`Nh`)按(交易日, 交易块)分组, 不跨块; 标签 = 本地时钟 ceil
//!   到周期, 但不超过块结束(A 股 1h: 10:00、11:00、11:30、14:00、15:00)。
//! - 日线窗口按交易日分组(夜盘 bar 归下一交易日), 标签 = 该交易日收盘,
//!   `interval_start` = 前一交易日收盘。
//! - 基础 bar 归属按区间内一点 `ts - 1` 判断。
//!
//! 没有解析器(或解析出错, 如日历越界)时退回本地时钟: 日内按 ceil 标签, 日线按
//! 本地日期、标签取当日最后一根基础 bar 的时间戳。
//!
//! 即时闭合: 基础周期已知(`base_interval_min`)或 bar 带 `interval_start` 时, bar 的
//! 区间终点到达窗口标签即闭合; 否则等下一根落入新窗口的 bar 到达才闭合。
use chrono::{Duration, NaiveDate, NaiveDateTime, NaiveTime};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::calendar::clock::LocalClock;
use crate::calendar::resolver::{SessionResolver, TradingBlock};
use crate::model::{AssetType, Bar};

const NS_PER_SEC: i64 = 1_000_000_000;

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
    /// 手写本地交易时段(按用户给定顺序); 给定则覆盖时段表。`end < start` 表示跨午夜。
    pub sessions: Vec<(NaiveTime, NaiveTime)>,
    /// "最后一段"在 `sessions` 中的下标: 不跨午夜的段中结束最晚者, 全部跨午夜时取
    /// 最后一项。日线窗口在这一段结束时即时闭合。
    last_session: Option<usize>,
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
            if e == s {
                return Err(format!(
                    "session_windows ({start}, {end}) 起点与终点相同, 时段长度为 0; \
                     跨午夜时段请把终点写得早于起点, 如 (\"21:00\", \"02:30\")"
                ));
            }
            parsed_sessions.push((s, e));
        }
        let last_session = Self::last_session_of(&parsed_sessions);
        Ok(Self {
            symbol,
            freq: parsed,
            freq_label: freq.trim().to_ascii_lowercase(),
            sessions: parsed_sessions,
            last_session,
        })
    }

    fn last_session_of(sessions: &[(NaiveTime, NaiveTime)]) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, (s, e)) in sessions.iter().enumerate() {
            if e > s && best.is_none_or(|b| *e > sessions[b].1) {
                best = Some(i);
            }
        }
        best.or_else(|| sessions.len().checked_sub(1))
    }

    fn matches(&self, symbol: &str) -> bool {
        self.symbol.as_deref().is_none_or(|s| s == symbol)
    }

    /// 本地时刻 `local`(基础 bar 区间内一点)所在的手写时段: (序号, 段起点, 段终点),
    /// 起止为本地时间(跨午夜段的终点在次日)。段外返回 `None`。
    fn session_at(&self, local: NaiveDateTime) -> Option<(i32, NaiveDateTime, NaiveDateTime)> {
        let t = local.time();
        let date = local.date();
        self.sessions.iter().enumerate().find_map(|(i, &(s, e))| {
            if s < e {
                (s <= t && t < e).then(|| (date.and_time(s), date.and_time(e)))
            } else if t >= s {
                Some((date.and_time(s), (date + Duration::days(1)).and_time(e)))
            } else if t < e {
                Some(((date - Duration::days(1)).and_time(s), date.and_time(e)))
            } else {
                None
            }
            .map(|(a, b)| (i as i32, a, b))
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowStateSnapshot {
    pub symbol: String,
    pub freq_label: String,
    pub label_ns: i64,
    /// 日内窗口的分组键 (交易日, 交易块序号或手写时段序号; 段外 -1)。
    /// 0.3.71 及更早的存档里是 (本地日期, 手写时段序号)。
    pub session_key: Option<(NaiveDate, i32)>,
    /// 日线窗口的分组键: 交易日(无解析器时为本地日期; 旧存档为本地日期)。
    pub local_date: NaiveDate,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    // 以下字段只能追加在末尾并带 `#[serde(default)]`(旧存档缺省)。
    /// 分组键里的块序号(与 `session_key` 第二项相同)。
    #[serde(default)]
    pub block_index: Option<i32>,
    /// 由会话解析器求得的交易日; 旧存档与无解析器时为 `None`。
    #[serde(default)]
    pub trading_day: Option<NaiveDate>,
    /// 窗口区间起点(UTC 纳秒)。
    #[serde(default)]
    pub interval_start: Option<i64>,
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
    trading_day: Option<NaiveDate>,
    interval_start: Option<i64>,
    open: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    volume: Decimal,
}

impl WindowState {
    fn open_from(bar: &Bar, freq: WindowFreq, place: &Placement) -> Self {
        Self {
            freq,
            label_ns: place.label,
            session_key: place.session_key,
            local_date: place.date,
            trading_day: place.trading_day,
            interval_start: place.interval_start,
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
            interval_start: self.interval_start,
        }
    }
}

/// 基础 bar 在某条订阅下的归属。
struct Placement {
    label: i64,
    session_key: Option<(NaiveDate, i32)>,
    /// 日线分组键(交易日或本地日期); 日内窗口仅记录。
    date: NaiveDate,
    trading_day: Option<NaiveDate>,
    interval_start: Option<i64>,
    /// 区间终点到达该时刻即闭合(`None` = 不即时闭合)。
    close_at: Option<i64>,
}

/// 品种所在交易日的区间缓存: 交易日 D 覆盖 `(lower, close]`。
#[derive(Debug, Clone, Copy)]
struct DaySpan {
    day: NaiveDate,
    lower: i64,
    close: i64,
    /// 日线窗口标签: 收盘时刻; 收盘恰为次日本地 0 点(如 ALWAYS_OPEN)时减 1ns,
    /// 与只有日期的日线改写(`_date_only_close_ns`)同口径, 标签仍落在交易日当天。
    label: i64,
    /// 前一交易日的窗口标签(同上规则), 作本交易日窗口的 `interval_start`。
    prev_label: Option<i64>,
}

/// 会话解析器句柄(`SessionResolver` 没有实现 `Debug`)。
#[derive(Clone)]
struct ResolverHandle(Arc<SessionResolver>);

impl std::fmt::Debug for ResolverHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SessionResolver")
    }
}

#[derive(Debug, Default)]
pub struct WindowAggregator {
    subscriptions: Vec<WindowSubscription>,
    base_interval_min: Option<u32>,
    clock: LocalClock,
    /// (symbol, freq_label) -> 在形成窗口
    states: HashMap<(String, String), WindowState>,
    resolver: Option<ResolverHandle>,
    asset_types: HashMap<String, AssetType>,
    /// 按品种缓存上一次命中的交易块。
    block_cache: HashMap<String, TradingBlock>,
    /// 按品种缓存上一次命中的交易日区间。
    day_cache: HashMap<String, DaySpan>,
}

/// bar 区间终点: 回测 tick 合成的终点戳 bar 打在边界前 1ns(`ts + 1` 是整秒),
/// 其区间终点为 `ts + 1`; 其余 bar(含区间未知的)为 `ts`。
fn bar_end(bar: &Bar) -> i64 {
    match bar.interval_start {
        Some(start) if start < bar.timestamp && (bar.timestamp + 1).rem_euclid(NS_PER_SEC) == 0 => {
            bar.timestamp + 1
        }
        _ => bar.timestamp,
    }
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
        self.block_cache.clear();
        self.day_cache.clear();
    }

    /// 设置会话解析器与各品种资产类型(决定时段表)。`None` 退回本地时钟切分。
    pub fn set_resolver(
        &mut self,
        resolver: Option<Arc<SessionResolver>>,
        asset_types: HashMap<String, AssetType>,
    ) {
        self.resolver = resolver.map(ResolverHandle);
        self.asset_types = asset_types;
        self.block_cache.clear();
        self.day_cache.clear();
    }

    /// 运行中补登记(或改写)单个品种的资产类型, 只作废该品种的区间缓存。
    pub fn set_asset_type(&mut self, symbol: &str, asset_type: AssetType) {
        if self.asset_types.get(symbol) == Some(&asset_type) {
            return;
        }
        self.asset_types.insert(symbol.to_string(), asset_type);
        self.block_cache.remove(symbol);
        self.day_cache.remove(symbol);
    }

    /// 品种是否已登记资产类型。
    pub fn has_asset_type(&self, symbol: &str) -> bool {
        self.asset_types.contains_key(symbol)
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
                         symbol={:?})但配置不一致(session_windows 或 symbol 范围不完全相同); \
                         请让重叠的订阅保持完全一致(session_windows 须逐项相同且顺序相同), 或改成互斥的 symbol 范围",
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

    /// `point` 所在的交易块(按品种缓存)。外层 `None` = 无解析器或解析出错(退回本地
    /// 时钟), 内层 `None` = 落在块外(午休、闭市)。
    fn block_for(&mut self, symbol: &str, point: i64) -> Option<Option<TradingBlock>> {
        if let Some(b) = self.block_cache.get(symbol)
            && b.start <= point
            && point < b.end
        {
            return Some(Some(*b));
        }
        let resolver = &self.resolver.as_ref()?.0;
        let asset_type = self.asset_types.get(symbol).copied();
        let block = resolver.block_at(symbol, asset_type, point).ok()?;
        if let Some(b) = block {
            self.block_cache.insert(symbol.to_string(), b);
        }
        Some(block)
    }

    /// `point` 所属的交易日区间(按品种缓存); 无解析器或解析出错返回 `None`。
    fn day_for(&mut self, symbol: &str, point: i64) -> Option<DaySpan> {
        if let Some(s) = self.day_cache.get(symbol)
            && s.lower < point
            && point <= s.close
        {
            return Some(*s);
        }
        let resolver = &self.resolver.as_ref()?.0;
        let asset_type = self.asset_types.get(symbol).copied();
        let day = resolver
            .resolve_instant(symbol, asset_type, point)
            .ok()?
            .trading_day;
        let (open, close) = resolver.day_bounds(symbol, asset_type, day).ok()?;
        let prev = resolver
            .previous_trading_day(symbol, asset_type, day)
            .ok()
            .and_then(|prev| {
                resolver
                    .day_bounds(symbol, asset_type, prev)
                    .ok()
                    .map(|(_, c)| (prev, c))
            });
        let span = DaySpan {
            day,
            // 前一交易日未知(日历首日)时只缓存开盘后的部分
            lower: prev.map_or(open - 1, |(_, c)| c),
            close,
            label: self.day_label(day, close),
            prev_label: prev.map(|(d, c)| self.day_label(d, c)),
        };
        self.day_cache.insert(symbol.to_string(), span);
        Some(span)
    }

    /// 交易日 `day` 的日线窗口标签: 收盘恰为次日本地 0 点时取前 1ns。
    fn day_label(&self, day: NaiveDate, close: i64) -> i64 {
        let next_midnight = (day + Duration::days(1))
            .and_hms_opt(0, 0, 0)
            .map(|t| self.clock.from_local(t));
        if next_midnight == Some(close) {
            close - 1
        } else {
            close
        }
    }

    /// 日内窗口(`period` 秒)的归属。
    fn place_intraday(
        &mut self,
        sub_idx: usize,
        symbol: &str,
        point: i64,
        local_ts: NaiveDateTime,
        period: i64,
    ) -> Placement {
        let period_ns = period * NS_PER_SEC;
        let clock_label = self.clock.ceil_label(local_ts, period);
        if self.subscriptions[sub_idx].sessions.is_empty() {
            match self.block_for(symbol, point) {
                Some(Some(block)) => {
                    return Placement {
                        label: clock_label.min(block.end),
                        session_key: Some((block.trading_day, block.index as i32)),
                        date: block.trading_day,
                        trading_day: Some(block.trading_day),
                        interval_start: Some((clock_label - period_ns).max(block.start)),
                        close_at: Some(clock_label.min(block.end)),
                    };
                }
                Some(None) => {
                    if let Some(span) = self.day_for(symbol, point) {
                        return Placement {
                            label: clock_label,
                            session_key: Some((span.day, -1)),
                            date: span.day,
                            trading_day: Some(span.day),
                            interval_start: Some(clock_label - period_ns),
                            close_at: Some(clock_label),
                        };
                    }
                }
                None => {}
            }
            return Placement {
                label: clock_label,
                session_key: None,
                date: local_ts.date(),
                trading_day: None,
                interval_start: Some(clock_label - period_ns),
                close_at: Some(clock_label),
            };
        }
        // 手写时段: 序号按用户列表, 交易日由解析器求(没有时取时段起点的本地日期)。
        let local_point = self.clock.to_local(point);
        let hit = self.subscriptions[sub_idx].session_at(local_point);
        let span = self.day_for(symbol, point);
        match hit {
            Some((idx, start, end)) => {
                let start_ns = self.clock.from_local(start);
                let end_ns = self.clock.from_local(end);
                let day = span.map_or(start.date(), |s| s.day);
                let label = clock_label.min(end_ns);
                Placement {
                    label,
                    session_key: Some((day, idx)),
                    date: day,
                    trading_day: span.map(|s| s.day),
                    interval_start: Some((clock_label - period_ns).max(start_ns)),
                    close_at: Some(label),
                }
            }
            None => {
                let day = span.map_or(local_point.date(), |s| s.day);
                Placement {
                    label: clock_label,
                    session_key: Some((day, -1)),
                    date: day,
                    trading_day: span.map(|s| s.day),
                    interval_start: Some(clock_label - period_ns),
                    close_at: Some(clock_label),
                }
            }
        }
    }

    /// 日线窗口的归属。
    fn place_daily(
        &mut self,
        sub_idx: usize,
        bar: &Bar,
        point: i64,
        local_ts: NaiveDateTime,
    ) -> Placement {
        let span = self.day_for(&bar.symbol, point);
        // 无解析器时按 bar 时间戳的本地日期分组(旧行为)。
        let date = span.map_or(local_ts.date(), |s| s.day);
        let sub = &self.subscriptions[sub_idx];
        // 手写时段: bar 落在"最后一段"内时, 在该段这一次的结束时刻闭合。段终点按段
        // 起点所在日期换算(跨午夜段的终点在起点次日), 不按交易日 D 的日期。
        let manual = sub.last_session.map(|last| {
            sub.session_at(self.clock.to_local(point))
                .filter(|(idx, _, _)| *idx as usize == last)
                .map(|(_, _, end)| self.clock.from_local(end))
        });
        match span {
            Some(s) => Placement {
                label: s.label,
                session_key: None,
                date,
                trading_day: Some(s.day),
                interval_start: s.prev_label,
                // 闭合判据用收盘本身(不是减 1ns 的标签), 次日 0 点那根 bar 也能闭合。
                close_at: manual.unwrap_or(Some(s.close)),
            },
            // 无解析器: 本地日期分组, 标签 = 当日最后一根基础 bar 的时间戳(旧行为)。
            None => Placement {
                label: bar.timestamp,
                session_key: None,
                date,
                trading_day: None,
                interval_start: None,
                close_at: manual.flatten(),
            },
        }
    }

    /// 喂入一根基础 bar, 返回本步闭合的窗口 bar(按周期从小到大)。
    pub fn update(&mut self, bar: &Bar) -> Vec<Bar> {
        if self.subscriptions.is_empty() {
            return Vec::new();
        }
        let local_ts = self.clock.to_local(bar.timestamp);
        // 区间内一点: 终点戳 bar 的 ts 是区间右端, 归属看 ts 前 1ns。
        let point = bar.timestamp.saturating_sub(1);
        let end = bar_end(bar);
        let base_known = self.base_interval_min.is_some() || bar.interval_start.is_some();
        // (总分钟数, freq_label, bar); label 作 tie-break, 保证 "1h" 与 "60min"
        // 这类分钟数相同但标签不同的周期有确定顺序(否则顺序依赖 HashMap 迭代,
        // 对 golden baseline 是隐患)。
        let mut closed: Vec<(u32, String, Bar)> = Vec::new();

        for idx in 0..self.subscriptions.len() {
            if !self.subscriptions[idx].matches(&bar.symbol) {
                continue;
            }
            let freq = self.subscriptions[idx].freq;
            let freq_label = self.subscriptions[idx].freq_label.clone();
            let daily = freq.period_seconds().is_none();
            let place = match freq.period_seconds() {
                Some(period) => self.place_intraday(idx, &bar.symbol, point, local_ts, period),
                None => self.place_daily(idx, bar, point, local_ts),
            };
            let key = (bar.symbol.clone(), freq_label.clone());
            if let Some(st) = self.states.get(&key) {
                let changed = if daily {
                    st.local_date != place.date
                } else {
                    st.label_ns != place.label || st.session_key != place.session_key
                };
                if changed {
                    closed.push((
                        freq.total_minutes(),
                        freq_label.clone(),
                        st.to_bar(&bar.symbol, &freq_label),
                    ));
                    self.states.remove(&key);
                }
            }
            let st = self
                .states
                .entry(key.clone())
                .and_modify(|s| {
                    s.merge(bar);
                    if daily {
                        // 日线标签与起点随交易日确定; 无解析器时标签跟随最后一根 bar。
                        s.label_ns = place.label;
                        s.interval_start = place.interval_start;
                    }
                })
                .or_insert_with(|| WindowState::open_from(bar, freq, &place));
            // 即时闭合: bar 区间 (point, end] 到达闭合时刻。日线手写时段且无解析器时,
            // 要求区间覆盖闭合时刻, 免得同一本地日期收盘后的 bar 一来就闭合。
            let reaches = place.close_at.is_some_and(|c| end >= c && point < c);
            if base_known && reaches {
                closed.push((
                    freq.total_minutes(),
                    freq_label.clone(),
                    st.to_bar(&bar.symbol, &freq_label),
                ));
                self.states.remove(&key);
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
                block_index: st.session_key.map(|k| k.1),
                trading_day: st.trading_day,
                interval_start: st.interval_start,
            })
            .collect();
        states.sort_by(|a, b| {
            a.symbol
                .cmp(&b.symbol)
                .then(a.freq_label.cmp(&b.freq_label))
        });
        WindowAggregatorSnapshot { states }
    }

    /// 恢复在形成窗口。旧存档(无 `trading_day`)的分组键按原样恢复: 与新规则下的键
    /// 不一致时, 该窗口在下一根 bar 到达时闭合(不报错)。
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
                    trading_day: s.trading_day,
                    interval_start: s.interval_start,
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
            interval_start: None,
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

    fn resolved_agg(
        subs: &[(&str, Option<Vec<(String, String)>>)],
        sessions: Vec<(&str, &str)>,
        symbol: &str,
        asset_type: AssetType,
    ) -> WindowAggregator {
        let mut a = agg(subs, Some(1));
        let clock = LocalClock::new(Some("Asia/Shanghai"), 28800);
        let sessions = sessions
            .into_iter()
            .map(|(k, t)| {
                (
                    k.to_string(),
                    crate::calendar::schedule::template(t).unwrap(),
                )
            })
            .collect();
        let resolver = SessionResolver::new(clock, None, sessions).unwrap();
        let mut types = HashMap::new();
        types.insert(symbol.to_string(), asset_type);
        a.set_resolver(Some(Arc::new(resolver)), types);
        a
    }

    #[test]
    fn stock_hour_window_clamps_at_lunch_and_closes_immediately() {
        let mut a = resolved_agg(&[("1h", None)], vec![], "S", AssetType::Stock);
        assert!(a.update(&bar(ns(2024, 1, 2, 11, 1), 10, "S")).is_empty());
        // 11:01～11:30 的窗口标签截到块结束 11:30, 在 11:30 那根即时闭合
        let closed = a.update(&bar(ns(2024, 1, 2, 11, 30), 11, "S"));
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].timestamp, ns(2024, 1, 2, 11, 30));
        assert_eq!(closed[0].interval_start, Some(ns(2024, 1, 2, 11, 0)));
        // 下午块: 13:01 开新窗口, 起点为块起点 13:00
        assert!(a.update(&bar(ns(2024, 1, 2, 13, 1), 12, "S")).is_empty());
        let cur = a.current("S", "1h").unwrap();
        assert_eq!(cur.timestamp, ns(2024, 1, 2, 14, 0));
        assert_eq!(cur.interval_start, Some(ns(2024, 1, 2, 13, 0)));
    }

    #[test]
    fn night_futures_windows_group_by_trading_day_across_midnight() {
        let mut a = resolved_agg(
            &[("1d", None), ("1h", None)],
            vec![("au", "CN_FUTURES_NIGHT_0230")],
            "au2406",
            AssetType::Futures,
        );
        // 周五日盘收盘: 日线(交易日 01-05)在 15:00 即时闭合
        let closed = a.update(&bar(ns(2024, 1, 5, 15, 0), 10, "au2406"));
        let day = closed
            .iter()
            .find(|b| b.freq.as_deref() == Some("1d"))
            .unwrap();
        assert_eq!(day.timestamp, ns(2024, 1, 5, 15, 0));
        assert_eq!(day.interval_start, Some(ns(2024, 1, 4, 15, 0)));
        // 周五夜盘跨午夜到周六 02:30: 同一块, 1h 窗口在 00:00 闭合, 不因换自然日断开
        a.update(&bar(ns(2024, 1, 5, 23, 30), 11, "au2406"));
        let closed = a.update(&bar(ns(2024, 1, 6, 0, 0), 12, "au2406"));
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].freq.as_deref(), Some("1h"));
        a.update(&bar(ns(2024, 1, 6, 2, 30), 13, "au2406"));
        // 周一日盘: 与周五夜盘同属交易日 01-08, 日线不在此处闭合
        let closed = a.update(&bar(ns(2024, 1, 8, 9, 1), 14, "au2406"));
        assert!(closed.iter().all(|b| b.freq.as_deref() != Some("1d")));
        let closed = a.update(&bar(ns(2024, 1, 8, 15, 0), 15, "au2406"));
        let day = closed
            .iter()
            .find(|b| b.freq.as_deref() == Some("1d"))
            .unwrap();
        assert_eq!(day.timestamp, ns(2024, 1, 8, 15, 0));
        assert_eq!(day.interval_start, Some(ns(2024, 1, 5, 15, 0)));
        assert_eq!(day.open, Decimal::from(11), "开盘取周五夜盘第一根");
        assert_eq!(day.close, Decimal::from(15));
    }

    #[test]
    fn tick_synthesized_bar_closes_window_at_boundary() {
        // 回测 tick 合成的终点戳 bar: ts = 边界 - 1ns, interval_start = 起点 - 1ns;
        // 基础周期未配置时也按 bar 区间即时闭合。
        let mut a = agg(&[("5min", None)], None);
        let mut synth = |m: u32| {
            let mut b = bar(ns(2024, 1, 2, 9, m) - 1, 10, "X");
            b.interval_start = Some(ns(2024, 1, 2, 9, m - 1) - 1);
            a.update(&b)
        };
        // synth(m) = 区间 [m-1, m) 的合成 bar; 09:30～09:35 窗口含 synth(31..=35)
        for m in 31..=34 {
            assert!(synth(m).is_empty(), "minute {m} must not close");
        }
        let closed = synth(35);
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].timestamp, ns(2024, 1, 2, 9, 35));
        assert_eq!(closed[0].volume, Decimal::from(50));
    }

    #[test]
    fn manual_sessions_accept_cross_midnight_and_pick_last_session() {
        let sessions = vec![
            ("21:00".to_string(), "02:30".to_string()),
            ("09:00".to_string(), "10:15".to_string()),
            ("13:30".to_string(), "15:00".to_string()),
            ("10:30".to_string(), "11:30".to_string()),
        ];
        let sub = WindowSubscription::parse(None, "1d", Some(sessions)).unwrap();
        assert_eq!(sub.last_session, Some(2), "不跨午夜的段中结束最晚者");
        let err = WindowSubscription::parse(
            None,
            "1d",
            Some(vec![("09:00".to_string(), "09:00".to_string())]),
        )
        .unwrap_err();
        assert!(err.contains("长度为 0"), "{err}");
        let only_night = WindowSubscription::parse(
            None,
            "1d",
            Some(vec![("21:00".to_string(), "02:30".to_string())]),
        )
        .unwrap();
        assert_eq!(only_night.last_session, Some(0));
        // 跨午夜段: 00:30 归入前一日 21:00 起的那一段
        let local = NaiveDate::from_ymd_opt(2024, 1, 6)
            .unwrap()
            .and_hms_opt(0, 30, 0)
            .unwrap();
        let (idx, start, end) = sub.session_at(local).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(start.date(), NaiveDate::from_ymd_opt(2024, 1, 5).unwrap());
        assert_eq!(end.date(), NaiveDate::from_ymd_opt(2024, 1, 6).unwrap());
    }

    #[test]
    fn legacy_snapshot_without_new_fields_restores_and_closes_on_key_change() {
        // 0.3.71 存档的窗口状态只有前 11 个字段(rmp_serde 按数组编码)
        #[derive(Serialize)]
        struct LegacyState {
            symbol: String,
            freq_label: String,
            label_ns: i64,
            session_key: Option<(NaiveDate, i32)>,
            local_date: NaiveDate,
            open: Decimal,
            high: Decimal,
            low: Decimal,
            close: Decimal,
            volume: Decimal,
        }
        #[derive(Serialize)]
        struct LegacySnapshot {
            states: Vec<LegacyState>,
        }
        let legacy = LegacySnapshot {
            states: vec![LegacyState {
                symbol: "S".to_string(),
                freq_label: "5min".to_string(),
                label_ns: ns(2024, 1, 2, 9, 35),
                session_key: None,
                local_date: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
                open: Decimal::from(7),
                high: Decimal::from(8),
                low: Decimal::from(6),
                close: Decimal::from(7),
                volume: Decimal::from(10),
            }],
        };
        let bytes = rmp_serde::to_vec(&legacy).unwrap();
        let decoded: WindowAggregatorSnapshot = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.states[0].trading_day, None);
        assert_eq!(decoded.states[0].interval_start, None);
        let mut a = resolved_agg(&[("5min", None)], vec![], "S", AssetType::Stock);
        a.restore(decoded);
        // 新键为 (交易日, 块 0), 与旧键 None 不一致: 下一根 bar 到达时闭合旧窗口, 不报错
        let closed = a.update(&bar(ns(2024, 1, 2, 9, 33), 10, "S"));
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].open, Decimal::from(7));
        assert_eq!(closed[0].timestamp, ns(2024, 1, 2, 9, 35));
        // 新窗口带新键, 快照往返保留新字段
        let snap = a.snapshot();
        assert_eq!(
            snap.states[0].trading_day,
            NaiveDate::from_ymd_opt(2024, 1, 2)
        );
        assert_eq!(snap.states[0].block_index, Some(0));
        assert_eq!(snap.states[0].interval_start, Some(ns(2024, 1, 2, 9, 30)));
    }

    #[test]
    fn all_cross_midnight_manual_daily_closes_at_segment_end_after_start_date() {
        // 只有夜盘一段: 交易日 01-08(周一)的夜盘从周五 21:00 到周六 02:30,
        // 闭合时刻按段起点日期换算为周六 02:30, 而不是交易日 D(周一)的 02:30。
        let night = Some(vec![("21:00".to_string(), "02:30".to_string())]);
        let mut a = resolved_agg(
            &[("1d", night)],
            vec![("au", "CN_FUTURES_NIGHT_0230")],
            "au2406",
            AssetType::Futures,
        );
        assert!(
            a.update(&bar(ns(2024, 1, 5, 21, 30), 10, "au2406"))
                .is_empty()
        );
        let closed = a.update(&bar(ns(2024, 1, 6, 2, 30), 11, "au2406"));
        assert_eq!(closed.len(), 1, "周六 02:30 夜盘结束即闭合");
        assert_eq!(closed[0].timestamp, ns(2024, 1, 8, 15, 0));
        assert_eq!(closed[0].close, Decimal::from(11));
    }

    #[test]
    fn always_open_daily_label_is_one_ns_before_next_midnight() {
        let mut a = WindowAggregator::with_clock(Some("America/New_York"), 0);
        a.configure(
            vec![WindowSubscription::parse(None, "1d", None).unwrap()],
            Some(1),
        )
        .unwrap();
        let clock = LocalClock::new(Some("America/New_York"), 0);
        let ny = |d: u32, h: u32, m: u32| {
            chrono_tz::America::New_York
                .with_ymd_and_hms(2024, 1, d, h, m, 0)
                .unwrap()
                .timestamp_nanos_opt()
                .unwrap()
        };
        let resolver = SessionResolver::new(
            clock,
            None,
            crate::calendar::resolver::effective_sessions(None, Some("America/New_York"), 0),
        )
        .unwrap();
        a.set_resolver(Some(Arc::new(resolver)), HashMap::new());
        assert!(a.update(&bar(ny(2, 15, 0), 10, "AAPL")).is_empty());
        // 00:00 那根 bar 的区间终点恰为收盘(次日 0 点): 按收盘判据即时闭合
        let closed = a.update(&bar(ny(3, 0, 0), 11, "AAPL"));
        assert_eq!(closed.len(), 1);
        assert_eq!(
            closed[0].timestamp,
            ny(3, 0, 0) - 1,
            "标签 = 01-02 23:59:59.999999999"
        );
        assert_eq!(closed[0].interval_start, Some(ny(2, 0, 0) - 1));
        assert_eq!(closed[0].close, Decimal::from(11));
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
