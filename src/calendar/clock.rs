//! 本地时钟：UTC 纳秒 ↔ 本地时间。有 IANA 名称时随夏令时，否则用固定偏移。

use chrono::{DateTime, NaiveDateTime, TimeZone, Timelike, Utc};
use chrono_tz::Tz;

/// 探测夏令时"跳变缺口"时, 向前步进的分钟数上限(6 小时足够覆盖现实世界的
/// 任何春季跳变缺口, 通常缺口只有 1 小时, 极少数地区 30 分钟)。
const MAX_DST_GAP_PROBE_MINUTES: i64 = 360;

/// 引擎时区: 有 IANA 名称用名称(随 DST), 否则用固定偏移秒数。
#[derive(Debug, Clone, Default)]
pub(crate) struct LocalClock {
    tz: Option<Tz>,
    offset_secs: i32,
}

impl LocalClock {
    pub(crate) fn new(tz_name: Option<&str>, offset_secs: i32) -> Self {
        Self {
            tz: tz_name.and_then(|n| n.parse::<Tz>().ok()),
            offset_secs,
        }
    }

    /// 两个时钟是否表示同一时区：都有 IANA 名称时比名称，都没有时比固定偏移。
    pub(crate) fn same_zone(&self, other: &LocalClock) -> bool {
        match (&self.tz, &other.tz) {
            (Some(a), Some(b)) => a == b,
            (None, None) => self.offset_secs == other.offset_secs,
            _ => false,
        }
    }

    /// 时区描述（报错用）：IANA 名称，或 `UTC+HH:MM` 形式的固定偏移。
    pub(crate) fn describe(&self) -> String {
        match &self.tz {
            Some(tz) => tz.name().to_string(),
            None => {
                let sign = if self.offset_secs < 0 { '-' } else { '+' };
                let abs = self.offset_secs.unsigned_abs();
                format!("UTC{sign}{:02}:{:02}", abs / 3600, abs % 3600 / 60)
            }
        }
    }

    /// 时区名是否解析成功（`new` 对无法解析的名称静默退回固定偏移）。
    pub(crate) fn has_valid_tz(&self) -> bool {
        self.tz.is_some()
    }

    pub(crate) fn to_local(&self, ts_ns: i64) -> NaiveDateTime {
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

    pub(crate) fn from_local(&self, local: NaiveDateTime) -> i64 {
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
    pub(crate) fn ceil_label(&self, local: NaiveDateTime, period_secs: i64) -> i64 {
        let secs_of_day = i64::from(local.num_seconds_from_midnight());
        // `i64::div_ceil` (signed) is unstable on this toolchain; compute manually.
        let ceiled = ((secs_of_day + period_secs - 1) / period_secs) * period_secs;
        let midnight = local.date().and_hms_opt(0, 0, 0).expect("midnight");
        self.from_local(midnight + chrono::Duration::seconds(ceiled))
    }
}
