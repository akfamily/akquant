//! 交易时段表：一个交易日内的若干时段，夜盘锚定到前一个交易日。

use crate::model::TradingSession;

/// 一天的秒数。
const DAY_SECS: u32 = 86_400;

/// 时段的锚定日期。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Anchor {
    /// 前一个交易日的自然日（夜盘：周一交易日的夜盘在上周五晚上）。
    PrevTradingDay,
    /// 交易日当天。
    TradingDay,
}

/// 一个时段：`[start_sec, end_sec)`，秒数相对锚定自然日的本地 0 点。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionSegment {
    pub anchor: Anchor,
    pub start_sec: u32,
    /// 可大于 86400，表示跨过午夜（如 21:00 至次日 02:30 写作 21:00-26:30）。
    pub end_sec: u32,
    pub session: TradingSession,
}

/// 用户未传交易日历时的缺省规则。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackCalendar {
    /// 周一至周五为交易日。
    Weekdays,
    /// 每个自然日都是交易日（全天候市场）。
    AllDays,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSchedule {
    pub name: String,
    pub segments: Vec<SessionSegment>,
    pub fallback_calendar: FallbackCalendar,
}

impl SessionSchedule {
    /// 校验并构造：至少一段；每段 start < end 且不超过 24 小时；按 (锚定, 开始) 排序后不重叠。
    pub fn new(
        name: impl Into<String>,
        mut segments: Vec<SessionSegment>,
        fallback_calendar: FallbackCalendar,
    ) -> Result<Self, String> {
        let name = name.into();
        if segments.is_empty() {
            return Err(format!("时段表 {name:?} 至少需要一个时段"));
        }
        segments.sort_by_key(|s| (s.anchor, s.start_sec));
        for seg in &segments {
            if seg.start_sec >= seg.end_sec || seg.end_sec - seg.start_sec > DAY_SECS {
                return Err(format!(
                    "时段表 {name:?} 的时段 {}-{} 无效：结束须晚于开始且不超过 24 小时",
                    fmt_clock(seg.start_sec),
                    fmt_clock(seg.end_sec)
                ));
            }
        }
        for pair in segments.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if a.anchor == b.anchor && a.end_sec > b.start_sec {
                return Err(format!(
                    "时段表 {name:?} 的时段 {}-{} 与 {}-{} 重叠",
                    fmt_clock(a.start_sec),
                    fmt_clock(a.end_sec),
                    fmt_clock(b.start_sec),
                    fmt_clock(b.end_sec)
                ));
            }
        }
        Ok(Self {
            name,
            segments,
            fallback_calendar,
        })
    }

    /// 是否有锚定到前一交易日的时段（夜盘）。
    pub fn has_night(&self) -> bool {
        self.segments
            .iter()
            .any(|s| s.anchor == Anchor::PrevTradingDay)
    }
}

fn fmt_clock(secs: u32) -> String {
    format!("{:02}:{:02}", secs / 3600, (secs % 3600) / 60)
}

/// 解析 `HH:MM` 或 `HH:MM:SS`；小时允许到 47（跨午夜写 25:00、26:30）。
pub fn parse_clock(s: &str) -> Result<u32, String> {
    let bad = || format!("时刻格式应为 HH:MM 或 HH:MM:SS，收到 {s:?}");
    let parts: Vec<&str> = s.trim().split(':').collect();
    if !(2..=3).contains(&parts.len()) {
        return Err(bad());
    }
    let mut nums = [0u32; 3];
    for (i, p) in parts.iter().enumerate() {
        nums[i] = p.parse::<u32>().map_err(|_| bad())?;
    }
    let (h, m, sec) = (nums[0], nums[1], nums[2]);
    if h >= 48 || m >= 60 || sec >= 60 {
        return Err(format!(
            "时刻越界：小时须小于 48（跨午夜写 25:00 这类），分、秒须小于 60，收到 {s:?}"
        ));
    }
    Ok(h * 3600 + m * 60 + sec)
}

/// 解析时段类型名（不区分大小写，接受 snake_case）。
pub fn parse_session(s: &str) -> Result<TradingSession, String> {
    let key: String = s.trim().to_ascii_lowercase().replace('_', "");
    Ok(match key.as_str() {
        "continuous" => TradingSession::Continuous,
        "openingauction" => TradingSession::OpeningAuction,
        "auctionhold" => TradingSession::AuctionHold,
        "closingauction" => TradingSession::ClosingAuction,
        "break" => TradingSession::Break,
        "closed" => TradingSession::Closed,
        "postclose" => TradingSession::PostClose,
        _ => {
            return Err(format!(
                "未知时段类型 {s:?}，可选：continuous / opening_auction / auction_hold / \
                 closing_auction / break / closed / post_close"
            ));
        }
    })
}

fn seg(anchor: Anchor, start: &str, end: &str, session: TradingSession) -> SessionSegment {
    SessionSegment {
        anchor,
        start_sec: parse_clock(start).expect("内置模板时刻"),
        end_sec: parse_clock(end).expect("内置模板时刻"),
        session,
    }
}

/// 内置模板名。
pub const TEMPLATE_NAMES: &[&str] = &[
    "CN_STOCK",
    "CN_FUND",
    "CN_OPTION",
    "CN_FUTURES_DAY",
    "CN_FUTURES_COMMODITY_DAY",
    "CN_FUTURES_CFFEX_STOCK_INDEX_DAY",
    "CN_FUTURES_CFFEX_BOND_DAY",
    "CN_FUTURES_NIGHT_23",
    "CN_FUTURES_NIGHT_01",
    "CN_FUTURES_NIGHT_0230",
    "ALWAYS_OPEN",
];

/// A 股、场内基金、上交所 ETF 期权的日内时段。
fn cn_exchange_day() -> Vec<SessionSegment> {
    use Anchor::TradingDay as D;
    use TradingSession as S;
    vec![
        seg(D, "09:15", "09:25", S::OpeningAuction),
        seg(D, "09:25", "09:30", S::AuctionHold),
        seg(D, "09:30", "11:30", S::Continuous),
        seg(D, "11:30", "13:00", S::Break),
        seg(D, "13:00", "14:57", S::Continuous),
        seg(D, "14:57", "15:00", S::ClosingAuction),
    ]
}

/// 商品期货日盘连续交易段（不含集合竞价）。
fn futures_day_continuous() -> Vec<SessionSegment> {
    use Anchor::TradingDay as D;
    use TradingSession as S;
    vec![
        seg(D, "09:00", "10:15", S::Continuous),
        seg(D, "10:15", "10:30", S::Break),
        seg(D, "10:30", "11:30", S::Continuous),
        seg(D, "11:30", "13:30", S::Break),
        seg(D, "13:30", "15:00", S::Continuous),
    ]
}

/// 有夜盘的商品期货：开盘集合竞价在夜盘前（20:55-20:59 报单，20:59-21:00 撮合）。
fn futures_night(night_end: &str) -> Vec<SessionSegment> {
    use Anchor::PrevTradingDay as P;
    use TradingSession as S;
    let mut v = vec![
        seg(P, "20:55", "20:59", S::OpeningAuction),
        seg(P, "20:59", "21:00", S::AuctionHold),
        seg(P, "21:00", night_end, S::Continuous),
    ];
    v.extend(futures_day_continuous());
    v
}

/// 中金所：股指期货收盘 15:00，国债期货收盘 15:15。
fn cffex_day(close: &str) -> Vec<SessionSegment> {
    use Anchor::TradingDay as D;
    use TradingSession as S;
    vec![
        seg(D, "09:25", "09:29", S::OpeningAuction),
        seg(D, "09:29", "09:30", S::AuctionHold),
        seg(D, "09:30", "11:30", S::Continuous),
        seg(D, "11:30", "13:00", S::Break),
        seg(D, "13:00", close, S::Continuous),
    ]
}

/// 按名称取内置模板（大小写不敏感）。
pub fn template(name: &str) -> Option<SessionSchedule> {
    use Anchor::TradingDay as D;
    use TradingSession as S;
    let upper = name.trim().to_ascii_uppercase();
    let (segments, fallback) = match upper.as_str() {
        "CN_STOCK" | "CN_FUND" | "CN_OPTION" => (cn_exchange_day(), FallbackCalendar::Weekdays),
        "CN_FUTURES_DAY" | "CN_FUTURES_COMMODITY_DAY" => {
            let mut v = vec![
                seg(D, "08:55", "08:59", S::OpeningAuction),
                seg(D, "08:59", "09:00", S::AuctionHold),
            ];
            v.extend(futures_day_continuous());
            (v, FallbackCalendar::Weekdays)
        }
        "CN_FUTURES_CFFEX_STOCK_INDEX_DAY" => (cffex_day("15:00"), FallbackCalendar::Weekdays),
        "CN_FUTURES_CFFEX_BOND_DAY" => (cffex_day("15:15"), FallbackCalendar::Weekdays),
        "CN_FUTURES_NIGHT_23" => (futures_night("23:00"), FallbackCalendar::Weekdays),
        "CN_FUTURES_NIGHT_01" => (futures_night("25:00"), FallbackCalendar::Weekdays),
        "CN_FUTURES_NIGHT_0230" => (futures_night("26:30"), FallbackCalendar::Weekdays),
        "ALWAYS_OPEN" => (
            vec![seg(D, "00:00", "24:00", S::Continuous)],
            FallbackCalendar::AllDays,
        ),
        _ => return None,
    };
    Some(SessionSchedule::new(upper, segments, fallback).expect("内置模板必须合法"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_template_name_builds() {
        for name in TEMPLATE_NAMES {
            let s = template(name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(s.name, *name);
        }
        assert!(template("cn_stock").is_some(), "大小写不敏感");
        assert!(template("NOPE").is_none());
    }

    #[test]
    fn night_templates_have_night_and_cross_midnight() {
        let s = template("CN_FUTURES_NIGHT_0230").unwrap();
        assert!(s.has_night());
        let night = s
            .segments
            .iter()
            .find(|x| x.anchor == Anchor::PrevTradingDay && x.session == TradingSession::Continuous)
            .unwrap();
        assert_eq!(night.start_sec, 21 * 3600);
        assert_eq!(night.end_sec, 26 * 3600 + 30 * 60);
        assert!(!template("CN_STOCK").unwrap().has_night());
    }

    #[test]
    fn parse_clock_accepts_cross_midnight_and_rejects_garbage() {
        assert_eq!(parse_clock("09:30").unwrap(), 9 * 3600 + 30 * 60);
        assert_eq!(parse_clock("15:00:01").unwrap(), 15 * 3600 + 1);
        assert_eq!(parse_clock("25:00").unwrap(), 25 * 3600);
        assert!(parse_clock("48:00").is_err());
        assert!(parse_clock("9").is_err());
        assert!(parse_clock("aa:bb").is_err());
    }

    #[test]
    fn new_rejects_overlap_and_inverted() {
        let a = SessionSegment {
            anchor: Anchor::TradingDay,
            start_sec: 3600,
            end_sec: 7200,
            session: TradingSession::Continuous,
        };
        let overlap = SessionSegment {
            start_sec: 5400,
            end_sec: 9000,
            ..a
        };
        assert!(SessionSchedule::new("x", vec![a, overlap], FallbackCalendar::Weekdays).is_err());
        let inverted = SessionSegment {
            start_sec: 7200,
            end_sec: 3600,
            ..a
        };
        assert!(SessionSchedule::new("x", vec![inverted], FallbackCalendar::Weekdays).is_err());
        assert!(SessionSchedule::new("x", vec![], FallbackCalendar::Weekdays).is_err());
    }

    #[test]
    fn parse_session_names() {
        assert_eq!(
            parse_session("Opening_Auction").unwrap(),
            TradingSession::OpeningAuction
        );
        assert_eq!(
            parse_session("continuous").unwrap(),
            TradingSession::Continuous
        );
        assert!(parse_session("normal").is_err());
    }
}
