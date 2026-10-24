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
                let scope = match a.anchor {
                    Anchor::PrevTradingDay => "夜盘（锚定前一交易日）",
                    Anchor::TradingDay => "交易日当天",
                };
                return Err(format!(
                    "时段表 {name:?} 的时段 {}-{} 与 {}-{} 重叠：同属{scope}的时段不得重叠，\
                     请调整其中一段的起止使其不重叠（如前一段的结束不晚于后一段的开始）",
                    fmt_clock(a.start_sec),
                    fmt_end_clock(a.end_sec),
                    fmt_clock(b.start_sec),
                    fmt_end_clock(b.end_sec)
                ));
            }
        }
        // 跨锚定：夜盘段平移到交易日坐标（相对交易日 0 点减 24 小时）后不得与当日段相交；
        // 夜盘段本身位于前一交易日的日历日上，不平移时也不得与（前一交易日的）日盘段相交。
        for night in segments
            .iter()
            .filter(|s| s.anchor == Anchor::PrevTradingDay)
        {
            let ns = i64::from(night.start_sec) - i64::from(DAY_SECS);
            let ne = i64::from(night.end_sec) - i64::from(DAY_SECS);
            for day in segments.iter().filter(|s| s.anchor == Anchor::TradingDay) {
                if ns < i64::from(day.end_sec) && i64::from(day.start_sec) < ne {
                    return Err(format!(
                        "时段表 {name:?} 的夜盘时段 {}-{}（锚定前一交易日）与当日时段 {}-{} \
                         重叠：夜盘跨午夜后不得与交易日当天的时段重叠，请调整其中一段的起止",
                        fmt_clock(night.start_sec),
                        fmt_end_clock(night.end_sec),
                        fmt_clock(day.start_sec),
                        fmt_end_clock(day.end_sec)
                    ));
                }
                if night.start_sec < day.end_sec && day.start_sec < night.end_sec {
                    return Err(format!(
                        "时段表 {name:?} 的夜盘时段 {}-{}（锚定前一交易日）与前一交易日的日盘时段 \
                         {}-{} 重叠：夜盘段位于前一交易日的日历日上，不得与该日的日盘时段重叠，\
                         请调整其中一段的起止使其不重叠",
                        fmt_clock(night.start_sec),
                        fmt_end_clock(night.end_sec),
                        fmt_clock(day.start_sec),
                        fmt_end_clock(day.end_sec)
                    ));
                }
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

/// 结束时刻按用户写法显示：跨午夜的 26:30 显示为 02:30，24:00 保持 24:00。
fn fmt_end_clock(secs: u32) -> String {
    fmt_clock(if secs > DAY_SECS {
        secs - DAY_SECS
    } else {
        secs
    })
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

/// 自定义时段表的一段：`(start, end, session[, anchor])`，原样保存用户写法（随存档保存）。
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CustomSegment {
    pub start: String,
    pub end: String,
    pub session: String,
    /// `"prev"`（前一交易日，夜盘）或 `"day"`（交易日当天）；None 按开始时刻推断。
    pub anchor: Option<String>,
}

/// 不写锚定时，开始时刻不早于它的段视为夜盘（锚定前一交易日）。
const NIGHT_START_SECS: u32 = 18 * 3600;

/// 自定义时段表里的时刻：严格两位 `HH:MM`（不带秒、不留空格），取值 00:00～24:00。
fn parse_custom_clock(name: &str, idx: usize, field: &str, s: &str) -> Result<u32, String> {
    let bad = || {
        format!(
            "sessions[{name:?}] 第 {idx} 段的{field} {s:?} 无效：须为两位小时、两位分钟的 HH:MM \
             格式（如 \"09:30\"，不带秒、不留空格），取值 00:00～24:00；跨午夜的夜盘请写成 \
             (\"21:00\", \"02:30\", ...)"
        )
    };
    let b = s.as_bytes();
    let well_formed =
        b.len() == 5 && b[2] == b':' && [0, 1, 3, 4].iter().all(|&i| b[i].is_ascii_digit());
    if !well_formed {
        return Err(bad());
    }
    let num = |i: usize| u32::from(b[i] - b'0') * 10 + u32::from(b[i + 1] - b'0');
    let (h, m) = (num(0), num(3));
    let secs = h * 3600 + m * 60;
    if m >= 60 || secs > DAY_SECS {
        return Err(bad());
    }
    Ok(secs)
}

/// 由用户写的段序列构造自定义时段表（缺省日历为工作日）。
///
/// `name` 为 `sessions` 的键，用于报错与时段表名。规则：`closed` 不允许写（段外即闭市）；
/// 未给锚定时 `start >= 18:00` 的段锚定前一交易日（夜盘）；`end <= start` 表示跨午夜，
/// 只允许夜盘段。
pub fn schedule_from_custom(name: &str, segs: &[CustomSegment]) -> Result<SessionSchedule, String> {
    if segs.is_empty() {
        return Err(format!(
            "sessions[{name:?}] 至少需要一个时段，如 [(\"09:30\", \"11:30\", \"continuous\")]"
        ));
    }
    let mut segments = Vec::with_capacity(segs.len());
    for (i, seg) in segs.iter().enumerate() {
        let idx = i + 1;
        let start = parse_custom_clock(name, idx, "开始时刻", &seg.start)?;
        if start >= DAY_SECS {
            return Err(format!(
                "sessions[{name:?}] 第 {idx} 段的开始时刻 {:?} 无效：须为 HH:MM 格式且早于 24:00",
                seg.start
            ));
        }
        let mut end = parse_custom_clock(name, idx, "结束时刻", &seg.end)?;
        let session = parse_session(&seg.session)
            .map_err(|e| format!("sessions[{name:?}] 第 {idx} 段：{e}"))?;
        if session == TradingSession::Closed {
            return Err(format!(
                "sessions[{name:?}] 第 {idx} 段的时段类型不能是 closed：段外即闭市，不要写 closed"
            ));
        }
        let anchor = match seg.anchor.as_deref().map(|a| a.trim().to_ascii_lowercase()) {
            None => {
                if start >= NIGHT_START_SECS {
                    Anchor::PrevTradingDay
                } else {
                    Anchor::TradingDay
                }
            }
            Some(a) if a == "prev" => Anchor::PrevTradingDay,
            Some(a) if a == "day" => Anchor::TradingDay,
            Some(_) => {
                return Err(format!(
                    "sessions[{name:?}] 第 {idx} 段的锚定 {:?} 无效：只能写 \"prev\"（前一交易日，\
                     夜盘）或 \"day\"（交易日当天），也可不写（开始不早于 18:00 视为夜盘）",
                    seg.anchor.as_deref().unwrap_or_default()
                ));
            }
        };
        if end == start {
            return Err(format!(
                "sessions[{name:?}] 第 {idx} 段 {}-{} 长度为 0：结束时刻须晚于开始时刻",
                seg.start, seg.end
            ));
        }
        if end < start {
            if anchor != Anchor::PrevTradingDay {
                return Err(format!(
                    "sessions[{name:?}] 第 {idx} 段 {}-{} 长度无效：结束早于开始表示跨午夜，\
                     只允许夜盘段（开始不早于 18:00，或第 4 项写 \"prev\"）；日盘段请让结束晚于开始",
                    seg.start, seg.end
                ));
            }
            end += DAY_SECS;
        }
        segments.push(SessionSegment {
            anchor,
            start_sec: start,
            end_sec: end,
            session,
        });
    }
    SessionSchedule::new(name, segments, FallbackCalendar::Weekdays)
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
                 closing_auction / break / post_close（段外即闭市，不要写 closed）"
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

    fn cs(start: &str, end: &str, session: &str, anchor: Option<&str>) -> CustomSegment {
        CustomSegment {
            start: start.to_string(),
            end: end.to_string(),
            session: session.to_string(),
            anchor: anchor.map(str::to_string),
        }
    }

    #[test]
    fn custom_schedule_anchors_night_and_wraps_midnight() {
        let s = schedule_from_custom(
            "xx",
            &[
                cs("21:00", "02:30", "continuous", None),
                cs("09:00", "11:30", "Continuous", None),
                cs("19:00", "20:00", "post_close", Some("DAY")),
            ],
        )
        .unwrap();
        assert_eq!(s.fallback_calendar, FallbackCalendar::Weekdays);
        assert!(s.has_night());
        let night = s
            .segments
            .iter()
            .find(|x| x.anchor == Anchor::PrevTradingDay)
            .unwrap();
        assert_eq!(night.start_sec, 21 * 3600);
        assert_eq!(night.end_sec, 26 * 3600 + 30 * 60);
        assert!(
            s.segments
                .iter()
                .any(|x| x.anchor == Anchor::TradingDay && x.start_sec == 19 * 3600)
        );
    }

    #[test]
    fn custom_schedule_rejects_invalid() {
        let err = |segs: &[CustomSegment]| schedule_from_custom("xx", segs).unwrap_err();
        assert!(err(&[]).contains("至少"));
        assert!(err(&[cs("09:30", "09:30", "continuous", None)]).contains("长度"));
        assert!(err(&[cs("11:30", "09:30", "continuous", None)]).contains("跨午夜"));
        assert!(err(&[cs("25:00", "26:00", "continuous", None)]).contains("HH:MM"));
        assert!(err(&[cs("24:00", "01:00", "continuous", None)]).contains("HH:MM"));
        assert!(err(&[cs("09:30", "11:30", "closed", None)]).contains("closed"));
        assert!(err(&[cs("09:30", "11:30", "normal", None)]).contains("未知时段类型"));
        assert!(err(&[cs("09:30", "11:30", "continuous", Some("-1d"))]).contains("锚定"));
        assert!(
            err(&[
                cs("21:00", "09:45", "continuous", None),
                cs("09:30", "11:30", "continuous", None),
            ])
            .contains("重叠")
        );
        // 显式锚定当天的跨午夜段不允许
        assert!(err(&[cs("21:00", "02:30", "continuous", Some("day"))]).contains("跨午夜"));
        // 夜盘段与前一交易日自身的日盘段重叠（夜盘落在前一交易日的日历日上）
        let e = err(&[
            cs("13:00", "19:00", "continuous", None),
            cs("18:30", "02:30", "continuous", None),
        ]);
        assert!(
            e.contains("前一交易日的日盘") && e.contains("18:30-02:30") && !e.contains("  "),
            "{e}"
        );
        let e = err(&[
            cs("09:30", "11:30", "continuous", None),
            cs("10:00", "11:00", "continuous", Some("prev")),
        ]);
        assert!(e.contains("前一交易日的日盘"), "{e}");
        // 同锚定重叠：指明两段，跨午夜结束按用户写法显示，并说明怎么改
        let e = err(&[
            cs("21:00", "23:00", "continuous", None),
            cs("22:00", "01:00", "continuous", None),
        ]);
        assert!(
            e.contains("21:00-23:00") && e.contains("22:00-01:00") && e.contains("调整"),
            "{e}"
        );
        assert!(!e.contains("25:00") && !e.contains("  "), "{e}");
        // 时刻须严格两位 HH:MM
        for bad in [
            "9:30", " 09:30", "09:30 ", "09:30:00", "0930", "09:60", "24:30", "+9:30",
        ] {
            assert!(
                err(&[cs(bad, "11:30", "continuous", None)]).contains("HH:MM"),
                "{bad:?}"
            );
        }
        // 夜盘紧接前一交易日日盘之后（不相交）合法
        assert!(
            schedule_from_custom(
                "xx",
                &[
                    cs("13:00", "18:30", "continuous", None),
                    cs("18:30", "02:30", "continuous", None),
                ]
            )
            .is_ok()
        );
        // 00:00-24:00 合法
        assert!(schedule_from_custom("xx", &[cs("00:00", "24:00", "continuous", None)]).is_ok());
    }

    #[test]
    fn new_rejects_cross_anchor_overlap() {
        let night = SessionSegment {
            anchor: Anchor::PrevTradingDay,
            start_sec: 21 * 3600,
            end_sec: 33 * 3600 + 45 * 60,
            session: TradingSession::Continuous,
        };
        let day = SessionSegment {
            anchor: Anchor::TradingDay,
            start_sec: 9 * 3600 + 30 * 60,
            end_sec: 11 * 3600 + 30 * 60,
            session: TradingSession::Continuous,
        };
        let e =
            SessionSchedule::new("x", vec![night, day], FallbackCalendar::Weekdays).unwrap_err();
        assert!(e.contains("重叠"), "{e}");
        let short_night = SessionSegment {
            end_sec: 26 * 3600 + 30 * 60,
            ..night
        };
        assert!(
            SessionSchedule::new("x", vec![short_night, day], FallbackCalendar::Weekdays).is_ok()
        );
    }

    #[test]
    fn custom_segment_roundtrips_msgpack() {
        let v = vec![(
            "xx".to_string(),
            vec![
                cs("21:00", "02:30", "continuous", None),
                cs("09:00", "10:00", "continuous", Some("day")),
            ],
        )];
        let bytes = rmp_serde::to_vec(&v).unwrap();
        let back: Vec<(String, Vec<CustomSegment>)> = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back, v);
    }
}
