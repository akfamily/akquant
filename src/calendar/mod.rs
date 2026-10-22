//! 交易时段、交易日与 bar 区间的统一底座。
//!
//! 设计见 `docs/superpowers/specs/2026-10-23-trading-session-calendar-design.md` §4.1。

// 阶段 2 接入引擎后删除
#![allow(dead_code)]

pub(crate) mod clock;
pub mod schedule;
