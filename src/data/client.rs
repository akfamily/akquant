use crate::data::columns::BarColumns;
use crate::error::AkQuantError;
use crate::event::Event;
use crate::log_context::{AkqLogContext, format_event_time_nanos, render_log_message};
use crate::model::Bar;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rust_decimal::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::sync::mpsc;
use std::time::Duration;

#[inline]
fn normalize_timestamp(ts: i64) -> i64 {
    let abs_ts = ts.abs();
    if abs_ts < 100_000_000_000 {
        // Seconds (< 10^11, up to year ~5138)
        ts * 1_000_000_000
    } else if abs_ts < 100_000_000_000_000 {
        // Milliseconds (< 10^14, up to year ~5138)
        ts * 1_000_000
    } else if abs_ts < 100_000_000_000_000_000 {
        // Microseconds (< 10^17, up to year ~5138)
        ts * 1_000
    } else {
        // Nanoseconds
        ts
    }
}

/// 提取事件时间戳 (无时间戳的事件回退为 0).
#[inline]
fn event_timestamp(event: &Event) -> i64 {
    match event {
        Event::Bar(b) => b.timestamp,
        Event::Tick(t) => t.timestamp,
        Event::ExecutionReport(_, Some(trade)) => trade.timestamp,
        _ => 0,
    }
}

/// Data Client Trait for streaming or in-memory data
pub trait DataClient: Send {
    fn peek_timestamp(&mut self) -> Option<i64>;
    fn next(&mut self) -> Option<Event>;
    fn add(&mut self, event: Event) -> Result<(), AkQuantError>;
    fn sort(&mut self);
    fn len_hint(&self) -> Option<usize>;
    fn progress_len_hint(&self) -> Option<usize> {
        self.len_hint()
    }

    /// 批量添加列式 Bar 数据. 默认逐行重构为 Bar 后调用 [`DataClient::add`];
    /// 支持列式存储的实现可覆盖以直接保存 (见 [`SimulatedDataClient`]).
    fn add_bar_columns(&mut self, columns: BarColumns) -> Result<(), AkQuantError> {
        for i in 0..columns.len() {
            self.add(Event::Bar(columns.reconstruct_bar(i)))?;
        }
        Ok(())
    }

    /// 是否为实时数据源
    fn is_live(&self) -> bool {
        false
    }

    /// 阻塞等待下一个事件的时间戳 (用于实时模式)
    fn wait_peek(&mut self, _timeout: Duration) -> Option<i64> {
        self.peek_timestamp()
    }
}

/// Simulated Data Client (In-Memory).
///
/// Bar 数据以列式 [`BarColumns`] 存储 (来自 `add_arrays`), 其余事件 (Tick /
/// `List[Bar]` 等) 存于 `events`; 迭代时按时间戳流式归并两者。
pub struct SimulatedDataClient {
    pub bars: BarColumns,
    pub bar_cursor: usize,
    pub events: VecDeque<Event>,
}

impl SimulatedDataClient {
    pub fn new() -> Self {
        Self {
            bars: BarColumns::default(),
            bar_cursor: 0,
            events: VecDeque::new(),
        }
    }

    /// 当前列式游标处的 Bar 时间戳 (已消费完则为 None).
    #[inline]
    fn peek_bar_timestamp(&self) -> Option<i64> {
        if self.bar_cursor < self.bars.len() {
            Some(self.bars.timestamp_at(self.bar_cursor))
        } else {
            None
        }
    }
}

impl DataClient for SimulatedDataClient {
    fn peek_timestamp(&mut self) -> Option<i64> {
        match (self.peek_bar_timestamp(), self.events.front().map(event_timestamp)) {
            (Some(b), Some(e)) => Some(b.min(e)),
            (Some(b), None) => Some(b),
            (None, Some(e)) => Some(e),
            (None, None) => None,
        }
    }

    fn next(&mut self) -> Option<Event> {
        let bar_ts = self.peek_bar_timestamp();
        let evt_ts = self.events.front().map(event_timestamp);
        // 列式 Bar 与事件按时间戳归并; 相等时列式 Bar 优先 (确定性 tie-break).
        let take_bar = match (bar_ts, evt_ts) {
            (Some(b), Some(e)) => b <= e,
            (Some(_), None) => true,
            _ => false,
        };
        if take_bar {
            let bar = self.bars.reconstruct_bar(self.bar_cursor);
            self.bar_cursor += 1;
            Some(Event::Bar(bar))
        } else {
            self.events.pop_front()
        }
    }

    fn add(&mut self, event: Event) -> Result<(), AkQuantError> {
        self.events.push_back(event);
        Ok(())
    }

    fn add_bar_columns(&mut self, columns: BarColumns) -> Result<(), AkQuantError> {
        self.bars.append(columns);
        Ok(())
    }

    fn sort(&mut self) {
        self.bars.sort_by_timestamp();
        self.events.make_contiguous().sort_by_key(event_timestamp);
    }

    fn len_hint(&self) -> Option<usize> {
        Some((self.bars.len() - self.bar_cursor) + self.events.len())
    }

    fn progress_len_hint(&self) -> Option<usize> {
        let mut timestamps = HashSet::new();
        for i in self.bar_cursor..self.bars.len() {
            let ts = self.bars.timestamp_at(i);
            if ts > 0 {
                timestamps.insert(ts);
            }
        }
        for event in &self.events {
            let ts = event_timestamp(event);
            if ts > 0 {
                timestamps.insert(ts);
            }
        }
        Some(timestamps.len())
    }
}

/// CSV Data Client (Streaming)
pub struct CsvDataClient {
    reader: csv::Reader<File>,
    current: Option<Event>,
    symbol: String,
}

impl CsvDataClient {
    pub fn new(path: &str, symbol: &str) -> PyResult<Self> {
        let file = File::open(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        Ok(Self {
            reader,
            current: None,
            symbol: symbol.to_string(),
        })
    }

    fn read_next(&mut self) -> Option<Event> {
        fn warn_invalid_numeric(field_name: &str, value: f64, symbol: &str, timestamp_ns: i64) {
            log::warn!(
                "{}",
                render_log_message(
                    format!("Invalid {field_name} {value}, defaulting to 0.0"),
                    AkqLogContext::new()
                        .phase("data")
                        .symbol(symbol)
                        .event_time(timestamp_ns)
                        .event_time_iso(format_event_time_nanos(timestamp_ns)),
                )
            );
        }

        // Assume CSV columns: timestamp, open, high, low, close, volume
        // Or using serde with a struct.
        // Let's use string records and parse manually for flexibility or define a struct.
        // Defining a struct is better.

        // Internal struct for CSV row
        #[derive(serde::Deserialize)]
        struct CsvRow {
            timestamp: i64,
            open: f64,
            high: f64,
            low: f64,
            close: f64,
            volume: f64,
        }

        let mut record = csv::StringRecord::new();
        if self.reader.read_record(&mut record).ok()? {
            // Deserialize
            let row: CsvRow = record.deserialize(self.reader.headers().ok()).ok()?;

            let normalized_timestamp = normalize_timestamp(row.timestamp);
            let bar = Bar {
                timestamp: normalized_timestamp,
                open: Decimal::from_f64(row.open).unwrap_or_else(|| {
                    warn_invalid_numeric(
                        "open price",
                        row.open,
                        self.symbol.as_str(),
                        normalized_timestamp,
                    );
                    Decimal::ZERO
                }),
                high: Decimal::from_f64(row.high).unwrap_or_else(|| {
                    warn_invalid_numeric(
                        "high price",
                        row.high,
                        self.symbol.as_str(),
                        normalized_timestamp,
                    );
                    Decimal::ZERO
                }),
                low: Decimal::from_f64(row.low).unwrap_or_else(|| {
                    warn_invalid_numeric(
                        "low price",
                        row.low,
                        self.symbol.as_str(),
                        normalized_timestamp,
                    );
                    Decimal::ZERO
                }),
                close: Decimal::from_f64(row.close).unwrap_or_else(|| {
                    warn_invalid_numeric(
                        "close price",
                        row.close,
                        self.symbol.as_str(),
                        normalized_timestamp,
                    );
                    Decimal::ZERO
                }),
                volume: Decimal::from_f64(row.volume).unwrap_or_else(|| {
                    warn_invalid_numeric(
                        "volume",
                        row.volume,
                        self.symbol.as_str(),
                        normalized_timestamp,
                    );
                    Decimal::ZERO
                }),
                symbol: self.symbol.clone(),
                extra: HashMap::new(),
            };
            Some(Event::Bar(bar))
        } else {
            None
        }
    }
}

impl DataClient for CsvDataClient {
    fn peek_timestamp(&mut self) -> Option<i64> {
        if self.current.is_none() {
            self.current = self.read_next();
        }

        self.current.as_ref().map(|e| match e {
            Event::Bar(b) => b.timestamp,
            Event::Tick(t) => t.timestamp,
            Event::ExecutionReport(_, Some(trade)) => trade.timestamp,
            _ => 0,
        })
    }

    fn next(&mut self) -> Option<Event> {
        if self.current.is_none() {
            self.current = self.read_next();
        }
        self.current.take()
    }

    fn add(&mut self, _event: Event) -> Result<(), AkQuantError> {
        Err(AkQuantError::DataError(
            "Cannot add data to a streaming CSV provider".to_string(),
        ))
    }

    fn sort(&mut self) {
        // Assume CSV is sorted or ignore
    }

    fn len_hint(&self) -> Option<usize> {
        None
    }
}

/// Realtime Data Client (Channel)
/// 适用于 CTP 等实时数据推送场景
pub struct RealtimeDataClient {
    rx: mpsc::Receiver<Event>,
    sender: mpsc::Sender<Event>, // Keep sender to clone for external use
    current: Option<Event>,
}

impl RealtimeDataClient {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            rx,
            sender: tx,
            current: None,
        }
    }

    pub fn get_sender(&self) -> mpsc::Sender<Event> {
        self.sender.clone()
    }
}

impl DataClient for RealtimeDataClient {
    fn peek_timestamp(&mut self) -> Option<i64> {
        // Try to read from channel non-blocking
        if self.current.is_none() {
            match self.rx.try_recv() {
                Ok(event) => self.current = Some(event),
                Err(_) => return None, // Empty or Disconnected
            }
        }

        self.current.as_ref().map(|e| match e {
            Event::Bar(b) => b.timestamp,
            Event::Tick(t) => t.timestamp,
            Event::ExecutionReport(_, Some(trade)) => trade.timestamp,
            _ => 0,
        })
    }

    fn next(&mut self) -> Option<Event> {
        if self.current.is_some() {
            return self.current.take();
        }
        self.rx.try_recv().ok()
    }

    fn add(&mut self, event: Event) -> Result<(), AkQuantError> {
        self.sender
            .send(event)
            .map_err(|e| AkQuantError::DataError(e.to_string()))
    }

    fn sort(&mut self) {
        // Live data cannot be sorted
    }

    fn len_hint(&self) -> Option<usize> {
        None
    }

    fn is_live(&self) -> bool {
        true
    }

    fn wait_peek(&mut self, timeout: Duration) -> Option<i64> {
        if self.current.is_some() {
            return self.peek_timestamp();
        }
        match self.rx.recv_timeout(timeout) {
            Ok(event) => {
                self.current = Some(event);
                self.peek_timestamp()
            }
            Err(_) => None,
        }
    }
}

pub enum FeedAction {
    Event(Box<Event>),
    Timer(i64),
    Wait,
    End,
}
