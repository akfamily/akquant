use crate::model::{Bar, Tick};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::sync::{mpsc, Arc, Mutex};
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

/// Data Client Trait for streaming or in-memory data
pub trait DataClient: Send {
    fn peek_timestamp(&mut self) -> Option<i64>;
    fn next(&mut self) -> Option<Event>;
    fn add(&mut self, event: Event) -> PyResult<()>;
    fn sort(&mut self);
    fn len_hint(&self) -> Option<usize>;

    /// 是否为实时数据源
    fn is_live(&self) -> bool {
        false
    }

    /// 阻塞等待下一个事件的时间戳 (用于实时模式)
    fn wait_peek(&mut self, _timeout: Duration) -> Option<i64> {
        self.peek_timestamp()
    }
}

/// Simulated Data Client (In-Memory)
pub struct SimulatedDataClient {
    pub events: VecDeque<Event>,
}

impl SimulatedDataClient {
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
        }
    }
}

impl DataClient for SimulatedDataClient {
    fn peek_timestamp(&mut self) -> Option<i64> {
        self.events.front().map(|e| match e {
            Event::Bar(b) => b.timestamp,
            Event::Tick(t) => t.timestamp,
        })
    }

    fn next(&mut self) -> Option<Event> {
        self.events.pop_front()
    }

    fn add(&mut self, event: Event) -> PyResult<()> {
        self.events.push_back(event);
        Ok(())
    }

    fn sort(&mut self) {
        self.events.make_contiguous().sort_by_key(|e| match e {
            Event::Bar(b) => b.timestamp,
            Event::Tick(t) => t.timestamp,
        });
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.events.len())
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

             let bar = Bar {
                 timestamp: normalize_timestamp(row.timestamp),
                 open: Decimal::from_f64(row.open).unwrap_or(Decimal::ZERO),
                 high: Decimal::from_f64(row.high).unwrap_or(Decimal::ZERO),
                 low: Decimal::from_f64(row.low).unwrap_or(Decimal::ZERO),
                 close: Decimal::from_f64(row.close).unwrap_or(Decimal::ZERO),
                 volume: Decimal::from_f64(row.volume).unwrap_or(Decimal::ZERO),
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
        })
    }

    fn next(&mut self) -> Option<Event> {
        if self.current.is_none() {
            self.current = self.read_next();
        }
        self.current.take()
    }

    fn add(&mut self, _event: Event) -> PyResult<()> {
        Err(PyTypeError::new_err("Cannot add data to a streaming CSV provider"))
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
        })
    }

    fn next(&mut self) -> Option<Event> {
        if self.current.is_some() {
            return self.current.take();
        }
        self.rx.try_recv().ok()
    }

    fn add(&mut self, event: Event) -> PyResult<()> {
        self.sender.send(event).map_err(|e| PyValueError::new_err(e.to_string()))
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

/// 从数组批量创建 Bar 列表 (Python 优化用)
#[gen_stub_pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn from_arrays(
    timestamps: Vec<i64>,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
    symbol: Option<String>,
    symbols: Option<Vec<String>>,
    extra: Option<HashMap<String, Vec<f64>>>,
) -> PyResult<Vec<Bar>> {
    let len = timestamps.len();
    if opens.len() != len
        || highs.len() != len
        || lows.len() != len
        || closes.len() != len
        || volumes.len() != len
    {
        return Err(PyValueError::new_err(
            "All arrays must have the same length",
        ));
    }

    if let Some(ref syms) = symbols
        && syms.len() != len
    {
        return Err(PyValueError::new_err(
            "symbols array must have the same length as other arrays",
        ));
    }

    // Check extra arrays length
    if let Some(ref extra_data) = extra {
        for (key, val) in extra_data {
            if val.len() != len {
                 return Err(PyValueError::new_err(format!(
                    "Extra array '{}' must have the same length as other arrays",
                    key
                )));
            }
        }
    }

    let mut bars = Vec::with_capacity(len);

    for i in 0..len {
        let sym = if let Some(ref syms) = symbols {
            syms[i].clone()
        } else if let Some(ref s) = symbol {
            s.clone()
        } else {
            "UNKNOWN".to_string()
        };

        let ts = timestamps[i];
        let normalized_ts = normalize_timestamp(ts);

        let mut bar_extra = HashMap::new();
        if let Some(ref extra_data) = extra {
            for (k, v) in extra_data {
                bar_extra.insert(k.clone(), v[i]);
            }
        }

        bars.push(Bar {
            timestamp: normalized_ts,
            open: Decimal::from_f64(opens[i]).unwrap_or(Decimal::ZERO),
            high: Decimal::from_f64(highs[i]).unwrap_or(Decimal::ZERO),
            low: Decimal::from_f64(lows[i]).unwrap_or(Decimal::ZERO),
            close: Decimal::from_f64(closes[i]).unwrap_or(Decimal::ZERO),
            volume: Decimal::from_f64(volumes[i]).unwrap_or(Decimal::ZERO),
            symbol: sym,
            extra: bar_extra,
        });
    }

    Ok(bars)
}

/// 事件类型枚举
#[derive(Debug, Clone)]
pub enum Event {
    Bar(Bar),
    Tick(Tick),
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
pub struct DataFeed {
    pub provider: Arc<Mutex<Box<dyn DataClient>>>,
    pub live_sender: Option<Arc<Mutex<mpsc::Sender<Event>>>>,
}

#[pymethods]
impl DataFeed {
    #[new]
    pub fn new() -> Self {
        DataFeed {
            provider: Arc::new(Mutex::new(Box::new(SimulatedDataClient::new()))),
            live_sender: None,
        }
    }

    #[staticmethod]
    pub fn from_csv(path: &str, symbol: &str) -> PyResult<Self> {
        let provider = CsvDataClient::new(path, symbol)?;
        Ok(DataFeed {
            provider: Arc::new(Mutex::new(Box::new(provider))),
            live_sender: None,
        })
    }

    /// 创建实时数据源 (Channel 模式)
    /// 适用于 CTP 等实时接口推送数据
    #[staticmethod]
    pub fn create_live() -> Self {
        let provider = RealtimeDataClient::new();
        let sender = provider.get_sender();
        DataFeed {
            provider: Arc::new(Mutex::new(Box::new(provider))),
            live_sender: Some(Arc::new(Mutex::new(sender))),
        }
    }

    /// 添加 Bar 数据
    pub fn add_bar(&mut self, bar: Bar) -> PyResult<()> {
        if let Some(sender_lock) = &self.live_sender {
            let sender = sender_lock.lock().unwrap();
            sender.send(Event::Bar(bar)).map_err(|e| PyValueError::new_err(e.to_string()))
        } else {
            let mut provider = self.provider.lock().unwrap();
            provider.add(Event::Bar(bar))
        }
    }

    /// 批量添加 Bar 数据 (优化)
    pub fn add_bars(&mut self, bars: Vec<Bar>) -> PyResult<()> {
        if let Some(sender_lock) = &self.live_sender {
            let sender = sender_lock.lock().unwrap();
            for bar in bars {
                sender.send(Event::Bar(bar)).map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
            Ok(())
        } else {
            let mut provider = self.provider.lock().unwrap();
            for bar in bars {
                provider.add(Event::Bar(bar))?;
            }
            Ok(())
        }
    }

    /// 从数组批量添加 Bar 数据 (高性能优化)
    #[allow(clippy::too_many_arguments)]
    pub fn add_arrays(
        &mut self,
        timestamps: Vec<i64>,
        opens: Vec<f64>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
        volumes: Vec<f64>,
        symbol: Option<String>,
        symbols: Option<Vec<String>>,
        extra: Option<HashMap<String, Vec<f64>>>,
    ) -> PyResult<()> {
        let bars = from_arrays(timestamps, opens, highs, lows, closes, volumes, symbol, symbols, extra)?;
        self.add_bars(bars)
    }

    /// 对事件队列按时间戳进行排序
    /// 在多标的批量加载后必须调用，以确保回测时序正确
    pub fn sort(&mut self) {
        let mut provider = self.provider.lock().unwrap();
        provider.sort();
    }

    /// 添加 Tick 数据
    pub fn add_tick(&mut self, tick: Tick) -> PyResult<()> {
        if let Some(sender_lock) = &self.live_sender {
            let sender = sender_lock.lock().unwrap();
            sender.send(Event::Tick(tick)).map_err(|e| PyValueError::new_err(e.to_string()))
        } else {
            let mut provider = self.provider.lock().unwrap();
            provider.add(Event::Tick(tick))
        }
    }
}

// Helper method for Engine
impl DataFeed {
    pub fn peek_timestamp(&self) -> Option<i64> {
        let mut provider = self.provider.lock().unwrap();
        provider.peek_timestamp()
    }

    pub fn next(&self) -> Option<Event> {
        let mut provider = self.provider.lock().unwrap();
        provider.next()
    }

    pub fn len_hint(&self) -> Option<usize> {
        let provider = self.provider.lock().unwrap();
        provider.len_hint()
    }

    pub fn is_live(&self) -> bool {
        let provider = self.provider.lock().unwrap();
        provider.is_live()
    }

    pub fn wait_peek(&self, timeout: Duration) -> Option<i64> {
        let mut provider = self.provider.lock().unwrap();
        provider.wait_peek(timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_arrays_valid() {
        let ts = vec![1625097600]; // 2021-07-01 00:00:00 UTC
        let opens = vec![100.0];
        let highs = vec![105.0];
        let lows = vec![95.0];
        let closes = vec![102.0];
        let volumes = vec![1000.0];
        let symbol = Some("AAPL".to_string());

        let result = from_arrays(ts, opens, highs, lows, closes, volumes, symbol, None, None);
        assert!(result.is_ok());

        let bars = result.unwrap();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].symbol, "AAPL");
        assert_eq!(bars[0].open, Decimal::from(100));
        assert_eq!(bars[0].timestamp, 1625097600000000000); // Normalized to nanoseconds
    }

    #[test]
    fn test_data_feed() {
        let mut feed = DataFeed::new();
        let bar = Bar {
            timestamp: 1000,
            open: Decimal::ZERO,
            high: Decimal::ZERO,
            low: Decimal::ZERO,
            close: Decimal::ZERO,
            volume: Decimal::ZERO,
            symbol: "TEST".to_string(),
            extra: std::collections::HashMap::new(),
        };

        feed.add_bar(bar.clone()).unwrap();
        // Since internal is hidden, test via public API behavior if possible,
        // or check behavior via mock engine.
        // For now, simple assertion that it didn't panic.
    }

    #[test]
    fn test_live_feed() {
        let feed = DataFeed::create_live();

        let bar = Bar {
            timestamp: 1625097600000000000,
            open: Decimal::ZERO,
            high: Decimal::ZERO,
            low: Decimal::ZERO,
            close: Decimal::ZERO,
            volume: Decimal::ZERO,
            symbol: "TEST".to_string(),
            extra: std::collections::HashMap::new(),
        };

        // Clone feed for thread
        let mut feed_clone = feed.clone();
        let bar_clone = bar.clone();

        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            feed_clone.add_bar(bar_clone).unwrap();
        });

        // Should wait and receive
        let ts = feed.wait_peek(Duration::from_secs(1));
        assert!(ts.is_some());
        assert_eq!(ts.unwrap(), 1625097600000000000);

        // Next should return event
        let event = feed.next();
        assert!(event.is_some());
        if let Event::Bar(b) = event.unwrap() {
            assert_eq!(b.timestamp, 1625097600000000000);
        } else {
            panic!("Expected Bar");
        }
    }
}
