//! 流式 Parquet 数据客户端 (C.2a, out-of-core 基座).
//!
//! 有界内存: 每次经 polars eager `ParquetReader::with_slice` 读取一个块
//! (chunk_size 行), 转为 `Bar` 缓冲后逐个产出; 内存占用约为 chunk_size 行,
//! 与文件总量无关。假定 parquet 已按 `timestamp` 升序 (规范格式)。
//!
//! 用 eager reader 而非 `LazyFrame::scan_parquet().slice()`: 二者切片语义等价,
//! 但 lazy 版会把 polars 的查询计划/表达式引擎链进扩展模块 (wheel +12MB)。
//!
//! 规范列: `timestamp` (i64 纳秒 UTC) + `open/high/low/close/volume` (f64) +
//! 可选 `symbol` (str)。价格重构用 `Decimal::from_f64`, 与 `from_arrays` 一致。

use std::collections::VecDeque;
use std::time::Duration;

use polars::prelude::*;
use rust_decimal::prelude::*;

use crate::data::client::DataClient;
use crate::error::AkQuantError;
use crate::event::Event;
use crate::model::Bar;

/// 默认分块行数.
pub const DEFAULT_CHUNK_ROWS: usize = 65_536;

pub struct ParquetStreamClient {
    path: String,
    default_symbol: String,
    chunk_size: usize,
    offset: usize,
    buffer: VecDeque<Bar>,
    exhausted: bool,
}

impl ParquetStreamClient {
    pub fn new(
        path: impl Into<String>,
        default_symbol: impl Into<String>,
        chunk_size: usize,
    ) -> Self {
        Self {
            path: path.into(),
            default_symbol: default_symbol.into(),
            chunk_size: chunk_size.max(1),
            offset: 0,
            buffer: VecDeque::new(),
            exhausted: false,
        }
    }

    #[inline]
    fn dec(value: f64) -> Decimal {
        Decimal::from_f64(value).unwrap_or(Decimal::ZERO)
    }

    fn read_chunk(&self) -> PolarsResult<DataFrame> {
        let file = std::fs::File::open(&self.path)?;
        ParquetReader::new(file)
            .with_slice(Some((self.offset, self.chunk_size)))
            .finish()
    }

    /// 读取下一块并填充缓冲区.
    fn fill_buffer(&mut self) {
        if self.exhausted {
            return;
        }
        let df = match self.read_chunk() {
            Ok(df) => df,
            Err(e) => {
                // 数据读取失败会静默截断数据流, 回测样本不完整, 属错误级。
                log::error!("ParquetStreamClient read failed ({}): {e}", self.path);
                self.exhausted = true;
                return;
            }
        };
        let n = df.height();
        if n < self.chunk_size {
            self.exhausted = true;
        }
        self.offset += n;
        if n == 0 {
            return;
        }
        if let Err(e) = self.push_rows(&df, n) {
            // 数据解析失败会静默截断数据流, 回测样本不完整, 属错误级。
            log::error!("ParquetStreamClient parse failed ({}): {e}", self.path);
            self.exhausted = true;
        }
    }

    fn push_rows(&mut self, df: &DataFrame, n: usize) -> PolarsResult<()> {
        let ts = df.column("timestamp")?.i64()?;
        let open = df.column("open")?.f64()?;
        let high = df.column("high")?.f64()?;
        let low = df.column("low")?.f64()?;
        let close = df.column("close")?.f64()?;
        let volume = df.column("volume")?.f64()?;
        let symbol = df.column("symbol").ok().and_then(|c| c.str().ok().cloned());

        for i in 0..n {
            let sym = match &symbol {
                Some(s) => s.get(i).unwrap_or(&self.default_symbol).to_string(),
                None => self.default_symbol.clone(),
            };
            self.buffer.push_back(Bar {
                timestamp: ts.get(i).unwrap_or(0),
                open: Self::dec(open.get(i).unwrap_or(0.0)),
                high: Self::dec(high.get(i).unwrap_or(0.0)),
                low: Self::dec(low.get(i).unwrap_or(0.0)),
                close: Self::dec(close.get(i).unwrap_or(0.0)),
                volume: Self::dec(volume.get(i).unwrap_or(0.0)),
                symbol: sym,
                extra: std::collections::HashMap::new(),
            });
        }
        Ok(())
    }
}

impl DataClient for ParquetStreamClient {
    fn peek_timestamp(&mut self) -> Option<i64> {
        if self.buffer.is_empty() {
            self.fill_buffer();
        }
        self.buffer.front().map(|b| b.timestamp)
    }

    fn next(&mut self) -> Option<Event> {
        if self.buffer.is_empty() {
            self.fill_buffer();
        }
        self.buffer.pop_front().map(Event::Bar)
    }

    fn add(&mut self, _event: Event) -> Result<(), AkQuantError> {
        // 只读流式源, 不支持追加事件.
        Err(AkQuantError::DataError(
            "ParquetStreamClient 为只读流式数据源, 不支持 add()".to_string(),
        ))
    }

    fn sort(&mut self) {
        // parquet 假定已按 timestamp 升序, 无需(也无法有界内存地)重排.
    }

    fn len_hint(&self) -> Option<usize> {
        // 流式读取, 总量未知.
        None
    }

    fn wait_peek(&mut self, _timeout: Duration) -> Option<i64> {
        self.peek_timestamp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_canonical_parquet(path: &std::path::Path, n: usize) {
        let ts: Vec<i64> = (0..n as i64)
            .map(|i| 1_600_000_000_000_000_000 + i * 60_000_000_000)
            .collect();
        let f: Vec<f64> = (0..n).map(|i| 10.0 + i as f64).collect();
        let syms: Vec<String> = (0..n).map(|_| "X".to_string()).collect();
        let mut df = DataFrame::new_infer_height(vec![
            Series::new("timestamp".into(), ts).into(),
            Series::new("open".into(), f.clone()).into(),
            Series::new("high".into(), f.clone()).into(),
            Series::new("low".into(), f.clone()).into(),
            Series::new("close".into(), f.clone()).into(),
            Series::new("volume".into(), f).into(),
            Series::new("symbol".into(), syms).into(),
        ])
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        ParquetWriter::new(file).finish(&mut df).unwrap();
    }

    #[test]
    fn test_stream_reads_all_bars_in_order_bounded() {
        let dir = std::env::temp_dir().join("akq_parquet_stream_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.parquet");
        write_canonical_parquet(&path, 10);

        let mut client =
            ParquetStreamClient::new(path.to_string_lossy().to_string(), "X", 3);

        let mut count = 0usize;
        let mut last_ts = i64::MIN;
        while let Some(event) = client.next() {
            if let Event::Bar(bar) = event {
                assert!(bar.timestamp > last_ts, "时间戳应升序");
                last_ts = bar.timestamp;
                assert_eq!(bar.symbol, "X");
                count += 1;
                // 有界内存: 缓冲区不超过 chunk_size.
                assert!(client.buffer.len() <= 3);
            }
        }
        assert_eq!(count, 10, "应读到全部 10 根 bar");
    }

    #[test]
    fn test_peek_matches_first() {
        let dir = std::env::temp_dir().join("akq_parquet_stream_peek");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.parquet");
        write_canonical_parquet(&path, 5);

        let mut client =
            ParquetStreamClient::new(path.to_string_lossy().to_string(), "X", 2);
        let peeked = client.peek_timestamp().unwrap();
        let first = match client.next().unwrap() {
            Event::Bar(b) => b.timestamp,
            _ => panic!(),
        };
        assert_eq!(peeked, first);
    }
}
