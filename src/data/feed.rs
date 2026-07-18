use crate::event::Event;
use crate::log_context::{AkqLogContext, format_event_time_nanos, render_log_message};
use crate::model::{Bar, Tick};
use chrono::Utc;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use super::client::{
    CsvDataClient, DataClient, FeedAction, RealtimeDataClient, SimulatedDataClient,
};
use super::columns::BarColumns;

fn clock_fallback_context(next_timer_ts: Option<i64>) -> AkqLogContext {
    let mut context = AkqLogContext::new().phase("data");
    if let Some(timer_ts) = next_timer_ts {
        context = context
            .event_time(timer_ts)
            .event_time_iso(format_event_time_nanos(timer_ts));
    }
    context
}

fn warn_clock_fallback(next_timer_ts: Option<i64>) {
    log::warn!(
        "{}",
        render_log_message(
            "Failed to get current timestamp, defaulting to 0",
            clock_fallback_context(next_timer_ts),
        )
    );
}

/// 数据源管理器.
///
/// 负责管理历史数据和实时数据流。
/// 支持 CSV 文件读取、数组批量加载和实时数据推送。
#[gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct DataFeed {
    pub provider: Arc<Mutex<Box<dyn DataClient>>>,
    pub live_sender: Option<Arc<Mutex<mpsc::Sender<Event>>>>,
}

#[pymethods]
impl DataFeed {
    /// 创建空的数据源.
    #[new]
    pub fn new() -> Self {
        DataFeed {
            provider: Arc::new(Mutex::new(Box::new(SimulatedDataClient::new()))),
            live_sender: None,
        }
    }

    /// 从 CSV 文件创建数据源.
    ///
    /// :param path: CSV 文件路径
    /// :param symbol: 标的代码
    /// :return: DataFeed 实例
    #[staticmethod]
    pub fn from_csv(path: &str, symbol: &str) -> PyResult<Self> {
        let provider = CsvDataClient::new(path, symbol)?;
        Ok(DataFeed {
            provider: Arc::new(Mutex::new(Box::new(provider))),
            live_sender: None,
        })
    }

    /// 创建实时数据源 (Channel 模式).
    ///
    /// 适用于 CTP 等实时接口推送数据。
    ///
    /// :return: DataFeed 实例
    #[staticmethod]
    pub fn create_live() -> Self {
        let provider = RealtimeDataClient::new();
        let sender = provider.get_sender();
        DataFeed {
            provider: Arc::new(Mutex::new(Box::new(provider))),
            live_sender: Some(Arc::new(Mutex::new(sender))),
        }
    }

    /// 添加 Bar 数据.
    ///
    /// :param bar: K 线数据
    pub fn add_bar(&mut self, bar: Bar) -> PyResult<()> {
        if let Some(sender_lock) = &self.live_sender {
            let sender = sender_lock.lock().unwrap();
            sender
                .send(Event::Bar(bar))
                .map_err(|e| PyValueError::new_err(e.to_string()))
        } else {
            let mut provider = self.provider.lock().unwrap();
            provider.add(Event::Bar(bar))?;
            Ok(())
        }
    }

    /// 批量添加 Bar 数据.
    ///
    /// :param bars: K 线数据列表
    pub fn add_bars(&mut self, bars: Vec<Bar>) -> PyResult<()> {
        if let Some(sender_lock) = &self.live_sender {
            let sender = sender_lock.lock().unwrap();
            for bar in bars {
                sender
                    .send(Event::Bar(bar))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
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

    /// 从数组批量添加 Bar 数据 (高性能优化 - Zero Copy).
    ///
    /// :param timestamps: 时间戳数组 (int64, 纳秒)
    /// :param opens: 开盘价数组 (float64)
    /// :param highs: 最高价数组 (float64)
    /// :param lows: 最低价数组 (float64)
    /// :param closes: 收盘价数组 (float64)
    /// :param volumes: 成交量数组 (float64)
    /// :param symbol: 标的代码 (可选，单一标的)
    /// :param symbols: 标的代码数组 (可选，多标的，长度需与数据一致)
    /// :param extra: 额外数据字典 {name: array} (可选)
    #[allow(clippy::too_many_arguments)]
    pub fn add_arrays(
        &mut self,
        timestamps: &Bound<'_, PyAny>,
        opens: &Bound<'_, PyAny>,
        highs: &Bound<'_, PyAny>,
        lows: &Bound<'_, PyAny>,
        closes: &Bound<'_, PyAny>,
        volumes: &Bound<'_, PyAny>,
        symbol: Option<String>,
        symbols: Option<Vec<String>>,
        extra: Option<HashMap<String, Py<PyAny>>>,
        py: Python<'_>,
    ) -> PyResult<()> {
        let columns = BarColumns::from_py_arrays(
            timestamps, opens, highs, lows, closes, volumes, symbol, symbols, extra, py,
        )?;
        if self.live_sender.is_some() {
            // 实时模式: 重构为 Bar 事件逐个发送
            self.add_bars(columns.to_bars())
        } else {
            let mut provider = self.provider.lock().unwrap();
            provider
                .add_bar_columns(columns)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }

    /// 对数据源进行排序 (按时间戳).
    pub fn sort(&self) {
        let mut provider = self.provider.lock().unwrap();
        provider.sort();
    }

    /// 添加 Tick 数据.
    ///
    /// :param tick: Tick 数据
    pub fn add_tick(&mut self, tick: Tick) -> PyResult<()> {
        if let Some(sender_lock) = &self.live_sender {
            let sender = sender_lock.lock().unwrap();
            sender
                .send(Event::Tick(tick))
                .map_err(|e| PyValueError::new_err(e.to_string()))
        } else {
            let mut provider = self.provider.lock().unwrap();
            provider.add(Event::Tick(tick))?;
            Ok(())
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

    pub fn progress_len_hint(&self) -> Option<usize> {
        let provider = self.provider.lock().unwrap();
        provider.progress_len_hint()
    }

    pub fn wait_peek(&self, timeout: Duration) -> Option<i64> {
        let mut provider = self.provider.lock().unwrap();
        provider.wait_peek(timeout)
    }

    pub fn is_live(&self) -> bool {
        let provider = self.provider.lock().unwrap();
        provider.is_live()
    }

    /// 等待下一个事件或直到下一个定时器触发 (用于实时模式)
    ///
    /// :param next_timer_timestamp: 下一个定时器的时间戳 (纳秒)
    /// :param py: Python 解释器 (用于释放 GIL)
    /// :return: 下一个事件的时间戳 (如果有)
    pub fn wait_next_event_or_timer(
        &self,
        next_timer_timestamp: Option<i64>,
        py: Python<'_>,
    ) -> Option<i64> {
        // If backtest mode (not live), return immediately
        if !self.is_live() {
            return self.peek_timestamp();
        }

        // Calculate timeout
        let timeout = if let Some(timer_ts) = next_timer_timestamp {
            let now = Utc::now().timestamp_nanos_opt().unwrap_or_else(|| {
                warn_clock_fallback(next_timer_timestamp);
                0
            });
            if timer_ts > now {
                let diff_ms = (timer_ts - now) / 1_000_000;
                if diff_ms > 0 {
                    Duration::from_millis(std::cmp::min(diff_ms as u64, 1000))
                } else {
                    Duration::ZERO
                }
            } else {
                Duration::ZERO
            }
        } else {
            Duration::from_secs(1)
        };

        if timeout > Duration::ZERO {
            let feed_clone = self.clone();
            // Release GIL and wait
            py.detach(move || feed_clone.wait_peek(timeout))
        } else {
            self.peek_timestamp()
        }
    }

    /// 获取下一个动作 (事件或定时器)
    pub fn next_action(&self, next_timer_ts: Option<i64>, py: Python) -> FeedAction {
        if !self.is_live() {
            let peek_ts = self.peek_timestamp();

            // Backtest logic
            match (peek_ts, next_timer_ts) {
                (Some(et), Some(tt)) => {
                    if tt <= et {
                        FeedAction::Timer(tt)
                    } else {
                        FeedAction::Event(Box::new(self.next().unwrap()))
                    }
                }
                (Some(_), None) => FeedAction::Event(Box::new(self.next().unwrap())),
                (None, Some(tt)) => FeedAction::Timer(tt),
                (None, None) => FeedAction::End,
            }
        } else {
            // Live logic
            // Calculate timeout
            let now = Utc::now().timestamp_nanos_opt().unwrap_or_else(|| {
                warn_clock_fallback(next_timer_ts);
                0
            });
            let timeout = if let Some(tt) = next_timer_ts {
                if tt > now {
                    let diff_ms = (tt - now) / 1_000_000;
                    Duration::from_millis(std::cmp::min(diff_ms as u64, 1000))
                } else {
                    Duration::ZERO
                }
            } else {
                Duration::from_secs(1)
            };

            let next_ts_opt = if timeout > Duration::ZERO {
                let feed_clone = self.clone();
                // Release GIL and wait
                py.detach(move || feed_clone.wait_peek(timeout))
            } else {
                self.peek_timestamp()
            };

            match next_ts_opt {
                Some(et) => {
                    // Event available
                    if let Some(tt) = next_timer_ts {
                        if tt <= et {
                            // Timer is earlier than event (and we are here so timer might be due or close?)
                            // In Live, check if timer is actually DUE (<= now).
                            if tt <= now {
                                FeedAction::Timer(tt)
                            } else {
                                // Timer not due, but event is here.
                                FeedAction::Event(Box::new(self.next().unwrap()))
                            }
                        } else {
                            // Event is earlier than timer
                            FeedAction::Event(Box::new(self.next().unwrap()))
                        }
                    } else {
                        FeedAction::Event(Box::new(self.next().unwrap()))
                    }
                }
                None => {
                    // Timeout (No event)
                    if let Some(tt) = next_timer_ts
                        && tt <= now
                    {
                        return FeedAction::Timer(tt);
                    }
                    FeedAction::Wait
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_fallback_context_uses_next_timer_timestamp_as_event_time() {
        let rendered = render_log_message(
            "clock-fallback",
            clock_fallback_context(Some(1_000_000_000)),
        );

        assert!(rendered.contains("\"phase\":\"data\""));
        assert!(rendered.contains("\"event_time_iso\":\"1970-01-01T00:00:01Z\""));
    }

    #[test]
    fn clock_fallback_context_omits_event_time_without_timer() {
        let rendered = render_log_message("clock-fallback", clock_fallback_context(None));

        assert!(rendered.contains("\"phase\":\"data\""));
        assert!(!rendered.contains("event_time_iso"));
    }
}
