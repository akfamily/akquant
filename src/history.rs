use crate::model::{Bar, Tick};
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct SymbolHistory {
    pub timestamps: VecDeque<i64>,
    pub opens: VecDeque<f64>,
    pub highs: VecDeque<f64>,
    pub lows: VecDeque<f64>,
    pub closes: VecDeque<f64>,
    pub volumes: VecDeque<f64>,
    pub extras: HashMap<String, VecDeque<f64>>,
    pub capacity: usize,
}

impl SymbolHistory {
    pub fn new(capacity: usize) -> Self {
        SymbolHistory {
            timestamps: VecDeque::with_capacity(capacity),
            opens: VecDeque::with_capacity(capacity),
            highs: VecDeque::with_capacity(capacity),
            lows: VecDeque::with_capacity(capacity),
            closes: VecDeque::with_capacity(capacity),
            volumes: VecDeque::with_capacity(capacity),
            extras: HashMap::new(),
            capacity,
        }
    }

    pub fn push(&mut self, bar: &Bar) {
        if self.capacity == 0 {
            return;
        }

        if self.timestamps.len() >= self.capacity {
            self.timestamps.pop_front();
            self.opens.pop_front();
            self.highs.pop_front();
            self.lows.pop_front();
            self.closes.pop_front();
            self.volumes.pop_front();
            for v in self.extras.values_mut() {
                v.pop_front();
            }
        }

        let cur_len = self.timestamps.len();
        for (k, _) in bar.extra.iter() {
            if !self.extras.contains_key(k) {
                let mut dq = VecDeque::with_capacity(self.capacity);
                for _ in 0..cur_len {
                    dq.push_back(f64::NAN);
                }
                self.extras.insert(k.clone(), dq);
            }
        }
        for (k, dq) in self.extras.iter_mut() {
            let val = bar.extra.get(k).cloned().unwrap_or(f64::NAN);
            dq.push_back(val);
        }
        self.timestamps.push_back(bar.timestamp);
        self.opens.push_back(bar.open.to_f64().unwrap_or(0.0));
        self.highs.push_back(bar.high.to_f64().unwrap_or(0.0));
        self.lows.push_back(bar.low.to_f64().unwrap_or(0.0));
        self.closes.push_back(bar.close.to_f64().unwrap_or(0.0));
        self.volumes.push_back(bar.volume.to_f64().unwrap_or(0.0));
    }

    /// 把 tick 以退化 bar 的形式写入历史.
    ///
    /// tick 只有 price 与 volume, 故 `open = high = low = close = price`。
    /// 引擎内部早有同一模式的先例: `pipeline/stages/data.rs` 在填补缺失价格时
    /// 就构造 `open=high=low=close=last_price` 的退化 bar 写入本结构。这样
    /// `context.rs` 的字段查询(`"close"` -> `closes`)无需任何改动即可用于
    /// tick 历史。
    ///
    /// **extras 必须补齐**: `Tick` 没有 `extra` 字段, 而同一 symbol 的 bar 可能
    /// 已经建立了 extras 列。若不为这些列 push NaN, 各列长度会失配,
    /// `get_history` 将返回错位数据。
    pub fn push_tick(&mut self, tick: &Tick) {
        if self.capacity == 0 {
            return;
        }

        if self.timestamps.len() >= self.capacity {
            self.timestamps.pop_front();
            self.opens.pop_front();
            self.highs.pop_front();
            self.lows.pop_front();
            self.closes.pop_front();
            self.volumes.pop_front();
            for v in self.extras.values_mut() {
                v.pop_front();
            }
        }

        // Tick 无 extra: 为所有已存在的 extras 列补 NaN, 保持各列等长。
        for dq in self.extras.values_mut() {
            dq.push_back(f64::NAN);
        }

        let price = tick.price.to_f64().unwrap_or(0.0);
        self.timestamps.push_back(tick.timestamp);
        self.opens.push_back(price);
        self.highs.push_back(price);
        self.lows.push_back(price);
        self.closes.push_back(price);
        self.volumes.push_back(tick.volume.to_f64().unwrap_or(0.0));
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolHistorySnapshot {
    pub timestamps: VecDeque<i64>,
    pub opens: VecDeque<f64>,
    pub highs: VecDeque<f64>,
    pub lows: VecDeque<f64>,
    pub closes: VecDeque<f64>,
    pub volumes: VecDeque<f64>,
    pub extras: HashMap<String, VecDeque<f64>>,
    pub capacity: usize,
}

impl From<&SymbolHistory> for SymbolHistorySnapshot {
    fn from(history: &SymbolHistory) -> Self {
        Self {
            timestamps: history.timestamps.clone(),
            opens: history.opens.clone(),
            highs: history.highs.clone(),
            lows: history.lows.clone(),
            closes: history.closes.clone(),
            volumes: history.volumes.clone(),
            extras: history.extras.clone(),
            capacity: history.capacity,
        }
    }
}

impl From<SymbolHistorySnapshot> for SymbolHistory {
    fn from(snapshot: SymbolHistorySnapshot) -> Self {
        Self {
            timestamps: snapshot.timestamps,
            opens: snapshot.opens,
            highs: snapshot.highs,
            lows: snapshot.lows,
            closes: snapshot.closes,
            volumes: snapshot.volumes,
            extras: snapshot.extras,
            capacity: snapshot.capacity,
        }
    }
}

#[derive(Debug)]
pub struct HistoryBuffer {
    pub data: HashMap<String, SymbolHistory>,
    pub previous_data: HashMap<String, SymbolHistory>,
    /// tick 序列。与 `data` 并列而非共用: tick 以退化 OHLC 写入, 若与 bar 混在
    /// 同一 SymbolHistory, 高频 tick 会把历史 bar 挤出窗口, 且完全静默
    /// (取到的值看起来就是价格)。
    pub tick_data: HashMap<String, SymbolHistory>,
    pub previous_tick_data: HashMap<String, SymbolHistory>,
    pub default_capacity: usize,
}

impl HistoryBuffer {
    pub fn new(default_capacity: usize) -> Self {
        HistoryBuffer {
            data: HashMap::new(),
            previous_data: HashMap::new(),
            tick_data: HashMap::new(),
            previous_tick_data: HashMap::new(),
            default_capacity,
        }
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        self.default_capacity = capacity;
        self.data.clear();
        self.previous_data.clear();
        self.tick_data.clear();
        self.previous_tick_data.clear();
    }

    pub fn set_capacity_preserve_existing(&mut self, capacity: usize) {
        self.default_capacity = capacity;
        self.previous_data.clear();
        self.previous_tick_data.clear();
        if capacity == 0 {
            self.data.clear();
            self.tick_data.clear();
            return;
        }

        for history in self.data.values_mut() {
            history.capacity = capacity;
            while history.timestamps.len() > capacity {
                history.timestamps.pop_front();
                history.opens.pop_front();
                history.highs.pop_front();
                history.lows.pop_front();
                history.closes.pop_front();
                history.volumes.pop_front();
                for values in history.extras.values_mut() {
                    values.pop_front();
                }
            }
        }

        for history in self.tick_data.values_mut() {
            history.capacity = capacity;
            while history.timestamps.len() > capacity {
                history.timestamps.pop_front();
                history.opens.pop_front();
                history.highs.pop_front();
                history.lows.pop_front();
                history.closes.pop_front();
                history.volumes.pop_front();
                for values in history.extras.values_mut() {
                    values.pop_front();
                }
            }
        }
    }

    pub fn update(&mut self, bar: &Bar) {
        if self.default_capacity == 0 {
            return;
        }

        let symbol = bar.symbol.clone();
        let history = self
            .data
            .entry(symbol.clone())
            .or_insert_with(|| SymbolHistory::new(self.default_capacity));
        if history.timestamps.is_empty() {
            self.previous_data.remove(&symbol);
        } else {
            self.previous_data.insert(symbol, history.clone());
        }

        history.push(bar);
    }

    /// `update` 的 tick 版本, 语义与之对称, 但落到独立的 `tick_data` 容器.
    ///
    /// 与 `update` 保持同样的 previous_data 快照时序: 首次写入某 symbol 时
    /// 清除其 previous 快照, 否则把当前状态存为 previous。
    ///
    /// **不写入 `data`**: tick 以退化 OHLC (`open=high=low=close=price`) 写入,
    /// 若与 bar 共用同一个 `SymbolHistory`, 高频 tick 会把历史 bar 挤出窗口,
    /// 且完全静默——取到的值看起来就是合法价格, 指标照算, 结果却全错。
    pub fn update_tick(&mut self, tick: &Tick) {
        if self.default_capacity == 0 {
            return;
        }

        let symbol = tick.symbol.clone();
        let history = self
            .tick_data
            .entry(symbol.clone())
            .or_insert_with(|| SymbolHistory::new(self.default_capacity));
        if history.timestamps.is_empty() {
            self.previous_tick_data.remove(&symbol);
        } else {
            self.previous_tick_data.insert(symbol, history.clone());
        }

        history.push_tick(tick);
    }

    pub fn get_history(&self, symbol: &str) -> Option<&SymbolHistory> {
        self.data.get(symbol)
    }

    pub fn get_previous_history(&self, symbol: &str) -> Option<&SymbolHistory> {
        self.previous_data.get(symbol)
    }

    pub fn get_tick_history(&self, symbol: &str) -> Option<&SymbolHistory> {
        self.tick_data.get(symbol)
    }

    pub fn get_previous_tick_history(&self, symbol: &str) -> Option<&SymbolHistory> {
        self.previous_tick_data.get(symbol)
    }

    /// 该 symbol 是否已有 bar 序列（供上层判断双流歧义）.
    pub fn has_bar_history(&self, symbol: &str) -> bool {
        self.data
            .get(symbol)
            .is_some_and(|h| !h.timestamps.is_empty())
    }

    /// 该 symbol 是否已有 tick 序列（供上层判断双流歧义）.
    pub fn has_tick_history(&self, symbol: &str) -> bool {
        self.tick_data
            .get(symbol)
            .is_some_and(|h| !h.timestamps.is_empty())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryBufferSnapshot {
    pub data: HashMap<String, SymbolHistorySnapshot>,
    pub default_capacity: usize,
}

impl From<&HistoryBuffer> for HistoryBufferSnapshot {
    fn from(buffer: &HistoryBuffer) -> Self {
        Self {
            data: buffer
                .data
                .iter()
                .map(|(symbol, history)| (symbol.clone(), SymbolHistorySnapshot::from(history)))
                .collect(),
            default_capacity: buffer.default_capacity,
        }
    }
}

impl From<HistoryBufferSnapshot> for HistoryBuffer {
    fn from(snapshot: HistoryBufferSnapshot) -> Self {
        Self {
            data: snapshot
                .data
                .into_iter()
                .map(|(symbol, history)| (symbol, SymbolHistory::from(history)))
                .collect(),
            previous_data: HashMap::new(),
            // 快照格式本身不携带 tick 序列（`HistoryBufferSnapshot` 无对应字段），
            // 恢复后 tick 容器留空，与 previous_data 的处理方式一致。
            tick_data: HashMap::new(),
            previous_tick_data: HashMap::new(),
            default_capacity: snapshot.default_capacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HistoryBuffer, HistoryBufferSnapshot};
    use crate::model::Bar;
    use rust_decimal::Decimal;
    use std::collections::HashMap;

    fn make_bar(timestamp: i64, close: i64, extra: Option<(&str, f64)>) -> Bar {
        let mut extra_map = HashMap::new();
        if let Some((key, value)) = extra {
            extra_map.insert(key.to_string(), value);
        }
        Bar {
            timestamp,
            open: Decimal::from(close),
            high: Decimal::from(close),
            low: Decimal::from(close),
            close: Decimal::from(close),
            volume: Decimal::from(1000),
            symbol: "TEST".to_string(),
            extra: extra_map,
        }
    }

    fn make_tick(timestamp: i64, price: i64) -> crate::model::Tick {
        crate::model::Tick {
            timestamp,
            price: Decimal::from(price),
            volume: Decimal::from(500),
            symbol: "TEST".to_string(),
        }
    }

    #[test]
    fn push_tick_writes_degenerate_ohlc() {
        use super::SymbolHistory;
        let mut history = SymbolHistory::new(4);
        history.push_tick(&make_tick(1, 10));

        assert_eq!(history.timestamps.len(), 1);
        assert_eq!(history.opens.len(), 1);
        assert_eq!(history.highs.len(), 1);
        assert_eq!(history.lows.len(), 1);
        assert_eq!(history.closes.len(), 1);
        assert_eq!(history.volumes.len(), 1);
        assert_eq!(history.opens[0], 10.0);
        assert_eq!(history.highs[0], 10.0);
        assert_eq!(history.lows[0], 10.0);
        assert_eq!(history.closes[0], 10.0);
        assert_eq!(history.volumes[0], 500.0);
    }

    #[test]
    fn push_tick_pads_existing_extras_with_nan() {
        // 回归防护: Tick 无 extra 字段。若 push_tick 不为已存在的 extras 列补
        // NaN, 各列长度会失配, get_history 返回错位数据。
        use super::SymbolHistory;
        let mut history = SymbolHistory::new(4);
        history.push(&make_bar(1, 10, Some(("factor", 1.5))));
        history.push_tick(&make_tick(2, 11));

        let factor = history.extras.get("factor").expect("factor column missing");
        assert_eq!(factor.len(), history.timestamps.len());
        assert_eq!(factor[0], 1.5);
        assert!(factor[1].is_nan());
    }

    #[test]
    fn push_tick_respects_capacity() {
        use super::SymbolHistory;
        let mut history = SymbolHistory::new(2);
        history.push_tick(&make_tick(1, 10));
        history.push_tick(&make_tick(2, 11));
        history.push_tick(&make_tick(3, 12));

        assert_eq!(history.timestamps.len(), 2);
        assert_eq!(history.timestamps[0], 2);
        assert_eq!(history.closes[1], 12.0);
    }

    #[test]
    fn push_tick_evicts_extras_columns_in_step() {
        // 回归防护: 淘汰与 extras 补齐的交互点。二者顺序若被重排, 列长会失配,
        // get_history 返回错位数据——且是静默的。
        use super::SymbolHistory;
        let mut history = SymbolHistory::new(2);
        history.push(&make_bar(1, 10, Some(("factor", 1.5))));
        history.push_tick(&make_tick(2, 11));
        history.push_tick(&make_tick(3, 12));

        let factor = history.extras.get("factor").expect("factor column missing");
        assert_eq!(history.timestamps.len(), 2);
        assert_eq!(factor.len(), history.timestamps.len());
        assert_eq!(history.timestamps[0], 2);
        assert!(factor[0].is_nan());
        assert!(factor[1].is_nan());
    }

    #[test]
    fn update_tick_creates_symbol_history() {
        // 分离之后 update_tick 落到 tick_data, 不再写入 bar 的 data 容器
        // (此前的断言 buffer.get_history(...) 正是要修复的 bug 本身)。
        let mut buffer = HistoryBuffer::new(3);
        buffer.update_tick(&make_tick(1, 10));

        let history = buffer
            .get_tick_history("TEST")
            .expect("tick history missing");
        assert_eq!(history.closes[0], 10.0);
    }

    #[test]
    fn tick_and_bar_histories_do_not_evict_each_other() {
        // 双流回归判据: 同一 symbol 交替写 bar 与 tick, 两条序列必须各自完整。
        // 修复前 tick 以退化 OHLC 挤占同一 SymbolHistory, 把历史 bar 挤出窗口。
        // 容量取 12(= 3*4 条 tick 总数), 确保 tick 序列不被自身容量淘汰,
        // 否则断言 12 与 FIFO 淘汰逻辑矛盾。
        let mut buffer = HistoryBuffer::new(12);
        for n in 1..=3 {
            for k in 0..4 {
                buffer.update_tick(&make_tick(n * 1000 + k, n * 10 + k));
            }
            buffer.update(&make_bar(n * 1000 + 9, n * 10 + 3, None));
        }
        let bars = buffer.get_history("TEST").expect("bar history missing");
        assert_eq!(
            bars.closes.iter().copied().collect::<Vec<f64>>(),
            vec![13.0, 23.0, 33.0]
        );
        let ticks = buffer
            .get_tick_history("TEST")
            .expect("tick history missing");
        assert_eq!(ticks.closes.len(), 12);
    }

    #[test]
    fn history_presence_flags_report_each_stream() {
        let mut buffer = HistoryBuffer::new(4);
        assert!(!buffer.has_bar_history("TEST"));
        assert!(!buffer.has_tick_history("TEST"));
        buffer.update(&make_bar(1, 10, None));
        assert!(buffer.has_bar_history("TEST"));
        assert!(!buffer.has_tick_history("TEST"));
        buffer.update_tick(&make_tick(2, 11));
        assert!(buffer.has_tick_history("TEST"));
    }

    #[test]
    fn test_history_buffer_snapshot_roundtrip_preserves_history() {
        let mut buffer = HistoryBuffer::new(4);
        buffer.update(&make_bar(1, 10, Some(("factor", 1.0))));
        buffer.update(&make_bar(2, 11, None));
        buffer.update(&make_bar(3, 12, Some(("factor", 3.0))));

        let snapshot = HistoryBufferSnapshot::from(&buffer);
        let encoded = rmp_serde::to_vec(&snapshot).expect("serialize history snapshot");
        let decoded: HistoryBufferSnapshot =
            rmp_serde::from_slice(&encoded).expect("deserialize history snapshot");
        let restored = HistoryBuffer::from(decoded);

        assert_eq!(restored.default_capacity, 4);
        let history = restored.get_history("TEST").expect("history for TEST");
        assert_eq!(history.capacity, 4);
        assert_eq!(
            history.timestamps.iter().copied().collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            history.closes.iter().copied().collect::<Vec<_>>(),
            vec![10.0, 11.0, 12.0]
        );
        let extras = history.extras.get("factor").expect("factor extra history");
        assert_eq!(extras.len(), 3);
        assert_eq!(extras[0], 1.0);
        assert!(extras[1].is_nan());
        assert_eq!(extras[2], 3.0);
    }

    #[test]
    fn test_set_capacity_preserve_existing_keeps_tail_window() {
        let mut buffer = HistoryBuffer::new(4);
        for (timestamp, close) in [(1, 10), (2, 11), (3, 12), (4, 13)] {
            buffer.update(&make_bar(timestamp, close, Some(("factor", close as f64))));
        }

        buffer.set_capacity_preserve_existing(2);

        assert_eq!(buffer.default_capacity, 2);
        let history = buffer.get_history("TEST").expect("history for TEST");
        assert_eq!(history.capacity, 2);
        assert_eq!(
            history.timestamps.iter().copied().collect::<Vec<_>>(),
            vec![3, 4]
        );
        assert_eq!(
            history.closes.iter().copied().collect::<Vec<_>>(),
            vec![12.0, 13.0]
        );
        let extras = history.extras.get("factor").expect("factor extra history");
        assert_eq!(extras.iter().copied().collect::<Vec<_>>(), vec![12.0, 13.0]);
    }
}
