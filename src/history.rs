use crate::model::{Bar, Tick};
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::iter::{Chain, Copied, Take};

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
    /// 最近一次写入的记录, 供 [`SymbolHistory::previous_view`] 现拼"写入前"的视图。
    last_push: LastPush,
}

/// 最近一次写入的记录。
///
/// 旧实现每次写入前把整个序列复制一份留作"上一视图", 代价与容量成正比
/// (默认容量 10000 时每根 bar 复制约 480KB)。现在只记下被挤掉的最老一行
/// 和本次新建的 extra 列, 上一视图 = 被挤掉的一行 + 当前序列去掉最后一行,
/// 与整份复制逐值相同。
#[derive(Debug, Clone, Default)]
enum LastPush {
    /// 没有上一视图: 写入前序列为空, 或调整容量 / 从存档恢复之后。
    #[default]
    None,
    Pushed {
        /// 写入时因满容量被挤掉的最老一行; 未满时为 None。
        evicted: Option<EvictedRow>,
        /// 本次写入新建的 extra 列(上一视图里没有这些列)。
        new_extras: Vec<String>,
    },
}

/// 被挤出序列的一行。
#[derive(Debug, Clone)]
struct EvictedRow {
    timestamp: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    extras: HashMap<String, f64>,
}

/// 视图里一列的迭代器: 可选的被挤掉的一行在前, 后接序列的前若干行。
pub type ColumnIter<'a, T> =
    Chain<std::option::IntoIter<T>, Take<Copied<std::collections::vec_deque::Iter<'a, T>>>>;

/// 序列的只读视图: 当前视图(序列本身)或上一视图(最近一次写入之前的样子)。
#[derive(Debug, Clone, Copy)]
pub struct HistoryView<'a> {
    history: &'a SymbolHistory,
    evicted: Option<&'a EvictedRow>,
    /// 取序列前多少行(上一视图去掉最后一行)。
    take: usize,
    /// 视图里不存在的 extra 列(上一视图隐藏最近一次写入新建的列)。
    hidden_extras: &'a [String],
}

impl<'a> HistoryView<'a> {
    /// 行数。
    pub fn len(&self) -> usize {
        self.take + usize::from(self.evicted.is_some())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn timestamps(&self) -> ColumnIter<'a, i64> {
        self.evicted
            .map(|row| row.timestamp)
            .into_iter()
            .chain(self.history.timestamps.iter().copied().take(self.take))
    }

    /// 按字段名取一列: `open` / `high` / `low` / `close` / `volume`, 其余按 extra 列查;
    /// 视图里没有该列时返回 None。
    pub fn field(&self, name: &str) -> Option<ColumnIter<'a, f64>> {
        let (column, evicted) = match name {
            "open" => (&self.history.opens, self.evicted.map(|row| row.open)),
            "high" => (&self.history.highs, self.evicted.map(|row| row.high)),
            "low" => (&self.history.lows, self.evicted.map(|row| row.low)),
            "close" => (&self.history.closes, self.evicted.map(|row| row.close)),
            "volume" => (&self.history.volumes, self.evicted.map(|row| row.volume)),
            _ => {
                if self.hidden_extras.iter().any(|key| key == name) {
                    return None;
                }
                let column = self.history.extras.get(name)?;
                // 挤掉那一行时该列已存在(写入先淘汰、后建新列), 一定有值
                let evicted = self
                    .evicted
                    .map(|row| row.extras.get(name).copied().unwrap_or(f64::NAN));
                (column, evicted)
            }
        };
        Some(
            evicted
                .into_iter()
                .chain(column.iter().copied().take(self.take)),
        )
    }

    /// 视图里存在的 extra 列名(排序)。
    pub fn extra_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .history
            .extras
            .keys()
            .filter(|key| !self.hidden_extras.contains(key))
            .cloned()
            .collect();
        keys.sort();
        keys
    }
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
            last_push: LastPush::None,
        }
    }

    /// 当前视图(序列本身)。
    pub fn view(&self) -> HistoryView<'_> {
        HistoryView {
            history: self,
            evicted: None,
            take: self.timestamps.len(),
            hidden_extras: &[],
        }
    }

    /// 上一视图: 最近一次写入之前的序列。写入前序列为空、或调整容量 / 从存档
    /// 恢复之后还没有写入时为 None。
    pub fn previous_view(&self) -> Option<HistoryView<'_>> {
        match &self.last_push {
            LastPush::None => None,
            LastPush::Pushed {
                evicted,
                new_extras,
            } => Some(HistoryView {
                history: self,
                evicted: evicted.as_ref(),
                take: self.timestamps.len().saturating_sub(1),
                hidden_extras: new_extras,
            }),
        }
    }

    /// 满容量时弹出最老一行并返回。
    fn evict_if_full(&mut self) -> Option<EvictedRow> {
        if self.timestamps.len() < self.capacity {
            return None;
        }
        let timestamp = self.timestamps.pop_front()?;
        let mut extras = HashMap::new();
        for (key, values) in self.extras.iter_mut() {
            if let Some(value) = values.pop_front() {
                extras.insert(key.clone(), value);
            }
        }
        Some(EvictedRow {
            timestamp,
            open: self.opens.pop_front().unwrap_or(f64::NAN),
            high: self.highs.pop_front().unwrap_or(f64::NAN),
            low: self.lows.pop_front().unwrap_or(f64::NAN),
            close: self.closes.pop_front().unwrap_or(f64::NAN),
            volume: self.volumes.pop_front().unwrap_or(f64::NAN),
            extras,
        })
    }

    /// 写入完成后记下本次写入: 写入前为空则没有上一视图。
    fn record_push(
        &mut self,
        was_empty: bool,
        evicted: Option<EvictedRow>,
        new_extras: Vec<String>,
    ) {
        self.last_push = if was_empty {
            LastPush::None
        } else {
            LastPush::Pushed {
                evicted,
                new_extras,
            }
        };
    }

    pub fn push(&mut self, bar: &Bar) {
        if self.capacity == 0 {
            return;
        }

        let was_empty = self.timestamps.is_empty();
        let evicted = self.evict_if_full();

        let cur_len = self.timestamps.len();
        let mut new_extras = Vec::new();
        for (k, _) in bar.extra.iter() {
            if !self.extras.contains_key(k) {
                let mut dq = VecDeque::with_capacity(self.capacity);
                for _ in 0..cur_len {
                    dq.push_back(f64::NAN);
                }
                self.extras.insert(k.clone(), dq);
                new_extras.push(k.clone());
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
        self.record_push(was_empty, evicted, new_extras);
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

        let was_empty = self.timestamps.is_empty();
        let evicted = self.evict_if_full();

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
        self.record_push(was_empty, evicted, Vec::new());
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
            // 上一视图是派生状态, 不入快照; 恢复后到下一次写入之前没有上一视图
            last_push: LastPush::None,
        }
    }
}

#[derive(Debug)]
pub struct HistoryBuffer {
    pub data: HashMap<String, SymbolHistory>,
    /// tick 序列。与 `data` 并列而非共用: tick 以退化 OHLC 写入, 若与 bar 混在
    /// 同一 SymbolHistory, 高频 tick 会把历史 bar 挤出窗口, 且完全静默
    /// (取到的值看起来就是价格)。
    pub tick_data: HashMap<String, SymbolHistory>,
    /// 窗口(多周期)序列, 键 (symbol, freq_label)。只存闭合的窗口 bar。
    pub window_data: HashMap<(String, String), SymbolHistory>,
    pub default_capacity: usize,
}

/// 把 `history` 裁剪到 `capacity`(FIFO 淘汰最旧), 三个 map(`data` /
/// `tick_data` / `window_data`)的裁剪循环共用此函数。
fn trim_to_capacity(history: &mut SymbolHistory, capacity: usize) {
    history.capacity = capacity;
    // 调整容量后没有上一视图(与旧实现清空整份复制一致)
    history.last_push = LastPush::None;
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

impl HistoryBuffer {
    pub fn new(default_capacity: usize) -> Self {
        HistoryBuffer {
            data: HashMap::new(),
            tick_data: HashMap::new(),
            window_data: HashMap::new(),
            default_capacity,
        }
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        self.default_capacity = capacity;
        self.data.clear();
        self.tick_data.clear();
        self.window_data.clear();
    }

    pub fn set_capacity_preserve_existing(&mut self, capacity: usize) {
        self.default_capacity = capacity;
        if capacity == 0 {
            self.data.clear();
            self.tick_data.clear();
            self.window_data.clear();
            return;
        }

        for history in self.data.values_mut() {
            trim_to_capacity(history, capacity);
        }

        for history in self.tick_data.values_mut() {
            trim_to_capacity(history, capacity);
        }

        for history in self.window_data.values_mut() {
            trim_to_capacity(history, capacity);
        }
    }

    pub fn update(&mut self, bar: &Bar) {
        if self.default_capacity == 0 {
            return;
        }

        let capacity = self.default_capacity;
        let history = match self.data.get_mut(&bar.symbol) {
            Some(history) => history,
            None => self
                .data
                .entry(bar.symbol.clone())
                .or_insert_with(|| SymbolHistory::new(capacity)),
        };
        history.push(bar);
    }

    /// `update` 的 tick 版本, 语义与之对称, 但落到独立的 `tick_data` 容器.
    ///
    /// 与 `update` 一样由写入本身记下上一视图(见 [`SymbolHistory::previous_view`])。
    ///
    /// **不写入 `data`**: tick 以退化 OHLC (`open=high=low=close=price`) 写入,
    /// 若与 bar 共用同一个 `SymbolHistory`, 高频 tick 会把历史 bar 挤出窗口,
    /// 且完全静默——取到的值看起来就是合法价格, 指标照算, 结果却全错。
    pub fn update_tick(&mut self, tick: &Tick) {
        if self.default_capacity == 0 {
            return;
        }

        let capacity = self.default_capacity;
        let history = match self.tick_data.get_mut(&tick.symbol) {
            Some(history) => history,
            None => self
                .tick_data
                .entry(tick.symbol.clone())
                .or_insert_with(|| SymbolHistory::new(capacity)),
        };
        history.push_tick(tick);
    }

    /// `update` 的窗口版本: 落到 `window_data[(symbol, freq)]`。`bar.freq` 为 None 时忽略
    /// (窗口 bar 一定带标签; 没标签说明调用方传错了对象)。
    ///
    /// 注: 任务书原文建议这里用 `debug_assert!(false, ...)`, 但本 crate 没有
    /// `[profile.test]` 覆盖, `cargo test` 默认开启 debug assertions, 会让这条
    /// assert 在 `update_window_ignores_bar_without_freq` 测试里直接 panic,
    /// 与该测试名字/断言(静默忽略、不 panic)矛盾。这里改为 `log::debug!`
    /// 发出信号但不中断执行, 以满足测试对"忽略"的定义。
    pub fn update_window(&mut self, bar: &Bar) {
        if self.default_capacity == 0 {
            return;
        }
        let Some(freq) = bar.freq.clone() else {
            log::debug!("update_window 收到不带 freq 的 bar: symbol={}", bar.symbol);
            return;
        };
        let history = self
            .window_data
            .entry((bar.symbol.clone(), freq))
            .or_insert_with(|| SymbolHistory::new(self.default_capacity));
        history.push(bar);
    }

    pub fn get_window_history(&self, symbol: &str, freq: &str) -> Option<&SymbolHistory> {
        self.window_data
            .get(&(symbol.to_string(), freq.to_string()))
    }

    pub fn has_window_history(&self, symbol: &str, freq: &str) -> bool {
        self.get_window_history(symbol, freq)
            .is_some_and(|h| !h.timestamps.is_empty())
    }

    /// 该 symbol 已有数据的窗口周期(排序), 供歧义报错文案使用。
    pub fn window_freqs(&self, symbol: &str) -> Vec<String> {
        let mut v: Vec<String> = self
            .window_data
            .iter()
            .filter(|((s, _), h)| s == symbol && !h.timestamps.is_empty())
            .map(|((_, f), _)| f.clone())
            .collect();
        v.sort();
        v
    }

    pub fn get_history(&self, symbol: &str) -> Option<&SymbolHistory> {
        self.data.get(symbol)
    }

    pub fn get_tick_history(&self, symbol: &str) -> Option<&SymbolHistory> {
        self.tick_data.get(symbol)
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
pub struct WindowSeriesSnapshot {
    pub symbol: String,
    pub freq: String,
    pub history: SymbolHistorySnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryBufferSnapshot {
    pub data: HashMap<String, SymbolHistorySnapshot>,
    // tick_data 与 data 对等(都是主序列, 不是派生缓存), 必须一起入快照,
    // 否则热重启一次 get_state_bytes -> load_state_bytes 往返就会把 tick
    // 历史整体清空且不报错。`#[serde(default)]` 让旧存档(无此字段)仍可
    // 反序列化, 恢复为空 tick 序列而不是报错。
    #[serde(default)]
    pub tick_data: HashMap<String, SymbolHistorySnapshot>,
    // window_data 用 Vec<WindowSeriesSnapshot> 而非以 (symbol, freq) 元组为键的
    // map: serde_json 不支持非字符串 map 键, 而既有测试用 serde_json 读旧快照。
    // `#[serde(default)]` 让旧存档(无此字段)仍可反序列化, 恢复为空窗口序列。
    #[serde(default)]
    pub window_data: Vec<WindowSeriesSnapshot>,
    pub default_capacity: usize,
}

impl From<&HistoryBuffer> for HistoryBufferSnapshot {
    fn from(buffer: &HistoryBuffer) -> Self {
        let mut window_data: Vec<WindowSeriesSnapshot> = buffer
            .window_data
            .iter()
            .map(|((symbol, freq), history)| WindowSeriesSnapshot {
                symbol: symbol.clone(),
                freq: freq.clone(),
                history: SymbolHistorySnapshot::from(history),
            })
            .collect();
        window_data.sort_by(|a, b| {
            (a.symbol.as_str(), a.freq.as_str()).cmp(&(b.symbol.as_str(), b.freq.as_str()))
        });
        Self {
            data: buffer
                .data
                .iter()
                .map(|(symbol, history)| (symbol.clone(), SymbolHistorySnapshot::from(history)))
                .collect(),
            tick_data: buffer
                .tick_data
                .iter()
                .map(|(symbol, history)| (symbol.clone(), SymbolHistorySnapshot::from(history)))
                .collect(),
            window_data,
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
            tick_data: snapshot
                .tick_data
                .into_iter()
                .map(|(symbol, history)| (symbol, SymbolHistory::from(history)))
                .collect(),
            window_data: snapshot
                .window_data
                .into_iter()
                .map(|entry| {
                    (
                        (entry.symbol, entry.freq),
                        SymbolHistory::from(entry.history),
                    )
                })
                .collect(),
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
            freq: None,
            interval_start: None,
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
    fn test_history_buffer_snapshot_roundtrip_preserves_tick_history() {
        // (a)(b): tick_data 是与 data 对等的主序列, 快照 round-trip 后必须能
        // 还原, 而不是像派生的上一视图那样恢复后留空。
        let mut buffer = HistoryBuffer::new(4);
        buffer.update_tick(&make_tick(1, 10));
        buffer.update_tick(&make_tick(2, 11));
        buffer.update_tick(&make_tick(3, 12));

        let snapshot = HistoryBufferSnapshot::from(&buffer);
        let encoded = rmp_serde::to_vec(&snapshot).expect("serialize history snapshot");
        let decoded: HistoryBufferSnapshot =
            rmp_serde::from_slice(&encoded).expect("deserialize history snapshot");
        let restored = HistoryBuffer::from(decoded);

        let history = restored
            .get_tick_history("TEST")
            .expect("tick history for TEST should survive snapshot round-trip");
        assert_eq!(history.capacity, 4);
        assert_eq!(
            history.timestamps.iter().copied().collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            history.closes.iter().copied().collect::<Vec<_>>(),
            vec![10.0, 11.0, 12.0]
        );
    }

    #[test]
    fn snapshot_without_tick_fields_restores_with_empty_tick_history() {
        // (c): 旧存档没有 tick 序列字段, 必须能恢复(tick 序列为空), 而不是
        // 反序列化失败。
        let json = r#"{"data":{},"default_capacity":16}"#;
        let snapshot: HistoryBufferSnapshot =
            serde_json::from_str(json).expect("旧快照应可反序列化");
        assert_eq!(snapshot.default_capacity, 16);
        assert!(snapshot.tick_data.is_empty());

        let restored = HistoryBuffer::from(snapshot);
        assert!(restored.tick_data.is_empty());
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

    #[test]
    fn batch_update_preserves_order_and_evicts_oldest() {
        let mut buffer = HistoryBuffer::new(3);
        for n in 1..=5 {
            buffer.update(&make_bar(n, n * 10, None));
        }
        let history = buffer.data.get("TEST").expect("symbol history missing");
        // 容量 3: 只保留最近三根, 且顺序是推入顺序(最老在前)。
        assert_eq!(history.timestamps.len(), 3);
        assert_eq!(
            history.timestamps.iter().copied().collect::<Vec<_>>(),
            vec![3, 4, 5]
        );
    }

    fn make_window_bar(timestamp: i64, close: i64, freq: &str) -> Bar {
        let mut b = make_bar(timestamp, close, None);
        b.freq = Some(freq.to_string());
        b
    }

    #[test]
    fn update_window_lands_in_third_bucket_keyed_by_freq() {
        let mut buffer = HistoryBuffer::new(4);
        buffer.update(&make_bar(1, 10, None));
        buffer.update_window(&make_window_bar(5, 50, "5min"));
        buffer.update_window(&make_window_bar(60, 600, "1h"));
        assert_eq!(buffer.get_history("TEST").unwrap().closes.len(), 1);
        assert_eq!(
            buffer.get_window_history("TEST", "5min").unwrap().closes[0],
            50.0
        );
        assert_eq!(
            buffer.get_window_history("TEST", "1h").unwrap().closes[0],
            600.0
        );
        assert!(buffer.get_window_history("TEST", "15min").is_none());
        assert!(buffer.has_window_history("TEST", "5min"));
        assert!(!buffer.has_window_history("TEST", "15min"));
        assert_eq!(
            buffer.window_freqs("TEST"),
            vec!["1h".to_string(), "5min".to_string()]
        );
    }

    #[test]
    fn update_window_ignores_bar_without_freq() {
        let mut buffer = HistoryBuffer::new(4);
        buffer.update_window(&make_bar(1, 10, None));
        assert!(buffer.window_data.is_empty());
    }

    #[test]
    fn window_previous_snapshot_tracks_like_bar() {
        let mut buffer = HistoryBuffer::new(4);
        buffer.update_window(&make_window_bar(5, 50, "5min"));
        let history = buffer.get_window_history("TEST", "5min").unwrap();
        assert!(history.previous_view().is_none());
        buffer.update_window(&make_window_bar(10, 60, "5min"));
        let history = buffer.get_window_history("TEST", "5min").unwrap();
        let prev = history.previous_view().unwrap();
        assert_eq!(prev.field("close").unwrap().collect::<Vec<_>>(), vec![50.0]);
    }

    #[test]
    fn set_capacity_preserve_existing_trims_window_series_too() {
        let mut buffer = HistoryBuffer::new(4);
        for ts in 1..=4 {
            buffer.update_window(&make_window_bar(ts, ts * 10, "5min"));
        }
        buffer.set_capacity_preserve_existing(2);
        let h = buffer.get_window_history("TEST", "5min").unwrap();
        assert_eq!(h.capacity, 2);
        assert_eq!(h.timestamps.iter().copied().collect::<Vec<_>>(), vec![3, 4]);
    }

    #[test]
    fn snapshot_roundtrip_preserves_window_series() {
        let mut buffer = HistoryBuffer::new(4);
        buffer.update_window(&make_window_bar(5, 50, "5min"));
        buffer.update_window(&make_window_bar(10, 60, "5min"));
        let snapshot = HistoryBufferSnapshot::from(&buffer);
        let encoded = rmp_serde::to_vec(&snapshot).unwrap();
        let decoded: HistoryBufferSnapshot = rmp_serde::from_slice(&encoded).unwrap();
        let restored = HistoryBuffer::from(decoded);
        let h = restored
            .get_window_history("TEST", "5min")
            .expect("window series survives");
        assert_eq!(
            h.closes.iter().copied().collect::<Vec<_>>(),
            vec![50.0, 60.0]
        );
    }

    #[test]
    fn snapshot_without_window_field_restores_empty_window_series() {
        let json = r#"{"data":{},"tick_data":{},"default_capacity":16}"#;
        let snapshot: HistoryBufferSnapshot = serde_json::from_str(json).unwrap();
        assert!(snapshot.window_data.is_empty());
        let restored = HistoryBuffer::from(snapshot);
        assert!(restored.window_data.is_empty());
    }

    /// 上一视图等价性的参照实现: 每次写入前把序列整份复制(旧实现的做法)。
    /// `None` 表示没有上一视图(写入前为空、或调整容量/从存档恢复之后)。
    #[derive(Default)]
    struct PreviousReference {
        bar: HashMap<String, Option<super::SymbolHistory>>,
        tick: HashMap<String, Option<super::SymbolHistory>>,
        window: HashMap<(String, String), Option<super::SymbolHistory>>,
    }

    impl PreviousReference {
        fn clear(&mut self) {
            for v in self.bar.values_mut() {
                *v = None;
            }
            for v in self.tick.values_mut() {
                *v = None;
            }
            for v in self.window.values_mut() {
                *v = None;
            }
        }
    }

    fn snapshot_before(history: Option<&super::SymbolHistory>) -> Option<super::SymbolHistory> {
        history.filter(|h| !h.timestamps.is_empty()).cloned()
    }

    fn bits(values: impl Iterator<Item = f64>) -> Vec<u64> {
        values.map(f64::to_bits).collect()
    }

    /// 逐字段比对视图与参照(NaN 按位比较)。
    fn assert_view_matches(
        view: Option<super::HistoryView<'_>>,
        reference: Option<&super::SymbolHistory>,
        what: &str,
    ) {
        match (view, reference) {
            (None, None) => {}
            (Some(view), Some(reference)) => {
                assert_eq!(
                    view.timestamps().collect::<Vec<_>>(),
                    reference.timestamps.iter().copied().collect::<Vec<_>>(),
                    "{what}: timestamps"
                );
                for (name, column) in [
                    ("open", &reference.opens),
                    ("high", &reference.highs),
                    ("low", &reference.lows),
                    ("close", &reference.closes),
                    ("volume", &reference.volumes),
                ] {
                    let got = view.field(name).map(bits);
                    assert_eq!(got, Some(bits(column.iter().copied())), "{what}: {name}");
                }
                let mut expected_keys: Vec<String> = reference.extras.keys().cloned().collect();
                expected_keys.sort();
                assert_eq!(view.extra_keys(), expected_keys, "{what}: extra keys");
                for key in ["x", "y", "z"] {
                    let got = view.field(key).map(bits);
                    let want = reference.extras.get(key).map(|c| bits(c.iter().copied()));
                    assert_eq!(got, want, "{what}: extra {key}");
                }
            }
            (view, reference) => panic!(
                "{what}: 视图存在性不一致, view={} reference={}",
                view.is_some(),
                reference.is_some()
            ),
        }
    }

    /// 上一视图与"写入前整份复制"逐字段一致: 覆盖淘汰/未淘汰、新增 extra 列、
    /// tick 与窗口序列、调整容量(含 0)、清空与存档往返。
    #[test]
    fn previous_view_matches_clone_before_push_reference() {
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move |modulo: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % modulo
        };
        let mut buffer = HistoryBuffer::new(3);
        let mut reference = PreviousReference::default();
        let symbols = ["A", "B"];
        let extra_keys = ["x", "y", "z"];

        for step in 0..4000_i64 {
            let ts = step + 1;
            match next(100) {
                0..=54 => {
                    let symbol = symbols[next(2) as usize];
                    let mut bar = make_bar(ts, ts % 97 + 1, None);
                    bar.symbol = symbol.to_string();
                    for key in extra_keys {
                        // 多数 bar 不带 extra; 偶尔带上, 触发新增列与 NaN 补齐
                        if next(5) == 0 {
                            bar.extra.insert(key.to_string(), (ts % 13) as f64);
                        }
                    }
                    let before = snapshot_before(buffer.get_history(symbol));
                    buffer.update(&bar);
                    if buffer.default_capacity > 0 {
                        reference.bar.insert(symbol.to_string(), before);
                    }
                }
                55..=74 => {
                    let symbol = symbols[next(2) as usize];
                    let mut tick = make_tick(ts, ts % 89 + 1);
                    tick.symbol = symbol.to_string();
                    let before = snapshot_before(buffer.get_tick_history(symbol));
                    buffer.update_tick(&tick);
                    if buffer.default_capacity > 0 {
                        reference.tick.insert(symbol.to_string(), before);
                    }
                }
                75..=89 => {
                    let freq = if next(2) == 0 { "5min" } else { "1h" };
                    let mut bar = make_window_bar(ts, ts % 83 + 1, freq);
                    if next(4) == 0 {
                        bar.extra.insert("x".to_string(), 1.5);
                    }
                    let before = snapshot_before(buffer.get_window_history("TEST", freq));
                    buffer.update_window(&bar);
                    if buffer.default_capacity > 0 {
                        reference
                            .window
                            .insert(("TEST".to_string(), freq.to_string()), before);
                    }
                }
                90..=95 => {
                    let capacity = [0_usize, 1, 2, 3, 5, 8][next(6) as usize];
                    buffer.set_capacity_preserve_existing(capacity);
                    reference.clear();
                }
                96..=97 => {
                    let restored = HistoryBuffer::from(HistoryBufferSnapshot::from(&buffer));
                    buffer = restored;
                    reference.clear();
                }
                _ => {
                    let capacity = [1_usize, 3, 4][next(3) as usize];
                    buffer.set_capacity(capacity);
                    reference = PreviousReference::default();
                }
            }

            for (symbol, expected) in &reference.bar {
                let view = buffer.get_history(symbol).and_then(|h| h.previous_view());
                assert_view_matches(
                    view,
                    expected.as_ref(),
                    &format!("step {step} bar {symbol}"),
                );
            }
            for (symbol, expected) in &reference.tick {
                let view = buffer
                    .get_tick_history(symbol)
                    .and_then(|h| h.previous_view());
                assert_view_matches(
                    view,
                    expected.as_ref(),
                    &format!("step {step} tick {symbol}"),
                );
            }
            for ((symbol, freq), expected) in &reference.window {
                let view = buffer
                    .get_window_history(symbol, freq)
                    .and_then(|h| h.previous_view());
                assert_view_matches(
                    view,
                    expected.as_ref(),
                    &format!("step {step} window {symbol} {freq}"),
                );
            }
            // 当前视图就是序列本身
            for symbol in symbols {
                if let Some(h) = buffer.get_history(symbol) {
                    assert_view_matches(
                        Some(h.view()),
                        Some(h),
                        &format!("step {step} current {symbol}"),
                    );
                }
            }
        }
    }
}
