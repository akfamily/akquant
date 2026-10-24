use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::prelude::*;
use std::collections::HashMap;

use super::feed::DataFeed;

const NS_PER_MIN: i64 = 60 * 1_000_000_000;

#[derive(Debug, Clone)]
struct ActiveBar {
    open: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    #[allow(dead_code)]
    volume_base: Decimal,
    volume_curr: Decimal,
    timestamp_min: i64,
}

#[gen_stub_pyclass]
#[pyclass]
/// K 线合成器.
///
/// 用于将 Tick 数据合成为 K 线 (Bar) 数据。封闭的 bar 恒用区间**终点**戳:
/// `timestamp = 下一区间起点 - 1ns`, `interval_start = 本区间起点 - 1ns`,
/// 区间为 `(interval_start, timestamp]`。实盘与回测同一口径。
pub struct BarAggregator {
    feed: DataFeed,
    active_bars: HashMap<String, ActiveBar>,
    last_cumulative_volumes: HashMap<String, Decimal>,
    interval_min: i64,
    /// `volume` 入参是否为累计量。
    ///
    /// live(CTP)推来的是当日累计量, 需算差分; 回测调用方构造的 `Tick` 携带的是
    /// **单笔**量, 直接计入即可。两者不可混用: 用累计口径处理单笔量会让每个
    /// symbol 的首笔量被丢弃(首次差分恒为 0)。
    volume_is_cumulative: bool,
    /// 每个品种最近一次封闭推出的区间编号。之后到达、区间编号不大于它的 tick
    /// 属于已推出的 bar, 只吸收累计量后丢弃(否则会重开旧区间、推出时间戳倒退的 bar)。
    last_emitted_key: HashMap<String, i64>,
    /// 每个品种因所属区间已封闭而丢弃的迟到 tick 数。
    late_ticks: HashMap<String, usize>,
}

impl BarAggregator {
    /// Rust 内部构造器(不含迁移参数)。
    pub fn new(feed: DataFeed, interval_min: Option<i64>, volume_is_cumulative: bool) -> Self {
        Self {
            feed,
            active_bars: HashMap::new(),
            last_cumulative_volumes: HashMap::new(),
            interval_min: interval_min.unwrap_or(1),
            volume_is_cumulative,
            last_emitted_key: HashMap::new(),
            late_ticks: HashMap::new(),
        }
    }

    fn interval_ns(&self) -> i64 {
        NS_PER_MIN * self.interval_min
    }

    /// 把在形成的 bar 封闭为终点戳 `Bar`(on_tick 与 flush_due 共用)。
    fn close_bar(&self, symbol: &str, bar: &ActiveBar) -> crate::model::Bar {
        let interval_ns = self.interval_ns();
        crate::model::Bar {
            // 区间终点 = 下一区间起点 - 1ns, 保证 bar 排在本区间所有 tick 之后、
            // 下一区间任何事件之前。
            timestamp: (bar.timestamp_min + 1) * interval_ns - 1,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume_curr,
            symbol: symbol.to_string(),
            extra: HashMap::new(),
            freq: None,
            // 区间为 (起点 - 1ns, 终点]
            interval_start: Some(bar.timestamp_min * interval_ns - 1),
        }
    }

    /// 封闭并推出 bar, 登记该品种最近推出的区间编号。
    fn emit_bar(&mut self, symbol: &str, bar: &ActiveBar) -> PyResult<()> {
        let finished = self.close_bar(symbol, bar);
        self.feed.add_bar(finished)?;
        self.last_emitted_key
            .insert(symbol.to_string(), bar.timestamp_min);
        Ok(())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl BarAggregator {
    /// 创建 K 线合成器.
    ///
    /// :param feed: 数据源 (合成的 Bar 将添加到此数据源)
    /// :param interval_min: K 线周期 (分钟, 默认 1 分钟)
    /// :param volume_is_cumulative: `on_tick` 的 `volume` 是否为累计量
    ///     (默认 True, 对应 CTP 的日累计 Volume)。回测喂单笔量时传 False。
    /// :param stamp_bar_at_interval_end: 已废弃, 仅为兼容旧调用保留。封闭 bar
    ///     恒用区间终点戳; 传 True 静默接受, 传 False 报错。
    #[new]
    #[pyo3(signature = (feed, interval_min=None, volume_is_cumulative=true, stamp_bar_at_interval_end=None))]
    pub fn py_new(
        feed: DataFeed,
        interval_min: Option<i64>,
        volume_is_cumulative: bool,
        stamp_bar_at_interval_end: Option<bool>,
    ) -> PyResult<Self> {
        if stamp_bar_at_interval_end == Some(false) {
            return Err(PyValueError::new_err(
                "实盘与回测统一用区间终点戳，已不支持起点戳；去掉该参数即可",
            ));
        }
        Ok(Self::new(feed, interval_min, volume_is_cumulative))
    }

    /// 处理新的 Tick 数据
    ///
    /// 所属区间已被封闭推出(例如已由 `flush_due` 补发)的迟到 tick 不再计入
    /// 任何 bar: 累计口径下仍更新累计基数, 计入 `late_tick_count`。
    ///
    /// :param symbol: 标的代码
    /// :param price: 最新价
    /// :param volume: 成交量。`volume_is_cumulative=True` 时为累计成交量
    ///     (TotalVolume); 为 False 时为本笔成交量。
    /// :param timestamp_ns: 时间戳 (纳秒)
    pub fn on_tick(
        &mut self,
        symbol: String,
        price: f64,
        volume: f64,
        timestamp_ns: i64,
    ) -> PyResult<()> {
        let price = Decimal::from_f64(price).unwrap_or(Decimal::ZERO);
        let volume = Decimal::from_f64(volume).unwrap_or(Decimal::ZERO);

        // Calculate current interval key
        let current_key = timestamp_ns / 1_000_000_000 / 60 / self.interval_min;

        let volume_delta = if self.volume_is_cumulative {
            // 累计口径(live/CTP): 首次播种为本次值, 之后算差分。
            if !self.last_cumulative_volumes.contains_key(&symbol) {
                self.last_cumulative_volumes.insert(symbol.clone(), volume);
            }
            let last_cum_vol = *self.last_cumulative_volumes.get(&symbol).unwrap();
            let delta = if volume >= last_cum_vol {
                volume - last_cum_vol
            } else {
                // Volume reset (e.g., new day or issue)
                volume
            };
            self.last_cumulative_volumes.insert(symbol.clone(), volume);
            delta
        } else {
            // 单笔口径(回测): volume 就是本笔成交量, 直接计入。
            volume
        };

        // 迟到 tick: 所属区间已推出, 累计量已在上面吸收, 这里只计数后丢弃。
        if self
            .last_emitted_key
            .get(&symbol)
            .is_some_and(|&emitted| current_key <= emitted)
        {
            *self.late_ticks.entry(symbol).or_insert(0) += 1;
            return Ok(());
        }

        if let Some(bar) = self.active_bars.get_mut(&symbol) {
            if bar.timestamp_min == current_key {
                // Update current bar
                if price > bar.high {
                    bar.high = price;
                }
                if price < bar.low {
                    bar.low = price;
                }
                bar.close = price;
                bar.volume_curr += volume_delta;
                return Ok(());
            }
            // Close current bar and start new one
            let finished = bar.clone();
            self.emit_bar(&symbol, &finished)?;
        }
        let new_bar = ActiveBar {
            open: price,
            high: price,
            low: price,
            close: price,
            volume_base: volume,
            volume_curr: volume_delta,
            timestamp_min: current_key,
        };
        self.active_bars.insert(symbol, new_bar);

        Ok(())
    }

    /// 封闭已到期的在形成 bar 并推入数据源.
    ///
    /// 区间终点 + `grace_ns` ≤ `now_ns` 的 bar 被封闭推出(时间戳同 `on_tick`
    /// 封闭的 bar)。用于实盘: 某品种下一区间迟迟没有 tick 时, 本区间的 bar
    /// 也能按时发出。之后到达的同区间 tick 视为迟到并丢弃。
    ///
    /// :param now_ns: 当前时刻 (纳秒)
    /// :param grace_ns: 区间结束后的宽限期 (纳秒, 默认 2 秒)
    /// :return: 本次推出的 bar 根数
    #[pyo3(signature = (now_ns, grace_ns=2_000_000_000))]
    pub fn flush_due(&mut self, now_ns: i64, grace_ns: i64) -> PyResult<usize> {
        let interval_ns = self.interval_ns();
        let mut due: Vec<(String, ActiveBar)> = self
            .active_bars
            .iter()
            .filter(|(_, bar)| (bar.timestamp_min + 1) * interval_ns + grace_ns <= now_ns)
            .map(|(symbol, bar)| (symbol.clone(), bar.clone()))
            .collect();
        // 按 (区间, 品种) 排序: 推出顺序确定, 时间戳不倒退
        due.sort_by(|a, b| (a.1.timestamp_min, &a.0).cmp(&(b.1.timestamp_min, &b.0)));
        for (symbol, bar) in &due {
            self.active_bars.remove(symbol);
            self.emit_bar(symbol, bar)?;
        }
        Ok(due.len())
    }

    /// 某品种因所属区间已封闭而丢弃的迟到 tick 数.
    ///
    /// :param symbol: 标的代码
    /// :return: 丢弃的 tick 数
    pub fn late_tick_count(&self, symbol: &str) -> usize {
        self.late_ticks.get(symbol).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataFeed;
    use rust_decimal_macros::dec;

    fn ns(minute: i64, second: i64) -> i64 {
        (minute * 60 + second) * 1_000_000_000
    }

    #[test]
    fn per_trade_mode_keeps_first_tick_volume() {
        // 回归防护: 累计口径下首次差分恒为 0, 会丢掉每个 symbol 的首笔量。
        // 单笔口径必须一笔不落。
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed, Some(1), false);

        agg.on_tick("X".to_string(), 10.0, 100.0, ns(1, 0)).unwrap();
        agg.on_tick("X".to_string(), 10.6, 200.0, ns(1, 15))
            .unwrap();
        agg.on_tick("X".to_string(), 9.8, 150.0, ns(1, 30)).unwrap();
        agg.on_tick("X".to_string(), 10.2, 50.0, ns(1, 45)).unwrap();

        let active = agg.active_bars.get("X").expect("active bar missing");
        assert_eq!(active.volume_curr, Decimal::from(500));
    }

    #[test]
    fn cumulative_mode_preserves_live_semantics() {
        // live(CTP)口径不得改变: 首笔播种, 之后算差分。
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed, Some(1), true);

        agg.on_tick("X".to_string(), 10.0, 50_000.0, ns(1, 0))
            .unwrap();
        agg.on_tick("X".to_string(), 10.1, 50_120.0, ns(1, 15))
            .unwrap();

        let active = agg.active_bars.get("X").expect("active bar missing");
        // 首笔播种为 50000(计 0), 第二笔差分 120。
        assert_eq!(active.volume_curr, Decimal::from(120));
    }

    #[test]
    fn interval_end_stamping_places_bar_after_its_ticks() {
        // 回归防护: 起点时间戳会让 bar 在回测 sort 后排到其源 tick 之前。
        //
        // 直接从 `feed` 读回封闭 bar 的真实时间戳(而非只看 active_bars 的内部
        // 状态推进), 这样才是真正验证了本次修复的目标属性。DataFeed 派生
        // Clone(共享同一个 Arc<Mutex<_>> provider), 故克隆一份留在测试里读回,
        // 另一份交给 aggregator 写入。
        let feed = DataFeed::new();
        let readback = feed.clone();
        let mut agg = BarAggregator::new(feed, Some(1), false);

        agg.on_tick("X".to_string(), 10.0, 100.0, ns(1, 0)).unwrap();
        agg.on_tick("X".to_string(), 20.0, 100.0, ns(1, 30))
            .unwrap();
        // 跨区间, 封闭第 1 分钟的 bar。
        agg.on_tick("X".to_string(), 11.0, 100.0, ns(2, 0)).unwrap();

        let event = readback.next().expect("封闭的 bar 未写入 feed");
        match event {
            crate::event::Event::Bar(bar) => {
                // 第 1 分钟的 bar 应打在 119_999_999_999ns, 即第 2 分钟起点前
                // 1ns, 晚于其最后贡献 tick(90s = 90_000_000_000ns)。
                assert_eq!(bar.timestamp, ns(2, 0) - 1);
                assert!(bar.timestamp >= ns(1, 30));
                // 区间起点 - 1ns: (起点 - 1, 终点 - 1] 与终点戳同口径
                assert_eq!(bar.interval_start, Some(ns(1, 0) - 1));
            }
            other => panic!("expected Event::Bar, got {other:?}"),
        }
    }

    #[test]
    fn bars_are_always_end_stamped_with_interval_start() {
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        agg.on_tick("X".into(), 10.0, 1.0, ns(1, 0)).unwrap();
        agg.on_tick("X".into(), 11.0, 1.0, ns(2, 0)).unwrap();
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].timestamp, ns(2, 0) - 1);
        assert_eq!(bars[0].interval_start, Some(ns(1, 0) - 1));
    }

    #[test]
    fn flush_due_emits_after_grace_and_drops_late_ticks() {
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), true);
        agg.on_tick("X".into(), 10.0, 100.0, ns(1, 0)).unwrap();
        agg.on_tick("X".into(), 10.5, 130.0, ns(1, 30)).unwrap();
        assert_eq!(agg.flush_due(ns(2, 1), 2_000_000_000).unwrap(), 0); // 未过宽限
        assert_eq!(agg.flush_due(ns(2, 2), 2_000_000_000).unwrap(), 1);
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        assert_eq!(
            (bars[0].timestamp, bars[0].close),
            (ns(2, 0) - 1, dec!(10.5))
        );
        // 累计口径: 首笔 100 播种计 0, 130 差分计 30
        assert_eq!(bars[0].volume, dec!(30));
        // 封闭后同一区间迟到的 tick: 丢弃, 但累计量被吸收
        agg.on_tick("X".into(), 10.7, 150.0, ns(1, 59)).unwrap();
        assert_eq!(agg.late_tick_count("X"), 1);
        assert_eq!(agg.late_tick_count("Y"), 0);
        assert!(feed.drain_bars_for_test().is_empty());
        agg.on_tick("X".into(), 10.8, 160.0, ns(2, 10)).unwrap();
        agg.on_tick("X".into(), 10.9, 170.0, ns(3, 0)).unwrap();
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].timestamp, ns(3, 0) - 1);
        assert_eq!((bars[0].open, bars[0].close), (dec!(10.8), dec!(10.8)));
        // 第 2 分钟只有 2:10 那一笔: 160-150=10。迟到 tick 的 20 已被吸收进
        // 累计基数, 不再计入任何一根; 3:00 那笔(170-160)属于第 3 分钟。
        assert_eq!(bars[0].volume, dec!(10));
    }

    #[test]
    fn flush_due_skips_bars_still_forming() {
        // 多品种: 只封闭已到期的区间, 仍在形成的不动
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        agg.on_tick("X".into(), 10.0, 1.0, ns(1, 0)).unwrap();
        agg.on_tick("Y".into(), 20.0, 1.0, ns(2, 0)).unwrap();
        assert_eq!(agg.flush_due(ns(2, 2), 2_000_000_000).unwrap(), 1);
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].symbol, "X");
        assert!(agg.active_bars.contains_key("Y"));
        assert!(!agg.active_bars.contains_key("X"));
    }
}
