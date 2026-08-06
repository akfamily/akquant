use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::prelude::*;
use std::collections::HashMap;

use super::feed::DataFeed;

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
/// 用于将 Tick 数据合成为 K 线 (Bar) 数据。
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
    /// 封闭的 bar 用区间**结束**时间戳而非起点。
    ///
    /// live 下用起点是安全的: bar 在下一区间首个 tick 到达时才发出, 墙钟顺序即因果
    /// 顺序。但回测把 tick 与合成 bar 放进同一个 feed 后会 `sort()`, 按起点排序会让
    /// bar 落到形成它的 tick **之前**, 策略于是读到未来的 OHLC。回测须传 true。
    stamp_bar_at_interval_end: bool,
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
    /// :param stamp_bar_at_interval_end: 封闭 bar 是否用区间结束时间戳
    ///     (默认 False, 保持 live 原有的区间起点语义)。回测必须传 True——
    ///     `run_backtest` 会把合成 bar 与源 tick 放进同一个 feed 再 `sort()`,
    ///     起点时间戳会让 bar 排到其源 tick 之前, 策略读到未来的 OHLC。
    #[new]
    #[pyo3(signature = (feed, interval_min=None, volume_is_cumulative=true, stamp_bar_at_interval_end=false))]
    pub fn new(
        feed: DataFeed,
        interval_min: Option<i64>,
        volume_is_cumulative: bool,
        stamp_bar_at_interval_end: bool,
    ) -> Self {
        Self {
            feed,
            active_bars: HashMap::new(),
            last_cumulative_volumes: HashMap::new(),
            interval_min: interval_min.unwrap_or(1),
            volume_is_cumulative,
            stamp_bar_at_interval_end,
        }
    }

    /// 处理新的 Tick 数据
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
            } else {
                // Close current bar and start new one
                let finished_bar = crate::model::Bar {
                    timestamp: if self.stamp_bar_at_interval_end {
                        // 区间结束 = 下一区间起点 - 1ns, 保证 bar 排在本区间所有
                        // tick 之后、下一区间任何事件之前。
                        (bar.timestamp_min + 1) * 60 * 1_000_000_000 * self.interval_min - 1
                    } else {
                        bar.timestamp_min * 60 * 1_000_000_000 * self.interval_min
                    },
                    open: bar.open,
                    high: bar.high,
                    low: bar.low,
                    close: bar.close,
                    volume: bar.volume_curr,
                    symbol: symbol.clone(),
                    extra: HashMap::new(),
                };
                self.feed.add_bar(finished_bar)?;

                // Start new bar
                let new_bar = ActiveBar {
                    open: price,
                    high: price,
                    low: price,
                    close: price,
                    volume_base: volume,
                    volume_curr: volume_delta,
                    timestamp_min: current_key,
                };
                self.active_bars.insert(symbol.clone(), new_bar);
            }
        } else {
            // First bar
            let new_bar = ActiveBar {
                open: price,
                high: price,
                low: price,
                close: price,
                volume_base: volume,
                volume_curr: volume_delta,
                timestamp_min: current_key,
            };
            self.active_bars.insert(symbol.clone(), new_bar);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataFeed;

    fn ns(minute: i64, second: i64) -> i64 {
        (minute * 60 + second) * 1_000_000_000
    }

    #[test]
    fn per_trade_mode_keeps_first_tick_volume() {
        // 回归防护: 累计口径下首次差分恒为 0, 会丢掉每个 symbol 的首笔量。
        // 单笔口径必须一笔不落。
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed, Some(1), false, false);

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
        let mut agg = BarAggregator::new(feed, Some(1), true, false);

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
        let mut agg = BarAggregator::new(feed, Some(1), false, true);

        agg.on_tick("X".to_string(), 10.0, 100.0, ns(1, 0)).unwrap();
        agg.on_tick("X".to_string(), 20.0, 100.0, ns(1, 30)).unwrap();
        // 跨区间, 封闭第 1 分钟的 bar。
        agg.on_tick("X".to_string(), 11.0, 100.0, ns(2, 0)).unwrap();

        let event = readback.next().expect("封闭的 bar 未写入 feed");
        match event {
            crate::event::Event::Bar(bar) => {
                // 第 1 分钟的 bar 应打在 119_999_999_999ns, 即第 2 分钟起点前
                // 1ns, 晚于其最后贡献 tick(90s = 90_000_000_000ns)。
                assert_eq!(bar.timestamp, ns(2, 0) - 1);
                assert!(bar.timestamp >= ns(1, 30));
            }
            other => panic!("expected Event::Bar, got {other:?}"),
        }
    }

    #[test]
    fn interval_start_stamping_is_the_default() {
        // live 语义不得改变: 默认仍用区间起点。
        let feed = DataFeed::new();
        let agg = BarAggregator::new(feed, Some(1), true, false);
        assert!(!agg.stamp_bar_at_interval_end);
    }
}
