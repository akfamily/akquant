use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use super::feed::DataFeed;
use crate::calendar::python::PySessionResolver;
use crate::calendar::resolver::SessionResolver;

const NS_PER_MIN: i64 = 60 * 1_000_000_000;

/// 收盘宽限钳制的收盘 bar 在 `extra` 里携带的“不晚于收盘的最后一笔价格”。
///
/// 钳制 bar 的时间戳是收盘时刻, close 却含收盘后宽限期内的成交价; 引擎据此把当日
/// 结算价设为该值(而非 close), 见 `pipeline/stages/data.rs`。
pub(crate) const BAR_CLOSE_PRICE_KEY: &str = "__akq_close_price";

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
    /// 补发基准: `flush_due` 在它 + 补发宽限之后推出。通常为区间终点; 开启收盘钳制时
    /// 收盘桶为 max(区间终点, 收盘 + 钳制宽限)。
    emit_after: i64,
    /// 开启收盘钳制时, 收盘桶对应的交易日收盘时刻。
    session_close: Option<i64>,
    /// 是否并入了收盘后宽限期内的 tick(钳制 bar)。
    has_after_close: bool,
}

/// 品种所属交易日的时刻表缓存, 见 [`SessionResolver::symbol_day_span`]。
#[derive(Debug, Clone, Copy)]
struct ClampSpan {
    prev_close: Option<i64>,
    first_start: i64,
    close: i64,
    /// 下一交易日首段起点(超出日历时为 None, 视为无界)。
    next_start: Option<i64>,
}

/// 收盘桶的钳制信息。
#[derive(Debug, Clone, Copy)]
struct ClampHit {
    /// 交易日收盘时刻。
    close: i64,
    /// 钳制窗口终点 min(收盘 + 钳制宽限, 下一交易日首段起点), 收盘桶按它推迟补发。
    clamp_end: i64,
    /// 本笔是否为收盘后的宽限 tick。
    after_close: bool,
}

/// 钳制窗口终点: min(收盘 + 宽限, 下一交易日首段起点)。
fn clamp_end_of(close: i64, grace_ns: i64, next_start: Option<i64>) -> i64 {
    let end = close.saturating_add(grace_ns);
    next_start.map_or(end, |n| end.min(n))
}

impl ClampSpan {
    /// `ts` 是否仍属于本交易日(缓存可用)。
    fn contains(&self, ts: i64) -> bool {
        ts <= self.close
            && match self.prev_close {
                Some(p) => ts > p,
                None => ts >= self.first_start,
            }
    }
}

/// 收盘宽限钳制配置与状态。
struct SessionClamp {
    resolver: Arc<SessionResolver>,
    grace_ns: i64,
    spans: HashMap<String, ClampSpan>,
    /// 每个品种不晚于其所属交易日收盘的最后一笔 `(时间戳, 价格)`。
    last_in_session: HashMap<String, (i64, Decimal)>,
}

#[gen_stub_pyclass]
#[pyclass]
/// K 线合成器.
///
/// 用于将 Tick 数据合成为 K 线 (Bar) 数据。按右闭区间 `(S, E]` 分桶(恰在整分
/// `E` 的 tick 属于以 `E` 结尾的 bar, 与 bar 时间戳口径、qmf_market 服务端 M1
/// 一致), 封闭的 bar 恒用区间**终点**戳: `timestamp = E`, `interval_start = S`。
/// 实盘与回测同一口径。
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
    /// 属于已推出的 bar: 不参与 OHLC(否则会重开旧区间、推出时间戳倒退的 bar),
    /// 其量计入下一根 bar(日累计量守恒)。
    last_emitted_key: HashMap<String, i64>,
    /// 每个品种的迟到 tick 数(所属区间已封闭推出)。
    late_ticks: HashMap<String, usize>,
    /// 迟到 tick 的量在该品种尚无在形成 bar 时暂存于此, 新建 bar 时并入。
    pending_late_volume: HashMap<String, Decimal>,
    /// 收盘宽限钳制(默认关闭, 只由实盘 runner 经网关钩子开启)。
    session_clamp: Option<SessionClamp>,
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
            pending_late_volume: HashMap::new(),
            session_clamp: None,
        }
    }

    /// 开启收盘宽限钳制(Rust 侧入口, Python 见 `set_session_clamp`)。
    ///
    /// tick 落在其品种交易日 D 的 `(收盘(D), 收盘(D) + grace_ns]` 且早于 D 的下一
    /// 交易日首段起点时, 归入以收盘(D) 结尾的那个桶; 该桶推迟到
    /// `收盘(D) + grace_ns` 之后才由 `flush_due` 补发。收盘按品种自己的时段表取。
    pub fn set_session_clamp_arc(
        &mut self,
        resolver: Arc<SessionResolver>,
        grace_ns: i64,
    ) -> Result<(), String> {
        if grace_ns < 0 {
            return Err(format!(
                "收盘钳制宽限 grace_ns 不能为负（收到 {grace_ns}），请传 0 或正的纳秒数"
            ));
        }
        self.session_clamp = Some(SessionClamp {
            resolver,
            grace_ns,
            spans: HashMap::new(),
            last_in_session: HashMap::new(),
        });
        Ok(())
    }

    /// 关闭收盘宽限钳制; 在形成的 bar 恢复按区间终点补发。
    pub fn clear_session_clamp(&mut self) {
        self.session_clamp = None;
        let interval_ns = self.interval_ns();
        for bar in self.active_bars.values_mut() {
            bar.emit_after = (bar.timestamp_min + 1) * interval_ns;
            bar.session_close = None;
            bar.has_after_close = false;
        }
    }

    /// 开启钳制时对 tick 分类: 落在收盘桶(或被钳进收盘桶)时返回钳制信息, 否则
    /// `None`(收盘后紧接下一交易日、没有闭市间隙时如 ALWAYS_OPEN, 也是 `None`)。
    /// 顺带登记品种不晚于收盘的最后价格。
    fn classify_for_clamp(&mut self, symbol: &str, ts: i64, price: Decimal) -> Option<ClampHit> {
        let interval_ns = self.interval_ns();
        let clamp = self.session_clamp.as_mut()?;
        let span = match clamp.spans.get(symbol) {
            Some(span) if span.contains(ts) => *span,
            _ => {
                let asset_type = clamp.resolver.registered_asset_type(symbol);
                // 解析失败(超出交易日历等)不钳, 按普通分桶
                let (prev_close, first_start, close, next_start) = clamp
                    .resolver
                    .symbol_day_span(symbol, asset_type, ts)
                    .ok()?;
                let span = ClampSpan {
                    prev_close,
                    first_start,
                    close,
                    next_start,
                };
                clamp.spans.insert(symbol.to_string(), span);
                span
            }
        };
        // 前一交易日收盘后的闭市间隙里、宽限之内: 钳到前一交易日的收盘桶
        if let Some(prev_close) = span.prev_close
            && ts < span.first_start
            && ts - prev_close <= clamp.grace_ns
        {
            return Some(ClampHit {
                close: prev_close,
                clamp_end: clamp_end_of(prev_close, clamp.grace_ns, Some(span.first_start)),
                after_close: true,
            });
        }
        match clamp.last_in_session.get_mut(symbol) {
            Some(slot) if slot.0 > ts => {}
            Some(slot) => *slot = (ts, price),
            None => {
                clamp
                    .last_in_session
                    .insert(symbol.to_string(), (ts, price));
            }
        }
        let key = |t: i64| (t - 1).div_euclid(interval_ns);
        let clamp_end = clamp_end_of(span.close, clamp.grace_ns, span.next_start);
        (key(ts) == key(span.close) && clamp_end > span.close).then_some(ClampHit {
            close: span.close,
            clamp_end,
            after_close: false,
        })
    }

    fn interval_ns(&self) -> i64 {
        NS_PER_MIN * self.interval_min
    }

    /// 右闭区间 `(key*I, (key+1)*I]` 的编号: `floor((ts - 1) / I)`。
    fn interval_key(&self, timestamp_ns: i64) -> i64 {
        (timestamp_ns - 1).div_euclid(self.interval_ns())
    }

    /// 把在形成的 bar 封闭为终点戳 `Bar`(on_tick 与 flush_due 共用)。
    fn close_bar(&self, symbol: &str, bar: &ActiveBar) -> crate::model::Bar {
        let interval_ns = self.interval_ns();
        crate::model::Bar {
            // 区间终点 E: 不早于本区间所有 tick(含恰在 E 的那笔), 早于下一区间
            // 任何 tick(都 > E)。同刻 E 的 tick 与 bar: 回测靠 feed 稳定排序
            // (tick 先入队), 实盘 bar 只在下一区间 tick 到达或补发时才推出。
            timestamp: (bar.timestamp_min + 1) * interval_ns,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume_curr,
            symbol: symbol.to_string(),
            extra: self.clamped_bar_extra(symbol, bar),
            freq: None,
            // 区间为 (起点, 终点]
            interval_start: Some(bar.timestamp_min * interval_ns),
        }
    }

    /// 钳制 bar 的 `extra`: 写入不晚于收盘的最后一笔价格(没有时不写)。
    fn clamped_bar_extra(&self, symbol: &str, bar: &ActiveBar) -> HashMap<String, f64> {
        let mut extra = HashMap::new();
        if bar.has_after_close
            && let (Some(close), Some(clamp)) = (bar.session_close, self.session_clamp.as_ref())
            && let Some(&(ts, price)) = clamp.last_in_session.get(symbol)
            && ts <= close
            && let Some(price) = price.to_f64()
        {
            extra.insert(BAR_CLOSE_PRICE_KEY.to_string(), price);
        }
        extra
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
    /// 所属区间已被封闭推出(例如已由 `flush_due` 补发)的迟到 tick 不参与
    /// OHLC, 其量计入下一根 bar(日累计量守恒), 并计入 `late_tick_count`。
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

        // 收盘宽限钳制(开启时): 收盘后宽限期内的 tick 归入以收盘结尾的桶
        let clamp_info = self.classify_for_clamp(&symbol, timestamp_ns, price);
        let after_close = clamp_info.is_some_and(|h| h.after_close);
        // 右闭区间编号: 恰在整分的 tick 属于以它结尾的那根 bar
        let current_key = match clamp_info {
            Some(hit) if hit.after_close => self.interval_key(hit.close),
            _ => self.interval_key(timestamp_ns),
        };

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

        // 迟到 tick: 所属区间已推出。OHLC 不动, 量计入下一根 bar——已有在形成
        // bar(区间必晚于已推出区间)就直接加上, 否则暂存到新建 bar 时并入。
        if self
            .last_emitted_key
            .get(&symbol)
            .is_some_and(|&emitted| current_key <= emitted)
        {
            if let Some(bar) = self.active_bars.get_mut(&symbol) {
                bar.volume_curr += volume_delta;
            } else {
                *self
                    .pending_late_volume
                    .entry(symbol.clone())
                    .or_insert(Decimal::ZERO) += volume_delta;
            }
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
                bar.has_after_close |= after_close;
                return Ok(());
            }
            // Close current bar and start new one
            let finished = bar.clone();
            self.emit_bar(&symbol, &finished)?;
        }
        let interval_end = (current_key + 1) * self.interval_ns();
        let new_bar = ActiveBar {
            open: price,
            high: price,
            low: price,
            close: price,
            volume_base: volume,
            volume_curr: volume_delta
                + self
                    .pending_late_volume
                    .remove(&symbol)
                    .unwrap_or(Decimal::ZERO),
            timestamp_min: current_key,
            emit_after: clamp_info.map_or(interval_end, |hit| interval_end.max(hit.clamp_end)),
            session_close: clamp_info.map(|hit| hit.close),
            has_after_close: after_close,
        };
        self.active_bars.insert(symbol, new_bar);

        Ok(())
    }

    /// 封闭已到期的在形成 bar 并推入数据源.
    ///
    /// 区间终点 `E` + `grace_ns` ≤ `now_ns` 的 bar 被封闭推出(时间戳同 `on_tick`
    /// 封闭的 bar, 即 `E`)。用于实盘: 某品种下一区间迟迟没有 tick 时, 本区间的
    /// bar 也能按时发出。之后到达的同区间 tick(含恰在 `E` 的)视为迟到: 不参与
    /// OHLC, 量计入下一根。
    ///
    /// :param now_ns: 当前时刻 (纳秒)
    /// :param grace_ns: 区间结束后的宽限期 (纳秒, 默认 2 秒)
    /// :return: 本次推出的 bar 根数
    #[pyo3(signature = (now_ns, grace_ns=2_000_000_000))]
    pub fn flush_due(&mut self, now_ns: i64, grace_ns: i64) -> PyResult<usize> {
        let mut due: Vec<(String, ActiveBar)> = self
            .active_bars
            .iter()
            .filter(|(_, bar)| bar.emit_after + grace_ns <= now_ns)
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

    /// 开启收盘宽限钳制(实盘).
    ///
    /// tick 时间戳落在其品种所属交易日 D 的 `(收盘(D), 收盘(D) + grace_ns]` 内、且
    /// 早于下一交易日首段起点时, 计入以收盘(D) 结尾的那根 bar(OHLC 与量都含它);
    /// 该 bar 推迟到 `收盘(D) + grace_ns` 再加 `flush_due` 的补发宽限之后才推出,
    /// 并在 `extra["__akq_close_price"]` 写入不晚于收盘的最后一笔价格(引擎按它
    /// 记当日结算价)。收盘按品种自己的时段表取; 午休不钳, ALWAYS_OPEN 不钳。
    /// 默认关闭, 由实盘 runner 经行情网关的 `configure_session_clamp` 钩子开启。
    ///
    /// :param resolver: 与引擎同配置的时段解析器(须已 `set_active` 全部合约)
    /// :param grace_ns: 收盘后宽限 (纳秒, 不能为负)
    #[pyo3(name = "set_session_clamp")]
    pub fn py_set_session_clamp(
        &mut self,
        resolver: PyRef<'_, PySessionResolver>,
        grace_ns: i64,
    ) -> PyResult<()> {
        self.set_session_clamp_arc(resolver.arc(), grace_ns)
            .map_err(PyValueError::new_err)
    }

    /// 关闭收盘宽限钳制.
    #[pyo3(name = "clear_session_clamp")]
    pub fn py_clear_session_clamp(&mut self) {
        self.clear_session_clamp();
    }

    /// 某品种的迟到 tick 数(所属区间已封闭推出, 量已计入下一根).
    ///
    /// :param symbol: 标的代码
    /// :return: 迟到 tick 数
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

        agg.on_tick("X".to_string(), 10.0, 100.0, ns(1, 1)).unwrap();
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

        agg.on_tick("X".to_string(), 10.0, 50_000.0, ns(1, 1))
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

        agg.on_tick("X".to_string(), 10.0, 100.0, ns(1, 1)).unwrap();
        agg.on_tick("X".to_string(), 20.0, 100.0, ns(1, 30))
            .unwrap();
        // 跨区间, 封闭第 1 分钟的 bar。
        agg.on_tick("X".to_string(), 11.0, 100.0, ns(2, 1)).unwrap();

        let event = readback.next().expect("封闭的 bar 未写入 feed");
        match event {
            crate::event::Event::Bar(bar) => {
                // (1:00, 2:00] 的 bar 应打在区间终点 120_000_000_000ns, 晚于其
                // 最后贡献 tick(90s = 90_000_000_000ns)。
                assert_eq!(bar.timestamp, ns(2, 0));
                assert!(bar.timestamp >= ns(1, 30));
                // 右闭区间 (起点, 终点]
                assert_eq!(bar.interval_start, Some(ns(1, 0)));
            }
            other => panic!("expected Event::Bar, got {other:?}"),
        }
    }

    #[test]
    fn bars_are_always_end_stamped_with_interval_start() {
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        agg.on_tick("X".into(), 10.0, 1.0, ns(1, 1)).unwrap();
        agg.on_tick("X".into(), 11.0, 1.0, ns(2, 1)).unwrap();
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].timestamp, ns(2, 0));
        assert_eq!(bars[0].interval_start, Some(ns(1, 0)));
    }

    #[test]
    fn flush_due_emits_after_grace_and_carries_late_volume() {
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), true);
        agg.on_tick("X".into(), 10.0, 100.0, ns(1, 1)).unwrap();
        agg.on_tick("X".into(), 10.5, 130.0, ns(1, 30)).unwrap();
        assert_eq!(agg.flush_due(ns(2, 1), 2_000_000_000).unwrap(), 0); // 未过宽限
        assert_eq!(agg.flush_due(ns(2, 2), 2_000_000_000).unwrap(), 1);
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        assert_eq!((bars[0].timestamp, bars[0].close), (ns(2, 0), dec!(10.5)));
        // 累计口径: 首笔 100 播种计 0, 130 差分计 30
        assert_eq!(bars[0].volume, dec!(30));
        // 封闭后同一区间迟到的 tick: 不参与 OHLC, 量(150-130=20)暂存待并入下一根
        agg.on_tick("X".into(), 10.7, 150.0, ns(1, 59)).unwrap();
        assert_eq!(agg.late_tick_count("X"), 1);
        assert_eq!(agg.late_tick_count("Y"), 0);
        assert!(feed.drain_bars_for_test().is_empty());
        agg.on_tick("X".into(), 10.8, 160.0, ns(2, 10)).unwrap();
        agg.on_tick("X".into(), 10.9, 170.0, ns(3, 1)).unwrap();
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].timestamp, ns(3, 0));
        assert_eq!((bars[0].open, bars[0].close), (dec!(10.8), dec!(10.8)));
        // 第 2 分钟: 迟到结转 20 + 2:10 那笔 160-150=10 = 30; OHLC 只来自 2:10,
        // 3:01 那笔(170-160)属于第 3 分钟。
        assert_eq!(bars[0].volume, dec!(30));
        // 守恒: 两根 bar 合计 60 = 第 3 分钟前最后累计量 160 - 首笔播种 100
    }

    #[test]
    fn late_tick_volume_goes_into_forming_bar() {
        // 迟到时该品种已有在形成 bar: 量直接加到它上面, OHLC 不动
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), true);
        agg.on_tick("X".into(), 10.0, 100.0, ns(1, 1)).unwrap();
        agg.on_tick("X".into(), 10.5, 130.0, ns(1, 30)).unwrap();
        assert_eq!(agg.flush_due(ns(2, 2), 2_000_000_000).unwrap(), 1);
        agg.on_tick("X".into(), 10.8, 160.0, ns(2, 5)).unwrap(); // 第 2 分钟 30
        agg.on_tick("X".into(), 99.0, 175.0, ns(1, 59)).unwrap(); // 迟到 15
        assert_eq!(agg.late_tick_count("X"), 1);
        agg.on_tick("X".into(), 10.9, 180.0, ns(3, 1)).unwrap(); // 第 3 分钟 5
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 2);
        let bar2 = &bars[1];
        assert_eq!(bar2.timestamp, ns(3, 0));
        assert_eq!(bar2.volume, dec!(45));
        assert_eq!(
            (bar2.open, bar2.high, bar2.low, bar2.close),
            (dec!(10.8), dec!(10.8), dec!(10.8), dec!(10.8))
        );
    }

    #[test]
    fn flush_due_skips_bars_still_forming() {
        // 多品种: 只封闭已到期的区间, 仍在形成的不动
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        agg.on_tick("X".into(), 10.0, 1.0, ns(1, 1)).unwrap();
        agg.on_tick("Y".into(), 20.0, 1.0, ns(2, 1)).unwrap();
        assert_eq!(agg.flush_due(ns(2, 2), 2_000_000_000).unwrap(), 1);
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].symbol, "X");
        assert!(agg.active_bars.contains_key("Y"));
        assert!(!agg.active_bars.contains_key("X"));
    }

    #[test]
    fn tick_exactly_on_boundary_belongs_to_bar_ending_there() {
        // 右闭区间 (S, E]: 恰在整分 E 的 tick 归入以 E 结尾的 bar
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        agg.on_tick("X".into(), 10.0, 1.0, ns(1, 30)).unwrap();
        agg.on_tick("X".into(), 12.0, 2.0, ns(2, 0)).unwrap();
        agg.on_tick("X".into(), 9.0, 4.0, ns(2, 30)).unwrap();
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        let bar = &bars[0];
        assert_eq!(bar.timestamp, ns(2, 0));
        assert_eq!(bar.interval_start, Some(ns(1, 0)));
        assert_eq!(
            (bar.open, bar.high, bar.low, bar.close, bar.volume),
            (dec!(10), dec!(12), dec!(10), dec!(12), dec!(3))
        );
        // bar 不早于本区间最后一笔 tick(恰在 E), 早于下一区间的 tick
        assert!(bar.timestamp >= ns(2, 0) && bar.timestamp < ns(2, 30));
    }

    #[test]
    fn close_tick_at_1500_goes_into_1459_1500_bar() {
        // 15:00:00 那笔收盘价与成交量属于 (14:59, 15:00] 的 bar, 不得落进
        // (15:00, 15:01] 而被归到下一交易日
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), true);
        let m1459 = 14 * 60 + 59;
        let m1500 = 15 * 60;
        agg.on_tick("X".into(), 100.0, 1_000.0, ns(m1459, 1))
            .unwrap();
        agg.on_tick("X".into(), 101.0, 1_010.0, ns(m1459, 30))
            .unwrap();
        agg.on_tick("X".into(), 102.0, 1_050.0, ns(m1500, 0))
            .unwrap();
        // 夜盘/次日首笔 tick 封闭 15:00 那根
        agg.on_tick("X".into(), 103.0, 5.0, ns(21 * 60, 1)).unwrap();
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        let bar = &bars[0];
        assert_eq!(bar.timestamp, ns(m1500, 0));
        assert_eq!(bar.interval_start, Some(ns(m1459, 0)));
        assert_eq!(bar.close, dec!(102));
        // 10 + 40: 15:00:00 那笔的 40 计入本根
        assert_eq!(bar.volume, dec!(50));
    }

    #[test]
    fn zero_and_negative_timestamps_use_floor_buckets() {
        // 右闭区间按 floor 取编号: ts=0 属于 (-I, 0], ts=1 属于 (0, I]; 截断除法
        // 会把 ts=0 与 ts=1 并进同一桶
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        agg.on_tick("X".into(), 1.0, 1.0, -ns(0, 30)).unwrap();
        agg.on_tick("X".into(), 2.0, 1.0, 0).unwrap();
        agg.on_tick("X".into(), 3.0, 1.0, 1).unwrap();
        agg.on_tick("X".into(), 4.0, 1.0, ns(1, 1)).unwrap();
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 2);
        assert_eq!(
            (bars[0].timestamp, bars[0].interval_start, bars[0].close),
            (0, Some(-ns(1, 0)), dec!(2))
        );
        assert_eq!(
            (bars[1].timestamp, bars[1].interval_start, bars[1].close),
            (ns(1, 0), Some(0), dec!(3))
        );
    }

    #[test]
    fn flush_due_counts_tick_on_boundary_as_part_of_flushed_bar() {
        // 补发后再来恰在 E 的 tick: 属于已推出的 (S, E], 记为迟到
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        agg.on_tick("X".into(), 10.0, 1.0, ns(1, 30)).unwrap();
        assert_eq!(agg.flush_due(ns(2, 1), 2_000_000_000).unwrap(), 0);
        assert_eq!(agg.flush_due(ns(2, 2), 2_000_000_000).unwrap(), 1);
        agg.on_tick("X".into(), 11.0, 1.0, ns(2, 0)).unwrap();
        assert_eq!(agg.late_tick_count("X"), 1);
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].timestamp, ns(2, 0));
    }

    // ---- 收盘宽限钳制 ----

    use crate::calendar::clock::LocalClock;
    use crate::calendar::resolver::SessionResolver;
    use crate::calendar::schedule::template;
    use crate::model::AssetType;
    use std::sync::Arc;

    const SEC: i64 = 1_000_000_000;

    /// 上海本地 2024-01-02（周二）h:m:s → UTC 纳秒。
    fn sh(h: u32, m: u32, s: u32) -> i64 {
        sh_on(2, h, m, s)
    }

    fn sh_on(day: u32, h: u32, m: u32, s: u32) -> i64 {
        LocalClock::new(Some("Asia/Shanghai"), 8 * 3600).from_local(
            chrono::NaiveDate::from_ymd_opt(2024, 1, day)
                .unwrap()
                .and_hms_opt(h, m, s)
                .unwrap(),
        )
    }

    fn clamp_resolver(
        sessions: Vec<(&str, &str)>,
        active: Vec<(&str, Option<AssetType>)>,
    ) -> Arc<SessionResolver> {
        let sessions = sessions
            .into_iter()
            .map(|(k, t)| (k.to_string(), template(t).unwrap()))
            .collect();
        let mut r = SessionResolver::new(
            LocalClock::new(Some("Asia/Shanghai"), 8 * 3600),
            None,
            sessions,
        )
        .unwrap();
        let active: Vec<(String, Option<AssetType>)> = active
            .into_iter()
            .map(|(s, t)| (s.to_string(), t))
            .collect();
        r.set_active(&active);
        Arc::new(r)
    }

    fn stock_resolver() -> Arc<SessionResolver> {
        clamp_resolver(vec![], vec![("600000", Some(AssetType::Stock))])
    }

    const GRACE: i64 = 57 * SEC; // day_close_delay=60 → 钳制宽限 57s
    const FLUSH: i64 = 2 * SEC;

    #[test]
    fn session_clamp_merges_grace_ticks_into_close_bar_and_defers_flush() {
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        agg.set_session_clamp_arc(stock_resolver(), GRACE).unwrap();
        let s = "600000".to_string();
        agg.on_tick(s.clone(), 10.0, 1.0, sh(14, 59, 30)).unwrap();
        agg.on_tick(s.clone(), 10.5, 2.0, sh(15, 0, 0)).unwrap();
        agg.on_tick(s.clone(), 11.0, 4.0, sh(15, 0, 20)).unwrap();
        agg.on_tick(s.clone(), 10.8, 8.0, sh(15, 0, 50)).unwrap();
        // 宽限 tick 没有封闭收盘 bar
        assert!(feed.drain_bars_for_test().is_empty());
        // 收盘 bar 推迟到 收盘 + 钳制宽限 + flush 宽限 = 15:00:59
        assert_eq!(agg.flush_due(sh(15, 0, 59) - 1, FLUSH).unwrap(), 0);
        assert_eq!(agg.flush_due(sh(15, 0, 59), FLUSH).unwrap(), 1);
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        let bar = &bars[0];
        assert_eq!(bar.timestamp, sh(15, 0, 0));
        assert_eq!(bar.interval_start, Some(sh(14, 59, 0)));
        assert_eq!(
            (bar.open, bar.high, bar.low, bar.close, bar.volume),
            (dec!(10), dec!(11), dec!(10), dec!(10.8), dec!(15))
        );
        // 不晚于收盘的最后一笔价格
        assert_eq!(bar.extra.get(BAR_CLOSE_PRICE_KEY), Some(&10.5));
        // 钳制宽限之后的 tick 按普通分桶进 (15:00, 15:01]
        agg.on_tick(s.clone(), 12.0, 1.0, sh(15, 0, 58)).unwrap();
        assert_eq!(agg.late_tick_count("600000"), 0);
        agg.on_tick(s.clone(), 12.5, 1.0, sh(15, 1, 10)).unwrap();
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].timestamp, sh(15, 1, 0));
        assert_eq!(bars[0].close, dec!(12));
        assert!(bars[0].extra.is_empty());
        // 15:01:10 那笔进 (15:01, 15:02], 不钳
        assert_eq!(agg.flush_due(sh(15, 2, 2), FLUSH).unwrap(), 1);
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars[0].timestamp, sh(15, 2, 0));
    }

    #[test]
    fn session_clamp_grace_only_close_bucket_marks_last_pre_close_price() {
        // 收盘桶内只有宽限 tick: 标记取不晚于收盘的最后一笔(在更早的桶里)
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        agg.set_session_clamp_arc(stock_resolver(), GRACE).unwrap();
        let s = "600000".to_string();
        agg.on_tick(s.clone(), 10.0, 1.0, sh(14, 58, 30)).unwrap();
        agg.on_tick(s.clone(), 11.0, 2.0, sh(15, 0, 20)).unwrap();
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].timestamp, sh(14, 59, 0));
        assert!(bars[0].extra.is_empty());
        assert_eq!(agg.flush_due(sh(15, 0, 58), FLUSH).unwrap(), 0);
        assert_eq!(agg.flush_due(sh(15, 0, 59), FLUSH).unwrap(), 1);
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars[0].timestamp, sh(15, 0, 0));
        assert_eq!(bars[0].close, dec!(11));
        assert_eq!(bars[0].extra.get(BAR_CLOSE_PRICE_KEY), Some(&10.0));
    }

    #[test]
    fn session_clamp_without_any_pre_close_tick_writes_no_marker() {
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        agg.set_session_clamp_arc(stock_resolver(), GRACE).unwrap();
        agg.on_tick("600000".into(), 11.0, 2.0, sh(15, 0, 20))
            .unwrap();
        assert_eq!(agg.flush_due(sh(15, 0, 59), FLUSH).unwrap(), 1);
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars[0].timestamp, sh(15, 0, 0));
        assert!(bars[0].extra.is_empty());
    }

    #[test]
    fn without_session_clamp_grace_tick_starts_next_bucket() {
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        let s = "600000".to_string();
        agg.on_tick(s.clone(), 10.0, 1.0, sh(14, 59, 30)).unwrap();
        agg.on_tick(s.clone(), 10.5, 2.0, sh(15, 0, 0)).unwrap();
        agg.on_tick(s.clone(), 11.0, 4.0, sh(15, 0, 20)).unwrap();
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        assert_eq!(
            (bars[0].timestamp, bars[0].close),
            (sh(15, 0, 0), dec!(10.5))
        );
        assert_eq!(bars[0].volume, dec!(3));
        assert!(bars[0].extra.is_empty());
        assert_eq!(agg.flush_due(sh(15, 1, 2), FLUSH).unwrap(), 1);
        let bars = feed.drain_bars_for_test();
        assert_eq!((bars[0].timestamp, bars[0].close), (sh(15, 1, 0), dec!(11)));
    }

    #[test]
    fn clear_session_clamp_restores_plain_bucketing() {
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        agg.set_session_clamp_arc(stock_resolver(), GRACE).unwrap();
        agg.clear_session_clamp();
        let s = "600000".to_string();
        agg.on_tick(s.clone(), 10.0, 1.0, sh(14, 59, 30)).unwrap();
        agg.on_tick(s.clone(), 11.0, 4.0, sh(15, 0, 20)).unwrap();
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].close, dec!(10));
    }

    #[test]
    fn session_clamp_ignored_for_always_open() {
        // ALWAYS_OPEN: 收盘即下一交易日 0 点开盘, 收盘后没有闭市间隙
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        agg.set_session_clamp_arc(clamp_resolver(vec![], vec![("BTC", None)]), GRACE)
            .unwrap();
        let s = "BTC".to_string();
        agg.on_tick(s.clone(), 10.0, 1.0, sh_on(2, 23, 59, 30))
            .unwrap();
        agg.on_tick(s.clone(), 11.0, 1.0, sh_on(3, 0, 0, 20))
            .unwrap();
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        assert_eq!(
            (bars[0].timestamp, bars[0].close),
            (sh_on(3, 0, 0, 0), dec!(10))
        );
        assert_eq!(agg.flush_due(sh_on(3, 0, 1, 2), FLUSH).unwrap(), 1);
        // 以交易日收盘(0 点)结尾的那根也不推迟: 收盘后没有闭市间隙
        agg.on_tick(s.clone(), 12.0, 1.0, sh_on(3, 23, 59, 30))
            .unwrap();
        assert_eq!(agg.flush_due(sh_on(4, 0, 0, 2), FLUSH).unwrap(), 1);
    }

    #[test]
    fn session_clamp_ignores_lunch_break() {
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        agg.set_session_clamp_arc(stock_resolver(), GRACE).unwrap();
        let s = "600000".to_string();
        agg.on_tick(s.clone(), 10.0, 1.0, sh(11, 29, 30)).unwrap();
        // 11:30 不是交易日收盘: 按普通到期补发, 不推迟
        assert_eq!(agg.flush_due(sh(11, 30, 2), FLUSH).unwrap(), 1);
        agg.on_tick(s.clone(), 11.0, 1.0, sh(11, 30, 20)).unwrap();
        assert_eq!(agg.late_tick_count("600000"), 0);
        assert_eq!(agg.flush_due(sh(11, 31, 2), FLUSH).unwrap(), 1);
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].timestamp, sh(11, 30, 0));
        assert_eq!(
            (bars[1].timestamp, bars[1].close),
            (sh(11, 31, 0), dec!(11))
        );
    }

    #[test]
    fn session_clamp_uses_each_symbols_own_close() {
        // 股票 15:00 收盘、国债期货 15:15 收盘: 股票按自己的收盘钳制, 不按会话级 15:15
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        let r = clamp_resolver(
            vec![("T", "CN_FUTURES_CFFEX_BOND_DAY")],
            vec![
                ("600000", Some(AssetType::Stock)),
                ("T2403", Some(AssetType::Futures)),
            ],
        );
        agg.set_session_clamp_arc(r, GRACE).unwrap();
        agg.on_tick("600000".into(), 10.0, 1.0, sh(14, 59, 30))
            .unwrap();
        agg.on_tick("600000".into(), 11.0, 1.0, sh(15, 0, 20))
            .unwrap();
        // 国债期货 15:00:20 仍在交易, 不钳
        agg.on_tick("T2403".into(), 100.0, 1.0, sh(14, 59, 30))
            .unwrap();
        agg.on_tick("T2403".into(), 101.0, 1.0, sh(15, 0, 20))
            .unwrap();
        let bars = feed.drain_bars_for_test();
        assert_eq!(bars.len(), 1);
        assert_eq!(
            (bars[0].symbol.as_str(), bars[0].close),
            ("T2403", dec!(100))
        );
        assert_eq!(agg.flush_due(sh(15, 0, 59), FLUSH).unwrap(), 1);
        let bars = feed.drain_bars_for_test();
        assert_eq!(
            (bars[0].symbol.as_str(), bars[0].close),
            ("600000", dec!(11))
        );
        assert_eq!(bars[0].timestamp, sh(15, 0, 0));
    }

    #[test]
    fn session_clamp_uses_registered_futures_schedule_for_night_gap() {
        // 期货(未写 sessions, 按资产类型取 CN_FUTURES_DAY): 资产类型来自 set_active
        let feed = DataFeed::new();
        let mut agg = BarAggregator::new(feed.clone(), Some(1), false);
        let r = clamp_resolver(vec![], vec![("rb2405", Some(AssetType::Futures))]);
        agg.set_session_clamp_arc(r, GRACE).unwrap();
        agg.on_tick("rb2405".into(), 10.0, 1.0, sh(14, 59, 30))
            .unwrap();
        agg.on_tick("rb2405".into(), 11.0, 1.0, sh(15, 0, 20))
            .unwrap();
        assert!(feed.drain_bars_for_test().is_empty());
        assert_eq!(agg.flush_due(sh(15, 0, 59), FLUSH).unwrap(), 1);
        let bars = feed.drain_bars_for_test();
        assert_eq!((bars[0].timestamp, bars[0].close), (sh(15, 0, 0), dec!(11)));
    }

    #[test]
    fn negative_session_clamp_grace_is_rejected() {
        let mut agg = BarAggregator::new(DataFeed::new(), Some(1), false);
        assert!(agg.set_session_clamp_arc(stock_resolver(), -1).is_err());
    }
}
