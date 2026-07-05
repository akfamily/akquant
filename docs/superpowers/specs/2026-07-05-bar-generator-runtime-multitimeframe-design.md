# 运行时多周期 Bar 聚合器（BarGenerator）设计

日期：2026-07-05
状态：设计待评审
关联：[[live-broker-readiness]] Strategy API v2 roadmap 遗留项「multi-timeframe Bar aggregation (vnpy BarGenerator)」。补 `docs/zh/advanced/multi_timeframe_feed_api.md` 里「运行时聚合」的空缺。

## 目标（一句话）

提供一个**策略内、流式**的 Bar 聚合器 `BarGenerator`：把 `on_bar` 进来的 Bar 流现场聚合成更高周期（5min/15min/小时/日线）并回调，**回测与实盘用同一句 `update_bar` 代码**，且同参数下与离线 `feed.resample` 产出一致。

## 背景（现状核实）

akquant 已有三块，均**不覆盖**运行时策略内聚合：
- **Feed `resample`/`replay`**（`feed_adapter.BasePandasFeedAdapter`，Python）：**离线** pandas 聚合，产出另一条低频 feed（例 14 用）。回测可行；**实盘不可**（无法离线重采样未来 bar），且需策略订阅多条 feed。签名 `resample(freq, agg=None, label="right", closed="right")`、`replay(freq, ..., session_windows=...)`，`SessionWindow=tuple[str,str]`，用 `pandas.to_offset`。
- **Rust `BarAggregator(feed, interval_min)`**：**tick→1min** 合成，挂 feed/gateway 层（CTP 实盘用）。只做 tick→1min，绑定 `DataFeed`，非策略内任意周期+回调。
- 无任何策略内 bar→更高周期的流式聚合器。

vnpy `BarGenerator` 正是填这个：单一流 → 运行时聚合 → 窗口闭合回调，回测/实盘同一套。

## 决策（用户已定）

- **API 形态**：vnpy 式**独立 helper**（策略里 `new` 一个，`on_bar` 里 `update_bar`，拿回调）。非框架声明式自动路由。
- **窗口对齐**：**时钟对齐 + 可选 session**（对齐 `:00/:05`、整点、日界；`label=right,closed=right`，与 `feed.resample` 一致；配 `session_windows` 则午休/收盘不跨段）。非 vnpy 纯分钟边界。
- **多标的可用**：内部按 `bar.symbol` 维护各自窗口状态，一个实例处理多标的。
- **闭合触发**：流式——新 bar 跨入下一桶时闭合上一窗口回调；`flush()` 收尾闭合；配 session 段末自动闭合。只回调**已闭合**窗口（不发半截，同 `emit_partial=False`）。
- **tick→bar 不在本次范围**：实盘 tick→1min 由 Rust `BarAggregator` 覆盖；本 helper 专注 bar→更高周期。`update_tick` 留后续。
- **实盘/回测同一套**：消费的 `on_bar` 在两模式同形状（P2 统一事件 / 回测 Rust `Bar`），`update_bar` 一句通用。

## 架构

**纯 Python，Rust 零改动**。新模块 `python/akquant/bar_generator.py`，从 `akquant/__init__.py` 导出 `BarGenerator`。聚合产物是**真正的 `Bar`**（`akquant.Bar(timestamp, open, high, low, close, volume, symbol, extra=None)`，Rust pyclass，Python 可构造，现有 ctp/examples/tests 已这么造），策略拿到后与 feed bar 无差别。

### 公有 API

```python
class BarGenerator:
    def __init__(
        self,
        on_window_bar: Callable[[Bar], None],
        window: int = 1,
        interval: str = "minute",          # "minute" | "hour" | "day"
        *,
        session_windows: list[tuple[str, str]] | None = None,
        timezone: str | None = None,       # 时钟/ session 对齐所在市场时区, 如 "Asia/Shanghai"
    ) -> None: ...

    def update_bar(self, bar: Bar) -> None:
        """喂入一根 bar(在策略 on_bar 里调; 回测/实盘同一句)。跨桶时闭合并回调上一窗口。"""

    def flush(self) -> None:
        """强制闭合所有 symbol 的当前未满窗口(回测收尾 / 数据结束)。"""

    def current(self, symbol: str) -> Bar | None:
        """返回某 symbol 正在形成(未闭合)的窗口 bar 快照; 无则 None。只读, 不触发回调。"""
```

### 窗口边界计算（与 feed.resample 对齐）

- `window+interval` → pandas freq：`(5,"minute")→"5min"`、`(1,"hour")→"1h"`、`(1,"day")→"1D"`（内部映射；用户 API 保持 vnpy 式 `window+interval`）。
- 桶键（bucket key）：把 bar 时间戳（纳秒 → 按 `timezone` 转本地）用 `pandas.to_offset(freq)` 向下取整到桶起点；`label=right,closed=right` → 闭合 bar 的 `timestamp` = 桶**右端**（窗口结束时刻），与 `feed.resample` 默认一致。
- 聚合：`open=第一根.open`、`high=max`、`low=min`、`close=最后一根.close`、`volume=sum`；`symbol` 透传；`extra` 默认丢弃（v1 不聚合 extra，见非目标）。
- **闭合语义（流式）**：维护每 symbol 的「当前桶键 + 累积 OHLCV」。`update_bar` 到来时算新 bar 的桶键：
  - 若与当前桶键相同 → 并入累积。
  - 若跨到新桶 → **闭合旧桶**（构造 `Bar` 调 `on_window_bar`），再以新 bar 起一个新桶。
  - 空穴（无成交的桶）**不补**（同 feed.resample「无成交不补值」；不发空 bar）。

### session 分段

- `session_windows`（本地时段，如 `[("09:30","11:30"),("13:00","15:00")]`）给定时：桶不得跨 session 边界；某 bar 属于新 session 且旧桶未闭合 → 先闭合旧桶（段末闭合）再起新桶。语义对齐 `feed.replay(align="session", session_windows=...)`。
- 不给 `session_windows` → 纯时钟对齐（`align` 近似 `global`）。

### 实盘/回测一致性

- 回测：`on_bar` 由 feed 驱动；`update_bar(bar)` 聚合。
- 实盘：`on_bar` 由 gateway（含 Rust `BarAggregator` tick→1min）驱动；同 `update_bar(bar)`。
- **同一实例、同一代码路径**；无模式分支。

## 测试策略

- **vs pandas resample 一致性（核心）**：造一段 1min bar，`BarGenerator(window=5,interval="minute")` 逐根 `update_bar`+`flush` 收集的窗口 bar，与 `pandas.resample("5min",label="right",closed="right").agg(OHLCV)` 的结果逐字段比对（浮点容差）。覆盖 5min/15min/1h/1D。
- **闭合时序**：窗口在下一桶首根到来时闭合（非到来即闭）；`flush` 闭合尾部半截窗口。
- **空穴不补**：跳空的桶不产出空 bar。
- **session 分段**：给 `session_windows`，午休前最后一窗在段末闭合，13:00 起新窗；11:30 与 13:00 不并入同一 5min 桶。
- **多标的**：交替喂两 symbol，各自独立聚合、回调带正确 symbol。
- **timezone**：给 `"Asia/Shanghai"`，日界/整点按本地对齐（造跨 UTC 日界样例验证）。
- **OHLCV 正确**：open=首、high=max、low=min、close=尾、volume=sum。
- **Bar 类型**：回调收到的是 `akquant.Bar` 且字段可读。
- **example 实跑**：新增 example 实际运行 exit 0（发布级要求）。
- **全量回归** `tests/` 全绿；ruff `check`+`format --check`；**Rust 零改动断言** `git diff --stat <merge-base>..HEAD -- '*.rs' 'python/akquant/akquant.pyi'` 空。

## 非目标（YAGNI）

- **tick→bar**（`update_tick`）：实盘由 Rust `BarAggregator` 覆盖，本次不做。
- **周/月周期**：需交易日历，留后续（v1 只 minute/hour/day）。
- **框架声明式自动路由**（`timeframes={...}`→`on_5m`）：本次只出独立 helper 基元；声明式糖留后续。
- **`extra` 字段聚合**：v1 丢弃（如需再定聚合规则）。
- **不改** `feed.resample`/`replay`/Rust `BarAggregator`/`run_backtest` 签名。

## 交付物

- `python/akquant/bar_generator.py`（`BarGenerator`）+ `akquant/__init__.py` 导出。
- `tests/test_bar_generator.py`（上述测试）。
- `examples/NN_bar_generator.py`（多周期策略示例，实跑 exit 0）。
- docs：`docs/zh/advanced/multi_timeframe_feed_api.md` 补「运行时聚合（BarGenerator）」小节，并按 CLAUDE.md 约定——若作教材配套则同步 `docs/zh/textbook/index.md` 映射表（本示例非教材章，预计不涉及）。

## 收尾

- 分支 `feat/bar-generator`（基于 dev）；中文 Conventional Commits + `--no-verify`；未获明确要求不 push。
- 最终 opus 全分支评审后合并 dev。
