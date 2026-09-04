# 可视化与报告

以下是 AKQuant 生成的交互式回测报告示例。您可以在此页面直接与图表进行交互，查看详细的回测数据。

<iframe src="../../assets/reports/akquant_report.html" width="100%" height="1000px" frameborder="0" style="border: 1px solid #eee; border-radius: 4px;"></iframe>

## 基准对比

`BacktestResult.viz.report` 支持直接传入基准收益率序列：

```python
benchmark_returns = (
    benchmark_df.set_index("date")["close"].pct_change().fillna(0.0)
)
result.viz.report(
    filename="akquant_report.html",
    benchmark=benchmark_returns,
    show=False,
)
```

报告会新增“基准对比 (Benchmark Comparison)”区块，提供累计超额收益、年化超额收益、跟踪误差、信息比率、Beta、Alpha 等指标，并展示策略/基准/超额三条累计收益曲线。

## 结构化 Benchmark Analysis

从当前版本开始，AKQuant 不再只有 HTML 报告里的基准对比区块，还提供可直接给前端、API 或离线分析复用的结构化 benchmark analysis：

```python
benchmark_returns = (
    benchmark_df.set_index("date")["close"].pct_change().fillna(0.0)
)

payload = result.benchmark_analysis(
    benchmark=benchmark_returns,
    curve_freq="D",
)

print(payload["schema_version"])
print(payload["summary"]["annual_excess"])
print(payload["series"][0])
```

返回 payload 主要包含：

- `schema_version`: 数据契约版本
- `available`: 当前 benchmark analysis 是否可用
- `reason`: 当 benchmark 无法对齐或输入非法时的原因
- `benchmark.label`: 基准显示名称
- `summary`: 汇总指标，如 `total_excess`、`annual_excess`、`tracking_error`、`information_ratio`、`beta`、`alpha`（收益/误差类字段为小数，前端展示百分比需 ×100；比率类字段无量纲不乘，详见 [回测结果与指标详解](./analysis.md#summary-字段单位)）
- `series`: 对齐后的逐日序列，包含策略收益、基准收益、超额收益及三条累计收益曲线
- `meta`: 对齐样本数、起止日期、年化因子等元信息

推荐实践：

- 后端负责准备 benchmark 收益率序列并调用 `result.benchmark_analysis(...)`
- 前端直接消费 `summary + series + meta`
- `result.viz.report(..., benchmark=...)` 与前端页面应复用同一份 benchmark analysis 逻辑，而不是各自重新计算

## 导出给前端或归档

如果需要把 benchmark analysis 固化为回测产物，可以直接导出：

```python
result.export_benchmark_analysis(
    path="artifacts/benchmark_analysis.json",
    benchmark=benchmark_returns,
    format="json",
    curve_freq="D",
)
```

也支持 `format="parquet"`，会输出：

- `series.parquet`: 逐点时间序列
- `metadata.json`: 汇总指标与元信息

## LWC 交互式交易复盘

`result.viz.review()` 基于 [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts) 生成**离线自包含**的单文件 HTML，在 K 线上标注买卖点，面向大数据量 / 日内的交易复盘。

它与 `result.viz.report()` 是**互补而非替代**：分析类图表（权益曲线、回撤、热力图、归因等）仍由 `report()` 的 plotly 负责；`review()` 只补足「交互式 K 线 + 买卖点」这一场景，适合逐笔复盘成交时机。

```python
# market_data 为单个 DataFrame 或 {symbol: df} 字典
path = result.viz.review(
    market_data=df,
    title="AKQuant 交易复盘",
    theme="dark",          # 初始主题 "light" / "dark"，页面内可即时切换
    filename="akquant_review.html",
    show=False,            # True 则自动打开浏览器
)
```

要点：

- 生成的 HTML 内联了 lightweight-charts，**无 CDN 依赖**，可离线打开与归档。
- 页面顶部有**明暗主题切换按钮**，`theme` 参数只决定初始主题；切换时即时重着色，无需重新生成文件。
- 多标的行情（`{symbol: df}`）会在页面顶部提供标的切换下拉，`initial_symbol` 可指定初始展示标的。
- 行情列名大小写不敏感，并兼容中文列名（`开盘/最高/最低/收盘/成交量/日期` 等）。
- 日频数据用 `YYYY-MM-DD` 时间轴，日内数据自动切换为带时分的时间轴。
- 面向**大数据量 / 日内**优化:payload 用向量化构建、时间戳自动去重,数万根 K 线也能流畅复盘(这正是相对 plotly 分析图的核心优势)。

完整示例见 `examples/67_lwc_trade_review.py`。
