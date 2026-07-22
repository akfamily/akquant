# 可视化与报告

以下是 AKQuant 生成的交互式回测报告示例。您可以在此页面直接与图表进行交互，查看详细的回测数据。

<iframe src="../../assets/reports/akquant_report.html" width="100%" height="1000px" frameborder="0" style="border: 1px solid #eee; border-radius: 4px;"></iframe>

## 基准对比

`BacktestResult.report` 支持直接传入基准收益率序列：

```python
benchmark_returns = (
    benchmark_df.set_index("date")["close"].pct_change().fillna(0.0)
)
result.report(
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
- `summary`: 汇总指标，如 `total_excess`、`annual_excess`、`tracking_error`、`information_ratio`、`beta`、`alpha`
- `series`: 对齐后的逐日序列，包含策略收益、基准收益、超额收益及三条累计收益曲线
- `meta`: 对齐样本数、起止日期、年化因子等元信息

推荐实践：

- 后端负责准备 benchmark 收益率序列并调用 `result.benchmark_analysis(...)`
- 前端直接消费 `summary + series + meta`
- `result.report(..., benchmark=...)` 与前端页面应复用同一份 benchmark analysis 逻辑，而不是各自重新计算

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

## Lightweight Charts 报告与交易复盘（plotly 替代）

`akquant.lwc` 提供基于 TradingView Lightweight Charts 的可视化路径，
不依赖 plotly；图表库已内嵌进报告文件，离线可打开：

```python
# 静态单文件报告：绩效指标 + 净值/回撤 + K线买卖点复盘
result.report_lwc(
    market_data={"600000": df1, "600004": df2},  # {symbol: DataFrame}
    filename="akquant_lwc_report.html",
    show=True,
)
```

报告的交易复盘区块支持**网页内热切换复盘标的**：在输入框中键入
（或从自动补全中选择）股票代码即可切换 K 线买卖点图与交易明细表；
点击明细行可定位到对应区间，悬停买卖点可查看该笔交易详情。

如需复盘报告中未内嵌的标的，可启动交互式复盘服务，页面输入任意
可解析的代码后由服务端按需加载：

```python
result.serve_review(
    market_data={"600000": df1},            # 预加载标的（可选）
    data_provider=lambda code: load_df(code),  # 按需解析新代码（可选）
    port=8765,
)
```

也可以直接调用模块级 API：`akquant.lwc.plot_report(...)` 与
`akquant.lwc.serve_review(...)`。
