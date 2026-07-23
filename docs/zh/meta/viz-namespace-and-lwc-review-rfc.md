# RFC:可视化接口收敛(`viz` 命名空间)与 LWC 交易复盘

> **状态**:提案(Proposed) · **日期**:2026-07-23 · **范围**:Python 结果层 `BacktestResult` 的**可视化接口**(`plot` / `plot_indicators` / `report` / `report_quantstats`)收敛为 `result.viz.*` 命名空间;新增 LWC(TradingView Lightweight Charts)**交易复盘**模块 `akquant.lwc`。**允许破坏性变更(硬改名,直接删除旧方法,不留兼容别名)**,与 [timer-api-rfc.md](timer-api-rfc.md)、[hooks-rfc.md](hooks-rfc.md) 的处理方式一致。
>
> 对标:**QuantStats**(`qs.reports.html` 单一报告入口)、**backtrader**(`cerebro.plot()` 单入口 + `Bokeh` 第三方后端并存)、**vectorbt**(`portfolio.plot()` + 各子图独立 `.plot()` 方法)、**TradingView Lightweight Charts**(金融时序专用、离线、~160KB)。
>
> 与引擎**正交**:本 RFC 只改**结果层可视化接口的命名/分层**并新增一个纯消费 `BacktestResult` 公开产物的可视化后端。**不碰引擎、不动 `__engine_rule_version__`、golden 测试零影响**。

---

## 0. 背景与动机

用户反馈:`result` 上的一组可视化方法 `plot()` / `plot_indicators()` / `report()` / `report_quantstats()` 看起来"为同一目的存在多套接口",存在**选择困难**。审计后厘清:真正的坏味道**不是功能重复**,而是**命名只编码了"介质",没编码"用户要干的事"**。

这些方法其实在**三个正交轴**上变化:

- **轴 A 介质**:交互 `Figure`(Notebook 里瞄一眼) vs 静态 HTML 文件(存档/发人);
- **轴 B 内容**:整体绩效 vs 单标的买卖点 vs 指标序列;
- **轴 C 引擎**:plotly 原生 vs QuantStats vs(拟新增)LWC。

而 `plot` / `report` 这组名字只暴露了**轴 A**,用户脑中想的是"我要看某只票的买卖点对不对"(轴 B),却被迫在"plot 还是 report"里纠结。**这是路牌问题,不是冗余问题。**

同时,现有 plotly 报告在**交互式 K 线复盘**上有真实短板:`report(include_trade_kline=True)` 的 K 线是**生成时锁定单标的、日内/大数据量卡顿**的静态图。这正是 LWC 的主场——需要一条**交互、多标的热切换、大数据量流畅**的复盘路径来补齐。

## 1. 目标与非目标

**目标**

- 把 `result` 上分散的可视化方法收敛进 **`result.viz.*`** 命名空间,一个入口 + Tab 补全即可发现全家桶,方法**按"用户要干的事"命名**(job-oriented),而非按介质/引擎。
- 提供**可视化决策表**(进文档),用两个问题("看什么 / 什么形态")替用户完成选择。
- 新增 **`result.viz.review()`**:基于 LWC 的交互式 K 线交易复盘,补齐 plotly 在大数据量/日内 K 线交互上的短板。
- 破坏性变更**集中一次**,配套更新示例、中英文档、教材映射表、CHANGELOG。

**非目标**

- **不替换 plotly**:分析类图表(月度热力图、收益分布、滚动指标、风控/归因)永远归 plotly,LWC 只做时间序列 K 线。二者是分工,非替代。
- **不做第一版服务端**:LWC v1 只出**静态自包含 HTML**;按需加载的 HTTP 服务后置(见 §9)。
- **不改数据导出**:`result.to_quantstats()` 返回 `pd.Series`,是数据方法非可视化,**留在 `result` 顶层不动**。
- **不碰引擎**:不动任何回测行为、不改 `__engine_rule_version__`。

## 2. 设计原则

1. **按 job 命名,不按工具命名**。方法名回答"用户要干什么"(dashboard/report/review/indicators),介质与引擎作为实现细节收在内部或参数。`report` 与 `quantstats` 是唯一例外——此处**引擎即 job**(用户明确"我就要 QuantStats 那套版式")。
2. **命名空间聚合,顶层瘦身**。`result.viz.` 一处发现全部可视化能力,`BacktestResult` 顶层不再被五个 viz 方法撑爆。
3. **可选依赖惰性化**。`viz` 属性本身零依赖(不触发 plotly/lwc 导入);重活到具体方法调用才 lazy import,完全沿用现有 `akquant[plot]` / `akquant[quantstats]` 策略,LWC 甚至零新增运行时依赖(纯 stdlib + 已有 pandas)。
4. **plotly 守分析,LWC 守 K 线**。分界线唯一:"这张图是不是随时间连续的价格序列?"是 → LWC 有资格;否(热力图/直方图/柱状/散点/分类占比)→ 只能 plotly。
5. **共享 normalizer,杜绝分叉**。行情数据的中英列名容错 / OHLCV 规整**只有一份**,plotly 与 LWC 共用,避免两处规则漂移。
6. **破坏性变更集中一次**,纯净断裂(不留 `DeprecationWarning` 垫片),一个 minor 版号内完成。

## 3. 现状审计

| API | 介质 | 内容 | 引擎 | 命名问题 | 处置 |
|---|---|---|---|---|---|
| `result.plot()` | 交互 Figure | 组合级面板 | plotly | 名字是介质非内容;`symbol` 参数早已 reserved 废弃 | **改名** `viz.dashboard()`,删废参 |
| `result.plot_indicators()` | 交互 Figure | 指标序列 | plotly | 同上 | **改名** `viz.indicators()` |
| `result.report()` | 静态 HTML | 全量(面板+K线+指标+风控+归因) | plotly | 与 `report_quantstats` 疑似重复 | **改名** `viz.report()`,文档点明与 quantstats 差异 |
| `result.report_quantstats()` | 静态 HTML | 组合级绩效 | quantstats | 冗余前缀;与 `report` 疑似重复 | **改名** `viz.quantstats()` |
| `result.to_quantstats()` | —(返回 Series) | 数据导出 | — | 无 | **保持不动**(非可视化) |
| —— | 交互 HTML | 单标的 K 线+买卖点 | LWC | —— | **新增** `viz.review()` |

**结论**:真正"能做同一件事"的只有 `report` vs `report_quantstats`(两个完整绩效 HTML 报告),且是**故意的双引擎**——`report()` akquant 原生、A股/中文优化、与引擎打通(风控拒单/强平/归因审计);`report_quantstats()` 是行业标准 QS 版式。二者该留,靠**文档点明差异**消除困惑。其余方法是"独立小挂件 vs 大报告内嵌区块"的正常二元性(类比 matplotlib 单图 vs subplot),非冗余。

## 4. 变更提案

### 4.1 收敛为 `result.viz.*` 命名空间

```python
result.viz.dashboard(show=True)               # 交互式总览(权益/回撤)→ 返回 Figure
result.viz.indicators(name=None, symbol=None) # 指标序列预览 → 返回 Figure
result.viz.report(market_data=None, ...)      # akquant 原生全量静态 HTML(含风控/归因/K线快照)
result.viz.quantstats(benchmark=None, ...)    # QuantStats 版式报告(引擎即 job)
result.viz.review(market_data, ...)           # ★新增:LWC 交互式 K 线复盘,多标的热切换
```

老 → 新映射(**破坏性,直接删旧名**):

| 旧 | 新 |
|---|---|
| `result.plot()` | `result.viz.dashboard()`(顺带删废参 `symbol`) |
| `result.plot_indicators()` | `result.viz.indicators()` |
| `result.report()` | `result.viz.report()` |
| `result.report_quantstats()` | `result.viz.quantstats()` |
| `result.to_quantstats()` | 不变(留顶层) |
| —— | `result.viz.review()`(新增) |

### 4.2 访问器机制(示意)

```python
# backtest/_viz.py —— 每个方法内部仍 lazy import,保持 plotly/lwc 可选
class VizNamespace:
    def __init__(self, result): self._r = result
    def dashboard(self, show=True, title="Backtest Result"):
        from ..plot import plot_dashboard
        return plot_dashboard(result=self._r, show=show, title=title)
    def review(self, market_data, *, title="AKQuant 交易复盘", show=False, **kw):
        from ..lwc import plot_kline_review
        return plot_kline_review(self._r, market_data, title=title, show=show, **kw)
    # report / quantstats / indicators 同样 lazy delegate

# backtest/result.py
@property
def viz(self) -> "VizNamespace":
    from ._viz import VizNamespace
    return VizNamespace(self)
```

`viz` 属性零依赖(不触发任何绘图库导入),重活到方法调用才 lazy import。

### 4.3 可视化决策表(进用户文档)

| 我想… | 当场交互看 | 要存档/发人的文件 |
|---|---|---|
| **整体绩效** | `viz.dashboard()` | `viz.report()` / `viz.quantstats()` |
| **单票买卖点** | `viz.review()`(LWC,多标的热切换) | `viz.report(include_trade_kline=True)` 静态快照 |
| **指标序列** | `viz.indicators()` | `viz.report(include_indicators=True)` |

`report` vs `quantstats` 文档点明:前者 akquant 原生、A股/中文、与引擎打通;后者 QS 标准版式。
`review` vs `report` 内 K 线:前者 LWC 交互、多标的、日内大数据量流畅;后者 plotly 静态、锁单标的。

## 5. LWC 交易复盘模块设计

### 5.1 定位与 scope

**只做一件事:交易复盘 K 线**——蜡烛 + 成交量 + 买卖 marker + 十字光标 tooltip + 页内多标的切换。分析类图表永远归 plotly,不碰。

**第一版只做静态自包含 HTML,砍掉服务端**:LWC 是 vendored JS,可把所有回测标的一起内嵌、页内输入框切换,**不需要服务器**,产物可直接发人/存档,天然满足仓库 "示例 exit 0" 约定(不阻塞、不开服务)。按需加载的 HTTP 服务是 20% 场景但成本最大(阻塞 / 开浏览器 / 跑用户回调 / XSS 面 / 生命周期),后置为 v3(见 §9)。

### 5.2 模块结构

```
python/akquant/
├── plot/
│   ├── _market_data.py    ← 新增:从 report.py 抽出的共享 normalizer
│   │                         (_normalize_market_data_frame / _resolve_market_data_column)
│   ├── utils.py           ← 复用现有 THEMES(红涨绿跌已在此)
│   └── report.py          ← 改为 from ._market_data import ...(去重)
└── lwc/
    ├── __init__.py        ← 只导出 plot_kline_review
    ├── review.py          ← 编排:build payload → render → 落盘/show
    ├── _payload.py        ← result + market_data → JSON(复用 plot._market_data)
    ├── _template.py       ← HTML/CSS/JS 模板,占位符 __TITLE__/__LWC_JS__/__APP_JSON__
    └── assets/
        ├── lightweight-charts.standalone.production.js
        └── LICENSE        ← Apache-2.0 NOTICE(vendored 合规硬要求)
```

**核心约束:`lwc/` 不重造 normalizer,复用 `plot/_market_data.py`**(§2 原则 5)。

### 5.3 公开 API

```python
# python/akquant/lwc/review.py
def plot_kline_review(
    result, market_data, *,
    title="AKQuant 交易复盘",
    filename="akquant_review.html",
    symbols=None,        # None = 全部有交易的标的
    plot_symbol=None,    # 初始显示;None = 交易最多的
    theme="light",       # 复用 plot.utils.THEMES
    show=False,
) -> str:                # 返回生成文件的绝对路径
    ...
```

用户侧一行:`result.viz.review(market_data, show=True)`。

### 5.4 数据契约(payload schema)

`_payload.py` 产出前端唯一认识的 JSON:

```
{
  "title": str,
  "theme": {"up": "#d32f2f", "down": "#2e7d32", ...},   # 从 THEMES 取,红涨绿跌
  "initialSymbol": str,
  "symbols": [str, ...],                                 # 输入框候选
  "payloads": {                                          # 每标的一份
    "600000": {
      "candles":  [{"time","open","high","low","close"}, ...],
      "volume":   [{"time","value","color"}, ...],
      "markers":  [{"time","position","shape","text":"B/S @price"}, ...]
    }
  }
}
```

- `candles/volume` ← `market_data` 经共享 normalizer 规整;
- `markers` ← `result.trades_df` 按 symbol 分组(方向→箭头,价格→text);
- **时间格式统一**:日频用 `"YYYY-MM-DD"`,日内用 UNIX 秒。A股日频 vs 日内的时间处理是主要坑,集中在 normalizer 里解决。

> **实现修订(P2)**:为支持页内明暗切换,最终 payload **不烘焙颜色**——`volume` 项为 `{time,value,up:bool}`、`marker` 项为 `{time,position,shape,text,buy:bool}`,并内联 `themes:{light,dark}` 两套色板 + `initial_theme`,由前端按当前主题动态上色。上方 schema 为设计初稿,以此修订为准。

### 5.5 渲染管线(安全 & 健壮)

`render_html(title, payload)`:

1. `app_json = json.dumps(payload, ensure_ascii=False)`,再 `.replace("<", "\\u003c")` 防 `</script>` 逃逸;
2. `title` 走 `html.escape()`;
3. 模板用 **`.replace("__X__", ...)` 占位符,不用 `str.format()`**(CSS 里的 `{}` 会炸 format——现有 plotly 模板即有此隐患,LWC 侧规避);
4. vendored JS 用 `.replace(..., 1)` 单次注入。

### 5.6 打包 & 依赖

- `pyproject.toml` 加 `include` 把 `lwc/assets/*` 打进 wheel/sdist(含 `LICENSE`,Apache-2.0 归属为合规硬要求);
- **零新增运行时依赖**:全 stdlib + 已有 pandas;LWC 路径甚至不需要 plotly。`akquant[plot]` / `akquant[quantstats]` extra 不受影响。

## 6. 测试

- `tests/test_viz_namespace.py`:`result.viz.*` 五方法可达;旧方法名(`plot`/`report`/…)**已删除**(`AttributeError`)确认破坏性到位;`viz` 属性不触发 plotly/lwc 导入。
- `tests/test_lwc_payload.py`:normalizer 复用后的中英列名 / dict·单表·长表三态 / 日频 vs 日内时间格式 / marker 方向映射 / 空 trades / 缺标的的降级。
- `tests/test_lwc_render.py`:XSS 注入(标题带 `<script>`、数据带 `</script>`)确认被转义;产物是合法自包含 HTML、离线可开(不发网络请求)。
- 示例 `examples/NN_lwc_review.py`:只调 `result.viz.review(...)` 静态产物,**实跑 exit 0**(不阻塞、不开服务)。

## 7. 破坏性变更与迁移

- **纯净断裂**:直接删除 `result.plot/plot_indicators/report/report_quantstats`,**不留 `DeprecationWarning` 垫片**(经确认)。
- 版本:API 破坏,建议 `0.3.x → 0.4.0`;不碰引擎,`__engine_rule_version__` 不变,golden 测试零影响。
- 配套一次性更新:仓库内所有 `examples/`、`docs/zh|en/guide/visualization.md`、教材映射表(`docs/zh/textbook/index.md`)与相关章「本章实践入口」、CHANGELOG 迁移说明(附老→新对照表)。

## 8. 落地阶段(P0→P2)

- **P0 命名空间收敛**(✅ 已落地):抽 `plot/_market_data.py` → 建 `backtest/_viz.py` + `result.viz` 属性 → 删旧方法 → 全仓 examples/docs 改引用 → 测试 + `check_docs_api_examples.py` 旧方法 guard。**此阶段不含 LWC,可独立合并**。
- **P1 LWC 静态复盘**(✅ 已落地):`lwc/` 骨架(`_payload`/`_template`/`review`)+ vendored `lightweight-charts@5.2.0` standalone JS + `plot_kline_review` + `viz.review()` + payload/render/安全性测试(`test_lwc_review.py`)+ 示例 `examples/67_lwc_trade_review.py`。
- **P2 打磨**(✅ 已落地):**页内明暗主题切换**(payload 改为主题无关——量柱带 `up`、marker 带 `buy` 布尔,颜色由前端按当前主题动态上色,切换只 `applyOptions`+重着色不重建数据);**日内大数据量**——candle/volume 向量化构建(numpy 数组替代 `iterrows`)+ 时间戳去重,6 万根 K 线压测 <5s。

> 说明:
> - P1 时间轴已按数据自动区分日频(`YYYY-MM-DD`)/日内(`UTCTimestamp` 秒);marker 时间对齐到最近 bar,并处理引擎 tz-aware 成交时间与行情 tz-naive 索引的时区差异。
> - P2 修了一个量纲坑:规范化后索引常为 `datetime64[us]`,向量化换算 UTC 秒前须先 `as_unit("ns")`,否则整段时间塌缩为同值。

## 9. 后续与非本 RFC 决策

- **服务端按需加载(v3,可选)**:若多标的多到不宜全内嵌,再引入 HTTP 服务;届时**倾向做成插件路径**(与 broker 插件同模式,走 entry-point),避免核心背 HTTP 服务/生命周期。
- **指标叠加(未来)**:复盘图可吃同一条 `IndicatorSink` 指标流,把回测指标叠加到 K 线上,与实盘/回测指标出口三处一致。**不进 v1**。

## 10. 备选方案与否决理由

- **用 LWC 重构/替换 plotly**:❌ 否决。plotly 报告 90% 内容非 K 线(热力图/直方图/散点/分类),LWC 原生不支持;重构 = 用专科件换掉功能完备的瑞士军刀,数周工作量 + 全新前端维护负债,赔上已成熟的 80%。
- **一个 `report(engine=, mode=, content=)` 巨型多 flag 函数**:❌ 否决。返回类型统一不了(`dashboard` 返 Figure、`report` 写文件),是"一函数二十开关"反模式。
- **保留旧方法名 + `DeprecationWarning` 垫片**:已评估,经确认采用**纯净断裂**,不保留。
- **直接采纳 PR #340**:❌ 否决。其 scope 过宽(内置 HTTP Web 应用)、服务端属应用层不该进核心、normalizer 重复造轮子、测试留在本地未提交。本 RFC 汲取其 LWC 选型与 XSS/打包经验,但按 akquant 既有分层重新设计。
