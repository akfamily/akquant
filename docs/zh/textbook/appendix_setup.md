# 附录 A：环境与复现

本附录给出一份**可复现**的环境基线，并说明为什么书中回测数字在你本机可能略有不同、以及如何把差异降到最小。环境搭建的入门步骤见第 1 章，本附录侧重"固定版本、固定口径、可复现"。

## A.1 推荐环境

| 项目 | 推荐 | 兼容范围 |
| :--- | :--- | :--- |
| 包管理 | [uv](https://github.com/astral-sh/uv) | —— |
| Python | 3.14 | ≥ 3.10（官方支持 3.10–3.14） |
| 框架 | AKQuant 0.2.45 | 随版本更新，复现时请锁定具体版本 |
| 成书工具 | MkDocs + Material 主题 | —— |

> AKQuant 底层为 Rust 扩展，通过 maturin 构建。普通读者直接 `pip install akquant` 即可获得预编译版本；若需从源码构建，则需额外安装 Rust 工具链与 maturin（详见仓库说明）。

## A.2 依赖版本基线

下表为 AKQuant 0.2.45 声明的主要依赖**下限**（实际安装版本可能更高）。复现书中结果时，建议用锁文件固定到一组确切版本。

| 依赖 | 最低版本 | 用途 |
| :--- | :--- | :--- |
| `akshare` | ≥ 1.0.0 | 行情/财务数据获取（第 3 章起） |
| `pandas` | ≥ 2.2.0 | 数据处理与时间序列 |
| `numpy` | ≥ 2.2.2 | 数值计算与向量化 |
| `polars` | ≥ 0.20.0 | 因子表达式引擎高性能计算（第 14 章） |
| `pyarrow` | ≥ 14.0.0 | Parquet 列式存储读写 |
| `plotly` | ≥ 5.0.0 | 交互式报告与权益曲线（第 13 章） |
| `quantstats` | ≥ 0.0.60 | 第三方绩效 Tearsheet（第 13 章） |
| `scikit-learn` | ≥ 1.0.0 | 机器学习模型（第 12 章） |
| `torch` | ≥ 2.0.0 | 深度学习（第 12 章选学部分） |
| `pydantic` | ≥ 2.0.0 | 参数模型与配置校验 |

## A.3 安装命令

在已激活的 uv 虚拟环境中（建议用清华源加速）：

```bash
uv venv --python 3.14
# Windows: .venv\Scripts\activate ；macOS/Linux: source .venv/bin/activate

uv pip install akquant akshare pandas plotly --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

> Windows 若出现中文输出乱码或编码报错，请在命令前加 `PYTHONIOENCODING=utf-8`，例如
> `PYTHONIOENCODING=utf-8 python examples/textbook/ch01_quickstart.py`。

## A.4 如何尽量复现书中回测数字

回测结果对数据、版本与成交口径都很敏感。要让你的输出尽量贴近书中示例，建议固定以下五项：

1. **锁定框架版本**：例如固定到 AKQuant 0.2.45，避免撮合或指标实现随版本演进而产生差异。
2. **固定数据区间与复权方式**：使用与示例相同的 `symbol`、`start_date`、`end_date` 与 `adjust="qfq"`。
3. **固定随机种子**：机器学习训练（第 12 章）与含随机性的参数搜索（第 11 章）务必设定种子。
4. **明确成交语义**：显式写出 `fill_policy`（`FillMode` 对象，默认 `NextOpen()`，即"当根算信号、下一根开盘成交"），并对齐手续费、印花税与滑点设置。
5. **统一时区口径**：引擎以 UTC 排序，本地时区仅用于展示，解读日志时间时注意区分（第 1、4 章）。

## A.5 已知的差异来源

即便锁定了版本，下列因素仍可能让你的数字与书中不完全一致——这是正常现象，重点在于理解流程而非复刻每一位小数：

- **数据源会被修订**：AKShare 背后的数据会随时间更新、补全或修正历史值，同一接口在不同日期可能返回略有差异的序列。
- **成本参数不同**：手续费率、最低佣金、印花税、过户费与滑点的设置直接改变收益曲线。
- **库与平台差异**：`numpy` / `pandas` 等版本、操作系统与 CPU 都可能带来浮点层面的细小差别。

## A.6 离线可复现实践

为保证"断网也能跑"，配套示例（如 [`examples/textbook/ch16_indicators.py`](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch16_indicators.py)）会在无法联网时回退到**内置合成数据**，并把联网抓取放在明确标注的可选路径上。教学与作业中推荐沿用这一约定：默认用离线数据保证可跑通，需要真实行情时再显式切换到 AKShare。

---

> 配套代码统一位于仓库 `examples/textbook/` 目录；各章"本章实践入口"给出了对应脚本与运行命令。
