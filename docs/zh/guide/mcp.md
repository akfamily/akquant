# MCP 服务：让 AI 助手写出能跑的策略

AKQuant 自带一个本地 MCP（Model Context Protocol）服务。接入 Claude Code、Cursor 等 AI 编程助手后，AI 可以查询与本机安装版本一致的 API、静态检查策略、准备数据、跑回测，并读到结构化的结果和报错，形成“写—跑—改”的闭环。

## 安装

```bash
pip install "akquant[mcp]"
```

安装后会多出一个 `akquant-mcp` 命令。它通过 stdio 与 AI 客户端通信，一般由客户端负责启动，不需要手动运行。

## 接入客户端

Claude Code（在项目根目录执行）：

```bash
claude mcp add akquant -- akquant-mcp
```

`akquant-mcp` 必须在 PATH 上；如果 AKQuant 装在 uv/venv 环境里，请改用 `claude mcp add akquant -- uv run akquant-mcp`，或写可执行文件的绝对路径。

Cursor 等使用 JSON 配置的客户端：

```json
{
  "mcpServers": {
    "akquant": {"command": "akquant-mcp", "args": ["--root", "/path/to/project"]}
  }
}
```

命令行参数：

- `--root`：项目根目录，不传时取客户端启动服务的目录，建议显式传入。策略文件和数据文件都必须位于这个目录内。
- `--max-runs`：保留最近多少次回测结果，默认 50。

## 工具清单

| 工具 | 作用 |
|---|---|
| `search_api` / `get_api` | 查 API 签名与文档（运行时内省，与本机版本一致） |
| `get_guide` | 写策略要点：骨架、下单、指标、数据格式、市场规则 |
| `check_strategy` | 静态检查策略文件，不执行代码 |
| `prepare_data` | 准备数据：本地 CSV/Parquet、AKShare（A 股个股日线）、合成数据 |
| `run_backtest` | 在子进程中回测，返回核心指标、前几笔成交和结构化报错 |
| `get_run_details` | 分页读取成交、委托、日频权益和日志 |

写作要点同时以 resource `akquant://guide/{topic}` 的形式提供。

## 限制

- 一期只支持类式策略，即 `Strategy` 子类；函数式写法请直接写脚本调用 `run_backtest`。
- `run_backtest` 的 `config` 只开放常用参数；期货合约参数、风控、参数优化、报告图表都不在一期范围内。

## 工作目录

数据缓存和回测结果默认保存在 `<项目根>/.akquant-mcp/`，可以用环境变量 `AKQUANT_MCP_HOME` 改到别处。建议把它加进 `.gitignore`：

```text
.akquant-mcp/
```

## 安全说明

这是本地开发工具，**不做沙箱**：`run_backtest` 在子进程中执行策略文件，信任程度与你自己运行 `python strategy.py` 相同。路径限制在项目根目录内，只是为了防止 AI 误读其他目录，不是安全隔离。不要用它运行来源不明的策略文件。
