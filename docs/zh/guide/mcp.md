# MCP 服务：让 AI 助手写出能跑的策略

AKQuant 自带一个本地 MCP（Model Context Protocol，模型上下文协议）服务。接入 WorkBuddy、Codex、Cursor、Claude 等 AI 助手后，助手可以查询与本机安装版本一致的 API、静态检查策略、准备数据、运行回测，并读到结构化的结果和报错，形成“写—跑—改”的闭环。

## 可以问什么 {: #examples }

- “用平安银行（000001）近三年的日线，跑一个 5 日与 20 日均线交叉的策略，报告收益、最大回撤和交易次数”
- “检查一下 `strategies/my_strategy.py` 有没有写错的地方，再用合成数据跑一遍”
- “把刚才的策略改成只在均线向上时开仓，再跑一次，和上一次的结果对比”
- “刚才那次回测的第 10 到 20 笔成交是什么？”

想让助手自己找数据、写策略、跑回测，请同时接入 AKShare 的官方 MCP，见[与 AKTools 联用](#aktools)。

## 接入客户端 {: #clients }

推荐用 [uv](https://docs.astral.sh/uv/) 的 `uvx` 启动：它会自动下载 AKQuant 并在独立环境中运行，**不需要事先安装 AKQuant，也不需要手动启动服务**。请先[安装 uv](https://docs.astral.sh/uv/getting-started/installation/)，再建好一个项目目录（下文的 `/path/to/akquant-demo`），策略文件和数据文件都要放在这个目录里。

=== "WorkBuddy"

    最简单的方法是直接对 WorkBuddy 说“安装 AKQuant 的官方 MCP，项目目录用 /path/to/akquant-demo”，它会自己完成配置。

    也可以手动添加：在 WorkBuddy 的连接器（MCP）设置中粘贴下面的配置，或者编辑用户目录下的 `.workbuddy/mcp.json`：

    ```json
    {
      "mcpServers": {
        "akquant": {
          "command": "uvx",
          "args": ["--from", "akquant[mcp]", "akquant-mcp", "--root", "/path/to/akquant-demo"],
          "env": {"UV_INDEX_URL": "https://mirrors.aliyun.com/pypi/simple/"}
        }
      }
    }
    ```

    `env` 中的镜像源用来加快国内下载，在国外可以去掉。连接器显示绿色圆点即表示已连接。

=== "Codex"

    在命令行中运行：

    ```bash
    codex mcp add akquant -- uvx --from "akquant[mcp]" akquant-mcp --root /path/to/akquant-demo
    ```

    第一次启动需要下载依赖，可以在 `~/.codex/config.toml` 的 `[mcp_servers.akquant]` 下加上 `startup_timeout_sec = 120` 延长等待时间，见[第一次启动很慢](#first-start)。

=== "Cursor"

    把下面的配置加入项目的 `.cursor/mcp.json`，或全局的 `~/.cursor/mcp.json`：

    ```json
    {
      "mcpServers": {
        "akquant": {
          "command": "uvx",
          "args": ["--from", "akquant[mcp]", "akquant-mcp", "--root", "/path/to/akquant-demo"]
        }
      }
    }
    ```

=== "Claude Code"

    在项目目录中运行：

    ```bash
    claude mcp add akquant -- uvx --from "akquant[mcp]" akquant-mcp --root /path/to/akquant-demo
    ```

=== "Claude Desktop"

    编辑 `claude_desktop_config.json`（macOS 上在 `~/Library/Application Support/Claude/`，Windows 上在 `%APPDATA%\Claude\`），加入与 Cursor 相同的 `mcpServers` 配置，然后重启 Claude Desktop。

配置说明：

- **`--root` 换成你自己的目录**，例如 macOS 上的 `/Users/<用户名>/akquant-demo`。Windows 路径写进 JSON 时用正斜杠（如 `D:/akquant-demo`），或把 `\` 写成 `\\`。不传 `--root` 时取客户端启动服务的目录，各客户端不一样，建议总是显式传入。
- **已经用 pip 安装了 AKQuant**（`pip install "akquant[mcp]"`）时，可以把 `uvx --from "akquant[mcp]" akquant-mcp` 换成 `akquant-mcp`；它必须在 PATH 上，装在 uv/venv 环境里时请写可执行文件的绝对路径。
- **桌面应用报找不到 `uvx`**（如 `spawn uvx ENOENT`）：Claude Desktop、Cursor 等不一定能读到命令行的 PATH，请把 `"command": "uvx"` 改为 `uvx` 的完整路径（Windows 上用 `where uvx` 查看，macOS 和 Linux 上用 `which uvx`）。
- `akquant[mcp]` 已包含 MCP SDK 与 AKShare，`prepare_data` 可以直接用 AKShare 取数。

### 第一次启动很慢 {: #first-start }

第一次启动时 `uvx` 要下载 AKQuant、AKShare 等依赖，可能超过客户端的等待时间，表现为连接超时或启动失败。可以先在命令行中运行一次，下载完成后再重启 AI 助手：

```bash
uvx --from "akquant[mcp]" akquant-mcp --help
```

国内网络下载较慢时，在配置中加上 `"env": {"UV_INDEX_URL": "https://mirrors.aliyun.com/pypi/simple/"}` 使用镜像源；在命令行中预先下载时，也可以先设置同名环境变量。

## 与 AKTools 联用：AI 自动回测 {: #aktools }

[AKTools](https://aktools.akfamily.xyz/guide/mcp/) 是 AKShare 的官方 MCP 服务。两个服务一起接入后，助手可以完成从数据到回测报告的全过程：

| 服务 | 负责 |
|---|---|
| AKTools（AKShare 官方 MCP） | 查找 AKShare 的接口、查看参数、取回数据 |
| AKQuant | 查询 AKQuant 的 API、编写并检查策略、准备行情数据、运行回测、读取成交与权益明细 |

在 WorkBuddy 中输入“安装 AKShare 和 AKQuant 各自的官方 MCP，运行一个示例策略并报告结果”，助手会自己完成安装，然后下载数据、编写策略、运行回测并解读结果。手动配置时把两个服务都加进去：

```json
{
  "mcpServers": {
    "aktools": {
      "command": "uvx",
      "args": ["--from", "aktools", "aktools-mcp"]
    },
    "akquant": {
      "command": "uvx",
      "args": ["--from", "akquant[mcp]", "akquant-mcp", "--root", "/path/to/akquant-demo"]
    }
  }
}
```

回测结果与改进建议由 AI 生成，仅供学习和研究，不构成投资建议。

## 命令行参数 {: #options }

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--root` | 客户端启动服务的目录 | 项目根目录。策略文件和数据文件都必须位于这个目录内 |
| `--max-runs` | 50 | 保留最近多少次回测结果 |

`akquant mcp` 与 `akquant-mcp` 是同一个命令，参数相同。

## 工具清单 {: #tools }

| 工具 | 作用 |
|---|---|
| `search_api` / `get_api` | 查 API 签名与文档（运行时内省，与本机版本一致） |
| `get_guide` | 写策略要点：骨架、下单、指标、数据格式、市场规则 |
| `check_strategy` | 静态检查策略文件，不执行代码 |
| `prepare_data` | 准备数据：本地 CSV/Parquet、AKShare（A 股个股日线）、合成数据 |
| `run_backtest` | 在子进程中回测，返回核心指标、前几笔成交和结构化报错 |
| `get_run_details` | 分页读取成交、委托、日频权益和日志 |

写作要点同时以 resource `akquant://guide/{topic}` 的形式提供。

## 限制 {: #limits }

- 一期只支持类式策略，即 `Strategy` 子类；函数式写法请直接写脚本调用 `run_backtest`。
- `run_backtest` 的 `config` 只开放常用参数；期货合约参数、风控、参数优化、报告图表都不在一期范围内。
- 只支持 stdio 方式，由客户端启动服务；不提供 HTTP 方式。

## 工作目录 {: #workdir }

数据缓存和回测结果默认保存在 `<项目根>/.akquant-mcp/`，可以用环境变量 `AKQUANT_MCP_HOME` 改到别处。建议把它加进 `.gitignore`：

```text
.akquant-mcp/
```

## 安全说明 {: #security }

这是本地开发工具，**不做沙箱**：`run_backtest` 在子进程中执行策略文件，信任程度与你自己运行 `python strategy.py` 相同。路径限制在项目根目录内，只是为了防止 AI 误读其他目录，不是安全隔离。请只运行自己或助手刚写好、看过内容的策略，不要运行来源不明的策略文件。
