# MCP Server: Let AI Assistants Write Strategies That Run

AKQuant ships a local MCP (Model Context Protocol) server. Once connected to WorkBuddy, Codex, Cursor, Claude or another AI assistant, the assistant can look up the API of the AKQuant version installed on your machine, statically check strategies, prepare data, run backtests, and read structured results and errors — closing the write–run–fix loop.

## What you can ask {: #examples }

- "Using Ping An Bank (000001) daily bars for the last three years, run a 5/20-day moving-average crossover and report return, max drawdown and trade count"
- "Check `strategies/my_strategy.py` for mistakes, then run it on synthetic data"
- "Only open positions when the moving average is rising, run it again and compare with the previous run"
- "What were trades 10 to 20 in that backtest?"

To let the assistant fetch data, write the strategy and backtest it on its own, also connect AKShare's official MCP server — see [Using with AKTools](#aktools).

## Connecting a client {: #clients }

We recommend starting it with [uv](https://docs.astral.sh/uv/)'s `uvx`: it downloads AKQuant and runs it in an isolated environment, so **you don't need to install AKQuant or start a server first**. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), then create a project directory (`/path/to/akquant-demo` below); strategy and data files must live inside it.

=== "WorkBuddy"

    The easiest way is to tell WorkBuddy "install AKQuant's official MCP, with /path/to/akquant-demo as the project directory"; it configures itself.

    To add it manually, paste the configuration below into WorkBuddy's connector (MCP) settings, or edit `.workbuddy/mcp.json` in your home directory:

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

    A green dot on the connector means it is connected.

=== "Codex"

    Run:

    ```bash
    codex mcp add akquant -- uvx --from "akquant[mcp]" akquant-mcp --root /path/to/akquant-demo
    ```

    The first start downloads dependencies; you can add `startup_timeout_sec = 120` under `[mcp_servers.akquant]` in `~/.codex/config.toml` to wait longer, see [Slow first start](#first-start).

=== "Cursor"

    Add the configuration below to the project's `.cursor/mcp.json`, or the global `~/.cursor/mcp.json`:

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

    Run in the project directory:

    ```bash
    claude mcp add akquant -- uvx --from "akquant[mcp]" akquant-mcp --root /path/to/akquant-demo
    ```

=== "Claude Desktop"

    Edit `claude_desktop_config.json` (in `~/Library/Application Support/Claude/` on macOS, `%APPDATA%\Claude\` on Windows), add the same `mcpServers` configuration as for Cursor, and restart Claude Desktop.

Notes:

- **Replace `--root` with your own directory**, e.g. `/Users/<you>/akquant-demo` on macOS. In JSON, write Windows paths with forward slashes or escape `\` as `\\`. Without `--root` the server uses the directory the client starts it in, which differs between clients, so always pass it.
- **If AKQuant is already installed with pip** (`pip install "akquant[mcp]"`), you can replace `uvx --from "akquant[mcp]" akquant-mcp` with `akquant-mcp`; it must be on PATH, or use the absolute path of the executable inside your uv/venv environment.
- **Desktop apps can't find `uvx`** (e.g. `spawn uvx ENOENT`): Claude Desktop, Cursor and similar apps may not see your shell's PATH; replace `"command": "uvx"` with the full path of `uvx` (`where uvx` on Windows, `which uvx` on macOS and Linux).
- `akquant[mcp]` includes the MCP SDK and AKShare, so `prepare_data` can fetch data from AKShare directly.

### Slow first start {: #first-start }

On first start `uvx` downloads AKQuant, AKShare and other dependencies, which can exceed the client's timeout and show up as a connection timeout or startup failure. Run it once in a terminal to download everything, then restart the assistant:

```bash
uvx --from "akquant[mcp]" akquant-mcp --help
```

In mainland China, add `"env": {"UV_INDEX_URL": "https://mirrors.aliyun.com/pypi/simple/"}` to the configuration to use a mirror; set the same environment variable before pre-downloading in a terminal.

## Using with AKTools: AI-driven backtests {: #aktools }

[AKTools](https://aktools.akfamily.xyz/guide/mcp/) is AKShare's official MCP server. With both connected, the assistant can go from data to backtest report on its own:

| Server | Responsible for |
|---|---|
| AKTools (AKShare's official MCP) | Finding AKShare interfaces, checking parameters, fetching data |
| AKQuant | Looking up AKQuant APIs, writing and checking strategies, preparing market data, running backtests, reading trades and equity details |

In WorkBuddy, say "install the official MCP servers of AKShare and AKQuant, run an example strategy and report the results"; the assistant installs both, downloads data, writes a strategy, runs the backtest and explains the results. To configure manually, add both servers:

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

Backtest results and suggestions are AI-generated, for learning and research only, and are not investment advice.

## Command-line options {: #options }

| Option | Default | Description |
|---|---|---|
| `--root` | The directory the client starts the server in | Project root. Strategy and data files must be inside it |
| `--max-runs` | 50 | How many recent backtest results to keep |

`akquant mcp` and `akquant-mcp` are the same command with the same options.

## Tools {: #tools }

| Tool | Purpose |
|---|---|
| `search_api` / `get_api` | Look up API signatures and docs (runtime introspection, matches the installed version) |
| `get_guide` | Strategy-writing notes: skeleton, orders, indicators, data format, market rules |
| `check_strategy` | Static check of a strategy file, without executing it |
| `prepare_data` | Prepare data: local CSV/Parquet, AKShare (A-share daily bars), synthetic |
| `run_backtest` | Run a backtest in a subprocess; returns core metrics, the first trades and structured errors |
| `get_run_details` | Page through trades, orders, daily equity and logs |

The writing notes are also available as the resource `akquant://guide/{topic}`.

## Limitations {: #limits }

- Phase 1 supports class-based strategies (`Strategy` subclasses) only; for function-style strategies, write a script that calls `run_backtest`.
- The `config` of `run_backtest` exposes common parameters only; futures contract parameters, risk controls, parameter optimization and report charts are out of scope for phase 1.
- stdio only: the client starts the server; there is no HTTP mode.

## Working directory {: #workdir }

Data cache and backtest results are stored in `<project root>/.akquant-mcp/` by default; set `AKQUANT_MCP_HOME` to move them. Add it to `.gitignore`:

```text
.akquant-mcp/
```

## Security {: #security }

This is a local development tool with **no sandbox**: `run_backtest` executes the strategy file in a subprocess, with the same trust as running `python strategy.py` yourself. Restricting paths to the project root only keeps the AI from reading other directories by mistake; it is not isolation. Only run strategies you or the assistant just wrote and reviewed; never run strategy files of unknown origin.
