# MCP Server: Help AI Assistants Write Strategies That Run

AKQuant ships a local MCP (Model Context Protocol) server. Once connected to an AI coding assistant such as Claude Code or Cursor, the AI can look up APIs matching your installed version, statically check a strategy, prepare data, run backtests, and read structured results and errors — a write–run–fix loop.

## Install

```bash
pip install "akquant[mcp]"
```

This adds an `akquant-mcp` command. It talks to the AI client over stdio and is normally launched by the client.

## Connect a client

Claude Code (run in the project root):

```bash
claude mcp add akquant -- akquant-mcp
```

`akquant-mcp` must be on PATH. If AKQuant is installed in a uv/venv environment, use `claude mcp add akquant -- uv run akquant-mcp` or the absolute path of the executable.

Clients configured with JSON (Cursor etc.):

```json
{
  "mcpServers": {
    "akquant": {"command": "akquant-mcp", "args": ["--root", "/path/to/project"]}
  }
}
```

Options:

- `--root`: project root. Without it the root is the directory the client launches the server in, so pass it explicitly. Strategy and data files must live under it.
- `--max-runs`: how many recent backtest runs to keep, default 50.

## Tools

| Tool | Purpose |
|---|---|
| `search_api` / `get_api` | Look up API signatures and docs (runtime introspection, matches your installed version) |
| `get_guide` | Strategy-writing notes: skeleton, orders, indicators, data format, market rules |
| `check_strategy` | Static check of a strategy file; does not execute it |
| `prepare_data` | Prepare data: local CSV/Parquet, AKShare (A-share daily bars), or synthetic |
| `run_backtest` | Backtest in a subprocess; returns key metrics, first trades and structured errors |
| `get_run_details` | Page through trades, orders, daily equity and logs |

The guides are also exposed as the resource `akquant://guide/{topic}`.

## Limitations

- Only class-based strategies (`Strategy` subclasses) are supported in this release; for function-style strategies, call `run_backtest` from a script.
- `run_backtest`'s `config` exposes common options only; futures contract parameters, risk config, optimization and reports are out of scope.

## Working directory

Cached data and run results go to `<project root>/.akquant-mcp/` by default; override with the `AKQUANT_MCP_HOME` environment variable. Add it to `.gitignore`:

```text
.akquant-mcp/
```

## Security

This is a local development tool and **is not a sandbox**: `run_backtest` executes your strategy file in a subprocess with the same trust as running `python strategy.py` yourself. The project-root path restriction only keeps the AI from wandering into other directories; it is not isolation. Do not use it to run strategy files from untrusted sources.
