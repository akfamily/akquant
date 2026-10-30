"""akquant.mcp.service.runs: 子进程回测、超时、清理与分页读取.

每次 run 启动一个 Python 子进程(冷启动约 1.5s), 本文件整体约 30~60s。
"""

from pathlib import Path
from typing import Callable

import pytest
from akquant.mcp.service import ServiceError
from akquant.mcp.service.data import prepare_data
from akquant.mcp.service.paths import ServiceConfig, make_config
from akquant.mcp.service.runs import get_run_details, run_backtest
from akquant.mcp.worker import METRIC_KEYS

WriteFile = Callable[[str, str], Path]

BUYER = """\
from akquant import IntParam, Strategy


class Buyer(Strategy):
    n = IntParam(3)

    def on_start(self) -> None:
        self.count = 0

    def on_bar(self, bar):
        self.count += 1
        if self.count == self.params.n:
            self.buy(symbol=bar.symbol, quantity=100)
        if self.count == self.params.n + 5:
            self.sell(symbol=bar.symbol, quantity=100)
"""

RAISER = (
    "from akquant import Strategy\n\n\n"
    "class S(Strategy):\n"
    "    def on_bar(self, bar):\n"
    "        raise ValueError('{tag}')\n"
)


def _dataset(cfg: ServiceConfig) -> str:
    out = prepare_data(cfg, {"source": "synthetic", "end": "2023-06-30"})
    return str(out["dataset_id"])


def test_ok_run_and_details(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """正常回测: 指标齐全, 明细可分页读取, 时间带时区."""
    write_file("buyer.py", BUYER)
    out = run_backtest(cfg, "buyer.py", _dataset(cfg), config={"initial_cash": 1e6})
    assert out["status"] == "ok", out
    assert set(out["metrics"]) == set(METRIC_KEYS)
    assert out["metrics"]["closed_trade_count"] == 1
    assert len(out["first_trades"]) == 1
    run_dir = cfg.runs_dir / out["run_id"]
    assert (run_dir / "summary.json").is_file()
    trades = get_run_details(cfg, out["run_id"], "trades")
    assert trades["total"] == 1 and trades["rows"][0]["symbol"] == "SYN001"
    equity = get_run_details(cfg, out["run_id"], "equity", limit=2)
    assert equity["total"] > 2 and len(equity["rows"]) == 2
    assert "+08:00" in equity["rows"][0]["timestamp"]
    beyond = get_run_details(cfg, out["run_id"], "orders", offset=10_000)
    assert beyond["rows"] == [] and beyond["total"] == 2
    logs = get_run_details(cfg, out["run_id"], "logs")
    assert logs["rows"] and logs["rows"][0]["line"] == 1


def test_strategy_params_and_attr(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """strategy_params 生效; 两个类时 strategy_attr 选中其一."""
    extra = "\n\nclass Other(Buyer):\n    pass\n"
    write_file("two.py", BUYER + extra)
    out = run_backtest(
        cfg,
        "two.py",
        _dataset(cfg),
        strategy_params={"n": 2},
        strategy_attr="Buyer",
    )
    assert out["status"] == "ok", out
    first = out["first_trades"][0]["entry_time"]
    assert first.startswith("2023-01-0")


def test_error_run_maps_user_frame(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """on_bar 抛错: status=error, phase=run, 行号指向 raise."""
    write_file("bad.py", RAISER.format(tag="boom"))
    out = run_backtest(cfg, "bad.py", _dataset(cfg))
    assert out["status"] == "error"
    assert out["error"]["phase"] == "run"
    assert out["error"]["user_frames"][-1]["line"] == 6
    assert any("boom" in line for line in out["log_tail"])


def test_rerun_after_edit_sees_new_code(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """同一路径改写后重跑, 拿到的是新代码(子进程不受模块缓存影响)."""
    dataset = _dataset(cfg)
    write_file("edit.py", RAISER.format(tag="version-A"))
    first = run_backtest(cfg, "edit.py", dataset)
    assert "version-A" in first["error"]["message"]
    write_file("edit.py", RAISER.format(tag="version-B"))
    second = run_backtest(cfg, "edit.py", dataset)
    assert "version-B" in second["error"]["message"]


def test_timeout(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """死循环策略超时被终止."""
    src = (
        "from akquant import Strategy\n\n\n"
        "class S(Strategy):\n"
        "    def on_bar(self, bar):\n"
        "        while True:\n"
        "            pass\n"
    )
    write_file("loop.py", src)
    out = run_backtest(cfg, "loop.py", _dataset(cfg), timeout_s=3)
    assert out["status"] == "timeout"
    assert out["elapsed_s"] < 30
    assert isinstance(out["log_tail"], list)
    assert out["message"]


def test_worker_crash(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """子进程硬退出没写 summary: status=error, phase=engine."""
    src = (
        "import os\n\nfrom akquant import Strategy\n\n\n"
        "class S(Strategy):\n"
        "    def on_bar(self, bar):\n"
        "        os._exit(3)\n"
    )
    write_file("crash.py", src)
    out = run_backtest(cfg, "crash.py", _dataset(cfg))
    assert out["status"] == "error"
    assert out["error"]["phase"] == "engine"
    assert out["error"]["type"] == "WorkerCrashed"
    assert "exit code 3" in out["error"]["message"]


def test_max_runs_prunes_oldest(tmp_path: Path, write_file: WriteFile) -> None:
    """max_runs=2 时第三次 run 前删除最旧的."""
    cfg = make_config(root=tmp_path, home=tmp_path / ".h", max_runs=2)
    write_file("syn.py", "class S(\n")
    dataset = _dataset(cfg)
    ids = [run_backtest(cfg, "syn.py", dataset)["run_id"] for _ in range(3)]
    left = sorted(p.name for p in cfg.runs_dir.iterdir())
    assert left == sorted(ids[1:])
    with pytest.raises(ServiceError, match="找不到"):
        get_run_details(cfg, ids[0], "logs")


def test_argument_validation(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """非法 config 字段、timeout、路径、id、section 都报 ServiceError."""
    write_file("buyer.py", BUYER)
    dataset = _dataset(cfg)
    with pytest.raises(ServiceError, match="config"):
        run_backtest(cfg, "buyer.py", dataset, config={"risk_config": {}})
    with pytest.raises(ServiceError, match="timeout_s"):
        run_backtest(cfg, "buyer.py", dataset, timeout_s=0)
    with pytest.raises(ServiceError):
        run_backtest(cfg, "../buyer.py", dataset)
    with pytest.raises(ServiceError, match="prepare_data"):
        run_backtest(cfg, "buyer.py", "0123456789ab")
    with pytest.raises(ServiceError):
        get_run_details(cfg, "../x", "logs")
    with pytest.raises(ServiceError):
        get_run_details(cfg, "20260101-000000-abcdef", "positions")


def _simple(body: str, head: str = "") -> str:
    return (
        f"{head}from akquant import Strategy\n\n\n"
        "class S(Strategy):\n"
        "    def on_bar(self, bar):\n"
        f"{body}"
    )


def test_root_stdlib_shadow_does_not_break_worker(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """项目根下的 random.py / akquant.py 不会遮蔽标准库或 akquant."""
    write_file("random.py", "x = 1\n")
    write_file("akquant.py", "y = 1\n")
    write_file("buyer.py", BUYER)
    out = run_backtest(cfg, "buyer.py", _dataset(cfg))
    assert out["status"] == "ok", out


def test_sibling_import_works(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """策略可导入同目录的辅助模块."""
    write_file("sub/helpers.py", "N = 100\n")
    src = _simple(
        "        if not getattr(self, 'done', False):\n"
        "            self.done = True\n"
        "            self.buy(symbol=bar.symbol, quantity=N)\n",
        head="from helpers import N\n",
    )
    write_file("sub/strat.py", src)
    out = run_backtest(cfg, "sub/strat.py", _dataset(cfg))
    assert out["status"] == "ok", out
    assert out["metrics"] is not None


def test_relative_path_resolves_against_root(
    tmp_path: Path, cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """策略里的相对路径按项目根解析."""
    src = (
        "from akquant import Strategy\n\n\n"
        "class S(Strategy):\n"
        "    def on_start(self):\n"
        "        open('marker.txt', 'w').close()\n\n"
        "    def on_bar(self, bar):\n"
        "        pass\n"
    )
    write_file("rel.py", src)
    out = run_backtest(cfg, "rel.py", _dataset(cfg))
    assert out["status"] == "ok", out
    assert (tmp_path / "marker.txt").exists()


def test_deprecation_warning_is_visible(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """DeprecationWarning 不被白名单吞掉: 进 warnings 也进日志."""
    src = _simple(
        "        if not getattr(self, 'done', False):\n"
        "            self.done = True\n"
        "            warnings.warn('legacy usage is deprecated', DeprecationWarning)\n",
        head="import warnings\n",
    )
    write_file("dep.py", src)
    out = run_backtest(cfg, "dep.py", _dataset(cfg))
    assert out["status"] == "ok", out
    assert any(w.startswith("DeprecationWarning") for w in out["warnings"])
    logs = get_run_details(cfg, out["run_id"], "logs", limit=1000)
    assert any("DeprecationWarning" in r["text"] for r in logs["rows"])


def test_warnings_capped_at_twenty(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """50 条不同的警告只保留前 20 条."""
    src = _simple(
        "        if not getattr(self, 'done', False):\n"
        "            self.done = True\n"
        "            for i in range(50):\n"
        "                warnings.warn(f'w{i}')\n",
        head="import warnings\n",
    )
    write_file("many.py", src)
    out = run_backtest(cfg, "many.py", _dataset(cfg))
    assert out["status"] == "ok", out
    assert len(out["warnings"]) == 20


def test_strategy_print_stays_out_of_parent_stdout(
    cfg: ServiceConfig, write_file: WriteFile, capfd: pytest.CaptureFixture[str]
) -> None:
    """策略 print 只进 output.log, 不污染父进程 stdout(MCP 协议通道)."""
    write_file("pr.py", _simple("        print('MARKER-xyz')\n"))
    out = run_backtest(cfg, "pr.py", _dataset(cfg))
    assert out["status"] == "ok", out
    assert "MARKER-xyz" not in capfd.readouterr().out
    logs = get_run_details(cfg, out["run_id"], "logs", limit=1000)
    assert any("MARKER-xyz" in r["text"] for r in logs["rows"])


def test_crash_message_includes_log_tail(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """崩溃信息带上 output.log 的最后一行."""
    src = _simple(
        "        print('LAST-LINE-q', flush=True)\n"
        "        import os\n"
        "        os._exit(4)\n"
    )
    write_file("c2.py", src)
    out = run_backtest(cfg, "c2.py", _dataset(cfg))
    assert out["error"]["type"] == "WorkerCrashed"
    assert "LAST-LINE-q" in out["error"]["message"]


def test_run_id_trailing_newline_rejected(cfg: ServiceConfig) -> None:
    """run_id 末尾带换行也不放行."""
    with pytest.raises(ServiceError, match="格式"):
        get_run_details(cfg, "20260101-000000-abcdef\n", "logs")


def test_sibling_file_base_runs_with_strategy_attr(
    cfg: ServiceConfig, write_file: WriteFile
) -> None:
    """基类在同目录另一文件里的策略, 指定 strategy_attr 可回测."""
    write_file(
        "base.py",
        "from akquant import Strategy\n\n\nclass MyBase(Strategy):\n"
        "    def go(self, bar):\n"
        "        if not getattr(self, 'done', False):\n"
        "            self.done = True\n"
        "            self.buy(symbol=bar.symbol, quantity=100)\n",
    )
    write_file(
        "s.py",
        "from base import MyBase\n\n\nclass S(MyBase):\n"
        "    def on_bar(self, bar):\n        self.go(bar)\n",
    )
    out = run_backtest(cfg, "s.py", _dataset(cfg), strategy_attr="S")
    assert out["status"] == "ok", out


def test_long_log_line_truncated(cfg: ServiceConfig, write_file: WriteFile) -> None:
    """超长日志行在 log_tail 里被截断."""
    src = _simple(
        "        if not getattr(self, 'p', False):\n"
        "            self.p = True\n"
        "            print('x' * 5000)\n"
    )
    write_file("long.py", src)
    out = run_backtest(cfg, "long.py", _dataset(cfg))
    assert out["status"] == "ok", out
    assert any(len(t) > 100 for t in out["log_tail"])
    assert all(len(t) <= 501 for t in out["log_tail"])
