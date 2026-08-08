import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import akquant as aq
import pandas as pd
import pytest


def _load_example_module(module_name: str, relative_path: str) -> ModuleType:
    """Load an example module from the repository by relative path."""
    root = Path(__file__).resolve().parents[1]
    module_path = root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load example module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def test_option_example_import_has_no_run_backtest_side_effect(
    monkeypatch: Any,
) -> None:
    """Importing the option example should not execute the backtest."""

    def _unexpected_run_backtest(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("run_backtest should not execute during import")

    monkeypatch.setattr(aq, "run_backtest", _unexpected_run_backtest)
    module = _load_example_module(
        "example_option_test_import",
        "examples/07_option_test.py",
    )
    assert hasattr(module, "build_data")
    assert hasattr(module, "build_config")
    assert hasattr(module, "main")


def test_option_example_builders_match_current_run_backtest_api() -> None:
    """Option example helpers should remain executable with the current API."""
    module = _load_example_module(
        "example_option_test_runtime",
        "examples/07_option_test.py",
    )
    data = module.build_data()
    config = module.build_config()
    assert set(data) == {"CALL_OPT", "UL"}
    assert all(isinstance(frame, pd.DataFrame) for frame in data.values())
    result = aq.run_backtest(
        data=data,
        strategy=module.OptionExpiryStrategy,
        config=config,
        commission_rate=0.0,
        show_progress=False,
    )
    assert result.metrics.end_market_value == 99900.0


def test_option_example_main_uses_keyword_arguments(monkeypatch: Any) -> None:
    """Option example main should call run_backtest with supported keywords."""
    module = _load_example_module(
        "example_option_test_main",
        "examples/07_option_test.py",
    )
    captured: dict[str, Any] = {}

    def _fake_run_backtest(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(
            orders=[],
            metrics=SimpleNamespace(end_market_value=99900.0),
            trades_df=pd.DataFrame(),
        )

    monkeypatch.setattr(module, "run_backtest", _fake_run_backtest)
    module.main()

    assert captured["strategy"] is module.OptionExpiryStrategy
    assert captured["commission_rate"] == 0.0
    assert captured["show_progress"] is False
    assert set(captured["data"]) == {"CALL_OPT", "UL"}


def test_textbook_futures_strategy_uses_short_for_bearish_signal(
    monkeypatch: Any,
) -> None:
    """Textbook futures strategy should open bearish positions via short()."""
    module = _load_example_module(
        "example_textbook_futures_strategy",
        "examples/textbook/ch07_futures.py",
    )
    strategy = module.FuturesTrendStrategy()
    captured: dict[str, Any] = {"short": None, "buy": None}

    monkeypatch.setattr(
        strategy,
        "get_history",
        lambda **_kwargs: pd.Series([100.0] * strategy.ma_window + [90.0]),
    )
    monkeypatch.setattr(strategy, "get_position", lambda _symbol: 0.0)
    monkeypatch.setattr(strategy, "log", lambda _message: None)
    monkeypatch.setattr(
        strategy,
        "short",
        lambda symbol, quantity: captured.__setitem__("short", (symbol, quantity)),
    )
    monkeypatch.setattr(
        strategy,
        "buy",
        lambda symbol, quantity: captured.__setitem__("buy", (symbol, quantity)),
    )

    bar = aq.Bar(
        timestamp=pd.Timestamp("2023-01-01 09:30:00", tz="UTC").value,
        open=90.0,
        high=90.0,
        low=90.0,
        close=90.0,
        volume=1000.0,
        symbol="RB2310",
    )
    strategy.on_bar(bar)

    assert captured["short"] == ("RB2310", 1)
    assert captured["buy"] is None


def test_textbook_dual_ma_examples_request_enough_warmup_bars() -> None:
    """Dual-MA textbook examples should align warmup with N+1 history windows."""
    ch05 = _load_example_module(
        "example_textbook_ch05_strategy",
        "examples/textbook/ch05_strategy.py",
    )
    ch10 = _load_example_module(
        "example_textbook_ch10_analysis",
        "examples/textbook/ch10_analysis.py",
    )

    ch05_strategy = ch05.MyFirstStrategy(short_window=5, long_window=20)
    ch10_strategy = ch10.AnalysisStrategy(short_window=5, long_window=20)

    assert ch05_strategy.warmup_period == 21
    assert ch10_strategy.warmup_period == 21


def test_textbook_futures_example_documents_fill_policy_and_bps_slippage() -> None:
    """Textbook futures example should retain safer fill/slippage configuration."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "examples" / "textbook" / "ch07_futures.py").read_text(
        encoding="utf-8"
    )
    assert "fill_policy=aq.CurrentClose()" in source
    assert 'slippage={"type": "percent", "value": 0.0002}' in source


def test_functional_warm_start_example_main_runs(capsys: Any) -> None:
    """Functional warm start example should run and print restored state markers."""
    module = _load_example_module(
        "example_functional_warm_start_demo",
        "examples/56_functional_warm_start_demo.py",
    )

    module.main()
    output = capsys.readouterr().out

    assert "phase1_events=start:restored=0|bar:10.00|bar:10.40" in output
    assert "resume:bars=2:starts=1" in output
    assert "processed_closes=10.00,10.40,10.80,11.20" in output
    assert "resume_count=1" in output
    assert "done_functional_warm_start_demo" in output


def test_functional_multi_slot_warm_start_example_main_runs(capsys: Any) -> None:
    """Functional multi-slot warm start example should restore both slot states."""
    module = _load_example_module(
        "example_functional_multi_slot_warm_start_demo",
        "examples/57_functional_multi_slot_warm_start_demo.py",
    )

    module.main()
    output = capsys.readouterr().out

    assert "alpha_events=alpha:start:restored=0" in output
    assert "beta_events=beta:start:restored=0" in output
    assert "alpha:resume:bars=2:starts=1" in output
    assert "beta:resume:bars=2:starts=1" in output
    assert "alpha_resume_count=1" in output
    assert "beta_resume_count=1" in output
    assert "slot_ids=['beta']" in output
    assert "done_functional_multi_slot_warm_start_demo" in output


def test_ch07_futures_example_reports_real_equity_and_flips() -> None:
    """ch07 期货示例须真正产出趋势反转，且期末权益不是兜底的 0.

    两个历史缺陷：``warmup_period`` 比 ``get_history(count=ma_window + 1)`` 少一根，
    均线恒为 NaN 导致信号永远锁在初值（示例从不走反手分支）；``metrics_df`` 里
    没有 ``end_portfolio_value`` 键，用错键会静默打印 0.00。
    """
    module = _load_example_module(
        "example_ch07_futures_metrics",
        "examples/textbook/ch07_futures.py",
    )

    df = module.generate_futures_data()
    result = aq.run_backtest(
        strategy=module.FuturesTrendStrategy,
        data=df,
        initial_cash=500_000.0,
        fill_policy=aq.CurrentClose(),
    )

    # warmup 足够 -> 均线有真值 -> 至少发生一次开仓
    assert len(result.orders_df) > 0

    metrics = result.metrics_df
    assert "end_portfolio_value" not in metrics.index
    assert "end_market_value" in metrics.index

    end_value = float(str(metrics.loc["end_market_value", "value"]))
    assert end_value > 0.0
    assert end_value == pytest.approx(float(result.equity_curve.iloc[-1]))
