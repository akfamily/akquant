"""run_grid_search 全部组合失败时直接报错, 不再返回全是 -999 的结果表."""

import sqlite3
from pathlib import Path
from typing import Any

import akquant
import numpy as np
import pandas as pd
import pytest
from akquant import Bar, IntParam, Strategy, run_grid_search


class FailingStrategy(Strategy):
    """on_bar 必然抛错的策略: 模拟策略/数据本身的问题."""

    n = IntParam(1, ge=1, le=100)

    def on_bar(self, bar: Bar) -> None:
        """Fail on every bar."""
        raise ValueError("boom: strategy bug")


class PartialStrategy(Strategy):
    """n == 1 时失败, 其余成功."""

    n = IntParam(1, ge=1, le=100)

    def on_bar(self, bar: Bar) -> None:
        """Fail only for n == 1."""
        if self.params.n == 1:
            raise ValueError("boom: n is one")


class LongRunFails(Strategy):
    """看到 8 根以上 bar 才失败: 训练窗口(10 根)失败, 样本外窗口(5 根)成功."""

    n = IntParam(1, ge=1, le=100)
    oos_runs: list = []

    def on_start(self) -> None:
        """Init counter."""
        self.count = 0

    def on_bar(self, bar: Bar) -> None:
        """Fail after 8 bars."""
        self.count += 1
        if self.count >= 8:
            raise ValueError("boom: too many bars")

    def on_stop(self) -> None:
        """Record runs that finished normally (only the short OOS runs)."""
        if self.count < 8:
            LongRunFails.oos_runs.append(self.params.n)


def _data(n: int = 5, symbol: str = "X") -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n, freq="D")
    close = 100 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
            "symbol": symbol,
        },
        index=idx,
    )


def _kwargs() -> dict[str, Any]:
    return {"data": _data(), "symbols": "X", "initial_cash": 100000.0}


def test_all_failed_raises_with_summary() -> None:
    """全部失败: 抛 RuntimeError, 带错误摘要、总组数与逃生口提示."""
    with pytest.raises(RuntimeError) as exc:
        run_grid_search(
            strategy=FailingStrategy,
            param_grid={"n": [1, 2, 3]},
            max_workers=1,
            **_kwargs(),
        )
    msg = str(exc.value)
    assert "3" in msg
    assert "boom: strategy bug" in msg
    assert "raise_on_all_failed=False" in msg
    assert "策略或数据本身" in msg


def test_all_failed_returns_table_when_opted_out() -> None:
    """raise_on_all_failed=False: 照旧返回结果表."""
    df = run_grid_search(
        strategy=FailingStrategy,
        param_grid={"n": [1, 2]},
        max_workers=1,
        raise_on_all_failed=False,
        **_kwargs(),
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert (df["sharpe_ratio"] == -999.0).all()


def test_partial_failure_does_not_raise() -> None:
    """部分失败: 不抛, 照常返回."""
    df = run_grid_search(
        strategy=PartialStrategy,
        param_grid={"n": [1, 2, 3]},
        max_workers=1,
        **_kwargs(),
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert (df["sharpe_ratio"] == -999.0).sum() == 1


def test_resume_with_db_success_then_all_new_failed(tmp_path: Path) -> None:
    """db_path 续跑: 库里有成功结果, 新组合全部失败时不抛."""
    db = str(tmp_path / "opt.db")
    first = run_grid_search(
        strategy=PartialStrategy,
        param_grid={"n": [2]},
        max_workers=1,
        db_path=db,
        **_kwargs(),
    )
    assert len(first) == 1
    # n=2 已在库里(成功), 本轮只新跑 n=1 且必然失败
    df = run_grid_search(
        strategy=PartialStrategy,
        param_grid={"n": [1, 2]},
        max_workers=1,
        db_path=db,
        **_kwargs(),
    )
    assert len(df) == 2
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM optimization_results").fetchone()
    assert rows[0] == 2


def test_all_failed_keeps_db_rows(tmp_path: Path) -> None:
    """全部失败抛错时, 已写进 db_path 的结果保留."""
    db = str(tmp_path / "opt.db")
    with pytest.raises(RuntimeError):
        run_grid_search(
            strategy=FailingStrategy,
            param_grid={"n": [1, 2]},
            max_workers=1,
            db_path=db,
            **_kwargs(),
        )
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM optimization_results").fetchone()
    assert rows[0] == 2


def test_walk_forward_skips_window_when_all_combinations_failed() -> None:
    """训练窗口全部失败: 该窗口被跳过, 不拿 -999 的参数去跑样本外."""
    LongRunFails.oos_runs = []
    results = akquant.run_walk_forward(
        strategy=LongRunFails,
        param_grid={"n": [1, 2]},
        data=_data(n=24),
        train_period=10,
        test_period=5,
        initial_cash=100_000.0,
        max_workers=1,
        show_progress=False,
        symbols="X",
    )
    assert isinstance(results, pd.DataFrame)
    assert results.empty
    assert LongRunFails.oos_runs == []


class DrawdownPartial(Strategy):
    """n == 1 失败, n == 2 成功(会买入持有, 有回撤指标)."""

    n = IntParam(1, ge=1, le=100)

    def on_bar(self, bar: Bar) -> None:
        """Fail for n == 1, buy once otherwise."""
        if self.params.n == 1:
            raise ValueError("boom: n is one")
        if self.get_position(bar.symbol) == 0:
            self.buy(bar.symbol, 100)

    def on_stop(self) -> None:
        """Record OOS-length successful runs."""
        DrawdownPartial.ok_runs.append(self.params.n)

    ok_runs: list = []


def test_failed_rows_sort_after_successes_under_ascending_metric() -> None:
    """max_drawdown_pct 升序: 失败组合不能因取不到指标而排到最前."""
    df = run_grid_search(
        strategy=DrawdownPartial,
        param_grid={"n": [1, 2]},
        max_workers=1,
        sort_by="max_drawdown_pct",
        ascending=True,
        **_kwargs(),
    )
    assert isinstance(df, pd.DataFrame)
    assert df["n"].tolist() == [2, 1]
    multi = run_grid_search(
        strategy=DrawdownPartial,
        param_grid={"n": [1, 2]},
        max_workers=1,
        sort_by=["max_drawdown_pct", "sharpe_ratio"],
        ascending=[True, True],
        **_kwargs(),
    )
    assert isinstance(multi, pd.DataFrame)
    assert multi["n"].tolist() == [2, 1]


def test_walk_forward_never_uses_failed_params_for_oos() -> None:
    """部分失败 + 升序指标: 样本外只用成功的 n=2."""
    DrawdownPartial.ok_runs = []
    results = akquant.run_walk_forward(
        strategy=DrawdownPartial,
        param_grid={"n": [1, 2]},
        data=_data(n=24),
        train_period=10,
        test_period=5,
        metric="max_drawdown_pct",
        ascending=True,
        initial_cash=100_000.0,
        max_workers=1,
        show_progress=False,
        symbols="X",
    )
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert set(results["n"]) == {2}


class MetaCall(type(Strategy)):  # type: ignore[misc]
    """Custom metaclass counting __call__ invocations."""

    calls = 0

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Count and delegate."""
        MetaCall.calls += 1
        return super().__call__(*args, **kwargs)


def test_instantiate_strategy_keeps_user_metaclass_call() -> None:
    """用户元类的 __call__ 仍被调用一次."""
    from akquant.strategy import instantiate_strategy

    class MS(Strategy, metaclass=MetaCall):
        n = IntParam(1)

    MetaCall.calls = 0
    inst = instantiate_strategy(MS, {"n": 4})
    assert MetaCall.calls == 1
    assert inst.params.n == 4
