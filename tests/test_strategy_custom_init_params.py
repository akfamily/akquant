"""自定义 __init__ 不接收 **kwargs 时也能拿到策略参数."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest, run_grid_search
from akquant.params import IntParam


def _data() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=30)
    close = 100 + np.arange(30, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
            "symbol": "X",
        },
        index=idx,
    )


class NoKwargsInit(Strategy):
    """__init__ 只收 self, 参数走类体声明."""

    n = IntParam(3, ge=1, le=50)
    seen: list = []

    def __init__(self) -> None:
        """Set custom state."""
        super().__init__()
        self.state = "ready"

    def on_start(self) -> None:
        """Subscribe."""
        self.subscribe("X")

    def on_bar(self, bar: Bar) -> None:
        """Record."""
        NoKwargsInit.seen.append((self.params.n, self.state))


class HalfMigrated(Strategy):
    """迁移了一半: __init__(self, fast=3, **kwargs)."""

    fast = IntParam(3, ge=1, le=50)
    got: list = []

    def __init__(self, fast: int = 3, **kwargs: Any) -> None:
        """Capture fast."""
        super().__init__(**kwargs)
        HalfMigrated.got.append(fast)

    def on_start(self) -> None:
        """Subscribe."""
        self.subscribe("X")

    def on_bar(self, bar: Bar) -> None:
        """Noop."""


class RequiredArg(Strategy):
    """__init__ 带必填的非参数位置参数."""

    n = IntParam(3, ge=1, le=50)

    def __init__(self, required_arg: int) -> None:
        """Need required_arg."""
        super().__init__()

    def on_bar(self, bar: Bar) -> None:
        """Noop."""


def test_init_without_kwargs_receives_params_in_backtest() -> None:
    """__init__(self) + strategy_params 跑通, params 生效, 自设状态保留."""
    NoKwargsInit.seen = []
    run_backtest(
        data=_data(),
        strategy=NoKwargsInit,
        symbols="X",
        initial_cash=100000.0,
        strategy_params={"n": 7},
    )
    assert NoKwargsInit.seen
    assert set(NoKwargsInit.seen) == {(7, "ready")}


def test_init_without_kwargs_grid_search() -> None:
    """__init__(self) 下 run_grid_search 每组都成功, 两组 params 不同."""
    df = run_grid_search(
        strategy=NoKwargsInit,
        param_grid={"n": [2, 5]},
        data=_data(),
        symbols="X",
        initial_cash=100000.0,
        max_workers=1,
    )
    assert isinstance(df, pd.DataFrame)
    if "error" in df.columns:
        assert df["error"].isna().all() or (df["error"].astype(str) == "").all()
    assert len(df) == 2
    assert sorted(df["n"].tolist()) == [2, 5]


def test_half_migrated_init_still_gets_fast() -> None:
    """__init__(self, fast=3, **kwargs) 仍收到传入的 fast."""
    HalfMigrated.got = []
    run_backtest(
        data=_data(),
        strategy=HalfMigrated,
        symbols="X",
        initial_cash=100000.0,
        strategy_params={"fast": 9},
    )
    assert HalfMigrated.got == [9]


def test_required_non_param_arg_still_fails() -> None:
    """必填的非参数位置参数仍然报错, 消息含 missing."""
    with pytest.raises(TypeError, match="missing"):
        run_backtest(
            data=_data(),
            strategy=RequiredArg,
            symbols="X",
            initial_cash=100000.0,
            strategy_params={"n": 5},
        )


def test_non_strict_params_now_take_effect() -> None:
    """strict=False 且 __init__(self): 参数不再回退成默认值."""
    NoKwargsInit.seen = []
    run_backtest(
        data=_data(),
        strategy=NoKwargsInit,
        symbols="X",
        initial_cash=100000.0,
        strategy_params={"n": 11},
        strict_strategy_params=False,
    )
    assert NoKwargsInit.seen
    assert {n for n, _ in NoKwargsInit.seen} == {11}


def test_direct_construction_keeps_python_semantics() -> None:
    """直接 NoKwargsInit(n=5) 仍按 Python 语义报错."""
    with pytest.raises(TypeError, match="unexpected keyword"):
        NoKwargsInit(n=5)  # type: ignore[call-arg]
