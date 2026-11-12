"""Logical operators in factor expressions: & | and or ~ (issue #446)."""

from datetime import date, timedelta
from typing import Any, List, cast

import polars as pl
import pytest
from akquant.factor import FactorEngine
from akquant.factor.parser import ExpressionParser

# 单标的 6 行：
#   close > open  -> T F T F T F
#   volume > 15   -> F T F F T T
#   open == 4     -> F F F T F F
_OPEN = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
_CLOSE = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]
_VOLUME = [10.0, 20.0, 5.0, 10.0, 50.0, 60.0]


def _frame() -> pl.DataFrame:
    n = len(_OPEN)
    return pl.DataFrame(
        {
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "symbol": ["A"] * n,
            "open": _OPEN,
            "close": _CLOSE,
            "volume": _VOLUME,
        }
    )


def _two_symbol_frame(rows: int = 30) -> pl.DataFrame:
    """Two symbols with different paths, long enough to fill a 20-row window."""
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(rows)]
    frames = []
    for symbol, phase in (("A", 0), ("B", 3)):
        close = [10.0 + ((i + phase) * 7 % 11) + i * 0.3 for i in range(rows)]
        open_ = [c - 0.5 + ((i + phase) % 3) * 0.5 for i, c in enumerate(close)]
        volume = [100.0 + ((i + phase) * 13 % 17) * 10 for i in range(rows)]
        frames.append(
            pl.DataFrame(
                {
                    "date": dates,
                    "symbol": [symbol] * rows,
                    "open": open_,
                    "high": [max(o, c) + 0.2 for o, c in zip(open_, close)],
                    "low": [min(o, c) - 0.2 for o, c in zip(open_, close)],
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pl.concat(frames)


def _run(expr: str, data: pl.DataFrame | None = None) -> List[Any]:
    # run_on_data 不读 catalog，与 issue 里的复现写法一致
    engine = FactorEngine(cast(Any, None))
    df = engine.run_on_data(_frame() if data is None else data, expr)
    return df.sort(["symbol", "date"])["factor_value"].to_list()


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("If((Close > Open) & (Volume > 15), 1, 0)", [0, 0, 0, 0, 1, 0]),
        ("If((Close > Open) | (Volume > 15), 1, 0)", [1, 1, 1, 0, 1, 1]),
        ("If(Close > Open and Volume > 15, 1, 0)", [0, 0, 0, 0, 1, 0]),
        ("If(Close > Open or Volume > 15, 1, 0)", [1, 1, 1, 0, 1, 1]),
        ("If(~(Close > Open), 1, 0)", [0, 1, 0, 1, 0, 1]),
        ("If(not (Close > Open), 1, 0)", [0, 1, 0, 1, 0, 1]),
        # 三个操作数
        ("If(Close > Open and Volume > 15 and Close > 5, 1, 0)", [0, 0, 0, 0, 1, 0]),
        ("If(Close > Open or Volume > 15 or Open == 4, 1, 0)", [1, 1, 1, 1, 1, 1]),
        (
            "If((Close > Open) | (Volume > 15) | (Open == 4), 1, 0)",
            [1, 1, 1, 1, 1, 1],
        ),
        # and 优先于 or，与 Python 一致
        ("If(Close > Open and Volume > 15 or Open == 4, 1, 0)", [0, 0, 0, 1, 1, 0]),
        ("If(Open == 4 or Close > Open and Volume > 15, 1, 0)", [0, 0, 0, 1, 1, 0]),
        # 取反与组合混用
        ("If(~(Close > Open) & (Volume > 15), 1, 0)", [0, 1, 0, 0, 0, 1]),
        ("If(not Close > Open and Volume > 15, 1, 0)", [0, 1, 0, 0, 0, 1]),
    ],
)
def test_logical_operators(expr: str, expected: List[int]) -> None:
    """Logical operators evaluate element-wise."""
    assert _run(expr) == expected


def test_null_follows_three_valued_logic() -> None:
    """Nulls from an unfilled window combine by three-valued logic."""
    # Ts_Mean(Volume, 3) 前两行为 null；之后 volume > 均值 -> F F T T
    assert _run("(Volume > Ts_Mean(Volume, 3)) & (Close > Open)") == [
        None,  # null & True
        False,  # null & False
        False,
        False,
        True,
        False,
    ]
    assert _run("(Volume > Ts_Mean(Volume, 3)) | (Close > Open)") == [
        True,  # null | True
        None,  # null | False
        True,
        False,
        True,
        True,
    ]
    # If 遇到 null 条件走“否”分支
    assert _run("If((Volume > Ts_Mean(Volume, 3)) & (Close > Open), 1, 0)") == [
        0,
        0,
        0,
        0,
        1,
        0,
    ]


@pytest.mark.parametrize(
    ("documented", "equivalent"),
    [
        # docs/zh/guide/factor.md §4.2
        (
            "If((Close > Open) & (Volume > Ts_Mean(Volume, 20)), 1, 0)",
            "If(Close > Open, If(Volume > Ts_Mean(Volume, 20), 1, 0), 0)",
        ),
        # docs/zh/textbook/14_factor.md §14.5.4
        (
            "If((Close == Ts_Max(Close, 20)) & (Volume < Ts_Mean(Volume, 20)), 1, 0)",
            "If(Close == Ts_Max(Close, 20), If(Volume < Ts_Mean(Volume, 20), 1, 0), 0)",
        ),
    ],
)
def test_documented_examples_run(documented: str, equivalent: str) -> None:
    """Documented examples run and match the nested-If spelling."""
    data = _two_symbol_frame()
    got = _run(documented, data)
    assert got == _run(equivalent, data)
    assert set(got) == {0, 1}


def test_logical_condition_inside_cross_partition_nesting() -> None:
    """A logical condition survives the CS-over-TS step split."""
    data = _two_symbol_frame()
    got = _run("Rank(If((Close > Open) & (Volume > Ts_Mean(Volume, 5)), 1, 0))", data)
    want = _run(
        "Rank(If(Close > Open, If(Volume > Ts_Mean(Volume, 5), 1, 0), 0))", data
    )
    assert got == want
    assert len(set(got)) > 1


def test_missing_parentheses_error_points_at_fix() -> None:
    """Missing parentheses around & operands produce an actionable error."""
    with pytest.raises(ValueError, match=r"parenthes") as excinfo:
        ExpressionParser().parse("If(Close > Open & Volume > 15, 1, 0)")
    assert "and" in str(excinfo.value)

    with pytest.raises(ValueError, match=r"parenthes"):
        ExpressionParser().parse("If(Close > Open | Volume > 15, 1, 0)")


def test_chained_comparison_error_suggests_split() -> None:
    """Real chained comparisons stay unsupported but suggest the split."""
    with pytest.raises(ValueError, match=r"Chained comparisons") as excinfo:
        ExpressionParser().parse("If(1 < Close < 5, 1, 0)")
    assert "&" in str(excinfo.value)
