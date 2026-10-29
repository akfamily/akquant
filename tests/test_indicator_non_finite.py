"""内置流式指标的非有限值规则(设计 C2).

任一输入为 NaN/±inf 时 update 返回 None, 且不改变任何状态: 对一个实例喂含坏行
的数据, 与对新实例只喂有效行, 两者在有效行上的输出必须逐位相同; 坏行之后的
value 必须等于坏行之前的 value。
"""

import math
from typing import Any

import akquant.akquant as ext
import akquant.talib as ta
import numpy as np
import pytest

#: 构造参数名 -> 取值; 新增指标类用了新参数名时这里要补, 否则测试报错提醒
_CTOR_VALUES: dict[str, Any] = {
    "period": 5,
    "fast_period": 3,
    "slow_period": 6,
    "signal_period": 3,
    "period1": 3,
    "period2": 5,
    "period3": 7,
    "fastk_period": 5,
    "slowk_period": 3,
    "slowd_period": 3,
    "multiplier": 2.0,
    "nbdev": 1.0,
    "c": 0.015,
    "vfactor": 0.7,
    "acceleration": 0.02,
    "maximum": 0.2,
    "fast_limit": 0.5,
    "slow_limit": 0.05,
}

_NOT_INDICATORS = {"WindowAggregator"}


def _indicator_classes() -> list[str]:
    return sorted(
        name
        for name in dir(ext)
        if isinstance(getattr(ext, name), type)
        and hasattr(getattr(ext, name), "update")
        and name not in _NOT_INDICATORS
    )


def _params(text_signature: str | None) -> list[str]:
    inner = (text_signature or "()").strip()[1:-1]
    return [
        part.strip()
        for part in inner.split(",")
        if part.strip() and part.strip() != "$self"
    ]


def _columns(length: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, length))
    other = close * 1.01 + np.sin(np.arange(length))
    return {
        "open": close + rng.normal(0.0, 0.2, length),
        "high": close + 0.6,
        "low": close - 0.6,
        "close": close,
        "value": close,
        "x": close,
        "left": close,
        "y": other,
        "right": other,
        "volume": rng.uniform(1e5, 2e5, length),
        "min_value": close - 1.0,
        "max_value": close + 1.0,
    }


def _flatten(out: object) -> tuple[float, ...] | None:
    if out is None:
        return None
    if isinstance(out, tuple):
        return tuple(float(item) for item in out)
    return (float(out),)  # type: ignore[arg-type]


def _same(left: tuple[float, ...] | None, right: tuple[float, ...] | None) -> bool:
    """逐位相同, NaN 与 NaN 视为相同(ACOS 等在定义域外本来就输出 NaN)."""
    if left is None or right is None:
        return left is right
    return len(left) == len(right) and all(
        a == b or (math.isnan(a) and math.isnan(b)) for a, b in zip(left, right)
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("name", _indicator_classes())
def test_non_finite_row_is_skipped_without_touching_state(
    name: str, bad: float
) -> None:
    """含 NaN/±inf 的行返回 None 且不留痕, 之后的输出与从没见过坏行一致."""
    cls = getattr(ext, name)
    ctor = [_CTOR_VALUES[param] for param in _params(cls.__text_signature__)]
    update_params = _params(cls.update.__text_signature__)
    length = 80
    columns = _columns(length)
    bad_rows = {30, 31, 32, 55}
    # 按坏行的序号轮流选列: 4 列输入的第 0~3 列各坏一次, 1~3 列输入则循环使用
    bad_column = {row: k for k, row in enumerate(sorted(bad_rows))}
    rows = []
    for i in range(length):
        row = [float(columns[param][i]) for param in update_params]
        if i in bad_rows:
            row[bad_column[i] % len(row)] = bad  # 每次只坏一列
        rows.append(row)

    dirty = cls(*ctor)
    clean = cls(*ctor)
    for i, row in enumerate(rows):
        before = _flatten(getattr(dirty, "value", None))
        out = dirty.update(*row)
        if i in bad_rows:
            assert out is None, f"{name}: 第 {i} 根含 {bad} 却出了值 {out}"
            assert _same(_flatten(getattr(dirty, "value", None)), before), (
                f"{name}: 第 {i} 根含 {bad} 改变了 value"
            )
        else:
            assert _same(_flatten(out), _flatten(clean.update(*row))), (
                f"{name}: 第 {i} 根的输出被前面的坏行污染"
            )


def test_every_constructor_parameter_has_a_value() -> None:
    """新指标类带来新构造参数名时, 这里要提醒补 ``_CTOR_VALUES``."""
    missing = {
        param
        for name in _indicator_classes()
        for param in _params(getattr(ext, name).__text_signature__)
        if param not in _CTOR_VALUES
    }
    assert not missing, f"_CTOR_VALUES 缺少构造参数: {sorted(missing)}"


def test_clip_with_inverted_bounds_does_not_panic() -> None:
    """下界大于上界时与 numpy/python 后端一致: 先交换上下界, 不 panic."""
    clip = ext.CLIP()
    assert clip.update(5.0, 3.0, 1.0) == 3.0
    assert clip.update(0.0, 3.0, 1.0) == 1.0


@pytest.mark.parametrize("name", sorted(ta.__all__))
def test_every_talib_function_skips_non_finite_rows(name: str) -> None:
    """``akquant.talib`` 的每个公开函数都带 ``skip_non_finite_rows`` 装饰器.

    装饰器经 ``functools.wraps`` 留下 ``__wrapped__``; 新增函数忘了装饰就会在这里失败。
    """
    assert name.isupper(), f"{name}: __all__ 里出现了非大写名字, 请确认是否该装饰"
    assert hasattr(getattr(ta, name), "__wrapped__"), (
        f"{name} 没有 @skip_non_finite_rows()"
    )


@pytest.mark.parametrize(
    ("name", "args", "nargs"),
    [
        ("BollingerBands", (0, 2.0), 1),
        ("STDDEV", (0, 1.0), 1),
        ("VAR", (0, 1.0), 1),
        ("CORREL", (0,), 2),
        ("MAXINDEX", (0,), 1),
        ("MININDEX", (0,), 1),
        ("MINMAXINDEX", (0,), 1),
    ],
)
def test_zero_period_does_not_panic(
    name: str, args: tuple[Any, ...], nargs: int
) -> None:
    """周期为 0 时构造可以, update 返回 None, 不 panic."""
    try:
        indicator = getattr(ext, name)(*args)
    except ValueError:
        return
    for k in range(5):
        assert indicator.update(*([100.0 + k] * nargs)) is None
