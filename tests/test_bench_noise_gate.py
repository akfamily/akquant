# -*- coding: utf-8 -*-
"""噪声闸门判定逻辑的单元测试(不跑回测).

闸门存在的理由:4500 标的负载的轮次离散度实测为 13.7%,大于 PR #352 声称的
+11.9~14.4%. 效应量小于噪声带时必须拒绝声称收益.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_engine_perbar.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("bench_engine_perbar", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # 必须先注册到 sys.modules: 脚本用了 `from __future__ import annotations`,
    # dataclasses 在 Python 3.12 上解析字符串注解时会反查 sys.modules[cls.__module__],
    # 不注册会在 @dataclass 装饰 RunStats 时抛 AttributeError.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_stats_uses_min_not_median() -> None:
    """bars/sec 必须取 min(min 对性能测量比 median 更稳)."""
    m = _load_module()
    stats = m.RunStats(profile="wide", bars=1000, rounds=[2.0, 4.0, 8.0])
    assert stats.min == 2.0
    assert stats.median == 4.0
    assert stats.max == 8.0
    assert stats.spread == 3.0  # 8/2 - 1
    assert stats.bars_per_sec == 500.0  # 1000 / min


def test_effect_inside_noise_band_is_inconclusive() -> None:
    """效应 12% 但噪声带 14% -> 必须 INCONCLUSIVE(这正是 #352 的情形)."""
    m = _load_module()
    base = m.RunStats(profile="wide", bars=1000, rounds=[1.000, 1.070, 1.140])
    cand = m.RunStats(profile="wide", bars=1000, rounds=[0.893, 0.950, 1.010])
    verdict, effect, noise = m.verdict(base, cand)
    assert verdict == "INCONCLUSIVE"
    assert effect < noise


def test_effect_above_noise_band_is_improved() -> None:
    """效应 60% 远超噪声带 2% -> IMPROVED."""
    m = _load_module()
    base = m.RunStats(profile="wide", bars=1000, rounds=[1.000, 1.010, 1.020])
    cand = m.RunStats(profile="wide", bars=1000, rounds=[0.625, 0.630, 0.635])
    verdict, effect, noise = m.verdict(base, cand)
    assert verdict == "IMPROVED"
    assert effect > noise


def test_regression_above_noise_band_is_regressed() -> None:
    """候选变慢且超噪声带 -> REGRESSED."""
    m = _load_module()
    base = m.RunStats(profile="wide", bars=1000, rounds=[0.625, 0.630, 0.635])
    cand = m.RunStats(profile="wide", bars=1000, rounds=[1.000, 1.010, 1.020])
    verdict, effect, noise = m.verdict(base, cand)
    assert verdict == "REGRESSED"
    assert effect < 0


def test_noise_band_must_use_base_spread_not_only_candidate() -> None:
    """噪声带必须是 base 与 cand 离散度的最大值, 不能只信候选或取两者较小值.

    这条测试专门用来抓这样的回归:有人"简化"实现, 认为新跑的候选轮次
    更可信, 于是把噪声带只算 cand.spread(或退化成 min(base, cand)),
    悄悄丢掉了 base.spread 这一侧. 前四个已有测试无法抓住这个回归 ——
    在它们的数值下, 只用 cand.spread、用 min(base, cand)、用 max(base,
    cand) 三种算法给出的判定完全相同, 回归可以从这些测试里"蒙混过关".

    构造: base 离散度故意设得很大(0.30), cand 离散度故意设得很小
    (0.05), 效应量卡在两者中间(0.15). 手算:
        cand: rounds=[1.000, 1.025, 1.050]
              min=1.000, max=1.050, spread = 1.050/1.000 - 1 = 0.05
        base: rounds=[1.150, 1.300, 1.495]
              min=1.150, max=1.495, spread = 1.495/1.150 - 1 = 0.30
        effect = base.min/cand.min - 1 = 1.150/1.000 - 1 = 0.15

    - 正确算法 max(base.spread, cand.spread) = 0.30 > 0.15 -> INCONCLUSIVE.
    - 错误算法"只用 cand.spread" = 0.05 < 0.15 -> 会误判为 IMPROVED.
    - 错误算法 min(base.spread, cand.spread) = 0.05, 同上, 也会误判为
      IMPROVED.
    因此本测试断言 INCONCLUSIVE 就足以让上述两种回归实现失败.
    """
    m = _load_module()
    base = m.RunStats(profile="wide", bars=1000, rounds=[1.150, 1.300, 1.495])
    cand = m.RunStats(profile="wide", bars=1000, rounds=[1.000, 1.025, 1.050])
    assert abs(base.spread - 0.30) < 1e-9
    assert abs(cand.spread - 0.05) < 1e-9
    verdict, effect, noise = m.verdict(base, cand)
    assert abs(effect - 0.15) < 1e-9
    assert cand.spread < effect < base.spread  # 效应卡在两侧离散度之间
    assert verdict == "INCONCLUSIVE"


def test_single_round_spread_is_zero() -> None:
    """只跑一轮时 max == min, 离散度必须恒为 0.0.

    这正是"单轮跑分"这个反面案例的根源: 没有第二个样本, 就没有离散度
    可言, spread 只能是 0.0, 噪声闸门因此彻底失去意义。main() 必须靠
    "拒绝 --rounds < 2" 来堵住这个口子, 而不是指望 verdict() 本身能替
    单轮跑分补上一个不存在的噪声估计。
    """
    m = _load_module()
    stats = m.RunStats(profile="narrow", bars=1000, rounds=[1.234])
    assert stats.min == 1.234
    assert stats.max == 1.234
    assert stats.spread == 0.0


def test_zero_effect_zero_noise_is_inconclusive_not_regressed() -> None:
    """噪声带为 0 且效应也为 0 时必须 INCONCLUSIVE, 不能落到 REGRESSED.

    base 与 cand 都只跑了一轮且耗时恰好相同, 于是 base.spread ==
    cand.spread == 0.0, noise_band == 0.0, effect == 0.0。守卫条件如果
    写成 `abs(effect) < noise_band`(严格小于), `0.0 < 0.0` 为 False,
    会直接跳过 INCONCLUSIVE 分支、落到 `effect > 0` 的判断, 而 0.0 不
    大于 0, 于是被误判为 REGRESSED —— 明明什么都没变化, 却报"变慢了"。
    必须改成 `<=` 才能让这种零效应/零噪声的比较正确落回 INCONCLUSIVE。
    这条测试就是用来在有人把 `<=` 悄悄改回 `<` 时立刻失败的。
    """
    m = _load_module()
    base = m.RunStats(profile="narrow", bars=1000, rounds=[1.5])
    cand = m.RunStats(profile="narrow", bars=1000, rounds=[1.5])
    assert base.spread == 0.0
    assert cand.spread == 0.0
    verdict, effect, noise = m.verdict(base, cand)
    assert effect == 0.0
    assert noise == 0.0
    assert verdict == "INCONCLUSIVE"
