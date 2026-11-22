"""波动率微笑与曲面拟合: SVI(Zeliade 准显式标定)与 SABR(Hagan 2002).

每个到期日拟合一条微笑(:class:`VolSmile`), 多个到期日组成曲面(:class:`VolSurface`),
期限方向在固定对数在值程度 ``k = ln(K/F)`` 上对总方差 ``w = σ²t`` 线性插值。

- SVI 原始参数(总方差口径): ``w(k) = a + b(ρ(k−m) + sqrt((k−m)² + σ²))``。固定
  (m, σ) 后对 (a, bρσ, bσ) 是带约束的线性最小二乘, 精确求解; 外层只在 (m, σ) 上用
  Nelder-Mead 搜索(Zeliade Systems 白皮书 zwp-0005)。约束域保证 ``a ≥ 0`` 与 Lee
  矩条件 ``b(1+|ρ|) ≤ 4``, 但不保证无蝶式套利, 拟合后检查并在
  :attr:`VolSmile.butterfly_min_g` 报告。
- SABR: β 固定, 拟合 α、ρ、ν, 隐含波动率用 Hagan 对数正态近似公式。

拟合在 Rust 中进行, 多条微笑并行。只检查、不修正套利: 蝶式套利看
``butterfly_min_g < 0``, 日历套利(总方差随期限下降)看
:attr:`VolSurface.calendar_arbitrage`。
"""

import math
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from ..akquant import _sabr_fit, _sabr_vol, _svi_fit, _svi_total_variance

SmileModel = Literal["svi", "sabr"]
FloatOrArray = Union[float, np.ndarray]

#: SVI 的 σ 搜索区间(对数在值程度单位)
_SVI_SIGMA_LOW = 0.005
_SVI_SIGMA_HIGH = 1.0
#: SVI 的 m 搜索区间: 数据 k 范围向两侧各扩展该倍数的跨度
_SVI_M_MARGIN = 1.0
#: Nelder-Mead 收敛判据: 单纯形各顶点参数差与目标函数(加权均方误差)差的上限
_FIT_XTOL = 1e-10
_FIT_FTOL = 1e-16
_FIT_MAX_ITER = 5000
#: 各模型拟合所需的最少报价数(SVI 5 个参数, SABR β 固定后 3 个)
_MIN_POINTS: Dict[str, int] = {"svi": 5, "sabr": 3}
#: 套利检查网格的点数与数值微分步长
_ARB_GRID = 201
_ARB_STEP = 1e-4
#: 日历套利判定容差(总方差)
_CALENDAR_TOL = 1e-10


@dataclass(frozen=True)
class VolSmile:
    """一个到期日的波动率微笑拟合结果.

    ``params``: SVI 为 a、b、rho、m、sigma(总方差口径); SABR 为 alpha、beta、rho、nu。
    ``rmse``: 拟合点上隐含波动率的加权均方根误差。``k_range``: 参与拟合的对数在值程度
    范围, 范围外是外推。``butterfly_min_g``: ``k_range`` 内 Gatheral 密度函数 g(k)
    的最小值, 小于 0 表示该范围内存在蝶式套利(隐含的风险中性密度为负)。
    ``converged``: 最优起点的 Nelder-Mead 是否在迭代上限内收敛。
    """

    model: str
    forward: float
    t: float
    params: Dict[str, float]
    rmse: float
    n_points: int
    converged: bool
    k_range: Tuple[float, float]
    butterfly_min_g: float
    expiry_date: Optional[int] = None

    @property
    def butterfly_arbitrage(self) -> bool:
        """拟合范围内是否存在蝶式套利."""
        return self.butterfly_min_g < 0

    def _total_variance_k(self, k: np.ndarray) -> np.ndarray:
        p = self.params
        if self.model == "svi":
            ones = np.ones_like(k)
            return np.asarray(
                _svi_total_variance(
                    k,
                    ones * p["a"],
                    ones * p["b"],
                    ones * p["rho"],
                    ones * p["m"],
                    ones * p["sigma"],
                )
            )
        n = len(k)
        vol = _sabr_vol(
            np.full(n, self.forward),
            self.forward * np.exp(k),
            np.full(n, self.t),
            np.full(n, p["alpha"]),
            p["beta"],
            np.full(n, p["rho"]),
            np.full(n, p["nu"]),
        )
        return np.asarray(vol) ** 2 * self.t

    def total_variance(self, strike: Any) -> FloatOrArray:
        """行权价处的总方差 σ²t."""
        scalar = np.ndim(strike) == 0
        k = np.log(np.atleast_1d(np.asarray(strike, dtype=float)) / self.forward)
        out = self._total_variance_k(np.ascontiguousarray(k.ravel()))
        return float(out[0]) if scalar else out.reshape(np.shape(strike))

    def vol(self, strike: Any) -> FloatOrArray:
        """行权价处的隐含波动率(年化)."""
        w = self.total_variance(strike)
        return math.sqrt(w / self.t) if isinstance(w, float) else np.sqrt(w / self.t)


@dataclass(frozen=True)
class VolSurface:
    """多个到期日的微笑组成的波动率曲面.

    ``smiles``: 拟合成功的到期日 → :class:`VolSmile`; ``failures``: 未能拟合的到期日
    → 原因; ``calendar_arbitrage``: 相邻到期日 (近, 远) 中, 在两者共同的拟合范围内
    远月总方差低于近月的对。
    """

    smiles: Dict[int, VolSmile]
    failures: Dict[int, str] = field(default_factory=dict)
    calendar_arbitrage: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def expiries(self) -> List[int]:
        """拟合成功的到期日, 按期限升序."""
        return sorted(self.smiles, key=lambda e: self.smiles[e].t)

    def smile(self, expiry_date: int) -> VolSmile:
        """取一个到期日的微笑; 该到期日未能拟合或不存在时报出原因."""
        if expiry_date in self.smiles:
            return self.smiles[expiry_date]
        if expiry_date in self.failures:
            raise ValueError(
                f"到期日 {expiry_date} 未能拟合: {self.failures[expiry_date]}"
            )
        raise ValueError(f"曲面中没有到期日 {expiry_date}")

    def vol(self, strike: Any, t: float) -> FloatOrArray:
        """任意期限 t(年)、行权价处的隐含波动率.

        在相邻两个已拟合到期日之间: 远期按期限线性插值, 在 ``k = ln(K/F(t))`` 处
        对两条微笑的总方差按期限线性插值。t 超出已拟合到期日的期限范围时报错, 不外推。
        """
        ordered = [self.smiles[e] for e in self.expiries]
        if not ordered:
            raise ValueError("曲面没有拟合成功的到期日")
        t = float(t)
        t_lo, t_hi = ordered[0].t, ordered[-1].t
        if not t_lo <= t <= t_hi:
            raise ValueError(
                f"t={t:g} 超出已拟合到期日的期限范围 [{t_lo:g}, {t_hi:g}], 不外推"
            )
        times = np.array([s.t for s in ordered])
        i = int(
            np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(ordered) - 1)
        )
        near = ordered[i]
        if near.t == t or i == len(ordered) - 1:
            return near.vol(strike)
        far = ordered[i + 1]
        frac = (t - near.t) / (far.t - near.t)
        forward = near.forward + frac * (far.forward - near.forward)
        k = np.log(np.asarray(strike, dtype=float) / forward)
        w = (1 - frac) * np.asarray(near.total_variance(near.forward * np.exp(k)))
        w = w + frac * np.asarray(far.total_variance(far.forward * np.exp(k)))
        out = np.sqrt(w / t)
        return float(out) if np.ndim(out) == 0 else out


def _butterfly_min_g(smile: VolSmile) -> float:
    """``k_range`` 内 g(k) = (1 − k w'/(2w))² − w'²/4·(1/w + 1/4) + w''/2 的最小值.

    w ≤ 0 的点 g 记为 -inf(总方差为负本身就是套利)。导数用中心差分。
    """
    lo, hi = smile.k_range
    k = np.linspace(lo, hi, _ARB_GRID)
    h = _ARB_STEP
    w = smile._total_variance_k(k)
    w_up = smile._total_variance_k(k + h)
    w_dn = smile._total_variance_k(k - h)
    d1 = (w_up - w_dn) / (2 * h)
    d2 = (w_up - 2 * w + w_dn) / (h * h)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1 - k * d1 / (2 * w)) ** 2 - d1**2 / 4 * (1 / w + 0.25) + d2 / 2
    g = np.where(w > 0, g, -np.inf)
    return float(np.min(g))


def _check_model(model: str, beta: float) -> None:
    if model not in _MIN_POINTS:
        raise ValueError(f"model 只接受 svi/sabr, 实际: {model!r}")
    if model == "sabr" and not 0.0 <= beta <= 1.0:
        raise ValueError(f"SABR 的 beta 须在 [0, 1] 内, 实际: {beta:g}")


def _fit_slices(
    model: str,
    beta: float,
    slices: Sequence[
        Tuple[Optional[int], float, float, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> List[VolSmile]:
    """一次批量拟合多条微笑; 每条为 (到期日, 远期, 期限, 行权价, 隐含波动率, 权重)."""
    if not slices:
        return []
    strikes = np.concatenate([s[3] for s in slices])
    vols = np.concatenate([s[4] for s in slices])
    weights = np.concatenate([s[5] for s in slices])
    forwards = np.array([s[1] for s in slices], dtype=float)
    ts = np.array([s[2] for s in slices], dtype=float)
    sizes = np.array([len(s[3]) for s in slices], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    fwd_rows = np.repeat(forwards, sizes)
    t_rows = np.repeat(ts, sizes)
    k = np.log(strikes / fwd_rows)
    if model == "svi":
        a, b, rho, m, sigma, _, converged = _svi_fit(
            k,
            vols**2 * t_rows,
            weights,
            offsets,
            _SVI_SIGMA_LOW,
            _SVI_SIGMA_HIGH,
            _SVI_M_MARGIN,
            _FIT_XTOL,
            _FIT_FTOL,
            _FIT_MAX_ITER,
        )
        params = [
            {"a": a[i], "b": b[i], "rho": rho[i], "m": m[i], "sigma": sigma[i]}
            for i in range(len(slices))
        ]
    else:
        alpha, rho, nu, _, converged = _sabr_fit(
            forwards,
            ts,
            strikes,
            vols,
            weights,
            offsets,
            beta,
            _FIT_XTOL,
            _FIT_FTOL,
            _FIT_MAX_ITER,
        )
        params = [
            {"alpha": alpha[i], "beta": beta, "rho": rho[i], "nu": nu[i]}
            for i in range(len(slices))
        ]
    smiles = []
    for i, (expiry, forward, t, strike, vol, weight) in enumerate(slices):
        seg = k[offsets[i] : offsets[i + 1]]
        draft = VolSmile(
            model=model,
            forward=forward,
            t=t,
            params={name: float(v) for name, v in params[i].items()},
            rmse=math.nan,
            n_points=len(strike),
            converged=bool(converged[i]),
            k_range=(float(seg.min()), float(seg.max())),
            butterfly_min_g=math.nan,
            expiry_date=expiry,
        )
        fitted = np.asarray(draft.vol(strike))
        rmse = math.sqrt(float(np.sum(weight * (fitted - vol) ** 2) / np.sum(weight)))
        smiles.append(
            replace(draft, rmse=rmse, butterfly_min_g=_butterfly_min_g(draft))
        )
    return smiles


def fit_smile(
    strikes: Any,
    ivs: Any,
    forward: float,
    t: float,
    model: SmileModel = "svi",
    weights: Any = None,
    beta: float = 1.0,
    expiry_date: Optional[int] = None,
) -> VolSmile:
    """
    拟合一个到期日的波动率微笑.

    :param strikes: 行权价数组(为正、互不相同)
    :param ivs: 对应的隐含波动率(年化, 为正)
    :param forward: 远期价格(商品期权为标的期货价格)
    :param t: 期限(年)
    :param model: ``"svi"`` (默认) 或 ``"sabr"``
    :param weights: 各点权重(非负), 不填等权; SVI 按总方差、SABR 按波动率加权最小二乘
    :param beta: SABR 的 β(固定), 默认 1(对数正态); SVI 忽略
    :param expiry_date: 到期日(YYYYMMDD), 仅作标注
    :return: :class:`VolSmile`
    """
    _check_model(model, beta)
    k_arr = np.asarray(strikes, dtype=float).ravel()
    v_arr = np.asarray(ivs, dtype=float).ravel()
    w_arr = (
        np.ones_like(k_arr)
        if weights is None
        else np.asarray(weights, dtype=float).ravel()
    )
    if not (len(k_arr) == len(v_arr) == len(w_arr)):
        raise ValueError(
            f"strikes、ivs、weights 长度须相同, 实际 {len(k_arr)}、{len(v_arr)}、"
            f"{len(w_arr)}"
        )
    if not (math.isfinite(forward) and forward > 0):
        raise ValueError(f"forward 须为正数, 实际: {forward}")
    if not (math.isfinite(t) and t > 0):
        raise ValueError(f"t 须为正数, 实际: {t}")
    if not np.all(np.isfinite(k_arr) & (k_arr > 0)):
        raise ValueError("strikes 须全部为正数")
    if len(np.unique(k_arr)) != len(k_arr):
        raise ValueError("strikes 不能重复")
    if not np.all(np.isfinite(v_arr) & (v_arr > 0)):
        raise ValueError("ivs 须全部为正数(先剔除算不出隐含波动率的点)")
    if not np.all(np.isfinite(w_arr) & (w_arr >= 0)) or not np.any(w_arr > 0):
        raise ValueError("weights 须非负且不全为 0")
    if len(k_arr) < _MIN_POINTS[model]:
        raise ValueError(
            f"{model} 拟合至少需要 {_MIN_POINTS[model]} 个报价, 实际 {len(k_arr)} 个"
        )
    return _fit_slices(
        model, beta, [(expiry_date, float(forward), float(t), k_arr, v_arr, w_arr)]
    )[0]


def _calendar_violations(smiles: List[VolSmile]) -> List[Tuple[int, int]]:
    """相邻到期日在共同拟合范围内远月总方差低于近月的 (近, 远) 对."""
    out = []
    for near, far in zip(smiles, smiles[1:]):
        lo = max(near.k_range[0], far.k_range[0])
        hi = min(near.k_range[1], far.k_range[1])
        if lo > hi:
            continue
        k = np.linspace(lo, hi, _ARB_GRID)
        if np.any(far._total_variance_k(k) < near._total_variance_k(k) - _CALENDAR_TOL):
            out.append((int(near.expiry_date or 0), int(far.expiry_date or 0)))
    return out


def fit_surface(
    chain: pd.DataFrame, model: SmileModel = "svi", beta: float = 1.0
) -> VolSurface:
    """
    由期权链拟合波动率曲面, 每个到期日一条微笑.

    按到期日分组, 每组用自己的远期价格, 所以可以把一个品种不同月份标的的期权链合在
    一起拟合(商品期权各月标的不同)。每个行权价只取虚值一侧: 行权价低于远期用认沽,
    不低于远期用认购; 剔除隐含波动率为 NaN 的合约。有效报价少于模型所需数量、远期
    或期限缺失的到期日记入 :attr:`VolSurface.failures`, 不参与曲面; 同一到期日的
    远期价格不一致(多个标的同一到期日)时报错。

    :param chain: 期权链, 需含 expiry_date、t、forward、strike、option_type、iv 列
        (:meth:`akquant.Strategy.option_chain` 的输出即可)
    :param model: ``"svi"`` (默认) 或 ``"sabr"``
    :param beta: SABR 的 β(固定), 默认 1
    :return: :class:`VolSurface`
    """
    _check_model(model, beta)
    required = ["expiry_date", "t", "forward", "strike", "option_type", "iv"]
    missing = [c for c in required if c not in chain.columns]
    if missing:
        raise ValueError(f"期权链缺少列: {missing}")
    slices = []
    failures: Dict[int, str] = {}
    for key, group in chain.groupby("expiry_date", sort=True):
        expiry = int(cast(Any, key))
        forwards = group["forward"].dropna().unique()
        if len(forwards) > 1:
            raise ValueError(
                f"到期日 {expiry} 的合约远期价格不一致({len(forwards)} 个), "
                "同一到期日的多个标的须分开拟合"
            )
        forward = float(group["forward"].iloc[0])
        t = float(group["t"].iloc[0])
        if not (math.isfinite(forward) and forward > 0):
            failures[expiry] = "远期价格缺失(标的没有最新价)"
            continue
        if not (math.isfinite(t) and t > 0):
            failures[expiry] = f"期限不为正: {t}"
            continue
        otm = np.where(group["strike"].to_numpy(dtype=float) < forward, "PUT", "CALL")
        rows = group[
            (group["option_type"].to_numpy() == otm)
            & np.isfinite(group["iv"].to_numpy(dtype=float))
        ]
        need = _MIN_POINTS[model]
        if len(rows) < need:
            failures[expiry] = (
                f"有隐含波动率的虚值合约 {len(rows)} 个, 少于 {model} 拟合所需的 "
                f"{need} 个"
            )
            continue
        strikes = rows["strike"].to_numpy(dtype=float)
        ivs = rows["iv"].to_numpy(dtype=float)
        slices.append((expiry, forward, t, strikes, ivs, np.ones_like(strikes)))
    fitted = sorted(_fit_slices(model, beta, slices), key=lambda s: s.t)
    return VolSurface(
        smiles={int(s.expiry_date or 0): s for s in fitted},
        failures=failures,
        calendar_arbitrage=_calendar_violations(fitted),
    )
