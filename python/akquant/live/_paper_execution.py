"""paper 会话的撮合参数: 费率、滑点、成交量限制.

paper 会话由 akquant 自己撮合, 与回测是同一个撮合器(``SimulatedExecutionClient``),
所以这些参数与 ``run_backtest`` 同名同义; ``broker_live`` 由柜台撮合, 成交价、成交量、
手续费都来自柜台回报, 传这些参数没有意义, 直接报错。

费率只作用于股票与基金(ETF): 未传的项沿用 paper 原有默认值(``use_china_market()``
的股票/基金默认费率, 见 ``src/market/stock.rs``/``fund.rs``), 不传任何费率参数时
完全不下发, 行为与此前一致。期货/期权沿用默认费率。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

# 与 Rust 默认值一致(``StockConfig::default`` / ``FundConfig::default``);
# tests/test_live_paper_execution_params.py 按成交手续费钉住这几个数。
PAPER_DEFAULT_COMMISSION_RATE = 0.0003
PAPER_DEFAULT_STOCK_STAMP_TAX_RATE = 0.0005
PAPER_DEFAULT_TRANSFER_FEE_RATE = 0.00001
PAPER_DEFAULT_MIN_COMMISSION = 5.0

_FEE_PARAM_NAMES = (
    "commission_rate",
    "stamp_tax_rate",
    "transfer_fee_rate",
    "min_commission",
)


@dataclass(frozen=True)
class PaperExecutionParams:
    """归一后的 paper 撮合参数; ``None`` 表示不覆盖."""

    commission_rate: Optional[float] = None
    stamp_tax_rate: Optional[float] = None
    transfer_fee_rate: Optional[float] = None
    min_commission: Optional[float] = None
    slippage: Optional[Dict[str, Any]] = None
    volume_limit_pct: Optional[float] = None

    @property
    def has_fees(self) -> bool:
        """是否传了任一费率参数."""
        return any(getattr(self, name) is not None for name in _FEE_PARAM_NAMES)


def _non_negative_float(name: str, value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"run_live({name}=) 须为数字, 收到 {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"run_live({name}=) 须为有限的非负数, 收到 {value!r}")
    return number


def _asset_name(instrument: Any) -> str:
    from ..backtest.engine import _asset_type_to_upper_name

    return str(_asset_type_to_upper_name(instrument.asset_type))


def _resolve_slippage(
    slippage: Any, instruments: Iterable[Any]
) -> Optional[Dict[str, Any]]:
    """与回测同一套规则归一滑点; ``ticks`` 按标的的 tick_size 换算成固定价差."""
    if slippage is None:
        return None
    from ..backtest.engine import _normalize_slippage_policy

    normalized = _normalize_slippage_policy(slippage, resolve_ticks=False)
    policy: Dict[str, Any] = {
        "type": str(normalized["type"]),
        "value": float(normalized["value"]),
    }
    if policy["type"] != "ticks":
        return policy
    tick_sizes = {
        float(tick)
        for inst in instruments
        if (tick := getattr(inst, "tick_size", None)) is not None
    }
    if not tick_sizes:
        raise ValueError("run_live(slippage={'type': 'ticks'}) 需要标的带 tick_size")
    if len(tick_sizes) != 1:
        raise ValueError(
            "run_live(slippage={'type': 'ticks'}) 要求所有标的的 tick_size 相同, "
            f"收到 {sorted(tick_sizes)}; 请改用 'fixed' 或 'percent'"
        )
    return {"type": "fixed", "value": float(policy["value"]) * tick_sizes.pop()}


def resolve_paper_execution_params(
    *,
    trading_mode: str,
    instruments: Iterable[Any],
    commission_rate: Any = None,
    stamp_tax_rate: Any = None,
    transfer_fee_rate: Any = None,
    min_commission: Any = None,
    slippage: Any = None,
    volume_limit_pct: Any = None,
) -> PaperExecutionParams:
    """校验并归一 paper 撮合参数(连网关之前调用).

    :raises ValueError: ``broker_live`` 下传了任一参数; 取值越界; 只有期货/期权
        标的却传了费率参数; ``ticks`` 滑点无法换算
    :raises TypeError: 数值参数类型错; 滑点给了裸数字
    """
    raw = {
        "commission_rate": commission_rate,
        "stamp_tax_rate": stamp_tax_rate,
        "transfer_fee_rate": transfer_fee_rate,
        "min_commission": min_commission,
        "slippage": slippage,
        "volume_limit_pct": volume_limit_pct,
    }
    given = [name for name, value in raw.items() if value is not None]
    if trading_mode == "broker_live":
        if given:
            raise ValueError(
                f"run_live({', '.join(given)}=) 只作用于 trading_mode='paper': "
                "broker_live 由柜台撮合, 成交价、成交量与手续费都以柜台回报为准"
            )
        return PaperExecutionParams()

    instrument_list = list(instruments or [])
    params = PaperExecutionParams(
        commission_rate=_non_negative_float("commission_rate", commission_rate),
        stamp_tax_rate=_non_negative_float("stamp_tax_rate", stamp_tax_rate),
        transfer_fee_rate=_non_negative_float("transfer_fee_rate", transfer_fee_rate),
        min_commission=_non_negative_float("min_commission", min_commission),
        slippage=_resolve_slippage(slippage, instrument_list),
        volume_limit_pct=_non_negative_float("volume_limit_pct", volume_limit_pct),
    )
    if params.has_fees and instrument_list:
        asset_names = {_asset_name(inst) for inst in instrument_list}
        if not asset_names & {"STOCK", "FUND"}:
            raise ValueError(
                "run_live 的费率参数只作用于股票与基金标的, 本次会话没有这类标的; "
                "期货/期权 paper 会话沿用默认费率"
            )
        others = sorted(asset_names - {"STOCK", "FUND"})
        if others:
            logger.warning(
                "run_live 的费率参数只作用于股票与基金标的, %s 标的沿用默认费率",
                "/".join(others),
            )
    return params


def apply_paper_execution_params(engine: Any, params: PaperExecutionParams) -> None:
    """把参数下发给引擎.

    须在 ``use_simulated_execution()`` 与 ``use_china_market()`` /
    ``use_china_futures_market()`` 之后调用: 后者会重建市场配置, 先下发会被冲掉。
    """
    if params.has_fees:
        commission = _or(params.commission_rate, PAPER_DEFAULT_COMMISSION_RATE)
        transfer = _or(params.transfer_fee_rate, PAPER_DEFAULT_TRANSFER_FEE_RATE)
        minimum = _or(params.min_commission, PAPER_DEFAULT_MIN_COMMISSION)
        engine.set_stock_fee_rules(
            commission,
            _or(params.stamp_tax_rate, PAPER_DEFAULT_STOCK_STAMP_TAX_RATE),
            transfer,
            minimum,
        )
        # 基金不收印花税(``FundConfig::default`` 为 0), stamp_tax_rate 不作用于基金
        engine.set_fund_fee_rules(commission, transfer, minimum)
    if params.slippage is not None:
        engine.set_slippage(params.slippage["type"], float(params.slippage["value"]))
    if params.volume_limit_pct is not None:
        engine.set_volume_limit(params.volume_limit_pct)


def _or(value: Optional[float], default: float) -> float:
    return default if value is None else value
