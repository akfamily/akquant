from typing import TYPE_CHECKING, Optional

from .config import RiskConfig as PyRiskConfig
from .log import build_log_extra, get_logger

if TYPE_CHECKING:
    from .akquant import Engine

logger = get_logger("risk")


def _build_risk_log_extra() -> dict[str, Optional[str]]:
    """Build structured context for risk configuration logs."""
    return build_log_extra(phase="risk")


def apply_risk_config(engine: "Engine", config: Optional[PyRiskConfig]) -> None:
    """
    Apply Python-side RiskConfig to the Rust Engine's RiskManager.

    :param engine: The backtest engine instance.
    :param config: The Python RiskConfig object.
    """
    if config is None:
        return

    # Get the Rust RiskConfig object from the engine's risk manager
    # Assuming engine.risk_manager.config is accessible and mutable
    # Or we can create a new one and assign it

    rust_config = engine.risk_manager.config

    if config.max_order_size is not None:
        rust_config.max_order_size = config.max_order_size

    if config.max_order_value is not None:
        rust_config.max_order_value = config.max_order_value

    if config.max_position_size is not None:
        rust_config.max_position_size = config.max_position_size

    if config.restricted_list is not None:
        rust_config.restricted_list = config.restricted_list

    if config.active is not None:
        rust_config.active = config.active

    if hasattr(config, "check_cash") and config.check_cash is not None:
        rust_config.check_cash = config.check_cash

    if config.safety_margin is not None:
        rust_config.safety_margin = config.safety_margin
    account_mode = getattr(config, "account_mode", None)
    if account_mode is not None:
        setattr(rust_config, "account_mode", str(account_mode))
    enable_short_sell = getattr(config, "enable_short_sell", None)
    if enable_short_sell is not None:
        setattr(rust_config, "enable_short_sell", bool(enable_short_sell))
    initial_margin_ratio = getattr(config, "initial_margin_ratio", None)
    if initial_margin_ratio is not None:
        setattr(rust_config, "initial_margin_ratio", float(initial_margin_ratio))
    maintenance_margin_ratio = getattr(config, "maintenance_margin_ratio", None)
    if maintenance_margin_ratio is not None:
        setattr(
            rust_config, "maintenance_margin_ratio", float(maintenance_margin_ratio)
        )
    financing_rate_annual = getattr(config, "financing_rate_annual", None)
    if financing_rate_annual is not None:
        setattr(rust_config, "financing_rate_annual", float(financing_rate_annual))
    borrow_rate_annual = getattr(config, "borrow_rate_annual", None)
    if borrow_rate_annual is not None:
        setattr(rust_config, "borrow_rate_annual", float(borrow_rate_annual))
    allow_force_liquidation = getattr(config, "allow_force_liquidation", None)
    if allow_force_liquidation is not None:
        setattr(rust_config, "allow_force_liquidation", bool(allow_force_liquidation))
    liquidation_priority = getattr(config, "liquidation_priority", None)
    if liquidation_priority is not None:
        setattr(rust_config, "liquidation_priority", str(liquidation_priority))

    # Use the dedicated setter method to ensure the update propagates to the Engine
    # Direct attribute assignment (engine.risk_manager.config = ...) might only
    # update a copy
    if hasattr(engine, "set_risk_config"):
        engine.set_risk_config(rust_config)
    else:
        # Fallback for older versions or if method is missing
        engine.risk_manager.config = rust_config

    # Apply dynamic rules (max_position_pct, sector_concentration)
    # These are not part of Rust RiskConfig struct but are handled by RiskManager
    rm = engine.risk_manager

    # Apply dynamic rules (max_position_pct, sector_concentration)
    # These are not part of Rust RiskConfig struct but are handled by RiskManager
    rm = engine.risk_manager

    if config.max_position_pct is not None:
        if hasattr(rm, "add_max_position_percent_rule"):
            rm.add_max_position_percent_rule(config.max_position_pct)
        else:
            logger.warning(
                "RiskManager does not support add_max_position_percent_rule.",
                extra=_build_risk_log_extra(),
            )

    if config.sector_concentration is not None:
        if hasattr(rm, "add_sector_concentration_rule"):
            if (
                isinstance(config.sector_concentration, (list, tuple))
                and len(config.sector_concentration) == 2
            ):
                rm.add_sector_concentration_rule(
                    config.sector_concentration[0], config.sector_concentration[1]
                )
            else:
                logger.warning(
                    "sector_concentration must be a tuple (limit, sector_map). "
                    "Rule ignored.",
                    extra=_build_risk_log_extra(),
                )
        else:
            logger.warning(
                "RiskManager does not support add_sector_concentration_rule.",
                extra=_build_risk_log_extra(),
            )

    if config.max_account_drawdown is not None:
        if hasattr(rm, "add_max_drawdown_rule"):
            rm.add_max_drawdown_rule(config.max_account_drawdown)
        else:
            logger.warning(
                "RiskManager does not support add_max_drawdown_rule.",
                extra=_build_risk_log_extra(),
            )

    if config.max_daily_loss is not None:
        if hasattr(rm, "add_max_daily_loss_rule"):
            rm.add_max_daily_loss_rule(config.max_daily_loss)
        else:
            logger.warning(
                "RiskManager does not support add_max_daily_loss_rule.",
                extra=_build_risk_log_extra(),
            )

    if config.stop_loss_threshold is not None:
        if hasattr(rm, "add_stop_loss_rule"):
            rm.add_stop_loss_rule(config.stop_loss_threshold)
        else:
            logger.warning(
                "RiskManager does not support add_stop_loss_rule.",
                extra=_build_risk_log_extra(),
            )

    # Update the engine's risk manager with the new rules
    # This is critical if engine.risk_manager returns a copy/clone
    engine.risk_manager = rm
