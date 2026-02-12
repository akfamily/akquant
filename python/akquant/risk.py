from typing import TYPE_CHECKING, Optional

from .config import RiskConfig as PyRiskConfig

if TYPE_CHECKING:
    from .akquant import Engine


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

    if config.safety_margin is not None:
        rust_config.safety_margin = config.safety_margin

    # Use the dedicated setter method to ensure the update propagates to the Engine
    # Direct attribute assignment (engine.risk_manager.config = ...) might only
    # update a copy
    if hasattr(engine, "set_risk_config"):
        engine.set_risk_config(rust_config)
    else:
        # Fallback for older versions or if method is missing
        engine.risk_manager.config = rust_config
