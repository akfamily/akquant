from typing import Any, Optional

from .log import get_logger
from .utils import parse_duration_to_bars

logger = get_logger("ml")


def _get_validation_model(strategy: Any) -> Any:
    """返回验证配置使用的模型模板."""
    template_model = getattr(strategy, "_ml_model_template", None)
    if template_model is not None:
        return template_model
    return getattr(strategy, "model", None)


def _get_validation_config(strategy: Any) -> Any:
    """返回验证配置对象."""
    model = _get_validation_model(strategy)
    return getattr(model, "validation_config", None)


def _resolve_validation_windows(strategy: Any) -> tuple[int, int, int]:
    """解析模型 walk-forward 配置窗口."""
    validation_config = _get_validation_config(strategy)
    if validation_config is None:
        return 0, 0, 0

    train_window = parse_duration_to_bars(
        validation_config.train_window,
        validation_config.frequency,
    )
    test_window = parse_duration_to_bars(
        validation_config.test_window,
        validation_config.frequency,
    )
    rolling_step = parse_duration_to_bars(
        validation_config.rolling_step,
        validation_config.frequency,
    )
    return train_window, test_window, rolling_step


def _effective_training_step(test_window: int, rolling_step: int) -> int:
    """计算有效训练步长."""
    if rolling_step > 0:
        return rolling_step
    if test_window > 0:
        return test_window
    return 0


def _validation_lifecycle_enabled(strategy: Any) -> bool:
    """返回是否启用验证窗口生命周期管理."""
    return bool(getattr(strategy, "_ml_validation_lifecycle", False))


def _clone_model_for_training(strategy: Any) -> Any:
    """为当前训练窗口构建待训练模型副本."""
    template_model = _get_validation_model(strategy)
    if template_model is None:
        return None

    validation_config = _get_validation_config(strategy)
    active_model = getattr(strategy, "_ml_active_model", None)
    if (
        validation_config is not None
        and bool(getattr(validation_config, "incremental", False))
        and active_model is not None
    ):
        return active_model.clone()
    return template_model.clone()


def activate_pending_model(strategy: Any) -> None:
    """在计划生效点激活待生效模型."""
    if not _validation_lifecycle_enabled(strategy):
        return

    pending_model = getattr(strategy, "_ml_pending_model", None)
    activation_bar = getattr(strategy, "_ml_pending_activation_bar", None)
    if pending_model is None or activation_bar is None:
        return
    if int(strategy._bar_count) < int(activation_bar):
        return

    strategy._ml_active_model = pending_model
    strategy.model = pending_model
    strategy._ml_active_window_index = int(
        getattr(strategy, "_ml_pending_window_index", 0)
    )
    strategy._ml_active_window_start_bar = getattr(
        strategy,
        "_ml_pending_window_start_bar",
        None,
    )
    strategy._ml_active_window_end_bar = getattr(
        strategy,
        "_ml_pending_window_end_bar",
        None,
    )
    strategy._ml_pending_model = None
    strategy._ml_pending_activation_bar = None
    strategy._ml_pending_window_index = 0
    strategy._ml_pending_window_start_bar = None
    strategy._ml_pending_window_end_bar = None


def auto_configure_model(strategy: Any) -> None:
    """应用模型校验配置（如滚动训练窗口参数）."""
    if strategy._model_configured:
        return

    if strategy.model and strategy.model.validation_config:
        try:
            train_window, test_window, rolling_step = _resolve_validation_windows(
                strategy
            )
            effective_step = _effective_training_step(test_window, rolling_step)
            strategy.set_rolling_window(train_window, effective_step)
            setattr(strategy, "_ml_validation_lifecycle", True)
            setattr(strategy, "_ml_model_template", strategy.model)
            setattr(strategy, "_ml_active_model", None)
            setattr(strategy, "_ml_pending_model", None)
            setattr(strategy, "_ml_pending_activation_bar", None)
            setattr(strategy, "_ml_active_window_index", 0)
            setattr(strategy, "_ml_active_window_start_bar", None)
            setattr(strategy, "_ml_active_window_end_bar", None)
            setattr(strategy, "_ml_pending_window_index", 0)
            setattr(strategy, "_ml_pending_window_start_bar", None)
            setattr(strategy, "_ml_pending_window_end_bar", None)
            setattr(strategy, "_rolling_test_window", test_window)
            setattr(strategy, "_rolling_last_train_bar", 0)
            setattr(strategy, "_rolling_window_index", 0)
            setattr(strategy, "_rolling_next_train_bar", max(train_window, 1))
        except Exception as e:
            logger.warning("Failed to configure model validation: %s", e)
    else:
        setattr(strategy, "_ml_validation_lifecycle", False)

    strategy._model_configured = True


def should_trigger_training(strategy: Any) -> bool:
    """返回当前 bar 是否应触发自动训练."""
    if strategy._rolling_step <= 0:
        return False

    validation_config = _get_validation_config(strategy)
    if validation_config is None:
        return bool(int(strategy._bar_count) % int(strategy._rolling_step) == 0)

    next_train_bar = int(
        getattr(strategy, "_rolling_next_train_bar", strategy._rolling_train_window)
    )
    return bool(
        int(strategy._bar_count)
        >= max(next_train_bar, int(strategy._rolling_train_window))
    )


def consume_training_trigger(strategy: Any) -> None:
    """消费一次训练触发并推进下一窗口."""
    validation_config = _get_validation_config(strategy)
    if validation_config is None or strategy._rolling_step <= 0:
        return

    current_bar = int(strategy._bar_count)
    next_window_index = int(getattr(strategy, "_rolling_window_index", 0)) + 1
    pending_start_bar = current_bar + 1
    pending_end_bar: Optional[int]
    if int(getattr(strategy, "_rolling_test_window", 0)) > 0:
        pending_end_bar = pending_start_bar + int(strategy._rolling_test_window) - 1
    else:
        pending_end_bar = None

    setattr(strategy, "_rolling_last_train_bar", current_bar)
    setattr(
        strategy,
        "_rolling_next_train_bar",
        current_bar + int(strategy._rolling_step),
    )
    setattr(strategy, "_rolling_window_index", next_window_index)
    setattr(strategy, "_ml_pending_activation_bar", pending_start_bar)
    setattr(strategy, "_ml_pending_window_index", next_window_index)
    setattr(strategy, "_ml_pending_window_start_bar", pending_start_bar)
    setattr(strategy, "_ml_pending_window_end_bar", pending_end_bar)


def begin_training_cycle(strategy: Any) -> Optional[tuple[Any, Any]]:
    """开始一次训练周期并临时挂载待训练模型."""
    if not _validation_lifecycle_enabled(strategy):
        return None

    training_model = _clone_model_for_training(strategy)
    if training_model is None:
        return None

    previous_public_model = getattr(strategy, "model", None)
    strategy.model = training_model
    return previous_public_model, training_model


def finalize_training_cycle(
    strategy: Any,
    cycle_state: Optional[tuple[Any, Any]],
) -> None:
    """结束训练周期并恢复对外模型引用."""
    if cycle_state is None:
        return

    previous_public_model, training_model = cycle_state
    strategy._ml_pending_model = training_model
    active_model = getattr(strategy, "_ml_active_model", None)
    if active_model is not None:
        strategy.model = active_model
        return
    strategy.model = previous_public_model


def is_model_ready(strategy: Any) -> bool:
    """返回当前是否已有可用于推理的活动模型."""
    if _validation_lifecycle_enabled(strategy):
        return getattr(strategy, "_ml_active_model", None) is not None
    return getattr(strategy, "model", None) is not None


def current_validation_window(strategy: Any) -> Optional[dict[str, Any]]:
    """返回当前验证窗口状态."""
    if not _validation_lifecycle_enabled(strategy):
        return None

    return {
        "is_model_ready": is_model_ready(strategy),
        "window_index": int(getattr(strategy, "_ml_active_window_index", 0)),
        "train_window": int(getattr(strategy, "_rolling_train_window", 0)),
        "test_window": int(getattr(strategy, "_rolling_test_window", 0)),
        "rolling_step": int(getattr(strategy, "_rolling_step", 0)),
        "active_start_bar": getattr(strategy, "_ml_active_window_start_bar", None),
        "active_end_bar": getattr(strategy, "_ml_active_window_end_bar", None),
        "pending_activation_bar": getattr(strategy, "_ml_pending_activation_bar", None),
        "pending_window_index": int(getattr(strategy, "_ml_pending_window_index", 0)),
        "next_train_bar": int(getattr(strategy, "_rolling_next_train_bar", 0)),
    }


def on_train_signal(strategy: Any, context: Any) -> None:
    """滚动训练信号回调."""
    if strategy.model:
        try:
            X_df, _ = strategy.get_rolling_data()

            validation_config = _get_validation_config(strategy)
            if validation_config and validation_config.verbose:
                ts_str = ""
                if strategy.current_bar:
                    ts_str = strategy.format_time(strategy.current_bar.timestamp)
                train_window = int(getattr(strategy, "_rolling_train_window", 0))
                test_window = int(getattr(strategy, "_rolling_test_window", 0))
                window_index = int(getattr(strategy, "_ml_pending_window_index", 0))
                activation_bar = getattr(strategy, "_ml_pending_activation_bar", None)
                logger.info(
                    "[%s] Auto-training triggered | Window=%s | Train Size=%s | "
                    "Train Window=%s | Test Window=%s | Activation Bar=%s",
                    ts_str,
                    window_index,
                    len(X_df),
                    train_window,
                    test_window,
                    activation_bar,
                )

            X, y = strategy.prepare_features(X_df, mode="training")
            strategy.model.fit(X, y)
        except NotImplementedError:
            pass
        except Exception as e:
            logger.warning("Auto-training failed at bar %s: %s", strategy._bar_count, e)
