from typing import Any, Optional, cast

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
        # 阈值语义, 不用取模: 取模对"任何一次被跳过的 bar 事件"都不健壮——
        # 例如 per-symbol warmup 门槛恰好挡住了 _bar_count 等于 step 倍数的
        # 那次事件, 取模下一次事件 _bar_count 已经往前走了一格, 模值再也凑
        # 不回 0, 这次训练永久丢失而非延后。阈值化后同样的跳过只会让触发
        # 顺延到下一根满足条件的 bar, 不会丢(状态推进见
        # consume_training_trigger)。没有任何跳过时, 首次触发时机与取模
        # 完全一致(均为 _bar_count == step 那一刻)。
        last_train_bar = int(getattr(strategy, "_last_train_bar_count", 0))
        return bool(
            int(strategy._bar_count) - last_train_bar >= int(strategy._rolling_step)
        )

    next_train_bar = int(
        getattr(strategy, "_rolling_next_train_bar", strategy._rolling_train_window)
    )
    return bool(
        int(strategy._bar_count)
        >= max(next_train_bar, int(strategy._rolling_train_window))
    )


def consume_training_trigger(strategy: Any) -> None:
    """消费一次训练触发并推进下一窗口."""
    if strategy._rolling_step <= 0:
        return

    current_bar = int(strategy._bar_count)
    validation_config = _get_validation_config(strategy)
    if validation_config is None:
        # 无 validation_config 分支用的是 should_trigger_training 里的阈值
        # 语义, 状态推进必须在这里完成、且必须在 return 之前——这个分支永远
        # 不会执行到下面 validation_config 专属的窗口 bookkeeping, 若状态
        # 更新只放在那之后, _last_train_bar_count 会永远停在 0, 阈值判断
        # 就退化成"每个 bar 都触发"。
        strategy._last_train_bar_count = current_bar
        return

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


#: `get_history`/`get_history_multi` 在双流(同一 symbol 同时存在 bar 与 tick
#: 两条历史序列)下省略 freq 时的歧义报错文案片段(见 src/context.rs 的
#: resolve_use_tick_history)。用它而不是异常类型来识别"是不是这个特定错误",
#: 因为 ValueError 本身太泛, 不能直接当成"双流歧义"的信号。
_DUAL_STREAM_AMBIGUITY_MARKER = "两条历史序列"


def _fetch_rolling_data_for_training(strategy: Any) -> tuple[Any, Any]:
    """取滚动训练用的 (X_df, y), 双流下自动退回 freq='bar'.

    默认按 ``freq=None`` 取一次: 单流下(只有 bar 或只有 tick)行为与此前完全
    一致——只有 bar 就取 bar, 只有 tick 就取 tick(退化 OHLC, 此前就是这个
    行为, 不新增也不修复)。

    只有当该 symbol 同时存在 bar 与 tick 两条历史序列(双流, on_bar 与
    on_tick 同时触发)导致 ``get_rolling_data()`` 按其既定歧义规则报错时,
    才捕获这个特定错误并显式退回 ``freq='bar'`` 重试一次: ML 训练需要真实
    OHLCV(``get_rolling_data`` 固定取 open/high/low/close/volume 五个字段),
    而 tick 序列没有真实 open/high/low(``freq='tick'`` 在这五个字段下必然
    报错, 见 ``_validate_field_for_freq``), 双流下能选的只有 bar。

    选"先按默认尝试、只在歧义报错时才退回 bar"而不是无条件传 ``freq='bar'``,
    是为了对纯 tick 单流场景零风险: 纯 tick 下 freq=None 本就直接命中 tick
    序列(has_bar_history 为 False, 不会走到下面的 except 分支), 不会被这次
    修复改变行为。实测确认: 纯 tick 单流下 ML 自动训练此前从未真正跑起来过
    (训练触发按 ``_bar_count`` 计数, 而 ``_bar_count`` 只在 on_bar 事件里
    才递增, 纯 tick feed 没有任何 bar 事件, ``_bar_count`` 恒为 0, 触发条件
    永远不满足)——所以这条路径本来就是死代码, 但仍选这个更保守的写法,
    是为了同时覆盖一个理论边界: 多 symbol 策略里 ``get_rolling_data()``
    默认解析到的 symbol 可能与驱动 ``_bar_count`` 的 bar 事件来自不同 symbol,
    那种情况下该 symbol 也可能是"纯 tick", 无条件传 ``freq='bar'`` 会让它从
    退化 OHLC 变成全 NaN——按这个写法则完全不受影响。
    """
    try:
        return cast("tuple[Any, Any]", strategy.get_rolling_data())
    except ValueError as e:
        if _DUAL_STREAM_AMBIGUITY_MARKER not in str(e):
            raise
        logger.info(
            "ML 自动训练: symbol 同时存在 bar 与 tick 两条历史序列(双流), "
            "已自动改用 freq='bar' 取真实 OHLCV(tick 无法提供有效 "
            "open/high/low, 无法用于训练)"
        )
        return cast("tuple[Any, Any]", strategy.get_rolling_data(freq="bar"))


def on_train_signal(strategy: Any, context: Any) -> None:
    """滚动训练信号回调."""
    if strategy.model:
        try:
            X_df, _ = _fetch_rolling_data_for_training(strategy)
        except Exception as e:
            # 取数/配置类错误(双流歧义之外的其他 ValueError、rolling window
            # 长度非法等): 用 ERROR 而不是 WARNING, 且把后果说破——此前这里
            # 与训练本身失败合用同一个 WARNING 分支, 双流歧义错误被这个分支
            # 悄悄降级成一行 WARNING, 回测照常跑完、照常出报告, 用户很容易
            # 以为训练正常发生了。这里刻意仍不无条件 re-raise 到调用方(不改变
            # "一次训练窗口的问题不应打断整场回测"的既有容错预期), 但必须让
            # "本次训练被跳过"这件事足够显眼。
            logger.error(
                "ML 自动训练取数失败, 本次训练窗口已跳过(bar=%s 的 ML 训练未生效): %s",
                strategy._bar_count,
                e,
            )
            return

        try:
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
