import os
import pickle
import warnings
from importlib import metadata
from typing import Any, Optional, Tuple

from .akquant import DataFeed, Engine
from .log import get_logger

logger = get_logger("checkpoint")

try:
    _VERSION = metadata.version("akquant")
except metadata.PackageNotFoundError:
    _VERSION = "0.0.0+unknown"


def _iter_strategy_graph(root_strategy: Any) -> list[Any]:
    """Collect primary and slot strategies for transient attr stripping."""
    stack = [root_strategy]
    collected: list[Any] = []
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        collected.append(current)
        raw_slots = getattr(current, "_slot_strategies", None)
        if isinstance(raw_slots, dict):
            for slot_strategy in raw_slots.values():
                stack.append(slot_strategy)
    return collected


def _extract_resolved_config(config: Any) -> Optional[dict[str, Any]]:
    """Normalize the save_checkpoint config argument into a plain dict or None.

    Accepts a BacktestResult (reads .resolved_config), a plain dict, or None.
    """
    if config is None:
        return None
    resolved = getattr(config, "resolved_config", None)
    if isinstance(resolved, dict):
        return dict(resolved)
    if isinstance(config, dict):
        return dict(config)
    return None


def save_checkpoint(
    engine: Engine,
    strategy: Any,
    filepath: str,
    *,
    config: Any = None,
) -> None:
    """
    保存当前运行状态到检查点文件 (checkpoint).

    :param engine: akquant.Engine 实例
    :param strategy: 策略实例
    :param filepath: 保存路径 (包含文件名)
    :param config: 可选。已解析运行时配置来源，用于热启动自动继承滑点/费率/
        fill_policy 等 (issue #282)。可传 BacktestResult (自动读取
        result.resolved_config) 或直接传配置 dict。为 None 时快照不含配置，
        热启动降级到逐项推导 (向后兼容)。
    """
    # 1. Get binary state from Rust Engine
    # Note: engine.get_state_bytes returns bytes directly in Python
    engine_bytes = engine.get_state_bytes()  # type: ignore[attr-defined]

    default_strategy_id = str(getattr(strategy, "_owner_strategy_id", "") or "").strip()
    if (
        not default_strategy_id
        and hasattr(engine, "get_default_strategy_id")
        and callable(getattr(engine, "get_default_strategy_id"))
    ):
        default_strategy_id = str(engine.get_default_strategy_id() or "").strip()
    if not default_strategy_id:
        default_strategy_id = "_default"

    slot_ids: list[str] = []
    slot_fetcher = None
    if hasattr(engine, "get_strategy_slot_ids") and callable(
        getattr(engine, "get_strategy_slot_ids")
    ):
        slot_fetcher = engine.get_strategy_slot_ids
    elif hasattr(engine, "get_strategy_slots") and callable(
        getattr(engine, "get_strategy_slots")
    ):
        slot_fetcher = engine.get_strategy_slots
    if slot_fetcher is not None:
        try:
            slot_ids = [
                str(slot_id).strip()
                for slot_id in slot_fetcher()
                if str(slot_id).strip()
            ]
        except Exception:
            slot_ids = []
    if not slot_ids:
        raw_slot_ids = getattr(strategy, "_strategy_slot_ids", [])
        if isinstance(raw_slot_ids, list):
            slot_ids = [
                str(slot_id).strip() for slot_id in raw_slot_ids if str(slot_id).strip()
            ]
    if default_strategy_id not in slot_ids:
        slot_ids.insert(0, default_strategy_id)

    backtest_config = _extract_resolved_config(config)

    snapshot = {
        "engine_state": engine_bytes,
        "strategy": strategy,
        "strategy_topology": {
            "default_strategy_id": default_strategy_id,
            "slot_ids": slot_ids,
        },
        "snapshot_features": {
            "history_buffer_snapshot": True,
            # 标记该存档由「已拆分 tick/bar 历史序列」的版本生成 (tick/bar dual
            # stream plan)。旧存档 (无此标记) 恢复时, Rust 侧的
            # `From<HistoryBufferSnapshot> for HistoryBuffer` 会把旧的共享
            # `data` 容器整体当作 bar 序列装载, 而 `tick_data` 因
            # `#[serde(default)]` 留空——见 load_checkpoint 里的告警。
            "history_tick_split": True,
        },
        "backtest_config": backtest_config,
        "version": _VERSION,
    }

    transient_backups: list[tuple[Any, dict[str, Any]]] = []
    for current_strategy in _iter_strategy_graph(strategy):
        attr_backup: dict[str, Any] = {}
        for attr_name in ("_engine", "_analyzer_manager"):
            if hasattr(current_strategy, attr_name):
                attr_backup[attr_name] = getattr(current_strategy, attr_name)
                delattr(current_strategy, attr_name)
        transient_backups.append((current_strategy, attr_backup))
    try:
        with open(filepath, "wb") as f:
            pickle.dump(snapshot, f)
    finally:
        for current_strategy, attr_backup in transient_backups:
            for attr_name, attr_value in attr_backup.items():
                setattr(current_strategy, attr_name, attr_value)
    logger.info("Checkpoint saved to %s", filepath)


def load_checkpoint(
    filepath: str, data_feed: Optional[DataFeed] = None
) -> Tuple[Engine, Any]:
    """
    从检查点文件恢复引擎与策略状态 (checkpoint restore).

    :param filepath: 检查点文件路径
    :param data_feed: 新的数据源 (可选). 如果为 None，则引擎无数据源，需后续手动添加.
    :return: (engine, strategy)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    with open(filepath, "rb") as f:
        snapshot = pickle.load(f)

    # 1. Restore Strategy
    strategy = snapshot["strategy"]
    topology = snapshot.get("strategy_topology", {})
    if not isinstance(topology, dict):
        topology = {}
    snapshot_features = snapshot.get("snapshot_features", {})
    if not isinstance(snapshot_features, dict):
        snapshot_features = {}

    history_buffer_snapshot_available = bool(
        snapshot_features.get("history_buffer_snapshot", False)
    )
    history_tick_split_available = bool(
        snapshot_features.get("history_tick_split", False)
    )
    if history_buffer_snapshot_available and not history_tick_split_available:
        # 该存档由「history buffer snapshot」功能已上线、但 tick/bar 历史序列
        # 尚未拆分的旧版本生成。当时 bar/tick 共用同一个 `data` 容器；恢复时
        # Rust 侧 `From<HistoryBufferSnapshot> for HistoryBuffer` 会把这个旧
        # `data` 整体当作 bar 序列装载, 新的 `tick_data` 容器留空。若原策略
        # 含 tick 数据, 续跑后同一 symbol 会同时拥有 bar 与 tick 两条历史序
        # 列: 取历史时可能撞“同时存在 bar 与 tick 两条历史序列”的 ValueError；
        # 若显式传 freq='tick' 规避, 又只能拿到续跑后新写入的 tick 数据,
        # 历史窗口会静默变短。不做自动迁移 (旧存档里的数据本就无法区分
        # 哪些是 tick、哪些是 bar), 仅提前告警, 建议用当前版本重新生成检查点。
        warning_message = (
            "Checkpoint '%s' was created before the tick/bar history-buffer "
            "split (snapshot_features missing 'history_tick_split'). If the "
            "original strategy consumed tick data, that data was stored in "
            "the old shared history buffer and will be restored as bar "
            "history; new ticks written after resume land in a separate "
            "tick history. get_history()/get_history_map() may then raise "
            '"two history sequences exist" for that symbol, or (if you '
            "pass freq='tick' to dodge that) silently return a truncated "
            "tick window. Regenerate the checkpoint with the current "
            "AKQuant version to avoid this." % filepath
        )
        warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
        logger.warning(warning_message)

    # 2. Initialize new Engine
    engine = Engine()

    # 3. Set DataFeed if provided
    if data_feed:
        engine.add_data(data_feed)

    # 4. Load Rust State
    # This restores portfolio, orders, and sets snapshot_time
    engine.load_state_bytes(snapshot["engine_state"])  # type: ignore[attr-defined]

    # 5. Restore Strategy reference to Engine if necessary
    # Note: Strategy class in akquant doesn't hold 'engine' reference explicitly.
    # But if user strategy does, they should handle it or we can try.
    if hasattr(strategy, "engine"):
        strategy.engine = engine
    default_strategy_id = str(topology.get("default_strategy_id", "") or "").strip()
    if default_strategy_id:
        setattr(strategy, "_owner_strategy_id", default_strategy_id)
    slot_ids = topology.get("slot_ids", [])
    if isinstance(slot_ids, list):
        normalized_slot_ids = [
            str(slot_id).strip() for slot_id in slot_ids if str(slot_id).strip()
        ]
        if normalized_slot_ids:
            setattr(strategy, "_strategy_slot_ids", normalized_slot_ids)
    setattr(strategy, "_warm_start_snapshot_features", dict(snapshot_features))
    # 携带检查点持久化的运行时配置，供 run_from_checkpoint 自动继承 (issue #282)
    restored_config = snapshot.get("backtest_config")
    setattr(
        strategy,
        "_warm_start_backtest_config",
        dict(restored_config) if isinstance(restored_config, dict) else None,
    )
    # print(f"Warm start loaded from {filepath}. Snapshot time: {engine.snapshot_time}")

    return engine, strategy
