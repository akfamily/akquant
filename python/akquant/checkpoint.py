import os
import pickle
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


def save_snapshot(engine: Engine, strategy: Any, filepath: str) -> None:
    """
    保存当前运行状态到文件.

    :param engine: akquant.Engine 实例
    :param strategy: 策略实例
    :param filepath: 保存路径 (包含文件名)
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

    snapshot = {
        "engine_state": engine_bytes,
        "strategy": strategy,
        "strategy_topology": {
            "default_strategy_id": default_strategy_id,
            "slot_ids": slot_ids,
        },
        "snapshot_features": {
            "history_buffer_snapshot": True,
        },
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
    logger.info("Snapshot saved to %s", filepath)


def warm_start(
    filepath: str, data_feed: Optional[DataFeed] = None
) -> Tuple[Engine, Any]:
    """
    从快照恢复并热启动.

    :param filepath: 快照文件路径
    :param data_feed: 新的数据源 (可选). 如果为 None，则引擎无数据源，需后续手动添加.
    :return: (engine, strategy)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Snapshot file not found: {filepath}")

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
    # print(f"Warm start loaded from {filepath}. Snapshot time: {engine.snapshot_time}")

    return engine, strategy
