#!/usr/bin/env bash
set -euo pipefail

uvx maturin develop
uv run --no-sync pre-commit run ruff-check --all-files
uv run --no-sync pre-commit run mypy --all-files
uv run pytest tests/test_engine.py -k "engine_oco_avoids_same_batch_double_fill"
uv run pytest tests/test_engine.py -k "engine_bracket_activates_exit_orders"
uv run pytest tests/test_strategy_extras.py -k "oco_group_prefers_engine_registration_when_available or oco_group_falls_back_to_deferred_engine_queue_on_runtime_error"
uv run pytest tests/test_strategy_extras.py -k "bracket_prefers_engine_registration_when_available or bracket_falls_back_to_deferred_engine_queue_on_runtime_error"
