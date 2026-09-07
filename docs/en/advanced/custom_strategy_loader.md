# Custom Strategy Loader

This page explains how to register and use a custom strategy loader in `akquant`, so strategy implementations can be loaded dynamically at runtime instead of being statically imported in your script.

## 1. When to use

- Strategies are distributed as encrypted artifacts and must be decrypted at load time
- Strategy source lives in a database, object storage, or a remote service rather than a local `.py` file
- Extra steps are required before loading (permission checks, signature verification, version selection)
- Injecting a mock strategy in test environments

## 2. Built-in loaders

| Name | Purpose |
| --- | --- |
| `python_plain` | Load from a plaintext `.py` file (default) |
| `encrypted_external` | Delegate to a `decrypt_and_load` callback that returns the strategy class |

Both `run_backtest` and `run_live` accept these three parameters:

```python
import akquant as aq

result = aq.run_backtest(
    data=bars,
    strategy_source="my_strategy.py",
    strategy_loader="python_plain",
    strategy_loader_options={"strategy_attr": "MyStrategy"},
    symbols="000001.SZ",
)
```

## 3. Core API

Importable from the `akquant` top level:

- `register_strategy_loader(name, loader)`
- `get_strategy_loader(name)`
- `register_plugin_strategy_loaders()`
- `resolve_strategy_input(...)`

The entry-point group constant `ENTRY_POINT_GROUP` lives in the `akquant.strategy_loader` submodule (consistent with the broker and signal-source plugin mechanisms, which also keep their group names out of the top level). Plugin authors just write the group name into `pyproject.toml` as shown in section 6; there is no need to read it at runtime.

## 4. Loader signature

```python
from typing import Any, Dict

def my_loader(source: Any, options: Dict[str, Any]) -> Any:
    """Return a Strategy subclass, a Strategy instance, or a (self, bar) callable."""
    ...
```

- `source` is whatever the caller passed as `strategy_source` (path string, `bytes`, or `PathLike`)
- `options` is the `strategy_loader_options` dict
- The return value must be a Strategy type/instance/callable, otherwise the framework raises `TypeError`

Exceptions raised inside a loader **propagate unchanged** — the framework does not swallow them. This lets a loader use exceptions to express "refuse to load".

## 5. Manual registration

```python
import akquant as aq

aq.register_strategy_loader("my_loader", my_loader)

result = aq.run_backtest(
    data=bars,
    strategy_source="payload.bin",
    strategy_loader="my_loader",
    symbols="000001.SZ",
)
```

## 6. Automatic registration via entry-point

Third-party or private packages can declare an `akquant.strategy_loaders` entry-point, so users **do not need to import the plugin package** in their scripts:

```toml
# plugin repo's pyproject.toml
[project.entry-points."akquant.strategy_loaders"]
my_loader = "my_pkg.loader:register"
```

```python
# my_pkg/loader.py
from akquant import register_strategy_loader

def register() -> None:
    """Called by akquant the first time a loader is resolved by name."""
    register_strategy_loader("my_loader", my_loader)
```

Once the plugin package is installed, just use it by name:

```python
result = aq.run_backtest(
    data=bars,
    strategy_source="payload.bin",
    strategy_loader="my_loader",
    symbols="000001.SZ",
)
```

The mechanism is fully symmetric with broker plugins (`akquant.brokers`) and signal source plugins (`akquant.signal_sources`).

### Discovery timing

Plugin discovery is **lazy**: it is triggered the first time `get_strategy_loader()` looks up a name, not during `import akquant`. This avoids unconditionally executing third-party code when the framework is imported, and keeps `import akquant` fast.

A single failing plugin is logged at `WARNING` and skipped; it does not affect other plugins or the framework itself.

## 7. Multiprocessing notes

`run_grid_search(max_workers>1)` pickles the strategy class into worker processes, and `pickle` only records `module.qualname`. A dynamically loaded strategy class must therefore satisfy two conditions, or workers cannot restore it:

1. **Stable module name** — loading the same source repeatedly yields the same module name
2. **Registered in `sys.modules`**, and importable by that name from a worker process

The built-in `python_plain` already satisfies both: it derives a stable, reversible module name from the source file path, and installs a `sys.meta_path` finder at `import akquant` time so workers can recover the source file from the module name.

A custom loader that also needs multiprocessing support must provide an equivalent mechanism. Mind the timing: a `spawn`-started worker restores the module while deserializing the task queue, before any user code runs, so the finder must be installed at module import time.

## 8. See also

- [Custom Broker Registry](custom_broker_registry.md)
- [Signal Ingestion](signal_ingestion.md)
- [Runtime Config Guide](runtime_config.md)
