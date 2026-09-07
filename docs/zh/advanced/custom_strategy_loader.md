# 自定义策略加载器

本页说明如何为 `akquant` 注册并使用自定义策略加载器，让策略实现按运行时配置动态加载，而不是在脚本里静态 `import`。

## 1. 适用场景

- 策略以加密产物分发，加载时需要先解密
- 策略源码存放在数据库、对象存储或远端服务，而非本地 `.py` 文件
- 加载前需要执行额外步骤（如权限校验、签名校验、版本选择）
- 测试环境下注入 mock 策略做联调

## 2. 内置加载器

| 名称 | 用途 |
| --- | --- |
| `python_plain` | 从明文 `.py` 文件加载（默认） |
| `encrypted_external` | 由外部回调 `decrypt_and_load` 解密并返回策略类 |

`run_backtest` 与 `run_live` 都接受这三个参数：

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

## 3. 核心 API

从 `akquant` 顶层可导入：

- `register_strategy_loader(name, loader)`
- `get_strategy_loader(name)`
- `register_plugin_strategy_loaders()`
- `resolve_strategy_input(...)`

entry-point 组名常量 `ENTRY_POINT_GROUP` 定义在 `akquant.strategy_loader` 子模块里（与 broker、信号源两个插件机制一致，组名不提升到顶层）。插件作者照下文第 6 节把组名写进 `pyproject.toml` 即可，运行期无需读取它。

## 4. 加载器签名

```python
from typing import Any, Dict

def my_loader(source: Any, options: Dict[str, Any]) -> Any:
    """返回 Strategy 子类、Strategy 实例，或 (self, bar) 形态的可调用对象."""
    ...
```

- `source` 即调用方传入的 `strategy_source`（路径字符串、`bytes` 或 `PathLike`）
- `options` 即 `strategy_loader_options` 字典
- 返回值必须是 Strategy 类型/实例/可调用对象，否则框架会报 `TypeError`

加载器内抛出的异常会**原样向上传播**，不会被框架吞掉——这使加载器可以用异常表达"拒绝加载"。

## 5. 手动注册

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

## 6. 通过 entry-point 自动注册

第三方或私有包可声明 `akquant.strategy_loaders` entry-point，使用者**无需在脚本里 import 插件包**：

```toml
# 插件仓的 pyproject.toml
[project.entry-points."akquant.strategy_loaders"]
my_loader = "my_pkg.loader:register"
```

```python
# my_pkg/loader.py
from akquant import register_strategy_loader

def register() -> None:
    """由 akquant 在首次按名解析加载器时自动调用."""
    register_strategy_loader("my_loader", my_loader)
```

装上插件包后直接按名使用即可：

```python
result = aq.run_backtest(
    data=bars,
    strategy_source="payload.bin",
    strategy_loader="my_loader",
    symbols="000001.SZ",
)
```

机制与 broker 插件（`akquant.brokers`）、信号源插件（`akquant.signal_sources`）完全对称。

### 发现时机

插件发现是**惰性**的：由 `get_strategy_loader()` 首次按名查表时触发，而非 `import akquant` 期间。这样既不会在导入框架时无条件执行第三方代码，也不会拖慢 `import akquant`。

单个插件加载失败只记 `WARNING` 并跳过，不影响其余插件与框架本身。

## 7. 多进程注意事项

`run_grid_search(max_workers>1)` 会把策略类 `pickle` 到子进程，而 `pickle` 只记录 `module.qualname`。因此动态加载的策略类必须满足两点，否则子进程无法还原：

1. **模块名稳定**——同一来源重复加载得到同一模块名
2. **模块已登记进 `sys.modules`**，且子进程能按该名字重新导入

内置 `python_plain` 已经满足：它按源文件路径生成稳定且可逆的模块名，并在 `import akquant` 时安装 `sys.meta_path` finder，使子进程能按模块名找回源文件。

自定义加载器若也要支持多进程，需自行提供等价机制。注意时机：`spawn` 启动的子进程在反序列化任务队列时就要还原该模块，早于任何用户代码执行，因此 finder 必须在模块导入期安装。

## 8. 相关页面

- [自定义 Broker 注册](custom_broker_registry.md)
- [信号接入](signal_ingestion.md)
- [运行时配置指南](runtime_config.md)
