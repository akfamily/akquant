from dataclasses import dataclass, field
from typing import Any, Protocol

from .log import get_logger

_logger = get_logger("analyzer")


class AnalyzerPlugin(Protocol):
    """Protocol for pluggable analyzers."""

    name: str

    def on_start(self, context: dict[str, Any]) -> None:
        """Handle backtest start event."""
        ...

    def on_bar(self, context: dict[str, Any]) -> None:
        """Handle bar event."""
        ...

    def on_trade(self, context: dict[str, Any]) -> None:
        """Handle trade event."""
        ...

    def on_finish(self, context: dict[str, Any]) -> dict[str, Any]:
        """Return final analyzer output."""
        ...


@dataclass
class AnalyzerManager:
    """Dispatch lifecycle events to registered analyzers.

    每个插件单独隔离: 一个插件的回调抛异常, 不影响其他插件收到同一次回调,
    也不中断回测。每个插件的每个回调只记第一次错误(带 traceback), 之后的
    同类错误只计数, 在 ``on_finish`` 时汇总一条, 避免按 bar 刷屏。
    ``on_finish`` 出错的插件不进输出, 其他插件的输出照常返回。
    """

    plugins: list[AnalyzerPlugin] = field(default_factory=list)
    _suppressed: dict[tuple[str, str], int] = field(
        default_factory=dict, init=False, repr=False
    )

    def register(self, plugin: AnalyzerPlugin) -> None:
        """Register a new analyzer plugin."""
        self.plugins.append(plugin)

    def _call(
        self, plugin: AnalyzerPlugin, method: str, context: dict[str, Any]
    ) -> tuple[bool, Any]:
        """调用一个插件的一个回调; 返回 (是否成功, 返回值)."""
        try:
            return True, getattr(plugin, method)(context)
        except Exception:
            key = (str(getattr(plugin, "name", plugin)), method)
            if key in self._suppressed:
                self._suppressed[key] += 1
            else:
                self._suppressed[key] = 0
                _logger.exception(
                    "Analyzer 插件 %r 的 %s 抛出异常, 已跳过该插件的这次回调"
                    "(其他插件不受影响; 同类错误之后只计数)",
                    key[0],
                    method,
                )
            return False, None

    def _dispatch(self, method: str, context: dict[str, Any]) -> None:
        for plugin in self.plugins:
            self._call(plugin, method, context)

    def on_start(self, context: dict[str, Any]) -> None:
        """Broadcast start event to all analyzers."""
        self._dispatch("on_start", context)

    def on_bar(self, context: dict[str, Any]) -> None:
        """Broadcast bar event to all analyzers."""
        self._dispatch("on_bar", context)

    def on_trade(self, context: dict[str, Any]) -> None:
        """Broadcast trade event to all analyzers."""
        self._dispatch("on_trade", context)

    def on_finish(self, context: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Collect analyzer outputs at the end of backtest."""
        result: dict[str, dict[str, Any]] = {}
        for plugin in self.plugins:
            ok, output = self._call(plugin, "on_finish", context)
            if ok:
                result[plugin.name] = output
        for (name, method), count in self._suppressed.items():
            if count > 0:
                _logger.error(
                    "Analyzer 插件 %r 的 %s 另有 %d 次异常已抑制(只记录了第一次)",
                    name,
                    method,
                    count,
                )
        return result


@dataclass
class AnalyzerTemplate:
    """Minimal analyzer template for plugin authors."""

    name: str = "template"
    seen_trades: int = 0

    def on_start(self, context: dict[str, Any]) -> None:
        """Handle start event."""
        _ = context

    def on_bar(self, context: dict[str, Any]) -> None:
        """Handle bar event."""
        _ = context

    def on_trade(self, context: dict[str, Any]) -> None:
        """Handle trade event."""
        _ = context
        self.seen_trades += 1

    def on_finish(self, context: dict[str, Any]) -> dict[str, Any]:
        """Return template summary output."""
        _ = context
        return {"seen_trades": self.seen_trades}
