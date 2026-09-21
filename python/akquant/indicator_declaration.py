"""声明式指标层: 声明体与绑定代理 (Internal).

``Strategy.I()`` 的承载模块。一次声明同时携带**计算**与**绘图**两组信息,
框架在每次行情回调前自动推进指标、记录历史值并按需上报绘图点。

设计契约见 ``docs/zh/meta/indicator-tradingview-rfc.md``:

- ``ind[0]`` 是当前 bar 的值(与 Pine 的 ``sma[0]`` 等价于 ``sma`` 一致),
  ``ind[n]`` 是前 n 根; 越界返回 ``None`` 而非抛异常——预热期天然越界。
- 回溯深度由 ``lookback`` 界定, 底层 ``deque(maxlen=lookback)`` **有界**:
  实盘是无限流, 无界缓冲必然泄漏。
"""

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Optional, Tuple, cast

from .akquant import clone_indicator as _clone_rust_indicator

#: 标签未知(基础周期不可解析)时 ``set_provisional`` 记的占位标签; 不发流事件。
UNKNOWN_LABEL = -1

#: 回溯缓冲默认深度。足够绝大多数形态判断(交叉、背离、N 根新高)。
DEFAULT_LOOKBACK = 128


def _normalize_value(value: Any) -> Any:
    """NaN 归一成 None: 两条路的"未就绪"对用户必须长一个样."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def peek_indicator(indicator: Any, args: Tuple[Any, ...]) -> Any:
    """试算不提交: 在副本上 ``update(*args)`` 并返回其 ``value``, 原对象不变.

    内建 Rust 指标经 :func:`akquant.clone_indicator` 复制(pyo3 不为 ``#[derive(Clone)]``
    暴露 ``__copy__``, ``copy`` / ``deepcopy`` / ``pickle`` 对它们一律 TypeError);
    用户自写的 Python 指标走 ``copy.deepcopy``。

    :param indicator: 有 ``update()`` 与 ``value`` 的增量指标实例
    :param args: 喂给 ``update`` 的位置参数(与 ``input_mode`` 口径一致)
    :return: 副本更新后的取值, 未就绪为 ``None``
    """
    try:
        clone = _clone_rust_indicator(indicator)
    except TypeError:
        clone = copy.deepcopy(indicator)
    clone.update(*args)
    return _normalize_value(getattr(clone, "value", None))


def end_stamp_label(timestamp_ns: int, interval_ns: int) -> int:
    """形成中基础 bar 将来闭合时的时间戳(区间末 -1ns).

    与 Rust ``BarAggregator`` 的 ``stamp_bar_at_interval_end`` 打戳公式逐字一致
    (``(min + 1) * interval - 1``)。双流(tick 与 bar 同时到)下 bar **必须**打区间
    末戳——否则与 tick 混推时间戳会倒退——所以凡是有 tick 可试算的会话, 这条
    公式就是闭合 bar 的真实时间戳; 跨午休/跨日也成立(按绝对时间取整, 不依赖
    相邻 bar 的间距)。运行期另有一道自校验: bar 真闭合时若与预测不符, 该
    symbol 停发临时点(见 ``Strategy._update_incremental_indicators``)。
    """
    step = int(interval_ns)
    return (int(timestamp_ns) // step + 1) * step - 1


@dataclass
class PartialBar:
    """从 tick 累出的、正在形成的基础 bar.

    形如 ``Bar``(open/high/low/close/volume/timestamp/symbol), 可直接交给
    ``Strategy._build_incremental_indicator_args`` 按 ``input_mode`` 取字段——
    这也让 H/L 类指标(ATR 等)在 tick 上有了真实的最高/最低价可试算, 而不是
    像纯 tick 路径那样只能跳过。
    """

    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_tick(cls, tick: Any) -> "PartialBar":
        """以第一笔 tick 开出一根形成中 bar."""
        price = float(tick.price)
        return cls(
            symbol=str(tick.symbol),
            timestamp=int(tick.timestamp),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=float(tick.volume),
        )

    def merge(self, tick: Any) -> None:
        """并入一笔 tick: 刷新 H/L/C, 累加单笔量."""
        price = float(tick.price)
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += float(tick.volume)
        self.timestamp = int(tick.timestamp)


def is_reportable_value(value: Any) -> bool:
    """未就绪的取值不上报: Pine 预热期是 ``na``, 不画.

    ``None`` 与 ``NaN`` 都算未就绪——原生 Rust 指标未满窗返回 ``None``,
    而用户自写指标惯用 ``float("nan")``(见 examples/60)。
    """
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


@dataclass
class IndicatorDeclaration:
    """一条 ``Strategy.I()`` 声明: 计算参数 + 绘图元数据 + 运行期状态."""

    # --- 计算 ---
    source: str = "close"
    symbols: Optional[set[str]] = None
    input_mode: str = "source"
    warmup_bars: int = 0
    factory: Optional[Callable[[], Any]] = None
    base_indicator: Any = None
    #: 该指标由哪个周期驱动: None=基础行情(bar/tick), 否则是 subscribe_bars 的
    #: 周期标签(如 "5min")。见 ``Strategy._update_incremental_indicators``。
    freq: Optional[str] = None
    #: True 表示向量化预计算指标(``indicator.py::Indicator``): 数据加载后整段
    #: 算完缓存, 运行期按时间戳 asof 查询, 而非逐 bar ``update()``。
    precomputed: bool = False

    # --- 序列回溯 ---
    lookback: int = DEFAULT_LOOKBACK
    #: symbol -> 最近 lookback 个**已确认**取值(含未就绪时的 None)
    history: Dict[str, Deque[Any]] = field(default_factory=dict)

    # --- 实时(intrabar) ---
    #: True 表示在 bar 未闭合期间试算临时值: 窗口周期指标(``freq`` 非 None)在
    #: 每根基础 bar 上用未闭合窗口快照试算; 基础周期指标(``freq`` 为 None)在
    #: 每笔 tick 上用形成中的 :class:`PartialBar` 试算(此时 tick 不再 ``update``)。
    intrabar: bool = False
    #: symbol -> 当前未确认的临时值。有临时值时 ``[0]`` 是它, ``[1]`` 是
    #: ``history[-1]`` —— 与 Pine 里 realtime bar 占据 index 0 一致。
    provisional: Dict[str, Any] = field(default_factory=dict)
    #: symbol -> 临时值所属窗口的标签(未闭合快照的 timestamp)。用来识别
    #: "窗口已滚动但确认值尚未入账"这一帧, 见 ``Strategy._peek_intrabar_indicators``。
    provisional_label: Dict[str, int] = field(default_factory=dict)

    # --- 绘图 ---
    plot: bool = False
    pane: int = 0
    render_type: str = "line"
    color: Optional[str] = None
    label: Optional[str] = None
    unit: Optional[str] = None
    precision: Optional[int] = None
    reference_lines: Optional[list[Dict[str, Any]]] = None
    scale_group: Optional[str] = None
    #: 多值指标的分量名。声明 outputs=("dif","dea","hist") 会让 MACD 产出三条
    #: 独立的线(indicator_key 形如 "macd.dif"), 而不是把元组丢给前端去拆。
    outputs: Optional[Tuple[str, ...]] = None

    # --- 运行期状态 ---
    primary_symbol: Optional[str] = None
    instances: Dict[str, Any] = field(default_factory=dict)
    # 下面两个集合只用于 H/L 类 input_mode 的会话级覆盖率核验(见
    # ``Strategy._check_incremental_hl_bar_coverage``): bar_seen_symbols 记录
    # 曾经收到过至少一个 bar 的 symbol; tick_only_symbols 记录曾经在
    # H/L 模式下被 tick 跳过更新的 symbol。会话结束时两者做差集, 差集非空
    # 说明该 symbol 全程只有 tick、从未有 bar, 需要报错而非静默不推进。
    bar_seen_symbols: set[str] = field(default_factory=set)
    tick_only_symbols: set[str] = field(default_factory=set)
    #: 曾在 tick 上做过 intrabar 试算的 symbol。会话结束时与 bar_seen_symbols
    #: 做差: 从未收到 bar 的 symbol 永远只试算不确认, 是配置错误(纯 tick 会话
    #: 开了 intrabar), 要报错而非静默。
    intrabar_symbols: set[str] = field(default_factory=set)

    def push_value(self, symbol: str, value: Any) -> None:
        """把一次更新后的取值写入该 symbol 的回溯缓冲.

        :raises ValueError: 声明了 ``outputs`` 但个数与指标实际分量数不符
        """
        self._check_outputs_arity(value)
        value = _normalize_value(value)
        buffer = self.history.get(symbol)
        if buffer is None:
            buffer = deque(maxlen=max(1, int(self.lookback)))
            self.history[symbol] = buffer
        buffer.append(value)

    def value_at(self, symbol: str, offset: int) -> Any:
        """取该 symbol 前 ``offset`` 根的取值; 越界返回 ``None``.

        有临时值时它占据 ``[0]``, 已确认序列整体后移一位(``[1]`` 是最近一根
        已确认)——这正是 Pine 里 realtime bar 的下标语义。
        """
        index = int(offset)
        if index < 0:
            return None
        if symbol in self.provisional:
            if index == 0:
                return self.provisional[symbol]
            index -= 1
        buffer = self.history.get(symbol)
        if buffer is None or index >= len(buffer):
            return None
        return buffer[-1 - index]

    def set_provisional(self, symbol: str, value: Any, label: int) -> None:
        """记下该 symbol 在未闭合窗口(标签 ``label``)上的临时值, 覆盖上一笔."""
        self._check_outputs_arity(value)
        self.provisional[symbol] = _normalize_value(value)
        self.provisional_label[symbol] = int(label)

    def pending_confirmation(self, symbol: str, label: int) -> bool:
        """快照已滚到新窗口, 但上一窗口的确认值还没入账.

        引擎在每根基础 bar 的 ``on_bar`` **之后**才派发闭合的窗口 bar; 在闭合那根
        基础 bar 上 ``current_window()`` 已是下一个窗口的快照。此时若照常试算, 副本
        里缺了刚闭合的那个窗口, 算出来的临时值跳过了一整根 —— 这一帧要跳过, 保留
        上一笔临时值作为即将确认的窗口的最佳估计。
        """
        previous = self.provisional_label.get(symbol)
        return (
            symbol in self.provisional
            and previous is not None
            and previous != int(label)
        )

    def clear_provisional(self, symbol: str) -> None:
        """窗口闭合、真值已入 ``history`` 后清掉临时值."""
        self.provisional.pop(symbol, None)
        self.provisional_label.pop(symbol, None)

    def is_confirmed(self, symbol: str) -> bool:
        """``[0]`` 是否为已确认值(没有挂起的临时值)."""
        return symbol not in self.provisional

    def _check_outputs_arity(self, value: Any) -> None:
        """校验 ``outputs`` 个数与指标实际分量数一致, 不符就 fail-fast.

        与是否绘图无关: 个数不符意味着 ``self.macd.hist`` 这类分量访问会
        静默取到错位或缺失的值, 比不画图严重得多。未就绪(非序列)时跳过
        ——那是预热期的正常状态, 不是配置错误。
        """
        if not self.outputs or not isinstance(value, (tuple, list)):
            return
        if len(value) != len(self.outputs):
            raise ValueError(
                f"outputs={self.outputs!r} 声明了 {len(self.outputs)} 个分量, "
                f"但指标实际产出 {len(value)} 个。请让两者个数一致"
            )

    def component_index(self, attr: str) -> Optional[int]:
        """分量名在 ``outputs`` 里的下标; 不是分量名则返回 ``None``."""
        if not self.outputs:
            return None
        try:
            return self.outputs.index(attr)
        except ValueError:
            return None

    def report_items(self, name: str, value: Any) -> list[Tuple[str, str, Any]]:
        """把一次取值摊平成待上报的 ``(indicator_key, display_name, value)``.

        多值指标按 ``outputs`` 展开成多条独立的线(``macd.dif`` 等), 前端的
        渲染单元是"一条线", 让它去解析元组等于把契约复杂度外推给消费者。

        :raises ValueError: ``outputs`` 个数与指标实际分量数不符
        """
        if not self.outputs:
            return [(name, self.label or name, value)]
        if not isinstance(value, (tuple, list)):
            # 未就绪时原生指标返回 None, 不是长度不符, 交给调用方过滤。
            return [(f"{name}.{out}", f"{name}.{out}", None) for out in self.outputs]
        self._check_outputs_arity(value)
        return [
            (f"{name}.{out}", f"{name}.{out}", item)
            for out, item in zip(self.outputs, value)
        ]


class IndicatorBinding:
    """按当前 symbol 访问底层指标实例与回溯缓冲的轻量代理.

    未知属性一律代理到底层指标实例, 所以 ``aq.SMA`` 之类的原生属性
    (``is_ready`` 等)照常可用。
    """

    def __init__(self, strategy: Any, name: str) -> None:
        """绑定策略实例与指标名, 延迟解析当前 symbol 对应的真实指标."""
        self._strategy = strategy
        self._name = name

    def get_instance(self, symbol: Optional[str] = None) -> Any:
        """返回指定 symbol 对应的底层指标实例."""
        return self._strategy._get_incremental_indicator_instance(self._name, symbol)

    def _declaration(self) -> IndicatorDeclaration:
        declaration = self._strategy._get_incremental_registration(self._name)
        return cast(IndicatorDeclaration, declaration)

    def _resolved_symbol(self, symbol: Optional[str] = None) -> str:
        declaration = self._declaration()
        resolved = self._strategy._resolve_incremental_indicator_symbol(
            declaration, symbol
        )
        return str(resolved)

    def __getitem__(self, offset: int) -> Any:
        """``ind[0]`` 当前值, ``ind[n]`` 前 n 根; 越界返回 ``None``."""
        return self._declaration().value_at(self._resolved_symbol(), offset)

    def __len__(self) -> int:
        """当前 symbol 已积累的回溯长度."""
        buffer = self._declaration().history.get(self._resolved_symbol())
        return 0 if buffer is None else len(buffer)

    @property
    def value(self) -> Any:
        """当前 symbol 的最新取值 —— ``[0]`` 的别名."""
        return self[0]

    @property
    def is_ready(self) -> bool:
        """当前 symbol 的 ready 状态."""
        return bool(getattr(self.get_instance(), "is_ready", False))

    @property
    def confirmed(self) -> bool:
        """``[0]`` 是否为已确认值; 未开 ``intrabar`` 时恒 True."""
        return self._declaration().is_confirmed(self._resolved_symbol())

    def update(self, *args: Any, **kwargs: Any) -> Any:
        """兼容手动调用 update 的现有写法."""
        return self.get_instance().update(*args, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        """分量名返回子绑定, 其余属性代理到底层当前 symbol 的指标实例."""
        # `_strategy`/`_name` 在 __init__ 里赋值前被访问(如 copy/pickle 探测)
        # 会无限递归, 这里直接拒绝下划线开头的名字。
        if attr.startswith("_"):
            raise AttributeError(attr)
        index = self._declaration().component_index(attr)
        if index is not None:
            return IndicatorComponentBinding(self, attr, index)
        return getattr(self.get_instance(), attr)


class IndicatorComponentBinding:
    """多值指标的单个分量视图: ``self.macd.dif[0]``.

    自身不持有数据, 每次从父绑定的元组取值里切出对应下标——保证分量与整体
    永远一致, 不会出现两套各自累积的缓冲漂移。
    """

    def __init__(self, parent: IndicatorBinding, name: str, index: int) -> None:
        """绑定父指标与分量下标."""
        self._parent = parent
        self._name = name
        self._index = index

    def __getitem__(self, offset: int) -> Any:
        """``comp[0]`` 当前值, ``comp[n]`` 前 n 根; 未就绪/越界返回 ``None``."""
        whole = self._parent[offset]
        if not isinstance(whole, (tuple, list)) or self._index >= len(whole):
            return None
        return whole[self._index]

    def __len__(self) -> int:
        """回溯长度与父指标一致."""
        return len(self._parent)

    @property
    def value(self) -> Any:
        """当前取值 —— ``[0]`` 的别名."""
        return self[0]
