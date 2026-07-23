r"""LWC 复盘 HTML 模板与安全渲染.

单文件自包含:内联 vendored LWC standalone JS + 数据 payload。渲染时:

- 标题经 :func:`html.escape` 转义,防 HTML 注入;
- payload 用 JSON 序列化后把 ``<`` / ``>`` / ``&`` 转成 ``\uXXXX``,
  防止 ``</script>`` 提前闭合 script 标签(XSS);
- 用占位符 ``.replace()`` 注入,避免 ``str.format`` 与 JS/CSS 花括号冲突。
"""

from __future__ import annotations

import html
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

_ASSETS = Path(__file__).parent / "assets"
_LWC_JS = _ASSETS / "lightweight-charts.standalone.production.js"


@lru_cache(maxsize=1)
def _load_lwc_js() -> str:
    """读取 vendored LWC standalone JS(带缓存)."""
    return _LWC_JS.read_text(encoding="utf-8")


def _safe_json(obj: Any) -> str:
    """JSON 序列化并转义可闭合 script 标签的字符,可安全嵌入 <script>."""
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


# 占位符用 %%NAME%% 形式,避开 JS/CSS 的 {}。页面 chrome 用 CSS 变量,
# 明暗切换时只改 <html data-theme>,无需重建 DOM。
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh" data-theme="%%INIT_THEME%%">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>%%TITLE%%</title>
<style>
  :root[data-theme="light"]{--bg:#ffffff;--text:#333333;--grid:#f0f0f0;}
  :root[data-theme="dark"]{--bg:#1e1e1e;--text:#e0e0e0;--grid:#2b2b2b;}
  html,body{margin:0;padding:0;background:var(--bg);color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
  #bar{display:flex;align-items:center;gap:12px;padding:10px 16px;
    border-bottom:1px solid var(--grid);}
  #bar h1{font-size:15px;font-weight:600;margin:0;}
  #sym,#theme-toggle{padding:4px 8px;background:var(--bg);color:var(--text);
    border:1px solid var(--grid);border-radius:4px;font-size:13px;cursor:pointer;}
  #sym{margin-left:auto;cursor:default;}
  #chart{width:100%;height:calc(100vh - 52px);}
  #empty{padding:40px;text-align:center;color:var(--text);opacity:.6;}
</style>
</head>
<body>
<div id="bar">
  <h1>%%TITLE%%</h1>
  <select id="sym" aria-label="标的选择"></select>
  <button id="theme-toggle" type="button" aria-label="切换明暗主题">🌙 暗色</button>
</div>
<div id="chart"></div>
<script>%%LWC_JS%%</script>
<script id="akq-data" type="application/json">%%DATA%%</script>
<script>%%APP_JS%%</script>
</body>
</html>
"""

# 前端逻辑:v5 API(addSeries(SeriesType,...) / createSeriesMarkers)。
# payload 主题无关(volume 带 up 布尔、marker 带 buy 布尔),颜色在此按当前
# 主题动态计算;切换主题只 applyOptions + 重新上色,不重建数据(大数据量友好)。
_APP_JS = """
(function(){
  var LWC = window.LightweightCharts;
  var cfg = JSON.parse(document.getElementById('akq-data').textContent);
  var themes = cfg.themes, all = cfg.symbols || [];
  var cur = cfg.initial_theme in themes ? cfg.initial_theme : 'light';
  var sel = document.getElementById('sym');
  var toggle = document.getElementById('theme-toggle');
  var host = document.getElementById('chart');
  if(!all.length){host.innerHTML='<div id="empty">无可复盘的标的数据</div>';return;}
  all.forEach(function(s,i){
    var o=document.createElement('option');o.value=i;o.textContent=s.symbol;
    sel.appendChild(o);
  });
  function T(){return themes[cur];}
  var chart = LWC.createChart(host, {
    autoSize:true,
    timeScale:{timeVisible:!!cfg.intraday,secondsVisible:false},
    crosshair:{mode:LWC.CrosshairMode.Normal}
  });
  var candle = chart.addSeries(LWC.CandlestickSeries,{},0);
  var vol = chart.addSeries(LWC.HistogramSeries,{
    priceFormat:{type:'volume'},priceScaleId:''
  },1);
  vol.priceScale().applyOptions({scaleMargins:{top:0.8,bottom:0}});
  if(chart.panes && chart.panes()[1]){chart.panes()[1].setHeight(120);}
  var markerPrim = LWC.createSeriesMarkers(candle, []);
  var curIdx = cfg.initial_symbol_index||0;

  function applyTheme(){
    var t = T();
    chart.applyOptions({
      layout:{background:{color:t.bg},textColor:t.text},
      grid:{vertLines:{color:t.grid},horzLines:{color:t.grid}},
      rightPriceScale:{borderColor:t.grid},
      timeScale:{borderColor:t.grid}
    });
    candle.applyOptions({
      upColor:t.up,downColor:t.down,borderUpColor:t.up,
      borderDownColor:t.down,wickUpColor:t.up,wickDownColor:t.down
    });
    document.documentElement.setAttribute('data-theme',cur);
    toggle.textContent = cur==='dark' ? '☀️ 亮色' : '🌙 暗色';
  }
  function colorVol(s){
    var t=T();
    return (s.volume||[]).map(function(v){
      return {time:v.time,value:v.value,color:v.up?t.up:t.down};
    });
  }
  function colorMarkers(s){
    var t=T();
    return (s.markers||[]).map(function(m){
      return {time:m.time,position:m.position,shape:m.shape,
        text:m.text,color:m.buy?t.up:t.down};
    });
  }
  function draw(i){
    var s = all[i]; if(!s) return;
    curIdx = i;
    candle.setData(s.candles||[]);
    vol.setData(colorVol(s));
    markerPrim.setMarkers(colorMarkers(s));
    chart.timeScale().fitContent();
  }
  function recolorCurrent(){
    var s = all[curIdx]; if(!s) return;
    vol.setData(colorVol(s));
    markerPrim.setMarkers(colorMarkers(s));
  }
  sel.addEventListener('change',function(){draw(parseInt(sel.value,10)||0);});
  toggle.addEventListener('click',function(){
    cur = cur==='dark' ? 'light' : 'dark';
    applyTheme(); recolorCurrent();
  });
  applyTheme();
  sel.value = curIdx; draw(curIdx);
})();
"""


def _to_js_theme(colors: dict[str, str]) -> dict[str, str]:
    """把 ``THEMES`` 条目转成前端用的短键色板."""
    return {
        "up": colors["up_color"],
        "down": colors["down_color"],
        "bg": colors["bg_color"],
        "grid": colors["grid_color"],
        "text": colors["text_color"],
    }


def render_review_html(
    payload: dict[str, Any],
    title: str,
    intraday: bool,
    themes: Optional[dict[str, dict[str, str]]] = None,
    initial_theme: str = "light",
    initial_symbol_index: int = 0,
) -> str:
    """把 payload 渲染成离线自包含的复盘 HTML 字符串(支持页内明暗切换).

    :param payload: :func:`.._payload.build_review_payload` 的返回值.
    :param title: 报告标题(将被 HTML 转义).
    :param intraday: 是否日内(影响时间轴显示时分).
    :param themes: ``{"light": {...}, "dark": {...}}`` 主题色板;缺省用 ``THEMES``.
    :param initial_theme: 初始主题键(``"light"`` / ``"dark"``).
    :param initial_symbol_index: 初始展示的标的下标.
    :return: 完整 HTML 文本.
    """
    if themes is None:
        from ..plot.utils import THEMES

        themes = THEMES
    init = initial_theme if initial_theme in themes else "light"
    data = dict(payload)
    data["themes"] = {name: _to_js_theme(cols) for name, cols in themes.items()}
    data["initial_theme"] = init
    data["intraday"] = bool(intraday)
    data["initial_symbol_index"] = int(initial_symbol_index)
    safe_title = html.escape(str(title))
    # 顺序敏感:先注入受控内容与静态 JS,最后注入 DATA,
    # 避免用户数据里的 "%%...%%" 字面量被后续替换误伤(DATA 之后无替换)。
    ordered = [
        ("%%TITLE%%", safe_title),
        ("%%INIT_THEME%%", init),
        ("%%APP_JS%%", _APP_JS),
        ("%%LWC_JS%%", _load_lwc_js()),
        ("%%DATA%%", _safe_json(data)),
    ]
    out = _HTML_TEMPLATE
    for token, value in ordered:
        out = out.replace(token, value)
    return out
