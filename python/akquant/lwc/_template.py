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
  #sym,#theme-toggle,#price-toggle,#drawings-toggle{padding:4px 8px;
    background:var(--bg);color:var(--text);
    border:1px solid var(--grid);border-radius:4px;font-size:13px;cursor:pointer;}
  #sym{margin-left:auto;cursor:default;}
  #chart-wrap{position:relative;}
  #chart{width:100%;height:calc(100vh - 52px);}
  #empty{padding:40px;text-align:center;color:var(--text);opacity:.6;}
  /* 策略表格浮层: 只覆盖主图 pane(与 TradingView 一致, 表格属于主图), 不盖成交量
     和副图; 宽高由 JS 按 chart.paneSize(0) 同步, 因此不含价格刻度与时间轴。
     不拦截鼠标(图表照常拖动/缩放); 超出主图的部分裁掉。TradingView 署名 logo
     在最下面的 pane(成交量)左下角, 不在主图内, bottom_left 不会盖住它。 */
  #akq-tables{position:absolute;top:0;left:0;width:0;height:0;overflow:hidden;
    pointer-events:none;z-index:5;}
  .akq-pos{position:absolute;display:flex;flex-direction:column;gap:6px;
    max-width:45%;}
  .akq-pos.top_left{top:8px;left:8px}
  .akq-pos.top_center{top:8px;left:50%;transform:translateX(-50%)}
  .akq-pos.top_right{top:8px;right:8px}
  .akq-pos.middle_left{top:50%;left:8px;transform:translateY(-50%)}
  .akq-pos.middle_center{top:50%;left:50%;transform:translate(-50%,-50%)}
  .akq-pos.middle_right{top:50%;right:8px;transform:translateY(-50%)}
  .akq-pos.bottom_left{bottom:8px;left:8px}
  .akq-pos.bottom_center{bottom:8px;left:50%;transform:translateX(-50%)}
  .akq-pos.bottom_right{bottom:8px;right:8px}
  .akq-table{--akq-border:var(--grid);border-collapse:collapse;font-size:12px;
    background:var(--bg);opacity:.92;border:1px solid var(--akq-border);}
  .akq-table td{padding:2px 8px;border:1px solid var(--akq-border);
    white-space:nowrap;}
  .akq-table td.n{text-align:right;font-variant-numeric:tabular-nums;}
  .akq-table tr.h td{font-weight:600;}
</style>
</head>
<body>
<div id="bar">
  <h1>%%TITLE%%</h1>
  <select id="sym" aria-label="标的选择"></select>
  <button id="drawings-toggle" type="button"
    aria-label="隐藏或显示策略绘图">隐藏绘图</button>
  <button id="price-toggle" type="button" aria-label="隐藏或显示价格 K 线"
    style="display:none">隐藏 K 线</button>
  <button id="theme-toggle" type="button" aria-label="切换明暗主题">🌙 暗色</button>
</div>
<div id="chart-wrap"><div id="chart"></div><div id="akq-tables"></div></div>
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
  // 隐藏价格 K 线: 只在当前标的有 candle 指标(如平均 K 线)时生效并显示按钮,
  // 否则隐藏后主图只剩空白。状态跨标的切换保留。
  var priceBtn = document.getElementById('price-toggle');
  var priceHidden = !!cfg.hide_price;
  var hasCandleInd = false;
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

  // ---- 策略绘图层(self.draw): line/box/label/polyline/fib 用挂在 K 线 series 上的
  // primitive 在 canvas 上画, table 用 HTML 浮层。这也是给平台前端的参考实现。
  // 点的 time 已在 Python 侧吸附到某根 bar, timeToCoordinate 才不会返回 null。
  // 颜色在 Python 侧已校验为 #hex; 这里再校验一次(数据可能来自别处), 空串取主题色。
  var drawingsHidden = false, curShapes = [];
  var FIB = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
  var LABEL_SIZE = {small:11, normal:12, large:14};
  var TABLE_POS = ['top_left','top_center','top_right','middle_left',
    'middle_center','middle_right','bottom_left','bottom_center','bottom_right'];
  function hex(c){
    return /^#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/.test(c||'') ? c : '';
  }
  function textLines(ctx, text){
    var lines = String(text).split('\\n'), w = 0;
    lines.forEach(function(l){ w = Math.max(w, ctx.measureText(l).width); });
    return {lines:lines, width:w};
  }
  function fillLines(ctx, lines, x, y, lineH){
    lines.forEach(function(l, k){ ctx.fillText(l, x, y + k*lineH); });
  }
  function drawShape(ctx, d, pts, r, W, H, Y){
    var st = d.style||{}, col = hex(st.color) || hex(st.border_color) || T().text;
    ctx.strokeStyle = col; ctx.fillStyle = col;
    ctx.lineWidth = Math.max(1, (st.width||1)*r);
    ctx.setLineDash(st.line_style==='dashed' ? [6*r,4*r]
      : (st.line_style==='dotted' ? [2*r,3*r] : []));
    if(d.kind==='line'){
      var a = pts[0], b = pts[1], dx = b[0]-a[0], dy = b[1]-a[1];
      var x0 = a[0], y0 = a[1], x1 = b[0], y1 = b[1];
      var yAt = function(x){ return a[1] + dy*(x-a[0])/dx; };
      if(st.extend==='right' || st.extend==='both'){
        // 射线: 从 p1 经 p2 延伸到画布边缘(p2 在 p1 左侧时向左延伸)
        if(dx!==0){ x1 = dx>0 ? W : 0; y1 = yAt(x1); }
        else if(dy!==0){ y1 = dy>0 ? H : 0; }
      }
      if(st.extend==='both'){
        if(dx!==0){ x0 = dx>0 ? 0 : W; y0 = yAt(x0); }
        else if(dy!==0){ y0 = dy>0 ? 0 : H; }
      }
      ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.stroke();
    } else if(d.kind==='box'){
      var bx = Math.min(pts[0][0],pts[1][0]), by = Math.min(pts[0][1],pts[1][1]);
      var bw = Math.abs(pts[1][0]-pts[0][0]), bh = Math.abs(pts[1][1]-pts[0][1]);
      if(hex(st.fill_color)){
        ctx.fillStyle = st.fill_color; ctx.fillRect(bx,by,bw,bh);
      }
      ctx.strokeRect(bx,by,bw,bh);
      if(d.text){
        var fs = 12*r; ctx.font = fs+'px sans-serif';
        var tb = textLines(ctx, d.text), lh = fs*1.25;
        ctx.fillStyle = T().text; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        fillLines(ctx, tb.lines, bx+bw/2, by+bh/2-(tb.lines.length-1)*lh/2, lh);
      }
    } else if(d.kind==='polyline'){
      ctx.beginPath(); ctx.moveTo(pts[0][0],pts[0][1]);
      for(var i=1;i<pts.length;i++) ctx.lineTo(pts[i][0],pts[i][1]);
      if(st.closed){
        ctx.closePath();
        if(hex(st.fill_color)){ ctx.fillStyle = st.fill_color; ctx.fill(); }
      }
      ctx.stroke();
    } else if(d.kind==='fib'){
      // 档位 0 在 p2、1 在 p1(TradingView 回撤惯例); 档线半透明, 文字不透明
      var p1 = d.points[0].price, p2 = d.points[1].price;
      var fx0 = Math.min(pts[0][0],pts[1][0]), fx1 = Math.max(pts[0][0],pts[1][0]);
      ctx.font = (11*r)+'px sans-serif'; ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      FIB.forEach(function(lv){
        var price = p2 + (p1-p2)*lv, y = Y(price); if(y===null) return;
        ctx.globalAlpha = 0.8;
        ctx.beginPath(); ctx.moveTo(fx0,y); ctx.lineTo(fx1,y); ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillText(lv.toFixed(3)+' ('+price.toFixed(2)+')', fx0+2*r, y-1*r);
      });
    } else if(d.kind==='label'){
      // label_style 是气泡指向: down=气泡在点上方指向下, up=在下方指向上,
      // left=在右侧指向左, right=在左侧指向右, none=只有文字(以点为中心)
      var ls = st.label_style || 'down', size = (LABEL_SIZE[st.size]||12)*r;
      ctx.font = size+'px sans-serif';
      var lb = textLines(ctx, d.text), lH = size*1.25;
      var pad = 4*r, w2 = lb.width+pad*2, h2 = lb.lines.length*lH+pad*2;
      var lx = pts[0][0], ly = pts[0][1], off = 6*r;
      var rx = lx-w2/2, ry = ly-h2-off, tri = null;
      if(ls==='up'){ ry = ly+off; tri = [[lx-off,ry],[lx+off,ry]]; }
      else if(ls==='left'){ rx = lx+off; ry = ly-h2/2;
        tri = [[rx,ly-off],[rx,ly+off]]; }
      else if(ls==='right'){ rx = lx-w2-off; ry = ly-h2/2;
        tri = [[rx+w2,ly-off],[rx+w2,ly+off]]; }
      else if(ls==='none'){ rx = lx-w2/2; ry = ly-h2/2; }
      else { tri = [[lx-off,ry+h2],[lx+off,ry+h2]]; }
      var bubble = hex(st.color) || T().text;
      if(ls!=='none'){
        ctx.fillStyle = bubble; ctx.fillRect(rx,ry,w2,h2);
        ctx.beginPath(); ctx.moveTo(tri[0][0],tri[0][1]);
        ctx.lineTo(tri[1][0],tri[1][1]);
        ctx.lineTo(lx,ly); ctx.closePath(); ctx.fill();
      }
      // 缺省文字色: 无气泡时同气泡色; 用户给了气泡色时白字(与主题无关, 否则暗色
      // 主题下深色字压在彩色气泡上看不清); 气泡也取主题色时用背景色反白。
      ctx.fillStyle = hex(st.text_color) || (ls==='none' ? bubble
        : (hex(st.color) ? '#ffffff' : T().bg));
      ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      fillLines(ctx, lb.lines, rx+pad, ry+pad, lH);
    }
  }
  var drawRenderer = {
    draw: function(target){
      // 主图每次重绘(autoSize 改尺寸、拖动 pane 分隔条、副图增删)都会走到这里,
      // 顺便核对主图尺寸; 变了就下一帧同步表格浮层, 不在绘制回调里改 DOM。
      target.useMediaCoordinateSpace(function(m){
        notePaneSize(m.mediaSize.width, m.mediaSize.height);
      });
      if(drawingsHidden || !curShapes.length) return;
      target.useBitmapCoordinateSpace(function(scope){
        var ctx = scope.context, r = scope.horizontalPixelRatio;
        var vr = scope.verticalPixelRatio, ts = chart.timeScale();
        var W = scope.bitmapSize.width, H = scope.bitmapSize.height;
        function X(t){ var x = ts.timeToCoordinate(t); return x===null ? null : x*r; }
        function Y(p){
          var y = candle.priceToCoordinate(p); return y===null ? null : y*vr;
        }
        curShapes.forEach(function(d){
          var pts = d.points.map(function(p){ return [X(p.time), Y(p.price)]; });
          if(pts.some(function(p){ return p[0]===null || p[1]===null; })) return;
          ctx.save();
          try{ drawShape(ctx, d, pts, r, W, H, Y); }
          finally{ ctx.restore(); }
        });
      });
    }
  };
  var drawView = {renderer: function(){ return drawRenderer; }};
  // LWC v5 series primitive: attachPrimitive 时回调 attached({chart, series,
  // requestUpdate}), 每帧取 paneViews()[i].renderer().draw(target)。
  var drawPrim = {
    _req: null,
    attached: function(p){ this._req = p.requestUpdate; },
    detached: function(){ this._req = null; },
    updateAllViews: function(){},
    paneViews: function(){ return [drawView]; },
    refresh: function(){ if(this._req) this._req(); }
  };
  candle.attachPrimitive(drawPrim);
  var tablesHost = document.getElementById('akq-tables');
  // 表格浮层只覆盖主图 pane: 位置/尺寸取自 LWC 公开 API(paneSize(0) 不含价格
  // 刻度; 左侧刻度默认隐藏, 宽度为 0, 显示时浮层随之右移)。
  var overlayKey = '', syncPending = false;
  function syncOverlay(){
    syncPending = false;
    var sz = chart.paneSize(0), left = chart.priceScale('left').width();
    var key = sz.width+'x'+sz.height+'+'+left;
    if(key===overlayKey) return;
    overlayKey = key;
    tablesHost.style.left = left+'px';
    tablesHost.style.width = sz.width+'px';
    tablesHost.style.height = sz.height+'px';
  }
  function notePaneSize(w, h){
    if(syncPending || overlayKey.indexOf(w+'x'+h+'+')===0) return;
    syncPending = true;
    requestAnimationFrame(syncOverlay);
  }
  function esc(s){
    return String(s==null ? '' : s).replace(/[&<>"']/g, function(c){
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
    });
  }
  // 显式背景色上的缺省文字色按亮度取深/浅, 不跟主题(否则暗色主题下浅色字压在
  // 用户给的浅色底上看不清); 透明度过半的底色仍跟主题。
  function ink(bg){
    if(!hex(bg) || (bg.length===9 && parseInt(bg.slice(7,9),16) < 128)) return '';
    var n = parseInt(bg.slice(1,7),16);
    var lum = 0.299*(n>>16) + 0.587*((n>>8)&255) + 0.114*(n&255);
    return lum > 150 ? '#212529' : '#f8f9fa';
  }
  // 同一位置的多张表纵向堆叠; Python 侧已按 (位置, 全局表在前, 创建顺序) 排好序。
  function drawTables(s){
    tablesHost.innerHTML = '';
    if(drawingsHidden || !s) return;
    var groups = {};
    (s.tables||[]).forEach(function(t){
      var tb = t.table || {}, pos = tb.position;
      if(TABLE_POS.indexOf(pos) < 0) return;
      if(!groups[pos]){
        groups[pos] = document.createElement('div');
        groups[pos].className = 'akq-pos '+pos;
        tablesHost.appendChild(groups[pos]);
      }
      var st = t.style||{}, css = '';
      if(hex(st.bg_color)) css += 'background:'+st.bg_color+';';
      if(ink(st.bg_color)) css += 'color:'+ink(st.bg_color)+';';
      if(hex(st.border_color)) css += '--akq-border:'+st.border_color+';';
      var html = '<table class="akq-table"'+(css ? ' style="'+css+'"' : '')+'>';
      (tb.rows||[]).forEach(function(row, ri){
        html += '<tr'+(ri < (tb.header_rows||0) ? ' class="h"' : '')+'>';
        row.forEach(function(c){
          c = c || {};
          var cs = (hex(c.bg_color) ? 'background:'+c.bg_color+';' : '')
            + (hex(c.text_color) ? 'color:'+c.text_color+';'
              : (ink(c.bg_color) ? 'color:'+ink(c.bg_color)+';' : ''));
          var num = typeof c.value==='number';
          html += '<td'+(num ? ' class="n"' : '')+(cs ? ' style="'+cs+'"' : '')+'>'
            + esc(c.text) + '</td>';
        });
        html += '</tr>';
      });
      groups[pos].insertAdjacentHTML('beforeend', html+'</table>');
    });
  }
  var drawBtn = document.getElementById('drawings-toggle');
  if(!all.some(function(s){
    return (s.drawings||[]).length || (s.tables||[]).length;
  })) drawBtn.style.display = 'none';
  drawBtn.addEventListener('click', function(){
    drawingsHidden = !drawingsHidden;
    drawBtn.textContent = drawingsHidden ? '显示绘图' : '隐藏绘图';
    drawTables(all[curIdx]); drawPrim.refresh();
  });
  // 指标 series 随标的切换重建; 元素 {series, ind, idx}。signal 类不建 series,
  // 作为 marker 与买卖点一起交给 markerPrim。
  var indSeries = [];
  var palette = cfg.palette || {};
  function indColor(ind, i){
    var p = palette[cur] || palette.light || ['#1976d2'];
    return ind.color || p[i % p.length];
  }
  var CANDLE_HOLLOW = 'rgba(0,0,0,0)';
  function tcmp(a,b){return a.time<b.time?-1:(a.time>b.time?1:0);}
  function clearIndicators(){
    indSeries.forEach(function(x){ try{ chart.removeSeries(x.series); }catch(e){} });
    indSeries = [];
  }
  function indOptions(ind, i){
    var c = indColor(ind, i);
    if(ind.series_type==='Candlestick'){
      // candle 指标绝不用主题的涨跌色(否则与价格 K 线分不清): 单色 c,
      // 涨为实心、跌为空心(TradingView hollow candles 惯例);
      // 点位为 {time,open,high,low,close}
      return {upColor:c, downColor:CANDLE_HOLLOW, borderUpColor:c,
        borderDownColor:c, wickUpColor:c, wickDownColor:c,
        priceLineVisible:false, lastValueVisible:true,
        title: ind.display_name || ind.series_key || ind.indicator_key};
    }
    var o = {color:c, lineWidth:2, priceLineVisible:false, lastValueVisible:true,
      title: ind.display_name || ind.series_key || ind.indicator_key};
    if(ind.series_type==='Area'){
      o.lineColor=c; o.topColor=c+'55'; o.bottomColor=c+'05';
    }
    if(ind.render_type==='scatter'){
      o.lineVisible=false; o.pointMarkersVisible=true;
    }
    return o;
  }
  function drawIndicators(s){
    clearIndicators();
    var signals = [];
    // payload 的 pane 已是 LWC 口径(0 主图, >=2 副图), 但可能不连续
    // (用户只声明了 pane 3); LWC 要求 paneIndex 连续, 这里按出现顺序压实。
    var paneMap = {}, nextPane = 2;
    (s.indicators||[]).forEach(function(ind, i){
      if(ind.series_type==='Marker'){
        (ind.points||[]).forEach(function(p){
          signals.push({time:p.time, position:'aboveBar', shape:'circle',
            text: ind.display_name || ind.series_key || ind.indicator_key,
            color: indColor(ind, i)});
        });
        return;
      }
      var ctor = LWC[ind.series_type + 'Series'] || LWC.LineSeries;
      var paneIdx = 0;
      if(ind.pane >= 2){
        if(!(ind.pane in paneMap)){ paneMap[ind.pane] = nextPane++; }
        paneIdx = paneMap[ind.pane];
      }
      var opts = indOptions(ind, i);
      var series = chart.addSeries(ctor, opts, paneIdx);
      series.setData(ind.points || []);
      (ind.price_lines||[]).forEach(function(pl){
        series.createPriceLine({price: pl.value, title: pl.label || '',
          color: pl.color || opts.color, lineWidth:1, lineStyle:2,
          axisLabelVisible:true});
      });
      if(paneIdx >= 2 && chart.panes && chart.panes()[paneIdx]){
        chart.panes()[paneIdx].setHeight(100);
      }
      indSeries.push({series:series, ind:ind, idx:i});
    });
    return signals;
  }
  function recolorIndicators(){
    indSeries.forEach(function(x){
      if(x.ind.color) return;  // 用户指定的颜色不随主题变
      x.series.applyOptions(indOptions(x.ind, x.idx));
    });
  }

  // 隐藏用透明色而不是 visible:false —— 买卖点 marker 挂在价格 series 上,
  // 隐藏 series 会连带隐藏 marker。
  function priceOptions(){
    var t = T();
    if(priceHidden && hasCandleInd){
      var h = CANDLE_HOLLOW;
      return {upColor:h,downColor:h,borderUpColor:h,borderDownColor:h,
        wickUpColor:h,wickDownColor:h,lastValueVisible:false,priceLineVisible:false};
    }
    return {upColor:t.up,downColor:t.down,borderUpColor:t.up,
      borderDownColor:t.down,wickUpColor:t.up,wickDownColor:t.down,
      lastValueVisible:true,priceLineVisible:true};
  }
  function applyPriceVisibility(){
    candle.applyOptions(priceOptions());
    priceBtn.style.display = hasCandleInd ? '' : 'none';
    priceBtn.textContent = priceHidden ? '显示 K 线' : '隐藏 K 线';
  }
  function applyTheme(){
    var t = T();
    chart.applyOptions({
      layout:{background:{color:t.bg},textColor:t.text},
      grid:{vertLines:{color:t.grid},horzLines:{color:t.grid}},
      rightPriceScale:{borderColor:t.grid},
      timeScale:{borderColor:t.grid}
    });
    candle.applyOptions(priceOptions());
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
  function allMarkers(s, signals){
    return colorMarkers(s).concat(signals || []).sort(tcmp);
  }
  var curSignals = [];
  function draw(i){
    var s = all[i]; if(!s) return;
    curIdx = i;
    candle.setData(s.candles||[]);
    vol.setData(colorVol(s));
    curSignals = drawIndicators(s);
    hasCandleInd = (s.indicators||[]).some(function(ind){
      return ind.series_type==='Candlestick';
    });
    applyPriceVisibility();
    markerPrim.setMarkers(allMarkers(s, curSignals));
    curShapes = s.drawings||[]; drawTables(s); drawPrim.refresh();
    chart.timeScale().fitContent();
    syncOverlay();  // 副图随标的重建, 主图高度可能变了
  }
  function recolorCurrent(){
    var s = all[curIdx]; if(!s) return;
    vol.setData(colorVol(s));
    recolorIndicators();
    markerPrim.setMarkers(allMarkers(s, curSignals));
    drawPrim.refresh();  // 缺省色取自主题
  }
  sel.addEventListener('change',function(){draw(parseInt(sel.value,10)||0);});
  priceBtn.addEventListener('click',function(){
    priceHidden = !priceHidden;
    applyPriceVisibility();
  });
  toggle.addEventListener('click',function(){
    cur = cur==='dark' ? 'light' : 'dark';
    applyTheme(); recolorCurrent();
  });
  applyTheme();
  sel.value = curIdx; draw(curIdx);
})();
"""


#: 未指定 color 的指标按主题轮转取色(明暗各一组, 与 K 线红绿避让)。
INDICATOR_PALETTE: dict[str, list[str]] = {
    "light": ["#1976d2", "#f57c00", "#7b1fa2", "#00897b", "#c2185b", "#5d4037"],
    "dark": ["#64b5f6", "#ffb74d", "#ba68c8", "#4db6ac", "#f06292", "#a1887f"],
}


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
    hide_price: bool = False,
) -> str:
    """把 payload 渲染成离线自包含的复盘 HTML 字符串(支持页内明暗切换).

    :param payload: :func:`.._payload.build_review_payload` 的返回值.
    :param title: 报告标题(将被 HTML 转义).
    :param intraday: 是否日内(影响时间轴显示时分).
    :param themes: ``{"light": {...}, "dark": {...}}`` 主题色板;缺省用 ``THEMES``.
    :param initial_theme: 初始主题键(``"light"`` / ``"dark"``).
    :param initial_symbol_index: 初始展示的标的下标.
    :param hide_price: 初始是否隐藏价格 K 线; 只对带 candle 指标的标的生效.
    :return: 完整 HTML 文本.
    """
    if themes is None:
        from ..plot.utils import THEMES

        themes = THEMES
    init = initial_theme if initial_theme in themes else "light"
    data = dict(payload)
    data["themes"] = {name: _to_js_theme(cols) for name, cols in themes.items()}
    data["palette"] = {
        name: INDICATOR_PALETTE.get(name, INDICATOR_PALETTE["light"]) for name in themes
    }
    data["initial_theme"] = init
    data["intraday"] = bool(intraday)
    data["initial_symbol_index"] = int(initial_symbol_index)
    data["hide_price"] = bool(hide_price)
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
