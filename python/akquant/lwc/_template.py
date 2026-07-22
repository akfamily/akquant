"""HTML/CSS/JS template for the lightweight-charts report page.

The template is a single self-contained HTML document. Placeholders are
substituted by :func:`akquant.lwc.report.render_html`:

- ``__TITLE__``: HTML-escaped report title (appears twice: <title> and <h1>).
- ``__LWC_JS__``: the vendored Lightweight Charts standalone bundle.
- ``__APP_JSON__``: JSON application payload (see ``report.build_app_data``).

Keeping the template in Python (rather than a separate file) avoids extra
package-data plumbing beyond the single vendored JS bundle.
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
:root {
  --up: #ef5350;
  --down: #26a69a;
  --accent: #2962ff;
  --border: #e5e7eb;
  --muted: #6b7280;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 24px 16px 48px;
  background: #f5f6fa;
  color: #1f2937;
  font-family: -apple-system, "Segoe UI", "PingFang SC",
    "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
}
.page { max-width: 1240px; margin: 0 auto; }
header h1 { font-size: 22px; margin: 0 0 4px; }
header .sub { color: var(--muted); font-size: 12px; margin-bottom: 16px; }
.card {
  background: #fff;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 18px;
  margin-bottom: 18px;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
.card h2 { font-size: 16px; margin: 0 0 12px; }
.card h3 { font-size: 13px; margin: 14px 0 8px; color: var(--muted); }
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 10px;
}
.metric {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 10px;
  background: #fafafa;
}
.metric .label { font-size: 12px; color: var(--muted); }
.metric .value { font-size: 17px; font-weight: 600; margin-top: 2px; }
.chart { width: 100%; position: relative; }
.chart-lg { height: 460px; }
.chart-md { height: 300px; }
.chart-sm { height: 160px; }
.controls {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}
.controls input {
  padding: 6px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 14px;
  width: 220px;
}
.controls button {
  padding: 6px 16px;
  border: none;
  border-radius: 6px;
  background: var(--accent);
  color: #fff;
  font-size: 14px;
  cursor: pointer;
}
.controls button:hover { background: #1e4fd8; }
.controls .status { font-size: 12px; color: var(--muted); }
.controls .status.error { color: var(--up); }
.tooltip {
  position: absolute;
  z-index: 20;
  display: none;
  max-width: 320px;
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.97);
  box-shadow: 0 2px 8px rgba(16, 24, 40, 0.12);
  font-size: 12px;
  line-height: 1.6;
  pointer-events: none;
}
.tooltip .win { color: var(--up); font-weight: 600; }
.tooltip .loss { color: var(--down); font-weight: 600; }
table.trades {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  margin-top: 12px;
}
table.trades th, table.trades td {
  border-bottom: 1px solid var(--border);
  padding: 5px 6px;
  text-align: right;
  white-space: nowrap;
}
table.trades th:first-child, table.trades td:first-child,
table.trades th:nth-child(2), table.trades td:nth-child(2) {
  text-align: left;
}
table.trades tbody tr { cursor: pointer; }
table.trades tbody tr:hover { background: #f3f6ff; }
.pos { color: var(--up); }
.neg { color: var(--down); }
.hint { color: var(--muted); font-size: 12px; }
</style>
<script>__LWC_JS__</script>
</head>
<body>
<div class="page">
<header>
  <h1>__TITLE__</h1>
  <div class="sub">AKQuant &times; TradingView Lightweight Charts 生成</div>
</header>

<section class="card" id="metrics-card">
  <h2>绩效指标</h2>
  <div class="metrics-grid" id="metrics"></div>
</section>

<section class="card">
  <h2>净值曲线</h2>
  <div id="equity-chart" class="chart chart-md"></div>
  <h3>回撤</h3>
  <div id="drawdown-chart" class="chart chart-sm"></div>
</section>

<section class="card">
  <h2>交易复盘（K线买卖点）</h2>
  <div class="controls">
    <input id="symbol-input" list="symbol-list"
           placeholder="输入股票代码后回车，如 600519.SH"/>
    <datalist id="symbol-list"></datalist>
    <button id="symbol-go" type="button">复盘</button>
    <span class="status" id="symbol-status"></span>
  </div>
  <div id="kline-chart" class="chart chart-lg">
    <div class="tooltip" id="trade-tooltip"></div>
  </div>
  <div class="hint">
    提示：点击交易明细行可定位到对应区间；K 线上悬停买卖点可查看交易详情。
  </div>
  <table class="trades" id="trades-table" style="display:none">
    <thead>
      <tr>
        <th>方向</th><th>开/平</th><th>开仓时间</th><th>开仓价</th>
        <th>平仓时间</th><th>平仓价</th><th>数量</th>
        <th>收益率</th><th>净利润</th>
      </tr>
    </thead>
    <tbody id="trades-body"></tbody>
  </table>
</section>
</div>

<script id="app-data" type="application/json">__APP_JSON__</script>
<script>
(function () {
  'use strict';
  var APP = JSON.parse(document.getElementById('app-data').textContent);
  var UP = '#ef5350';
  var DOWN = '#26a69a';

  function el(id) { return document.getElementById(id); }

  function baseOpts(height) {
    return {
      height: height,
      layout: {
        background: { type: 'solid', color: '#ffffff' },
        textColor: '#374151'
      },
      grid: {
        vertLines: { color: '#f3f4f6' },
        horzLines: { color: '#f3f4f6' }
      },
      rightPriceScale: { borderColor: '#e5e7eb' },
      timeScale: { borderColor: '#e5e7eb', timeVisible: true },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal }
    };
  }

  function watchResize(chart, node) {
    if (typeof ResizeObserver === 'undefined') { return; }
    var ro = new ResizeObserver(function (entries) {
      var w = entries[0].contentRect.width;
      if (w > 0) { chart.applyOptions({ width: w }); }
    });
    ro.observe(node);
  }

  /* ---------------- metrics ---------------- */
  (function renderMetrics() {
    var box = el('metrics');
    if (!APP.metrics || !APP.metrics.length) {
      el('metrics-card').style.display = 'none';
      return;
    }
    APP.metrics.forEach(function (m) {
      var div = document.createElement('div');
      div.className = 'metric';
      div.innerHTML = '<div class="label"></div><div class="value"></div>';
      div.querySelector('.label').textContent = m.label;
      div.querySelector('.value').textContent = m.value;
      box.appendChild(div);
    });
  })();

  /* ---------------- equity & drawdown ---------------- */
  function makeLineChart(nodeId, data, color, isPct) {
    var node = el(nodeId);
    if (!data || !data.length) {
      node.innerHTML = '<div class="hint">无数据</div>';
      return null;
    }
    var chart = LightweightCharts.createChart(node, baseOpts(node.clientHeight));
    var series = chart.addAreaSeries({
      lineColor: color,
      topColor: color.replace(')', ', 0.28)').replace('rgb', 'rgba'),
      bottomColor: color.replace(')', ', 0.02)').replace('rgb', 'rgba'),
      lineWidth: 2,
      priceLineVisible: false
    });
    if (isPct) {
      series.applyOptions({
        priceFormat: {
          type: 'custom',
          minMove: 0.0001,
          formatter: function (v) { return (v * 100).toFixed(2) + '%'; }
        }
      });
    }
    series.setData(data);
    chart.timeScale().fitContent();
    watchResize(chart, node);
    return chart;
  }

  var eqChart = makeLineChart(
    'equity-chart', APP.equity, 'rgb(41, 98, 255)', false);
  var ddChart = makeLineChart(
    'drawdown-chart', APP.drawdown, 'rgb(239, 83, 80)', true);

  (function syncCharts(a, b) {
    if (!a || !b) { return; }
    var syncing = false;
    function hook(src, dst) {
      src.timeScale().subscribeVisibleLogicalRangeChange(function (range) {
        if (syncing || !range) { return; }
        syncing = true;
        dst.timeScale().setVisibleLogicalRange(range);
        syncing = false;
      });
    }
    hook(a, b);
    hook(b, a);
  })(eqChart, ddChart);

  /* ---------------- kline / trade review ---------------- */
  var klineNode = el('kline-chart');
  var klineChart = LightweightCharts.createChart(
    klineNode, baseOpts(klineNode.clientHeight));
  var candleSeries = klineChart.addCandlestickSeries({
    upColor: UP,
    downColor: DOWN,
    borderVisible: false,
    wickUpColor: UP,
    wickDownColor: DOWN
  });
  candleSeries.priceScale().applyOptions({
    scaleMargins: { top: 0.08, bottom: 0.28 }
  });
  var volumeSeries = klineChart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: ''
  });
  klineChart.priceScale('').applyOptions({
    scaleMargins: { top: 0.82, bottom: 0 }
  });
  watchResize(klineChart, klineNode);

  var currentSymbol = null;

  function setStatus(text, isError) {
    var node = el('symbol-status');
    node.textContent = text;
    node.className = isError ? 'status error' : 'status';
  }

  function addSymbolOption(code) {
    var list = el('symbol-list');
    var exists = Array.prototype.some.call(
      list.options, function (o) { return o.value === code; });
    if (!exists) {
      var opt = document.createElement('option');
      opt.value = code;
      list.appendChild(opt);
    }
  }

  (APP.symbols || []).forEach(addSymbolOption);

  function fmtPct(x) {
    return x == null ? '-' : (x * 100).toFixed(2) + '%';
  }
  function fmtNum(x) {
    if (x == null) { return '-'; }
    return x.toLocaleString('zh-CN', { maximumFractionDigits: 2 });
  }
  function fmtSigned(x) {
    if (x == null) { return '-'; }
    var s = x >= 0 ? '+' : '';
    return s + fmtNum(x);
  }
  function pnlClass(x) {
    if (x == null) { return ''; }
    return x >= 0 ? 'pos' : 'neg';
  }

  function renderKline(payload) {
    candleSeries.setData(payload.candles || []);
    volumeSeries.setData(payload.volumes || []);
    candleSeries.setMarkers(payload.markers || []);
    payload._times = (payload.candles || []).map(function (c) {
      return c.time;
    });
    klineChart.timeScale().fitContent();
  }

  function renderTradesTable(payload) {
    var tbody = el('trades-body');
    var table = el('trades-table');
    tbody.innerHTML = '';
    var trades = payload.trades || [];
    table.style.display = trades.length ? '' : 'none';
    trades.forEach(function (t) {
      var tr = document.createElement('tr');
      var cells = [
        t.side === 'short' ? '空' : '多',
        (t.side === 'short' ? 'S→B' : 'B→S'),
        t.entry_label,
        fmtNum(t.entry_price),
        t.exit_label,
        fmtNum(t.exit_price),
        fmtNum(t.quantity),
        fmtPct(t.return_pct),
        fmtSigned(t.net_pnl)
      ];
      cells.forEach(function (text, idx) {
        var td = document.createElement('td');
        td.textContent = text == null ? '-' : text;
        if (idx >= 7) { td.className = pnlClass(t.net_pnl); }
        tr.appendChild(td);
      });
      tr.addEventListener('click', function () {
        focusTrade(payload, t);
      });
      tbody.appendChild(tr);
    });
  }

  function focusTrade(payload, trade) {
    var times = payload._times || [];
    var i = times.indexOf(trade.entry_time);
    var j = times.indexOf(trade.exit_time);
    if (i < 0) { i = 0; }
    if (j < 0) { j = times.length - 1; }
    klineChart.timeScale().setVisibleLogicalRange({
      from: Math.max(0, i - 10),
      to: Math.min(times.length + 10, j + 10)
    });
  }

  function tradeTipHtml(trades, time) {
    var rows = trades.map(function (t) {
      var isEntry = t.entry_time === time;
      var pnl = fmtSigned(t.net_pnl);
      var cls = pnlClass(t.net_pnl);
      var head = isEntry ? '开仓' : '平仓';
      var dir = t.side === 'short' ? '空头' : '多头';
      return '<div><b>' + head + ' · ' + dir + '</b><br/>' +
        '开仓 ' + t.entry_label + ' @ ' + fmtNum(t.entry_price) + '<br/>' +
        '平仓 ' + t.exit_label + ' @ ' + fmtNum(t.exit_price) + '<br/>' +
        '数量 ' + fmtNum(t.quantity) +
        ' &nbsp;收益率 ' + fmtPct(t.return_pct) +
        ' &nbsp;净利润 <span class="' + cls + '">' + pnl + '</span></div>';
    });
    return rows.join('<hr style="border:none;' +
      'border-top:1px solid #eee;margin:6px 0"/>');
  }

  klineChart.subscribeCrosshairMove(function (param) {
    var tip = el('trade-tooltip');
    if (!param.time || !param.point || currentSymbol == null) {
      tip.style.display = 'none';
      return;
    }
    var payload = APP.payloads[currentSymbol];
    if (!payload) {
      tip.style.display = 'none';
      return;
    }
    var time = param.time;
    var hits = (payload.trades || []).filter(function (t) {
      return t.entry_time === time || t.exit_time === time;
    });
    if (!hits.length) {
      tip.style.display = 'none';
      return;
    }
    tip.innerHTML = tradeTipHtml(hits, time);
    tip.style.display = 'block';
    var rect = klineNode.getBoundingClientRect();
    var x = Math.min(param.point.x + 16, rect.width - 330);
    var y = Math.min(param.point.y + 16, rect.height - 140);
    tip.style.left = Math.max(0, x) + 'px';
    tip.style.top = Math.max(0, y) + 'px';
  });

  function ensurePayload(code) {
    if (APP.payloads[code]) {
      return Promise.resolve(APP.payloads[code]);
    }
    if (!APP.serverMode) {
      return Promise.reject(new Error('报告中未包含该标的的行情数据'));
    }
    return fetch('/api/symbol?code=' + encodeURIComponent(code))
      .then(function (resp) {
        if (!resp.ok) {
          return resp.json().then(function (body) {
            throw new Error(body.error || ('HTTP ' + resp.status));
          });
        }
        return resp.json();
      })
      .then(function (payload) {
        APP.payloads[code] = payload;
        addSymbolOption(code);
        return payload;
      });
  }

  function switchSymbol(code) {
    code = (code || '').trim();
    if (!code) { return; }
    setStatus('加载中…');
    ensurePayload(code).then(function (payload) {
      currentSymbol = code;
      el('symbol-input').value = code;
      renderKline(payload);
      renderTradesTable(payload);
      setStatus(
        '当前复盘：' + code +
        '（' + (payload.candles || []).length + ' 根K线，' +
        (payload.trades || []).length + ' 笔交易）'
      );
    }).catch(function (err) {
      setStatus('无法加载 ' + code + '：' + err.message, true);
    });
  }

  el('symbol-go').addEventListener('click', function () {
    switchSymbol(el('symbol-input').value);
  });
  el('symbol-input').addEventListener('keydown', function (ev) {
    if (ev.key === 'Enter') { switchSymbol(el('symbol-input').value); }
  });

  if (APP.initialSymbol) {
    switchSymbol(APP.initialSymbol);
  } else {
    setStatus('请输入股票代码开始复盘');
  }
})();
</script>
</body>
</html>
"""
