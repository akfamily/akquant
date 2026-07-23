# Vendored third-party assets

## lightweight-charts.standalone.production.js

- **Library**: TradingView Lightweight Charts™
- **Version**: 5.2.0
- **Source**: https://www.npmjs.com/package/lightweight-charts (`dist/lightweight-charts.standalone.production.js`)
- **License**: Apache License 2.0 (see `LICENSE.lightweight-charts`)
- **Copyright**: Copyright (c) 2026 TradingView, Inc.

该文件为 IIFE standalone 构建,加载后暴露全局 `window.LightweightCharts`。
akquant 将其内联进 `result.viz.review()` 生成的 HTML,使复盘报告
**离线自包含**(无 CDN 依赖)。升级时:`npm pack lightweight-charts@<ver>`,
取 `package/dist/lightweight-charts.standalone.production.js` 与 `package/LICENSE`
覆盖,并同步本文件的版本号。

Lightweight Charts™ is a trademark of TradingView, Inc.
