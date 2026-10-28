# Chapter 6: A-Share Market Microstructure and Strategy Practice

This chapter is currently maintained in Chinese first.

- Chinese chapter: [第 6 章：A 股市场微观结构与策略实战](../../zh/textbook/06_stock_a.md)
- Commission note: Chinese chapter now documents `percent`, `fixed`, and `per_unit` commission modes; A-share default remains `percent + min_commission`.
- Session note: the backtest matches by the real A-share sessions; orders during `OpeningAuction` / `AuctionHold` / `ClosingAuction` are matched in one batch at the auction price (auction-session data first, otherwise the first continuous bar's open or the close-time bar's close). The 09:20-09:25 no-cancel rule and STAR-market after-hours fixed-price trading are not simulated. Day-end settlement runs at the trading-day close and its snapshot is stamped at the close.
- Textbook home: [Chinese textbook index](../../zh/textbook/index.md)
- Practice links:
  - Primary example: [examples/textbook/ch06_stock_a.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch06_stock_a.py)
  - Extended example: [examples/20_risk_management_demo.py](https://github.com/akfamily/akquant/blob/main/examples/20_risk_management_demo.py)
  - Functional twin example: [examples/textbook/ch06_stock_a_functional.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch06_stock_a_functional.py)
  - Guide: [Quant Basics Guide](../guide/quant_basics.md)
