# Chapter 13: Strategy Visualization and Report Analysis

This chapter is currently maintained in Chinese first.

- Chinese chapter: [第 13 章：策略可视化与报表分析](../../zh/textbook/13_visualization.md)
- Textbook home: [Chinese textbook index](../../zh/textbook/index.md)
- Practice links:
  - Primary example: [examples/textbook/ch13_visualization.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch13_visualization.py)
  - Extended example: [examples/11_plot_visualization.py](https://github.com/akfamily/akquant/blob/main/examples/11_plot_visualization.py)
  - Functional twin example: [examples/textbook/ch13_visualization_functional.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch13_visualization_functional.py)
  - Guide: [Visualization Guide](../guide/visualization.md)

Key update in this chapter:

- `result.viz.report(..., benchmark=...)` now supports benchmark comparison in the built-in HTML report.
- The report includes cumulative excess return, annual excess return, tracking error, information ratio, beta, and alpha.
- `result.benchmark_analysis(...)` exposes the same benchmark logic as a structured payload for frontends and APIs.
- `result.export_benchmark_analysis(...)` can persist benchmark analysis as JSON or parquet artifacts.
