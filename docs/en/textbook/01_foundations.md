# Chapter 1: Quantitative Investment Overview and Environment Setup

This chapter is currently maintained in Chinese first.

- Chinese chapter: [第 1 章：量化投资概述与环境搭建](../../zh/textbook/01_foundations.md)
- Textbook home: [Chinese textbook index](../../zh/textbook/index.md)
- Time semantics note:
  - `bar.timestamp` is a UTC nanosecond timestamp.
  - `bar.timestamp_iso` is a UTC ISO 8601 string.
  - `self.now`, `self.to_local_time(...)`, and `self.format_time(...)` are the recommended display-layer helpers for local timezone rendering.
  - A Beijing market close such as `2023-01-03 15:00:00+08:00` may therefore appear as `2023-01-03T07:00:00Z` in structured fields. This is expected and represents the same instant.
- Practice links:
  - Primary example: [examples/textbook/ch01_quickstart.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch01_quickstart.py)
  - Extended example: [examples/01_quickstart.py](https://github.com/akfamily/akquant/blob/main/examples/01_quickstart.py)
  - Functional twin example: [examples/textbook/ch01_quickstart_functional.py](https://github.com/akfamily/akquant/blob/main/examples/textbook/ch01_quickstart_functional.py)
  - Guide: [Quickstart Guide](../start/quickstart.md)
