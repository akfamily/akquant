1. 必须激活本地的 conda 环境，才行执行相关命令，通过 `conda activate <env_name>` 激活环境。
2. 生成的代码符合 pep8 规范
3. 生成的代码要通过 `ruff check .` 和 `mypy .` 检查。
4. 如果有开源项目或者文档，则使用 context7 的 mcp
5. 本项目是一个基于 AKQuant 的回测框架，用于回测股票、期货、期权等金融产品。
6. 文档或者注释中出现 akquant，则用 AKQuant 替换。
7. 本项目主要参考：https://github.com/nautechsystems/nautilus_trader 量化框架
8. 本项目用 PyO3 绑定 Rust 代码，实现高性能的回测引擎。绑定的 API 参考：https://github.com/nautechsystems/nautilus_trader
