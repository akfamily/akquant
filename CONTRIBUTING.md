# 贡献指南 (Contributing Guide)

感谢你对 AKQuant 的关注！我们需要你的帮助来让这个项目变得更好。无论你是修复 Bug、改进文档，还是增加新功能，我们都非常欢迎！

为了方便“萌新”上手，我们准备了这份详细的 GitHub 合作开发指南。

## 🚀 开发流程 (Workflow)

我们采用 **Git Flow** 的简化模式进行开发。

- **`main` 分支**: 稳定分支，对应 PyPI 发布的版本。
- **`dev` 分支**: 开发分支，所有的日常开发和 PR 都应合并到此分支。

### 1. Fork & Clone (复刻与克隆)

1.  **Fork 项目**: 点击 GitHub 页面右上角的 `Fork` 按钮，将 `akquant` 仓库复刻到你自己的账号下。
2.  **Clone 到本地**:
    ```bash
    # 将 <your-username> 替换为你的 GitHub 用户名
    git clone https://github.com/<your-username>/akquant.git
    cd akquant
    ```
3.  **设置上游仓库 (Upstream)**:
    为了保持你的代码与官方仓库同步，需要添加上游仓库地址：
    ```bash
    git remote add upstream https://github.com/akfamily/akquant.git
    ```

### 2. 环境搭建 (Setup)

本项目混合了 Rust 和 Python，请按以下步骤配置环境：

1.  **安装 Rust**: [官网下载](https://www.rust-lang.org/tools/install)
2.  **创建 Python 虚拟环境 (推荐 uv)**:
    ```bash
    uv venv --python 3.10
    # Windows:
    .venv\Scripts\activate
    # macOS / Linux:
    source .venv/bin/activate
    ```
3.  **安装依赖与编译**:
    ```bash
    # 安装开发依赖
    uv pip install -e ".[dev,ml,plot]"

    # 编译 Rust 扩展 (开发模式)
    # 注意：不要使用 `uv run maturin develop`，它会先尝试安装当前项目，
    # 对于以 maturin 作为 build backend 的仓库可能卡在 Preparing packages。
    uvx maturin develop
    ```

### 3. 开始开发 (Coding)

1.  **同步最新代码**:
    每次开发前，先确保你的本地 `dev` 分支是最新的：
    ```bash
    git checkout dev
    git pull upstream dev
    ```

2.  **创建功能分支**:
    **不要**直接在 `dev` 或 `main` 上修改。请为每个任务创建一个新分支：
    ```bash
    git checkout -b feature/my-new-feature
    # 或者修复 bug
    git checkout -b fix/bug-fix-name
    ```

3.  **编写代码**:
    *   遵循 PEP 8 编码规范。
    *   确保添加了类型注解 (Type Hints)。
    *   如果是 Python 代码，请在提交前运行检查（包含 ruff、ruff-format、mypy 与文档检查）：
        ```bash
        uv run pre-commit run --all-files
        ```
        说明：
        *   ruff 与 mypy 的版本在 `.pre-commit-config.yaml` 与 `pyproject.toml` 中钉为一致；改代码过程中可用 `uv run ruff check .` 快速自查，但以 pre-commit 结果为准。
        *   不要直接运行 `uv run mypy .`：开发环境中的 numpy 2.5 存根用了需 Python 3.12 的语法，按项目的 3.11 目标检查会产生误报，环境里的 torch 等可选依赖也会改变结论。pre-commit 的 mypy hook 使用钉死版本的独立环境，不受影响。
        *   仓库中的 mypy hook 已配置为以项目根目录运行并读取 `pyproject.toml`，用于避免 `__init__.py` 与 `__init__.pyi` 的重复模块报错。
        *   如果你修改了 pre-commit 配置或首次运行遇到环境问题，可执行：
            ```bash
            pre-commit clean
            pre-commit install --install-hooks
            ```

### 4. 提交与推送 (Commit & Push)

1.  **提交代码**:
    ```bash
    git add .
    git commit -m "feat: 添加了xxx功能"
    ```
    *(推荐使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式)*

2.  **推送到你的 Fork 仓库**:
    ```bash
    git push origin feature/my-new-feature
    ```

### 5. 版本与 Changelog 规则 (Versioning & Changelog)

为避免版本信息分散或互相冲突，AKQuant 统一采用以下规则：

- `pyproject.toml` 中的 `project.version` 作为 Python 包版本号来源。
- `Cargo.toml` 中的版本号必须与 `pyproject.toml` 保持一致。
- 正式发布由形如 `v0.2.10` 的 Git tag 触发。
- `CHANGELOG.md` 只保留 `Unreleased` 和当前 `0.2.x` 主线版本记录。
- 更早的版本历史不再继续堆积在 `CHANGELOG.md` 中，统一以 Git tag、GitHub Releases 或历史提交记录为准。

维护 `CHANGELOG.md` 时请遵循以下约定：

- 日常开发只更新 `Unreleased` 区块。
- 发版时，将 `Unreleased` 中与本次发布相关的内容整理为对应的 `0.2.x` 正式版本条目。
- 变更说明优先写“用户可感知的行为变化”，避免只写纯内部重构细节。
- 建议使用 `Added`、`Changed`、`Fixed`、`Removed` 分类。

### 6. 测试与验证 (Testing)

为确保代码质量与回测结果的一致性，本项目引入了**黄金测试套件 (Golden Test Suite)**。

1.  **运行常规单元测试**:
    ```bash
    uv run pytest
    ```

    Rust 核心测试（推荐使用项目脚本，自动处理 macOS + uv 环境动态库路径）：
    ```bash
    ./scripts/cargo-test.sh -q
    ```

2.  **运行黄金测试 (Golden Tests)**:
    黄金测试用于捕捉核心算法变更导致的非预期行为（如 PnL 计算、撮合逻辑差异）。
    ```bash
    uv run pytest tests/golden/test_golden.py
    ```
    *   如果测试失败，请检查差异是否符合预期。
    *   如果是**预期内的算法改进**（例如修复了撮合 bug 导致成交价格变动），你需要更新基线：
        ```bash
        uv run python tests/golden/runner.py --generate-baseline
        ```
        并在 PR 中说明导致基线变更的原因。

3.  **代码风格检查**:
    ```bash
    uv run pre-commit run --all-files
    ```
    如需仅检查本次变更文档，可使用：
    ```bash
    uv run python scripts/check_docs_links.py --changed-only --from-rev origin/main --to-rev HEAD
    uv run python scripts/check_docs_api_examples.py --changed-only --from-rev origin/main --to-rev HEAD
    ```

---

## ✅ 提交前的检查清单

在提交 PR 之前，请检查：

- [ ] 代码可以通过 `uvx maturin develop` 编译成功。
- [ ] 运行了 `uv run pre-commit run --all-files` 且全部通过（含 ruff、mypy、文档链接与文档 API 示例检查）。
- [ ] 如果是新功能，是否添加了简单的测试或示例？
- [ ] 文档是否已更新？

## ❓ 遇到问题？

如果你在配置环境或提交代码时遇到困难，欢迎在 [Issues](https://github.com/akfamily/akquant/issues) 中提问，我们会尽快回复！

再次感谢你的贡献！🎉
