# 附录 B：引用、许可与勘误

## B.1 如何引用本教材

如果本教材对你的课程、研究或项目有帮助，欢迎引用。由于教材随仓库持续更新，建议在引用时注明**访问日期**与所用 **AKQuant 版本**。

**纯文本格式：**

> AKQuant Developers. 《量化投资：从理论到实战——基于 AKQuant 框架》. AKQuant 文档, 2026. https://github.com/akfamily/akquant （访问于 YYYY-MM-DD，AKQuant v0.2.45）.

**BibTeX 格式：**

```bibtex
@misc{akquant_textbook,
  author       = {{AKQuant Developers}},
  title        = {量化投资：从理论到实战——基于 AKQuant 框架},
  year         = {2026},
  howpublished = {\url{https://github.com/akfamily/akquant}},
  note         = {AKQuant 文档；访问于 YYYY-MM-DD，AKQuant v0.2.45}
}
```

> 引用前请把 `YYYY-MM-DD` 替换为实际访问日期，并把版本号替换为你实际使用的 AKQuant 版本。

## B.2 许可

本仓库（含源码、示例与文档）整体采用 **MIT 许可证**，版权归 *AKQuant developers*（Copyright © 2026）所有。完整条款以仓库根目录的 [`LICENSE`](https://github.com/akfamily/akquant/blob/main/LICENSE) 文件为准。

在遵守 MIT 许可（保留版权与许可声明）的前提下，你可以自由地复制、修改、分发与用于教学。若你计划大规模二次发布或商用本教材正文，建议先在仓库提 Issue 与维护者沟通，以便明确署名与版本对应关系。

## B.3 勘误与反馈

教材难免有疏漏。如果你发现**事实错误、失效链接、示例无法运行、文献信息有误或表述不清**，欢迎反馈：

- **提交 Issue**：前往 [github.com/akfamily/akquant/issues](https://github.com/akfamily/akquant/issues)，尽量附上章节定位、复现步骤与环境信息（系统、Python 与 AKQuant 版本）。
- **提交 PR**：协作约定以仓库 `dev` 为开发分支、`main` 为 PR 目标分支；文档类改动建议使用中文 Conventional Commits（如 `docs(textbook): ...`）。改动示例脚本时请先真正运行通过（exit 0），并通过 `uv run ruff check` 与 `uv run ruff format --check`。

高质量的勘误与补充会被吸收进后续版本，并在变更记录中体现。

## B.4 免责声明

本教材内容仅供**教育与研究用途**，不构成任何投资建议。书中所有策略、参数与回测结果均为教学示例：

- **回测不代表未来收益**：历史表现无法保证未来结果，过拟合会让样本内的漂亮曲线在实盘中迅速失效（参见第 11 章）。
- **实盘有风险**：杠杆、流动性、系统故障与极端行情都可能造成重大损失（参见第 15 章的事故复盘）。
- **请独立决策并自担风险**：在投入真实资金前，务必充分理解策略逻辑、成本结构与风险约束，并在模拟盘中长期验证。

---

> 愿你在量化研究的道路上既保持**对数据的敬畏**，也保持**对逻辑的执着**。
