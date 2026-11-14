# SpreadsheetBench Verified 400 与节点 verifier 准备记录

日期：2026-11-07。服务器仓库：`/root/autodl-tmp/GraphSkillEvo`，分支 `feature/grpo-verifier-editor`。本次准备数据、依赖、官方评测源码与节点检查代码，未执行真实题目，未生成 SFT 数据，未启动 SFT、强化学习或进化搜索。

## 1. 数据与运行环境

- 官方发布地址：<https://huggingface.co/datasets/KAKA22/SpreadsheetBench/blob/main/spreadsheetbench_verified_400.tar.gz>。
- 原始压缩包：`data/downloads/verified_400.official.tar.gz`。
- 解压目录：`data/spreadsheetbench_verified_400`。
- 压缩包 SHA-256：`10ef893dd29cb13ab97143ea787e68cdc9574a13873ab9a54e50b31dc03fc949`，与官方发布文件一致。
- 已核对 400 个唯一任务、800 个工作簿的 ZIP 完整性、输入与答案配对、文件哈希。没有在这些任务上运行 agent 或评分。
- 复用 `data/spreadsheetbench_id_split`：train 80、val 40、test 280，互不重叠并覆盖全部 400 题，没有重新随机划分。
- 395 题使用带编号的 init/golden 文件；5 题使用 `initial.xlsx/golden.xlsx`。任务 `42930` 的原始答案文件名为 `1_43930_golden.xlsx`；以显式兼容映射处理，不重命名或修改原始文件。逐题映射写入数据审计 JSON。

Python 运行依赖：openpyxl 3.1.5、pandas 2.2.3、numpy 1.26.4、tqdm 4.67.1。新增 `requirements-spreadsheet-preparation.txt`，保留已有模型和训练依赖版本，未按官方仓库旧的完整推理 requirements 降级整个环境。

Ubuntu 源的 LibreOffice 为 7.3，不满足官方说明中的 7.5+。已另装官方 LibreOffice 7.6.7.2，执行路径为 `/opt/libreoffice7.6/program/soffice`，新后端显式使用此路径。普通 `libreoffice` 命令仍可能指向 Ubuntu 的 7.3，不应拿它代替上述路径。

安装包来自 Document Foundation 官方归档，SHA-256：`5fbd379bd9cedb037fa00b6e7e830619bff503a9451bc321e2b4e8d646081920`。

## 2. 官方评测与节点检查分离

官方仓库：<https://github.com/RUCKBReasoning/SpreadsheetBench>，固定提交：`49b73a94775fb489063f60ca1865e3a650079a79`。

`vendor/SpreadsheetBench` 保存该提交未修改的 `evaluation/evaluation.py`、`evaluation/open_spreadsheet.py`、README 和 requirements；这是固定源码快照，不是完整 Git 克隆。文件哈希保存在 `configs/editor/spreadsheet_official.lock.json`，加载时校验。

新增 `graphskillevo/editor/spreadsheet_official.py`：

1. 用官方 `just_open_libreoffice` 重算**预测文件的临时副本**，不修改原预测、输入或答案。
2. 调用官方 `compare_workbooks`，不使用节点 verifier 代替任务评分。
3. 适配 Verified400 的文件命名和公开的 `answer_sheet` 字段。官方旧 CLI 硬编码原版三测试用例及旧文件名，因此不直接运行它，也不修改它的比较函数。
4. 不向 worker 返回标准答案值、官方逐单元格对错或评测诊断。重算基础设施失败会抛异常，不伪造成任务得零分。

## 3. 原始技能图与初始检查

唯一技能来源是 `graphskillevo/initial_skills/spreadsheetbench/init_skill.md`，不读取 `ckpt` 中的进化最佳 skill。

原 Markdown 未修改，其 SHA-256 为 `8229fedeee7abaf003b68036ec9b45daa01fd6af2566134503ee8055d15e3a11`。保留 6 个粗粒度节点、一条工作流、节点正文和全局指导。生成物位于 `examples/editor/spreadsheet_verified400_initial/`，包含图 JSON、worker 文档、来源记录和仅用于离线检查的合成 manifest。

| 原节点 | 稳定 ID | 初始检查 |
|---|---|---|
| Explore Input File | `n_cd272bc98375` | 节点结束时输入工作簿是否可打开 |
| Select Library | `n_e656c84b04cf` | 已捕获脚本的静态导入是否符合库约束 |
| Write `solution.py` | `n_934fa4a006f8` | Python 语法；INPUT_PATH/OUTPUT_PATH 赋值声明 |
| Save Output | `n_7f636116d9b4` | 脚本是否声明对 OUTPUT_PATH 的保存调用 |
| Execute | `n_e3a67d9540e9` | 实际执行回执；输出文件是否存在 |
| Confirm | `n_eebdd24a77d9` | 输出可打开；Excel 错误单元格；公式缓存可观察性 |

原工作流中 Save Output 在 Execute 前，因此这里检查保存调用的静态声明，实际落盘检查放在 Execute 后。若 Select Library 结束时还没有脚本，该检查返回 fail（实际捕获到脚本不存在）；没有可靠节点快照则返回 unknown。可根据后续真实轨迹修改检查时机或条件，不能用后来的脚本补填早期证据。

这些检查描述可观察事实：语法正确不等于脚本能运行；有保存调用不等于已经保存；没有 Excel 错误不等于答案正确；存在公式缓存不代表缓存新鲜或正确。

## 4. 固定接口、可编辑条件与证据

新增 `graphskillevo/editor/spreadsheet.py`，扩展原 `CheckEngine`，沿用事务式检查编辑与同一不可变证据重检流程。

固定接口包括：`readable`、`exists`、`no_cell_errors`、`formula_cache`、`syntax`、`library_policy`、`paths_defined`、`save_declared`、`execution_success`、`sheets_present`。

条件示例：

```json
{
  "check_id": "output_readable",
  "applies_when": "always",
  "target": "output",
  "requirement": "Captured output workbook can be opened",
  "evidence_needed": ["output"],
  "method": "rule",
  "rule": {"kind": "spreadsheet", "path": "/output", "value": {"check": "readable"}}
}
```

编辑器可以增删检查、修改接口选择、目标与受支持参数（如 allowed/names 列表），不能修改检查器代码或执行任意 Python 谓词。新的接口说明已进入编辑器提示。

输出沿用 `pass/fail/unknown/not_applicable`，每项带节点 ID、检查 ID、快照 ID、事件位置和证据键；快照内容有哈希，工作簿证据带文件哈希与单元格引用。

工作簿信息包括工作表名称、行列维度、合并区域、表名、命名区域、工作表状态、有限单元格样本、公式/缓存、Excel 错误。扫描有单元格和工作表数量上限；未完整扫描不能宣称整个工作簿没有错误或没有公式。没有公式时缓存检查为不适用；缓存缺失时无法区分尚未计算与合法空结果，返回无法判断。

脚本使用 AST 静态检查，不执行检查条件或待检查代码；动态导入、exec/eval 等无法静态确认的库约束返回无法判断。

## 5. 接入执行器和编辑流程

在原 ReAct 执行器上增加**可选** evidence_session；原有调用默认行为保持兼容。新 SpreadsheetBackend 使用以下工具：

- `node_checkpoint(start/end/skip)`：真实调用时记录节点边界，end 由 harness 立即采集输入、脚本和输出。文本 NODE 标记不产生文件快照。
- `execute_solution`：执行一次 solution.py，记录实际返回码、超时和 stdout/stderr，并绑定当前节点访问。重访节点不会沿用旧执行回执。

检查结果只写入诊断文件，不返回失败即重试的控制信号。新路径关闭原 bash 自动检查文本，保留普通执行错误反馈；最终正确率完全来自官方评价器。测试已验证：节点检查失败可以和官方任务成功同时存在，不会强制把最终得分改成零。

新增 `backend: spreadsheet`，由现有 `run_editor` / runner 构建，可将 support 的真实 Trace 送入原检查编辑、重检和图编辑流程。此项是代码接入，未启动适配或训练。

后端每次使用新输出目录，只向工作区复制输入，不复制标准答案。轨迹、检查结果和官方评分分别保存。当前仍沿用原仓库宿主机 bash 执行方式，不构成操作系统权限沙箱。

## 6. 离线验证

完整测试：**73 passed**；包括原有回归、新增表格检查、证据缺失、时间点隔离、节点重访、路径拒绝、编辑重检、模拟 worker 后端、官方比较与 LibreOffice 重算测试。三个警告来自既有 Transformers/PEFT 测试。

合成重算测试将 `=1+2` 与字面值 3 比较，使用真实官方重算和比较函数通过，且原预测与答案文件哈希不变。没有以真实题目求解来测试。

离线 preflight 返回 model_loaded=false、api_called=false、training_started=false。全套回归含既有临时微型模型接口测试，不下载生产模型或执行训练更新。

另检查 Python 编译、pip check、git diff --check、原始 skill 与已有实验文件哈希。

## 7. 可运行命令

以下命令仅用于离线检查，均从仓库根目录执行。写文件的脚本拒绝覆盖已有输出，请换一个新的输出路径。

```bash
cd /root/autodl-tmp/GraphSkillEvo

/opt/libreoffice7.6/program/soffice --version
python -m pip check
python -m pytest tests/rl tests/editor -q

python -m graphskillevo.run_editor \
  --config configs/editor/spreadsheet_prepared.yaml \
  --manifest examples/editor/spreadsheet_verified400_initial/preflight_manifest.json

python -m scripts.audit_spreadsheet_data \
  --data-root data/spreadsheetbench_verified_400 \
  --archive data/downloads/verified_400.official.tar.gz \
  --splits data/spreadsheetbench_id_split \
  --out /root/autodl-tmp/spreadsheet-data-audit-new.json

# 将来已有真实节点轨迹时，只重检，不重新执行任务：
python -m scripts.check_spreadsheet_trace \
  --graph examples/editor/spreadsheet_verified400_initial/initial_graph.json \
  --trace /absolute/path/to/editor_trace.json \
  --out /absolute/path/to/new-node-checks.json
```

非登录 SSH 命令可使用 `/root/miniconda3/bin/python`，避免 PATH 中找不到 python。示例 preflight manifest 只含合成占位任务，不能拿来做真实适配或训练。

## 8. 剩余问题

1. 尚未调用真实 worker API，真实模型是否稳定调用 checkpoint、是否遗漏或提前结束节点，尚待后续授权的单题联调。未记录到的节点必须保留无法判断。
2. 新采集路径支持 ReAct；原 single/multi codegen 路径未获得虚构的中间节点快照，也未自动切换为新后端。
3. 初始条件以文件、语法、执行和可观察错误为主，不能自动证明复杂业务语义、上游根因或整题正确性；早期节点的条件和时机还需要真实轨迹校准。
4. 公式由 LibreOffice 重算，复杂 Excel 专有函数、外部链接及格式兼容性没有在 400 个真实任务上验证。
5. 宿主机 bash 尚无权限沙箱；真实执行前应确认所需的文件访问隔离。评测接口独立于 verifier，但这不等同于系统级隔离。
6. 没有创建真实 support/reward/query 实验面板，没有生成 SFT 记录或准备训练授权；原 benchmark 划分仅作为现有数据划分保留。

## 9. 留存记录

服务器准备目录：`/root/autodl-tmp/spreadsheet-preparation-2026-11-07/`。

- `before-code.tar.gz`、`before-status.txt`：修改前代码和 Git 状态。
- `existing-results-sha256.json`、`final-integrity.json`：已有实验文件保护与最终核验。
- `data-audit.json`：400 题配对、划分和 800 个工作簿哈希。
- `tests-final.log`、`spreadsheet-tests.log`：完整回归与表格专项测试。
- `preflight.json`、`initial-graph.json`：离线检查及初始图来源。
- `environment.freeze.txt`、pip/apt/LibreOffice 安装日志：环境记录。

本地实现副本在 `D:\project\skill_optimization\implementation`，服务器同名记录保存在 `docs/SPREADSHEET_VERIFIED400_PREPARATION.md`。
