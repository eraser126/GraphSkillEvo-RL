# 服务器代码完善记录

日期：2026-11-02。依据《强化学习技能编辑器与可学习节点验证项目设计》完善代码。本次只准备代码、依赖和离线验证，没有启动训练或真实 benchmark 实验。

## 代码位置与兼容性

- 服务器仓库：`/root/autodl-tmp/GraphSkillEvo`。
- 当前分支：`feature/grpo-verifier-editor`。
- 上游基线：`722a8a913b05dc9d1b4078f0bec604cde4dc7529`。
- 原 EA 入口：`python -m graphskillevo.run`，保留。
- 早期算子选择 PPO：`python -m graphskillevo.run_rl`，保留。
- 新 LLM 编辑器入口：`python -m graphskillevo.run_editor`。
- 本地新增代码副本：`D:\project\skill_optimization\implementation`。

本次新增路径训练的是语言模型编辑器，不把早期小型 PPO 分类控制器称为语言模型训练。改动在原仓库内，未推送或创建提交；新分支保留此前的未提交修改。

## 已实现的代码

| 模块 | 实现内容 |
|---|---|
| `editor/schema.py` | 原 Markdown 导入、稳定节点 ID、工作流与检查条件、原风格执行文档渲染；整体补丁校验与回滚；允许环与重访 |
| `editor/evidence.py`、`recorder.py` | 节点时刻快照、内容哈希、不可变记录、采集 hook；标记与证据对齐；区分明确跳过、未对齐和图外标记 |
| `editor/verification.py` | 固定规则与可选固定 LLM 检查；可编辑条件；pass、fail、unknown、not_applicable；缺证据不假定通过 |
| `editor/diagnosis.py` | 检查与任务结果矛盾、跳过和重访、实际产物引用的上游候选；不宣称因果定位 |
| `editor/episode.py` | 同一编辑器先改检查条件，再重检同一份证据，最后生成图补丁；奖励任务不进入提示上下文 |
| `editor/policy.py` | 本地 causal LLM、精确 completion token 概率、可选 LoRA；API 编辑器仅作冻结基线 |
| `editor/grpo.py` | 组相对优势、completion-token clipped GRPO；两阶段共享序列奖励；梯度有限性与裁剪；不训练 worker |
| `editor/runner.py` | 固定面板对照、训练 checkpoint、独立 dev 选择、冻结单候选适配、独立 query 评价及制品哈希 |
| `editor/data.py` | skill/family 分组，support/reward/query 任务和 payload 重叠检查，训练许可与来源；禁止 SkillsBench 单元进入权重训练 |
| `editor/backends.py` | 原 EnvAdapter 桥接，以及外部 harness 的隔离请求／响应接口 |
| `editor/skillsbench.py` | 固定源码 commit、技能覆盖层、保留原技能元信息与其他技能、官方 BenchFlow CLI 封装；默认只生成离线执行计划 |
| `editor/accounting.py` | 持久调用与 token 记录，记录可见内部 API 调用；异常不伪造为能力失败 |

另有 `configs/editor/minimal.yaml`、手写契约 fixture、离线转换脚本、可选依赖文件及服务器 `docs/EDITOR_PREPARATION.md`。服务器也保存本次项目设计原文。

## 验证情况

截至完成时，58 项离线测试通过，包含原 20 项 PPO 回归和新增测试。语法检查、三个入口帮助、`git diff --check` 和 `pip check` 均通过。

验证覆盖：事务补丁、旧版本拒绝、节点删除后的引用完整性、检查不泄漏到 worker 文档、缺快照为 unknown、实际产物依赖回看、检查修改进入第二阶段提示、基础设施错误传播、数据隔离、默认运行保护、GRPO 概率和梯度、冻结适配、官方技能正文覆盖与结果归一化。

模型接口检查使用临时创建的未训练微型 GPT-2，测试本地加载、短生成、概率重算一致及 LoRA 保存／重载。GRPO 更新契约用不修改参数的模拟 optimizer 检查；未执行真实 optimizer 参数更新，没有 toy 学习曲线实验。

离线配置检查明确返回 `model_loaded=false`、`api_called=false`、`training_started=false`。手写两步 replay 是程序契约检查，不是模型自主修复结果。官方表格技能已离线转换为 9 个节点的 JSON 示例，未执行任务。

服务器现有 PyTorch 2.1.2+cu118 保持不变；新增 Transformers 4.44.2、PEFT 0.13.2、Accelerate 0.34.2、Tokenizers 0.19.1、Safetensors 0.4.5。完整版本见 `docs/environment-editor-prepared.freeze.txt`。这是兼容当前 Python 3.10 / Torch 2.1 的准备环境，不保证所有最新模型架构都支持。

## 仍需真实环境验证的部分

当前没有选择或下载生产用编辑器模型，没有真实 API 凭据配置，没有完整训练数据或经过审查的 support/query 划分，也没有启动 BenchFlow、Docker 或真实 SkillsBench。

SkillsBench wrapper 已实现并用文件 fixture 测试，但需要在独立 Python 3.12+ 的 BenchFlow 环境中核对固定版本、结果 usage 字段和真实任务。普通轨迹日志不能替代节点时刻快照；需要将 recorder 接入实际 harness hook，缺失时保守返回 unknown。程序中没有制造中间状态。

已保存模型、optimizer 和 RNG 状态，但尚未开放自动断点恢复命令；真实训练中断后不能换新输出目录、重置预算并把它当作续训。新技能适配首版采用单候选和固定预算，不用 query 选图。

预算在操作边界检查，不是 API 服务商级的硬费用上限。外部 wrapper 必须报告其内部调用和 token，旧 adapter 或 SDK 失败请求的不可见用量仍需联调核验。

## 安全的离线命令

```bash
cd /root/autodl-tmp/GraphSkillEvo
python -m graphskillevo.run_editor --config configs/editor/minimal.yaml --manifest examples/editor/manifest.json
python -m pytest tests/rl tests/editor -q
```

默认不会加载模型或发出模型请求。真实运行要显式 `--allow-api`；训练还需 `--allow-training` 和可训练数据、模型目录。示例 manifest 标记为 explore，不可拿来启动训练。

## 日志与备份

- 离线测试：`/root/autodl-tmp/editor_offline_tests_2026-11-02.log`。
- 离线配置检查：`/root/autodl-tmp/editor_preflight_2026-11-02.json`。
- 依赖安装：`/root/autodl-tmp/editor_dependency_setup_2026-11-02.log`。
- 改动前快照：`/root/autodl-tmp/GraphSkillEvo-before-verifier-2026-11-02.tar.gz`。
- 完整补丁：`/root/autodl-tmp/GraphSkillEvo-editor-prepared-2026-11-02.patch`。
- 完整源码备份：`/root/autodl-tmp/GraphSkillEvo-editor-prepared-2026-11-02.tar.gz`。

下一步是核验真实任务与模型的运行条件，并在单独授权后进行最小真实联调；本次没有开启该阶段。
