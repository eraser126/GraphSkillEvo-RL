# 单面板批次技能编辑器准备记录（2026-11-08）

依据《项目设计_批次强化学习技能图与Verifier_精简版_2026-11-08.md》。本次只更新代码、任务元数据、模型和运行依赖，不调用真实 worker/editor API，不生成 SFT 示范，不进行参数训练。

## 新入口与训练逻辑

保留原 EA、早期 PPO 和旧 `run_editor` 入口；新方案使用：

```bash
python -m graphskillevo.run_batch_editor
```

单面板 manifest v2 将任务划分为 train/val/test，允许同一 skill/family 横跨任务划分，但拒绝重复任务 ID、重复有效 payload 和私有答案字段。原 80/40/280 ID 划分保持不变。

一轮组更新：同一初始图执行一批相关任务一次 → 同一份结果提供诊断和基准 → 当前编辑器冻结参数采样四个候选 → 候选逐一执行同一批任务 → 全部评分完成后调用 GRPO 更新。B=4/K=4 通常为二十次完整执行；无效和 NOOP 候选单独记录，可减少执行。没有额外 support/reward 双面板。

每个候选仍是“修改检查条件 → 原证据重检 → 修改图”两个编辑回合。检查-only 的修改不会凭空获得任务收益。新配置默认仅用批次官方平均分差；成本系数默认零，调用和 token 仍记录，后续可显式固定成本惩罚。

按任务指令关键词给出可审查的初始 family 标签，族内打乱；尾批尽量重排为多题，极少数单独的 family 可进入明确的混合尾批。不删除任务，不强制事前固定成功/失败比例。真实初始执行后成功与失败都保留，轨迹用于共同诊断。该标签是工程启发式，不是已验证的语义聚类。

默认每批从原始图开始。`state_bank_size>0` 可开启仅来自训练候选的中间图池，与原始图交替作为状态；只保存训练面板上取得正向改善的合法候选。默认关闭，不把图池描述成独立的进化算法。

验证阶段使用更新后、冻结的编辑器在本批训练证据上再生成一个确定性修复，用 val 成绩选择 checkpoint/图；val 结果不进入编辑器提示或梯度。测试是独立命令，不在训练循环中访问 test，不用 test 挑选候选。

## 证据与执行

- 新增任务指令和输出契约进入 Trace；记录普通 bash、write_file 的动作、观察及脚本正文。
- 节点结束时保存脚本源码摘要与明确截断标记，保留完整原始对话文件；不从最终状态倒填中间证据。
- 大证据提示采用标注过的部分展示，保留任务要求、最近动作/错误及节点快照；重检仍使用完整不可变 Trace。模型上下文不足会明确失败，不伪装为无效任务。
- 基础执行配置的超时统一传入工具及模型请求，新配置为 600 秒。
- E0 事后诊断仍是本次实现范围。E1 在线检查、节点重试与环境恢复没有在本次隐式加入。

## 初始模型与依赖

模型：`Qwen/Qwen3-4B-Instruct-2507`。

固定 revision：`cdbee75f17c01a7cc42f958dc650907174af0554`。

服务器目录：`/root/autodl-tmp/models/Qwen3-4B-Instruct-2507`。固定文件清单见 `configs/editor/qwen3_model.lock.json`，下载元数据和逐分片校验记录在服务器准备目录。

单独环境：`/root/autodl-tmp/envs/editor-qwen3`，不覆盖原环境；Transformers 4.51.3、Tokenizers 0.21.4、PEFT 0.15.2、Accelerate 1.6.0、Safetensors 0.5.3。当前继承原 PyTorch 2.1.2+cu118；导入、tokenizer 和微型 Qwen3 前向已检查，但没有加载 4B 权重。实际 GPU 主机还需要验证 CUDA/PyTorch、显存和 LoRA 训练兼容性，不把 CPU 准备环境称为已验证的 GPU 训练环境。

当前服务器仍无 GPU。新入口在生产训练前检查 CUDA，不会尝试在 2 GB 内存上加载 4B 模型训练。执行 agent 的 API 配置仍需提供，当前没有配置或泄露密钥。

## 安全的离线命令

```bash
cd /root/autodl-tmp/GraphSkillEvo

/root/miniconda3/bin/python -m pytest tests/editor tests/rl -q

/root/autodl-tmp/envs/editor-qwen3/bin/python -m graphskillevo.run_batch_editor \
  --config configs/editor/spreadsheet_batch.yaml \
  --manifest examples/editor/spreadsheet_verified400_initial/batch_manifest.json
```

默认 preflight，不加载模型、不调用 API。任务 manifest 只含原数据集的任务元信息，不含生成轨迹或 SFT 标签；`training_allowed=false`。后续明确启动训练时，可用 `scripts.prepare_batch_manifest --permit-training-data --out NEW_FILE` 新建允许训练的数据配置，再配合 `--allow-training`，不会覆盖原 manifest。

## 后续明确启动时的命令模板（本次未执行）

所有运行输出必须使用新的目录。下列模式都必须显式 `--allow-api`。

```bash
# 只采集指定训练批次，不编辑、不训练
python -m graphskillevo.run_batch_editor --config configs/editor/spreadsheet_batch.yaml \
  --manifest examples/editor/spreadsheet_verified400_initial/batch_manifest.json \
  --mode collect --batch-index 0 --out-root /root/autodl-tmp/NEW_collection --allow-api

# 冻结 API 编辑器生成四个候选并评分，不更新参数
python -m graphskillevo.run_batch_editor --config configs/editor/spreadsheet_batch.yaml \
  --manifest examples/editor/spreadsheet_verified400_initial/batch_manifest.json \
  --mode candidates --api-editor --out-root /root/autodl-tmp/NEW_candidates --allow-api

# 仅在已准备 CUDA、生产模型和允许训练的数据配置后使用
python -m graphskillevo.run_batch_editor --config configs/editor/spreadsheet_batch.yaml \
  --manifest /absolute/path/to/training_manifest.json --mode train \
  --model-path /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
  --out-root /root/autodl-tmp/NEW_training --allow-api --allow-training

# 冻结最终图测试，不再选择或编辑；会执行整个 test 划分
python -m graphskillevo.run_batch_editor --config configs/editor/spreadsheet_batch.yaml \
  --manifest examples/editor/spreadsheet_verified400_initial/batch_manifest.json \
  --mode evaluate --graph /absolute/path/to/selected_graph.json \
  --out-root /root/autodl-tmp/NEW_test --allow-api --allow-test
```

`adapt` 模式允许冻结编辑器用指定训练批次生成单个图，输出 `adapted_graph.json`；不访问测试答案。对“未见技能的编辑能力”做更强的评估，还需要独立准备未用于训练/选 checkpoint 的证据和评价任务，不能把这张最终图的测试分数等同于该能力已经被证明。

## 限制与记录

检查条件的有效性、真实检查点调用率、上下文长度和奖励差异仍待真实联调。检查通过不等于节点完全正确。宿主机 bash 不是权限沙箱；自动断点续训未开放，已有输出目录会拒绝运行；保存的 RNG/optimizer/state bank 只为后续恢复提供基础。

服务器准备目录：`/root/autodl-tmp/batch-editor-preparation-2026-11-08/`。含修改前备份、原结果哈希、测试日志、模型官方清单、环境安装与校验记录。本次不创建 Git 提交，不改原始 skill 或已有实验结果。

## 最终检查结果

- 原环境：80 项离线测试通过；独立 Qwen3 环境：80 项通过。包含原 EA/PPO 相关回归、新的单面板批次流程、先生成整组再更新、任务隔离、证据进入提示及压缩保留目标/最近错误等检查。
- 新环境 tokenizer 与模型配置可离线读取，微型随机 Qwen3 前向输出有限；未加载生产 4B 权重，未执行真实模型训练。
- 新入口离线 preflight 显示 80/40/280、四候选，真实训练任务组成 21 批，每批 2–4 题；model_loaded/api_called/training_started 均为 false。
- 初始 skill 和修改前记录的五个实验文件哈希保持一致；编译与依赖检查通过。
- 模型分片按官方 LFS SHA-256 核对，普通文件按官方 Git blob ID 核对；逐张量名称、形状、偏移及索引引用另做检查。官方索引 `metadata.total_size` 为 8,045,591,552，实际 BF16 张量合计 8,044,936,192，相差 655,360 字节。保留该上游原始汇总字段，不修改文件；完整性判断以匹配官方哈希和逐张量检查为依据，差异记入 `model-integrity.json`。
- 中间因权重传输期间资源竞争而中断的测试日志保留；最终完成的日志为 `tests-legacy-final.log` 和 `tests-qwen-clean.log`，不把中断运行计作通过。
