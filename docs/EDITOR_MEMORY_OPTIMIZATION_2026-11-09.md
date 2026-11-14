# 编辑器显存优化记录（2026-11-09）

目标：在 RTX 4090 24GB 上保留现有训练算法与配置，降低本地 Qwen3 编辑器的显存占用。服务器仓库为 `/root/autodl-tmp/GraphSkillEvo`。

## 保留的训练逻辑

- 单面板多任务批次 B=4，每组候选 K=4，每个候选两轮编辑。
- 当前 GRPO 的奖励、优势、裁剪损失、候选/轮次权重、梯度累积与每组参数更新时机。
- Qwen3-4B-Instruct-2507、BF16 基座、LoRA rank=8、学习率及数据划分。
- 总上下文上限 16384、输出上限 2048；不缩短轨迹或静默截断证据。
- 任务执行器、verifier、官方评分器、初始图和历史结果。

## 实现变化

`graphskillevo/editor/policy.py`：启用 non-reentrant 梯度检查点，只打开 decoder stack 的检查点训练标志，内部 dropout 继续关闭。对支持 `logits_to_keep` 的模型，只计算 completion 对齐所需的 logits；按 128 token 分块计算完整词表的精确 log-softmax，检查点重算各块，降低 FP32 中间张量占用。保留全部输入与原有因果位置关系；未改变采样温度、top-k/p 或 token 损失定义。

`graphskillevo/editor/grpo.py`：每轮反向传播结束后及时释放局部张量引用，仍按原规则累积梯度并更新一次。

新增 `tests/editor/test_memory.py`，对照原始 dense 实现检查概率、梯度、LoRA 梯度及 dropout 行为。新增 `scripts/check_editor_memory.py`，只做合成输入反向传播和显存测试，无 optimizer.step、任务执行或 API 调用。

## 环境原因

原编辑器环境 PyTorch 2.1.2+cu118 在 RTX 4090（sm89）上不支持 head_dim=128 的 Flash Attention 训练路径。独立 attention 探针已复现该限制；长序列回退路径曾申请约 32 GiB 临时显存。编辑器独立虚拟环境升级到 PyTorch 2.5.1 和匹配的 torchvision 0.20.1，系统旧环境保留。复现依赖见 `requirements-editor-cuda24gb.txt`。

## 验证结果

独立环境最终为 PyTorch 2.5.1+cu124 / Transformers 4.51.3 / PEFT 0.15.2，模型 attention 路径为 SDPA；系统 PyTorch 2.1.2+cu118 未改变。

全部 89 项离线测试通过（1 条已有 GPT2/LoRA Conv1D 兼容提示）。实机使用原 Qwen3-4B、BF16、LoRA rank=8，预留 Adam 状态后的合成反向传播结果如下：

| 输入 token | 输出 token | 峰值已分配 GiB | 峰值保留 GiB | 结果 |
|---:|---:|---:|---:|---|
| 2048 | 2 | 8.688 | 8.994 | 通过 |
| 6144 | 2048 | 11.573 | 12.836 | 通过 |
| 14336 | 2048 | 15.338 | 17.080 | 通过 |

三组均取得 504 个适配器梯度张量，梯度有限且存在非零值；测试前后可训练参数逐项相等。未运行 optimizer.step，未执行 benchmark，未调用 API。测试结束后 GPU 显存占用为 0 MiB。

概率和梯度等价性测试覆盖 GPT2、Qwen3、带非零 LoRA B 权重的 Qwen3，以及单 token 对齐。测试以原始完整 logits + log-softmax 为参考，检查数值容差内的一致性；不承诺不同 CUDA 内核逐 bit 一致。

备份逐文件核对确认，已有代码仅 policy.py、grpo.py 改变，批次配置和其他编辑执行逻辑未变；已记录的 5 个历史最佳 skill 文件 SHA-256 相同。`gpu-probe-final.json`、`regression.log`、`preservation-audit.json` 为对应证据。

## 可重复的离线检查命令

```bash
cd /root/autodl-tmp/GraphSkillEvo
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  /root/autodl-tmp/envs/editor-qwen3/bin/python -m pytest -q tests/editor tests/rl

OMP_NUM_THREADS=1 /root/autodl-tmp/envs/editor-qwen3/bin/python -m scripts.check_editor_memory \
  --model /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
  --out /root/autodl-tmp/editor-memory-2026-11-09/recheck-NEW.json \
  --allow-model-load
```

输出文件须使用新名字，脚本拒绝覆盖。显存测试预留 Adam 一、二阶状态等量存储，并验证梯度有限且非零、可训练参数没有被修改。这仍不等于完整 benchmark/RL 运行，不能证明候选质量、奖励差异或收敛性。

修改前代码备份与诊断日志位于 `/root/autodl-tmp/editor-memory-2026-11-09/`。本次不启动正式训练，也不消费任务 API。
