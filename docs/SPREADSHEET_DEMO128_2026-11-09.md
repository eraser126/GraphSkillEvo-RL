# SpreadsheetBench Demo128 任务划分与覆盖

扩大 demo 为 128 题：训练 80、验证 16、测试 32。完整 400 题清单保持不变。训练保留原来全部 80 题；验证、测试只在各自原始划分中选取，不将原测试题移入训练集。未根据执行成败或官方答案选题。

| 划分 | 单元格级操作（官方） | 工作表级操作（官方） | 合计 |
|---|---:|---:|---:|
| 训练 | 53 | 27 | 80 |
| 验证 | 12 | 4 | 16 |
| 测试 | 22 | 10 | 32 |

## 操作覆盖

在两种官方大类之外，按任务指令建立多标签筛查。同一道题可有多个标签，因此以下计数不能相加为题数。优先纳入稀少标签，再填充官方大类的比例；固定随机种子 20261109。

| 启发式操作标签 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| 汇总、计算与取整 | 33 | 9 | 15 |
| 缺失值、重复值与清洗相关 | 28 | 4 | 6 |
| 条件逻辑 | 42 | 11 | 19 |
| 跨工作表相关 | 5 | 5 | 5 |
| 日期时间 | 25 | 5 | 11 |
| 筛选、排序与删除排除 | 28 | 3 | 6 |
| 格式相关 | 27 | 3 | 10 |
| 查找、匹配与间隔引用 | 33 | 9 | 14 |
| 表格重组、插入与合并相关 | 14 | 2 | 3 |
| 文本处理 | 24 | 6 | 12 |

三个集合均覆盖筛查识别的十类操作。初次筛查未识别的三题已阅读指令，分别涉及排除特定项、OFFSET 间隔引用、按指定倍数取整，并补充了匹配规则。

这些是透明的关键词标签，不是官方细分类，也不等于全部题目完成了人工语义标注。例如指令提到格式，不代表格式一定是该题的主要评分目标。此划分用于广覆盖的探索 demo，不能宣称穷尽 benchmark 的全部操作、难度分布或子类型。保留原划分，尚未完成同源任务/高度相似指令的泄漏审计。

## 文件与使用

服务器仓库 `/root/autodl-tmp/GraphSkillEvo` 下：

- `examples/editor/spreadsheet_verified400_initial/demo128_manifest.json`：可直接供现有入口读取的独立清单。
- `examples/editor/spreadsheet_verified400_initial/demo128_manifest.selection.json`：选题规则、逐题指令与标签、覆盖统计。
- `scripts/prepare_spreadsheet_demo.py`：可重复的选题脚本。拒绝覆盖已有输出。

本地对应文件位于 `D:\project\skill_optimization\implementation` 下。

服务器离线 preflight 已通过，显示 80/16/32、21 个训练批次、每批 2–4 题、K=4。未改变训练配置、原有 family 分批逻辑、初始图或 verifier。清单 `training_allowed=false`；本次只准备清单，不执行任务、调用 API 或启动训练。

```bash
cd /root/autodl-tmp/GraphSkillEvo
/root/autodl-tmp/envs/editor-qwen3/bin/python -m graphskillevo.run_batch_editor \
  --config configs/editor/spreadsheet_batch.yaml \
  --manifest examples/editor/spreadsheet_verified400_initial/demo128_manifest.json
```

上面命令仅作 preflight。后续需要真正启动时，明确使用 demo128 清单，不要误用完整 400 题清单；训练数据授权需另行显式开启。当前实验入口未被切换为自动执行。

80 道训练题一轮仍为最多 21 次有效更新；候选奖励相同会跳过更新。验证集缩小可减少周期性评估开销；32 道测试题适合初步比较，后续应保留剩余题目用于扩大验证。
