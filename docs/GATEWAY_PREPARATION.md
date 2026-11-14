# 网关 API 接入记录

服务器仓库：`/root/autodl-tmp/GraphSkillEvo`。

- 原始网关：`https://maas.tatucloud.com`，兼容客户端使用 `https://maas.tatucloud.com/v1/`。
- 执行模型：`gpt-6.1-sol`；推理强度沿用 `medium`，未启用 pro。
- 私有配置：`/root/.config/graphskillevo/gateway.json`，目录 0700、文件 0600，密钥不写入仓库、文档、命令行参数或输出配置。
- 通过正常项目入口配置模型时自动读取；可用 `GRAPHSKILLEVO_GATEWAY_FILE` 指定另一个私有文件，或以 `GATEWAY_BASE_URL/GATEWAY_API_KEY/GATEWAY_MODEL_NAME` 环境变量覆盖。
- API worker 和显式选择的冻结 API 编辑基线使用该网关；强化学习的本地 Qwen3 编辑器仍从 `--model-path` 加载，不改成训练 API 模型。

修改包括私有配置读取模块、模型初始化接入、GPT-6.1 Sol 的 Responses API 路由，以及 Responses 工具上下文和图像输入的保留。没有自动退回其他模型或静默删除推理设置。

只读连接检查：`GET /v1/models` 返回 HTTP 200，列表中存在所配置的模型。此结果只证明认证和模型列表接口可用，不证明第三方网关完整支持 Responses 生成、推理参数或多轮工具调用；本次没有发起生成请求、执行 benchmark 或启动训练。

项目文本文件扫描未发现真实密钥。修改前备份及脱敏状态保存在 `/root/autodl-tmp/gateway-preparation-2026-11-08/`。离线回归测试记录在该目录的 `tests-final.log`。
