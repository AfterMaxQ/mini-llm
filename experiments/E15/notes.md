# E15：4B 教师的工具任务表现

## 当前有效证据

E15-R02 完成了冻结 Pi dev16 的进程执行，但本机 API 未把 Pi 的文本块数组还原为用户文本。逐题运行记录显示，16 个冻结任务对应 10 种用户提示，而每题首次请求均为相同的 2,266-token 模型输入；因此 R02 仅保留为接线诊断，不作为教师能力成绩。

修复后的教师对照尚未完成。有效比较需使用接线已通过输入哈希检查的 E14 Pi dev 运行作为学生同集参照；在有效教师结果产生前，不对教师相对学生的表现下结论。

## 证据索引

- E15-R02 原始配置、逐题记录、模型 API 轨迹与进程结果：`.local/runs/E15-R02/`
- E15-R02 状态摘要：`experiments/E15/runs/E15-R02.json`；标记为 `invalid_harness`，原始文件保持不变。
- Pi 文本块接线审计与受影响运行清单：`experiments/E13/prompt-delivery-audit.json`
- E15 教师条件：`configs/pi-agent-e15-teacher-dev.json`；本地模型清单：`.local/models/Qwen3-4B/download-manifest.json`
- 冻结任务：`configs/subsets-frozen.json` 的 `pi_dev`
