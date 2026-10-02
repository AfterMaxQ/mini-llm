# E11：rank 与学习率对照执行计划

由当前主 Agent 使用 superpowers:writing-plans 整理，并使用 superpowers:executing-plans 执行。不使用子 Agent、不采用 TDD；沿用已批准的[项目规格](../specs/2026-09-30-agent-training-lab-design.md)。

## 固定条件

rank 对照使用同一份 5,000 条训练轨迹、相同模板和完整 500 条 dev，seed 17、一个 epoch、1,813 次更新，每 100 步及末步计算完整 dev loss。学习率对照固定 rank 16，同一份 1,000 条训练轨迹与 100 条固定 dev，初始、每 200 步及末步验证。NF4、BF16 计算、alpha=32、dropout=0.05、micro-batch=1、梯度累积=8、warmup 3%、最大长度 2048 和缓存释放保持一致；每轮完成独立工具生成，test 不参与选择。

固定学习率 1e-4，比较 rank 8 和 16。rank 16 复用已结束且通过完整生成核验的 E09 5k 条件。alpha 固定意味着 alpha/r 分别为 4、2，结论比较的是这套 rank 设置，不能把差异只归因于参数量。rank 32 的配置和参数量计算保留为可选扩展，不填写未运行成绩。

学习率实验固定 rank 16，比较 1e-4 和 5e-5；两组从同一个官方学生检查点重新训练。按固定 dev 的整条轨迹通过数比较，并列时选较低学习率，调用轮、不调用轮和截断另列。不同训练规模或验证清单不混为同一组提升；2e-4 为可选扩展。

## 入口与准备

- [x] 创建 `configs/sft-rank8.json`、`configs/sft-rank32.json` 和 `configs/rank.json`。与有效的 5k 基础配置逐字段比较，只有 rank 改变。
- [x] 在 `scripts/scale.py` 支持每项任务显式指定训练配置，原 E09/E10 配置行为不变。rank 队列启动前必须核验匹配的 E09 参考训练、适配器哈希和完整 dev 工具评测。
- [x] 用本地学生权重文件的真实投影形状计算各 rank 的适配器参数量，核对 rank 16 与已保存实际训练参数统计一致。该检查只读权重元数据，不构造 GPU 模型，不作为 E11 训练结果。
- [x] 保存检查命令、代码/config 哈希、模型 revision、输入数据与参考条件，并压缩归档准备证据。公共仓库只保存配置、中文主笔记、精简准备结果与本计划，不把待运行条件写成已完成实验。
- [x] 用 `scripts/notes_tuning.py` 整理准备与实际运行，`scripts/sft.py`、`scripts/offline_eval.py` 将 E11 文档回调指向该笔记。`scripts/report.py` 使用章节标题换页，避免单独的分页段落产生空白页。
- [x] 完成 rank 8 与 E09 rank 16 的同条件 5k 对照，核对实际参数、完整 loss、保存恢复和全部 500 条 dev 工具结果，保留失败和截断。
- [x] 使用 `configs/learning-rate.json` 顺序运行 1k 学习率对照。两个配置只有学习率不同；每轮核对固定 100 条 dev 的全部决策轮，每个条件另开运行号。
- [x] 归纳 E11 中文主笔记和第二册，加入真实曲线、表现比较、失败与选择理由；实际查看 Word 导出的变化页，归档并提交推送 main。

修改范围为上述配置、队列入口、笔记生成、文档回调和准备记录；不修改 README、当前训练条件、冻结数据、历史结果和已有 checkpoint。GPU 队列正在运行时只做 CPU 准备，继续使用既有单队列锁。
