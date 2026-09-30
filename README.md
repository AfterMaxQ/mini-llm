# MiniLLM

一个在 Windows 单卡上研究工具调用微调的实验项目。使用 Qwen3-1.7B 和公开工具对话数据，从数据划分、监督遮罩、LoRA 更新、小样本过拟合到断点恢复，逐步检查训练是否正确。

## 安装

需要 Python 3.11、支持 CUDA 的 NVIDIA 显卡。生成 Word 时还需要本机 Microsoft Word 和微软雅黑字体。

```powershell
python -m venv .local/venv-train
.local/venv-train/Scripts/python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.local/venv-train/Scripts/python.exe -m pip install -r configs/requirements-train.txt
$py = '.local/venv-train/Scripts/python.exe'
```

模型和数据版本固定在 `configs/models.json` 与 `configs/data-sources.json`。下载脚本核验源文件哈希，已有完整文件可重复使用。

```powershell
& $py scripts/download_assets.py --kind student
& $py scripts/download_assets.py --kind data
& $py scripts/download_assets.py --kind glaive
```

## 按顺序运行

先检查真实 CUDA 前向、反向与 NF4 运算，再整理数据和模板。

```powershell
& $py scripts/environment.py --experiment E00
& $py scripts/environment.py --experiment E01
& $py scripts/data.py
& $py scripts/template_probe.py
& $py scripts/mechanisms.py --experiment E04
& $py scripts/mechanisms.py --experiment E05
& $py scripts/tiny_train.py
& $py scripts/resume_probe.py
```

数据按任务组划分，训练集的 1k、5k 是 10k 的子集，dev 与 test 各 500 条。复现数据通过全部冻结文件哈希核对后，训练脚本才会读取。50 条抽查及其中的标注疑点见 [数据实验笔记](experiments/E02/notes.md)。

小样本实验保留 32 条独立轨迹，展开每个 assistant 回复，只监督当前回复。每 25 次参数更新检查全部标注工具调用，达到 29/32 后结束，最多更新 300 次。这个检查衡量训练样本拟合程度；评估输入使用标注历史。

各次运行自动分配 `E编号-R编号`，保存配置、源码快照、逐条结果和 checkpoint。恢复脚本默认使用本机最近一次已通过的 E06，也可通过 `--from-run` 指定运行号。

## 查看实验记录

[实验索引](docs/实验索引.md)连接各实验的中文笔记和精简结果。Word 正文来自相同笔记，证据索引位于阶段总结之后。

```powershell
& $py scripts/report.py --volume 01 --render
```

该命令生成环境与训练机制分册，调用本机 Word 导出 PDF，并渲染页面图片，供检查中文、图表和分页。完整依赖版本见 `configs/environment-train.lock.txt`；图表旁的来源文件记录数据哈希与运行号。
