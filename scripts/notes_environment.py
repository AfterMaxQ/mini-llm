"""用 E00/E01 原始结果生成环境笔记和真实显存图。"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from lab import ROOT, sha256, write_json


def latest(experiment):
    paths = sorted((ROOT / f"experiments/{experiment}/runs").glob("*.json"))
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def main():
    env, cuda = latest("E00"), latest("E01")
    assert env["status"] == cuda["status"] == "completed"
    folder = ROOT / "experiments/E00"
    figures = folder / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    gpu = env["gpu"]
    with (folder / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "measured_at", "total_mib", "used_mib", "free_mib"])
        writer.writerow([env["run_id"], env["finished"], gpu["total_mib"], gpu["used_mib"], gpu["free_mib"]])
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    figure, axis = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    values = [gpu["total_mib"], gpu["used_mib"], gpu["free_mib"]]
    axis.bar([0, 1, 2], values, color=["#6B7280", "#D97706", "#059669"], width=0.55)
    axis.set_xticks([0, 1, 2], ["总显存", "已用显存", "空闲显存"], fontproperties=font)
    axis.set_ylabel("显存（MiB）", fontproperties=font)
    axis.set_title("RTX 4070 SUPER：环境检查时的显存快照", fontproperties=font)
    axis.set_ylim(0, max(values) * 1.2)
    axis.yaxis.grid(True, color="#E5E7EB", linewidth=0.6)
    axis.set_axisbelow(True)
    for index, value in enumerate(values):
        axis.text(index, value + 180, f"{value:,}", ha="center", fontsize=10)
    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)
    image = figures / "gpu-memory.png"
    figure.savefig(image, dpi=300)
    plt.close(figure)
    write_json(figures / "gpu-memory.source.json", {"run_id": env["run_id"], "source": "../metrics.csv", "source_sha256": sha256(folder / "metrics.csv"), "image_sha256": sha256(image)})
    (folder / "notes.md").write_text(f"""# E00：有显卡，为什么 Python 还是不能训练？

问题出在 PyTorch 安装包。本机原来的 Python 装了 CPU 版 PyTorch，显卡驱动能正常工作，Python 却用不上它。换到项目内的独立环境，安装带 CUDA 支持的版本后，GPU 才被 PyTorch 识别。原来的 Python 没有改动。

## 先分清驱动和 Python 的问题

看到 nvidia-smi 能列出显卡，很容易以为训练环境已经好了。它检查的是驱动；torch.cuda.is_available() 检查的是当前 Python 能否使用 CUDA。两边各查一次，问题就能定位到安装包，而不用先重装驱动。

在准备训练的 Python 环境中，核心检查只有几行：

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA 构建:", torch.version.cuda)
print("能否使用 GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("设备:", torch.cuda.get_device_name(0))
```

这里必须用训练环境的 Python 来跑。系统里另一个 Python 装着什么包，不会自动变成当前环境的能力。

| 检查项 | 实际结果 |
| --- | --- |
| 运行编号 | {env['run_id']} |
| 显卡与驱动 | {gpu['name']}；{gpu['driver']} |
| 原环境 PyTorch | {env['global_python']['torch']}，CUDA 不可用 |
| 独立环境 PyTorch | {env['torch']}，识别 CUDA {env['cuda_build']} |

## 下载和报告还遇到了什么？

下载公开资源时，第一次卡在 HTTPS 证书验证。改用 Windows 系统信任库后，请求成功了，全程保留证书校验。接着，Hub 下载接口缺少 HEAD 元数据，于是改成按固定版本直接 GET 下载，再核对来源提供的文件哈希。模型大文件的整包响应又等待过久，分段请求能够正常返回，便改用可续传的分段下载，最后仍核对整个文件的哈希。

报告也遇到了一次接口错误：Word 的应用对象没有预期的窗口句柄属性。改从已经打开的文档窗口读取句柄后，PDF 导出成功。随后查看实际页面，确认中文和图表能够正常显示。

## 12GB 显存，现在能用多少？

![图 E00-1：环境检查时的显存快照，单位 MiB；运行 {env['run_id']}。](figures/gpu-memory.png)

这次检查有 {gpu['free_mib']:,} MiB 空闲显存。不过，这张图只记录一个时间点：桌面程序的占用会变，加载模型后还会增加激活、梯度和优化器状态。它可以帮助安排实验，不能提前证明某个模型一定练得动。

环境到这里解决了“Python 看不见显卡”的问题。接下来还要实际做前向和反向传播；只有计算和梯度正常，才适合继续模型训练。

复习时可以回看两个问题：nvidia-smi 成功为什么不等于 PyTorch 能使用 GPU？因为驱动和 Python 安装包是两层条件。空闲显存为什么不是固定预算？因为它会随其他程序和训练阶段变化，后续仍要测训练峰值。
""", encoding="utf-8")
    probe, nf4 = cuda["cuda_probe"], cuda["nf4_probe"]
    folder = ROOT / "experiments/E01"
    (folder / "notes.md").write_text(f"""# E01：先用小矩阵，把 CUDA 和 NF4 跑一遍

这一轮的结论是：原生 Windows 环境能完成 BF16 前向、反向，也能运行 NF4 量化层并传回有效梯度。暂时没有迁移 Linux 的理由，可以继续检查真实模型。

## 为什么先不急着加载语言模型？

如果模型训练一上来报错，原因可能是模板、数据、显存，也可能是底层计算。先用小矩阵单独检查 CUDA，可以把这些问题分开。

BF16 是一种 16 位浮点格式。本轮用 1024×1024 的矩阵做前向和反向，随机种子固定为 42。先预热 5 次，再测量 20 次；计时前后等待 GPU 完成工作，避免只测到 Python 提交任务的时间。

接着检查 NF4。它把权重压缩为四位表示，计算时再按块解量化到 BF16。本轮用一个 64 输入、32 输出的线性层，同时检查输出、输入梯度，以及它和原始权重输出的差异。

## 实际计算的结果

| 检查项 | 实测结果 |
| --- | --- |
| 运行编号 | {cuda['run_id']} |
| 20 次前向、反向总耗时 | {probe['total_seconds']:.4f} s |
| 矩阵计算峰值分配 | {probe['peak_allocated_mib']:.2f} MiB |
| NF4 相对 L2 输出误差 | {nf4['relative_l2_error']:.4f} |
| 输出和输入梯度 | 均为有限数 |

两项运算都没有出现 NaN 或无穷值。NF4 的权重以 uint8 打包保存，输出仍按 BF16 计算，反向也能继续传到输入。

## 8.91% 的误差，该怎么读？

相对 L2 误差是“输出差值的长度 / 原输出的长度”。这里约为 {nf4['relative_l2_error'] * 100:.2f}%，说明量化后的随机线性层输出与原输出存在近似误差。它不是任务准确率，更不能写成模型准确率下降了 8.91%。

这轮的随机矩阵 loss 也不属于语言模型训练曲线。现在能确认的是底层运算可用；模型能否正确学习，还要继续检查监督位置、LoRA 更新和小样本过拟合。

复习问题的参考解释：CUDA 异步执行，计时前后同步才能测到实际完成的耗时；四位权重存储与 BF16 计算可以共存，因为压缩的是权重的存储表示，而不是所有中间计算。
""", encoding="utf-8")


if __name__ == "__main__":
    main()
