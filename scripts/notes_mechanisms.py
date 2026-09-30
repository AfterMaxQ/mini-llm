"""根据 E03—E05 的真实结果写机制实验正文。"""
import json

from lab import ROOT


def main():
    e3 = json.loads(sorted((ROOT / "experiments/E03/runs").glob("*.json"))[-1].read_text(encoding="utf-8"))
    e4 = json.loads(sorted((ROOT / "experiments/E04/runs").glob("*.json"))[-1].read_text(encoding="utf-8"))
    e5 = json.loads(sorted((ROOT / "experiments/E05/runs").glob("*.json"))[-1].read_text(encoding="utf-8"))
    assert e3["status"] == e4["status"] == e5["status"] == "completed"
    (ROOT / "experiments/E03/notes.md").write_text(f"""# E03：模型到底在学哪一段文字？

20 个模板检查样本已经通过：增加监督标记后，渲染文字和 token 都与官方模板一致；系统、用户和工具返回没有参与 loss，assistant 的结束 token 被保留为监督目标。

## 把一条对话展开看

训练时不能只看 messages 里的角色名。工具定义会进入系统提示，工具返回在 Qwen3 模板中又以用户消息包裹；它们最后都变成一串 token。监督遮罩决定哪些位置算错、产生梯度。

官方模板没有直接给出所需的 assistant 监督区间。本轮只在 assistant 分支加了 generation 标记，没有更换工具格式或改写提示。每条样本都同时渲染原模板和加标记模板，核对全文、token 序列和非思考推理前缀。

![图 E03-1：{e3['run_id']}，真实 token 局部窗口；灰色为系统，蓝色为用户或工具返回，绿色为 assistant。](figures/assistant-mask.png)

绿色位置的 mask 为 1，参与 loss；其余位置为 0，对应标签 -100。这里监督的是完整 assistant 消息，包含角色前缀和结束标记，不只是调用 JSON。原始逐 token 对照还能查看 token id、文本片段和标签。

## 20 条检查覆盖了什么？

样本包含单次、多轮、并行和不调用工具的记录，最长为 {max(r['tokens'] for r in e3['results'])} token。另有一条从真实轨迹派生的空返回检查：只把工具返回改成空列表，用来确认遮罩仍然正确。它属于机制检查，不算真实执行或能力成绩。

E06 暴露多轮训练与当前轮推理的空 think 前缀差异后，又补查了每个 assistant 回复。按当前回复展开训练单元，逐条确认训练前缀与非思考推理前缀相同；此前的回复只作为上下文，当前回复和 EOS 才参与这一单元的 loss。
""", encoding="utf-8")
    (ROOT / "experiments/E04/notes.md").write_text(f"""# E04：自己算一次 loss，检查标签有没有放错

手工计算的交叉熵与模型返回的 loss 完全一致。两条完整样本经过 padding 后形成 {e4['input_shape'][0]}×{e4['input_shape'][1]} 的输入，assistant-only 有 {e4['assistant_tokens']} 个监督 token；{e4['padding_tokens']} 个 padding 位置都被排除。

## 下一 token，具体对齐到哪里？

语言模型在第 t 个位置的输出预测第 t+1 个 token。因此，计算交叉熵时把 logits 去掉最后一个位置，把 labels 去掉第一个位置，再按 -100 忽略不监督的标签。这里用同一份 logits 和标签重新计算了一遍，检查框架和手算是否对得上。

| loss 口径 | 监督 token 数 | 框架返回 | 手工计算 |
| --- | --- | --- | --- |
| assistant-only | {e4['assistant_tokens']} | {e4['assistant_framework_loss']:.6f} | {e4['assistant_manual_loss']:.6f} |
| 全文，排除 padding | {e4['full_tokens']} | {e4['full_framework_loss']:.6f} | {e4['full_manual_loss']:.6f} |

模型、输入和初始权重相同，只改变 labels。两种口径的绝对差值都为零，模型加载与前向峰值分配约 {e4['peak_allocated_mib']:.0f} MiB。

## 两个 loss，能直接比较谁更好吗？

不能。全文 loss 还在预测用户提示和工具返回，assistant-only 则只计算模型自己的消息；参与计算的位置和分母已经不同。这里的比较用来验证监督设置，不用于宣布某种训练方式能力更好。

复习问题的答案：用户和工具内容仍保留在输入中，让模型理解上下文；只是它们的标签设为 -100，不直接贡献这轮的训练损失。padding 同样忽略，防止把补齐长度的 token 当作任务目标。
""", encoding="utf-8")
    steps = e5["steps"]
    (ROOT / "experiments/E05/notes.md").write_text(f"""# E05：LoRA 的 A 第一轮没梯度，是不是坏了？

不是。这次检查里，第一步 A 的梯度为零，B 有梯度；B 更新后，第二步 A 也获得了非零梯度。两步更新前后，全部冻结参数的 SHA256 相同，适配器的 SHA256 改变，说明实际更新发生在 LoRA 上。

## 先看初始化，再判断梯度

LoRA 用两块较小的矩阵 A 和 B 表示增量权重，增量为 B×A×alpha/r。默认初始化时 B 为零，所以第一步更新 A 不会改变这个乘积，A 的梯度也为零。这时如果要求所有 LoRA 参数一开始都有梯度，反而会把正常初始化当成错误。

本轮固定 rank=16、alpha=32、dropout=0.05，使用两条实际工具轨迹，学习率 1e-4。优化器的 weight decay 设为零，避免把权重衰减与梯度更新混在一起。共有 {e5['trainable_parameters']:,} 个可训练参数，只开放 LoRA 参数。

| 更新步数 | A 有非零梯度的张量数 | B 有非零梯度的张量数 | loss |
| --- | --- | --- | --- |
| 1 | {steps[0]['A']['nonzero_tensors']}/{steps[0]['A']['tensors']} | {steps[0]['B']['nonzero_tensors']}/{steps[0]['B']['tensors']} | {steps[0]['loss']:.4f} |
| 2 | {steps[1]['A']['nonzero_tensors']}/{steps[1]['A']['tensors']} | {steps[1]['B']['nonzero_tensors']}/{steps[1]['B']['tensors']} | {steps[1]['loss']:.4f} |

## 参数动了，还不等于模型学会了

基础权重冻结、梯度有效和适配器更新，回答的是“训练代码有没有真的在工作”。两步 loss 的变化不能作为微调效果。下一轮会保留训练前的调用结果，再用 32 条完整轨迹检查能否拟合到明确的正确率门槛。

复习问题：只检查 requires_grad 是否足够？不够，它只说明允许求梯度。本轮还检查了梯度数值、更新前后的参数指纹，以及冻结参数是否保持不变。
""", encoding="utf-8")


if __name__ == "__main__":
    main()
