# E04：自己算一次 loss，检查标签有没有放错

手工计算的交叉熵与模型返回的 loss 完全一致。两条完整样本经过 padding 后形成 2×538 的输入，assistant-only 有 139 个监督 token；94 个 padding 位置都被排除。

## 下一 token，具体对齐到哪里？

语言模型在第 t 个位置的输出预测第 t+1 个 token。因此，计算交叉熵时把 logits 去掉最后一个位置，把 labels 去掉第一个位置，再按 -100 忽略不监督的标签。这里用同一份 logits 和标签重新计算了一遍，检查框架和手算是否对得上。

实际手算用的是下面这段。output 来自模型对同一批 inputs 的前向，没有再生成一份不同的预测。

```python
import torch.nn.functional as F

logits = output.logits[:, :-1].float()
labels = inputs["labels"][:, 1:]
manual = F.cross_entropy(
    logits.reshape(-1, logits.shape[-1]),
    labels.reshape(-1), ignore_index=-100,
)
```

| loss 口径 | 监督 token 数 | 框架返回 | 手工计算 |
| --- | --- | --- | --- |
| assistant-only | 139 | 1.762018 | 1.762018 |
| 全文，排除 padding | 980 | 2.762061 | 2.762061 |

模型、输入和初始权重相同，只改变 labels。两种口径的绝对差值都为零，模型加载与前向峰值分配约 3794 MiB。

## 两个 loss，能直接比较谁更好吗？

不能。全文 loss 还在预测用户提示和工具返回，assistant-only 则只计算模型自己的消息；参与计算的位置和分母已经不同。这里的比较用来验证监督设置，不用于宣布某种训练方式能力更好。

复习问题的答案：用户和工具内容仍保留在输入中，让模型理解上下文；只是它们的标签设为 -100，不直接贡献这轮的训练损失。padding 同样忽略，防止把补齐长度的 token 当作任务目标。
