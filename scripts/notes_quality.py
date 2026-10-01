"""整理真实筛选、抽查和训练结果，数据准备不记作模型效果。"""
import csv
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from lab import ROOT, sha256, write_json
from notes_sft import draw_curve


def main():
    folder = ROOT / "experiments/E12"
    frozen_path = ROOT / "configs/data-quality-frozen.json"
    if not frozen_path.exists():
        return
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    result = json.loads((ROOT / ".local/runs" / frozen["run_id"] / "result.json").read_text(encoding="utf-8"))
    review = json.loads((folder / "sample-review.json").read_text(encoding="utf-8"))
    assert sha256(folder / "sample-review.json") == result["review_sha256"]
    assert result["status"] == "data_ready_pending_training" and result["inspection_status"] == "completed"
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    categories = Counter()
    for row in result["matched_strata"]:
        assert row["reference"] == row["filtered"]
        categories[row["category"]] += row["filtered"]
    csv_path = folder / "data-comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "baseline", "filtered"])
        writer.writerows((key, value, value) for key, value in categories.items())
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    pairs = [("单次调用", "single_call"), ("多轮调用", "multi_turn"), ("不调用", "no_call"), ("并行调用", "parallel")]
    fig, axis = plt.subplots(figsize=(6.4, 3.3), layout="constrained")
    for offset, label, color, hatch in [(-0.18, "基础清洗", "#6B7280", ""), (0.18, "额外筛选", "#059669", "//")]:
        bars = axis.bar([i + offset for i in range(4)], [categories[key] for _, key in pairs],
                        width=0.34, color=color, hatch=hatch, label=label)
        axis.bar_label(bars, labels=[f"{categories[key]:,}" for _, key in pairs], fontsize=8, padding=3)
    axis.set_xticks(range(4), [label for label, _ in pairs], fontproperties=font)
    axis.set_ylabel("独立训练轨迹数（条）", fontproperties=font)
    axis.set_title("两份 5k 数据的类别数完全匹配", fontproperties=font)
    axis.set_ylim(0, max(categories.values()) * 1.2)
    axis.legend(prop=font)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)
    path = figures / "matched-categories.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    write_json(figures / "matched-categories.source.json", {"run_id": result["run_id"],
               "source": "../data-comparison.csv", "source_sha256": sha256(csv_path), "image_sha256": sha256(path),
               "result_sha256": sha256(ROOT / ".local/runs" / result["run_id"] / "result.json")})
    train = result["train"]
    note = f'''# E12：筛得更严格，数据就一定更好吗？

目前只能确认数据条件，效果还要等训练。两份数据都保留 5,000 条独立轨迹，来源和四类任务数量一致；额外筛选保留原来的 {result['baseline_retained']:,} 条，替换 {result['baseline_replaced']:,} 条。抽查同时发现了误删和漏检，因此这里的“通过筛选”只表示符合固定规则。

## 第一版为什么不能直接拿去训练？

第一轮从既有训练区的 {result['candidate_train_groups']:,} 个任务组中选出了 5,000 条。逐条检查 25 条通过和 25 条剔除样本时，发现数字提取把句末的 `$200.`、`27.` 漏掉了。正常的折扣计算、加法和随机数调用被误删，原来的占位检测也没认出 `QR_CODE_IMAGE_DATA`。

这轮数据保留为失败记录，没有用于模型训练。第二轮另开运行，修正句号识别，增加这个明确的占位字符串，然后重新扫描全部候选训练组。对同样的输入，旧规则漏掉 200 和 27，新规则能正确读出。紧连单位的 `5000mAh` 仍不支持，记录为当前规则的局限。

## 第二轮具体筛了什么？

调用中的字符串和数值，要能在此前用户消息或工具返回里找到表面出处。字符串忽略大小写和标点，按连续词项匹配；数字按数值比较。数组和嵌套对象逐项检查。此前 assistant 自己说过的话不能充当参数依据，否则模型先猜一个值，再调用它，也会被误当作有来源。

```python
# 只用当前调用之前的用户和工具消息检查出处
if message["role"] in ["user", "tool"]:
    evidence.append(message["content"])
# 不把 assistant 自己的说法加入 evidence
```

另外检查工具返回中的固定占位文本。布尔值、null 和对象键名不做来源判断；空字符串按这轮固定策略剔除，不代表它违反 schema。不调用的任务仍沿用已有检查，额外规则没有重新判断是否该调用。公开工具没有在本机执行，返回是否真实也没有由这一步保证。

第二轮通过 {result['accepted']:,}/{result['candidate_train_groups']:,} 条，剔除 {result['rejected']:,} 条。按轨迹计，字符串出处未确认 {result['reason_counts']['unverified_string']:,} 条、数值出处未确认 {result['reason_counts']['unverified_number']:,} 条、占位返回 {result['reason_counts']['placeholder_tool_return']:,} 条。同一条可能命中多个理由，三项不能相加当作剔除总数。完整扫描和文件整理耗时 {result['screening_seconds']:.2f} 秒，不包括后续阅读抽查和 token 展开。

## 换了样本，还能公平比较吗？

候选只来自 E02 原有训练区。重新取出的前 10,000 条与冻结训练文件逐条一致，每组仍只选同一代表样本；没有为凑数重新拆组。随后沿用原组序，在每个“来源 × 类别”配额中取通过样本。dev 的 500 条原样复用，test 没有进入筛选和抽查。

| 来源 | 类别 | 基础清洗 | 额外筛选 |
| --- | --- | --- | --- |
'''
    labels = {"single_call": "单次调用", "multi_turn": "多轮调用", "no_call": "不调用", "parallel": "并行调用"}
    for row in result["matched_strata"]:
        note += f"| {row['source']} | {labels[row['category']]} | {row['reference']:,} | {row['filtered']:,} |\n"
    note += f'''
![图 E12-1：两份训练数据各 5,000 条，四类数量匹配；表中进一步核对来源。](figures/matched-categories.png)

| 实际训练输入 | 基础清洗 | 额外筛选 |
| --- | --- | --- |
| 当前回复单元 | 14,502 | {train['assistant_units']:,} |
| 监督 token | 638,598 | {train['supervised_tokens']:,} |
| 最大回复单元输入长度（token） | 2,034 | {train['max_sequence_length']:,} |
| 一个 epoch 的更新次数，梯度累积 8 | 1,813 | {(train['assistant_units']+7)//8:,} |

新数据已在 CPU 上用同一模板和监督遮罩展开，全部单元都有监督 token，官方文本与标记模板的文本及 token 一致，没有截断调用。两份数据的轨迹数相同，监督量和具体任务内容仍有差异；后续表现变化同时包含筛选策略和样本替换的影响。

## 抽查有没有证明这批数据更可靠？

没有证明整体更可靠。第二轮仍随机读了 25 条通过和 25 条剔除的候选训练记录。通过样本有 21 条未见明显问题，另 4 条有回复或标注疑点。例如发票调用本身参数正确，最终回答却说会通过邮件送达，工具定义和返回都没提供邮件动作；还有食谱提到“剩余配料”，返回却没有列出这些配料。规则只看参数出处，抓不到这些问题。

剔除样本里，14 条有合理的转换或检索写法，例如 Apple 转 AAPL、Euros 转 EUR、bike 转 biking、按类别检索时使用空字符串。它们不满足连续文字匹配，却不能直接判作错误。另有 7 条参数选择缺少可核对依据，1 条明确是二维码占位返回，3 条混合了合理转换和其他疑点。这些是当前主 Agent 对 50 条标注的阅读判断，没有把未核验的选择都算成错误标签。

一个真正应该保留的例子，是用户先搜到一本书，下一轮要求找同一作者的其他书。作者来自上一轮工具返回，而不是这一轮用户原文。另一种需要警惕的例子，是用户说“今天”，标注却直接填入某个 2022 年日期，前文没有基准时间。固定规则能挡住后者，也可能误删前者的合理改写。

50 条采用通过与剔除各半的抽样，没有按总体比例抽取，也没有逐条真实执行，因此不计算整批数据的“准确率”。这次抽查更有用的结论是：筛选会改变任务难度和表达方式，后续若分数上升，还要检查是否只学会了更直接的参数复制。

## 训练与完整 dev 结果

正式对照固定 rank 16、学习率 1e-4、seed 17、一个 epoch，精度、模板、缓存策略和固定提示与 E09 5k 相同。基础清洗条件只在训练及全部 500 条 dev 工具生成结束后复用。每 100 步计算完整 dev loss；最终工具评测仍包含 500 条轨迹、928 个决策轮。

'''
    runs = []
    for path in sorted((ROOT / ".local/runs").glob("E12-R*/config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if "train_size" in config:
            runs.append(path.parent)
    if not runs:
        note += "新数据尚未开始 GPU 训练，当前不填写通过率或提升值。现有 GPU 队列继续按原顺序运行。\n"
    for run in runs:
        progress = json.loads((run / "progress.json").read_text(encoding="utf-8")) if (run / "progress.json").exists() else {}
        ended = json.loads((run / "result.json").read_text(encoding="utf-8")) if (run / "result.json").exists() else {}
        state = {"trained_pending_tool_eval": "训练完成，完整工具评测另查", "failed": "失败，原记录保留"}.get(ended.get("status"), "运行中")
        note += f"\n{run.name}：{state}，已更新 {ended.get('steps', progress.get('latest', {}).get('step', 0)):,} 次。\n"
        curve = draw_curve(run.name)
        if curve and curve["train"]:
            note += f"\n![图 E12：{run.name} 的真实训练与完整 dev loss，未平滑。](figures/{curve['path'].name})\n"
    for path in sorted((folder / "runs").glob("*.json")):
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        config = evaluation["config"]
        if evaluation["status"] != "completed" or config.get("kind") != "tool_eval":
            continue
        summary = evaluation["summaries"][evaluation["selected_prompt"]]
        assert summary["trajectories"] == summary["evaluated_trajectories"] == 500 and summary["decision_turns"] == 928
        assert summary["rows_sha256"] == sha256(ROOT / ".local/runs" / evaluation["run_id"] / (evaluation["selected_prompt"] + ".jsonl"))
        assert config["frozen_prompt_sha256"] == sha256(ROOT / "configs/prompt-frozen.json")
        assert config["adapter_sha256"] == sha256(ROOT / ".local/runs" / config["source_train_run"] / "selected-adapter/adapter_model.safetensors")
        note += f"\n{evaluation['run_id']} 完整 dev：整条轨迹 {summary['trajectory_passed']}/500，调用轮 {summary['call_turn_passed']}/568，不调用轮 {summary['no_call_turn_passed']}/360，截断 {summary['truncated']}。\n"
    (folder / "notes.md").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
