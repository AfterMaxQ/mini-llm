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
    evaluations = []
    for path in sorted((folder / "runs").glob("*.json")):
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        if evaluation.get("status") == "completed" and evaluation.get("config", {}).get("kind") == "tool_eval":
            evaluations.append(evaluation)
    evaluated_train_runs = {item["config"].get("source_train_run") for item in evaluations}
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

这组实验先把筛选规则说清楚，再看两组各 1,000 条训练数据在同一份 dev 上的表现。筛选前后各有 5,000 条独立轨迹，来源和四类任务数量配平；额外筛选保留原来的 {result['baseline_retained']:,} 条，替换 {result['baseline_replaced']:,} 条。抽查同时发现了误删和漏检，因此这里的“通过筛选”只表示符合固定规则。

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

## 1k 训练对照怎么做？

训练对照各用 1,000 条独立轨迹。基础组沿用原 1k 样本，额外筛选组从已复查的 5k 数据池中抽取，逐个匹配“来源 × 类别”的数量；原 5k 筛选记录仍用于解释规则和抽查过程。两组固定 rank 16、学习率 1e-4、seed 17、一个 epoch，模板、精度与缓存策略相同。

两组使用同一份预先冻结的 100 条 dev，四类任务都覆盖，包含原 dev 全部 10 条并行调用。初始、每 200 步和末步计算 token 加权 loss，选择最低 loss 的 checkpoint，再用同样的固定提示生成工具决策。基础组使用 E11 的同条件 1k 运行，不能拿验证方法不同的旧运行直接作参考。100 条成绩只代表这份固定样本，不当作完整 dev 的分数。

'''
    runs = []
    for path in sorted((ROOT / ".local/runs").glob("E12-R*/config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if "train_size" in config:
            runs.append(path.parent)
    if not runs:
        note += "新数据尚未开始 GPU 训练，当前不填写通过率或提升值。先完成基础组，再训练额外筛选组。\n"
    for run in runs:
        progress = json.loads((run / "progress.json").read_text(encoding="utf-8")) if (run / "progress.json").exists() else {}
        ended = json.loads((run / "result.json").read_text(encoding="utf-8")) if (run / "result.json").exists() else {}
        if ended.get("status") == "trained_pending_tool_eval":
            state = "训练完成，工具评测已记录" if run.name in evaluated_train_runs else "训练完成，工具评测待核对"
        else:
            state = {"failed": "失败，原记录保留"}.get(ended.get("status"), "运行中")
        note += f"\n{run.name}：{state}，已更新 {ended.get('steps', progress.get('latest', {}).get('step', 0)):,} 次。\n"
        curve = draw_curve(run.name)
        if curve and curve["train"]:
            note += f"\n![图 E12：{run.name} 的真实训练与固定 dev loss，未平滑。](figures/{curve['path'].name})\n"
    for evaluation in evaluations:
        config = evaluation["config"]
        summary = evaluation["summaries"][evaluation["selected_prompt"]]
        records = [json.loads(line) for line in (ROOT / ".local/data/processed" / config["data_run"] / (config["split"] + ".jsonl")).read_text(encoding="utf-8").splitlines()]
        turns = sum(m["role"] == "assistant" and bool(m.get("tool_calls") or record["messages"][i - 1]["role"] == "user")
                    for record in records for i, m in enumerate(record["messages"]))
        assert summary["trajectories"] == summary["evaluated_trajectories"] == len(records) and summary["decision_turns"] == turns
        assert summary["rows_sha256"] == sha256(ROOT / ".local/runs" / evaluation["run_id"] / (evaluation["selected_prompt"] + ".jsonl"))
        assert config["frozen_prompt_sha256"] == sha256(ROOT / "configs/prompt-frozen.json")
        assert config["adapter_sha256"] == sha256(ROOT / ".local/runs" / config["source_train_run"] / "selected-adapter/adapter_model.safetensors")
        note += f"\n{evaluation['run_id']} 固定 dev：整条轨迹 {summary['trajectory_passed']}/{summary['trajectories']}，调用轮 {summary['call_turn_passed']}/{summary['call_turns']}，不调用轮 {summary['no_call_turn_passed']}/{summary['no_call_turns']}，截断 {summary['truncated']}。\n"
    budget = json.loads((ROOT / "configs/quality.json").read_text(encoding="utf-8"))
    reference = budget["reference"]
    reference_config = json.loads((ROOT / reference["training_config"]).read_text(encoding="utf-8"))
    quality_config = json.loads((ROOT / budget["training_config"]).read_text(encoding="utf-8"))
    shared_config = ("seed", "rank", "learning_rate", "epochs", "micro_batch",
                     "gradient_accumulation", "max_sequence_length", "precision", "model_revision")
    assert all(reference_config[key] == quality_config[key] for key in shared_config)
    reference_train = None
    for path in sorted((ROOT / "experiments" / reference["experiment"] / "runs").glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        config = run.get("config", {})
        if (config.get("train_size") == reference["size"] and config.get("seed") == reference["seed"]
                and config.get("data_run") == reference_config["data_run"]
                and config.get("rank") == reference_config["rank"]
                and config.get("learning_rate") == reference_config["learning_rate"]):
            reference_train = run
            break
    quality_train = None
    job = budget["jobs"][0]
    for path in sorted((folder / "runs").glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        config = run.get("config", {})
        if config.get("train_size") == job["size"] and config.get("seed") == job["seed"]:
            quality_train = run
            break
    if reference_train and quality_train:
        reference_eval = None
        for path in sorted((ROOT / "experiments" / reference["experiment"] / "runs").glob("*.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            if (item.get("status") == "completed" and item.get("config", {}).get("kind") == "tool_eval"
                    and item["config"].get("source_train_run") == reference_train["run_id"]):
                reference_eval = item
                break
        quality_eval = next((item for item in evaluations if item["config"].get("source_train_run") == quality_train["run_id"]), None)
        if reference_eval and quality_eval:
            assert all(reference_train["config"][key] == quality_train["config"][key] for key in shared_config)
            reference_dev = reference_train.get("train", {}).get("data_file_hashes", {}).get("dev.jsonl")
            quality_dev = quality_train.get("train", {}).get("data_file_hashes", {}).get("dev.jsonl")
            reference_summary = reference_eval["summaries"][reference_eval["selected_prompt"]]
            quality_summary = quality_eval["summaries"][quality_eval["selected_prompt"]]
            assert reference_dev == quality_dev
            assert reference_eval["config"]["frozen_prompt_sha256"] == quality_eval["config"]["frozen_prompt_sha256"]
            assert reference_eval["config"]["data_run"] == quality_eval["config"]["data_run"]
            assert reference_eval["selected_prompt"] == quality_eval["selected_prompt"] == "few_shot"
            assert reference_summary["trajectories"] == quality_summary["trajectories"] == 100
            comparison_rows = []
            for category, label in [("single_call", "单次调用"), ("multi_turn", "多轮调用"),
                                    ("no_call", "不调用"), ("parallel", "并行调用")]:
                base = reference_summary["categories"][category]
                filtered = quality_summary["categories"][category]
                assert base["samples"] == filtered["samples"]
                comparison_rows.append((label, base["passed"], filtered["passed"], base["samples"]))
            comparison_csv = folder / "quality-evaluation.csv"
            with comparison_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["category", "reference_passed", "filtered_passed", "denominator"])
                writer.writerows(comparison_rows)
            fig, axis = plt.subplots(figsize=(6.4, 3.3), layout="constrained")
            positions = list(range(len(comparison_rows)))
            for offset, column, label, color, hatch in [
                    (-0.18, 1, "基础清洗", "#6B7280", ""),
                    (0.18, 2, "额外筛选", "#059669", "//")]:
                values = [row[column] / row[3] * 100 for row in comparison_rows]
                bars = axis.bar([i + offset for i in positions], values, width=0.34,
                                color=color, hatch=hatch, label=label)
                axis.bar_label(bars, labels=[f"{row[column]}/{row[3]}" for row in comparison_rows],
                               fontsize=8, padding=3)
            axis.set_xticks(positions, [row[0] for row in comparison_rows], fontproperties=font)
            axis.set_ylabel("固定 dev 轨迹通过率（%）", fontproperties=font)
            axis.set_ylim(0, 120)
            axis.legend(prop=font)
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
            axis.set_axisbelow(True)
            for side in ["top", "right"]:
                axis.spines[side].set_visible(False)
            comparison_figure = figures / "quality-evaluation.png"
            fig.savefig(comparison_figure, dpi=300)
            plt.close(fig)
            write_json(figures / "quality-evaluation.source.json", {
                "reference_run": reference_eval["run_id"],
                "reference_result_sha256": sha256(ROOT / "experiments" / reference["experiment"] / "runs" / f"{reference_eval['run_id']}.json"),
                "quality_run": quality_eval["run_id"],
                "quality_result_sha256": sha256(folder / "runs" / f"{quality_eval['run_id']}.json"),
                "source": "../quality-evaluation.csv",
                "source_sha256": sha256(comparison_csv),
                "image_sha256": sha256(comparison_figure),
            })
            category_table = "\n".join(
                f"| {label} | {base}/{denominator} | {filtered}/{denominator} |"
                for label, base, filtered, denominator in comparison_rows)
            note += f'''\n## 额外筛选后，固定 dev 上的表现怎样？

基础组与额外筛选组都使用 rank 16、学习率 1e-4、seed 17 和一个 epoch；验证数据与提示哈希相同。基础组来自 E11，筛选组从匹配后的数据池抽取。固定 100 条 dev 上，整条轨迹通过数为 {reference_summary['trajectory_passed']}/100 与 {quality_summary['trajectory_passed']}/100，差值 {quality_summary['trajectory_passed'] - reference_summary['trajectory_passed']:+d} 条。下面按任务类型拆开看：

![图 E12-2：同一份固定 dev 上，两组的分类型轨迹通过率。](figures/quality-evaluation.png)

| 任务类型 | 基础组 | 额外筛选组 |
| --- | ---: | ---: |
{category_table}

调用轮分别通过 {reference_summary['call_turn_passed']}/{reference_summary['call_turns']} 与 {quality_summary['call_turn_passed']}/{quality_summary['call_turns']}；不调用轮分别通过 {reference_summary['no_call_turn_passed']}/{reference_summary['no_call_turns']} 与 {quality_summary['no_call_turn_passed']}/{quality_summary['no_call_turns']}。错误记录为 {json.dumps(reference_summary['errors'], ensure_ascii=False)} 与 {json.dumps(quality_summary['errors'], ensure_ascii=False)}，截断数为 {reference_summary['truncated']} 与 {quality_summary['truncated']}。

表格里的“不调用”按整条任务轨迹计数；这里的“不调用轮”则是轨迹中的单次决策，统计口径和分母不同。

在这 100 条固定样本上，额外筛选组多通过 {quality_summary['trajectory_passed'] - reference_summary['trajectory_passed']} 条轨迹；不调用决策轮多通过 {quality_summary['no_call_turn_passed'] - reference_summary['no_call_turn_passed']} 条，其余三类轨迹也有增加。这个结果值得继续看，但还不是筛选规则的因果证明：这里只有一个训练 seed，且筛选组替换了部分样本，规则变化与样本内容变化没有拆开。
'''
    (folder / "notes.md").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
