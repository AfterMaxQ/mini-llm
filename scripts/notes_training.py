"""按真实 E06 日志替换当前章节，不叠加临时汇总。"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from lab import ROOT, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = ROOT / ".local/runs" / args.run
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    baseline = json.loads((run / "baseline.json").read_text(encoding="utf-8")) if (run / "baseline.json").exists() else None
    progress = json.loads((run / "progress.json").read_text(encoding="utf-8")) if (run / "progress.json").exists() else None
    result = json.loads((run / "result.json").read_text(encoding="utf-8")) if (run / "result.json").exists() else None
    units = json.loads((run / "training-units.json").read_text(encoding="utf-8")) if (run / "training-units.json").exists() else {}
    rows = list(csv.DictReader((run / "metrics.csv").open(encoding="utf-8"))) if (run / "metrics.csv").exists() else []
    directory = ROOT / "experiments/E06"; directory.mkdir(parents=True, exist_ok=True)
    status = result["status"] if result else "running"
    if result and result.get("final", {}).get("passed", 0) >= 29:
        opening = f"32 条训练轨迹中，最后有 {result['final']['passed']} 条的全部工具调用通过严格检查，达到了这轮的 29/32 门槛。它说明训练链路能拟合这些样本，还不能说明模型学会了处理新任务。"
    else:
        opening = "这轮还在进行，暂时没有过拟合通过的结论。先记录训练前的表现，再检查 loss 下降以后，模型能否真正复现这些样本的工具名和参数。"
    note = f"""# E06：32 条样本，能不能真正学进去？

{opening}

## 为什么只用 32 条？

这里的小规模是排错手段。连训练样本都记不住时，直接扩大数据量很难判断是数据难、监督位置错，还是参数没有更新。32 条达到门槛后，还要继续 1k、5k、10k 的独立验证与测试比较。

样本包含 16 条单次调用、8 条多轮和 8 条并行调用。每条保留完整消息和工具定义。每轮检查工具调用时，输入使用该位置之前的标注历史；它没有执行真实工具，也不属于 Agent 成功率。

## 这一轮怎么训练

QLoRA 使用 NF4 权重和 BF16 计算，rank=16、alpha=32、dropout=0.05。学习率为 1e-4，每条样本单独前向，累计 8 次梯度后更新一次。每 25 次更新检查全部 32 条轨迹，最多更新 300 次。调用的工具名、参数与标注严格一致才算通过，截断或无法解析都计为失败。

本轮运行编号为 {args.run}。
"""
    if units.get("policy") == "assistant_action":
        note += f"\n32 条独立轨迹展开成 {units['assistant_units']} 个当前回复单元。每次前向使用一个单元，历史消息保留为输入，只有当前 assistant 回复参与 loss；轨迹数量没有增加到 {units['assistant_units']} 条。\n"
    if baseline:
        note += f"\n训练前通过 {baseline['passed']}/32，共检查 {baseline['tool_turns']} 轮工具调用。这个起点很重要：后续不能把原模型已经会做的事情全算成微调的贡献。\n"
    issue_path = ROOT / "experiments/E06/R01-template-mismatch.json"
    if issue_path.exists() and args.run != "E06-R01":
        issue = json.loads(issue_path.read_text(encoding="utf-8"))
        previous = issue["last_metrics"]
        note += f"""
## 第一轮 loss 很低，为什么还有回复不调用工具？

R01 的原模型起点是 9/32。第 25 次更新检查通过 25/32，第 50 次为 28/32；到第 {previous['step']} 次，训练 loss 已降到 {previous['train_loss']:.6f}，仍不能据此宣布通过。查看失败回复时，有些只说“我来帮你查询”，没有输出调用。

对比一个阶乘样本才看到差异：完整多轮训练的早期 assistant 直接接回复文字，当前轮推理前缀却多了空的 think 段。遮罩和梯度虽然正确，输入仍没有完全对齐。这也解释了为什么单看 loss 会误判。

R01 在确认差异后停止，训练日志、两次完整调用检查和 checkpoint 均保留。R02 仍使用同一批 32 条轨迹和官方模板，只把每条 assistant 回复展开成一个训练单元，前面的对话作为上下文，当前回复作为目标。实际前缀已经重新检查，手工 loss 和 LoRA 更新也重跑过了。

训练单元的分母改变了，所以两轮 loss 不直接比较大小。这里比较的是同 32 条完整轨迹能否复现全部标注调用。
"""
    if rows:
        shutil_source = directory / "metrics.csv"
        shutil_source.write_bytes((run / "metrics.csv").read_bytes())
        figures = directory / "figures"; figures.mkdir(exist_ok=True)
        font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
        fig, axis = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
        axis.plot([int(r["step"]) for r in rows], [float(r["train_loss"]) for r in rows], color="#059669", linewidth=1.2)
        axis.set_xlabel("optimizer step（每 8 次梯度累计更新一次）", fontproperties=font)
        axis.set_ylabel("assistant-only 交叉熵", fontproperties=font)
        axis.set_title("32 条训练样本的原始 loss，未平滑", fontproperties=font)
        axis.yaxis.grid(True, color="#E5E7EB"); axis.set_axisbelow(True)
        for side in ["top", "right"]: axis.spines[side].set_visible(False)
        path = figures / "loss.png"; fig.savefig(path, dpi=300); plt.close(fig)
        write_json(figures / "loss.source.json", {"run_id": args.run, "points": len(rows), "smoothing": None,
                    "source_sha256": sha256(shutil_source), "image_sha256": sha256(path)})
        note += f"\n## Loss 降了，调用也正确了吗？\n\n![图 E06-1：{args.run}，32 条样本；每个点是一次 optimizer step 的真实平均 loss。](figures/loss.png)\n\n"
        note += f"目前记录了 {len(rows)} 次更新，第一步 loss 为 {float(rows[0]['train_loss']):.4f}，最新为 {float(rows[-1]['train_loss']):.4f}。这是训练集曲线，不能当成验证集的泛化结果。\n\n"
        checks = [r for r in rows if r["train_passed"] != ""]
        if checks:
            note += "| 更新步数 | 完整轨迹通过数 |\n| --- | --- |\n"
            note += "\n".join(f"| {r['step']} | {r['train_passed']}/32 |" for r in checks) + "\n"
        else:
            note += "尚未到第一次完整调用检查，不能仅凭 loss 下降宣布通过。\n"
    note += "\n每次检查后同时保存适配器、优化器、步数与随机状态。下一轮会实际加载并续训，检查保存的文件是否足以恢复同一次实验。\n"
    (directory / "notes.md").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
