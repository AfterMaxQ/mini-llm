"""按 E14 冻结数据、训练日志和评测结果更新领域实验记录。"""
import argparse
import json
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from lab import ROOT, sha256, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def update_notes(run_id, pi_run_id=None):
    run = ROOT / ".local/runs" / run_id
    config = read(run / "config.json")
    assert config.get("domain_manifest_sha256") == sha256(ROOT / "configs/domain-mix-frozen.json")
    prep = next((p for p in sorted((ROOT / ".local/runs").glob("E14-R*/result.json"))
                 if read(p).get("status") == "domain_mix_frozen"), None)
    assert prep is not None
    prep_result = read(prep)
    progress_path, result_path = run / "progress.json", run / "result.json"
    progress = read(progress_path) if progress_path.exists() else {}
    result = read(result_path) if result_path.exists() else {}
    metrics_path = run / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()] if metrics_path.exists() else []

    lines = ["## 领域混合微调与迁移评测", "", "### 训练样本如何组成？", "",
             "训练集由512条公开工具轨迹和512条Pi规则参考轨迹组成。公开样本按来源与调用类别分层抽取；Pi样本从16个训练模板族各取32条。样本清单在查看模型成绩前冻结，固定100条公开dev继续用于loss选择和工具决策评测。", "",
             f"冻结记录确认了{prep_result['public_trajectories']}条公开轨迹、{prep_result['pi_trajectories']}条Pi轨迹，Pi覆盖{prep_result['pi_families']}个训练模板族；训练条数按独立轨迹计算。Pi dev和test采用单独模板族，本轮训练不会读取test。", "",
             "| 来源 | 类别 | 轨迹数 |", "| --- | --- | ---: |"]
    mixed = [json.loads(line) for line in (ROOT / ".local/data/processed/E14-domain/train-1024.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    strata = Counter((row["source"], row["category"]) for row in mixed)
    for (source, category), count in sorted(strata.items()):
        lines.append(f"| {source} | {category} | {count} |")
    lines += ["", "Pi每类都是64条：该类两个独立训练模板族各抽32条；公开部分按原1k样本的来源和类别占比分层。", "",
              "### 训练进展与验证曲线", ""]
    if result.get("status") == "trained_pending_tool_eval":
        lines.append(f"训练已结束：共更新{result['steps']}步，最低冻结dev loss为{result['selected_dev_loss']:.6f}。这项loss用于挑选checkpoint，不等同于Agent任务成功率。")
    elif result.get("status") == "failed":
        lines.append("训练未成功，失败配置和已产生的checkpoint保持原样；本次不记作领域能力结果。")
    elif progress:
        latest = progress.get("latest", {})
        lines.append(f"训练仍在进行：最近记录到第{latest.get('step', 0)}/{progress.get('max_steps', 0)}步；当前train loss为{latest.get('loss', float('nan')):.6f}，目前最低dev loss为{progress.get('best_metric', float('nan')):.6f}。尚未结束的曲线只描述运行进度。")
    else:
        lines.append("混合数据与训练配置已冻结，尚无训练进度记录。")
    failed = []
    for path in sorted((ROOT / ".local/runs").glob("E14-R*/result.json")):
        if read(path).get("status") == "failed":
            config_path = path.parent / "config.json"
            if config_path.exists() and read(config_path).get("operation") == "domain_mix_source_hash_preflight":
                failed.append(read(path))
    if failed:
        lines += ["", "首次冻结检查因来源文件哈希配置不匹配而停止；核对本机文件后修正配置，再按相同抽样规则完成冻结。错误发生在训练集写入前，失败记录保留。"]
    lines.append("")
    if metrics:
        figure = ROOT / "experiments/E14/figures/domain-training-loss.png"
        figure.parent.mkdir(parents=True, exist_ok=True)
        font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
        train_rows = [row for row in metrics if "loss" in row]
        dev_rows = [row for row in metrics if "eval_loss" in row]
        fig, axis = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
        if train_rows:
            axis.plot([row["step"] for row in train_rows], [row["loss"] for row in train_rows],
                      color="#2563EB", linewidth=1.2, label="训练 loss")
        if dev_rows:
            axis.plot([row["step"] for row in dev_rows], [row["eval_loss"] for row in dev_rows],
                      marker="o", color="#059669", linewidth=1.2, label="dev loss")
        axis.set_xlabel("optimizer step", fontproperties=font)
        axis.set_ylabel("token 加权交叉熵", fontproperties=font)
        axis.set_title("E14 领域混合训练的实际记录", fontproperties=font)
        axis.grid(color="#E5E7EB", linewidth=0.6)
        axis.legend(prop=font)
        for side in ["top", "right"]:
            axis.spines[side].set_visible(False)
        fig.savefig(figure, dpi=300)
        plt.close(fig)
        write_json(figure.with_suffix(".source.json"), {"metrics_sha256": sha256(metrics_path),
                   "image_sha256": sha256(figure), "training_run": run_id,
                   "train_points": len(train_rows), "dev_points": len(dev_rows)})
        lines.append("![E14领域混合训练的真实train/dev loss曲线。](figures/domain-training-loss.png)")
    else:
        lines.append("训练曲线尚未产生；开始训练后会按实际日志绘制。")
    lines += ["", "### 固定公开集和 Pi 任务的实际成绩", ""]
    evaluations = []
    for path in sorted((ROOT / "experiments/E14/runs").glob("*.json")):
        item = read(path)
        if item.get("config", {}).get("kind") == "tool_eval" and item["config"].get("source_train_run") == run_id:
            evaluations.append(item)
    if evaluations:
        item = evaluations[-1]
        summary = item["summaries"][item["selected_prompt"]]
        lines.append(f"固定公开dev通过{summary['trajectory_passed']}/{summary['trajectories']}条；逐轮结果保留了{summary['evaluated_trajectories']}条完整分母。")
    else:
        lines.append("模型训练完成后，固定100条公开dev会评估工具决策；Pi dev16与test40还需通过本机Agent容器分别运行。")
    pi_results = []
    for path in sorted((ROOT / "experiments/E14/runs").glob("*.json")):
        item = read(path)
        if item.get("operation", "").startswith("pi_model_agent_"):
            pi_results.append(item)
    for item in pi_results:
        lines.append(f"Pi {item['split']}：{item['passed_tasks']}/{item['target_tasks']}个任务通过；超时、截断或工具错误都留在预定分母中。")
    if pi_run_id:
        live = ROOT / ".local/runs" / pi_run_id
        pi_config_path, pi_progress_path = live / "config.json", live / "progress.json"
        if pi_config_path.exists() and pi_progress_path.exists() and not (live / "result.json").exists():
            pi_config, pi_progress = read(pi_config_path), read(pi_progress_path)
            lines.append(f"Pi {pi_config['split']}当前完成{pi_progress['completed']}/{pi_progress['target']}个任务，通过{pi_progress['passed']}个；未完成条目仍计入固定分母。")
    lines += ["", "公开dev的轨迹通过数、Pi任务完成数和token loss回答的是不同问题，不能互相替代。规则参考本身不是Agent成绩；具体错误要回到工具调用、返回和最终文件状态判断。", ""]

    path = ROOT / "experiments/E14/notes.md"
    existing = path.read_text(encoding="utf-8").rstrip()
    title = "## 领域混合微调与迁移评测"
    if title in existing:
        existing = existing.split(title, 1)[0].rstrip()
    path.write_text(existing + "\n\n" + "\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--pi-run")
    args = parser.parse_args()
    update_notes(args.run, args.pi_run)


if __name__ == "__main__":
    main()
