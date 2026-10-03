"""从逐任务结果更新 Pi 模型接入记录。"""
import argparse
import json
import subprocess
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from lab import ROOT, sha256, write_json

LABELS = {
    "missing_information": "缺信息先询问",
    "failure_recovery": "失败后恢复",
    "read_locate": "读取定位",
    "invalid_path_recovery": "无效路径恢复",
    "multi_file": "多文件修改",
    "no_tool": "无需调用工具",
    "function_fix": "函数修复",
    "config_change": "配置修改",
}


def load_run(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    folder = path.parent
    progress_path, result_path = folder / "progress.json", folder / "result.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    public_path = ROOT / "experiments/E13/runs" / f"{config['run_id']}.json"
    public = json.loads(public_path.read_text(encoding="utf-8")) if public_path.exists() else {}
    rows = progress.get("rows", [])
    assert config["operation"] == "pi_model_agent_dev"
    assert len(rows) == progress.get("completed", len(rows)) <= config["task_count"]
    assert [row["task_id"] for row in rows] == config["task_ids"][:len(rows)]
    if result:
        assert result["records_sha256"] == sha256(folder / "records.jsonl")
        assert result["evaluated_tasks"] == len(rows)
    return {"id": config["run_id"], "config": config, "progress": progress,
            "result": result, "rows": rows, "folder": folder,
            "validity": public.get("validity", "eligible")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    assert args.run.startswith("E13-R") and "/" not in args.run and "\\" not in args.run
    base_note = (ROOT / "experiments/E13/notes.md").read_text(encoding="utf-8")
    marker = "## 微调模型接入 Pi 后，能不能独立完成开发任务？"
    if marker in base_note:
        base_note = base_note.split(marker, 1)[0].rstrip()
    else:
        base_note = subprocess.check_output(
            ["git", "show", "HEAD:experiments/E13/notes.md"], cwd=ROOT, text=True, encoding="utf-8")
        base_note = base_note.split(marker, 1)[0].rstrip()
    all_runs = [load_run(path) for path in sorted((ROOT / ".local/runs").glob("E13-R*/config.json"))]
    assert all_runs and all_runs[-1]["id"] == args.run
    runs = [run for run in all_runs if run["validity"] != "invalid_harness"]
    assert runs and runs[-1]["id"] == args.run
    base_config = all_runs[0]["config"]
    for run in all_runs[1:]:
        config = run["config"]
        for key in ("source_run", "source_eval_run", "adapter_sha256", "task_ids", "temperature",
                    "inference_seed", "context_window", "max_new_tokens", "task_timeout_seconds", "max_tool_calls"):
            assert config[key] == base_config[key], f"E13对照的共同条件变化：{key}"

    selected_eval = json.loads((ROOT / "experiments/E09/runs/E09-R16.json").read_text(encoding="utf-8"))
    assert selected_eval["status"] == "completed" and selected_eval["selected_prompt"] == "few_shot"
    selected_score = selected_eval["summaries"]["few_shot"]["trajectory_passed"]
    categories = list(LABELS)
    counts = {}
    for run in runs:
        rows = run["rows"]
        counts[run["id"]] = {
            "done": Counter(row["category"] for row in rows),
            "passed": Counter(row["category"] for row in rows if row.get("passed")),
            "failed": Counter(row["category"] for row in rows if not row.get("passed")),
            "tool_calls": [event for row in rows for event in row.get("observed_tool_calls", [])],
            "model_requests": [request for row in rows for request in row.get("model_requests", [])],
            "failures": [row for row in rows if not row.get("passed")],
            "truncations": sum(bool(request.get("truncated")) for row in rows
                                for request in row.get("model_requests", [])),
        }

    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig, axis = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
    y = list(range(len(categories)))
    for index, run in enumerate(runs):
        run_counts = counts[run["id"]]
        offset = 0 if len(runs) == 1 else (0.16 if index % 2 == 0 else -0.16)
        positions = [value + offset for value in y]
        done = [run_counts["done"][category] for category in categories]
        passed = [run_counts["passed"][category] for category in categories]
        failed = [run_counts["failed"][category] for category in categories]
        remaining = [2 - value for value in done]
        axis.barh(positions, passed, height=0.28, color="#16856B", label="通过" if index == 0 else None)
        axis.barh(positions, failed, height=0.28, left=passed, color="#C66B5D",
                  label="已运行未通过" if index == 0 else None)
        axis.barh(positions, remaining, height=0.28, left=done, color="#D1D5DB",
                  label="尚未运行" if index == 0 else None)
    axis.set_yticks(y, [LABELS[category] for category in categories], fontproperties=font)
    axis.invert_yaxis()
    axis.set_xlim(0, 2.45)
    axis.set_xticks([0, 1, 2])
    axis.set_xlabel("冻结任务数（每类 2 条）", fontproperties=font)
    axis.set_title("E13 Pi dev：冻结任务逐类结果", fontproperties=font)
    axis.legend(prop=font, ncols=3, loc="lower right")
    axis.xaxis.grid(True, color="#E5E7EB", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)
    figures = ROOT / "experiments/E13/figures"
    figures.mkdir(parents=True, exist_ok=True)
    image = figures / "pi-agent-dev.png"
    fig.savefig(image, dpi=300)
    plt.close(fig)
    inputs = []
    for run in all_runs:
        for name in ("progress.json", "result.json", "records.jsonl"):
            path = run["folder"] / name
            if path.exists():
                inputs.append({"run_id": run["id"], "file": name, "sha256": sha256(path)})
    write_json(image.with_suffix(".source.json"), {
        "runs": inputs, "excluded_runs": [run["id"] for run in all_runs if run["validity"] == "invalid_harness"],
        "image_sha256": sha256(image), "task_count_per_run": 16,
    })

    note = base_note + f'''
## 微调模型接入 Pi 后，能不能独立完成开发任务？

### 这轮评测检查什么？

E13 把已微调的 Qwen3-1.7B 接入 Pi 0.99.1，让模型在真实会话里选择 read、edit、write、bash 工具，并根据工具返回继续处理。每道题使用新的隔离容器和 Agent 会话；最终文件状态、必要的命令输出以及任务要求共同决定是否通过。16 道 dev 任务是从 40 个场景中按类别冻结的子集，每类 2 条。

模型使用 E09-R15 适配器；E09 固定 dev 结果为 {selected_score}/500。Pi 条件沿用 {base_config['temperature']} 温度、推理 seed {base_config['inference_seed']}、{base_config['context_window']:,} token 上下文、每题最多 {base_config['task_timeout_seconds']} 秒和 {base_config['max_tool_calls']} 次工具调用。有效成绩仅统计通过提示内容核验的运行。

### 输入检查发现了什么？

Pi 以 `[{{"type":"text","text":"…"}}]` 形式交付用户消息；历史 API 归一化代码没有提取文本块，直接把数组传给 tokenizer。R01/R02 的归档代码和逐题事件均保留了这一输入形态，因此两轮虽完成进程执行，却没有有效送达任务文本，不能作为模型能力成绩。修复后的 API 只接受受支持的文本块，并在每个任务记录任务提示哈希和模型输入提示哈希；不同冻结任务若渲染成相同输入会立即停止。

### 实际结果如何？

| 运行 | 提示条件 | 任务通过 | 工具调用 | 工具错误返回 | 输出截断 |
| --- | --- | ---: | ---: | ---: | ---: |
'''
    metrics = {}
    for run in runs:
        run_counts = counts[run["id"]]
        rows = run["rows"]
        passed_total = sum(bool(row.get("passed")) for row in rows)
        tool_calls = run_counts["tool_calls"]
        tool_errors = sum(bool(event.get("is_error")) for event in tool_calls)
        prompt = run["config"].get("selected_prompt", "简短 system prompt")
        note += (f"| {run['id']} | {prompt} | {passed_total}/{run['config']['task_count']} | "
                 f"{len(tool_calls)} | {tool_errors} | {run_counts['truncations']} |\n")
        metrics[run["id"]] = passed_total
    note += "\n| 任务类型 | " + " | ".join(run["id"] + " 通过" for run in runs) + " |\n"
    note += "| --- | " + " | ".join("---:" for _ in runs) + " |\n"
    for category in categories:
        note += f"| {LABELS[category]} | " + " | ".join(
            f"{counts[run['id']]['passed'][category]}/{counts[run['id']]['done'][category]}"
            for run in runs) + " |\n"
    note += "\n![E13 Pi dev 冻结任务的逐类结果；每类分母均为冻结的 2 条。](figures/pi-agent-dev.png)\n"
    note += "\n### 失败例子具体卡在哪里？\n\n"
    for run in runs:
        failures = counts[run["id"]]["failures"]
        if not failures:
            note += f"{run['id']} 当前已运行任务没有未通过项；还没运行的题仍在原分母中。\n\n"
            continue
        note += f"{run['id']} 的逐任务判据记录到 {len(failures)} 条未通过，代表例子如下：\n\n"
        for row in failures[:3]:
            reasons = "；".join(row.get("judgement", {}).get("reasons", [])) or "未通过"
            note += f"- `{row['task_id']}`（{LABELS.get(row['category'], row['category'])}）：{reasons}。\n"
        note += "\n"
    note += "### 现在能下什么结论？\n\n"
    if not runs[-1]["result"]:
        note += "有效运行尚未结束，暂不归纳 Pi dev 成绩。\n"
    elif len(runs) == 1:
        note += (f"修正后的有效运行 {runs[0]['id']} 在冻结 Pi dev 子集上通过 "
                 f"{metrics[runs[0]['id']]}/16。该分数只代表这16条任务，不能外推到Pi全量或官方基准。\n")
    else:
        note += ("表中只纳入已核验任务文本确实进入模型输入的运行；各分数限于相同冻结 Pi dev 子集。"
                 "被排除的旧运行仍留在证据索引与审计记录中，不参与图表和结论。\n")
    note += "\n输入完整性审计：`experiments/E13/prompt-delivery-audit.json`。旧运行原始记录保留，但因任务文本接线不完整而排除："
    note += "、".join(run["id"] for run in all_runs if run["validity"] == "invalid_harness") + "。\n"
    note += "\n## 证据索引\n\n实际配置、逐条任务、模型请求轨迹和精简结果均按运行号分开保存：\n"
    for run in all_runs:
        local_run = f".local/runs/{run['id']}"
        exclusion = "；无效接线，不参与能力比较" if run["validity"] == "invalid_harness" else ""
        note += (f"\n- `{local_run}/config.json`、`{local_run}/result.json`、`{local_run}/progress.json`"
                 f"、`{local_run}/records.jsonl`、`{local_run}/model-api.jsonl`"
                 f"、`{local_run}/model-api.stderr.txt`；公开摘要：`experiments/E13/runs/{run['id']}.json`{exclusion}\n")
    note += "\n输入完整性审计记录：`experiments/E13/prompt-delivery-audit.json`。\n"
    note += ("\n抽样配置：`configs/pi-agent-eval.json`；冻结 dev id：`configs/subsets-frozen.json` 的 `pi_dev`；"
             "few-shot 来源：`configs/prompt-frozen.json`；E09 模型选择成绩：`experiments/E09/runs/E09-R16.json`。"
             "图表数据哈希：`experiments/E13/figures/pi-agent-dev.source.json`。\n")
    (ROOT / "experiments/E13/notes.md").write_bytes(note.encode("utf-8"))


if __name__ == "__main__":
    main()
