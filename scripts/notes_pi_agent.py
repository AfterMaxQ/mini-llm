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
    rows = progress.get("rows", [])
    assert config["operation"] == "pi_model_agent_dev"
    assert len(rows) == progress.get("completed", len(rows)) <= config["task_count"]
    assert [row["task_id"] for row in rows] == config["task_ids"][:len(rows)]
    if result:
        assert result["records_sha256"] == sha256(folder / "records.jsonl")
        assert result["evaluated_tasks"] == len(rows)
    return {"id": config["run_id"], "config": config, "progress": progress,
            "result": result, "rows": rows, "folder": folder}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    assert args.run.startswith("E13-R") and "/" not in args.run and "\\" not in args.run
    base_note = (ROOT / "experiments/E13/notes.md").read_text(encoding="utf-8")
    marker = "## 接下来还缺什么？"
    if marker in base_note:
        base_note = base_note.split(marker, 1)[0].rstrip()
    else:
        base_note = subprocess.check_output(
            ["git", "show", "HEAD:experiments/E13/notes.md"], cwd=ROOT, text=True, encoding="utf-8")
        base_note = base_note.split(marker, 1)[0].rstrip()
    runs = [load_run(path) for path in sorted((ROOT / ".local/runs").glob("E13-R*/config.json"))]
    assert runs and runs[-1]["id"] == args.run
    base_config = runs[0]["config"]
    for run in runs[1:]:
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
    offsets = [0.16, -0.16]
    for index, run in enumerate(runs):
        run_counts = counts[run["id"]]
        positions = [value + offsets[index % 2] for value in y]
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
    axis.set_title("E13 Pi dev：不同提示下的逐类任务结果", fontproperties=font)
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
    for run in runs:
        for name in ("progress.json", "result.json", "records.jsonl"):
            path = run["folder"] / name
            if path.exists():
                inputs.append({"run_id": run["id"], "file": name, "sha256": sha256(path)})
    write_json(image.with_suffix(".source.json"), {
        "runs": inputs, "image_sha256": sha256(image), "task_count_per_run": 16,
    })

    note = base_note + f'''
## 微调模型接入 Pi 后，能不能独立完成开发任务？

### 这轮评测检查什么？

E13 把已微调的 Qwen3-1.7B 接入 Pi 0.99.1，让模型在真实会话里选择 read、edit、write、bash 工具，并根据工具返回继续处理。每道题使用新的隔离容器和 Agent 会话；最终文件状态、必要的命令输出以及任务要求共同决定是否通过。16 道 dev 任务是从 40 个场景中按类别冻结的子集，每类 2 条。

模型始终使用 E09-R15 适配器，E09 固定 dev 结果为 {selected_score}/500。Pi 条件沿用 {base_config['temperature']} 温度、推理 seed {base_config['inference_seed']}、{base_config['context_window']:,} token 上下文、每题最多 {base_config['task_timeout_seconds']} 秒和 {base_config['max_tool_calls']} 次工具调用。下面两轮的差别是送给模型的系统提示。

### 为什么同一个适配器要看两种提示？

第一轮 E13-R01 用了简短的 Pi 项目说明。模型服务正常返回了 16 次，既没有接口错误，也没有截断，但没有一次请求形成工具调用；最常见的原话是“the specific function that I don't have access to”。所以 R01 证明了模型在这套提示下没有动手，尚未走到工具执行和返回匹配这一步。

对照 E09-R16 的离线评测后发现，411/500 的模型选择成绩使用的是冻结的 few-shot 提示。Pi 接入轮没有复用这段示例，提示并不匹配。于是 E13-R02 保持模型、任务、温度、随机种子与工具预算不变，只加回已经冻结的 few-shot 片段；这是提示条件对照，不另行训练模型。

```text
E13-R01：简短 Pi system prompt
E13-R02：同一 system prompt + E09 冻结 few-shot 片段
```

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
    note += "\n![E13 Pi dev 两种提示条件下的逐类结果；每类分母均为冻结的 2 条。](figures/pi-agent-dev.png)\n"
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
        if run["id"] == "E13-R01" and failures:
            answer = failures[0].get("answer", "").replace("\n", " ")[:180]
            note += f"\nR01 的实际回答节选：\n\n```text\n{answer}\n```\n\n"
        else:
            note += "\n"
        if run["id"] == "E13-R02":
            observed = counts[run["id"]]["tool_calls"]
            commands = Counter(
                (event.get("tool"), json.dumps(event.get("input", {}), ensure_ascii=False, sort_keys=True))
                for event in observed
            )
            if observed:
                example = observed[0]
                output = ""
                for block in example.get("result", {}).get("content", []):
                    if block.get("type") == "text":
                        output = block.get("text", "")
                        break
                note += (f"R02 虽然每道题都触发了一次工具调用，但 {len(observed)} 次调用的组合只有 "
                         f"{len(commands)} 种；最常见的是 `{example['tool']}` 执行 "
                         f"`{example.get('input', {}).get('command', '')}`。例如 `{failures[0]['task_id']}` "
                         f"返回了文件列表（{output.strip().replace(chr(10), '、')}），却没有继续读取租户配置或询问缺失信息。"
                         "这说明 Pi 确实收到了工具调用并执行了命令，但工具协议跑通不等于任务做对。\n\n"
                         "```json\n"
                         + json.dumps({"tool": example["tool"], **example.get("input", {})}, ensure_ascii=False)
                         + "\n```\n\n")
    note += "### 现在能下什么结论？\n\n"
    if len(runs) < 2 or not runs[-1]["result"]:
        note += f"当前完成的提示条件中，最高为 {max(metrics.values(), default=0)}/16；对照仍在运行，暂不归纳最终提示差异。\n"
    else:
        first, second = runs[0], runs[-1]
        delta = metrics[second["id"]] - metrics[first["id"]]
        word = "增加" if delta > 0 else "减少" if delta < 0 else "未变"
        note += (f"在相同 16 条冻结任务上，R01 为 {metrics[first['id']]}/16，R02 为 {metrics[second['id']]}/16，"
                 f"加回冻结 few-shot 后通过数{word}{abs(delta)} 条。这个差异来自提示条件，不是训练收益；"
                 "两轮只代表这组 Pi dev 样本，不能外推到 40 条 Pi dev 全量、Pi test 或官方基准。\n")
    note += "\n## 证据索引\n\n实际配置、逐条任务、模型请求轨迹和精简结果均按运行号分开保存：\n"
    for run in runs:
        local_run = f".local/runs/{run['id']}"
        note += (f"\n- `{local_run}/config.json`、`{local_run}/result.json`、`{local_run}/progress.json`"
                 f"、`{local_run}/records.jsonl`、`{local_run}/model-api.jsonl`"
                 f"、`{local_run}/model-api.stderr.txt`；公开摘要：`experiments/E13/runs/{run['id']}.json`\n")
    note += ("\n抽样配置：`configs/pi-agent-eval.json`；冻结 dev id：`configs/subsets-frozen.json` 的 `pi_dev`；"
             "few-shot 来源：`configs/prompt-frozen.json`；E09 模型选择成绩：`experiments/E09/runs/E09-R16.json`。"
             "图表数据哈希：`experiments/E13/figures/pi-agent-dev.source.json`。\n")
    (ROOT / "experiments/E13/notes.md").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
