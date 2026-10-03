"""从固定 Pi dev 运行记录生成教师验证笔记。"""
import argparse
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from lab import ROOT

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


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def update(run_id):
    folder = ROOT / ".local/runs" / run_id
    config = read(folder / "config.json")
    progress_path, result_path = folder / "progress.json", folder / "result.json"
    progress = read(progress_path) if progress_path.exists() else {}
    result = read(result_path) if result_path.exists() else {}
    student_run = config["source_eval_run"]
    student_eval = read(ROOT / "experiments/E14/runs" / f"{student_run}.json")
    assert student_eval.get("validity") != "invalid_harness"
    rows = progress.get("rows", [])
    assert config["experiment"] == "E15" and config["operation"] == "pi_teacher_agent_dev"
    passed = sum(bool(row.get("passed")) for row in rows)
    timeouts = sum(bool(row.get("agent_timed_out")) for row in rows)
    truncations = sum(bool(request.get("truncated")) for row in rows
                      for request in row.get("model_requests", []))
    categories = Counter(row["category"] for row in rows)
    category_passed = Counter(row["category"] for row in rows if row.get("passed"))

    figure = ROOT / "experiments/E15/figures/E15-1.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
        keys = list(LABELS)
        fig, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
        values = [category_passed[key] for key in keys]
        axis.barh(range(len(keys)), values, color="#2f855a", label="通过")
        axis.barh(range(len(keys)), [2 - value for value in values], left=values,
                  color="#cbd5e0", label="未通过")
        axis.set_yticks(range(len(keys)), [LABELS[key] for key in keys], fontproperties=font)
        axis.set_xlim(0, 2)
        axis.set_xticks([0, 1, 2])
        axis.set_xlabel("固定 dev 任务数")
        axis.set_title(f"Qwen3-4B NF4 教师 Pi dev：{len(rows)}/16 已评")
        axis.legend(prop=font, loc="lower right")
        axis.invert_yaxis()
        fig.savefig(figure, dpi=160)
        plt.close(fig)

    status = result.get("status", progress.get("status", "prepared"))
    summary = (f"本轮已完成{len(rows)}/16条固定dev任务，通过{passed}条，超时{timeouts}条，截断回复{truncations}条。"
               if status == "completed" else
               f"当前已完成{len(rows)}/16条固定dev任务，通过{passed}条，超时{timeouts}条，截断回复{truncations}条；未完成任务仍保留在16条分母中。")
    lines = [
        "# E15：4B 教师的工具任务表现",
        "",
        f"本轮直接在本机加载 Qwen3-4B 基础模型，以 NF4 量化和该模型自带的 Qwen3 chat template 接入 Pi。模型没有经过本项目的 LoRA 微调；评测使用冻结的 Pi dev16、同一 few-shot 提示和隔离工具容器。E14 学生有效接入运行 {student_run} 在同一固定 dev 子集通过 {student_eval['passed_tasks']}/{student_eval['target_tasks']}，作为同集参照。结论以实际任务记录为准，不从参数量推断教师更可靠。",
        "",
        "## 固定条件",
        "",
        "| 条件 | 配置 |",
        "| --- | --- |",
        f"| 教师 | `Qwen/Qwen3-4B`，revision `{config['model_revision']}`，NF4 双重量化 |",
        f"| tokenizer 模板 SHA-256 | `{config['tokenizer_template_sha256']}`，与 E14 学生的 Qwen3-1.7B 模板相同 |",
        f"| 模型文件清单 SHA-256 | `{config['model_manifest_sha256']}` |",
        f"| 提示 | 冻结 `few_shot`，SHA-256 `{config['prompt_sha256']}` |",
        "| 任务 | `configs/subsets-frozen.json` 的 `pi_dev`，8类各2条 |",
        f"| 解码预算 | context {config['context_window']}，每次最多生成{config['max_new_tokens']} token，温度{config['temperature']}，seed {config['inference_seed']} |",
        f"| Agent 预算 | 每题最多{config['task_timeout_seconds']}秒、{config['max_tool_calls']}次工具调用 |",
        "",
        "## 运行结果",
        "",
        summary,
        "",
    ]
    if rows:
        lines += ["![图 E15-1：八类固定 Pi dev 任务的实际通过数与未通过数](figures/E15-1.png)", ""]
        lines += ["| 任务类别 | 已评 | 通过 | 超时 |", "| --- | ---: | ---: | ---: |"]
        timed = Counter(row["category"] for row in rows if row.get("agent_timed_out"))
        for key, label in LABELS.items():
            if categories[key]:
                lines.append(f"| {label} | {categories[key]} | {category_passed[key]} | {timed[key]} |")
        lines.append("")
    failure = next((row for row in rows if not row.get("passed")), None)
    if failure:
        reasons = "；".join(failure.get("judgement", {}).get("reasons", [])) or "最终文件状态未满足冻结判据"
        answer = " ".join(failure.get("answer", "").split())[:240].replace("`", "\\`")
        lines += ["## 一个实际失败例子", "",
                  f"任务 `{failure['task_id']}`（{LABELS.get(failure['category'], failure['category'])}）未通过：{reasons}。",
                  f"模型最终回复：`{answer or '无最终文本回复'}`。该例保留在原始16题分母中。", ""]
    if (ROOT / "experiments/E15/runs/E15-R02.json").exists():
        old = read(ROOT / "experiments/E15/runs/E15-R02.json")
        if old.get("validity") == "invalid_harness":
            lines += ["## 输入完整性复核", "",
                      f"E15-R02 的16题原始记录保留，但 Pi 文本块未被旧 API 转成模型输入；其 `{old['passed_tasks']}/{old['target_tasks']}` 观测不作为教师能力成绩。修复后的本轮逐题保存任务提示与模型输入哈希，并检查不同任务输入不碰撞。审计见 `experiments/E13/prompt-delivery-audit.json`。", ""]
    lines += [
        "## 本机加载入口",
        "",
        "```python",
        "model = AutoModelForCausalLM.from_pretrained(",
        "    model_path, local_files_only=True, **model_loading_kwargs(load_in_4bit=True)",
        ")",
        "```",
        "",
        "模型和 tokenizer 均从本地 revision 读取；接口只绑定 `127.0.0.1`，Pi 每题使用新的无网络、只读根文件系统容器。",
        "",
        "## 证据索引",
        "",
        "| 内容 | 实际记录 |",
        "| --- | --- |",
        f"| 教师运行配置、逐题进度、结果与模型回复 | `{folder.relative_to(ROOT).as_posix()}/config.json`；`{folder.relative_to(ROOT).as_posix()}/progress.json`；`{folder.relative_to(ROOT).as_posix()}/result.json` |",
        f"| 教师本机 API 请求轨迹 | `{folder.relative_to(ROOT).as_posix()}/model-api.jsonl` |",
        "| 教师模型来源与 NF4 配置 | `configs/pi-agent-e15-teacher-dev.json`；`.local/models/Qwen3-4B/download-manifest.json` |",
        f"| 冻结任务及学生同集参照 | `configs/subsets-frozen.json`；`experiments/E14/runs/{student_run}.json` |",
        "",
    ]
    path = ROOT / "experiments/E15/notes.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("\n".join(lines).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    assert args.run.startswith("E15-R") and "/" not in args.run and "\\" not in args.run
    update(args.run)


if __name__ == "__main__":
    main()
