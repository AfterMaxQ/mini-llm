"""按已有运行与笔记刷新实验总索引。"""
import json

from lab import ROOT, now

TITLES = ["环境", "CUDA 与 NF4", "公开数据", "模板与遮罩", "手工核对 loss", "LoRA 参数更新", "32 条过拟合", "保存恢复",
          "提示词基线", "1k/5k/10k 微调", "LoRA 与 QLoRA", "rank 与学习率", "数据质量", "Pi 接入", "领域轨迹",
          "教师验证", "教师生成筛选", "匹配蒸馏", "蒸馏规模", "合并与转换", "量化", "推理性能", "BFCL", "Pi 重复评测", "通用回归", "误差复盘"]


def main():
    lines = ["# MiniLLM 实验索引", "", f"最近整理：{now()}。状态来自实际运行记录，未执行的实验保持待执行。", "",
             "| 实验 | 内容 | 当前记录 | 笔记 |", "| --- | --- | --- | --- |"]
    for number, title in enumerate(TITLES):
        experiment = f"E{number:02d}"
        folder = ROOT / "experiments" / experiment
        runs = sorted((folder / "runs").glob("*.json"))
        state = "待执行"
        if runs:
            data = json.loads(runs[-1].read_text(encoding="utf-8"))
            translated = {"completed": "已执行", "failed": "失败，证据保留", "criterion_not_met": "未达到门槛", "awaiting_sample_review": "等待样本复查", "stopped_for_template_mismatch": "模板差异，已停止并保留证据", "interrupted_for_memory_pressure": "显存压力，停止并保留证据", "trained_pending_tool_eval": "训练完成，待工具评测"}
            state = f"{data['run_id']}：{translated.get(data['status'], data['status'])}"
        progress = sorted((ROOT / ".local/runs").glob(f"{experiment}-R*/progress.json"))
        if progress and not (progress[-1].parent / "result.json").exists():
            data = json.loads(progress[-1].read_text(encoding="utf-8"))
            state = f"{progress[-1].parent.name}：进行中"
            if number == 6:
                state += f"，更新 {data['latest']['step']} 次；最近检查 {data['latest_eval']['passed']}/32"
            elif "latest" in data:
                state += f"，更新 {data['latest']['step']} 次"
            elif "summary" in data:
                state += f"，{data['current_prompt']} 已记录 {data['summary']['decision_turns']} 个决策轮"
        note = f"[阅读](../experiments/{experiment}/notes.md)" if (folder / "notes.md").exists() else "—"
        lines.append(f"| {experiment} | {title} | {state} | {note} |")
    lines += ["", "## 阅读与复查", "", "实验笔记按问题和实际过程展开；各实验 runs 中保存精简结果，图表附带来源哈希。Word 正文来自同一份 Markdown，文件与归档位置集中放在分册总结后的证据索引。", "",
              "[实验规格](superpowers/specs/2026-09-30-agent-training-lab-design.md)；[执行计划](superpowers/plans/2026-10-01-local-experiments.md)。", "",
              "公开数据的工具返回属于来源标注，不等于本机真实执行。32 条过拟合属于训练机制检查；正式微调、Agent、蒸馏、标准评测及性能实验分别保留自己的分母和条件。"]
    (ROOT / "docs/实验索引.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
