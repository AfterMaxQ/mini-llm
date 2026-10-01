"""按已有运行与笔记刷新实验总索引。"""
import json
import subprocess

from lab import ROOT, now
from scale import alive

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
        preparation = folder / 'preparation.json'
        if preparation.exists():
            prepared = json.loads(preparation.read_text(encoding='utf-8'))
            if prepared.get('status') == 'tool_chain_verified':
                state = '工具链准备已核验，模型 Agent 接入待执行'
            elif prepared.get('status') == 'configuration_verified':
                state = '配置与参数量已核验，正式 GPU 对照待执行'
        if runs:
            data = json.loads(runs[-1].read_text(encoding="utf-8"))
            translated = {"completed": "已执行", "failed": "失败，证据保留", "criterion_not_met": "未达到门槛", "awaiting_sample_review": "等待样本复查", "stopped_for_template_mismatch": "模板差异，已停止并保留证据", "interrupted_for_memory_pressure": "显存压力，停止并保留证据", "trained_pending_tool_eval": "训练完成，待工具评测", "data_ready_pending_training": "匹配数据与50条复查完成，正式训练待执行", "reference_pilot_verified": "8类训练任务原型已核验，正式轨迹与迁移训练待执行"}
            translated['pilot_encoding_verified']='8条原型的34个回复单元已核验，正式轨迹与迁移训练待执行'
            translated['reference_extension_verified']='新增8个训练模板已核验，正式轨迹与迁移训练待执行'
            translated['reference_encoding_verified']='新增参考训练格式已核验，正式轨迹与迁移训练待执行'
            translated['training_requests_frozen']='2000个训练场景已冻结，逐条参考执行待完成'
            translated['reference_batch_verified']='1000条规则参考已执行，训练格式与迁移待核对'
            translated['dev_reference_verified']='40个dev场景与错误判据已执行，模型迁移待评测'
            translated['dev_lengths_verified']='40个dev参考历史长度已核验，模型迁移待评测'
            translated['split_inspection_verified']='train/dev近似重复已筛查，领域训练与模型迁移待执行'
            translated['test_reference_verified']='100个test场景参考与判据已核验，模型评测待执行'
            translated['test_lengths_verified']='100个test参考历史长度已核验，模型评测待执行'
            translated['test_split_inspection_verified']='train/dev/test划分筛查完成，模型迁移与最终评测待执行'
            if data.get('operation')=='pi_reference_encoding' and data.get('independent_trajectories')==1000:
                translated['reference_encoding_verified']='1000条规则参考的训练格式已核验，领域训练待执行'
            state = f"{data['run_id']}：{translated.get(data['status'], data['status'])}"
        unfinished = [p for p in sorted((ROOT / ".local/runs").glob(f"{experiment}-R*/config.json"))
                      if not (p.parent / "result.json").exists()]
        if unfinished:
            current = unfinished[-1].parent
            config = json.loads(unfinished[-1].read_text(encoding="utf-8"))
            state = f"{current.name}：进行中" if alive(config) else f"{current.name}：进程已结束，待核对结束记录"
            data = json.loads((current / 'progress.json').read_text(encoding='utf-8')) if (current / 'progress.json').exists() else {}
            if number == 6:
                if 'latest' in data:
                    state += f"，更新 {data['latest']['step']} 次；最近检查 {data['latest_eval']['passed']}/32"
            elif "latest" in data:
                state += f"，更新 {data['latest']['step']} 次"
            elif "summary" in data:
                state += f"，{data['current_prompt']} 已记录 {data['summary']['decision_turns']} 个决策轮"
            elif config.get('operation')=='pi_reference_batch' and 'completed' in data:
                state += f"，参考过程 {data['completed']}/{data['target']}，有效 {data['valid']} 条"
            elif config.get('operation') in ['pi_dev_probe','pi_test_probe'] and 'completed' in data:
                state += f"，{config['split']}参考 {data['completed']}/{data['target']}"
            elif (current / 'validation-progress.json').exists():
                validation=json.loads((current / 'validation-progress.json').read_text(encoding='utf-8'))
                state += f"，第 {validation['step']} 步完整 dev loss 已检查 {validation['units']}/{validation['total_units']} 个回复"
            elif alive(config):
                state += '，准备数据与模型'
        if number==14:
            actual=[json.loads(p.read_text(encoding='utf-8')) for p in runs]
            if any(r.get('operation')=='pi_reference_encoding' and r.get('status')=='reference_encoding_verified' and r.get('independent_trajectories')==1000 for r in actual):
                state += '；1000条规则参考格式已核验'
            if any(r.get('status')=='dev_reference_verified' for r in actual):
                state += '；40个dev场景已冻结，模型成绩待测'
            if any(r.get('status')=='test_reference_verified' for r in actual):
                state += '；100个test场景已冻结，模型成绩待测'
        note = f"[阅读](../experiments/{experiment}/notes.md)" if (folder / "notes.md").exists() else "—"
        lines.append(f"| {experiment} | {title} | {state} | {note} |")
    lines += ["", "## 阅读与复查", "", "实验笔记按问题和实际过程展开；各实验 runs 中保存精简结果，图表附带来源哈希。Word 正文来自同一份 Markdown，文件与归档位置集中放在分册总结后的证据索引。", "",
              "[实验规格](superpowers/specs/2026-09-30-agent-training-lab-design.md)；[执行计划](superpowers/plans/2026-10-01-local-experiments.md)。", "",
              "公开数据的工具返回属于来源标注，不等于本机真实执行。32 条过拟合属于训练机制检查；正式微调、Agent、蒸馏、标准评测及性能实验分别保留自己的分母和条件。"]
    books=subprocess.check_output(['git','ls-files','-z','--','docs/reports/*.docx'],cwd=ROOT).decode('utf-8').split('\0')
    if any(books):
        lines += ['', '## 实验册', '']
        lines += [f"- [{file.rsplit('/',1)[-1].removesuffix('.docx')}]({file.removeprefix('docs/')})" for file in books if file]
    (ROOT / "docs/实验索引.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
