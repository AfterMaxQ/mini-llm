"""按已有运行与笔记刷新实验总索引。"""
import json
import psutil
import subprocess

from lab import ROOT, now
from scale import alive

TITLES = ["环境", "CUDA 与 NF4", "公开数据", "模板与遮罩", "手工核对 loss", "LoRA 参数更新", "32 条过拟合", "保存恢复",
          "提示词基线", "1k/5k/10k 微调", "LoRA 与 QLoRA", "rank 与学习率", "数据质量", "Pi 接入", "领域轨迹",
          "教师验证", "教师生成筛选", "匹配蒸馏", "蒸馏规模", "合并与转换", "量化", "推理性能", "BFCL", "Pi 重复评测", "通用回归", "误差复盘"]


def process_identity_alive(config):
    identity = config.get("process_identity")
    if not identity:
        return False
    try:
        process = psutil.Process(identity["pid"])
        return (abs(process.create_time() - identity["created"]) < 0.1 and
                process.cmdline() == identity["command"] and process.is_running())
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
        return False


def pi_timeout_count(run_id):
    result_path = ROOT / "experiments/E14/runs" / f"{run_id}.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if "timed_out_tasks" in result:
            return result["timed_out_tasks"]
    path = ROOT / ".local/runs" / run_id / "records.jsonl"
    if not path.exists():
        return 0
    return sum(bool(json.loads(line).get("agent_timed_out"))
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


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
            translated['domain_mix_frozen']='512条公开轨迹与512条Pi轨迹已冻结，正式微调待执行'
            translated['training_requests_frozen']='2000个训练场景已冻结，逐条参考执行待完成'
            translated['reference_batch_verified']='1000条规则参考已执行，训练格式与迁移待核对'
            translated['dev_reference_verified']='40个dev场景与错误判据已执行，模型迁移待评测'
            translated['dev_lengths_verified']='40个dev参考历史长度已核验，模型迁移待评测'
            translated['split_inspection_verified']='train/dev近似重复已筛查，领域训练与模型迁移待执行'
            translated['test_reference_verified']='100个test场景参考与判据已核验，模型评测待执行'
            translated['test_lengths_verified']='100个test参考历史长度已核验，模型评测待执行'
            translated['test_split_inspection_verified']='train/dev/test划分筛查完成，模型迁移与最终评测待执行'
            translated['benchmark_preparation_verified']='13类BFCL清单与判分接口已核验，模型成绩待测'
            translated['preparation_verified_not_scored']='ARC-Challenge官方test与全量评分请求已核验，模型评测待执行'
            translated['preparation_failed']='准备失败，错误证据保留'
            translated['evaluation_environment_verified_not_scored']='ARC-Challenge评分环境已锁定，模型评测待执行'
            if data.get('operation')=='pi_reference_encoding' and data.get('independent_trajectories')==1000:
                translated['reference_encoding_verified']='1000条规则参考的训练格式已核验，领域训练待执行'
            state = f"{data['run_id']}：{translated.get(data['status'], data['status'])}"
        if number==11 and runs:
            evaluations=[json.loads(p.read_text(encoding='utf-8')) for p in runs]
            complete=[r for r in evaluations if r.get('status')=='completed' and r.get('config',{}).get('kind')=='tool_eval']
            if complete:
                focused=[r for r in complete if r['config'].get('data_run')=='focused-public']
                trains={r['run_id']:r['config'] for r in evaluations if r.get('config',{}).get('train_size')==1000}
                rates={trains[r['config']['source_train_run']]['learning_rate'] for r in focused
                       if r['config']['source_train_run'] in trains}
                state='5k rank对照完成；1k学习率对照'+('完成' if {0.0001,0.00005}<=rates else f'已评测{len(rates)}/2组')
        unfinished = [p for p in sorted((ROOT / ".local/runs").glob(f"{experiment}-R*/config.json"))
                      if not (p.parent / "result.json").exists()]
        if unfinished:
            current = unfinished[-1].parent
            config = json.loads(unfinished[-1].read_text(encoding="utf-8"))
            resumes=sorted(current.glob('resume-*.json'))
            if resumes:
                resumed=json.loads(resumes[-1].read_text(encoding='utf-8'))
                if {'process_id','started','command'}<=resumed.keys():config=resumed
            running=alive(config) or process_identity_alive(config)
            state = f"{current.name}：进行中" if running else f"{current.name}：进程已结束，待核对结束记录"
            data = json.loads((current / 'progress.json').read_text(encoding='utf-8')) if (current / 'progress.json').exists() else {}
            if not running and data.get('status')=='interrupted':
                state=f"{current.name}：Windows 重启中断，保留断点待恢复"
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
            elif config.get('operation') in ['pi_model_agent_dev','pi_model_agent_test'] and 'completed' in data:
                state += f"，Pi {config['split']}任务 {data['completed']}/{data['target']}，通过 {data['passed']}"
            elif (current / 'validation-progress.json').exists():
                validation=json.loads((current / 'validation-progress.json').read_text(encoding='utf-8'))
                state += f"，第 {validation['step']} 步冻结 dev loss 已检查 {validation['units']}/{validation['total_units']} 个回复"
            elif alive(config):
                state += '，准备数据与模型'
        if number==14:
            actual=[json.loads(p.read_text(encoding='utf-8')) for p in runs]
            mix=next((r for r in actual if r.get('status')=='domain_mix_frozen'),None)
            train=next((p for p in sorted((ROOT/'.local/runs').glob('E14-R*/result.json'))
                        if json.loads(p.read_text(encoding='utf-8')).get('status')=='trained_pending_tool_eval'),None)
            agent=[r for r in actual if r.get('operation','').startswith('pi_model_agent_')]
            invalid_harness=[r for r in agent if r.get('status')=='aborted_invalid_harness' or r.get('validity')=='invalid_harness']
            eligible=[r for r in agent if r.get('status')=='completed' and r.get('validity')!='invalid_harness'
                      and not r.get('config',{}).get('condition')]
            if mix:
                detail='512条公开轨迹与512条Pi轨迹已冻结'
                failed=[r for r in actual if r.get('status')=='failed']
                if failed:
                    detail+=f"；{failed[-1]['run_id']}失败记录保留"
                if train:
                    train_result=json.loads(train.read_text(encoding='utf-8'))
                    detail+=f"；领域微调{train_result['run_id']}已完成"
                public=[r for r in actual if r.get('config',{}).get('kind')=='tool_eval' and r.get('config',{}).get('data_run')=='E14-domain' and r.get('status')=='completed']
                if public:
                    score=public[-1]['summaries'][public[-1]['selected_prompt']]['trajectory_passed']
                    detail+=f"；固定公开dev {score}/100"
                for split,count in [('dev',16),('test',40)]:
                    matches=[r for r in eligible if r.get('split')==split]
                    match=matches[-1] if matches else None
                    if match:
                        detail+=f"；Pi {split} {match['run_id']} {match['passed_tasks']}/{count}"
                        timeouts=pi_timeout_count(match['run_id'])
                        if timeouts:
                            detail+=f"，{timeouts}项超时"
                    else:
                        detail+=f"；Pi {split}{count}待评"
                focused=[r for r in actual if r.get('status')=='focused_tool_data_frozen']
                if focused:
                    detail+='；专项工具集64条已冻结'
                for condition,label in [('pi_focused_baseline','短提示原模型'),('pi_focused_sft','同提示专项模型')]:
                    matches=[r for r in agent if r.get('status')=='completed' and r.get('config',{}).get('condition')==condition]
                    if matches:
                        match=matches[-1]
                        detail+=f"；{label} {match['run_id']} {match['passed_tasks']}/{match['target_tasks']}"
                if invalid_harness:
                    record=invalid_harness[-1]
                    detail+=f"；文本块接线审计判无效的旧运行：{', '.join(r['run_id'] for r in invalid_harness)}，原始记录保留"
                harness_smokes=[r for r in actual if r.get('operation',r.get('config',{}).get('operation'))=='pi_path_harness_smoke'
                                and r.get('status')=='harness_verified']
                harness_smoke=harness_smokes[-1] if harness_smokes else None
                if harness_smoke:
                    detail+=f"；{harness_smoke['run_id']}路径映射核验通过，不含模型推理"
                latest_agent=eligible[-1]['run_id'] if eligible else mix['run_id']
                state=f"{latest_agent}：{detail}"
                if unfinished:
                    state=f"{unfinished[-1].parent.name}：进行中；{detail}"
        if number==13 and runs:
            actual=[json.loads(p.read_text(encoding='utf-8')) for p in runs]
            agent=[r for r in actual if r.get('operation')=='pi_model_agent_dev']
            invalid=[r for r in agent if r.get('validity')=='invalid_harness']
            eligible=[r for r in agent if r.get('status')=='completed' and r.get('validity')!='invalid_harness']
            if eligible:
                record=eligible[-1]
                detail=f"Pi dev {record['passed_tasks']}/{record['target_tasks']}；固定16条有效接线观察"
                if invalid:
                    detail+=f"；排除的旧接线运行：{', '.join(r['run_id'] for r in invalid)}，原始记录保留"
                state=f"{record['run_id']}：{detail}"
            elif invalid:
                state=f"{invalid[-1]['run_id']}：历史运行文本块接线无效，记录保留；修复后评测待执行"
        if number==15 and runs:
            actual=[json.loads(p.read_text(encoding='utf-8')) for p in runs]
            agent=[r for r in actual if r.get('operation')=='pi_teacher_agent_dev']
            invalid=[r for r in agent if r.get('validity')=='invalid_harness']
            eligible=[r for r in agent if r.get('status')=='completed' and r.get('validity')!='invalid_harness']
            if eligible:
                record=eligible[-1]
                state=f"{record['run_id']}：教师 Pi dev {record['passed_tasks']}/{record['target_tasks']}"
            elif invalid:
                state=f"{invalid[-1]['run_id']}：文本块接线审计判无效，原始记录保留；有效教师评测待执行"
            active=[path for path in unfinished if process_identity_alive(json.loads(path.read_text(encoding='utf-8')))]
            if active:
                current=active[-1].parent
                progress=json.loads((current/'progress.json').read_text(encoding='utf-8')) if (current/'progress.json').exists() else {}
                state=f"{current.name}：教师 Pi dev {progress.get('completed',0)}/16进行中；{state}"
        active=[path for path in unfinished if process_identity_alive(json.loads(path.read_text(encoding='utf-8')))]
        if number==16 and active:
            current=active[-1].parent
            progress=json.loads((current/'progress.json').read_text(encoding='utf-8')) if (current/'progress.json').exists() else {}
            state=f"{current.name}：训练请求{progress.get('completed',0)}/{progress.get('target',512)}，初筛接受{progress.get('accepted',0)}；完整编码待核对"
        note = f"[阅读](../experiments/{experiment}/notes.md)" if (folder / "notes.md").exists() else "—"
        lines.append(f"| {experiment} | {title} | {state} | {note} |")
    lines += ["", "## 阅读与复查", "", "实验笔记按问题和实际过程展开；各实验 runs 中保存精简结果，图表附带来源哈希。Word 正文来自同一份 Markdown，文件与归档位置集中放在分册总结后的证据索引。", "",
              "[实验规格](superpowers/specs/2026-09-30-agent-training-lab-design.md)；[执行计划](superpowers/plans/2026-10-01-local-experiments.md)。", "",
              "公开数据的工具返回属于来源标注，不等于本机真实执行。32 条过拟合属于训练机制检查；正式微调、Agent、蒸馏、标准评测及性能实验分别保留自己的分母和条件。"]
    books=subprocess.check_output(['git','ls-files','-z','--','docs/reports/*.docx'],cwd=ROOT).decode('utf-8').split('\0')
    if any(books):
        lines += ['', '## 实验册', '']
        for file in filter(None, books):
            target = file.removeprefix('docs/')
            if ' ' in target:
                target = '<'+target+'>'
            lines.append(f"- [{file.rsplit('/',1)[-1].removesuffix('.docx')}]({target})")
    (ROOT / "docs/实验索引.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
