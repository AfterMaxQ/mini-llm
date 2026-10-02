"""从实际多次 SFT 运行归纳规模实验；正在训练时替换当前进展。"""
import csv
import hashlib
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from lab import ROOT, sha256, write_json


def tool_results():
    completed=[];active=[]
    for path in sorted((ROOT/'.local/runs').glob('E09-R*/config.json')):
        config=json.loads(path.read_text(encoding='utf-8'))
        if config.get('kind')!='tool_eval':continue
        source=ROOT/'.local/runs'/config['source_train_run']
        training=json.loads((source/'config.json').read_text(encoding='utf-8'))
        result=path.parent/'result.json'
        if result.exists():
            value=json.loads(result.read_text(encoding='utf-8'))
            if value['status']!='completed':continue
            summary=value['summaries'][value['selected_prompt']]
            assert summary['trajectories']==summary['evaluated_trajectories']==500
            assert summary['decision_turns']==928
            assert config['frozen_prompt_sha256']==sha256(ROOT/'configs/prompt-frozen.json')
            assert config['adapter_sha256']==sha256(source/'selected-adapter/adapter_model.safetensors')
            assert summary['rows_sha256']==sha256(path.parent/(value['selected_prompt']+'.jsonl'))
            completed.append({'train_run':source.name,'eval_run':path.parent.name,
                              'size':training['train_size'],'seed':training['seed'],
                              'summary':summary,'raw_rows_sha256':summary['rows_sha256']})
        elif (path.parent/'progress.json').exists():
            progress=json.loads((path.parent/'progress.json').read_text(encoding='utf-8'))
            active.append((source.name,path.parent.name,progress['summary']['decision_turns']))
    return completed,active


def draw_comparison(completed):
    if not completed:return None
    baseline=json.loads((ROOT/'experiments/E08/summary.json').read_text(encoding='utf-8'))
    frozen=json.loads((ROOT/'configs/prompt-frozen.json').read_text(encoding='utf-8'))
    baseline=baseline[frozen['selected_prompt']]
    folder=ROOT/'experiments/E09'
    rows=[{'train_run':'官方学生','eval_run':frozen['source_run'],'size':0,'seed':'',
           'passed':baseline['trajectory_passed'],'denominator':500,
           'raw_rows_sha256':baseline['rows_sha256']}]
    rows += [{'train_run':r['train_run'],'eval_run':r['eval_run'],'size':r['size'],
              'seed':r['seed'],'passed':r['summary']['trajectory_passed'],'denominator':500,
              'raw_rows_sha256':r['raw_rows_sha256']} for r in completed]
    data=folder/'dev-tool-comparison.csv'
    with data.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    fig,axis=plt.subplots(figsize=(6.4,3.5),layout='constrained')
    labels=['提示词基线\n原模型 + 三例']+[f"{r['size']//1000}k / seed {r['seed']}\n{r['train_run']}" for r in completed]
    rates=[100*r['passed']/500 for r in rows]
    bars=axis.bar(range(len(rows)),rates,color=['#2563EB']+['#059669']*len(completed))
    axis.set_xticks(range(len(rows)),labels,fontproperties=font)
    axis.set_ylim(0,105);axis.set_ylabel('整条轨迹通过率（%）',fontproperties=font)
    axis.set_title('相同固定提示与完整 dev；各训练 seed 单独显示',fontproperties=font)
    axis.yaxis.grid(True,color='#E5E7EB');axis.set_axisbelow(True)
    for bar,row in zip(bars,rows):
        axis.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
                  f"{100*row['passed']/500:.1f}%\n{row['passed']}/500",ha='center',fontproperties=font,fontsize=9)
    for side in ['top','right']:axis.spines[side].set_visible(False)
    path=folder/'figures/dev-tool-comparison.png';fig.savefig(path,dpi=300);plt.close(fig)
    write_json(path.with_suffix('.source.json'),{'source_file':data.name,'source_sha256':sha256(data),
               'image_sha256':sha256(path),'denominator':500,'rows':rows,
               'scope':'同一dev与固定提示；训练seed分别保留，不作为独立任务样本合并'})
    return path


def put_evidence(evidence,label,files):
    row=f'| {label} | {files} |';prefix=f'| {label} |';lines=[];seen=False
    for line in evidence.splitlines():
        if line.startswith(prefix):
            if not seen:lines.append(row)
            seen=True
        else:lines.append(line)
    result='\n'.join(lines)+'\n'
    return result if seen else result.replace('| 后续训练与生成结果 |',row+'\n| 后续训练与生成结果 |')


def update_summary(completed):
    path=ROOT/'docs/reports/summaries/02.md'
    if not path.exists():return
    evidence='## 证据索引'+path.read_text(encoding='utf-8').split('## 证据索引',1)[1]
    pi_summary=ROOT/'docs/reports/summaries/02B.md'
    if pi_summary.exists():
        for line in pi_summary.read_text(encoding='utf-8').splitlines():
            if line.startswith('| Pi ') and line not in evidence:
                evidence=evidence.replace('| 后续训练与生成结果 |',line+'\n| 后续训练与生成结果 |')
    rows=[('工具生成比较','experiments/E09/dev-tool-comparison.csv'),
          ('1k 配对变化与失败例子','experiments/E09/E09-R06-comparison.json；experiments/E09/E09-R06-paired.csv'),
          ('5k 配对变化与失败例子','experiments/E09/E09-R10-comparison.json；experiments/E09/E09-R10-paired.csv'),
          ('Windows 记录写入排查','experiments/E09/runs/E09-R07.json；experiments/E09/runs/E09-R08.json'),
          ('Pi 工具链准备','experiments/E13/preparation.json；experiments/E13/notes.md'),
          ('Pi 领域混合来源哈希失败与修正','experiments/E14/runs/E14-R17.json；configs/domain-mix.json'),
          ('Pi 512+512冻结清单与数据构建','scripts/domain_mix.py；configs/domain-mix-frozen.json；experiments/E14/runs/E14-R18.json'),
          ('Pi 领域微调及固定公开dev','configs/sft-domain.json；configs/offline-eval-domain.json；configs/scale-domain.json'),
          ('Pi 实际工具定义与容器参数','configs/pi-tools.json；configs/pi-sandbox.json'),
          ('Pi 工具适配与核验入口','scripts/pi_sandbox.mjs；scripts/pi_tools_probe.mjs'),
          ('rank 对照准备','experiments/E11/preparation.json；configs/rank.json；configs/sft-rank8.json；configs/sft-rank32.json'),
          ('等量数据筛选与复查','configs/data-quality.json；configs/data-quality-frozen.json；experiments/E12/sample-review.json；experiments/E12/data-comparison.csv'),
          ('数据质量对照入口','scripts/data_quality.py；configs/quality.json；configs/sft-quality-focused.json'),
          ('E11 rank训练与评测','experiments/E11/runs/E11-R01.json；experiments/E11/runs/E11-R02.json；experiments/E11/rank-tool-comparison.csv'),
          ('1k训练与固定评测清单','configs/sft-focused.json；configs/sft-lr-low.json；configs/learning-rate.json；configs/subsets-frozen.json'),
          ('筛选失败与重新冻结','experiments/E12/runs/E12-R01.json；experiments/E12/runs/E12-R02.json'),
          ('Pi 任务参考与判据核验','experiments/E14/runs/E14-R01.json；experiments/E14/reference-calls.csv；experiments/E14/reference-events.jsonl'),
          ('Pi 训练任务原型与核验入口','configs/pi-task-catalog.json；scripts/pi_tasks.mjs；scripts/pi_task_probe.mjs'),
          ('Pi 多轮训练格式与监督检查','configs/pi-reference-data.json；scripts/pi_reference_data.py；experiments/E14/runs/E14-R02.json；experiments/E14/current-reply-mask.csv')]
    interruption=ROOT/'.local/runs/E09-R13/interruption-20261001T211127+08.json'
    if interruption.exists():
        run_record=ROOT/'experiments/E09/runs/E09-R13.json'
        interruption_evidence='experiments/E09/notes.md；.local/runs/E09-R13'
        if run_record.exists():interruption_evidence='experiments/E09/notes.md；experiments/E09/runs/E09-R13.json；.local/archive/E09/E09-R13.zip'
        rows.append(('E09-R13 重启中断与断点恢复',interruption_evidence))
    rows.append(('Pi 场景去重与失败记录','experiments/E14/scene-generation-audit.json；experiments/E14/runs/E14-R05.json；experiments/E14/runs/E14-R06.json；experiments/E14/runs/E14-R07.json'))
    for label,files in rows:
        if (ROOT/files.split('；')[0]).exists():
            evidence=put_evidence(evidence,label,files)
    intro='# 从提示词基线走向正式微调\n\n原模型在相同完整 dev 上，零样本通过 268/500，加入三条固定示例后通过 355/500。少样本提示提高了调用轮表现，也增加了本该询问或不调用时的误调用。后续模型都沿用这个已冻结的提示。\n\n'
    if completed:
        r=completed[-1];s=r['summary']
        intro+=f"最近完成的 E09 规模组 {r['eval_run']} 使用 {r['size']:,} 条训练轨迹，完整 dev 通过 {s['trajectory_passed']}/500；调用轮为 {s['call_turn_passed']}/568，不调用轮为 {s['no_call_turn_passed']}/360。loss 与实际工具决策分开看，不能把更容易生成调用当成任务成功率提升。\n\n"
    ten_k=sorted((r for r in completed if r['size']==10000),key=lambda r:r['seed'])
    if ten_k:
        assert [r['seed'] for r in ten_k]==[17,42,2026]
        passed=[r['summary']['trajectory_passed'] for r in ten_k]
        no_call=[r['summary']['no_call_turn_passed'] for r in ten_k]
        intro+=f"10k 三个 seed（17、42、2026）分别通过 {passed[0]}、{passed[1]}、{passed[2]}/500；不调用轮落在 {min(no_call)}—{max(no_call)}/360。三轮在同一份 dev 上完成，不能当作独立测试集成绩。\n\n"
    lr_runs={run_id:json.loads((ROOT/'experiments/E11/runs'/f'{run_id}.json').read_text(encoding='utf-8'))
             for run_id in ('E11-R03','E11-R04','E11-R05','E11-R06')}
    lr_low,lr_low_eval,lr_high,lr_high_eval=(lr_runs[key] for key in ('E11-R03','E11-R04','E11-R05','E11-R06'))
    if all(item.get('status') in {'trained_pending_tool_eval','completed'} for item in lr_runs.values()):
        assert (lr_low['config']['train_size'],lr_low['config']['rank'],lr_low['config']['seed']) == (1000,16,17)
        assert (lr_high['config']['train_size'],lr_high['config']['rank'],lr_high['config']['seed']) == (1000,16,17)
        assert lr_low['config']['data_file_hashes']['train-1000.jsonl']==lr_high['config']['data_file_hashes']['train-1000.jsonl']
        low=lr_low_eval['summaries'][lr_low_eval['selected_prompt']]
        high=lr_high_eval['summaries'][lr_high_eval['selected_prompt']]
        assert low['trajectories']==high['trajectories']==100
        intro+=f"1k 学习率对照使用相同训练轨迹、rank 16、seed 17 和固定 100 条 dev：1e-4 组通过 {low['trajectory_passed']}/100，5e-5 组通过 {high['trajectory_passed']}/100；调用轮为 {low['call_turn_passed']}/112 与 {high['call_turn_passed']}/112，不调用轮为 {low['no_call_turn_passed']}/72 与 {high['no_call_turn_passed']}/72。5e-5 组有 {high['truncated']} 条截断，仍保留在分母中。该单 seed 对照只说明本轮结果。\n\n"
    quality_train=json.loads((ROOT/'experiments/E12/runs/E12-R03.json').read_text(encoding='utf-8'))
    quality_eval=json.loads((ROOT/'experiments/E12/runs/E12-R04.json').read_text(encoding='utf-8'))
    if quality_eval.get('status')=='completed' and quality_train.get('status')=='trained_pending_tool_eval':
        base=lr_low_eval['summaries'][lr_low_eval['selected_prompt']]
        quality=quality_eval['summaries'][quality_eval['selected_prompt']]
        assert quality['trajectories']==base['trajectories']==100
        assert quality_train['config']['rank']==lr_low['config']['rank']==16
        assert quality_train['config']['learning_rate']==lr_low['config']['learning_rate']==1e-4
        intro+=f"数据质量对照在固定 100 条 dev 上，基础组通过 {base['trajectory_passed']}/100，额外筛选组通过 {quality['trajectory_passed']}/100。两组各用 1k、rank 16、学习率 1e-4、seed 17，但筛选组替换了部分样本且只有一个 seed；这是本次数据组的差异，不能归因于筛选规则本身。\n\n"
    precision_runs={}
    for run_id in ('E09-R10','E10-R02'):
        result_path=ROOT/'experiments'/run_id[:3]/'runs'/f'{run_id}.json'
        if result_path.exists():
            record=json.loads(result_path.read_text(encoding='utf-8'))
            if record.get('status')=='completed':
                precision_runs[run_id]=record['summaries'][record['selected_prompt']]
    if len(precision_runs)==2:
        nf4=precision_runs['E09-R10'];bf16=precision_runs['E10-R02']
        intro+=f"精度对照沿用同一生成后端和完整 dev：NF4 训练组通过 {nf4['trajectory_passed']}/500，BF16 训练组通过 {bf16['trajectory_passed']}/500；调用轮分别为 {nf4['call_turn_passed']}/568 与 {bf16['call_turn_passed']}/568，不调用轮为 {nf4['no_call_turn_passed']}/360 与 {bf16['no_call_turn_passed']}/360。三条轨迹的差异说明这批结果接近，不能据此单独断言精度优劣。\n\n"
    intro+='训练中遇到的 LoRA 重复准备和显存缓存问题，连同处理过程与失败条件都保留在正文。各组效果按自己的冻结数据与验证条件比较；最终配置只由 dev 选择，test 留作独立检查。\n\n'
    pi_intro='# 从工具链到可执行的任务数据\n\n'
    preparation=ROOT/'experiments/E13/preparation.json'
    if preparation.exists():
        p=json.loads(preparation.read_text(encoding='utf-8'))
        if p['status']=='tool_chain_verified':
            pi_intro+=f"Pi 的四个工具已在本机任务容器实际核验：{p['request_count']} 次请求中，{p['normal_returns']} 次正常返回，{p['expected_error_returns']} 次触发预期错误或预算限制。最终文件、逐次返回与隔离设置都保留了证据；这轮探针请求不是模型 Agent 成绩。\n\n"
    if (ROOT/'experiments/E11/preparation.json').exists():
        intro+='rank 对照使用同一份 5k 数据，学习率对照另在同一份 1k 数据上固定 rank 16。两组各自使用相同的验证清单和保存规则，分开比较；配置选择只使用 dev。\n\n'
    if (ROOT/'configs/data-quality-frozen.json').exists():
        intro+='数据质量准备保留 5k 筛选池与阅读抽查，训练比较使用来源和类别匹配的两份 1k 数据。规则的误删和漏检、筛选成本、监督量与模型表现分别记录，效果不能由“通过筛选”直接推出。\n\n'
    if (ROOT/'experiments/E14/notes.md').exists():
        pi_intro+='八类 Pi 训练任务原型的参考操作已实际执行，同一判据也拒绝了初始错误和明确错误候选。它们帮助检查任务是否判得准；正式轨迹规模、独立任务集和模型迁移表现继续分别验证。\n\n'
    domain_runs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'experiments/E14/runs').glob('*.json'))]
    domain_train=next((r for r in domain_runs if r.get('status')=='trained_pending_tool_eval'),None)
    domain_evals=[r for r in domain_runs if r.get('config',{}).get('kind')=='tool_eval'
                  and r['config'].get('source_train_run')=='E14-R19']
    if domain_train and domain_evals:
        evaluation=domain_evals[-1]['summaries'][domain_evals[-1]['selected_prompt']]
        pi_intro+=f"领域混合模型使用512条公开轨迹和512条Pi参考轨迹训练，更新{domain_train['steps']}步，最低冻结dev loss为{domain_train['selected_dev_loss']:.6f}；固定100条公开dev通过{evaluation['trajectory_passed']}/100。Pi Agent迁移另用隔离任务集评估。\n\n"
        evidence=put_evidence(evidence,'E14领域混合训练与固定公开dev','experiments/E14/runs/E14-R19.json；experiments/E14/runs/E14-R20.json')
    extension=ROOT/'experiments/E14/runs/E14-R03.json'
    if extension.exists() and json.loads(extension.read_text(encoding='utf-8'))['status']=='reference_extension_verified':
        pi_intro+='新增八个训练模板改变了配置层级、单位换算、去重顺序和故障原因，两批共16个训练模板族。参考步骤由规则给定，逐条执行不等于模型自主完成任务。\n\n'
        evidence=put_evidence(evidence,'Pi 新增模板与错误方案','scripts/pi_tasks_extended.mjs；experiments/E14/runs/E14-R03.json；experiments/E14/extension-calls.csv；experiments/E14/extension-events.jsonl')
    converted=ROOT/'experiments/E14/runs/E14-R04.json'
    if converted.exists() and json.loads(converted.read_text(encoding='utf-8'))['status']=='reference_encoding_verified':
        evidence=put_evidence(evidence,'Pi 新增参考的训练格式','configs/pi-reference-extension.json；experiments/E14/runs/E14-R04.json')
    scenes=ROOT/'experiments/E14/runs/E14-R07.json'
    if scenes.exists() and json.loads(scenes.read_text(encoding='utf-8'))['status']=='training_requests_frozen':
        evidence=put_evidence(evidence,'Pi 训练场景与批量参考入口','configs/pi-task-data.json；configs/pi-reference-batch.json；scripts/pi_task_data.py；scripts/pi_reference_batch.mjs')
    pi_runs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'experiments/E14/runs').glob('*.json'))]
    batch=[r for r in pi_runs if r.get('status')=='reference_batch_verified']
    if batch:
        r=batch[-1]
        pi_intro+=f"正式批量执行已获得1,000条有效规则参考，共{r['reference_tool_calls']:,}次工具调用，保留{r['reference_error_returns']}次预期错误返回。它们来自16个训练模板族，教师生成与模型迁移仍各自记录。\n\n"
        evidence=put_evidence(evidence,'Pi 1000条实际规则参考',f"experiments/E14/runs/{r['run_id']}.json；experiments/E14/batch-reference.csv；experiments/E14/batch-audit.json")
        encoded=[v for v in pi_runs if v.get('status')=='reference_encoding_verified' and v['config']['source_run']==r['run_id']]
        if encoded:
            v=encoded[-1]
            pi_intro+=f"这批过程展开为{v['assistant_units']:,}个当前回复单元，{v['supervised_tokens']:,}个监督token。实际返回、模板和遮罩都已核验，最长{v['max_sequence_length']:,}token，没有截断；领域微调与模型迁移成绩另行评估。\n\n"
            evidence=put_evidence(evidence,'Pi 1000条参考训练格式',f"scripts/pi_reference_data.py；experiments/E14/runs/{v['run_id']}.json")
    dev=[r for r in pi_runs if r.get('status')=='dev_reference_verified']
    if dev:
        r=dev[-1]
        pi_intro+='40个dev场景已用独立模板与仓库族建立，参考操作和错误候选逐条在容器执行。它们分属八个任务族，每族五个场景；模型迁移成绩与最终test仍要单独评测。\n\n'
        evidence=put_evidence(evidence,'Pi dev任务与判据核验',f"configs/pi-dev-tasks.json；scripts/pi_tasks_dev.mjs；scripts/pi_dev_probe.mjs；experiments/E14/runs/{r['run_id']}.json")
        if (ROOT/'experiments/E14/dev-audit.json').exists():
            evidence=put_evidence(evidence,'Pi dev逐条结果与划分复核','experiments/E14/dev-reference.csv；experiments/E14/dev-audit.json')
    lengths=[r for r in pi_runs if r.get('status')=='dev_lengths_verified']
    if lengths:
        r=lengths[-1]
        evidence=put_evidence(evidence,'Pi dev参考历史长度',f"scripts/pi_dev_lengths.py；experiments/E14/runs/{r['run_id']}.json；experiments/E14/dev-lengths.csv")
    tests=[r for r in pi_runs if r.get('status')=='test_reference_verified']
    failed_tests=[r for r in pi_runs if r.get('operation')=='pi_test_probe' and r['status']=='failed']
    if failed_tests:
        r=failed_tests[-1]
        evidence=put_evidence(evidence,'Pi test生成失败记录',f"experiments/E14/runs/{r['run_id']}.json")
    if tests:
        r=tests[-1]
        pi_intro+='最终100个test场景已冻结，16个新任务族按预定类别配额分配。题目、参考与错误判据已实际执行，模型没有接触test；后续选择配置仍只用dev。\n\n'
        evidence=put_evidence(evidence,'Pi test任务与判据核验',f"configs/pi-test-tasks.json；scripts/pi_tasks_test.mjs；scripts/pi_eval_tasks.mjs；experiments/E14/runs/{r['run_id']}.json")
        if (ROOT/'experiments/E14/test-audit.json').exists():
            evidence=put_evidence(evidence,'Pi test逐条结果与归档复核','experiments/E14/test-reference.csv；experiments/E14/test-audit.json')
    test_lengths=[r for r in pi_runs if r.get('status')=='test_lengths_verified']
    if test_lengths:
        r=test_lengths[-1]
        evidence=put_evidence(evidence,'Pi test参考历史长度',f"scripts/pi_eval_lengths.py；experiments/E14/runs/{r['run_id']}.json；experiments/E14/test-lengths.csv")
    test_inspections=[r for r in pi_runs if r.get('status')=='test_split_inspection_verified']
    if test_inspections:
        r=test_inspections[-1]
        evidence=put_evidence(evidence,'Pi test与train/dev划分筛查',f"scripts/pi_split_inspect.py；experiments/E14/runs/{r['run_id']}.json")
    inspection=[r for r in pi_runs if r.get('status')=='split_inspection_verified']
    if inspection:
        r=inspection[-1]
        evidence=put_evidence(evidence,'Pi train/dev近似重复筛查',f"scripts/pi_split_inspect.py；experiments/E14/runs/{r['run_id']}.json")
    for record in pi_runs:
        if not record.get('operation','').startswith('pi_model_agent_'):
            continue
        record_path='experiments/E14/runs/'+record['run_id']+'.json'
        if not any(record_path in line for line in evidence.splitlines()):
            evidence=put_evidence(evidence,'E14 '+record['run_id']+' Pi Agent记录',record_path)
    path_smoke=next((record for record in pi_runs if record.get('operation')=='pi_path_harness_smoke'
                     and record.get('status')=='harness_verified'),None)
    if path_smoke:
        files=f"scripts/pi_path_smoke.mjs；experiments/E14/runs/{path_smoke['run_id']}.json"
        if not any(path_smoke['run_id']+'.json' in line for line in evidence.splitlines()):
            evidence=put_evidence(evidence,'Pi Agent路径映射核验（未运行模型）',files)
        pi_intro+='Pi Agent首次dev模型运行在前2/16个任务时因容器cwd路径映射错误中止，不计模型成绩；单独烟测确认会话工作目录及SessionManager目录均为`/workspace`，隔离write/read往返成功。烟测未请求模型，因此Pi dev模型成绩仍待完整冻结16题评测。\n\n'
    deduplicated=[];seen_files=set()
    for line in evidence.splitlines():
        if line.startswith('| ') and line.count('|') >= 3:
            files=line.rsplit('|',2)[1].strip()
            if files in seen_files:
                continue
            seen_files.add(files)
        deduplicated.append(line)
    evidence='\n'.join(deduplicated)+'\n'
    pi_rows=[line for line in evidence.splitlines() if line.startswith('| Pi ')]
    pi_evidence_rows=[
        ('Pi 实际工具定义与容器参数','configs/pi-tools.json；configs/pi-sandbox.json'),
        ('Pi 工具适配与核验入口','scripts/pi_sandbox.mjs；scripts/pi_tools_probe.mjs'),
        ('Pi 工具链准备','experiments/E13/preparation.json；experiments/E13/notes.md'),
        ('Pi 任务参考与判据核验','experiments/E14/runs/E14-R01.json；experiments/E14/reference-calls.csv；experiments/E14/reference-events.jsonl'),
        ('Pi 训练任务原型与核验入口','configs/pi-task-catalog.json；scripts/pi_tasks.mjs；scripts/pi_task_probe.mjs'),
        ('Pi 多轮训练格式与监督检查','configs/pi-reference-data.json；scripts/pi_reference_data.py；experiments/E14/runs/E14-R02.json；experiments/E14/current-reply-mask.csv'),
        ('Pi 新增模板与错误方案','scripts/pi_tasks_extended.mjs；experiments/E14/runs/E14-R03.json；experiments/E14/extension-calls.csv；experiments/E14/extension-events.jsonl'),
        ('Pi 新增参考的训练格式','configs/pi-reference-extension.json；experiments/E14/runs/E14-R04.json'),
        ('Pi 训练场景与批量参考入口','configs/pi-task-data.json；configs/pi-reference-batch.json；scripts/pi_task_data.py；scripts/pi_reference_batch.mjs'),
        ('Pi 场景去重与失败记录','experiments/E14/scene-generation-audit.json；experiments/E14/runs/E14-R05.json；experiments/E14/runs/E14-R06.json；experiments/E14/runs/E14-R07.json'),
        ('Pi dev任务与判据核验','configs/pi-dev-tasks.json；scripts/pi_tasks_dev.mjs；scripts/pi_dev_probe.mjs；experiments/E14/runs/E14-R09.json'),
        ('Pi dev参考历史长度','scripts/pi_dev_lengths.py；experiments/E14/runs/E14-R10.json；experiments/E14/dev-lengths.csv'),
        ('Pi dev逐条结果与划分复核','experiments/E14/dev-reference.csv；experiments/E14/dev-audit.json'),
        ('Pi 1000条实际规则参考','experiments/E14/runs/E14-R08.json；experiments/E14/batch-reference.csv；experiments/E14/batch-audit.json'),
        ('Pi 1000条参考训练格式','scripts/pi_reference_data.py；experiments/E14/runs/E14-R11.json'),
        ('Pi train/dev近似重复筛查','scripts/pi_split_inspect.py；experiments/E14/runs/E14-R12.json'),
        ('Pi test任务与判据核验','configs/pi-test-tasks.json；scripts/pi_tasks_test.mjs；scripts/pi_eval_tasks.mjs；experiments/E14/runs/E14-R14.json'),
        ('Pi test逐条结果与归档复核','experiments/E14/test-reference.csv；experiments/E14/test-audit.json'),
        ('Pi test参考历史长度','scripts/pi_eval_lengths.py；experiments/E14/runs/E14-R15.json；experiments/E14/test-lengths.csv'),
        ('Pi test与train/dev划分筛查','scripts/pi_split_inspect.py；experiments/E14/runs/E14-R16.json'),
        ('Pi test生成失败记录','experiments/E14/runs/E14-R13.json'),
        ('Pi 领域混合来源哈希失败与修正','experiments/E14/runs/E14-R17.json；configs/domain-mix.json'),
        ('Pi 512+512冻结清单与数据构建','scripts/domain_mix.py；configs/domain-mix-frozen.json；experiments/E14/runs/E14-R18.json'),
        ('Pi 领域微调及固定公开dev','configs/sft-domain.json；configs/offline-eval-domain.json；configs/scale-domain.json'),
        ('Pi 接线失败运行（不计模型成绩）','experiments/E14/runs/E14-R21.json'),
        ('Pi 会话路径映射烟测（不含模型推理）','scripts/pi_path_smoke.mjs；experiments/E14/runs/E14-R22.json'),
    ]
    present_labels={line.split('|',2)[1].strip() for line in pi_rows}
    for label,files in pi_evidence_rows:
        if label in present_labels or not all((ROOT/file).exists() for file in files.split('；')):
            continue
        pi_rows.append(f'| {label} | {files} |')
    main_evidence='\n'.join(line for line in evidence.splitlines() if not line.startswith('| Pi '))+'\n'
    if path_smoke:
        smoke_files=f"scripts/pi_path_smoke.mjs；experiments/E14/runs/{path_smoke['run_id']}.json"
        row=f'| E14 R22 路径映射烟测（未运行模型） | {smoke_files} |'
        lines=main_evidence.splitlines();found=False
        for index,line in enumerate(lines):
            if line.startswith('| E14 R22 路径映射烟测（未运行模型） |'):
                lines[index]=row;found=True
        if not found:
            marker='| 后续训练与生成结果 |'
            marker_index=next((index for index,line in enumerate(lines) if line.startswith(marker)),len(lines))
            lines.insert(marker_index,row)
        main_evidence='\n'.join(lines)+'\n'
    pi_evidence='## 证据索引\n\n| 内容 | 对应记录 |\n| --- | --- |\n'+'\n'.join(pi_rows)+'\n'
    pi_evidence+='\n完整工具返回、文件状态与容器记录保存在本地运行档案，公开结果给出来源哈希。理解回顾仍未回答，参考解释不代表用户已掌握。\n'
    path.write_text(intro+main_evidence,encoding='utf-8')
    if pi_rows:pi_summary.write_text(pi_intro+pi_evidence,encoding='utf-8')


def draw_curve(run_id):
    metrics=ROOT/'.local/runs'/run_id/'metrics.jsonl'
    if not metrics.exists():return None
    raw=metrics.read_bytes();rows=[json.loads(x) for x in raw.decode('utf-8').splitlines()]
    folder=ROOT/'experiments'/run_id.split('-')[0];folder.mkdir(parents=True,exist_ok=True)
    csv_path=folder/(run_id+'-metrics.csv')
    fields=['time','step','loss','eval_loss','grad_norm','learning_rate','epoch','num_tokens','supervised_tokens','step_seconds','peak_allocated_mib','peak_reserved_mib','allocated_mib','reserved_mib','device_free_mib','reserved_before_step_mib','reserved_after_release_mib']
    with csv_path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    train=[r for r in rows if 'loss' in r];dev=[r for r in rows if 'eval_loss' in r]
    if not train:return {'rows':rows,'train':train,'dev':dev}
    figures=folder/'figures';figures.mkdir(exist_ok=True)
    path=figures/(run_id+'-loss.png');source=path.with_suffix('.source.json')
    saved=json.loads(source.read_text(encoding='utf-8')) if source.exists() else {}
    segments=[]
    for row in train:
        if segments and int(row['step']) < int(segments[-1][-1]['step']):
            segments.append([])
        if not segments:segments.append([])
        segments[-1].append(row)
    needs_resume_plot = len(segments)>1 and saved.get('plot_version')!=3
    if saved.get('source_sha256')!=sha256(csv_path) or needs_resume_plot:
        font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
        fig,axis=plt.subplots(figsize=(6.4,3.5),layout='constrained')
        resume_files=sorted((ROOT/'.local/runs'/run_id).glob('resume-*.json'))
        resume=json.loads(resume_files[-1].read_text(encoding='utf-8')) if resume_files else {}
        for index,segment in enumerate(segments):
            if len(segments)==1:
                label='训练：每次更新';color='#059669';style='-'
            elif index==0:
                label='首次运行（含未保存尾段）';color='#94A3B8';style='--'
            else:
                checkpoint=resume.get('checkpoint','未知 checkpoint')
                label=f'{checkpoint} 恢复' if index==1 else f'后续恢复段 {index}'
                color='#059669' if index==1 else '#2563EB';style='-'
            axis.plot([r['step'] for r in segment],[r['loss'] for r in segment],
                      color=color,linestyle=style,linewidth=1.0,label=label)
        if dev:
            count=dev[-1].get('eval_trajectories',500)
            axis.plot([r['step'] for r in dev],[r['eval_loss'] for r in dev],color='#059669',linestyle='None',marker='o',label=f'dev {count} 条：实测点，token 加权')
        axis.set_xlabel('optimizer step',fontproperties=font);axis.set_ylabel('assistant-only 交叉熵',fontproperties=font)
        axis.set_title(run_id+'：真实训练与验证 loss，未平滑',fontproperties=font)
        axis.legend(prop=font,frameon=False);axis.yaxis.grid(True,color='#E5E7EB');axis.set_axisbelow(True)
        for side in ['top','right']:axis.spines[side].set_visible(False)
        fig.savefig(path,dpi=300);plt.close(fig)
        write_json(source,{'run_id':run_id,'plot_version':3 if len(segments)>1 else saved.get('plot_version',2),
                   'source_file':csv_path.name,'source_sha256':sha256(csv_path),
                   'raw_metrics_prefix_sha256':hashlib.sha256(raw).hexdigest(),'image_sha256':sha256(path),
                   'train_points':len(train),'train_segments':[{'start_step':int(s[0]['step']),
                       'end_step':int(s[-1]['step']),'points':len(s)} for s in segments],
                   'resume_checkpoint':resume.get('checkpoint'),'validation_points':len(dev),'smoothing':None})
    return {'rows':rows,'train':train,'train_segments':segments,'dev':dev,'path':path}


def main():
    runs=sorted((ROOT/'.local/runs').glob('E09-R*/config.json'))
    if not runs:return
    entries=[]
    for path in runs:
        config=json.loads(path.read_text(encoding='utf-8'));run=path.parent
        if 'train_size' not in config:continue
        result=json.loads((run/'result.json').read_text(encoding='utf-8')) if (run/'result.json').exists() else None
        progress=json.loads((run/'progress.json').read_text(encoding='utf-8')) if (run/'progress.json').exists() else {}
        data=json.loads((run/'training-data.json').read_text(encoding='utf-8')) if (run/'training-data.json').exists() else {}
        entries.append((run.name,config,result,progress,data))
    if not entries:return
    completed,active=tool_results()
    evaluated={r['train_run'] for r in completed}
    name,config,result,progress,data=entries[-1]
    opening='正式微调还没有完成规模比较，暂时不能判断增加数据是否有用。当前先训练 1k，再以相同主要配置完成 5k 和 10k；dev loss 与工具生成结果分别保留。'
    if completed:
        latest=completed[-1];s=latest['summary']
        baseline=json.loads((ROOT/'experiments/E08/summary.json').read_text(encoding='utf-8'))['few_shot']
        opening=f"这轮用了 {latest['size']:,} 条训练轨迹，完整 dev 通过 {s['trajectory_passed']}/500；原模型加固定示例为 {baseline['trajectory_passed']}/500。调用轮相差 {s['call_turn_passed']-baseline['call_turn_passed']:+d} 条，不调用轮相差 {s['no_call_turn_passed']-baseline['no_call_turn_passed']:+d} 条。先把这两类决策分开，才看得清模型具体改变了什么；其余规模和 seed 仍需继续比较。"
    note=f"""# E09：从 1k 到 10k，模型学到的是格式还是任务？

{opening}

## 先把数据量和训练量分清楚

1k、5k、10k 表示独立轨迹数量。每条轨迹展开为多个当前回复单元，原有的完整历史和工具定义都保留，只监督当前 assistant；因此一个 epoch 是全部展开单元各参与一次训练。三个训练集嵌套，但 token 数、参数更新次数也随数据增加，这个实验比较的是扩大数据后的整体效果。

这轮使用 TRL SFTTrainer、PEFT 和 NF4 QLoRA，BF16 混合精度，packing 关闭。初始 rank=16、alpha=32、dropout=0.05，单卡每次一条，累计 8 次梯度后更新。AdamW 学习率从 warmup 升到 1e-4，再线性下降，warmup 比例 3%，梯度裁剪为 1.0。训练只跑一个 epoch，不因 loss 好看而追加轮数。

实际训练参数中，与监督和更新频率直接有关的是这些：

```python
training_args = SFTConfig(
    output_dir=str(run_directory),
    num_train_epochs=1,
    assistant_only_loss=True, packing=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4, warmup_ratio=0.03,
    bf16=True, eval_steps=100, save_steps=100,
    dataset_kwargs={{"skip_prepare_dataset": True}},
)
```

这里跳过的是重复预处理：输入和 labels 已经逐条生成并检查，Trainer 读取这些真实 token；assistant-only 没有被改成全文 loss。其余优化器、检查点和保存参数也保存在运行配置里。

训练保留数据本身的通用系统提示。工具生成评估沿用 E08 在 dev 选择的固定提示；所有模型使用同一种评测条件。训练记录中明确保存这两个提示条件。

## checkpoint 为什么不能只看训练 loss？

每 100 次更新检查全部 500 条 dev 的所有当前回复，最后不足 100 步的一段也检查。验证交叉熵按实际监督 token 数加权；不把长回复和短回复的平均 loss 简单再平均。最低完整 dev loss 的 checkpoint 被选中，平局保留更早者。最终 test 不参与选择。

loss 选择结束后，还要真正生成工具调用。即使 dev loss 下降，也可能出现少调用、参数猜测或多轮错误，所以完成训练和完成泛化评测是两个状态。

## 目前实际运行了什么？

| 运行 | 独立轨迹 | seed | 当前步数 | 最佳 checkpoint 的 dev loss | 状态 |
| --- | --- | --- | --- | --- | --- |
"""
    for name,c,r,p,d in entries:
        last=p.get('latest',{});best=r.get('selected_dev_loss') if r else p.get('best_metric')
        interruption_files=sorted((ROOT/'.local/runs'/name).glob('interruption-*.json'))
        interruption=json.loads(interruption_files[-1].read_text(encoding='utf-8')) if interruption_files else None
        resume_files=sorted((ROOT/'.local/runs'/name).glob('resume-*.json'))
        if not r or r['status']!='trained_pending_tool_eval':
            states=sorted((ROOT/'.local/runs'/name).glob('checkpoint-*/trainer_state.json'),key=lambda x:int(x.parent.name.split('-')[-1]))
            if states:best=json.loads(states[-1].read_text(encoding='utf-8'))['best_metric']
        status={'trained_pending_tool_eval':'训练完成，待工具评测','failed':'失败，保留记录','interrupted_for_memory_pressure':'显存压力，停止并保留记录'}.get(r['status'],r['status']) if r else '运行中'
        if interruption and not r and p.get('status')=='interrupted':
            status=f"Windows 重启中断；待从 {interruption['checkpoint']['directory']} 恢复"
        elif interruption and not r and resume_files and p.get('status')=='running':
            status=f"从 {interruption['checkpoint']['directory']} 恢复训练中"
        elif interruption and r and r['status']=='trained_pending_tool_eval':
            status=f"从 {interruption['checkpoint']['directory']} 恢复后训练完成，待工具评测"
        if name in evaluated:status='训练与完整 dev 工具评测结束'
        if name in evaluated and interruption:status='断点恢复后训练与完整 dev 工具评测结束'
        note+=f"| {name} | {c['train_size']} | {c['seed']} | {r.get('steps',last.get('step',0)) if r else last.get('step',0)} | {f'{best:.5f}' if best is not None else '尚未保存'} | {status} |\n"
    if data:
        note+=f"\n当前运行展开 {data['train']['assistant_units']:,} 个回复单元，共 {data['train']['supervised_tokens']:,} 个监督 token；最长 {data['train']['max_sequence_length']} token。全部单元已检查官方渲染、非思考前缀和监督位置，没有截断调用。\n"
    if result and result['status']=='failed':
        error=result['error'].splitlines()[-1]
        if 'WinError 5' in error:error='PermissionError: [WinError 5]，Windows 拒绝替换进度记录'
        note+=f"\n这次运行失败，原配置和错误已保存。直接原因是：{error}。调整条件后需要新的运行号，不能把失败覆盖掉。\n"
    for name,c,r,p,d in entries:
        interruption_files=sorted((ROOT/'.local/runs'/name).glob('interruption-*.json'))
        if not interruption_files:continue
        event=json.loads(interruption_files[-1].read_text(encoding='utf-8'))
        checkpoint=event['checkpoint']['directory'];step=event['training']['latest_step']
        note+=f"\n## 10k seed 42 这一轮为什么停在半路？\n\n{name} 跑到第 {step} 次参数更新时，训练日志没有 Python traceback 或 CUDA OOM 记录。Windows 在同一秒留下了重启事件：由 StartMenuExperienceHost 发出重启请求，训练子进程随后以 `0x40010004` 结束。系统事件与训练日志的时间能对上。\n\n最后一次完整 dev 检查在第 400 步，loss 为 {event['training']['best_dev_loss']:.5f}。保存的 `{checkpoint}` 有 {event['checkpoint']['verified_files']} 个文件，清单里的 SHA-256 都已复核；第 401—{step} 步没有 checkpoint，不能算进已保存训练量。恢复时沿用这份 10k、seed 42 配置和同一运行号：\n\n```powershell\npython scripts/sft.py --config configs/sft.json --experiment E09 --size 10000 --seed 42 --resume-run {name}\n```\n\n这次只是系统重启后的断点续训，不是换参数重跑。重启原因、最后日志、队列报错和 checkpoint 校验结果均保留在本地运行档案。\n"
    folder=ROOT/'experiments/E09';folder.mkdir(parents=True,exist_ok=True);(folder/'notes.md').write_text(note,encoding='utf-8')
    curves={entry[0]:draw_curve(entry[0]) for entry in entries}
    curve_name=entries[-1][0]
    current_curve=curves[curve_name]
    if not current_curve or not current_curve['train']:
        candidates=[name for name,curve in curves.items() if curve and curve['train']]
        if candidates:curve_name=candidates[-1];current_curve=curves[curve_name]
    if current_curve and current_curve['train']:
        note+=f"\n![图 E09-1：{curve_name}，逐步训练 loss 与完整 dev 测量点。](figures/{current_curve['path'].name})\n"
        note+='\n训练点是 TRL 每次参数更新的 loss，验证点来自完整 dev 的监督 token 加权交叉熵。它们的聚合窗口不同，先观察各自趋势，再结合工具生成结果判断；不能把一两个低点当成能力改善。\n'
        if len(current_curve['train_segments'])>1:
            first_end=int(current_curve['train_segments'][0][-1]['step'])
            resume=current_curve['path'].with_suffix('.source.json')
            source=json.loads(resume.read_text(encoding='utf-8'))
            checkpoint=source.get('resume_checkpoint','未知 checkpoint')
            note+=f"\nE09-R13 的训练日志跨过一次断点：首次运行记录到第 {first_end} 步，但最后可恢复权重只到 `{checkpoint}`；恢复后的 loss 从下一段单独绘制。图里保留了重启前未保存的尾段记录，也保留恢复后的实际曲线，没有把两次进程的 loss 连成一条连续训练轨迹。\n"
    first=next((entry for entry in entries if entry[0]=='E09-R05'),None)
    if first and first[2] and first[2]['status']=='trained_pending_tool_eval':
        r=first[2];initial=curves[first[0]]['dev'][0]['eval_loss']
        note+=f"\n已完成的 1k 主运行共有 {r['train']['assistant_units']:,} 个回复单元、{r['train']['supervised_tokens']:,} 个监督 token，更新 {r['steps']} 次。完整 dev loss 从 {initial:.5f} 降到 {r['selected_dev_loss']:.5f}，选择了 {r['selected_checkpoint']}。下面的生成结果才用来检查，这种对标注文本的拟合是否变成了更可靠的决策。\n"
    fivek=next((entry for entry in reversed(entries) if entry[1]['train_size']==5000 and entry[2] and entry[2]['status']=='trained_pending_tool_eval'),None)
    if fivek:
        r=fivek[2]
        outcome='选中适配器的同一500条dev工具生成评测也已完成，下面结合实际回复判断规模变化。' if fivek[0] in evaluated else '选中适配器随后单独做同一500条dev的工具生成评测，结果完整后再讨论规模变化。'
        note+=f"\n5k 的 {fivek[0]} 也完成了一个epoch，{r['train']['assistant_units']:,} 个回复单元共有 {r['train']['supervised_tokens']:,} 个监督 token，实际更新 {r['steps']:,} 次。最低完整 dev loss 为 {r['selected_dev_loss']:.5f}，选择 {r['selected_checkpoint']}；参数训练耗时 {r['train_metrics']['train_runtime']/3600:.2f} 小时，包含中途完整验证与保存。{outcome}\n"
    note+='\n这一步要观察两条线：训练 loss 是否继续下降，完整 dev loss 是否也下降。只有两条线和工具结果放在一起，才有依据区分记住训练内容与适应新任务。\n'
    if active:
        note+='\n## 生成评测进行到哪一步？\n\n'
        for train,run,turns in active:
            note+=f'{train} 的独立生成评测为 {run}，已落盘 {turns}/928 个决策轮。还没有覆盖完整 500 条轨迹，暂时不把当前通过数写成最终成绩。\n\n'
    if completed:
        note+='\n## loss 降下来了，工具调用也改善了吗？\n\n'
        latest=completed[-1];s=latest['summary']
        note+=f"最近完成的 {latest['train_run']} 通过 {s['trajectory_passed']}/500；调用轮为 {s['call_turn_passed']}/568，不调用轮为 {s['no_call_turn_passed']}/360。是否比原模型好，要在相同条件下逐项比较。\n\n"
        note+='这里只列已经完成全部 dev 的运行。原模型和适配器使用 E08 选择的同一提示、贪心解码、相同标注历史和生成预算；每一项都可以核对逐条回复，失败仍留在分母中。\n\n'
        note+='| 训练运行 | 数据量 / seed | 整条轨迹通过 | 调用轮通过 | 不调用轮通过 | 截断 |\n| --- | --- | --- | --- | --- | --- |\n'
        frozen=json.loads((ROOT/'configs/prompt-frozen.json').read_text(encoding='utf-8'))
        baseline=json.loads((ROOT/'experiments/E08/summary.json').read_text(encoding='utf-8'))[frozen['selected_prompt']]
        note+=f"| 原模型，固定提示 | 无微调 | {baseline['trajectory_passed']}/500 | {baseline['call_turn_passed']}/568 | {baseline['no_call_turn_passed']}/360 | {baseline['truncated']} |\n"
        for row in completed:
            s=row['summary']
            note+=f"| {row['train_run']} | {row['size']:,} / {row['seed']} | {s['trajectory_passed']}/500 | {s['call_turn_passed']}/568 | {s['no_call_turn_passed']}/360 | {s['truncated']} |\n"
        plot=draw_comparison(completed)
        note+=f'\n![图 E09-3：相同完整 dev 的真实生成结果，各训练 seed 单独显示。](figures/{plot.name})\n'
        latest=completed[-1];s=latest['summary'];delta=s['trajectory_passed']-baseline['trajectory_passed']
        note+=f"\n最近完成的 {latest['train_run']} 通过 {s['trajectory_passed']}/500，与原模型相差 {delta:+d} 条，即 {100*delta/500:+.1f} 个百分点。调用轮相差 {s['call_turn_passed']-baseline['call_turn_passed']:+d} 条，不调用轮相差 {s['no_call_turn_passed']-baseline['no_call_turn_passed']:+d} 条。先分别看这两类决策，才能判断模型是否只是更愿意调工具。\n"
        note+='\n这些仍是 dev 成绩，用于探索规模与选配置；最终 test 还没有用于本阶段选择。数据更多也意味着监督 token 和更新次数更多，规模差异不能直接归因于某一种训练机制。\n'
        analysis_path=folder/'E09-R06-comparison.json'
        if analysis_path.exists():
            a=json.loads(analysis_path.read_text(encoding='utf-8'));low,high=a['paired_bootstrap']['percentile_interval']
            note+=f"\n## 只少通过五条，是原来的题都差了一点吗？\n\n不是。1k 这轮和提示词基线有 {a['both_passed']} 条共同通过、{a['both_failed']} 条共同失败；另外 {a['student_only']} 条原来错的轨迹被修好，{a['baseline_only']} 条原来对的轨迹退步了。最后相差五条，是这两边变化抵消后的结果。逐轮看也有 {a['decision_improvements']} 次改对和 {a['decision_regressions']} 次改错，不能说模型几乎没有变化。\n"
            note+=f"\n把同一批 500 条轨迹成对重抽 10,000 次，差值的 95% 区间为 {low:.1f} 到 {high:+.1f} 个百分点，覆盖零。这里先把结论收住：本轮没有证明整体改善，也没有充分依据把这一点差距推广为稳定退步。区间只反映这份 dev 的轨迹差异，不包含重新训练不同 seed 的波动。\n"
            note+='\n## 是不是每个缺参数的问题都变差了？\n\n也不是。上一节的待办例子 glaive-60626，这轮已经改成先询问任务和优先级；原模型猜参数的问题得到了修正。但 glaive-19711 只说想听音乐，没有给出类型，原模型会问想听什么，微调后却直接调用了下面的工具：\n\n```json\n{"name":"play_music","arguments":{"genre":"pop"}}\n```\n\n这条 JSON 和参数类型都有效，问题在于 pop 是模型自行补出的。两个例子都来自本轮实际回复。接下来继续原定 5k、10k 比较，观察这种多调用、少询问的倾向是否改变；数据质量和学习率的对照再单独排查，当前不为了让成绩变好而改评分。\n'
    fivek_analysis=folder/'E09-R10-comparison.json'
    if fivek_analysis.exists():
        a=json.loads(fivek_analysis.read_text(encoding='utf-8'));low,high=a['paired_bootstrap']['percentile_interval']
        note+=f"\n## 增加到5k后，哪些轨迹变了？64条修好，13条退步\n\n和固定提示的原模型相比，5k 这轮有 {a['both_passed']} 条共同通过、{a['both_failed']} 条共同失败；{a['student_only']} 条原来错的轨迹被修好，{a['baseline_only']} 条原来对的轨迹退步，净增加51条。按同一500条轨迹配对重抽10,000次，提升的95%区间为 {low:.1f}—{high:.1f} 个百分点。这次区间没有覆盖零，可以说这份dev上有改善；它仍不能代表最终test，也没有包含不同训练seed的波动。\n"
        note+='\n变化主要体现在不调用的决策：5k通过342/360，1k只有276/360；调用轮则从474/568变为482/568。模型更能等参数补齐，但并行任务仍只有2/10条轨迹通过。数据量、监督token和更新次数都增加了，暂时不能把原因单独归到某一种训练机制。\n'
        note+='\n待办任务 glaive-60626 很适合说明这轮的变化。用户只说想加一条待办时，5k模型会先询问内容和优先级，修正了原模型自行猜参数的问题。但在补齐“buy groceries”和“high”后的另一个决策轮，它返回了下面的调用：\n\n```json\n{"name":"make_todo","arguments":{"priority":"high","task":"Buy groceries"}}\n```\n\n标注里的任务内容是小写的“buy groceries”。这次只改了首字母，仍被既定的严格字符串评分判为失败，所以整条轨迹没有通过。这不是缺参数错误，却也不能临时放宽评分把它算对。一个决策改好了，不代表整条轨迹都修好了。这里各轮输入使用标注历史，尚未执行模型自己的连续Agent对话；接着完成原定10k和三个训练seed，再看收益是否稳定。\n'
    diagnosis=ROOT/'experiments/E09/runs/E09-R02.json'
    if diagnosis.exists():
        probe=json.loads(diagnosis.read_text(encoding='utf-8'))
        if probe.get('kind')=='compatibility_probe' or probe.get('config',{}).get('kind')=='compatibility_probe':
            note+=f"\n## 第一次反向为何没有梯度？\n\nR01 已完整算出初始 dev loss 2.18203，但第一次反向失败。R02 单独复现了边界：手动准备的 LoRA 有 {probe['trainable_before']:,} 个可训练参数，交给 TRL 的准备函数后变为 {probe['trainable_after']}，loss 也不再带梯度。原因是这版 TRL 又调用了一次 k-bit 准备，冻结了已经添加的适配器。\n\n后续把裸 NF4 模型和 LoRA 配置一起交给 Trainer，让它先准备基础权重再创建适配器；构造结束时核对可训练参数。失败和单独复现的记录都保留，正式训练使用新的运行号，不改数据量和监督方式。\n"
    stopped=ROOT/'.local/runs/E09-R03/result.json'
    if stopped.exists() and json.loads(stopped.read_text(encoding='utf-8'))['status']=='interrupted_for_memory_pressure':
        previous=[json.loads(x) for x in (stopped.parent/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        slow=[r['step_seconds'] for r in previous if r.get('step',0)>=18 and 'step_seconds' in r]
        peaks=[r['peak_allocated_mib'] for r in previous if r.get('step',0)>=18 and 'step_seconds' in r]
        note+=f"\n## 梯度恢复了，为什么每一步又变得很慢？\n\nR03 已经有有效梯度，但约第 18 次更新开始，单步耗时的中位数达到 {statistics.median(slow):.1f} 秒。停止前实际记录的整卡占用为 11,818 MiB，运行中的张量峰值却只有约 {min(peaks):,.0f}—{max(peaks):,.0f} MiB。停止本项目进程后，整卡占用回到 1,026 MiB。这轮保留已有更新，但没有到第一次 checkpoint，不计作完成训练。\n\n下一轮 R04 把张量分配量和缓存保留量分开记录，并在更新前释放未使用缓存。模型参数、梯度和优化器状态仍在；轨迹数量、seed、学习率和更新次数都没有改变。这是新的资源条件，不能把两轮耗时直接合成一次训练。\n"
        if curves.get('E09-R04'):
            current=[r for r in curves['E09-R04']['train'] if r['step']>=18]
            if current:
                note+=f"\nR04 已记录第 18—{current[-1]['step']} 次更新，这段训练单步中位数为 {statistics.median(r['step_seconds'] for r in current):.2f} 秒；本段最大缓存保留量为 {max(r['peak_reserved_mib'] for r in current)/1024:.2f} GiB。训练阶段恢复了正常步时，完整流程还需要看验证阶段。\n"
        note+='\n排查时要分开看这几个数。allocated 是仍被张量使用的显存，reserved 还包含分配器保留的缓存；只记录其中一个，容易看漏资源压力。\n\n```python\nallocated = torch.cuda.memory_allocated()\nreserved = torch.cuda.memory_reserved()\n# 释放没有张量占用的缓存，不释放模型权重\ntorch.cuda.empty_cache()\n```\n'
    stopped_validation=ROOT/'.local/runs/E09-R04/result.json'
    if stopped_validation.exists() and json.loads(stopped_validation.read_text(encoding='utf-8'))['status']=='interrupted_for_memory_pressure':
        note+='\n## 验证还在慢，训练阶段的调整够了吗？\n\n还不够。R04 第 100 步完整 dev loss 为 0.18030，checkpoint 也已保存；第 200 步验证又累积缓存，停止前整卡占用 11,769 MiB。第 200 步没有完整验证结果，因此图中没有补上这个点，也没有把这轮写成训练完成。\n\nR05 重新从同一官方学生开始训练，同时在每个验证回复结束后释放空闲缓存，单独记录释放前后占用。计算仍覆盖全部 500 条轨迹的 1,484 个回复，按相同监督 token 加权。这次改变针对资源管理，样本量、目标标签和评分方式都保持原定条件。\n'
        if curves.get('E09-R04'):
            curve=curves['E09-R04']
            note+=f"\n![图 E09-2：R04 保留的 {len(curve['train'])} 次训练更新；仅显示已完成的 {len(curve['dev'])} 次完整 dev 测量，第 200 步验证未完成。](figures/{curve['path'].name})\n"
    io_failure=ROOT/'.local/runs/E09-R07/result.json'
    io_probe=ROOT/'.local/runs/E09-R08/result.json'
    if io_failure.exists() and 'WinError 5' in json.loads(io_failure.read_text(encoding='utf-8')).get('error',''):
        note+='\n## 这次没有 OOM，为什么 5k 还是停了？\n\nR07 已训练 100 次，完整 dev loss 为 0.18383；模型计算和验证都已完成，随后替换进度记录时出现 WinError 5。失败发生在保存 checkpoint 之前，因此这一轮没有可恢复的模型断点，不能把完成验证当成完成训练。已有 100 次更新、两次完整 dev 测量和原始报错全部保留。\n'
        if io_probe.exists():
            probe=json.loads(io_probe.read_text(encoding='utf-8'))
            note+='\nR08 用真实 Windows 文件句柄复现了边界：文件还被读取时，旧写法替换失败；关闭读句柄后，同一替换成功。原失败时具体是哪一个进程持有句柄还没有确认，不能直接归因到某个程序。\n'
            if probe.get('replacement_verification')=='completed':
                note+=f"\n新的写法让各次写入使用独立临时文件，在短暂占用时有限重试。实际核验中，读句柄在 0.20 秒后释放，替换在 {probe['new_writer_seconds']:.2f} 秒后成功，读取到的内容正确；持续无法写入仍会报错。后续按同样 5k 数据、seed 和训练参数重新开始，改动针对记录写入，不减少训练量。\n"
    (folder/'notes.md').write_text(note,encoding='utf-8')
    update_summary(completed)


if __name__=='__main__':main()
