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
    labels=['原模型\n少样本提示']+[f"{r['size']//1000}k / seed {r['seed']}\n{r['train_run']}" for r in completed]
    rates=[100*r['passed']/500 for r in rows]
    bars=axis.bar(range(len(rows)),rates,color=['#6B7280']+['#059669']*len(completed))
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


def draw_curve(run_id):
    metrics=ROOT/'.local/runs'/run_id/'metrics.jsonl'
    if not metrics.exists():return None
    raw=metrics.read_bytes();rows=[json.loads(x) for x in raw.decode('utf-8').splitlines()]
    folder=ROOT/'experiments/E09';csv_path=folder/(run_id+'-metrics.csv')
    fields=['time','step','loss','eval_loss','grad_norm','learning_rate','epoch','num_tokens','supervised_tokens','step_seconds','peak_allocated_mib','peak_reserved_mib','allocated_mib','reserved_mib','device_free_mib','reserved_before_step_mib','reserved_after_release_mib']
    with csv_path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    train=[r for r in rows if 'loss' in r];dev=[r for r in rows if 'eval_loss' in r]
    if not train:return {'rows':rows,'train':train,'dev':dev}
    figures=folder/'figures';figures.mkdir(exist_ok=True)
    path=figures/(run_id+'-loss.png');source=path.with_suffix('.source.json')
    saved=json.loads(source.read_text(encoding='utf-8')) if source.exists() else {}
    if saved.get('source_sha256')!=sha256(csv_path) or saved.get('plot_version')!=2:
        font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
        fig,axis=plt.subplots(figsize=(6.4,3.5),layout='constrained')
        axis.plot([r['step'] for r in train],[r['loss'] for r in train],color='#059669',linewidth=1.0,label='训练：每次更新')
        if dev:axis.plot([r['step'] for r in dev],[r['eval_loss'] for r in dev],color='#059669',linestyle='None',marker='o',label='完整 dev：实测点，token 加权')
        axis.set_xlabel('optimizer step',fontproperties=font);axis.set_ylabel('assistant-only 交叉熵',fontproperties=font)
        axis.set_title(run_id+'：真实训练与验证 loss，未平滑',fontproperties=font)
        axis.legend(prop=font,frameon=False);axis.yaxis.grid(True,color='#E5E7EB');axis.set_axisbelow(True)
        for side in ['top','right']:axis.spines[side].set_visible(False)
        fig.savefig(path,dpi=300);plt.close(fig)
        write_json(source,{'run_id':run_id,'plot_version':2,'source_file':csv_path.name,'source_sha256':sha256(csv_path),
                   'raw_metrics_prefix_sha256':hashlib.sha256(raw).hexdigest(),'image_sha256':sha256(path),
                   'train_points':len(train),'validation_points':len(dev),'smoothing':None})
    return {'rows':rows,'train':train,'dev':dev,'path':path}


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
        opening=f"{latest['train_run']} 的完整 dev 已生成并检查，整条轨迹通过 {s['trajectory_passed']}/500。下面同时看调用轮、不调用轮和 loss；尚未完成的规模或 seed 继续保留为待比较，不能用单轮结果代替全部结论。"
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
        if not r or r['status']!='trained_pending_tool_eval':
            states=sorted((ROOT/'.local/runs'/name).glob('checkpoint-*/trainer_state.json'),key=lambda x:int(x.parent.name.split('-')[-1]))
            if states:best=json.loads(states[-1].read_text(encoding='utf-8'))['best_metric']
        status={'trained_pending_tool_eval':'训练完成，待工具评测','failed':'失败，保留记录','interrupted_for_memory_pressure':'显存压力，停止并保留记录'}.get(r['status'],r['status']) if r else '运行中'
        if name in evaluated:status='训练与完整 dev 工具评测结束'
        note+=f"| {name} | {c['train_size']} | {c['seed']} | {r.get('steps',last.get('step',0)) if r else last.get('step',0)} | {f'{best:.5f}' if best is not None else '尚未保存'} | {status} |\n"
    if data:
        note+=f"\n当前运行展开 {data['train']['assistant_units']:,} 个回复单元，共 {data['train']['supervised_tokens']:,} 个监督 token；最长 {data['train']['max_sequence_length']} token。全部单元已检查官方渲染、非思考前缀和监督位置，没有截断调用。\n"
    if result and result['status']=='failed':
        note+=f"\n这次运行失败，原配置和错误已保存。直接原因是：{result['error'].splitlines()[-1]}。调整条件后需要新的运行号，不能把失败覆盖掉。\n"
    folder=ROOT/'experiments/E09';folder.mkdir(parents=True,exist_ok=True);(folder/'notes.md').write_text(note,encoding='utf-8')
    curves={entry[0]:draw_curve(entry[0]) for entry in entries}
    current_curve=curves[entries[-1][0]]
    if current_curve and current_curve['train']:
        note+=f"\n![图 E09-1：{entries[-1][0]}，训练 {len(current_curve['train'])} 个点、完整 dev {len(current_curve['dev'])} 个点；保留原始波动。](figures/{current_curve['path'].name})\n"
        note+='\n训练点是 TRL 每次参数更新的 loss，验证点来自完整 dev 的监督 token 加权交叉熵。它们的聚合窗口不同，先观察各自趋势，再结合工具生成结果判断；不能把一两个低点当成能力改善。\n'
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
    (folder/'notes.md').write_text(note,encoding='utf-8')


if __name__=='__main__':main()
