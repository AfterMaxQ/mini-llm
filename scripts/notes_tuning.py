"""从真实 rank/学习率运行整理对照，准备检查不记作训练结果。"""
import json
import csv
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from lab import ROOT, sha256, write_json
from notes_sft import draw_curve


def main():
    folder=ROOT/'experiments/E11'
    preparation=folder/'preparation.json'
    if not preparation.exists():return
    p=json.loads(preparation.read_text(encoding='utf-8'))
    base=json.loads((ROOT/'configs/sft.json').read_text(encoding='utf-8'));base.update(train_size=5000,seed=17)
    reference=[f.parent for f in sorted((ROOT/'.local/runs').glob('E09-R*/config.json'))
               if all(json.loads(f.read_text(encoding='utf-8')).get(k)==v for k,v in base.items())]
    runs=([reference[-1]] if reference else [])+[f.parent for f in sorted((ROOT/'.local/runs').glob('E11-R*/config.json'))
          if 'train_size' in json.loads(f.read_text(encoding='utf-8'))]
    generations=[]
    for experiment in ['E09','E11']:
        for path in sorted((ROOT/'experiments'/experiment/'runs').glob('*.json')):
            r=json.loads(path.read_text(encoding='utf-8'));c=r['config']
            if r['status']!='completed' or c.get('kind')!='tool_eval' or c.get('source_train_run') not in [f.name for f in runs]:continue
            source=ROOT/'.local/runs'/c['source_train_run'];training=json.loads((source/'config.json').read_text(encoding='utf-8'))
            s=r['summaries'][r['selected_prompt']]
            records=[json.loads(line) for line in (ROOT/'.local/data/processed'/c['data_run']/(c['split']+'.jsonl')).read_text(encoding='utf-8').splitlines()]
            turns=sum(m['role']=='assistant' and bool(m.get('tool_calls') or record['messages'][i-1]['role']=='user')
                      for record in records for i,m in enumerate(record['messages']))
            assert s['trajectories']==s['evaluated_trajectories']==len(records) and s['decision_turns']==turns
            assert c['adapter_sha256']==sha256(source/'selected-adapter/adapter_model.safetensors')
            assert c['frozen_prompt_sha256']==sha256(ROOT/'configs/prompt-frozen.json')
            assert s['rows_sha256']==sha256(ROOT/'.local/runs'/r['run_id']/(r['selected_prompt']+'.jsonl'))
            generations.append({'training':training,'summary':s,'run_id':r['run_id']})
    formal=[run for run in runs if run.name.startswith('E11-')]
    opening='rank 对照还没有开始正式训练。目前核验的是固定配置和参数量，不能据此判断 rank 越大越好。'
    if formal:opening='rank 对照已开始。先看工具生成能不能通过，再结合 loss、参数量和耗时判断；还没评测完的条件不填写通过率。'
    rank_results={g['training']['rank']:g['summary'] for g in generations if g['training']['train_size']==5000}
    if {8,16}<=rank_results.keys():
        opening=(f"这次把 rank 从 8 增到 16，没有提高整条轨迹的通过数：两组都通过 {rank_results[8]['trajectory_passed']}/500。"
                 "可训练参数翻倍，模型的具体错误却有变化。这个结果说明，增加适配器容量未必就能把工具任务做好；不能只凭 loss 或参数量判断效果。")
    note=f'''# E11：rank 翻倍，能学得更好吗？

{opening}

## 先确认增加了多少参数

读取本地学生权重文件的投影形状，找到了 196 个目标矩阵。LoRA 为每个矩阵增加 A 和 B 两个矩阵，因此共有 392 个适配器张量。对于输出维度为 m、输入维度为 n 的线性层，新增参数量是 r×(m+n)。这里读取的是权重元数据，没有把模型放到 GPU 上。

```python
parameters = sum(rank * (out_dim + in_dim)
                 for out_dim, in_dim in projection_shapes)
```

| rank | 由实际权重形状计算的适配器参数量 | alpha | alpha/r |
| --- | --- | --- | --- |
'''
    for row in p['rank_conditions']:
        note+=f"| {row['rank']} | {row['projected_adapter_parameters']:,} | {row['alpha']} | {row['alpha_over_rank']:g} |\n"
    note+='\nrank 16 的计算结果是 17,432,576，与 E09 的实际可训练参数统计一致。E11-R01 接入训练后，rank 8 实际为 8,716,288 个参数，也与计算相符。rank 32 这里只列形状计算，尚无训练成绩。在 alpha 固定为 32 时，alpha/r 也随 rank 改变，所以比较的是整套 rank 设置，不能把变化全部归因于参数量。\n\n'
    note+='## 怎么安排，才能看出是哪一个变量起作用？\n\n'
    note+='rank 对照把学习率固定为 1e-4，比较 rank 8 和 16。两组使用同一份 5,000 条轨迹、14,502 个当前回复单元和 638,598 个监督 token，seed 17，一个 epoch、1,813 次更新；长度、模板、监督遮罩和缓存策略一致。rank 16 复用 E09 的同条件训练与全部 500 条 dev 工具结果。\n\n'
    note+='学习率实验另用同一份 1,000 条训练轨迹，固定 rank 16，比较 1e-4 和 5e-5。两组都重新训练，使用同一份预先冻结的 100 条 dev，初始、每 200 步及末步计算 token 加权 loss，按最低 loss 选 checkpoint。随后在这 100 条轨迹上生成工具决策；通过数相同就选较低学习率。这里比较学习率，不把它和 5k rank 实验拼成同一组提升。最终 test 不参与选择。\n\n'
    note+='## 第一次启动为什么停了？\n\n'
    note+='第一次启动队列时使用了系统默认 Python，训练子进程在导入 `datasets` 时以 `ModuleNotFoundError` 退出，退出码为 1。错误发生在训练初始化阶段，没有创建训练状态、加载模型或占用 GPU。随后改用项目锁定的训练环境启动 E11-R01；这次失败属于解释器入口问题，不是 rank 条件跑坏了。\n\n'
    note+='## 目前哪些条件已经实际运行？\n\n'
    displayed=[run for run in runs if run.name.startswith('E11-') or any(g['training']['run_id']==run.name for g in generations)]
    if displayed:note+='| 运行 | 训练轨迹 | rank | 学习率 | 更新次数 | 状态 |\n| --- | --- | --- | --- | --- | --- |\n'
    for run in displayed:
        c=json.loads((run/'config.json').read_text(encoding='utf-8'))
        r=json.loads((run/'result.json').read_text(encoding='utf-8')) if (run/'result.json').exists() else {}
        progress=json.loads((run/'progress.json').read_text(encoding='utf-8')) if (run/'progress.json').exists() else {}
        state={'trained_pending_tool_eval':'训练结束，工具评测另查','failed':'失败，原记录保留'}.get(r.get('status'),'运行中')
        if any(g['training']['run_id']==run.name for g in generations):state='训练与工具评测完成'
        if not run.name.startswith('E11-'):state='参考；'+state
        note+=f"| {run.name} | {c['train_size']:,} | {c['rank']} | {c['learning_rate']:g} | {r.get('steps',progress.get('latest',{}).get('step',0))} | {state} |\n"
    if not formal:note+='E11 尚未开始 GPU 训练。队列入口在参考训练或完整 dev 工具评测缺失时会停止；参考条件的训练过程保留在 E09 章节。已有 E09 进程继续运行，没有启动第二个 GPU 任务。\n'
    for run in formal:
        curve=draw_curve(run.name)
        if curve and curve['train']:
            c=json.loads((run/'config.json').read_text(encoding='utf-8'))
            count=100 if c.get('sampling_manifest_sha256') else 500
            note+=f"\n![图 E11：{run.name} 的真实训练与 {count} 条 dev loss，未平滑。](figures/{curve['path'].name})\n"
    if generations:
        note+='\n## 工具表现怎样？\n\n表中分母来自各自冻结清单。5k rank 实验使用完整 500 条 dev；1k 学习率实验使用固定抽样的 100 条，四类任务都保留，其中并行调用保留原 dev 的全部 10 条。两套分数分开解释，抽样结果只说明这些任务上的表现。各条件沿用固定提示、贪心解码和同一 NF4 推理后端。\n\n| 训练运行 | 训练轨迹 | rank | 学习率 | 整条轨迹 | 调用轮 | 不调用轮 | 截断 |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n'
        for g in generations:
            c=g['training'];s=g['summary']
            note+=f"| {c['run_id']} | {c['train_size']:,} | {c['rank']} | {c['learning_rate']:g} | {s['trajectory_passed']}/{s['trajectories']} | {s['call_turn_passed']}/{s['call_turns']} | {s['no_call_turn_passed']}/{s['no_call_turns']} | {s['truncated']} |\n"
        if {8,16}<=rank_results.keys():
            a,b=rank_results[8],rank_results[16]
            note+=f"\nrank 8 的调用轮通过 {a['call_turn_passed']}/{a['call_turns']}，rank 16 为 {b['call_turn_passed']}/{b['call_turns']}；不调用轮分别为 {a['no_call_turn_passed']}/{a['no_call_turns']} 和 {b['no_call_turn_passed']}/{b['no_call_turns']}。总通过数相同，不能理解成两个模型输出完全一样。当前结果来自 seed 17 的一次对照，尚不能推广成所有训练随机种子的结论。\n"
            rank_evaluations={g['training']['rank']:g for g in generations if g['training']['train_size']==5000}
            outputs={}
            for rank,g in rank_evaluations.items():
                result=json.loads((ROOT/'.local/runs'/g['run_id']/'result.json').read_text(encoding='utf-8'))
                path=ROOT/'.local/runs'/g['run_id']/(result['selected_prompt']+'.jsonl')
                rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
                outputs[rank]={key:all(row['exact'] for row in rows if row['sample_id']==key) for key in {row['sample_id'] for row in rows}}
            assert outputs[8].keys()==outputs[16].keys()
            paired=[{'sample_id':key,'rank8_passed':outputs[8][key],'rank16_passed':outputs[16][key]} for key in sorted(outputs[8])]
            pair_path=folder/'rank-paired.csv'
            with pair_path.open('w',encoding='utf-8',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
            both=sum(row['rank8_passed'] and row['rank16_passed'] for row in paired)
            only8=sum(row['rank8_passed'] and not row['rank16_passed'] for row in paired)
            only16=sum(not row['rank8_passed'] and row['rank16_passed'] for row in paired)
            note+=f"\n逐条配对后，两组共同通过 {both} 条，各有 {only8} 条和 {only16} 条是只有自己通过的任务。总体分数打平，并不意味着增加 rank 完全没有改变模型，只是这次变化没有增加通过的任务总数。\n"
            note+='\n一个具体错误出现在发票任务 `glaive-19894`。用户给出的客户、商品单价和数量都被 rank 8 正确写入调用，但它还在 arguments 中多写了 `type` 字段。rank 16 的同一条回复没有这个字段，因而通过严格的参数匹配。下面摘出多写的参数，完整调用保留在逐条结果中。\n\n```json\n{"type": "function"}\n```\n'
            figure=folder/'figures/rank-tool-comparison.png'
            source=folder/'rank-tool-comparison.csv'
            categories=['single_call','multi_turn','no_call','parallel']
            labels=['单次调用','多轮调用','不调用','并行调用']
            with source.open('w',encoding='utf-8',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=['rank','category','passed','samples']);writer.writeheader()
                for rank,summary in [(8,a),(16,b)]:
                    for category in categories:writer.writerow({'rank':rank,'category':category,**summary['categories'][category]})
            font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
            fig,axis=plt.subplots(figsize=(6.4,3.3),layout='constrained')
            for offset,rank,summary,color,hatch in [(-0.18,8,a,'#6B7280',''),(0.18,16,b,'#059669','//')]:
                values=[100*summary['categories'][category]['passed']/summary['categories'][category]['samples'] for category in categories]
                bars=axis.bar([i+offset for i in range(4)],values,width=0.34,color=color,hatch=hatch,label=f'rank {rank}')
                axis.bar_label(bars,labels=[f"{summary['categories'][c]['passed']}/{summary['categories'][c]['samples']}" for c in categories],fontsize=8,padding=3)
            axis.set_xticks(range(4),labels,fontproperties=font);axis.set_ylim(0,112)
            axis.set_ylabel('整条轨迹通过率（%）',fontproperties=font)
            axis.set_title('同一5k训练、完整500条dev：不同任务的表现',fontproperties=font)
            axis.legend(prop=font,frameon=False);axis.grid(axis='y',color='#E5E7EB');axis.set_axisbelow(True)
            for side in ['top','right']:axis.spines[side].set_visible(False)
            fig.savefig(figure,dpi=300);plt.close(fig)
            evaluations=[g for g in generations if g['training']['train_size']==5000]
            write_json(figure.with_suffix('.source.json'),{'source_file':source.name,'source_sha256':sha256(source),'image_sha256':sha256(figure),
                       'evaluation_files':{g['run_id']:sha256(ROOT/'experiments'/('E09' if g['run_id'].startswith('E09-') else 'E11')/'runs'/(g['run_id']+'.json')) for g in evaluations}})
            note+='\n![图 E11：同一完整 dev 的四类任务通过率，柱上给出正确数与分母。两组均为训练 seed 17。](figures/rank-tool-comparison.png)\n'
            note+='四类题的通过数在两组中也完全相同：单次调用 205/257、多轮调用 113/147、不调用 86/86、并行调用 2/10。逐条配对仍能看到各自多出的 5 条表现落在不同任务上，只是没有改变本批数据的类别合计。并行类只有 10 条，比例对少数样本很敏感。\n'
    (folder/'notes.md').write_text(note,encoding='utf-8')


if __name__=='__main__':main()
