"""从真实 rank/学习率运行整理对照，准备检查不记作训练结果。"""
import json
from lab import ROOT, sha256
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
            assert s['trajectories']==s['evaluated_trajectories']==500 and s['decision_turns']==928
            assert c['adapter_sha256']==sha256(source/'selected-adapter/adapter_model.safetensors')
            assert c['frozen_prompt_sha256']==sha256(ROOT/'configs/prompt-frozen.json')
            assert s['rows_sha256']==sha256(ROOT/'.local/runs'/r['run_id']/(r['selected_prompt']+'.jsonl'))
            generations.append({'training':training,'summary':s,'run_id':r['run_id']})
    formal=[run for run in runs if run.name.startswith('E11-')]
    opening='rank 对照还没有开始正式训练。目前核验的是固定配置和参数量，不能据此判断 rank 越大越好。'
    if formal:opening='rank 与学习率对照已经开始，下面保留全部实际运行。loss 和工具生成分别核对，尚未完成完整 dev 的条件不填最终成绩。'
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
    note+='\nrank 16 的计算结果是 17,432,576，与 E09 已保存的实际可训练参数统计一致。rank 8 和 32 的值仍是按形状计算的预期量，正式训练接入后还要核对实际参数统计。参数翻倍不代表效果翻倍；在 alpha 固定为 32 的条件下，增量权重的缩放也随 rank 改变。\n\n'
    note+='## 怎么安排，才能看出是哪一个变量起作用？\n\n'
    note+='先把学习率固定为 1e-4，比较 rank 8、16、32。三组使用同一份 5,000 条轨迹、14,502 个当前回复单元和 638,598 个监督 token，seed 17，训练一个 epoch、更新 1,813 次；长度、模板、监督遮罩和缓存策略保持一致。rank 16 复用 E09 同条件结果，必须完成训练和全部 500 条 dev 的工具生成后，才作为有效参考。\n\n'
    note+='根据整条轨迹通过数选择 rank，并列时选较小 rank。然后固定这个 rank，再比较 5e-5、1e-4、2e-4 三个学习率；已经完成的 1e-4 条件直接复用，并列时选较低学习率。两阶段都只用 dev，最终 test 不参与选择。调用轮、不调用轮和截断数仍单独展示，避免只看一个总分。\n\n'
    note+='## 目前哪些条件已经实际运行？\n\n'
    displayed=[run for run in runs if run.name.startswith('E11-') or any(g['training']['run_id']==run.name for g in generations)]
    if displayed:note+='| 运行 | rank | 学习率 | 更新次数 | 状态 |\n| --- | --- | --- | --- | --- |\n'
    for run in displayed:
        c=json.loads((run/'config.json').read_text(encoding='utf-8'))
        r=json.loads((run/'result.json').read_text(encoding='utf-8')) if (run/'result.json').exists() else {}
        progress=json.loads((run/'progress.json').read_text(encoding='utf-8')) if (run/'progress.json').exists() else {}
        state={'trained_pending_tool_eval':'训练结束，工具评测另查','failed':'失败，原记录保留'}.get(r.get('status'),'运行中')
        if not run.name.startswith('E11-'):state='E09 参考条件；'+state
        note+=f"| {run.name} | {c['rank']} | {c['learning_rate']:g} | {r.get('steps',progress.get('latest',{}).get('step',0))} | {state} |\n"
    if not formal:note+='E11 尚未开始 GPU 训练。队列入口在参考训练或完整 dev 工具评测缺失时会停止；参考条件的训练过程保留在 E09 章节。已有 E09 进程继续运行，没有启动第二个 GPU 任务。\n'
    for run in formal:
        curve=draw_curve(run.name)
        if curve and curve['train']:
            note+=f"\n![图 E11：{run.name} 的真实训练与完整 dev loss，未平滑。](figures/{curve['path'].name})\n"
    if generations:
        note+='\n## 完整 dev 的工具表现怎样？\n\n这里只列实际完成全部 500 条轨迹、928 个决策轮的结果。各条件沿用固定提示、贪心解码和同一 NF4 推理后端。\n\n| 训练运行 | rank | 学习率 | 整条轨迹 | 调用轮 | 不调用轮 | 截断 |\n| --- | --- | --- | --- | --- | --- | --- |\n'
        for g in generations:
            c=g['training'];s=g['summary']
            note+=f"| {c['run_id']} | {c['rank']} | {c['learning_rate']:g} | {s['trajectory_passed']}/500 | {s['call_turn_passed']}/568 | {s['no_call_turn_passed']}/360 | {s['truncated']} |\n"
    note+='\n选择配置前还要看失败条件、实际参数 dtype、显存和耗时。准备检查只确认入口与计算条件，表现比较等正式结果到齐后再归纳。\n'
    (folder/'notes.md').write_text(note,encoding='utf-8')


if __name__=='__main__':main()
