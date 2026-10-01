"""只从实际训练与完整生成结果整理 BF16/NF4 对照。"""
import json
import statistics

from lab import ROOT, sha256
from notes_sft import draw_curve, tool_results


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    runs=[p.parent for p in sorted((ROOT/'.local/runs').glob('E10-R*/config.json'))
          if 'train_size' in read(p)]
    if not runs:return
    reference=[r for r in tool_results()[0] if r['size']==5000 and r['seed']==17]
    assert reference,'先完成 E09 同条件 5k 训练及完整 dev 工具评测'
    reference=reference[-1]
    base=ROOT/'.local/runs'/reference['train_run'];base_config=read(base/'config.json')
    keys=set(read(ROOT/'configs/sft.json'))-{'precision','load_in_4bit'}
    folder=ROOT/'experiments/E10';folder.mkdir(parents=True,exist_ok=True)
    entries=[]
    for run in [base,*runs]:
        config=read(run/'config.json')
        assert all(config[k]==base_config[k] for k in keys),'精度对照的训练条件不匹配'
        assert config['evaluation_prompt_sha256']==sha256(ROOT/'configs/prompt-frozen.json')
        result=read(run/'result.json') if (run/'result.json').exists() else {}
        curve=draw_curve(run.name)
        metrics=curve['train'] if curve else []
        evaluations=[]
        for path in sorted((ROOT/'.local/runs').glob('*/config.json')):
            cfg=read(path);out=path.parent/'result.json'
            if cfg.get('kind')!='tool_eval' or cfg.get('source_train_run')!=run.name or not out.exists():continue
            generated=read(out)
            if generated['status']!='completed':continue
            summary=generated['summaries'][generated['selected_prompt']]
            assert summary['trajectories']==summary['evaluated_trajectories']==500 and summary['decision_turns']==928
            assert cfg['frozen_prompt_sha256']==config['evaluation_prompt_sha256']
            assert cfg['adapter_sha256']==sha256(run/'selected-adapter/adapter_model.safetensors')
            assert summary['rows_sha256']==sha256(path.parent/(generated['selected_prompt']+'.jsonl'))
            evaluations.append((path.parent.name,summary))
        trainable=read(run/'trainable-parameters.json') if (run/'trainable-parameters.json').exists() else {}
        entries.append({'run':run,'config':config,'result':result,'curve':curve,'metrics':metrics,
                        'evaluation':evaluations[-1] if evaluations else None,'trainable':trainable})
    latest=entries[-1];finished=latest['evaluation'];result=latest['result']
    if finished:
        a=reference['summary']['trajectory_passed'];b=finished[1]['trajectory_passed']
        opening=f'两组已完成相同完整 dev：NF4 训练通过 {a}/500，BF16 训练通过 {b}/500，相差 {b-a:+d} 条。先看训练成本和两类决策，再解释精度的影响。'
    elif result.get('status')=='failed':
        opening='BF16 条件本次运行失败，还没有可比较的完整工具成绩。下面保留实际执行到的位置；同条件 NF4 的完整结果继续作为参考。'
    else:
        opening='BF16 对照正在执行，目前还不能回答质量和成本是否改善。NF4 参考已经完成，BF16 的训练、完整验证与生成评测分别记录。'
    note=f'''# E10：QLoRA 省了多少显存，效果是否变化？

{opening}

## 怎样让这个比较尽量公平？

两组使用相同的 5,000 条轨迹、14,502 个回复单元和 638,598 个监督 token。seed 为 17，rank=16、alpha=32、dropout=0.05；学习率为 1e-4，单卡 batch=1，累积 8 次梯度，训练一个 epoch，共 1,813 次更新。NF4 参考沿用已完成的 E09 运行，没有为了这个对照再训练一遍。

主变量是基础权重以 BF16 还是 NF4 存储。量化模型需要 k-bit 准备，普通模型走正常 PEFT 分支；库还可能改变适配器和非量化层的 dtype，所以它是两套实际训练方案的比较，不能把所有差异都归因于四位编码。下面记录实际可训练参数和 dtype。

```python
options = dict(torch_dtype=torch.bfloat16, device_map={{"": 0}})
if load_in_4bit:
    options["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
```

`load_in_4bit` 是存储方式开关；`options` 传给同一个官方模型加载入口。无论哪一组，模型都交给 TRL 准备后再添加 LoRA，模板、监督位置和缓存释放方式保持一致。

| 运行 / 存储方式 | 可训练参数 | 适配器 dtype | 基础层 dtype |
| --- | --- | --- | --- |
'''
    for entry in entries:
        t=entry['trainable'];name=entry['run'].name
        note+=f"| {name} / {'NF4' if entry['config'].get('load_in_4bit',True) else 'BF16'} | {t.get('parameters','尚未记录')} | {', '.join(t.get('dtypes',[])) or '尚未记录'} | {', '.join(t.get('base_dtypes',[])) or '原参考未另记此项'} |\n"
    note+='\n## 训练省下了哪些资源？\n\n显存表来自训练更新时 PyTorch 实际记录的数值；allocated 表示张量占用，reserved 还包含分配器保留的缓存。更新耗时包括该步计算与缓存释放，不含后续完整验证和 Word 整理。失败条件只列已有测量，不把缺失的最终成绩写成零。\n\n| 运行 | 已记录更新 | 更新耗时中位数（s） | 最大 allocated / reserved（MiB） | 训练状态 |\n| --- | --- | --- | --- | --- |\n'
    for entry in entries:
        rows=entry['metrics'];state=entry['result'].get('status','running')
        label={'running':'进行中','failed':'失败，保留记录','trained_pending_tool_eval':'一个 epoch 已完成'}.get(state,state)
        seconds=f"{statistics.median(r['step_seconds'] for r in rows):.2f}" if rows else '未测得'
        memory=f"{max(r['peak_allocated_mib'] for r in rows):.0f} / {max(r['peak_reserved_mib'] for r in rows):.0f}" if rows else '未测得'
        note+=f"| {entry['run'].name} | {rows[-1]['step'] if rows else 0} | {seconds} | {memory} | {label} |\n"
    note+='\n| 运行 | 训练循环耗时（s） | 监督 tokens/s | 最低完整 dev loss |\n| --- | --- | --- | --- |\n'
    for entry in entries:
        result=entry['result'];runtime=result.get('train_metrics',{}).get('train_runtime')
        if runtime:
            tokens=result['train']['supervised_tokens']
            note+=f"| {entry['run'].name} | {runtime:.0f} | {tokens/runtime:.2f} | {result['selected_dev_loss']:.5f} |\n"
        else:note+=f"| {entry['run'].name} | 未完成 | 未完成 | 未完成 |\n"
    note+='\n训练循环耗时使用 Trainer 的实际计时，包含循环内完整验证和报告回调，不含模型加载及训练前验证；监督吞吐按本轮真实目标 token 总数除以这段耗时计算，不能与纯推理 tokens/s 混用。\n'
    if latest['curve'] and latest['curve']['train']:
        c=latest['curve']
        note+=f"\n![图 E10-1：{latest['run'].name} 的真实 loss；训练 {len(c['train'])} 点，完整 dev {len(c['dev'])} 点，未平滑。](figures/{c['path'].name})\n\n训练和验证 loss 分别保留原始测量。验证覆盖全部 500 条轨迹的 1,484 个回复，按 66,664 个监督 token 加权；缺失的验证点不插值补齐。\n"
    note+='\n## 训练精度不同，生成评测也一起改吗？\n\n这一步的生成后端保持一致：两组适配器都加载到相同 NF4 学生，用固定提示、标注历史和贪心解码评测。这样比较的是训练后适配器的变化；BF16 推理本身的质量和速度另做部署实验。\n\n| 训练运行 | 完整轨迹通过 | 调用轮通过 | 不调用轮通过 | 截断 |\n| --- | --- | --- | --- | --- |\n'
    for entry in entries:
        evaluated=entry['evaluation']
        if evaluated:
            s=evaluated[1];note+=f"| {entry['run'].name} | {s['trajectory_passed']}/500 | {s['call_turn_passed']}/568 | {s['no_call_turn_passed']}/360 | {s['truncated']} |\n"
        else:note+=f"| {entry['run'].name} | 未完成 | 未完成 | 未完成 | 未完成 |\n"
    for entry in entries[1:]:
        if entry['result'].get('status')=='failed':
            error=entry['result']['error'].splitlines()[-1]
            reason='实际发生 CUDA 显存不足' if 'out of memory' in error.lower() else '实际运行报错，完整堆栈保存在证据记录中'
            note+=f"\n## {entry['run'].name} 为什么没有完成？\n\n{reason}。已有配置、指标和原始报错保留；数据量、上下文和评测分母没有缩小。不能用失败前的几次低 loss 代替完整训练或工具结果。\n"
    (folder/'notes.md').write_text(note,encoding='utf-8')
    summary=ROOT/'docs/reports/summaries/02.md'
    if summary.exists():
        content=summary.read_text(encoding='utf-8');body,evidence=content.split('## 证据索引',1)
        body=body.split('## 精度对照',1)[0].rstrip()+f'\n\n## 精度对照\n\n{opening}\n\n'
        if '| 精度对照 |' not in evidence:
            evidence=evidence.replace('| 后续训练与生成结果 |','| 精度对照 | experiments/E10/notes.md；experiments/E10/runs；configs/sft-bf16.json |\n| 后续训练与生成结果 |')
        summary.write_text(body+'## 证据索引'+evidence,encoding='utf-8')


if __name__=='__main__':main()
