"""把正在进行的基线结果写成一段可读的实验笔记。"""
import argparse
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from lab import ROOT,sha256,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",required=True);args=parser.parse_args()
    run=ROOT/".local/runs"/args.run
    config=json.loads((run/"config.json").read_text(encoding="utf-8"))
    result=json.loads((run/"result.json").read_text(encoding="utf-8")) if (run/"result.json").exists() else None
    progress=json.loads((run/"progress.json").read_text(encoding="utf-8")) if (run/"progress.json").exists() else {}
    summaries=result["summaries"] if result and "summaries" in result else progress.get("completed_prompts",{})
    if result and result["status"]=="completed":
        best=result["selected_prompt"];s=summaries[best];label='零样本提示' if best=='zero_shot' else '三条示例的少样本提示'
        opening=f"完整 dev 的 500 条轨迹中，{label}通过 {s['trajectory_passed']}/500，成为后续统一使用的评测提示。这个起点留好后，再看微调能增加多少正确调用；选择没有使用最终 test。"
    elif 'zero_shot' in summaries:
        opening=f"零样本已经检查完全部 500 条轨迹，通过 {summaries['zero_shot']['trajectory_passed']}/500。少样本仍在运行，暂时不能判断三条示例是否值得加入。"
    else:
        opening="完整 dev 基线还在运行，零样本与少样本谁更好暂时没有结论。每批结果都保存，等两种提示检查完同一批 500 条轨迹后再选择。"
    note=f"""# E08：不训练，原模型能做到哪一步？

{opening}

## 先留一个可信的起点

32 条拟合通过，只说明训练代码能把训练样本学进去。正式微调前，先让原模型回答从未用于训练的 dev。否则，后面看到的正确调用，很容易被全部归功于微调。

这轮比较两种提示。零样本沿用数据的通用系统提示；少样本增加三条来自 1k 训练集的示例，分别展示单次、并行和不调用工具。示例包含自己的工具定义，并明确说明那些工具不能拿来处理当前任务。两种提示都保留当前任务的完整工具列表和历史。

## 500 条轨迹，怎样算通过？

每次工具调用都检查。紧接用户的无调用回复也检查，防止模型遇到任何问题都硬调工具；工具返回后的普通总结不重复充当“无调用成功”。输入使用该轮之前的标注历史，因此这里测的是工具决策，还没有执行真实工具。

单轮通过需要工具名和参数都符合标注，且生成没有截断、超时或解析错误。一个回复里的并行调用按无序多重集合比较；一条轨迹的所有决策都通过，整条才通过。输出上限 {config['max_new_tokens']} token，上下文预算 {config['context_length']}，贪心解码，一批 {config['batch_size']} 条。超限和失败留在分母里。

本轮为 {args.run}。少样本示例固定在开始时，不随 dev 的错误换题；两种提示结束后按轨迹通过数选择，平局选较短的零样本提示。

生成时显式关闭采样，下面是实际参数中的核心几项。inputs 已通过官方模板渲染，并使用非思考前缀。

```python
output = model.generate(
    **inputs, do_sample=False, use_cache=True,
    max_new_tokens=1024,
    temperature=None, top_p=None, top_k=None,
)
```

这样固定生成策略，是为了让两种提示和后续模型接受同样的比较条件。
"""
    if summaries:
        note+="\n## 已完成的比较\n\n| 提示 | 完整轨迹通过 | 工具调用轮通过 | 不调用轮通过 | 截断 |\n| --- | --- | --- | --- | --- |\n"
        for name,s in summaries.items():
            note+=f"| {name} | {s['trajectory_passed']}/{s['trajectories']} | {s['call_turn_passed']}/{s['call_turns']} | {s['no_call_turn_passed']}/{s['no_call_turns']} | {s['truncated']} |\n"
    if not result or result['status']!='completed':
        current=progress.get('summary',{})
        note+=f"\n当前处理 {progress.get('current_prompt','尚未生成')}，已记录 {current.get('decision_turns',0)} 个决策轮。部分结果不用于宣布哪种提示更好。\n"
    elif len(summaries)==2:
        directory=ROOT/'experiments/E08';figures=directory/'figures';figures.mkdir(parents=True,exist_ok=True)
        write_json(directory/'summary.json',summaries)
        font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
        fig,axis=plt.subplots(figsize=(6.4,3.4),layout='constrained')
        values=[100*summaries[name]['trajectory_passed']/500 for name in ['zero_shot','few_shot']]
        bars=axis.bar(['零样本','少样本（3条示例）'],values,color=['#6B7280','#2563EB'],width=.55)
        axis.set_ylim(0,100);axis.set_ylabel('整条轨迹通过率（%）',fontproperties=font)
        axis.set_title('原模型提示对照：同一 dev，n=500 条独立轨迹',fontproperties=font)
        axis.set_xticks(range(2),['零样本','少样本（3条示例）'],fontproperties=font)
        axis.bar_label(bars,labels=[f'{value:.1f}%\n{summaries[name]["trajectory_passed"]}/500' for value,name in zip(values,['zero_shot','few_shot'])],padding=4)
        axis.yaxis.grid(True,color='#E5E7EB');axis.set_axisbelow(True)
        for side in ['top','right']:axis.spines[side].set_visible(False)
        path=figures/'prompt-comparison.png';fig.savefig(path,dpi=300);plt.close(fig)
        write_json(figures/'prompt-comparison.source.json',{'run_id':args.run,'source_sha256':sha256(directory/'summary.json'),
                         'image_sha256':sha256(path),'independent_trajectories':500,'inference':'greedy, batch=4'})
        with (directory/'decisions.csv').open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['prompt','sample_id','message_index','expected_call','predicted_call','exact','name_exact','truncated','error'],extrasaction='ignore')
            writer.writeheader()
            for name in ['zero_shot','few_shot']:
                rows=[json.loads(x) for x in (run/f'{name}.jsonl').read_text(encoding='utf-8').splitlines()]
                assert len(rows)==summaries[name]['decision_turns'] and sum(r['exact'] for r in rows)==summaries[name]['turn_passed']
                writer.writerows({'prompt':name,**r} for r in rows)
        note+='\n![图 E08-1：两种提示在相同 500 条 dev 上的整条轨迹通过率，贪心解码。](figures/prompt-comparison.png)\n'
        delta=summaries['few_shot']['trajectory_passed']-summaries['zero_shot']['trajectory_passed']
        direction='多' if delta>=0 else '少'
        note+=f"\n少样本比零样本{direction}通过 {abs(delta)} 条，相差 {abs(delta)/5:.1f} 个百分点。这是本 dev 的实际差值，后续固定选择的提示，再让训练条件接受同样的检查。并行类别只有 10 条，分类成绩需要结合这个样本量看。\n"
        zero,few=summaries['zero_shot'],summaries['few_shot']
        note+=f"\n示例让模型更愿意调用工具：调用轮从 {zero['call_turn_passed']}/{zero['call_turns']} 提高到 {few['call_turn_passed']}/{few['call_turns']}。代价也很明显，不调用轮从 {zero['no_call_turn_passed']}/{zero['no_call_turns']} 降到 {few['no_call_turn_passed']}/{few['no_call_turns']}。后面不能只盯着总体通过率，还要看微调能否减少这种过度调用。\n"
        example=next(r for r in rows if r['sample_id']=='glaive-60626' and r['message_index']==2)
        note+='\n## JSON 能解析，为什么仍然判错？\n\n例如，glaive-60626 的用户只说想往待办清单加任务，还没有给具体任务和优先级。标注要求先询问，少样本提示下模型却直接输出了下面的调用。这是本轮实际回复，格式检查通过，任务判断却错了。\n\n```json\n'+json.dumps(example['parsed'][0],ensure_ascii=False,indent=2)+'\n```\n\n模型把“想加任务”这句话当成了任务内容，又自行填了 medium 优先级。因此，可解析 JSON 和正确工具选择还不够，参数也要能从请求或工具返回中找到依据。\n'
    folder=ROOT/'experiments/E08';folder.mkdir(parents=True,exist_ok=True);(folder/'notes.md').write_text(note,encoding='utf-8')


if __name__=='__main__':main()
