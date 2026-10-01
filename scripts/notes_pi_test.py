"""把最终任务的真实参考核验归入E14笔记，模型成绩另记。"""
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from lab import ROOT, sha256, write_json


def test_note(folder, labels):
    runs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((folder/'runs').glob('*.json'))]
    active=[]
    for path in sorted((ROOT/'.local/runs').glob('E14-R*/config.json')):
        config=json.loads(path.read_text(encoding='utf-8'))
        if config.get('operation')=='pi_test_probe' and not (path.parent/'result.json').exists():active.append(path.parent)
    completed=[r for r in runs if r.get('status')=='test_reference_verified']
    failed=[r for r in runs if r.get('operation')=='pi_test_probe' and r['status']=='failed']
    note=''
    if active:
        current=active[-1]; progress=json.loads((current/'progress.json').read_text(encoding='utf-8')) if (current/'progress.json').exists() else {'completed':0}
        note+=f'\n## 最终测试题准备到哪一步？正在检查100个场景的参考与判据\n\n{current.name} 已保存{progress["completed"]}/100个场景的参考核验。按预先分配的配额使用16个新任务族，逐条检查初始错误、明确错误候选和参考过程；全部通过后才冻结。此时没有调用模型，也没有最终成绩。\n'
    if failed:
        r=failed[-1]; failure=json.loads((ROOT/'.local/runs'/r['run_id']/'failure.json').read_text(encoding='utf-8'))
        partial=failure.get('partial_task') or {}; reasons=partial.get('reference',{}).get('reasons',[])
        syntax='SyntaxError: Invalid or unexpected token' in partial.get('reference',{}).get('execution',{}).get('stderr','')
        heading='参考解为什么也没通过？生成的检查文件先出了语法错误' if syntax else '这一轮为什么停了？参考没有通过现有判据'
        note+=f'\n## {heading}\n\n{r["run_id"]} 完成了前{r["task_count"]}个场景，停止在{partial.get("task_id","场景清单检查")}。失败时的文件、调用和返回都保留，修复后另开运行。\n'
        if syntax:
            note+='\n编号样例含末尾换行，生成器把它直接拼进单引号字符串，写出的JavaScript无法解析。容器中的实际文件和Node报错对应得上；这次不能把问题归到编号函数。后续用JSON.stringify序列化整组样例，让换行以转义形式进入文件。题目规则和检查样例没有删减，完整100题重新核验。\n\n```javascript\n// 样例先当数据保存，再序列化成合法的源码字面量\nconst cases = [["AB-123\\n", false]];\nconst literal = JSON.stringify(cases);\n```\n'
    if not completed:return note
    r=completed[-1]; source=ROOT/'.local/runs'/r['run_id']
    assert sha256(source/'records.json')==r['records_sha256'] and sha256(source/'tasks.json')==r['tasks_sha256']
    records=json.loads((source/'records.json').read_text(encoding='utf-8')); tasks=json.loads((source/'tasks.json').read_text(encoding='utf-8'))
    assert len(records)==len(tasks)==r['task_count']==r['reference_passed']==r['initial_rejected']==r['negative_rejected']==100
    descriptions=['依赖图间接影响、有序拦截链','规则插入位置、共享配额解绑','拓扑排序与环检测、递归合并删除',
        '校验和编解码、权限集合往返','跨调用正则状态、嵌套默认对象共享','页码基数、删除或归档约定',
        '加权均值、按公式约分F1','空查询别名恢复、分片最新事件归并']
    rows=[]
    for category,description in zip(labels,descriptions):
        group=[entry for entry in records if entry['category']==category]; calls=[len(entry['reference_events']) for entry in group]
        rows.append(f'| {labels[category]} | {len(group)} | {description} | {min(calls)}—{max(calls)} |')
    empty=sum(any(e['tool']=='bash' and not e['is_error'] and any(b['type']=='text' and b['text'].strip()=='[]' for b in e.get('result',{}).get('content',[])) for e in entry['reference_events']) for entry in records)
    note+=f'''
## 最终100题能用了么？题目和参考已核验，模型还没跑

{r['run_id']} 实际检查了100个场景：参考100/100通过，100个初始错误和100个明确错误候选都被拒绝。共{r['reference_tool_calls']}次参考工具调用，保留{r['reference_error_returns']}次预期错误，以及{empty}个场景中的真实空查询；错误候选另调用工具{r['diagnostic_tool_calls']}次。本轮200个独立容器均已移除，参考与错误候选没有共用状态。

配额先为八类各留10题，再把余下20题分给更需要追踪状态、维护多个文件或恢复错误的类别。没有参考模型成绩来删题或调分母。

| 类别 | 场景（个） | 两个新任务族检查什么 | 参考调用（次） |
| --- | --- | --- | --- |
{chr(10).join(rows)}

这100个场景分属16个模板族，仍不是100种完全不同的问题。测试题要求的算法和状态处理与train、dev有所区别：函数题要考虑图的依赖与环、对象递归及输入引用，多文件题要让读写双方遵守相同契约。它们属于自编的小仓库任务集，正式结果按任务和任务族分别展示，不代替标准榜单。

清单已经冻结。此后用dev选提示词和checkpoint，教师只处理train；最终test等方案确定后统一运行，每个任务重复三次，失败、超时和截断仍留在100题的分母里。

## 答案算对了，空查询题就算成功吗？还要真的完成恢复过程

其中一个场景第一次查询旧key，工具正常退出，却返回空数组。参考过程读项目说明和别名，再用当前key查到数据，只汇总done记录。错误候选直接给出完全正确的最终JSON，却没有经历空查询，仍被判据拒绝。这道题明确要求先查旧key，再恢复；检查的是这个过程，而不只是最终数字。

```javascript
// 实际判据中的空查询检查
block.type === 'text' && block.text.trim() === '[]'
```

共享默认配置的错误候选也值得看：只复制顶层对象，cache和tags仍指向原对象。第一次返回值改了tags，下一次就拿不到原始默认值。参考解复制嵌套数据，保留显式false、retries=0和未指定的默认字段，原检查才全部通过。

```javascript
// 只复制顶层，嵌套对象仍会共享
const result = {{...defaults}};
// 本轮参考从独立默认对象开始，再应用覆盖项
const result = structuredClone(defaults);
```

这些明确错误候选只能说明判据能拦住本轮检查过的错误。询问类仍需复查模型回复的语义，代码题的固定样例也不能证明覆盖所有实现。正式模型轨迹要继续检查实际调用、返回和文件状态。
'''
    lengths=[v for v in runs if v.get('status')=='test_lengths_verified' and v['source_run']==r['run_id']]
    if lengths:
        v=lengths[-1]; raw=ROOT/'.local/runs'/v['run_id']/'lengths.csv'; assert sha256(raw)==v['lengths_sha256']
        data=folder/'test-lengths.csv';data.write_bytes(raw.read_bytes())
        with data.open(encoding='utf-8') as handle:points=list(csv.DictReader(handle))
        maxima=[max(int(p['reference_full_tokens']) for p in points if p['category']==c) for c in labels]
        font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');fig,axis=plt.subplots(figsize=(6.4,3.2),layout='constrained')
        axis.barh(range(8),maxima,color='#6B7280');axis.set_yticks(range(8),list(labels.values()),fontproperties=font);axis.invert_yaxis()
        axis.axvline(8192,color='#9CA3AF',linestyle='--',linewidth=1);axis.text(8070,-.5,'8192预算',ha='right',fontproperties=font,fontsize=9)
        axis.set_xlim(0,9000);axis.set_xlabel('完整参考历史长度（token，含工具定义）',fontproperties=font)
        axis.set_title('100个test场景：各类最长参考历史，未调用模型',fontproperties=font)
        axis.xaxis.grid(True,color='#E5E7EB');axis.set_axisbelow(True)
        for i,value in enumerate(maxima):axis.text(value+80,i,str(value),va='center',fontsize=9)
        for side in ['top','right']:axis.spines[side].set_visible(False)
        image=folder/'figures/test-lengths.png';fig.savefig(image,dpi=300);plt.close(fig)
        write_json(image.with_suffix('.source.json'),{'source_file':data.name,'source_sha256':sha256(data),'image_sha256':sha256(image),
            'run_id':v['run_id'],'reference_run':r['run_id'],'task_count':100,'reply_points':len(points),'scope':v['scope']})
        note+=f'''
## 题目变复杂，会不会先撞上下文？本轮参考最长{v['max_reference_full_tokens']:,}token

{v['run_id']} 用固定系统提示、实际工具定义、原样返回和非思考模板检查{v['reply_points']}个回复位置。最长生成前缀{v['max_prompt_tokens']:,}token，带上当前参考回复最长{v['max_reference_full_tokens']:,}token，超过8192的数量为零。本轮只运行tokenizer，没有初始化CUDA或生成训练标签。

![图 E14-4：100个test场景的实际参考历史，各类最长值；虚线为固定预算。](figures/test-lengths.png)

这是参考解的长度。模型可能反复读取或改错后重试，正式评测仍记录实际上下文、工具次数和超时，不能用参考长度代替模型轨迹长度。
'''
    inspections=[v for v in runs if v.get('status')=='test_split_inspection_verified' and v['config']['test_run']==r['run_id']]
    if inspections:
        v=inspections[-1]; train=v['comparisons']['train'];dev=v['comparisons']['dev']
        note+=f'\n重复检查也已完成：test与2,000条train请求比较{train["pairs_checked"]:,}对，与40个dev比较{dev["pairs_checked"]:,}对。场景、模板族和仓库族交叉为零；数字归一后的精确重复及达到0.8阈值的字符5-gram组合均为零，最高相似度分别为{train["max_jaccard"]:.4f}和{dev["max_jaccard"]:.4f}。这能复查具体筛查规则，仍不能排除语义相似或预训练污染。\n'
    return note
