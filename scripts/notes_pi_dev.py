"""从实际 dev 核验记录整理任务说明，供 E14 主笔记持续更新。"""
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from lab import ROOT, sha256, write_json


def dev_note(folder, labels):
    runs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((folder/'runs').glob('*.json'))]
    completed = [r for r in runs if r.get('status') == 'dev_reference_verified']
    if not completed: return ''
    result = completed[-1]; local = ROOT/'.local/runs'/result['run_id']
    records = json.loads((local/'records.json').read_text(encoding='utf-8'))
    tasks = json.loads((local/'tasks.json').read_text(encoding='utf-8'))
    assert sha256(local/'records.json') == result['records_sha256']
    assert sha256(local/'tasks.json') == result['tasks_sha256']
    assert len(records) == result['task_count'] == result['reference_passed'] == result['initial_rejected'] == result['negative_rejected'] == 40
    assert all(r['reference']['passed'] and not r['initial']['passed'] and not r['negative']['passed'] for r in records)
    assert all(r['control_removed']['removed'] and r['reference_removed']['removed'] for r in records)
    descriptions = ['按两个功能开关和路由表定位实际处理函数', '只更新指定租户的 email 订阅行',
        '解析逗号、引号、转义引号及尾部空列', '同步函数导出、转发入口和调用方名称',
        '等待异步格式化，保留异常向上传播', '两个租户都匹配时询问目标，不写配置',
        '按题中定义计算20个测量值的p95', '从旧路径错误找到截止日前最近成功报告']
    rows = []
    for category, description in zip(labels, descriptions):
        group = [r for r in records if r['category'] == category]
        assert len(group) == 5
        calls = [len(r['reference_events']) for r in group]
        call_range=str(calls[0]) if min(calls)==max(calls) else f'{min(calls)}—{max(calls)}'
        rows.append(f"| {labels[category]} | {description} | 5 | {call_range} |")
    csv_task = next(t for t in tasks if t['category'] == 'function_fix')
    case = next(r for r in records if r['task_id'] == csv_task['task_id'])
    assert case['negative']['execution']['code'] != 0
    note = f'''
## dev能检查迁移吗？40个场景已冻结，按八个新任务族观察

{result['run_id']} 实际执行了40个dev场景，八类各5个。规则参考40/40通过，40个初始错误和40个明确构造的错误候选都被拒绝。参考过程调用工具{result['reference_tool_calls']}次，保留{result['reference_error_returns']}次预期错误返回；错误候选另用了{result['diagnostic_tool_calls']}次工具。每个场景各用一个检查错误候选的容器和一个执行参考解的容器，本轮80个容器均已移除。

| 类别 | dev具体检查什么 | 场景（个） | 每条参考调用（次） |
| --- | --- | --- | --- |
{chr(10).join(rows)}

训练区有16个模板族，这轮dev另用8个模板族和仓库族，两边完整问题与初始文件的精确重复数为零。差别还需要看实际任务：训练配置题修改嵌套值或环境覆盖，dev题要在数组中同时匹配租户和通道；训练函数题处理空白或去重，dev题要按引号状态解析CSV。任务结构变了，模型是否能迁移到这些结构，要等它自己运行后判断。

40个场景也不是40种题型。每个新模板有5个不同场景，族内仍有相似性，最终同时列任务成绩和任务族成绩。dev可用于选择方案；最终100个test继续另建任务族，不从dev换几个数值获得。

## 去掉引号再按逗号分割，为什么会错？引号内的逗号也是字段内容

CSV任务中的一个实际样例是下面这行，前两列分别应是a,b和c，而不是a、b、c三个字段：

```text
"a,b",c
```

错误候选去掉所有引号后再split，遇到这个样例就失去字段边界；它在容器中执行后，被原检查判为失败。参考解逐字符维护引号状态，只有引号外的逗号才结束当前字段，还处理转义双引号和最后一个空字段。这次只要求单行CSV，没有把跨行字段写成已覆盖能力。

## 正常结果看着对，恢复过程就对了吗？还要保留异常

异步格式化任务最初把Promise拼进字符串，先运行原命令确实报错。错误候选绕过格式化函数后，普通数字看起来正常，但负值本应抛出的异常被吞掉，仍被拒绝。参考解等待异步结果，正常、零值和异常传播都通过了原检查。成功判据检查行为，不能只看一句“完成了”。

这批参考解是任务规则给出的步骤，没有模型自主调用，尚无Agent成功率。询问类目前只查必要词和禁止修改，语义判据仍较粗；模型接入后需要结合实际回复逐条复查。最终文件、工具返回和错误候选都保留，用来定位以后真正的模型失败。
'''
    inspected=[r for r in runs if r.get('status')=='split_inspection_verified' and r['config']['dev_run']==result['run_id']]
    if inspected:
        inspected=inspected[-1]
        assert sha256(ROOT/'.local/runs'/inspected['run_id']/'inspection.json')==inspected['inspection_sha256']
        note+=f"\n近似重复也做了一轮检查：问题和初始文件统一小写、归一数字后，按字符5-gram的Jaccard相似度比较train与dev的80,000对组合，阈值固定0.8。归一后的精确重复为零，达到阈值的组合也为零，最高相似度为{inspected['max_jaccard']:.4f}。这种筛查还不能排除语义上的相似，也查不到预训练阶段见过哪些样本，所以仍要结合上面的任务结构区别理解。\n"
    measured = [r for r in runs if r.get('status') == 'dev_lengths_verified' and r['source_run'] == result['run_id']]
    if measured:
        length = measured[-1]; source = ROOT/'.local/runs'/length['run_id']/'lengths.csv'
        assert sha256(source) == length['lengths_sha256']
        data = folder/'dev-lengths.csv'; data.write_bytes(source.read_bytes())
        with data.open(encoding='utf-8') as handle: points = list(csv.DictReader(handle))
        maxima = [max(int(p['reference_full_tokens']) for p in points if p['category'] == c) for c in labels]
        font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
        fig, axis = plt.subplots(figsize=(6.4, 3.2), layout='constrained')
        axis.barh(range(8), maxima, color='#6B7280')
        axis.set_yticks(range(8), list(labels.values()), fontproperties=font); axis.invert_yaxis()
        axis.axvline(8192, color='#9CA3AF', linestyle='--', linewidth=1)
        axis.text(8070, -.5, '8192预算', ha='right', fontproperties=font, fontsize=9)
        axis.set_xlim(0, 9000); axis.set_xlabel('完整参考历史长度（token，含工具定义）', fontproperties=font)
        axis.set_title('40个dev场景：各类最长参考历史，未调用模型', fontproperties=font)
        axis.xaxis.grid(True, color='#E5E7EB'); axis.set_axisbelow(True)
        for index, value in enumerate(maxima): axis.text(value+80, index, str(value), va='center', fontsize=9)
        for side in ['top', 'right']: axis.spines[side].set_visible(False)
        image = folder/'figures/dev-lengths.png'; fig.savefig(image, dpi=300); plt.close(fig)
        write_json(image.with_suffix('.source.json'), {'source_file': data.name, 'source_sha256': sha256(data),
            'image_sha256': sha256(image), 'run_id': length['run_id'], 'reference_run': result['run_id'],
            'task_count': 40, 'reply_points': len(points), 'scope': length['scope']})
        note += f'''
## 工具返回都带上，会超过上下文吗？这批参考没有超过8192

{length['run_id']} 用固定系统提示、真实工具定义及原样工具返回，检查{length['reply_points']}个回复位置。最长推理前缀{length['max_prompt_tokens']:,}token，带上该次参考回复后最长{length['max_reference_full_tokens']:,}token；超8192的数量为零。只运行tokenizer，没有加载模型、初始化CUDA或生成训练标签。

![图 E14-3：每类5个场景的最长完整参考历史；虚线为固定上下文预算。](figures/dev-lengths.png)

图里测的是参考解走过的历史。模型可能多读文件、反复失败或生成更长回复，所以这不能保证它的实际轨迹也在预算内；正式运行仍逐条记录长度、截断和超时，保留在40个任务的分母里。
'''
    return note
