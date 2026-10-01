"""把真实任务判据核验整理为中文笔记和调用统计。"""
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
from lab import ROOT, sha256, write_json


LABELS = {'read_locate': '读取定位', 'config_change': '配置修改', 'function_fix': '函数修复',
          'multi_file': '多文件一致性', 'failure_recovery': '失败恢复',
          'missing_information': '缺参数询问', 'no_tool': '无需工具回答',
          'invalid_path_recovery': '无效路径恢复'}


def main():
    folder = ROOT / 'experiments/E14'
    runs = sorted((folder / 'runs').glob('*.json'))
    result = next(json.loads(p.read_text(encoding='utf-8')) for p in reversed(runs)
                  if json.loads(p.read_text(encoding='utf-8')).get('status') == 'reference_pilot_verified')
    run = ROOT / '.local/runs' / result['run_id']
    assert sha256(run / 'records.json') == result['records_sha256']
    assert sha256(run / 'tasks.json') == result['tasks_sha256']
    records = json.loads((run / 'records.json').read_text(encoding='utf-8'))
    assert len(records) == result['task_count'] == 8
    assert result['reference_passed'] == result['initial_rejected'] == result['negative_rejected'] == 8
    assert result['model_calls'] == result['dev_tasks'] == result['test_tasks'] == 0
    rows = []
    for record, summary in zip(records, result['per_task']):
        events = record['reference_events']
        assert record['task_id'] == summary['task_id']
        assert record['reference']['passed'] and not record['initial']['passed'] and not record['negative']['passed']
        assert record['control_removed']['removed'] and record['reference_removed']['removed']
        calls, errors = len(events), sum(e['is_error'] for e in events)
        assert (calls, errors) == (summary['reference_calls'], summary['error_returns'])
        rows.append({'task_id': record['task_id'], 'category': record['category'], 'split': record['split'],
                     'reference_calls': calls, 'normal_returns': calls-errors, 'error_returns': errors,
                     'reference_passed': True, 'initial_rejected': True, 'negative_rejected': True})
    assert sum(r['reference_calls'] for r in rows) == result['reference_tool_calls']
    assert sum(r['error_returns'] for r in rows) == result['reference_error_returns']
    data = folder / 'reference-calls.csv'
    with data.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    fig, axis = plt.subplots(figsize=(6.4, 3.6), layout='constrained')
    positions = range(len(rows)); normal = [r['normal_returns'] for r in rows]; errors = [r['error_returns'] for r in rows]
    axis.barh(positions, normal, color='#6B7280', label='正常返回')
    axis.barh(positions, errors, left=normal, color='#9CA3AF', hatch='///', label='预期错误返回')
    axis.set_yticks(list(positions), [LABELS[r['category']] for r in rows], fontproperties=font)
    axis.invert_yaxis(); axis.set_xlim(0, 8); axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlabel('参考操作的工具调用次数（次）', fontproperties=font)
    axis.set_title('八个训练任务原型：真实调用，含失败后的恢复', fontproperties=font)
    axis.legend(prop=font, frameon=False, loc='lower right')
    axis.xaxis.grid(True, color='#E5E7EB'); axis.set_axisbelow(True)
    for i, row in enumerate(rows): axis.text(row['reference_calls']+.12, i, str(row['reference_calls']), va='center')
    for side in ['top', 'right']: axis.spines[side].set_visible(False)
    figures = folder / 'figures'; figures.mkdir(exist_ok=True)
    image = figures / 'reference-calls.png'; fig.savefig(image, dpi=300); plt.close(fig)
    write_json(image.with_suffix('.source.json'), {'source_file': data.name, 'source_sha256': sha256(data),
               'image_sha256': sha256(image), 'raw_records_sha256': result['records_sha256'],
               'run_id': result['run_id'], 'tasks': 8, 'model_calls': 0,
               'scope': '训练区原型的规则参考操作；不表示模型Agent表现'})
    table = '\n'.join(f"| {LABELS[r['category']]} | {r['reference_calls']} | {r['error_returns']} | 通过 | 拒绝 |"
                      for r in rows)
    note = f"""# E14：先把任务判准，再让模型来做

怎样才算任务做成了？这次先把判据跑了一遍。八类训练任务原型的参考操作全部通过；同一批任务的初始错误候选和刻意构造的错误候选，各有 8 个被拒绝。检查依据是最终文件、实际运行结果和必要的回复内容，模型说一句“已完成”不能代替这些结果。

这轮是 {result['run_id']}，从创建任务容器到完成核验用了 {result['duration_seconds']:.1f} 秒，不含写任务和整理文档的时间。参考过程共调用工具 {result['reference_tool_calls']} 次，包含 {result['reference_error_returns']} 次预期错误；另外有 {result['diagnostic_tool_calls']} 次调用用于构造错误候选。模型调用数仍为 0，八个原型只属于训练区，正式的 1,000 条轨迹和独立 dev、test 还没有完成。

## 为什么不直接从“把一个函数改对”开始算成绩？

单独改一行代码，容易忽略真正做任务时的上下文。这次把代码放进可离线运行的小项目：入口可能经过两个模块才找到目标，配置里有容易改错的相邻字段，多文件修改要同时对齐生产端、展示端和 schema。还保留了不应使用的旧文件，检查参考操作有没有沿着当前项目结构找到答案。

每个原型在两个独立容器里运行。第一个容器先检查原始错误状态或错误答案，再执行一个明确错误的操作；第二个容器从原始文件重新开始，运行规则给出的参考步骤。文件初始化和最终判据由核验程序执行，不混入参考轨迹的工具调用次数。

| 任务类别 | 参考调用（次） | 错误返回（次） | 参考结果 | 错误候选 |
| --- | --- | --- | --- | --- |
{table}

![图 E14-1：8 个训练区原型的真实参考调用；错误返回计入调用次数，模型调用为 0。](figures/reference-calls.png)

多文件任务实际用了七次工具调用，无需工具的回答则一次也没有调用。两种恢复任务各保留一次失败返回，随后根据项目内容继续处理。这张图说明参考过程实际做了哪些操作，不能据此推断模型会选对工具，也不能把调用次数多当成任务更难的充分证据。

## 只输出“通过”，能不能蒙混过去？不能

函数修复、多文件修改和失败恢复这三个任务，都试过把检查脚本改成直接输出预期的成功文本。命令的退出码变成了 0，输出也对上了，但判据仍然拒绝：检查文件的哈希与初始值不同。这样可以挡住“修改检查，让错误代码看起来通过”的做法。

下面是实际实现中检查受保护文件的核心逻辑。`files` 是结束时的文件快照，`task.files` 保存初始文本，`allowed_changes` 只列本次允许修改的文件。

```javascript
for (const [name, value] of Object.entries(task.files)) {{
  if (!task.allowed_changes.includes(name) &&
      byPath.get(name)?.sha256 !== digest(value)) {{
    reasons.push('protected_file_changed:' + name);
  }}
}}
```

配置任务还做了一个更隐蔽的错误：search 的缓存容量按要求改成 120，却顺手把 admin 的容量从 20 改成 21。文件可以解析，但完整 JSON 状态比较仍然拒绝。只检查目标字段变成 120，会漏掉这个额外改动。

## 错误返回有什么用？字符串修复这次就靠它定位

失败恢复任务先运行现有检查，真实返回退出码 1：输入含连续空格时，得到 `a- b`，预期是 `a-b`。读回函数后可以看到，旧写法只替换一次普通空格。参考操作把它改成匹配连续空白的正则，再运行原检查，五个样例全部通过。

```javascript
// 旧写法只替换第一个普通空格
value.trim().replace(' ', '-').toLowerCase();
// 修复后同时处理连续空格、制表符和换行
value.trim().replace(/\\s+/g, '-').toLowerCase();
```

这五个样例涵盖连续空格、制表符、换行、普通字符和纯空白。失败的第一次调用仍留在轨迹里，不能为了让过程显得顺利而删掉它。最终判据又独立运行一次原检查，核对退出码和输出。

路径恢复也保留了真实 ENOENT：旧库存文件不存在。参考步骤继续读项目说明和导出清单，找到当前库存，数量 7、12、4 相加得到 23。错误候选则新建了一个同样合计 23 的旧路径文件，即使答案里的数对了，判据仍因新增文件而拒绝。

## 信息没给全时，问清楚比猜一个数更合适

用户只说把上传超时“调大一些”，没有给目标秒数。参考操作读到当前为 30 秒后询问目标值，保留原配置。错误候选一边问目标值，一边已经改成了 60 秒；回复看起来谨慎，文件却已被擅自修改，所以仍被拒绝。

无需工具的任务则直接解释“每秒最多允许八次请求”。错误候选答案正确，却先去读了无关配置；因为任务明确要求只根据已给文字回答，这次额外调用也被拒绝。这里考察的是有没有遵守具体任务约束，不是规定所有解释性问题都禁止使用工具。

## 这轮确认了什么，还要补什么？

现在确认了八个原型能真实执行，判据能够发现这批明确构造的错误，文件快照和失败返回也能保留下来。每个任务的参考调用都在 12 次预算内；容器无网络、没有宿主目录挂载，也没有 GPU 请求。本轮创建的 16 个容器结束后都已移除。

原型的覆盖还有限。代码判据使用任务自带的固定样例，不能保证发现所有绕过方式；询问与解释类回复只做必要词语检查，真实模型接入后还要逐条复查语义。后续扩展训练任务结构，并为 dev 和 test 单独建立模板族、仓库族，避免只换几个变量名就把同一道题算成未见过的任务。

复习时可以先看两个问题。为什么命令输出正确还可能失败？因为检查文件或不该动的配置也可能被改过，输出只是成功条件的一部分。为什么参考操作 8/8 不能写成 Agent 成功率 100%？因为步骤由规则预先给定，还没有让模型自己选择工具、读取错误并决定下一步。
"""
    (folder / 'notes.md').write_text(note, encoding='utf-8')


if __name__ == '__main__':
    main()
