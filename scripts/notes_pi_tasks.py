"""把真实任务判据核验整理为中文笔记和调用统计。"""
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from lab import ROOT, sha256, write_json


LABELS = {'read_locate': '读取定位', 'config_change': '配置修改', 'function_fix': '函数修复',
          'multi_file': '多文件一致性', 'failure_recovery': '失败恢复',
          'missing_information': '缺参数询问', 'no_tool': '无需工具回答',
          'invalid_path_recovery': '无效路径恢复'}


def encoding_note(folder, source_run):
    completed = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((folder/'runs').glob('*.json'))]
    matching = [r for r in completed if r.get('status')=='pilot_encoding_verified' and r['config']['source_run']==source_run]
    if not matching: return ''
    result = matching[-1]; run = ROOT/'.local/runs'/result['run_id']
    assert sha256(run/'units.jsonl')==result['units_sha256']
    assert sha256(run/'encoded.jsonl')==result['encoded_sha256']
    units = [json.loads(line) for line in (run/'units.jsonl').read_text(encoding='utf-8').splitlines()]
    encoded = [json.loads(line) for line in (run/'encoded.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(units)==result['assistant_units']==34
    example = next(e for e in encoded if e['sample_id']=='pilot-failure_recovery' and e['message_index']==4)
    meta = next(u for u in units if u['sample_id']==example['sample_id'] and u['message_index']==4)
    mask = [int(x!=-100) for x in example['labels']]
    assert sum(mask)==meta['target_tokens'] and not any(mask[:meta['target_start_token']])
    data = folder/'current-reply-mask.csv'
    with data.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['token_position','supervised'])
        writer.writerows(enumerate(mask))
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    fig,axes=plt.subplots(3,1,figsize=(6.4,3.0),gridspec_kw={'height_ratios':[1,1,.35]},layout='constrained')
    colors=ListedColormap(['#D1D5DB','#059669'])
    for axis,start,label in zip(axes[:2],[0,len(mask)-64],['完整单元','末尾64个token']):
        axis.imshow([mask[start:]],extent=[start,len(mask),0,1],aspect='auto',interpolation='nearest',cmap=colors,vmin=0,vmax=1)
        axis.set_yticks([.5],[label],fontproperties=font);axis.tick_params(axis='y',length=0)
    axes[0].set_title(f'失败返回后的下一步：输入{len(mask)}个token，只监督当前{sum(mask)}个',fontproperties=font)
    axes[1].set_xlabel('token位置（下图放大末尾区间）',fontproperties=font)
    axes[2].axis('off')
    axes[2].legend(handles=[Patch(color='#D1D5DB',label='历史与工具返回：忽略'),Patch(color='#059669',label='当前回复：监督')],
                   prop=font,frameon=False,loc='center',ncol=2)
    image=folder/'figures/current-reply-mask.png';fig.savefig(image,dpi=300);plt.close(fig)
    write_json(image.with_suffix('.source.json'),{'source_file':data.name,'source_sha256':sha256(data),
                'image_sha256':sha256(image),'run_id':result['run_id'],'encoded_sha256':result['encoded_sha256'],
                'sample_id':meta['sample_id'],'message_index':4,'input_tokens':len(mask),'target_tokens':sum(mask),
                'scope':'真实训练单元的监督遮罩；不是模型表现'})
    table=[]
    for category,label in LABELS.items():
        group=[u for u in units if u['category']==category]
        table.append(f"| {label} | {len(group)} | {max(u['input_tokens'] for u in group)} | {sum(u['target_tokens'] for u in group)} |")
    return f"""
## 工具轨迹怎么变成训练样本？保留历史，每次只学当前回复

{result['run_id']} 把这八条参考过程转换成了 {result['assistant_units']} 个回复单元，共 {result['supervised_tokens']:,} 个监督 token。一次工具调用是一个 assistant 回复，轨迹结束时的总结或询问又是一个回复；所以“八条轨迹”并不等于“八个训练单元”。这些答案和步骤仍来自已执行的任务规则，没有混入教师生成。

转换时逐个保留调用 id、工具名、参数和真实文本返回，两次错误也没有丢掉。官方 Qwen3 模板在文本中呈现工具名和参数，不把本轮 Pi 的调用 id 当作模型要生成的内容；数据里仍保存 id，便于核对请求与返回是否对应。34 个单元逐个检查了模板文本、token、推理前缀和监督位置，带监督标记的模板与官方模板渲染一致。

| 任务类别 | 回复单元（个） | 最长输入（token） | 监督量（token） |
| --- | --- | --- | --- |
{chr(10).join(table)}

这里的输入长度包含工具定义和完整历史。最长单元为 {result['max_sequence_length']} token，34 个单元都在 2048 以内，没有截断。这只是八个短项目的实测长度，后续扩展任务仍要逐条检查，不能据此认定所有领域轨迹都适合 2048。

## 前面报错了，会不会连报错文本也一起训练？这次没有

字符串修复中，第一次命令失败后，下一步应读回函数。这个真实单元有 {meta['input_tokens']} 个输入 token，监督位置只有当前回复的 {meta['target_tokens']} 个 token；此前的调用、错误返回和问题都被遮掉，只作为上下文。监督量包含官方模板在 assistant 分支中的标记和结束 token，不能把它全当成普通答案文字的长度。

![图 E14-2：{result['run_id']} 的一个真实单元；上图显示完整输入，下图放大末尾64个token，灰色位置的label为-100。](figures/current-reply-mask.png)

错误返回留在输入里，模型有机会根据它决定下一步；loss 的目标仍是当前回复。这次全部 34 个单元的历史监督量都为零，图中选的是明确经历过命令失败的单元。检查通过说明数据表示与遮罩对应，还不能说明模型训练后学会了恢复。

该单元的目标里实际包含下面这次调用：

```json
{{"name": "read", "arguments": {{"path": "src/slug.mjs"}}}}
```

已经得到可追溯的原型训练格式，正式 1,000 条有效轨迹和领域训练还要继续完成。本轮只调用 tokenizer，未初始化 CUDA，也未进行模型前向或更新参数。
"""


def extension_note(folder):
    runs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((folder/'runs').glob('*.json'))]
    matching = [r for r in runs if r.get('status') == 'reference_extension_verified']
    if not matching: return ''
    result = matching[-1]; run = ROOT/'.local/runs'/result['run_id']
    assert sha256(run/'records.json') == result['records_sha256']
    assert sha256(run/'tasks.json') == result['tasks_sha256']
    assert result['task_count'] == result['reference_passed'] == result['initial_rejected'] == result['negative_rejected'] == 8
    assert result['model_calls'] == result['dev_tasks'] == result['test_tasks'] == 0
    details = ['按配置从注册表选择存储实现', '只加 staging 覆盖，保留生产默认值',
               '去重后按最后出现的位置排序', '毫秒改秒，保持真实延时', '模块移动后的相对导入修复',
               '提醒时间没有给时区，先询问', '根据已知速率计算窗口预算', '按商品状态联表统计库存']
    rows = [f"| {LABELS[r['category']]} | {description} | {r['reference_calls']} |"
            for r,description in zip(result['per_task'],details)]
    encoded = [r for r in runs if r.get('status')=='reference_encoding_verified' and r['config']['source_run']==result['run_id']]
    pending = '新增八条尚未完成训练格式转换，'
    format_note = ''
    if encoded:
        data=encoded[-1]; local=ROOT/'.local/runs'/data['run_id']
        assert sha256(local/'units.jsonl')==data['units_sha256'] and sha256(local/'encoded.jsonl')==data['encoded_sha256']
        assert sha256(ROOT/data['processed_file'])==data['processed_sha256']
        examples=json.loads((local/'mask-examples.json').read_text(encoding='utf-8'))
        assert len(examples)==2 and all(e['history_supervised_tokens']==0 for e in examples)
        pending=''
        format_note=f'''
## 新增过程能完整表示吗？37个单元已逐条核验

{data['run_id']} 将新增八条过程转换为{data['assistant_units']}个回复单元，共{data['supervised_tokens']:,}个监督token，最长输入{data['max_sequence_length']}token。29次调用的参数、实际返回和最终回复逐条与原记录比对，解码监督目标也对应当前动作；没有截断，未初始化CUDA。

导入失败后的下一步是读迁移说明，这个单元输入{examples[0]['input_tokens']}token，只监督当前{examples[0]['target_tokens']}个；旧路径错误后的下一步是读项目说明，输入{examples[1]['input_tokens']}token，只监督当前{examples[1]['target_tokens']}个。两个错误返回都留在上下文中，历史监督量为零。这里验证了表示和遮罩，没有进行参数训练。

两批现有16条规则参考合计71个回复单元、2,563个监督token。它们各自属于原模板族，不能把展开单元当成新增独立任务，也不能由最长1,497token推断正式扩展数据都适合2048；后续数据继续逐条检查长度。
'''
    return f'''
## 再加八个模板，解题过程变在哪里？

{result['run_id']} 新增了八个训练模板族。参考操作8/8通过，初始错误和八个明确构造的错误方案全部被拒绝。新增参考共调用工具{result['reference_tool_calls']}次，包含{result['reference_error_returns']}次真实错误返回；错误方案另用了{result['diagnostic_tool_calls']}次工具。容器执行与记录耗时{result['duration_seconds']:.1f}秒，这个时间不含任务编写与文档整理。

| 类别 | 新模板需要处理什么 | 参考调用（次） |
| --- | --- | --- |
{chr(10).join(rows)}

这次改变了数据结构或错误原因。原来的配置任务直接修改一个值，新任务要区分默认值和环境覆盖；原来迁移指标字段，新任务还要做单位换算；路径恢复从单文件求和变成两份数据联表，并排除停售和未知商品。

## 字段改好了，为什么仍然判失败？实际延时没有保持

时间单位迁移的错误方案把refreshMs改成refreshSeconds，却保留2500这个数，读取函数也直接返回它。最终输出看似仍是2500毫秒，但新配置的含义已变成2500秒，而且其他秒数无法正确换算。判据同时核对配置值与读取函数：2500毫秒应写成2.5秒，0.125秒应读成125毫秒。错误方案的配置检查与执行检查都失败，规则参考保留了实际延时。

参考修复中的这行是真实执行过的代码：

```javascript
export const delay = config => config.refreshSeconds * 1000;
```

## 倒序去重能保留最后对象，为什么也失败？顺序和输入都要检查

新去重任务要求a、b、a返回b、a。错误方案先反转输入数组再去重，虽然能留下最后一个a，却改变了输入，也没有遵守输出顺序。现有检查比较独立的筛选结果，并在调用前后核对输入序列；本轮执行了42组样例，其中40组由固定seed=73生成。42组函数样例仍属于一个任务，不能算成42条领域轨迹。

移动模块的恢复任务也保留了第一次命令的真实导入错误。参考步骤根据迁移说明修正相对导入，再跑原检查；错误方案补回旧路径文件，被新增文件和执行检查拒绝。故障原因与上一批的空白正则错误不同。

目前两批共16个训练模板族，各有一条已验证参考过程。所有任务仍在train；{pending}正式1,000条有效训练记录、独立dev/test和模型迁移评测继续分别验收。参考步骤由规则给定，没有模型自主选择工具；这16条不能写成模型成功率。判据仍有有限样例和询问语义只查必要词的局限，后续模型结果要保留逐条复查。
'''+format_note


def generation_note(folder):
    runs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((folder/'runs').glob('*.json'))]
    frozen=[r for r in runs if r.get('status')=='training_requests_frozen']
    if not frozen:return ''
    data=frozen[-1];assert sha256(ROOT/data['requests_file'])==data['requests_sha256']
    note=f'''
## 有2,000个场景，是否已经有2,000条有效轨迹？还没有，先逐条执行

{data['run_id']} 冻结了2,000个训练场景，16个模板族各125个，八类各250个。前1,000个场景作为本轮规则参考目标，八类各125个。清单改变实际配置、边界样例、选择的模块、时间和库存关系；这里只统计待执行场景，不把生成的参考步骤直接算成成功轨迹。

前两轮都被精确去重拦住。第一轮实际只有1,979个不同场景，时间单位任务出现21个重复；第二轮调整延时取值后，提醒任务仍有7个重复，只有1,993个不同场景。随机取值的范围有限，id不同也会抽到同一组实际内容。两轮没有用于参考批量执行。随后为延时和日期使用确定的取值序列，保留其他场景变化，新的2,000份完整问题与文件内容没有精确重复。

这批数据仍共享16个模板族，属于训练区的场景增广。精确去重不能消除这种结构相似性，模型接触过的处理方式也不会因此变成未见任务。后续dev/test另建模板和仓库族；当前没有冻结这两个任务集，也没有让教师接触它们。
'''
    batch=[]
    for config_file in sorted((ROOT/'.local/runs').glob('E14-R*/config.json')):
        config=json.loads(config_file.read_text(encoding='utf-8'))
        if config.get('operation')=='pi_reference_batch' and config.get('source_scene_run')==data['run_id']:batch.append(config_file.parent)
    if batch:
        run=batch[-1];progress=json.loads((run/'progress.json').read_text(encoding='utf-8'))
        outcome='整轮还在继续'
        if (run/'result.json').exists():
            result=json.loads((run/'result.json').read_text(encoding='utf-8'))
            outcome='整轮参考执行已结束' if result['status']=='reference_batch_verified' else '本轮未达到有效轨迹目标，保留已执行结果'
        note+=f"\n{run.name} 已实际处理{progress['completed']}/{progress['target']}个场景，其中{progress['valid']}条参考通过，{progress['failed']}条失败；已记录{progress['reference_tool_calls']}次工具调用和{progress['reference_error_returns']}次错误返回。{outcome}，失败保留在原目标分母。每条过程使用独立容器，执行结束核对最终文件，再移除本轮容器；本阶段没有调用模型。\n"
        encoded=[r for r in runs if r.get('operation')=='pi_reference_encoding' and
                 r.get('status')=='reference_encoding_verified' and r['config']['source_run']==run.name]
        if encoded:
            r=encoded[-1]
            note+=f"\n完整参考过程随后在 {r['run_id']} 转成训练格式：{r['independent_trajectories']:,}个场景展开为{r['assistant_units']:,}个当前回复单元，共{r['supervised_tokens']:,}个监督token，最长{r['max_sequence_length']:,}个token。每个单元都核对了实际工具返回、官方模板和监督位置，没有截断。数据仍属于上述16个训练模板族；格式核验通过后，领域训练与独立任务迁移评测继续单独进行。\n"
    return note


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
    (folder / 'notes.md').write_text(note+encoding_note(folder,result['run_id'])+extension_note(folder)+generation_note(folder), encoding='utf-8')


if __name__ == '__main__':
    main()
