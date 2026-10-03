"""从实际 Pi 运行生成教师对照笔记。"""
import argparse
import json
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from lab import ROOT, sha256, write_json

LABELS = {'missing_information': '缺信息先询问', 'failure_recovery': '失败后恢复',
          'read_locate': '读取定位', 'invalid_path_recovery': '无效路径恢复',
          'multi_file': '多文件修改', 'no_tool': '无需调用工具',
          'function_fix': '函数修复', 'config_change': '配置修改'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metrics(run):
    folder = ROOT / '.local/runs' / run
    rows = read(folder / 'progress.json')['rows']
    result = read(folder / 'result.json') if (folder / 'result.json').exists() else {}
    requests = [request for row in rows for request in row.get('model_requests', [])]
    return {'run': run, 'config': read(folder / 'config.json'), 'rows': rows, 'result': result,
            'passed': sum(bool(row.get('passed')) for row in rows), 'requests': len(requests),
            'timeouts': sum(bool(row.get('agent_timed_out')) for row in rows),
            'budget': sum(bool(row.get('tool_budget_exhausted')) for row in rows),
            'truncated': sum(bool(request.get('truncated')) for request in requests),
            'malformed': sum('<tool_call>' in request['raw_response'] and not request['tool_calls'] for request in requests),
            'seconds': result.get('elapsed_seconds'),
            'source': {'records_sha256': sha256(folder / 'records.jsonl'), 'config_sha256': sha256(folder / 'config.json')}}


def update(run):
    current = metrics(run)
    teachers = [metrics(path.stem) for path in sorted((ROOT / 'experiments/E15/runs').glob('*.json'))
                if read(path).get('validity') == 'valid_harness']
    if run not in [item['run'] for item in teachers]:
        teachers.append(current)
    for item in teachers:
        student = read(ROOT / 'experiments/E14/runs' / f"{item['config']['source_eval_run']}.json")
        assert student.get('validity') == 'valid_harness' and student['input_delivery_verified']
        item['student'] = student
    config, rows = current['config'], current['rows']
    complete = [item for item in teachers if item['seconds'] is not None]
    conclusion = ('已完成的教师条件均为0/16，与各自学生参照相比增加0个百分点；这批任务尚未观察到教师更可靠。教师回复不能直接当作优质标签，训练场景仍需逐条执行并筛选。'
                  if complete and all(item['passed'] == 0 for item in complete) else
                  '教师表现以同提示、同任务的实际对照为准；未完成任务保留在16题分母中。')
    lines = ['# E15：4B 教师能否更可靠地完成工具任务', '',
             '本机Qwen3-4B基础模型以NF4量化接入Pi，与Qwen3-1.7B领域适配器使用同一冻结dev16、同一工具和解码预算。每个教师条件绑定使用相同提示的学生运行。任务成绩由实际文件状态、检查结果和最终回答决定。', '',
             '## 同条件对照', '', '| 提示条件 | 学生 | 学生成绩 | 教师 | 教师成绩 | 教师耗时 |', '| --- | --- | ---: | --- | ---: | ---: |']
    for item in teachers:
        student = item['student']
        label = '简短工具提示' if item['config'].get('condition') == 'pi_focused_teacher' else '原few-shot提示'
        score = f"{item['passed']}/16" if item['result'].get('status') == 'completed' else f"已评{len(item['rows'])}/16，通过{item['passed']}"
        elapsed = f"{item['seconds']}秒" if item['seconds'] is not None else '进行中'
        lines.append(f"| {label} | {student['run_id']} | {student['passed_tasks']}/16 | {item['run']} | {score} | {elapsed} |")
    lines += ['', conclusion, '', '## 预算与错误记录', '',
              f"输入上限{config['context_window']} token，生成上限{config['max_new_tokens']} token；温度{config['temperature']}、推理seed{config['inference_seed']}，关闭thinking。每题最多{config['max_tool_calls']}次工具调用、{config['task_timeout_seconds']}秒。预算耗尽立即结束并记失败，超时和截断也保留。", '',
              '| 教师运行 | API请求 | 任务超时 | 预算耗尽 | 截断回复 | 调用解析失败 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for item in teachers:
        lines.append(f"| {item['run']} | {item['requests']} | {item['timeouts']} | {item['budget']} | {item['truncated']} | {item['malformed']} |")
    lines += ['', '调用解析失败统计的是回复中出现工具调用标记、但没有任何调用成功解析的请求数。']
    figure = ROOT / 'experiments/E15/figures/E15-1.png'
    if rows:
        figure.parent.mkdir(parents=True, exist_ok=True)
        font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout='constrained')
        totals = Counter(row['category'] for row in rows)
        passed = Counter(row['category'] for row in rows if row.get('passed'))
        values = [passed[key] for key in LABELS]
        axes[0].barh(range(8), values, color='#2f855a', label='通过')
        axes[0].barh(range(8), [totals[key] - passed[key] for key in LABELS], left=values, color='#cbd5e0', label='未通过')
        axes[0].set_yticks(range(8), LABELS.values(), fontproperties=font)
        axes[0].set_xlim(0, 2)
        axes[0].set_xticks([0, 1, 2])
        axes[0].set_xlabel('每类任务数', fontproperties=font)
        axes[0].set_title(f"{run}：已评{len(rows)}/16，通过{current['passed']}", fontproperties=font)
        axes[0].invert_yaxis()
        axes[0].legend(prop=font, loc='lower right')
        labels, times = [], []
        for item in complete:
            student = metrics(item['student']['run_id'])
            labels += [f"学生 {student['run']}", f"教师 {item['run']}"]
            times += [student['seconds'] / 60, item['seconds'] / 60]
        axes[1].barh(range(len(times)), times, color=['#718096', '#2b6cb0'] * len(complete))
        axes[1].set_yticks(range(len(times)), labels, fontproperties=font)
        for index, value in enumerate(times):
            axes[1].text(value + .1, index, f'{value:.2f}', va='center')
        axes[1].set_xlim(0, max(times, default=1) * 1.3)
        axes[1].set_xlabel('完整16题耗时（分钟）', fontproperties=font)
        axes[1].set_title('同提示、同任务的实测耗时', fontproperties=font)
        axes[1].invert_yaxis()
        fig.savefig(figure, dpi=300)
        plt.close(fig)
        write_json(figure.with_suffix('.source.json'), {'runs': {item['run']: item['source'] for item in teachers},
                   'student_runs': {item['student']['run_id']: metrics(item['student']['run_id'])['source'] for item in teachers},
                   'figure_sha256': sha256(figure)})
        lines += ['', '![图 E15-1：当前教师八类实际计数与已完成配对评测耗时](figures/E15-1.png)', '']
    original = next((item for item in teachers if item['run'] == 'E15-R03'), None)
    if original:
        failure = next(row for row in original['rows'] if row['category'] == 'failure_recovery')
        lines += ['## 错误发生在哪一步', '',
                  '原提示教师在缺信息任务中没有先读取已有配置，就要求用户补充文件里已有的信息；修复时修改受保护的检查文件；路径定位时猜测不存在的文件。部分工具参数的JSON字符串引号没有正确转义，导致调用无法解析。这分别属于任务策略、文件边界和调用格式问题。', '',
                  f"实际任务 `{failure['task_id']}` 未通过的判据为：{'；'.join(failure['judgement']['reasons'])}。判据检查修复后的文件与测试，模型解释不能代替修复。", '',
                  '```python', 'accepted = task_passed and schema_valid and arguments_valid',
                  'accepted = accepted and not (timed_out or truncated)', '```', '']
    lines += ['## 阶段结论', '', conclusion, '',
              '教师使用本地固定revision和模型自带的Qwen3模板，没有经过本项目的LoRA训练。学生、教师模板一致。旧E15-R02的文本块接线无效，原始记录保留且不参与能力比较；有效条件逐题核对任务提示与模型输入哈希。', '',
              '## 证据索引', '', '| 内容 | 对应记录 |', '| --- | --- |']
    for item in teachers:
        lines.append(f"| {item['run']}逐题执行与本机生成 | `.local/runs/{item['run']}/records.jsonl`；`model-api.jsonl`；`tasks.json`；`config.json`；`result.json` |")
    lines += ['| 教师来源和条件 | `.local/models/Qwen3-4B/download-manifest.json`；`configs/pi-agent-e15-teacher-dev.json`；`configs/pi-agent-e15-teacher-focused-dev.json` |',
              '| 任务、提示和输入审计 | `configs/subsets-frozen.json`；`configs/prompt-frozen.json`；`configs/pi-focused-prompt.json`；`experiments/E13/prompt-delivery-audit.json` |',
              '| 图表来源 | `experiments/E15/figures/E15-1.source.json` |', '']
    (ROOT / 'experiments/E15/notes.md').write_text('\n'.join(lines), encoding='utf-8')
    summary = ROOT / 'docs/reports/summaries/03.md'
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text('# 教师与蒸馏阶段总结\n\n' + conclusion + '\n\n'
                       '各条件保留自己的提示、任务分母、耗时和错误。未执行的蒸馏训练与规模比较不计为结果。\n\n'
                       '## 证据索引\n\n| 内容 | 对应记录 |\n| --- | --- |\n'
                       '| 教师对照 | `experiments/E15/notes.md`；`experiments/E15/runs/` |\n'
                       '| 输入交付与图表来源 | `experiments/E13/prompt-delivery-audit.json`；`experiments/E15/figures/E15-1.source.json` |\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    assert args.run.startswith('E15-R') and '/' not in args.run and '\\' not in args.run
    update(args.run)
