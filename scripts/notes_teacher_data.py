"""依据逐条真实执行记录更新教师轨迹接受率。"""
import argparse
import hashlib
import json
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from lab import ROOT, sha256, write_json


def update(run):
    folder = ROOT / '.local/runs' / run
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    records_snapshot = (folder / 'records.jsonl').read_bytes()
    rows = [json.loads(line) for line in records_snapshot.decode('utf-8').splitlines()]
    accepted = sum(row['teacher_filter']['accepted'] for row in rows)
    reasons = Counter(reason for row in rows for reason in row['teacher_filter']['reasons'])
    ended = (folder / 'result.json').exists()
    lines = ['# E16：教师轨迹的执行与筛选', '',
             f"教师处理冻结train请求。首批{config['initial_requests']}条覆盖八类、16个训练模板族；有效量不足{config['valid_target']}条时，每批扩展{config['extension_batch']}条，最多{config['maximum_requests']}条。模型输入只有任务提示、工具定义和实际工具返回，不包含参考解。", '',
             f"实际已处理{len(rows)}条，执行与格式初筛接受{accepted}条。所有拒绝、截断、超时和预算耗尽均保留；初筛轨迹还要经过完整历史的token编码与监督边界检查，才能计为可训练样本。", '',
             '## 每条轨迹检查什么', '',
             '每题在新的无网络容器中执行。最终状态通过、工具名和参数schema正确、调用与返回一一对应且没有截断或超时的轨迹，才进入完整编码。发生错误后成功恢复的过程可以保留；JSON正确而任务失败仍被拒绝。', '',
             '```python', 'eligible = task_passed and valid_schema and matched_results',
             'eligible = eligible and not (timeout or truncated or budget_exhausted)', '```', '',
             '| 拒绝原因 | 涉及轨迹数 |', '| --- | ---: |']
    lines += [f'| {reason} | {count} |' for reason, count in sorted(reasons.items())]
    figure = ROOT / 'experiments/E16/figures/E16-1.png'
    if rows:
        figure.parent.mkdir(parents=True, exist_ok=True)
        font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
        running, cumulative = 0, []
        for row in rows:
            running += row['teacher_filter']['accepted']
            cumulative.append(running)
        fig, axis = plt.subplots(figsize=(8, 4), layout='constrained')
        axis.plot(range(1, len(rows) + 1), cumulative, color='#2b6cb0')
        axis.set_xlabel('已处理训练请求', fontproperties=font)
        axis.set_ylabel('累计初筛接受轨迹', fontproperties=font)
        axis.set_title('真实执行记录的累计接受数量', fontproperties=font)
        axis.set_ylim(bottom=0)
        fig.savefig(figure, dpi=300)
        plt.close(fig)
        write_json(figure.with_suffix('.source.json'), {'run_id': run, 'records_sha256': hashlib.sha256(records_snapshot).hexdigest(),
                   'figure_sha256': sha256(figure), 'requests': len(rows), 'accepted': accepted})
        lines += ['', '![图 E16-1：逐条执行的累计初筛接受数量](figures/E16-1.png)', '']
    inspection = ROOT / 'experiments/E16/encoding-inspection.json'
    if inspection.exists():
        checked = json.loads(inspection.read_text(encoding='utf-8'))
        if checked['run_id'] == run:
            lines += ['## Pi 消息怎样变成训练单元', '',
                      'Pi 的原生 system 消息可以只有空 content，实际提示保存在结构化 sections 中。转换时使用同版本 Pi 渲染器恢复完整系统提示，包括工作目录；不能把空 content 当成空提示，也不能把 system 角色当作用户消息。', '',
                      f"对已落盘的{checked['requests']}条请求快照检查，初筛接受的{checked['preliminary_accepted']}条均完整编码成功，展开为{checked['assistant_units']}个当前回复训练单元，共{checked['supervised_tokens']}个监督token；最长输入{checked['max_input_tokens']} tokens。这是运行中快照，尚未达到256条目标。", '',
                      f"{checked['prefix_checks']}个推理前缀的token哈希与实际API记录逐一相同，规范化消息哈希也相同；历史消息与工具返回不参与当前回复loss。检查没有初始化CUDA，不占用教师推理GPU。", '',
                      '```python', 'history = messages[:target_message_index]',
                      'assert sha256(render_tokens(history)) == request["prompt_sha256"]',
                      'labels[:target_start] = -100', '```', '']
    exported = folder / 'teacher-export.json'
    if exported.exists():
        data = json.loads(exported.read_text(encoding='utf-8'))
        lines += ['## 编码结果', '', f"完整编码后接受{data['accepted']}条，拒绝{data['rejected']}条；256条目标{'达到' if data['target_met'] else '尚未达到'}。只有真实有效数量足够时才导出128/256条训练文件，较小集合为较大集合的前缀。", '']
    lines += ['## 阶段结论', '',
              f"{'请求执行已结束' if ended else '请求仍在顺序执行'}；当前初筛{accepted}/{len(rows)}。这是训练请求的接受率，不能作为dev或test成功率。", '',
              '## 证据索引', '', '| 内容 | 对应记录 |', '| --- | --- |',
              f'| 冻结请求与教师条件 | `configs/teacher-requests-frozen.json`；`.local/runs/{run}/config.json` |',
              f'| 逐题执行与本机生成 | `.local/runs/{run}/records.jsonl`；`model-api.jsonl`；`tasks.json` |']
    if exported.exists():
        lines.append(f'| 编码、拒绝与训练文件清单 | `.local/runs/{run}/teacher-export.json`；`teacher-rejections.json`；`teacher-units.json` |')
    if inspection.exists():
        evidence = checked['evidence']
        lines.append(f"| 编码快照核对 | `experiments/E16/encoding-inspection.json`；`{evidence['snapshot']}`；`{evidence['inspection']}` |")
    lines.append('')
    (ROOT / 'experiments/E16').mkdir(exist_ok=True)
    (ROOT / 'experiments/E16/notes.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    assert args.run.startswith('E16-R') and '/' not in args.run and '\\' not in args.run
    update(args.run)
