"""记录专项训练的真实预算、loss 与同提示 Pi 对照。"""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from lab import ROOT, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--pi-run')
    args = parser.parse_args()
    run = ROOT / '.local/runs' / args.run
    progress = json.loads((run / 'progress.json').read_text(encoding='utf-8')) if (run / 'progress.json').exists() else {}
    result = json.loads((run / 'result.json').read_text(encoding='utf-8')) if (run / 'result.json').exists() else {}
    manifest = json.loads((ROOT / 'configs/pi-focused-data.json').read_text(encoding='utf-8'))
    lines = ['## 针对性工具微调', '',
             '专项集取自已执行通过的Pi训练参考：16个训练模板族各4条，合计64条；八类各8条。工具调用、原始返回和最终答案保持一致，只把系统提示统一为推理时使用的简短Pi工具示例。原始训练集与dev、test划分保持各自来源。', '',
             '从E14-R19适配器开始，rank16、seed17、学习率5e-5、梯度累积8，完整轨迹按当前assistant回复展开，训练1个epoch。固定100条公开dev保留初始与末次token加权loss；Pi成功率使用固定16条dev，短提示下的原适配器与新适配器分别运行。', '']
    lines += [f"64条轨迹展开为{manifest['train']['assistant_units']}个回复单元，最长{manifest['train']['max_sequence_length']}token；工具选择、参数、错误返回后的下一步，以及最终回答都在监督中。", '',
              '### 训练是否实际更新？', '']
    if result.get('status') == 'trained_pending_tool_eval':
        lines += [f"{args.run}已更新{result['steps']}步；选中的{result['selected_checkpoint']}公开dev loss为{result['selected_dev_loss']:.6f}。任务效果以实际Pi评测为准。", '']
        audit_path = ROOT / 'experiments/E14/focused-training-audit.json'
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding='utf-8'))
            lines += [f"训练耗时{audit['train_runtime_seconds']:.1f}秒，{audit['changed_adapter_tensors']}/{audit['adapter_tensors']}个LoRA张量发生实际变化，梯度和loss均有限。固定100条公开dev loss从{audit['initial_public_dev_loss']:.6f}变为{audit['final_public_dev_loss']:.6f}，没有下降；专项训练loss低不能替代独立任务验证。", '']
    elif progress:
        lines += [f"{args.run}当前更新{progress.get('latest', {}).get('step', 0)}/{progress.get('max_steps', 0)}步，尚无完整任务效果结论。", '']
    metrics_path = run / 'metrics.jsonl'
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding='utf-8').splitlines()] if metrics_path.exists() else []
    if metrics:
        figure = ROOT / 'experiments/E14/figures/focused-training-loss.png'
        font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
        fig, axis = plt.subplots(figsize=(6.4, 3.4), layout='constrained')
        for key, label, color in [('loss', '专项训练 loss', '#2563EB'), ('eval_loss', '公开 dev loss', '#059669')]:
            points = [row for row in metrics if key in row]
            if points:
                axis.plot([row['step'] for row in points], [row[key] for row in points], color=color, marker='o' if key == 'eval_loss' else None, label=label)
        axis.set_xlabel('optimizer step')
        axis.set_ylabel('token 加权交叉熵', fontproperties=font)
        axis.set_title('专项工具微调的实际 train/dev loss', fontproperties=font)
        axis.grid(color='#E5E7EB', linewidth=0.6)
        axis.legend(prop=font)
        fig.savefig(figure, dpi=300)
        plt.close(fig)
        write_json(figure.with_suffix('.source.json'), {'metrics_sha256': sha256(metrics_path), 'image_sha256': sha256(figure), 'training_run': args.run})
        lines += ['![专项工具训练的真实loss曲线。](figures/focused-training-loss.png)', '']
    evaluations = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((ROOT / 'experiments/E14/runs').glob('*.json'))]
    matched = [item for item in evaluations if item.get('status') == 'completed' and item.get('config', {}).get('condition') in ['pi_focused_baseline', 'pi_focused_sft']]
    lines += ['### 同提示下能否完成新任务？', '', '| 模型 | 运行 | Pi dev | 耗时 | 超时 | 工具预算耗尽 |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for item in matched:
        rows = [json.loads(line) for line in (ROOT / '.local/runs' / item['run_id'] / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
        label = '原领域适配器' if item['config']['condition'] == 'pi_focused_baseline' else '专项适配器'
        lines.append(f"| {label} | {item['run_id']} | {item['passed_tasks']}/{item['target_tasks']} | {item['elapsed_seconds']}秒 | {sum(bool(row['agent_timed_out']) for row in rows)} | {sum(bool(row.get('tool_budget_exhausted')) for row in rows)} |")
    lines += ['', '两组使用相同短提示、任务ID、seed17和12次工具调用上限；dev包含独立模板族。完成训练或降低loss，都需要通过这组执行结果验证迁移效果。', '']
    if len(matched) == 2:
        before, after = matched
        gain = after['passed_tasks'] - before['passed_tasks']
        lines += [f"专项微调新增通过{gain}个任务，成功率变化{gain / 16 * 100:+.1f}个百分点。" + ('本轮未证明独立模板上的能力提升，不能把训练loss下降称为任务效果改善。' if gain <= 0 else '这个结论只覆盖冻结16条dev，最终test仍使用独立分母。'), '']
    lines += ['### 一条实际失败说明什么？', '',
              '短提示原模型在目标租户缺失的任务中先读取不存在的`src/config.mjs`，再多次运行不存在的`checks.mjs`，最终耗尽工具预算。正确的决策需要识别用户缺失的信息并询问；重复执行同一失败命令没有补充信息。该轨迹说明训练模板中的恢复动作不能直接等同于新仓库上的恢复能力。', '',
              '```python', '# 专项数据只复用执行通过的训练参考', "assert record['split'] == 'train'", "assert record['validation']['execution'] == 'reference_passed'", '```', '']
    lines += ['### 证据索引', '', '- configs/pi-focused-data.json；configs/pi-focused-prompt.json；configs/sft-pi-focused.json',
              f'- experiments/E14/runs/{args.run}.json；.local/runs/{args.run}/metrics.jsonl',
              '- configs/pi-agent-focused-baseline-dev.json；scripts/pi_agent_audit.py',
              '- experiments/E14/runs/E14-R30.json；experiments/E14/runs/E14-R31.json；.local/data/processed/E14-focused-tools/train-64.jsonl', '']
    if (ROOT / 'experiments/E14/focused-training-audit.json').exists():
        lines.insert(len(lines) - 1, '- experiments/E14/focused-training-audit.json；.local/runs/E14-R32/trainable-parameters.json')
    if args.pi_run:
        lines.insert(len(lines) - 1, f'- experiments/E14/runs/{args.pi_run}.json；.local/runs/{args.pi_run}/records.jsonl')
    if (ROOT / 'configs/pi-agent-focused-trained-dev.json').exists():
        lines.insert(len(lines) - 1, '- configs/pi-agent-focused-trained-dev.json')
    path = ROOT / 'experiments/E14/notes.md'
    existing = path.read_text(encoding='utf-8').split('## 针对性工具微调', 1)[0].rstrip()
    path.write_text(existing + '\n\n' + '\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
