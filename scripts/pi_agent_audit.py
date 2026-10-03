"""核对已完成的模型 Agent 分母、提示交付和原始文件哈希。"""
import argparse
import hashlib
import json
from collections import Counter

from lab import ROOT, sha256, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(run_id):
    directory = ROOT / '.local/runs' / run_id
    result = read(directory / 'result.json')
    config = read(directory / 'config.json')
    rows = [json.loads(line) for line in (directory / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
    tasks = read(directory / 'tasks.json')
    trace = [json.loads(line) for line in (directory / 'model-api.jsonl').read_text(encoding='utf-8').splitlines()]
    assert result['status'] == 'completed'
    frozen = read(ROOT / 'configs/subsets-frozen.json')[config['frozen_subset']]
    assert config['task_ids'] == frozen['ids'] and config['ids_sha256'] == frozen['ids_sha256']
    assert sha256(ROOT / config['prompt_config']) == config['prompt_sha256']
    if config['experiment'] == 'E15':
        assert sha256(ROOT / config['model_manifest_path']) == config['model_manifest_sha256']
        assert config['tokenizer_template_sha256'] == config['student_tokenizer_template_sha256']
    else:
        assert sha256(ROOT / config['adapter'] / 'adapter_model.safetensors') == config['adapter_sha256']
    assert len(rows) == len(tasks) == result['target_tasks'] == result['evaluated_tasks']
    assert [row['task_id'] for row in rows] == config['task_ids'] == [task['task_id'] for task in tasks]
    assert Counter(row['category'] for row in rows) == config['category_quotas']
    assert sum(row['passed'] for row in rows) == result['passed_tasks']
    for name, key in [('records.jsonl', 'records_sha256'), ('tasks.json', 'tasks_sha256'),
                      ('model-api.jsonl', 'model_api_trace_sha256')]:
        assert sha256(directory / name) == result[key]
    pairs = set()
    for row, task in zip(rows, tasks, strict=True):
        assert row['container_removed']
        assert row['isolation'] == {'network': 'none', 'readonly_root': True, 'user': '1000:1000', 'binds': [], 'gpu_requests': []}
        assert row['task_prompt_sha256'] == hashlib.sha256(task['prompt'].encode()).hexdigest()
        assert row['model_requests']
        assert row['model_input_prompt_sha256'] == row['model_requests'][0]['prompt_sha256']
        pairs.add((row['task_prompt_sha256'], row['model_input_prompt_sha256']))
    assert len(pairs) == len({a for a, _ in pairs}) == len({b for _, b in pairs})
    calls = [call for row in rows for call in row['observed_tool_calls']]
    details = {'tasks': len(rows), 'unique_task_prompts': len(pairs), 'unique_model_input_prompts': len(pairs),
               'api_requests': len(trace), 'prompt_hash_match_per_task': True,
               'records_sha256': result['records_sha256'], 'model_api_trace_sha256': result['model_api_trace_sha256'],
               'result_sha256': sha256(directory / 'result.json'), 'eligible_for_model_comparison': True}
    path = ROOT / 'experiments/E13/prompt-delivery-audit.json'
    saved = read(path)
    saved['corrected_run_verification'][run_id] = details
    write_json(path, saved)
    public = ROOT / 'experiments' / config['experiment'] / 'runs' / f'{run_id}.json'
    summary = read(public)
    summary.update(validity='valid_harness', input_delivery_verified=True,
                   timed_out_tasks=sum(bool(row['agent_timed_out']) for row in rows),
                   tool_budget_exhausted_tasks=sum(bool(row.get('tool_budget_exhausted')) for row in rows),
                   observed_tool_requests=len(calls),
                   budget_rejections=sum(any('任务超过工具调用预算' in block.get('text', '')
                                             for block in call.get('result', {}).get('content', [])) for call in calls),
                   non_timeout_failed_tasks=sum(not row['passed'] and not row['agent_timed_out'] for row in rows))
    write_json(public, summary)
    print(json.dumps({'run_id': run_id, **details}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+')
    for run_id in parser.parse_args().runs:
        audit(run_id)
