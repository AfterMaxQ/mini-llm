"""只检查 dev 参考历史的长度，不生成训练文件，也不加载模型。"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter

import torch
from lab import ROOT, start_run, finish_run, sha256, write_json
from templates import tokenizer_and_template
from pi_reference_data import check_arguments, digest


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); args = parser.parse_args()
    source = ROOT / '.local/runs' / args.source
    result = json.loads((source / 'result.json').read_text(encoding='utf-8'))
    assert result['status'] == 'dev_reference_verified' and result['task_count'] == 40
    for name in ['records', 'tasks']:
        assert sha256(source / (name + '.json')) == result[name + '_sha256']
    records = json.loads((source / 'records.json').read_text(encoding='utf-8'))
    tasks = json.loads((source / 'tasks.json').read_text(encoding='utf-8'))
    assert sha256(ROOT / result['frozen_dev_file']) == result['frozen_dev_sha256']
    system = json.loads((ROOT / 'configs/pi-reference-data.json').read_text(encoding='utf-8'))['system_prompt']
    installed = json.loads((ROOT / 'configs/pi-tools.json').read_text(encoding='utf-8'))
    tools = [{'type': 'function', 'function': t} for t in installed['tools']]
    definitions = {t['function']['name']: t['function']['parameters'] for t in tools}
    config = {'operation': 'pi_dev_lengths', 'source_run': args.source, 'model': None,
              'command': [sys.executable, *sys.argv], 'system_prompt': system,
              'source_records_sha256': result['records_sha256'], 'source_tasks_sha256': result['tasks_sha256'],
              'frozen_dev_sha256': result['frozen_dev_sha256'], 'tools_schema_sha256': digest(tools),
              'context_tokens': 8192, 'enable_thinking': False,
              'scope': '40个dev场景的已执行参考历史长度；不生成训练标签，不代表模型轨迹长度'}
    run = start_run('E14', config)
    try:
        tokenizer, original, _ = tokenizer_and_template()
        rows = []
        for task, record in zip(tasks, records):
            assert task['task_id'] == record['task_id'] and task['split'] == record['split'] == 'dev'
            assert record['reference']['passed'] and not record['initial']['passed'] and not record['negative']['passed']
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': task['prompt']}]
            ids = []
            def measure(reply, kind):
                prefix = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                    enable_thinking=False, add_generation_prompt=True)
                full = tokenizer.apply_chat_template([*messages, reply], tools=tools, tokenize=False,
                    enable_thinking=False, add_generation_prompt=False)
                assert full.startswith(prefix)
                prompt_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
                full_ids = tokenizer(full, add_special_tokens=False)['input_ids']
                rows.append({'task_id': task['task_id'], 'category': task['category'], 'split': 'dev',
                    'reply_index': len(ids), 'kind': kind, 'prompt_tokens': len(prompt_ids),
                    'reference_full_tokens': len(full_ids), 'full_text_sha256': hashlib.sha256(full.encode()).hexdigest()})
                messages.append(reply)
            for event, action in zip(record['reference_events'], task['reference'], strict=True):
                assert event['tool'] == action['tool'] and event['input'] == action['input']
                check_arguments(event['input'], definitions[event['tool']])
                call_id = event['call_id']; assert call_id not in ids
                measure({'role': 'assistant', 'content': '', 'tool_calls': [{'id': call_id, 'type': 'function',
                    'function': {'name': event['tool'], 'arguments': event['input']}}]}, 'tool_call')
                ids.append(call_id)
                if 'error' in event:
                    assert event['is_error']; content = event['error']
                else:
                    blocks = event['result']['content']; assert all(b['type'] == 'text' for b in blocks)
                    content = '\n'.join(b['text'] for b in blocks)
                messages.append({'role': 'tool', 'name': event['tool'], 'tool_call_id': call_id, 'content': content})
            measure({'role': 'assistant', 'content': record['reference_answer']}, 'final_answer')
            assert ids == [m['tool_call_id'] for m in messages if m['role'] == 'tool']
        assert len(rows) == result['reference_tool_calls'] + 40
        assert not torch.cuda.is_initialized()
        assert max(r['reference_full_tokens'] for r in rows) <= config['context_tokens']
        file = run / 'lengths.csv'
        with file.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        summary = {'status': 'dev_lengths_verified', 'exit_code': 0, 'operation': 'pi_dev_lengths',
            'task_count': 40, 'reply_points': len(rows), 'model_calls': 0, 'cuda_initialized': False,
            'max_prompt_tokens': max(r['prompt_tokens'] for r in rows),
            'max_reference_full_tokens': max(r['reference_full_tokens'] for r in rows),
            'points_over_8192': sum(r['reference_full_tokens'] > 8192 for r in rows),
            'categories': dict(Counter(t['category'] for t in tasks)), 'lengths_sha256': sha256(file),
            'template_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'source_run': args.source, 'frozen_dev_sha256': result['frozen_dev_sha256'], 'scope': config['scope']}
        finish_run(run, summary); print(json.dumps(summary, ensure_ascii=False))
    except Exception:
        import traceback
        finish_run(run, {'status': 'failed', 'exit_code': 1, 'operation': 'pi_dev_lengths', 'error': traceback.format_exc()})
        raise


if __name__ == '__main__': main()
