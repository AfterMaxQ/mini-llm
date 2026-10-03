"""冻结教师训练请求，并将真实执行通过的轨迹转换为学生监督数据。"""
import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict

from lab import ROOT, sha256, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def freeze():
    source = read(ROOT / '.local/runs/E14-R07/result.json')
    budget = read(ROOT / 'configs/experiment-budget.json')
    assert source['status'] == 'training_requests_frozen'
    raw = ROOT / source['requests_file']
    assert sha256(raw) == source['requests_sha256']
    tasks = [json.loads(line) for line in raw.read_text(encoding='utf-8').splitlines()]
    groups = defaultdict(list)
    for task in tasks:
        assert task['split'] == 'train'
        groups[task['template_family']].append(task)
    assert len(groups) == 16
    teacher = budget['teacher']
    for group in groups.values():
        group.sort(key=lambda task: hashlib.sha256(f"{budget['sampling_seed']}:{task['task_id']}".encode()).hexdigest())
    selected = [groups[family][index] for index in range(teacher['maximum_requests'] // 16)
                for family in sorted(groups)]
    manifest = {'version': 1, 'sampling_seed': budget['sampling_seed'], 'source_run': 'E14-R07',
                'source_file': source['requests_file'], 'source_sha256': sha256(raw),
                **{key: teacher[key] for key in ['initial_requests', 'extension_batch', 'maximum_requests', 'valid_target']},
                'ids': [task['task_id'] for task in selected],
                'strata': dict(Counter(task['category'] for task in selected)),
                'family_counts': dict(Counter(task['template_family'] for task in selected)),
                'initial_strata': dict(Counter(task['category'] for task in selected[:teacher['initial_requests']])),
                'selection': '按训练模板族分层，种子哈希排序后轮流取样；仅使用train场景，不读取模型成绩'}
    path = ROOT / 'configs/teacher-requests-frozen.json'
    if path.exists():
        assert read(path) == manifest, '冻结请求不可覆盖'
    else:
        write_json(path, manifest)
    print(json.dumps({'manifest': path.relative_to(ROOT).as_posix(), 'sha256': sha256(path),
                      'maximum_requests': len(selected), 'initial_strata': manifest['initial_strata']}, ensure_ascii=False))


def system_prompt(config):
    custom = read(ROOT / 'configs/pi-reference-data.json')['system_prompt'] + '\n\n' + read(ROOT / config['prompt_config'])['suffix']
    node = config['command'][0]
    script = ("import{readFileSync}from'node:fs';import path from'node:path';"
              "import{buildSystemPrompt}from'./.local/pi/node_modules/@earendil-works/pi-coding-agent/dist/core/system-prompt.js';"
              "process.stdout.write(buildSystemPrompt({customPrompt:readFileSync(0,'utf8'),cwd:path.resolve('/workspace')}));")
    # Pi 把系统提示保存在 sections 中；使用同版本渲染器恢复实际发送的文本。
    return subprocess.check_output([node, '--input-type=module', '-e', script], cwd=ROOT,
                                   input=custom.encode('utf-8')).decode('utf-8')


def convert(run_id, snapshot=None):
    from pi_agent_api import normalize_messages
    from pi_reference_data import check_arguments, digest
    from templates import action_records, encode_action, tokenizer_and_template
    folder = ROOT / '.local/runs' / run_id
    config = read(folder / 'config.json')
    assert config['experiment'] == 'E16'
    if snapshot is None:
        result = read(folder / 'result.json')
        assert result['status'] == 'completed'
        for name, key in [('records.jsonl', 'records_sha256'), ('tasks.json', 'tasks_sha256'),
                          ('model-api.jsonl', 'model_api_trace_sha256')]:
            assert sha256(folder / name) == result[key]
    raw = snapshot if snapshot is not None else folder / 'records.jsonl'
    rows = [json.loads(line) for line in raw.read_text(encoding='utf-8').splitlines()]
    tasks = read(folder / 'tasks.json')
    if snapshot is None:
        assert len(rows) == result['evaluated_tasks']
    assert [row['task_id'] for row in rows] == [task['task_id'] for task in tasks[:len(rows)]]
    tools = [{'type': 'function', 'function': tool} for tool in read(ROOT / 'configs/pi-tools.json')['tools']]
    definitions = {tool['function']['name']: tool['function'] for tool in tools}
    schema_hash = digest(tools)
    tokenizer, original, marked = tokenizer_and_template()
    prompt = system_prompt(config)
    records, rejections, units = [], [], []
    for row, task in zip(rows, tasks):
        reasons = list(row['teacher_filter']['reasons'])
        record = None
        if not reasons:
            try:
                assert row['passed'] and row['container_removed']
                assert row['task_prompt_sha256'] == hashlib.sha256(task['prompt'].encode()).hexdigest()
                assert row['model_input_prompt_sha256'] == row['model_requests'][0]['prompt_sha256']
                messages = [{'role': 'system', 'content': prompt}]
                for index, event in enumerate(row['agent_events']):
                    if event['role'] == 'system':
                        assert index == 0 and event['content'] in ('', prompt), 'unexpected_system_message'
                        continue
                    blocks = event['content']
                    assert isinstance(blocks, list), 'expected_native_text_blocks'
                    text = ''.join(block['text'] for block in blocks if block['type'] == 'text')
                    if event['role'] == 'assistant':
                        calls = []
                        for call in [block for block in blocks if block['type'] == 'toolCall']:
                            check_arguments(call['arguments'], definitions[call['name']]['parameters'])
                            calls.append({'id': call['id'], 'type': 'function',
                                          'function': {'name': call['name'], 'arguments': call['arguments']}})
                        messages.append({'role': 'assistant', 'content': text, **({'tool_calls': calls} if calls else {})})
                    elif event['role'] == 'toolResult':
                        messages.append({'role': 'tool', 'name': event['toolName'],
                                         'tool_call_id': event['toolCallId'], 'content': text})
                    else:
                        assert event['role'] == 'user' and text == task['prompt']
                        messages.append({'role': 'user', 'content': text})
                assert messages[-1]['role'] == 'assistant' and not messages[-1].get('tool_calls')
                assert messages[-1]['content'].strip(), 'missing_final_answer'
                record = {'sample_id': task['task_id'], 'split': 'train', 'category': task['category'],
                          'template_family': task['template_family'], 'repository_family': task['repository_family'],
                          'group_id': digest([task['template_family'], task['repository_family']]),
                          'source': 'minillm_pi_teacher', 'source_run_id': run_id, 'source_id': task['task_id'],
                          'source_revision': config['code_snapshot_sha256'], 'teacher_run_id': run_id,
                          'license': None, 'license_note': '项目自编任务，当前仓库未声明数据许可',
                          'messages': messages, 'tools': tools, 'tools_sha256': schema_hash,
                          'reference_answer_source': 'executed_teacher_trajectory',
                          'validation': {'format': True, 'arguments': True, 'execution': 'teacher_passed'}}
                current_units = []
                actions = action_records(record)
                assert len(actions) == len(row['model_requests']), 'unmatched_model_request'
                for action, request in zip(actions, row['model_requests']):
                    encoded = encode_action(tokenizer, marked, action)
                    text = tokenizer.apply_chat_template(action['messages'], tools=tools, tokenize=False,
                                                         enable_thinking=False, add_generation_prompt=False)
                    tagged = tokenizer.apply_chat_template(action['messages'], tools=tools, chat_template=marked,
                                                           tokenize=False, enable_thinking=False, add_generation_prompt=False)
                    prefix = tokenizer.apply_chat_template(action['messages'][:-1], tools=tools, tokenize=False,
                                                           enable_thinking=False, add_generation_prompt=True)
                    assert text == tagged and text.startswith(prefix)
                    assert tokenizer(text, add_special_tokens=False)['input_ids'] == encoded['input_ids']
                    prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
                    assert hashlib.sha256(json.dumps(prefix_ids, separators=(',', ':')).encode()).hexdigest() == request['prompt_sha256'], 'inference_prefix_tokens_mismatch'
                    history_hash = hashlib.sha256(json.dumps(normalize_messages(action['messages'][:-1]), ensure_ascii=False,
                                                             sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    assert history_hash == request['messages_sha256'], 'inference_history_mismatch'
                    mask = encoded['assistant_masks']
                    start = next(index for index, value in enumerate(mask) if value)
                    assert all(label == -100 for label in encoded['labels'][:start])
                    assert len(encoded['input_ids']) <= config['context_window'], 'full_history_over_context'
                    targets = sum(label != -100 for label in encoded['labels'][1:])
                    assert targets > 0 and tokenizer.eos_token_id in encoded['labels']
                    current_units.append({'sample_id': task['task_id'], 'message_index': action['target_message_index'],
                                          'input_tokens': len(encoded['input_ids']), 'supervised_tokens': targets,
                                          'inference_prefix_sha256': request['prompt_sha256']})
                assert current_units
                record['token_count'] = max(unit['input_tokens'] for unit in current_units)
                record['target_token_count'] = sum(unit['supervised_tokens'] for unit in current_units)
                units.extend(current_units)
            except (AssertionError, KeyError, TypeError, ValueError) as error:
                reasons.append('encoding_or_protocol:' + str(error))
        if reasons:
            rejections.append({'task_id': task['task_id'], 'reasons': reasons})
        else:
            records.append(record)
    import torch
    assert not torch.cuda.is_initialized(), '轨迹编码不得占用GPU'
    return config, rows, records, rejections, units, prompt, original, schema_hash, sha256(raw)


def export(run_id):
    config, rows, records, rejections, units, prompt, original, schema_hash, records_hash = convert(run_id)
    folder = ROOT / '.local/runs' / run_id
    output = ROOT / '.local/data/processed' / run_id
    output.mkdir(parents=True, exist_ok=True)
    selected = records[:config['valid_target']]
    paths = []
    for size in [128, 256]:
        if len(selected) < size:
            continue
        file = output / f'train-teacher-{size}.jsonl'
        data = ''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in selected[:size]).encode()
        if file.exists():
            assert file.read_bytes() == data, '已导出数据不可覆盖'
        else:
            file.write_bytes(data)
        paths.append({'file': file.relative_to(ROOT).as_posix(), 'sha256': sha256(file), 'trajectories': size})
    manifest = {'run_id': run_id, 'requests': len(rows), 'accepted': len(records), 'rejected': len(rejections),
                'valid_target': config['valid_target'], 'target_met': len(records) >= config['valid_target'],
                'categories': dict(Counter(record['category'] for record in records)), 'files': paths,
                'selected_ids': [record['sample_id'] for record in selected],
                'system_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'template_sha256': hashlib.sha256(original.encode()).hexdigest(), 'tools_schema_sha256': schema_hash,
                'raw_records_sha256': records_hash, 'converter_sha256': sha256(ROOT / 'scripts/pi_teacher_data.py'),
                'selection': '保留执行通过且完整编码成功的轨迹，按冻结请求顺序取前256条；128条为前缀'}
    write_json(output / 'manifest.json', manifest)
    write_json(folder / 'teacher-export.json', manifest)
    write_json(folder / 'teacher-rejections.json', rejections)
    write_json(folder / 'teacher-units.json', units)
    public = ROOT / 'experiments/E16/runs' / f'{run_id}.json'
    summary = read(public)
    summary.update(encoded_trajectories=len(records), encoded_target_met=manifest['target_met'],
                   export_manifest_sha256=sha256(folder / 'teacher-export.json'))
    write_json(public, summary)
    print(json.dumps(manifest, ensure_ascii=False))


def inspect(run_id, snapshot):
    config, rows, records, rejections, units, prompt, original, schema_hash, records_hash = convert(run_id, snapshot)
    result = {'run_id': run_id, 'operation': 'teacher_encoding_snapshot_inspection', 'requests': len(rows),
              'preliminary_accepted': sum(row['teacher_filter']['accepted'] for row in rows),
              'encoded_trajectories': len(records), 'assistant_units': len(units),
              'supervised_tokens': sum(unit['supervised_tokens'] for unit in units),
              'max_input_tokens': max((unit['input_tokens'] for unit in units), default=0),
              'prefix_checks': len(units), 'cuda_initialized': False,
              'categories': dict(Counter(record['category'] for record in records)),
              'encoding_rejections': [row for row in rejections if any(reason.startswith('encoding_or_protocol:') for reason in row['reasons'])],
              'snapshot_sha256': records_hash, 'converter_sha256': sha256(ROOT / 'scripts/pi_teacher_data.py'),
              'system_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
              'template_sha256': hashlib.sha256(original.encode()).hexdigest(), 'tools_schema_sha256': schema_hash,
              'units': units}
    write_json(snapshot.with_suffix('.inspection.json'), result)
    print(json.dumps({key: value for key, value in result.items() if key != 'units'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['freeze', 'export', 'inspect'])
    parser.add_argument('--run')
    parser.add_argument('--snapshot')
    args = parser.parse_args()
    if args.operation == 'freeze':
        freeze()
    else:
        assert args.run and args.run.startswith('E16-R') and '/' not in args.run and '\\' not in args.run
        if args.operation == 'inspect':
            assert args.snapshot
            inspect(args.run, ROOT / args.snapshot)
        else:
            export(args.run)
