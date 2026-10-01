"""从真实 Pi 规则执行记录生成训练格式，检查完整历史与当前回复监督。"""
import argparse
import hashlib
import json
import sys
from collections import Counter

import torch
from lab import ROOT, start_run, finish_run, sha256, write_json
from templates import action_records, encode_action, tokenizer_and_template


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_arguments(value, schema):
    kind = schema.get('type')
    if kind == 'object':
        assert isinstance(value, dict) and set(schema.get('required', [])).issubset(value)
        assert set(value).issubset(schema.get('properties', {}))
        for name, item in value.items(): check_arguments(item, schema['properties'][name])
    elif kind == 'array':
        assert isinstance(value, list)
        for item in value: check_arguments(item, schema['items'])
    elif kind == 'string': assert isinstance(value, str)
    elif kind == 'number': assert type(value) in (int, float)
    elif kind == 'boolean': assert type(value) is bool
    else: raise ValueError(f'当前Pi参数中出现尚未核对的schema类型：{kind}')


def convert(source, task, tools, config, schema_hash):
    messages = [{'role': 'system', 'content': config['system_prompt']}, {'role': 'user', 'content': task['prompt']}]
    definitions = {t['function']['name']: t['function'] for t in tools}
    metadata = []
    assert source['task_id'] == task['task_id'] and source['split'] == task['split'] == 'train'
    assert source['reference']['passed'] and len(source['reference_events']) == len(task['reference'])
    for event, action in zip(source['reference_events'], task['reference']):
        assert event['tool'] == action['tool'] and event['input'] == action['input']
        check_arguments(event['input'], definitions[event['tool']]['parameters'])
        call_id = event['call_id']
        messages.append({'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': call_id, 'type': 'function', 'function': {'name': event['tool'], 'arguments': event['input']}}]})
        if 'error' in event:
            assert event['is_error']
            content = event['error']
        else:
            blocks = event['result']['content']
            assert all(block['type'] == 'text' for block in blocks), '当前文本训练格式不能静默丢掉非文本返回'
            content = '\n'.join(block['text'] for block in blocks)
        messages.append({'role': 'tool', 'name': event['tool'], 'tool_call_id': call_id, 'content': content})
        metadata.append({'call_id': call_id, 'is_error': event['is_error'], 'event_sha256': digest(event),
                         'content_sha256': hashlib.sha256(content.encode()).hexdigest(),
                         'execution': event.get('result', {}).get('structuredContent')})
    messages.append({'role': 'assistant', 'content': source['reference_answer']})
    ids = [m['tool_calls'][0]['id'] for m in messages if m.get('tool_calls')]
    assert len(ids) == len(set(ids))
    assert ids == [m['tool_call_id'] for m in messages if m['role'] == 'tool']
    for index, message in enumerate(messages):
        if message['role'] == 'tool': assert messages[index-1]['tool_calls'][0]['id'] == message['tool_call_id']
    return {'sample_id': task['task_id'], 'split': 'train', 'category': task['category'],
            'template_family': task['template_family'], 'repository_family': task['repository_family'],
            'group_id': digest([task['template_family'], task['repository_family']]),
            'source': 'minillm_pi_rules', 'source_run_id': config['source_run'], 'source_id': task['task_id'],
            'messages': messages, 'tools': tools, 'tools_sha256': schema_hash, 'tool_result_metadata': metadata,
            'teacher_run_id': None, 'reference_answer_source': 'executed_rule_reference',
            'validation': {'format': True, 'arguments': True, 'execution': 'reference_passed'},
            'rejection_reason': None}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='configs/pi-reference-data.json'); args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    source_run = ROOT / '.local/runs' / config['source_run']
    source_result = json.loads((source_run / 'result.json').read_text(encoding='utf-8'))
    assert source_result['status'] == 'reference_pilot_verified' and source_result['model_calls'] == 0
    assert sha256(source_run / 'records.json') == source_result['records_sha256']
    assert sha256(source_run / 'tasks.json') == source_result['tasks_sha256']
    source = json.loads((source_run / 'records.json').read_text(encoding='utf-8'))
    tasks = json.loads((source_run / 'tasks.json').read_text(encoding='utf-8'))
    assert len(source) == len(tasks) == 8
    installed = json.loads((ROOT / 'configs/pi-tools.json').read_text(encoding='utf-8'))
    tools = [{'type': 'function', 'function': tool} for tool in installed['tools']]
    schema_hash = digest(tools)
    config.update({'command': sys.argv, 'operation': 'pi_reference_encoding',
                   'source_records_sha256': sha256(source_run / 'records.json'),
                   'source_tasks_sha256': sha256(source_run / 'tasks.json'),
                   'source_result_sha256': sha256(source_run / 'result.json'),
                   'pi_tools_file_sha256': sha256(ROOT / 'configs/pi-tools.json'), 'tools_schema_sha256': schema_hash,
                   'source_code_snapshot_sha256': source_result['config']['code_snapshot_sha256'] if 'config' in source_result else json.loads((source_run/'config.json').read_text(encoding='utf-8'))['code_snapshot_sha256']})
    run = start_run('E14', config)
    try:
        tokenizer, original, marked = tokenizer_and_template()
        records = [convert(s, t, tools, config, schema_hash) for s, t in zip(source, tasks)]
        assert len({r['sample_id'] for r in records}) == 8
        units = []; encoded = []; boundaries = []
        for record in records:
            target_count = 0
            for action in action_records(record):
                item = encode_action(tokenizer, marked, action)
                text = tokenizer.apply_chat_template(action['messages'], tools=tools, tokenize=False,
                            enable_thinking=False, add_generation_prompt=False)
                tagged = tokenizer.apply_chat_template(action['messages'], tools=tools, chat_template=marked,
                            tokenize=False, enable_thinking=False, add_generation_prompt=False)
                prefix = tokenizer.apply_chat_template(action['messages'][:-1], tools=tools, tokenize=False,
                            enable_thinking=False, add_generation_prompt=True)
                assert text == tagged and text.startswith(prefix)
                assert tokenizer(text, add_special_tokens=False)['input_ids'] == item['input_ids']
                full = tokenizer.apply_chat_template(action['messages'], tools=tools, chat_template=marked,
                            tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
                            enable_thinking=False, add_generation_prompt=False)
                starts = [i for i, value in enumerate(full['assistant_masks']) if value and (i==0 or not full['assistant_masks'][i-1])]
                start = starts[-1]
                assert all(x == -100 for x in item['labels'][:start])
                assert item['assistant_masks'][start:] == full['assistant_masks'][start:]
                assert all(y == (x if active else -100) for x,y,active in zip(item['input_ids'],item['labels'],item['assistant_masks']))
                targets = sum(x != -100 for x in item['labels'][1:])
                assert targets > 0 and tokenizer.eos_token_id in item['labels']
                assert len(item['input_ids']) <= config['context_limit'], '完整轨迹超出部署预算，不能截断凑成可训练样本'
                target_count += targets
                unit = {'sample_id': record['sample_id'], 'category': record['category'], 'message_index': action['target_message_index'],
                        'input_tokens': len(item['input_ids']), 'target_tokens': targets, 'target_start_token': start,
                        'input_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                        'target_text': tokenizer.decode([x for x in item['labels'] if x != -100])}
                units.append(unit); encoded.append({**item, 'sample_id': record['sample_id'], 'message_index': action['target_message_index']})
                if record['category'] == 'failure_recovery' and action['target_message_index'] == 4:
                    tool = action['messages'][-2]
                    assert tool['role'] == 'tool' and 'Command exited with code 1' in tool['content']
                    assert "'a- b'" in tool['content'] and tool['content'] in prefix
                    boundaries.append({'sample_id':record['sample_id'],'message_index':4,'preceding_error':tool['content'],
                                       'tool_call_id':tool['tool_call_id'],'target_text':unit['target_text'],
                                       'input_tokens':len(item['input_ids']),'target_tokens':targets,'history_supervised_tokens':sum(x!=-100 for x in item['labels'][:start])})
            record['token_count'] = units[-1]['input_tokens']; record['target_token_count'] = target_count
        assert len(units) == 34 and len(boundaries) == 1
        assert sum(m['is_error'] for r in records for m in r['tool_result_metadata']) == 2
        assert not torch.cuda.is_initialized()
        output = ROOT / '.local/data/processed' / run.name; output.mkdir(parents=True, exist_ok=False)
        raw = output / 'train-pilot.jsonl'
        raw.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
        for name, values in [('units.jsonl', units), ('encoded.jsonl', encoded)]:
            (run / name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values),encoding='utf-8')
        write_json(run / 'mask-examples.json', boundaries)
        result = {'status':'pilot_encoding_verified','exit_code':0,'operation':'pi_reference_encoding',
                  'independent_trajectories':8,'assistant_units':len(units),'tool_calls':26,'error_returns':2,
                  'input_tokens':sum(u['input_tokens'] for u in units),'supervised_tokens':sum(u['target_tokens'] for u in units),
                  'max_sequence_length':max(u['input_tokens'] for u in units),
                  'length_inspection':{str(n):{'units_over_limit':sum(u['input_tokens']>n for u in units),
                    'trajectories_over_limit':len({u['sample_id'] for u in units if u['input_tokens']>n})} for n in config['training_lengths_to_inspect']},
                  'categories':dict(Counter(r['category'] for r in records)),'prefix_text_token_mask_checks':len(units),
                  'cuda_initialized':False,'truncated_units':0,'model_calls':0,'tools_schema_sha256':schema_hash,
                  'template_sha256':hashlib.sha256(original.encode()).hexdigest(),
                  'marked_template_sha256':hashlib.sha256(marked.encode()).hexdigest(),
                  'processed_file':raw.relative_to(ROOT).as_posix(),'processed_sha256':sha256(raw),
                  'units_sha256':sha256(run/'units.jsonl'),'encoded_sha256':sha256(run/'encoded.jsonl'),
                  'mask_examples_sha256':sha256(run/'mask-examples.json'),
                  'scope':'八个真实训练原型的格式与遮罩核验；未满足正式轨迹规模，未进行Pi训练'}
        write_json(output / 'manifest.json', result)
        finish_run(run, result)
        print(json.dumps(result, ensure_ascii=False))
    except Exception:
        import traceback
        finish_run(run, {'status':'failed','exit_code':1,'operation':'pi_reference_encoding','error':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
