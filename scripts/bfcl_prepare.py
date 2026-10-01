"""冻结 BFCL 单轮数据，并记录官方判分接口的实际行为；不调用模型。"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import sys
import traceback

from lab import ROOT, finish_run, sha256, start_run, write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def freeze_manifest(config, result):
    return {**{key: result[key] for key in ['run_id', 'dataset_version', 'total_questions', 'categories',
                                          'samples_sha256', 'checker_sources_sha256']},
            **{key: config[key] for key in ['package_version', 'test_group', 'registry_name', 'mode',
                                          'format_conversion', 'environment_lock_sha256']}}


def fixtures():
    def function(name, properties, required):
        return {'name': name, 'description': '手写接口检查，非基准题目。',
                'parameters': {'type': 'dict', 'properties': properties, 'required': required}}
    add = function('math.add', {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, ['a', 'b'])
    city = function('weather.lookup', {'city': {'type': 'string'}}, ['city'])
    truth = [{'math.add': {'a': [2], 'b': [3]}}]
    cases = []

    def case(name, category, functions, answers, calls=None, raw=None):
        if raw is None:
            raw = '\n'.join('<tool_call>\n'+json.dumps(call, ensure_ascii=False)+'\n</tool_call>'
                            for call in (calls or []))
        cases.append({'id': name, 'category': category, 'function': functions,
                      'ground_truth': answers, 'raw': raw})

    good = {'name': 'math.add', 'arguments': {'a': 2, 'b': 3}}
    for name, call in [
        ('python-correct', good),
        ('python-wrong-value', {'name': 'math.add', 'arguments': {'a': 2, 'b': 4}}),
        ('python-wrong-type', {'name': 'math.add', 'arguments': {'a': '2', 'b': 3}}),
        ('python-missing-required', {'name': 'math.add', 'arguments': {'a': 2}}),
        ('python-extra-parameter', {'name': 'math.add', 'arguments': {'a': 2, 'b': 3, 'c': 0}}),
        ('python-wrong-function', {'name': 'math.subtract', 'arguments': {'a': 2, 'b': 3}}),
        ('python-bool-integer', {'name': 'math.add', 'arguments': {'a': True, 'b': 3}})]:
        case(name, 'simple_python', [add], truth, [call])
    case('python-string-normalization', 'simple_python', [city],
         [{'weather.lookup': {'city': ['New York']}}],
         [{'name': 'weather.lookup', 'arguments': {'city': 'new-york'}}])
    for language, value_type in [('java', 'integer'), ('javascript', 'integer')]:
        function_def = function('math.add', {'a': {'type': value_type}, 'b': {'type': value_type}}, ['a', 'b'])
        for representation, arguments in [('native-json', {'a': 2, 'b': 3}),
                                           ('source-literals', {'a': '2', 'b': '3'})]:
            case(language+'-'+representation, 'simple_'+language, [function_def], truth,
                 [{'name': 'math.add', 'arguments': arguments}])
    weather = {'name': 'weather.lookup', 'arguments': {'city': 'Paris'}}
    city_truth = [{'weather.lookup': {'city': ['Paris']}}]
    case('multiple-selection', 'multiple', [add, city], city_truth, [weather])
    case('parallel-reversed', 'parallel', [add], truth+[{'math.add': {'a': [5], 'b': [6]}}],
         [{'name': 'math.add', 'arguments': {'a': 5, 'b': 6}}, good])
    case('parallel-missing-call', 'parallel', [add], truth+[{'math.add': {'a': [5], 'b': [6]}}], [good])
    case('parallel-multiple-reversed', 'parallel_multiple', [add, city], truth+city_truth, [weather, good])
    malformed = '<tool_call>\n{"name":"math.add","arguments":BROKEN}\n</tool_call>'
    for category in ['irrelevance', 'live_irrelevance', 'live_relevance']:
        for name, raw in [('empty', ''), ('prose', '需要更多信息。'),
                          ('valid-call', '<tool_call>\n'+json.dumps(good)+'\n</tool_call>'),
                          ('malformed-call', malformed)]:
            case(category+'-'+name, category, [add], None, raw=raw)
    case('mixed-valid-malformed', 'simple_python', [add], truth,
         raw='<tool_call>\n'+json.dumps(good)+'\n</tool_call>\n'+malformed)
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/bfcl.json')
    args = parser.parse_args()
    config = json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    actual_version = importlib.metadata.version('bfcl-eval')
    if actual_version != config['package_version']:
        raise RuntimeError(f'BFCL 版本不一致：{actual_version}')
    os.environ['BFCL_PROJECT_ROOT'] = str(ROOT/'.local/bfcl')
    # 准备阶段的判分不需要网络，连向本机服务也视为意外调用。
    attempts = []
    def blocked(*args, **kwargs):
        attempts.append(str(args[1:]))
        raise RuntimeError('BFCL 准备检查禁止网络连接')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = blocked
    run = start_run('E22', {**config, 'operation': 'bfcl_preparation', 'package_version_actual': actual_version,
                            'environment_lock_sha256': sha256(ROOT/'configs/environment-bfcl.lock.txt'),
                            'command': sys.argv, 'model_inference': False})
    try:
        shutil.copy2(ROOT/'configs/environment-bfcl.lock.txt', run/'environment.lock.txt')
        from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING, VERSION_PREFIX
        from bfcl_eval.constants.eval_config import PACKAGE_ROOT, PROMPT_PATH, POSSIBLE_ANSWER_PATH
        from bfcl_eval.constants.enums import Language, ReturnFormat
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
        from bfcl_eval.eval_checker.eval_runner import (_evaluate_single_ast_entry,
                                                      _evaluate_single_relevance_entry, get_handler)
        assert TEST_COLLECTION_MAPPING[config['test_group']] == config['categories']
        assert len(config['categories']) == 13 and config['mode'] == 'FC'
        assert MODEL_CONFIG_MAPPING[config['registry_name']].is_fc_model
        frozen = ROOT/'.local/bfcl/data'/run.name
        frozen.mkdir(parents=True, exist_ok=False)
        category_rows, items, all_ids = [], [], set()
        for category in config['categories']:
            name = f'{VERSION_PREFIX}_{category}.json'
            question_path, answer_path = PROMPT_PATH/name, POSSIBLE_ANSWER_PATH/name
            questions = load_rows(question_path)
            ids = [q['id'] for q in questions]
            assert len(ids) == len(set(ids)) and not all_ids.intersection(ids)
            all_ids.update(ids)
            relevance = 'relevance' in category
            answers = load_rows(answer_path) if not relevance else []
            if not relevance:
                assert len(answers) == len(questions)
                assert [a['id'] for a in answers] == ids, f'{category}答案id或顺序不一致'
            shutil.copy2(question_path, frozen/name)
            if answers:
                (frozen/'possible_answer').mkdir(exist_ok=True)
                shutil.copy2(answer_path, frozen/'possible_answer'/name)
            row = {'category': category, 'questions': len(questions), 'answers': len(answers),
                   'checker': 'relevance' if relevance else 'AST',
                   'language': 'java' if category=='simple_java' else 'javascript' if category=='simple_javascript' else 'python',
                   'question_sha256': sha256(question_path), 'answer_sha256': sha256(answer_path) if answers else '',
                   'ids_sha256': digest(ids)}
            category_rows.append(row)
            for index, question in enumerate(questions):
                items.append({'id': ids[index], 'category': category, 'index': index,
                              'question_sha256': digest(question),
                              'answer_sha256': digest(answers[index]) if answers else None})
        source_hashes = {p.relative_to(PACKAGE_ROOT).as_posix(): sha256(p)
                         for p in sorted(PACKAGE_ROOT.rglob('*.py')) if '__pycache__' not in p.parts}
        write_json(run/'checker-sources.json', source_hashes)
        write_json(run/'samples.json', items)
        write_json(run/'categories.json', category_rows)
        handler = get_handler(config['registry_name'])
        probes = []
        for entry in fixtures():
            write_json(run/'current-probe.json', entry)
            category = entry['category']
            prompt = {'id': entry['id'], 'function': entry['function'],
                      'question': [[{'role': 'user', 'content': '接口检查，无模型推理。'}]]}
            if 'relevance' in category:
                outcome = _evaluate_single_relevance_entry(handler, entry['id'], entry['raw'],
                                                           prompt, config['registry_name'], category)
            else:
                language = Language.JAVA if category=='simple_java' else Language.JAVASCRIPT if category=='simple_javascript' else Language.PYTHON
                outcome = _evaluate_single_ast_entry(handler, entry['id'], entry['raw'], entry['ground_truth'],
                                                     prompt, config['registry_name'], category, language,
                                                     ReturnFormat(language.value))
            try:
                decoded = handler.decode_ast(entry['raw'], ReturnFormat.PYTHON, False)
            except Exception as error:
                decoded = {'decode_exception': repr(error)}
            probes.append({**entry, 'decoded': decoded, 'official_result': outcome})
            write_json(run/'interface-probes.json', probes)
        assert not attempts
        public = ROOT/'experiments/E22'
        public.mkdir(parents=True, exist_ok=True)
        with (public/'categories.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(category_rows[0]))
            writer.writeheader(); writer.writerows(category_rows)
        compact = [{'id': p['id'], 'category': p['category'], 'valid': p['official_result']['valid'],
                    'decoded_calls': len(p['decoded']) if isinstance(p['decoded'], list) else None,
                    'error_type': p['official_result'].get('error_type', '')} for p in probes]
        result = finish_run(run, {'status': 'benchmark_preparation_verified', 'operation': 'bfcl_preparation',
                                 'dataset_version': VERSION_PREFIX, 'categories': category_rows,
                                 'total_questions': len(items), 'unique_ids': len(all_ids),
                                 'ast_questions': sum(r['questions'] for r in category_rows if r['checker']=='AST'),
                                 'relevance_questions': sum(r['questions'] for r in category_rows if r['checker']=='relevance'),
                                 'source_modules': len(source_hashes), 'checker_sources_sha256': sha256(run/'checker-sources.json'),
                                 'samples_sha256': sha256(run/'samples.json'), 'categories_csv_sha256': sha256(public/'categories.csv'),
                                 'interface_probes_sha256': sha256(run/'interface-probes.json'), 'interface_probes': compact,
                                 'network_attempts': len(attempts), 'model_inference': False,
                                 'model_scores': None, 'exit_code': 0})
        write_json(ROOT/'configs/bfcl-frozen.json', freeze_manifest(
            json.loads((run/'config.json').read_text(encoding='utf-8')), result))
        print(json.dumps({'run_id': run.name, 'questions': len(items), 'probes': compact}, ensure_ascii=False), flush=True)
    except Exception as error:
        (run/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        finish_run(run, {'status': 'failed', 'operation': 'bfcl_preparation', 'error': repr(error),
                         'model_inference': False, 'network_attempts': len(attempts), 'exit_code': 1})
        raise


if __name__ == '__main__':
    main()
