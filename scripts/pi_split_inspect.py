"""检查训练场景与dev的近似重复；阈值只用于筛查，不替代结构复查。"""
import argparse
import json
import re
import sys
import time
from lab import ROOT, start_run, finish_run, sha256, write_json


def inspect(training, tasks, threshold=.8, source_name='train', target_name='dev'):
    def normalized(task):
        value = json.dumps({'prompt': task['prompt'], 'files': task['files']}, ensure_ascii=False, sort_keys=True)
        return re.sub(r'\d+(?:\.\d+)?', '<N>', value).lower()
    train_text = [normalized(t) for t in training]; dev_text = [normalized(t) for t in tasks]
    def grams(text): return {text[i:i+5] for i in range(len(text)-4)}
    train_grams = [grams(t) for t in train_text]; dev_grams = [grams(t) for t in dev_text]
    nearest = []; flagged = 0
    for task, features in zip(tasks, dev_grams):
        scores = [len(features & other) / len(features | other) for other in train_grams]
        flagged += sum(value >= threshold for value in scores)
        index = max(range(len(scores)), key=scores.__getitem__)
        nearest.append({target_name+'_task_id': task['task_id'], source_name+'_task_id': training[index]['task_id'], 'jaccard': scores[index]})
    return {'method': '完整问题和初始文件：小写、数字归一后字符5-gram集合Jaccard',
        'threshold': threshold, 'pairs_checked': len(tasks)*len(training),
        'normalized_exact_overlap': len(set(train_text) & set(dev_text)), 'pairs_above_threshold': flagged,
        'nearest_per_'+target_name: nearest, 'max_jaccard': max(x['jaccard'] for x in nearest),
        'scope': '可复现的近似重复筛查；未证明没有语义相似或预训练污染'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--training', required=True); parser.add_argument('--dev', required=True)
    parser.add_argument('--test')
    args = parser.parse_args()
    train_run = ROOT/'.local/runs'/args.training; dev_run = ROOT/'.local/runs'/args.dev
    train_result = json.loads((train_run/'result.json').read_text(encoding='utf-8'))
    dev_result = json.loads((dev_run/'result.json').read_text(encoding='utf-8'))
    assert train_result['status'] == 'training_requests_frozen' and dev_result['status'] == 'dev_reference_verified'
    train_file = ROOT/train_result['requests_file']; dev_file = dev_run/'tasks.json'
    assert sha256(train_file) == train_result['requests_sha256'] and sha256(dev_file) == dev_result['tasks_sha256']
    training = [json.loads(line) for line in train_file.read_text(encoding='utf-8').splitlines()]
    tasks = json.loads(dev_file.read_text(encoding='utf-8'))
    assert len(training) == 2000 and len(tasks) == 40
    target_run = ROOT/'.local/runs'/args.test if args.test else dev_run
    target_result = json.loads((target_run/'result.json').read_text(encoding='utf-8'))
    if args.test:
        assert target_result['status'] == 'test_reference_verified' and target_result['task_count'] == 100
        assert sha256(target_run/'tasks.json') == target_result['tasks_sha256']
        target_tasks = json.loads((target_run/'tasks.json').read_text(encoding='utf-8'))
    config = {'operation': 'pi_test_split_inspection' if args.test else 'pi_split_inspection', 'command': [sys.executable, *sys.argv], 'model': None,
        'training_run': args.training, 'dev_run': args.dev, 'training_sha256': sha256(train_file),
        'dev_sha256': sha256(dev_file), 'threshold': .8, 'character_ngrams': 5}
    if args.test: config.update({'test_run': args.test, 'test_sha256': sha256(target_run/'tasks.json')})
    run = start_run('E14', config); started = time.perf_counter()
    try:
        if args.test:
            comparisons={}
            for name,data in [('train',training),('dev',tasks)]:
                inspected=inspect(data,target_tasks,config['threshold'],name,'test')
                inspected['exact_overlap']={key:len({t[key] for t in data}&{t[key] for t in target_tasks})
                    for key in ['scene_sha256','template_family','repository_family']}
                assert not any(inspected['exact_overlap'].values())
                assert inspected['normalized_exact_overlap'] == inspected['pairs_above_threshold'] == 0
                comparisons[name]=inspected
            result={'comparisons':comparisons,'pairs_checked':sum(v['pairs_checked'] for v in comparisons.values()),
                'test_tasks':100,'normalized_exact_overlap':0,'pairs_above_threshold':0}
        else: result = inspect(training, tasks, config['threshold'])
        write_json(run/'inspection.json', result)
        assert result['normalized_exact_overlap'] == result['pairs_above_threshold'] == 0
        finish_run(run, {**result, 'status': 'test_split_inspection_verified' if args.test else 'split_inspection_verified', 'exit_code': 0, 'operation': config['operation'],
            'model_calls': 0, 'inspection_sha256': sha256(run/'inspection.json'), 'duration_seconds': time.perf_counter()-started})
        print(json.dumps({k: v for k, v in result.items() if not k.startswith('nearest_per_') and k!='comparisons'}, ensure_ascii=False))
    except Exception:
        import traceback
        finish_run(run, {'status': 'failed', 'exit_code': 1, 'operation': config['operation'], 'error': traceback.format_exc()})
        raise


if __name__ == '__main__': main()
