"""复用已执行通过的训练参考，冻结与 Pi 推理提示一致的专项数据。"""
import copy
import hashlib
import json
from collections import Counter, defaultdict

from lab import ROOT, finish_run, sha256, start_run, write_json
from prepare_sft import prepare


def main():
    source = ROOT / '.local/data/processed/E14-R11/train-reference.jsonl'
    source_config = json.loads((ROOT / 'configs/domain-mix.json').read_text(encoding='utf-8'))
    assert sha256(source) == source_config['pi_source_sha256']
    records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    groups = defaultdict(list)
    for record in records:
        assert record['split'] == 'train' and record['validation']['execution'] == 'reference_passed'
        groups[record['template_family']].append(record)
    assert len(groups) == 16
    prompt_path = ROOT / 'configs/pi-focused-prompt.json'
    suffix = json.loads(prompt_path.read_text(encoding='utf-8'))['suffix']
    selected = []
    for family, items in sorted(groups.items()):
        items.sort(key=lambda item: hashlib.sha256(('2026:' + item['sample_id']).encode()).hexdigest())
        selected.extend(copy.deepcopy(items[:4]))
    assert len(selected) == 64 and set(Counter(r['category'] for r in selected).values()) == {8}
    for record in selected:
        record['messages'][0]['content'] += '\n\n' + suffix
    directory = ROOT / '.local/data/processed/E14-focused-tools'
    directory.mkdir(parents=True, exist_ok=False)
    train = directory / 'train-64.jsonl'
    train.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected), encoding='utf-8')
    dev_source = ROOT / '.local/data/processed/focused-public/dev.jsonl'
    assert sha256(dev_source) == source_config['dev_source_sha256']
    (directory / 'dev.jsonl').write_bytes(dev_source.read_bytes())
    _, train_manifest = prepare('train-64.jsonl', 'E14-focused-tools')
    _, dev_manifest = prepare('dev.jsonl', 'E14-focused-tools')
    assert train_manifest['max_sequence_length'] <= 4096 and dev_manifest['independent_trajectories'] == 100
    manifest = {'operation': 'focused_tool_data', 'status': 'focused_tool_data_frozen', 'exit_code': 0,
                'source_run': 'E14-R11', 'source_sha256': sha256(source), 'prompt_sha256': sha256(prompt_path),
                'selected_ids': [r['sample_id'] for r in selected], 'categories': dict(Counter(r['category'] for r in selected)),
                'families': dict(Counter(r['template_family'] for r in selected)),
                'train': train_manifest, 'dev': dev_manifest, 'model_calls': 0,
                'target_policy': '工具调用、返回和最终回答复用已执行通过的训练参考；只统一系统提示，不读取test标签',
                'files': {p.name: sha256(p) for p in (train, directory / 'dev.jsonl')}}
    frozen_path = ROOT / 'configs/pi-focused-data.json'
    assert not frozen_path.exists()
    write_json(frozen_path, manifest)
    config = json.loads((ROOT / 'configs/sft-domain.json').read_text(encoding='utf-8'))
    config.pop('domain_manifest_sha256')
    config.update(data_run='E14-focused-tools', train_size=64, learning_rate=5e-5, max_sequence_length=4096,
                  initial_adapter='.local/runs/E14-R19/selected-adapter',
                  initial_adapter_sha256=sha256(ROOT / '.local/runs/E14-R19/selected-adapter/adapter_model.safetensors'),
                  training_prompt='与configs/pi-focused-prompt.json相同的Pi任务提示',
                  evaluation_prompt_config='configs/pi-focused-prompt.json',
                  data_file_hashes=manifest['files'], notes_script='scripts/notes_focused.py')
    write_json(ROOT / 'configs/sft-pi-focused.json', config)
    run = start_run('E14', {'operation': 'focused_tool_data', 'source_run': 'E14-R11',
                            'manifest_sha256': sha256(frozen_path)})
    finish_run(run, manifest)
    print(json.dumps({'run_id': run.name, 'trajectories': 64, 'assistant_units': train_manifest['assistant_units'],
                      'max_tokens': train_manifest['max_sequence_length'], 'dev_trajectories': 100}, ensure_ascii=False))


if __name__ == '__main__':
    main()
