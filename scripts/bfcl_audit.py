"""复核冻结清单，并归档基准原文件、检查器源码和依赖排错证据。"""
import argparse
import csv
import json
import zipfile

from lab import ROOT, now, sha256, write_json
from archive import archive_run
from bfcl_prepare import digest, freeze_manifest, load_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    run = ROOT/'.local/runs'/args.source
    result = json.loads((run/'result.json').read_text(encoding='utf-8'))
    public = ROOT/'experiments/E22/runs'/(args.source+'.json')
    published = json.loads(public.read_text(encoding='utf-8'))
    assert published['raw_result_sha256'] == sha256(run/'result.json')
    assert result['status'] == 'benchmark_preparation_verified' and result['model_scores'] is None
    for key, filename in [('samples_sha256', 'samples.json'), ('checker_sources_sha256', 'checker-sources.json'),
                          ('interface_probes_sha256', 'interface-probes.json')]:
        assert result[key] == sha256(run/filename)
    config = json.loads((run/'config.json').read_text(encoding='utf-8'))
    assert config['environment_lock_sha256'] == sha256(run/'environment.lock.txt')
    write_json(ROOT/'configs/bfcl-frozen.json', freeze_manifest(config, result))
    frozen = ROOT/'.local/bfcl/data'/args.source
    samples = json.loads((run/'samples.json').read_text(encoding='utf-8'))
    reconstructed = []
    for row in result['categories']:
        filename = result['dataset_version']+'_'+row['category']+'.json'
        questions = load_rows(frozen/filename)
        assert len(questions) == row['questions'] and sha256(frozen/filename) == row['question_sha256']
        assert digest([q['id'] for q in questions]) == row['ids_sha256']
        answers = []
        if row['checker'] == 'AST':
            answer_path = frozen/'possible_answer'/filename
            answers = load_rows(answer_path)
            assert len(answers) == row['answers'] and sha256(answer_path) == row['answer_sha256']
            assert [q['id'] for q in questions] == [a['id'] for a in answers]
        for index, question in enumerate(questions):
            reconstructed.append({'id': question['id'], 'category': row['category'], 'index': index,
                                  'question_sha256': digest(question),
                                  'answer_sha256': digest(answers[index]) if answers else None})
    assert reconstructed == samples and len({r['id'] for r in samples}) == result['total_questions']
    with (ROOT/'experiments/E22/categories.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 13 and sum(int(r['questions']) for r in rows) == len(samples)
    assert sha256(ROOT/'experiments/E22/categories.csv') == result['categories_csv_sha256']
    probes = json.loads((run/'interface-probes.json').read_text(encoding='utf-8'))
    assert len(probes) == len(result['interface_probes']) == 29
    for full, compact in zip(probes, result['interface_probes']):
        assert full['id'] == compact['id'] and full['official_result']['valid'] == compact['valid']
    package = ROOT/'.local/venv-bfcl/Lib/site-packages/bfcl_eval'
    sources = json.loads((run/'checker-sources.json').read_text(encoding='utf-8'))
    assert all(sha256(package/name) == value for name, value in sources.items())
    directory = ROOT/'.local/archive/E22'
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory/(args.source+'-benchmark-manifest.json')
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert sha256(directory/saved['archive']) == saved['archive_sha256']
        assert all(sha256(ROOT/f['file']) == f['sha256'] for f in saved['files'])
    else:
        files = sorted(p for p in frozen.rglob('*') if p.is_file())
        files += [package/name for name in sources]
        files += [p for p in (ROOT/'.local/bfcl').glob('*.txt') if p.is_file()]
        items = [{'file': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files]
        archive = directory/(args.source+'-benchmark.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
            for path in files:
                zipped.write(path, path.relative_to(ROOT).as_posix())
        with zipfile.ZipFile(archive) as zipped:
            for item in items:
                import hashlib
                assert hashlib.sha256(zipped.read(item['file'])).hexdigest() == item['sha256']
        saved = {'run_id': args.source, 'archived': now(), 'archive': archive.name,
                 'archive_sha256': sha256(archive), 'files': items}
        write_json(manifest_path, saved)
    run_archives = []
    for path in sorted((ROOT/'experiments/E22/runs').glob('*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('operation') == 'bfcl_preparation':
            manifest = archive_run(record['run_id'])
            run_archives.append({'run_id': record['run_id'], 'archive_sha256': manifest['archive_sha256']})
    audit = {'source_run': args.source, 'checked': now(), 'categories': len(rows), 'questions': len(samples),
             'ast_questions': sum(int(r['questions']) for r in rows if r['checker']=='AST'),
             'relevance_questions': sum(int(r['questions']) for r in rows if r['checker']=='relevance'),
             'unique_ids': len({r['id'] for r in samples}), 'probes': len(probes), 'source_modules': len(sources),
             'categories_csv_sha256': result['categories_csv_sha256'], 'benchmark_archive_sha256': saved['archive_sha256'],
             'archived_files': len(saved['files']), 'run_archives': run_archives,
             'model_inference': False, 'model_scores': None}
    write_json(ROOT/'experiments/E22/preparation-audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
