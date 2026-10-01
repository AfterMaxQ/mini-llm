"""压缩已结束运行的轻量证据，给仍保留在原位的大文件建立哈希清单。"""
import argparse
import json
import zipfile

from lab import ROOT, now, sha256, write_json


def archive_run(run_id):
    run=ROOT/'.local/runs'/run_id
    if not (run/'result.json').exists():
        raise RuntimeError('运行尚未结束，不能作为定稿归档')
    directory=ROOT/'.local/archive'/run_id.split('-')[0]
    directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/(run_id+'-manifest.json')
    if manifest.exists():
        saved=json.loads(manifest.read_text(encoding='utf-8'))
        assert sha256(directory/saved['archive'])==saved['archive_sha256']
        assert all(sha256(ROOT/f['file'])==f['sha256'] for f in saved['files'])
        return saved
    files=sorted(p for p in run.rglob('*') if p.is_file())
    log=ROOT/'.local/logs'/(run_id+'.log')
    if log.exists():files.append(log)
    if (run/'external-log.json').exists():
        logs=json.loads((run/'external-log.json').read_text(encoding='utf-8'))
        for name in ('file','stdout','stderr'):
            if name not in logs:continue
            external=(ROOT/logs[name]).resolve()
            assert external.is_relative_to((ROOT/'.local/logs').resolve())
            if external.exists() and external not in files:files.append(external)
    items=[{'file':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in files]
    path=directory/(run_id+'.zip')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as zipped:
        for p in files:
            if p.suffix not in {'.pt','.safetensors','.bin'}:
                zipped.write(p,p.relative_to(ROOT))
    saved={'run_id':run_id,'archived':now(),'archive':path.name,'archive_sha256':sha256(path),
           'files':items,'policy':'原始文件与关键checkpoint仍保留；压缩包保存轻量证据，大文件通过清单追溯'}
    write_json(manifest,saved)
    return saved


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+');args=parser.parse_args()
    for run_id in args.runs:
        saved=archive_run(run_id)
        print(json.dumps({'run_id':run_id,'files':len(saved['files']),'archive_sha256':saved['archive_sha256']}),flush=True)
