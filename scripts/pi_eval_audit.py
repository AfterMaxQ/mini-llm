"""从原始字节和逐条事件复核dev/test参考，保存精简表与数据归档。"""
import argparse
import base64
import csv
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from lab import ROOT, now, sha256, write_json


def scene(task):
    value=json.dumps({'prompt':task['prompt'],'files':task['files']},ensure_ascii=False,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);args=parser.parse_args()
    source=ROOT/'.local/runs'/args.source;result=json.loads((source/'result.json').read_text(encoding='utf-8'))
    split='test' if result['status']=='test_reference_verified' else 'dev';count=100 if split=='test' else 40
    assert result['status']==split+'_reference_verified' and result['exit_code']==0 and result['model_calls']==0
    assert sha256(source/'records.json')==result['records_sha256'] and sha256(source/'tasks.json')==result['tasks_sha256']
    tasks=json.loads((source/'tasks.json').read_text(encoding='utf-8'));records=json.loads((source/'records.json').read_text(encoding='utf-8'))
    frozen=ROOT/result['frozen_'+split+'_file'];assert sha256(frozen)==result['frozen_'+split+'_sha256']
    assert json.loads(frozen.read_text(encoding='utf-8'))==tasks and len(tasks)==len(records)==count
    assert len({t['task_id'] for t in tasks})==len({t['scene_sha256'] for t in tasks})==count
    assert Counter(t['category'] for t in tasks)==result['categories']
    names=[];rows=[];all_ids=[]
    for task,record in zip(tasks,records,strict=True):
        assert task['split']==record['split']==split and task['task_id']==record['task_id']
        assert scene(task)==task['scene_sha256']==record['scene_sha256']
        for key in ['template_family','repository_family']:assert task[key]==record[key]
        assert record['reference']['passed'] and not record['initial']['passed'] and not record['negative']['passed']
        assert not record['reference']['reasons'] and record['initial']['reasons'] and record['negative']['reasons']
        for kind in ['control','reference']:
            assert record[kind+'_isolation']=={'network':'none','readonly_root':True,'user':'1000:1000','binds':[],'gpu_requests':[]}
            assert record[kind+'_removed']['removed'];names.append(record[kind+'_removed']['container'])
        events=record['reference_events'];assert len(events)==len(task['reference'])<=12
        for event,action in zip(events,task['reference'],strict=True):
            assert event['tool']==action['tool'] and event['input']==action['input'];all_ids.append(event['call_id'])
            assert event['seconds']>=0 and ('result' in event or event['is_error'])
        state={file['path']:file for file in record['reference']['files']};assert set(state)==set(task['files'])
        for name,file in state.items():
            data=base64.b64decode(file['base64']);assert len(data)==file['bytes'] and hashlib.sha256(data).hexdigest()==file['sha256']
            if name not in task['allowed_changes']:assert data.decode('utf-8')==task['files'][name]
        expected=task['expected']
        if 'json_file' in expected:assert json.loads(base64.b64decode(state[expected['json_file']]['base64']))==expected['json_value']
        if 'check_output' in expected:
            execution=record['reference']['execution'];assert execution['code']==0 and execution['stdout'].strip()==expected['check_output']
        if 'answer' in expected:
            if isinstance(expected['answer'],str):assert record['reference_answer']==expected['answer']
            else:assert json.loads(record['reference_answer'])==expected['answer']
        if 'answer_contains' in expected:assert all(word in record['reference_answer'] for word in expected['answer_contains'])
        if 'max_tool_calls' in expected:assert len(events)<=expected['max_tool_calls']
        if expected.get('must_observe_error'):assert any(event['is_error'] for event in events)
        empty=any(event['tool']=='bash' and not event['is_error'] and any(block['type']=='text' and block['text'].strip()=='[]'
            for block in event.get('result',{}).get('content',[])) for event in events)
        if expected.get('must_observe_empty_result'):assert empty
        rows.append({'task_id':task['task_id'],'category':task['category'],'template_family':task['template_family'],
            'repository_family':task['repository_family'],'scene_sha256':task['scene_sha256'],'reference_calls':len(events),
            'error_returns':sum(event['is_error'] for event in events),'empty_query_observed':empty,
            'reference_passed':True,'initial_rejected':True,'negative_rejected':True,'negative_reasons':'|'.join(record['negative']['reasons'])})
    assert len(set(names))==len(names)==count*2 and len(set(all_ids))==len(all_ids)
    existing=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines()
    assert not set(names).intersection(existing)
    calls=sum(row['reference_calls'] for row in rows);errors=sum(row['error_returns'] for row in rows)
    assert calls==result['reference_tool_calls'] and errors==result['reference_error_returns']
    diagnostic=sum(len(record['control_events']) for record in records);assert diagnostic==result['diagnostic_tool_calls']
    csv_file=ROOT/f'experiments/E14/{split}-reference.csv'
    with csv_file.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    directory=ROOT/'.local/archive/E14';directory.mkdir(parents=True,exist_ok=True)
    data_zip=directory/f'{args.source}-{split}-data.zip';manifest=directory/f'{args.source}-{split}-data-manifest.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text(encoding='utf-8'));assert sha256(data_zip)==previous['sha256']
        assert previous['files'][0]['sha256']==sha256(frozen)
    else:
        with zipfile.ZipFile(data_zip,'w',zipfile.ZIP_DEFLATED) as archive:archive.write(frozen,frozen.relative_to(ROOT))
        with zipfile.ZipFile(data_zip) as archive:assert hashlib.sha256(archive.read(frozen.relative_to(ROOT).as_posix())).hexdigest()==sha256(frozen)
        write_json(manifest,{'created':now(),'file':data_zip.name,'sha256':sha256(data_zip),
            'files':[{'file':frozen.relative_to(ROOT).as_posix(),'sha256':sha256(frozen),'bytes':frozen.stat().st_size}]})
    audit={'source_run':args.source,'checked_at':now(),'task_count':count,'template_families':len({t['template_family'] for t in tasks}),
        'categories':result['categories'],'reference_passed':count,'initial_rejected':count,'negative_rejected':count,
        'reference_tool_calls':calls,'reference_error_returns':errors,'empty_query_scenes':sum(row['empty_query_observed'] for row in rows),
        'diagnostic_tool_calls':diagnostic,'removed_containers':count*2,'remaining_owned_containers':0,
        'records_sha256':result['records_sha256'],'tasks_sha256':result['tasks_sha256'],'frozen_data_sha256':sha256(frozen),
        'compact_csv_sha256':sha256(csv_file),'data_archive_sha256':sha256(data_zip),'model_calls':0,
        'scope':'实际规则参考、文件字节和明确错误候选复核；场景共享任务族，不代表模型成绩'}
    write_json(ROOT/f'experiments/E14/{split}-audit.json',audit);print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
