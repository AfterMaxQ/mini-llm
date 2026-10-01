"""配对同一份完整 dev：轨迹胜负、逐轮变化和重采样区间。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from lab import ROOT, sha256, write_json
from offline_eval import canonical
from templates import read_records


def compare(eval_run):
    folder=ROOT/'.local/runs'/eval_run
    result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    frozen=json.loads((ROOT/'configs/prompt-frozen.json').read_text(encoding='utf-8'))
    assert result['status']=='completed' and config['split']=='dev'
    assert config['frozen_prompt_sha256']==sha256(ROOT/'configs/prompt-frozen.json')
    paths=[ROOT/'.local/runs'/frozen['source_run']/(frozen['selected_prompt']+'.jsonl'),
           folder/(result['selected_prompt']+'.jsonl')]
    records=read_records('dev.jsonl',config['data_run'])
    by_id={r['sample_id']:r for r in records};all_rows=[]
    for path in paths:
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert len(rows)==928 and {r['sample_id'] for r in rows}==set(by_id)
        all_rows.append({(r['sample_id'],r['message_index']):r for r in rows})
        assert len(all_rows[-1])==len(rows)
    baseline,student=all_rows
    assert baseline.keys()==student.keys()
    assert all(canonical(baseline[k]['expected'])==canonical(student[k]['expected']) for k in baseline)
    paired=[]
    for sample_id,record in by_id.items():
        keys=[k for k in baseline if k[0]==sample_id]
        a=all(baseline[k]['exact'] for k in keys);b=all(student[k]['exact'] for k in keys)
        paired.append({'sample_id':sample_id,'category':record['category'],
                       'baseline_passed':int(a),'student_passed':int(b),'difference':int(b)-int(a)})
    assert len(paired)==500
    public=ROOT/'experiments/E09'
    data=public/(eval_run+'-paired.csv')
    with data.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    differences=np.array([r['difference'] for r in paired],dtype=np.int8)
    # 同一批轨迹成对抽样；这个区间不包含重新训练的 seed 波动。
    generator=np.random.default_rng(2026)
    boot=differences[generator.integers(0,500,size=(10000,500))].mean(axis=1)*100
    cases=[]
    for preferred,improvement in [('glaive-60626',True),('glaive-19711',False)]:
        changed=[k for k in baseline if baseline[k]['exact']!=student[k]['exact']
                 and student[k]['exact']==improvement]
        if not changed:continue
        key=next((k for k in changed if k[0]==preferred),changed[0]);sample_id=key[0]
        a,b=baseline[key],student[key];message=by_id[sample_id]['messages'][key[1]-1]
        cases.append({'sample_id':sample_id,'message_index':key[1],'user':message['content'],
                      'expected':a['expected'],'baseline':a['response'],'student':b['response'],
                      'baseline_passed':a['exact'],'student_passed':b['exact']})
    summary={'eval_run':eval_run,'train_run':config['source_train_run'],'trajectories':500,
             'baseline_passed':sum(r['baseline_passed'] for r in paired),
             'student_passed':sum(r['student_passed'] for r in paired),
             'both_passed':sum(r['baseline_passed'] and r['student_passed'] for r in paired),
             'baseline_only':sum(r['difference']==-1 for r in paired),
             'student_only':sum(r['difference']==1 for r in paired),
             'both_failed':sum(not r['baseline_passed'] and not r['student_passed'] for r in paired),
             'decision_improvements':sum(not baseline[k]['exact'] and student[k]['exact'] for k in baseline),
             'decision_regressions':sum(baseline[k]['exact'] and not student[k]['exact'] for k in baseline),
             'difference_percentage_points':float(differences.mean()*100),
             'paired_bootstrap':{'repeats':10000,'seed':2026,'generator':'NumPy PCG64',
               'interval_percent':95,'percentile_interval':np.quantile(boot,[0.025,0.975]).tolist(),
               'unit':'一条完整dev轨迹；配对百分比差值，不含训练seed波动'},
             'paired_csv_sha256':sha256(data),'source_rows_sha256':[sha256(p) for p in paths],
             'analysis_code_sha256':sha256(Path(__file__)),'cases':cases}
    baseline_result=json.loads((paths[0].parent/'result.json').read_text(encoding='utf-8'))
    assert summary['baseline_passed']==baseline_result['summaries'][frozen['selected_prompt']]['trajectory_passed']
    assert summary['student_passed']==result['summaries'][result['selected_prompt']]['trajectory_passed']
    write_json(public/(eval_run+'-comparison.json'),summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('eval_run');args=parser.parse_args()
    print(json.dumps(compare(args.eval_run),ensure_ascii=False),flush=True)
