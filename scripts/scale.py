"""顺序执行既定规模条件；等待已开始的本项目训练，避免重复占用显卡。"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import psutil

from archive import archive_run
from lab import ROOT, now, sha256, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def alive(config):
    try:
        process=psutil.Process(config['process_id'])
        age=abs(process.create_time()-datetime.fromisoformat(config['started']).timestamp())
        actual=process.cmdline()[1:];expected=config['command'][1:]
        same=len(actual)==len(expected) and all(a.replace('\\','/').lower()==b.replace('\\','/').lower() for a,b in zip(actual,expected))
        return age<120 and same and process.is_running()
    except (psutil.NoSuchProcess,psutil.AccessDenied,KeyError):
        return False


def active_run_config(run):
    resumes=sorted(run.glob('resume-*.json'))
    if resumes:
        config=read(resumes[-1])
        if {'process_id','started','command'}<=config.keys():return config
    return read(run/'config.json')


def launch(command,label,log_directory=None):
    log_directory=log_directory or ROOT/'.local/logs'
    log_directory.mkdir(parents=True,exist_ok=True)
    log=log_directory/(label+'-'+str(time.time_ns())+'.log')
    write_json(ROOT/'.local/scale-state.json',{'status':'running','time':now(),'command':command,'log':log.relative_to(ROOT).as_posix()})
    with log.open('a',encoding='utf-8') as handle:
        handle.write('\n启动：'+now()+'\n');handle.flush()
        result=subprocess.run([sys.executable,*command],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{label} 退出码 {result.returncode}，先保留失败证据并排错')
    return log


def verify_reference(reference):
    desired={**read(ROOT/reference['training_config']),'train_size':reference['size'],'seed':reference['seed'],
             'evaluation_prompt_sha256':sha256(ROOT/'configs/prompt-frozen.json')}
    matches=[p.parent for p in sorted((ROOT/'.local/runs').glob(reference['experiment']+'-R*/config.json'))
             if all(read(p).get(k)==v for k,v in desired.items())]
    if not matches or not (matches[-1]/'result.json').exists():
        raise RuntimeError('同条件参考训练尚未结束，不能启动对照队列')
    run=matches[-1];result=read(run/'result.json')
    assert result['status']=='trained_pending_tool_eval' and not alive(read(run/'config.json'))
    assert result['train']['independent_trajectories']==reference['size'] and result['dev']['independent_trajectories']==500
    expected_steps=(result['train']['assistant_units']+desired['gradient_accumulation']-1)//desired['gradient_accumulation']
    assert result['steps']==expected_steps and desired['epochs']==1
    adapter_hash=sha256(run/'selected-adapter/adapter_model.safetensors')
    evaluations=[]
    for path in (ROOT/'experiments'/reference['experiment']/'runs').glob('*.json'):
        data=read(path);config=data['config']
        if (data['status']=='completed' and config.get('kind')=='tool_eval' and config.get('source_train_run')==run.name
            and config.get('adapter_sha256')==adapter_hash and config.get('frozen_prompt_sha256')==desired['evaluation_prompt_sha256']):
            summary=data['summaries'][data['selected_prompt']]
            assert summary['trajectories']==summary['evaluated_trajectories']==500 and summary['decision_turns']==928
            assert sha256(ROOT/'.local/runs'/data['run_id']/(data['selected_prompt']+'.jsonl'))==summary['rows_sha256']
            evaluations.append(data['run_id'])
    if not evaluations:raise RuntimeError('同条件参考尚未完成完整 dev 工具评测')
    print(json.dumps({'reference_train':run.name,'reference_dev':evaluations,'adapter_sha256':adapter_hash}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/scale.json');args=parser.parse_args()
    plan=read(ROOT/args.config)
    lock=ROOT/'.local/scale-lock.json'
    if lock.exists():
        previous=read(lock)
        if psutil.pid_exists(previous['pid']) and abs(psutil.Process(previous['pid']).create_time()-previous['created'])<1:
            raise RuntimeError('规模实验队列已经运行，不能重复启动')
    write_json(lock,{'pid':os.getpid(),'created':psutil.Process().create_time(),'command':sys.argv})
    try:
        if 'reference' in plan:verify_reference(plan['reference'])
        for job in plan['jobs']:
            training_config=job.get('training_config',plan['training_config'])
            base=read(ROOT/training_config)
            desired={**base,'train_size':job['size'],'seed':job['seed'],
                     'evaluation_prompt_sha256':sha256(ROOT/'configs/prompt-frozen.json')}
            candidates=[]
            for path in sorted((ROOT/'.local/runs').glob(plan['experiment']+'-R*/config.json')):
                config=read(path)
                if all(config.get(k)==v for k,v in desired.items()):candidates.append(path.parent)
            run=candidates[-1] if candidates else None
            if run and not (run/'result.json').exists():
                config=active_run_config(run)
                if alive(config):print('等待已开始的训练：'+run.name,flush=True)
                while not (run/'result.json').exists() and alive(config):
                    write_json(ROOT/'.local/scale-state.json',{'status':'waiting_existing','time':now(),'run_id':run.name,'job':job})
                    time.sleep(20)
                if not (run/'result.json').exists():
                    checkpoints=sorted((p for p in run.glob('checkpoint-*/manifest.json')
                                        if (p.parent/'trainer_state.json').exists()),
                                       key=lambda p:int(p.parent.name.split('-')[-1]))
                    if not checkpoints:
                        raise RuntimeError(f'{run.name} 进程已结束且没有可恢复 checkpoint；保留失败证据，不另起同条件训练')
                    resume=['scripts/sft.py','--config',training_config,'--experiment',plan['experiment'],
                            '--size',str(job['size']),'--seed',str(job['seed']),'--resume-run',run.name]
                    launch(resume,f"scale-{job['size']}-{job['seed']}-resume",run)
                # result 写入后，训练还会收尾文档；等模型进程真正退出再做生成。
                while alive(config):time.sleep(5)
            if not run or read(run/'result.json')['status']!='trained_pending_tool_eval':
                log=launch(['scripts/sft.py','--config',training_config,'--experiment',plan['experiment'],
                        '--size',str(job['size']),'--seed',str(job['seed'])],f"scale-{job['size']}-{job['seed']}-train")
                candidates=sorted((ROOT/'.local/runs').glob(plan['experiment']+'-R*/config.json'))
                run=next(p.parent for p in reversed(candidates) if all(read(p).get(k)==v for k,v in desired.items()))
                write_json(run/'external-log.json',{'file':log.relative_to(ROOT).as_posix()})
            result=read(run/'result.json')
            assert result['status']=='trained_pending_tool_eval' and result['steps']>0
            adapter=run/'selected-adapter';adapter_hash=sha256(adapter/'adapter_model.safetensors')
            evaluations=[]
            for path in (ROOT/'experiments'/plan['experiment']/'runs').glob('*.json'):
                data=read(path);config=data['config']
                if (data['status']=='completed' and config.get('kind')=='tool_eval' and config.get('source_train_run')==run.name
                    and config.get('adapter_sha256')==adapter_hash and config.get('frozen_prompt_sha256')==desired['evaluation_prompt_sha256']):
                    evaluations.append(data)
            if not evaluations:
                log=launch(['scripts/offline_eval.py','--experiment',plan['experiment'],'--adapter',adapter.relative_to(ROOT).as_posix(),
                        '--source-run',run.name],f"scale-{job['size']}-{job['seed']}-dev")
                evaluations=[read(p) for p in sorted((ROOT/'experiments'/plan['experiment']/'runs').glob('*.json'))
                             if read(p).get('config',{}).get('source_train_run')==run.name and read(p)['status']=='completed']
                write_json(ROOT/'.local/runs'/evaluations[-1]['run_id']/'external-log.json',{'file':log.relative_to(ROOT).as_posix()})
            evaluation=evaluations[-1]
            summary=evaluation['summaries'][evaluation['selected_prompt']]
            assert summary['trajectories']==summary['evaluated_trajectories']==500 and summary['decision_turns']==928
            for run_id in [run.name,evaluation['run_id']]:archive_run(run_id)
            print(json.dumps({'time':now(),'job':job,'train_run':run.name,'eval_run':evaluation['run_id'],
                              'trajectory_passed':summary['trajectory_passed'],'denominator':500},ensure_ascii=False),flush=True)
        write_json(ROOT/'.local/scale-state.json',{'status':'completed','time':now(),'scope':plan['experiment']+'训练与完整dev工具评测；后续实验继续按spec执行'})
    except Exception as error:
        write_json(ROOT/'.local/scale-state.json',{'status':'failed','time':now(),'error':str(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
