"""生成训练场景清单；改变真实内容，不把模板变体当作新模板。"""
import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import date, timedelta
from lab import ROOT, start_run, finish_run, sha256, write_json


def formatted(value): return json.dumps(value, ensure_ascii=False, indent=2)+'\n'
def hashed(value): return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def replace_strings(value, changes):
    if isinstance(value, str):
        for old,new in changes:value=value.replace(old,new)
        return value
    if isinstance(value, list):return [replace_strings(v,changes) for v in value]
    if isinstance(value, dict):return {replace_strings(k,changes):replace_strings(v,changes) for k,v in value.items()}
    return value


def variant(original, number, rng):
    t=copy.deepcopy(original);t.pop('negative',None);t.pop('wrong_answer',None)
    suffix=f'{number:04d}';family=t['template_family'];parameters={}
    def rename(changes):
        nonlocal t
        t=replace_strings(t,changes);t['template_family']=original['template_family'];t['repository_family']=original['repository_family']
    def file(name,value): t['files'][name]=formatted(value)
    def expected(value): t['expected']['json_value']=value
    if family=='train-router-window-budget':
        symbol='allowService'+suffix;key='serviceRequests'+suffix
        rename([('allowedInWindow',symbol),('windowRequests',key),('window-budget.mjs','window-'+suffix+'.mjs')])
        parameters={'symbol':symbol,'config_key':key}
    elif family=='train-profile-cache-json':
        service='catalog_'+suffix;rename([('search',service),('admin','ops_'+suffix)])
        old=rng.randint(35,80);target=rng.randint(100,500);admin=rng.randint(5,30)
        value=json.loads(t['files']['settings.json']);value['profiles'][service]['cache']['maxItems']=old;value['profiles']['ops_'+suffix]['cache']['maxItems']=admin
        file('settings.json',value);new=copy.deepcopy(value);new['profiles'][service]['cache']['maxItems']=target;expected(new)
        action=t['reference'][1]['input']['edits'][0];action.update(oldText=f'"maxItems": {old}',newText=f'"maxItems": {target}')
        t['prompt']=f'请把settings.json中{service}配置的缓存容量maxItems调整为{target}。其他配置保持原值，确认配置能解析。'
        parameters={'service':service,'old':old,'target':target,'other_capacity':admin}
    elif family=='train-half-open-booking':
        symbol='overlaps'+suffix;rename([('overlaps',symbol),('interval.mjs','interval-'+suffix+'.mjs')])
        start=rng.randint(-100,100);width=rng.randint(2,12)
        cases=[({'start':start,'end':start+width},{'start':start+width,'end':start+2*width},False),
               ({'start':start,'end':start+width+1},{'start':start+width,'end':start+2*width},True),
               ({'start':start,'end':start},{'start':start-1,'end':start+1},False),
               ({'start':start+width,'end':start},{'start':start,'end':start+width+1},False),
               ({'start':start-5,'end':start},{'start':start-2,'end':start+1},True),
               ({'start':start,'end':start+3*width},{'start':start+width,'end':start+2*width},True)]
        t['files']['checks.mjs']=f"import assert from 'node:assert/strict';\nimport {{{symbol}}} from './src/interval-{suffix}.mjs';\nconst cases={json.dumps(cases)};\nfor(const [a,b,result] of cases)assert.equal({symbol}(a,b),result);\nconsole.log('interval checks: 6 passed');\n"
        parameters={'start':start,'width':width,'symbol':symbol}
    elif family=='train-metrics-contract-migration':
        rename([('request_count','received_'+suffix),('served_requests','processed_'+suffix)])
        parameters={'old_key':'received_'+suffix,'new_key':'processed_'+suffix}
    elif family=='train-slug-whitespace-regression':
        rename([('slug.mjs','slug-'+suffix+'.mjs')])
        a='Item'+suffix;b='Batch'+str(rng.randint(100,999));cases=[(f' {a}  {b} ',f'{a.lower()}-{b.lower()}'),('C\tD','c-d'),('E\nF','e-f'),('x','x'),('  ','')]
        t['files']['checks.mjs']=f"import assert from 'node:assert/strict';\nimport {{normalize}} from './src/slug-{suffix}.mjs';\nfor(const [x,y] of {json.dumps(cases)})assert.equal(normalize(x),y);\nconsole.log('slug regression: 5 passed');\n"
        parameters={'first_word':a,'second_word':b}
    elif family=='train-unset-upload-timeout':
        timeout=rng.randint(10,90);rename([('upload.json','upload-'+suffix+'.json')]);value={'timeoutSeconds':timeout,'retries':rng.randint(1,4)};file('upload-'+suffix+'.json',value)
        parameters=value
    elif family=='train-explain-request-rate':
        rate=number+2;t['prompt']=f'只根据这句话回答，不查看文件：系统每秒允许{rate}次请求。请用一句话说明每秒最多能请求多少次，不修改项目。'
        t['expected'].update(answer_contains=['每秒',str(rate)],answer=f'这个限制表示每秒最多允许发起{rate}次请求。')
        parameters={'requests_per_second':rate}
    elif family=='train-stale-inventory-path':
        quantities=[rng.randint(1,80) for _ in range(3)];rename([('stock.csv','stock-'+suffix+'.csv')])
        current='exports/warehouse/stock-'+suffix+'.csv';t['files'][current]='sku,quantity\n'+''.join(f'P{suffix}{i},{n}\n' for i,n in enumerate(quantities))
        t['expected']['answer']={'file':current,'total':sum(quantities)};parameters={'quantities':quantities,'current_file':current}
    elif family=='train-selected-storage-registry':
        rename([('disk.mjs','disk-'+suffix+'.mjs'),('memory.mjs','memory-'+suffix+'.mjs')])
        chosen=rng.choice(['disk','memory']);value=json.loads(t['files']['config.json']);value['storage']['backend']=chosen;file('config.json',value)
        module='storage/'+chosen+'-'+suffix+'.mjs';symbol='readChunk' if chosen=='disk' else 'lookup'
        t['reference'][2]['input']['path']=module;t['expected']['answer']={'file':module,'symbol':symbol,'config':'storage.backend'}
        parameters={'selected_backend':chosen,'current_file':module}
    elif family=='train-layered-environment-override':
        base={'workers':rng.randint(3,8),'cache':{'slots':rng.randint(80,200),'ttlSeconds':rng.randint(60,180)}}
        extra={'workers':rng.randint(1,2),'cache':{'ttlSeconds':rng.randint(10,45)}};target=rng.randint(12,50)
        file('defaults.json',base);file('staging.json',extra);new=copy.deepcopy(extra);new['cache']['slots']=target;expected(new)
        t['reference'][2]['input']['content']=formatted(new);merged={**base,**new,'cache':{**base['cache'],**new['cache']}}
        t['files']['checks.mjs']=f"import assert from 'node:assert/strict';\nimport {{load}} from './lib/load.mjs';\nassert.deepEqual(load('staging'),{json.dumps(merged)});\nassert.deepEqual(load('production'),{json.dumps(base)});\nconsole.log('layered config: passed');\n"
        t['prompt']=f'staging继承defaults.json，再应用staging.json。只把staging缓存槽位改成{target}；生产默认值、staging的TTL与worker数量保持原值。保留检查并运行。';parameters={'defaults':base,'staging':extra,'target':target}
    elif family=='train-last-occurrence-order':
        seed=rng.randrange(1,2**31);t['files']['checks.mjs']=t['files']['checks.mjs'].replace('seed=73',f'seed={seed}')
        rename([('latest.mjs','latest-'+suffix+'.mjs')]);parameters={'case_seed':seed,'file':'src/latest-'+suffix+'.mjs'}
    elif family=='train-duration-unit-migration':
        milliseconds=625+((number-1)//16)*125;retries=rng.randint(1,5);cfg={'refreshMs':milliseconds,'retries':retries};new={'refreshSeconds':milliseconds/1000,'retries':retries}
        file('config.json',cfg);file('notes/old-config.json',cfg);expected(new);t['reference'][3]['input']['content']=formatted(new)
        checks=t['files']['checks.mjs'].replace('{refreshSeconds:2.5,retries:3}',json.dumps(new)).replace('delay(config),2500',f'delay(config),{milliseconds}')
        t['files']['checks.mjs']=checks;t['prompt']=f'刷新配置从refreshMs迁移为refreshSeconds，同步config.json、schema.json和延时模块。保持实际延时{milliseconds}毫秒及retries={retries}，允许小数秒。保留原检查与旧版归档。'
        parameters={'milliseconds':milliseconds,'seconds':milliseconds/1000,'retries':retries}
    elif family=='train-moved-module-import':
        rename([('order.mjs','order-'+suffix+'.mjs'),('fees.mjs','fees-'+suffix+'.mjs')]);parameters={'entry':'src/order-'+suffix+'.mjs','module':'src/pricing/fees-'+suffix+'.mjs'}
    elif family=='train-unspecified-schedule-zone':
        zone=rng.choice(['UTC','Asia/Tokyo','Europe/London']);value=json.loads(t['files']['schedule.json']);value.update(timezone=zone,localTime=f'{rng.randint(6,8):02d}:00');file('schedule.json',value)
        hour=rng.randint(9,17);requested=(date(2026,11,1)+timedelta(days=(number-1)//16)).isoformat()
        t['prompt']=f'请把提醒设为{requested}的{hour}:00。schedule.json现用{zone}，但这次没有说明按哪个时区。先查看配置并确认缺少的信息，不要猜我的时区。';parameters={'current':value,'requested_hour':hour,'requested_date':requested}
    elif family=='train-given-rate-budget-calculation':
        rate=rng.randint(2,30);seconds=number+10;answer={'maxRequests':rate*seconds,'seconds':seconds,'requestsPerSecond':rate};t['expected']['answer']=answer
        t['prompt']=f'仅根据已给信息计算，不查看文件或运行命令：速率上限每秒{rate}次，窗口{seconds}秒。忽略突发额度，最多请求多少次？用JSON返回maxRequests、seconds、requestsPerSecond。';parameters=answer
    elif family=='train-current-stock-active-join':
        rename([('current-stock.json','current-stock-'+suffix+'.json')]);current='reports/current-stock-'+suffix+'.json'
        skus=['P'+suffix+str(i) for i in range(3)];active=set(rng.sample(skus,2));quantities=[rng.randint(1,70) for _ in skus]
        file('products.json',[{'sku':sku,'active':sku in active} for sku in skus]);file(current,[{'sku':sku,'quantity':n} for sku,n in zip(skus,quantities)]+[{'sku':'UNKNOWN','quantity':999}])
        t['expected']['answer']={'file':current,'total':sum(n for sku,n in zip(skus,quantities) if sku in active),'includedSkus':sorted(active)}
        parameters={'skus':skus,'active':sorted(active),'quantities':quantities}
    else:raise ValueError(family)
    t.update(task_id=f'train-{suffix}-{original["category"]}',split='train',variant_parameters=parameters,
             source_prototype_id=original['task_id'],source_run=original['source_run'],reference_source='task_rules_pending_execution')
    return t


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/pi-task-data.json');args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding='utf-8'));sources=[];prototypes=[]
    for name in config['source_runs']:
        folder=ROOT/'.local/runs'/name;result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        assert result['reference_passed']==result['task_count'] and sha256(folder/'tasks.json')==result['tasks_sha256']
        for task in json.loads((folder/'tasks.json').read_text(encoding='utf-8')):prototypes.append({**task,'source_run':name})
        sources.append({'run_id':name,'tasks_sha256':sha256(folder/'tasks.json'),'result_sha256':sha256(folder/'result.json')})
    assert len(prototypes)==len({p['template_family'] for p in prototypes})==16
    config.update(command=[sys.executable,*sys.argv],sources=sources);run=start_run('E14',config)
    try:
        rng=random.Random(config['seed']);tasks=[variant(prototypes[i%16],i+1,rng) for i in range(config['training_requests'])]
        hashes=[hashed({'prompt':t['prompt'],'files':t['files']}) for t in tasks]
        repeated={h for h,n in Counter(hashes).items() if n>1}
        if repeated:write_json(run/'duplicate-scenes.json',[{'scene_sha256':h,'task_id':t['task_id'],
            'family':t['template_family'],'parameters':t['variant_parameters']} for t,h in zip(tasks,hashes) if h in repeated])
        assert len(hashes)==len(set(hashes))==2000, '场景存在重复，不能冻结请求清单'
        ids=[t['task_id'] for t in tasks];assert len(ids)==len(set(ids))
        output=ROOT/'.local/data/pi'/run.name;output.mkdir(parents=True,exist_ok=False);raw=output/'requests.jsonl'
        raw.write_text(''.join(json.dumps({**t,'scene_sha256':h},ensure_ascii=False)+'\n' for t,h in zip(tasks,hashes)),encoding='utf-8')
        result={'status':'training_requests_frozen','exit_code':0,'operation':'pi_training_scene_generation',
                'requests':len(tasks),'reference_request_target':config['reference_requests'],'template_families':16,
                'family_counts':dict(Counter(t['template_family'] for t in tasks)),
                'categories':dict(Counter(t['category'] for t in tasks)),
                'reference_prefix_categories':dict(Counter(t['category'] for t in tasks[:config['reference_requests']])),
                'exact_scene_duplicates':0,'near_repeat_policy':'16个共享模板族，族内变体全部train；不宣称2000个独立模板',
                'requests_file':raw.relative_to(ROOT).as_posix(),'requests_sha256':sha256(raw),'scene_hashes_sha256':hashed(hashes),
                'valid_executed_references':0,'model_calls':0,'dev_tasks':0,'test_tasks':0,
                'scope':'2000个训练场景已冻结，参考步骤尚待逐条本机执行；数量不计为已完成有效轨迹'}
        write_json(output/'manifest.json',result);finish_run(run,result);print(json.dumps(result,ensure_ascii=False))
    except Exception:
        import traceback
        finish_run(run,{'status':'failed','exit_code':1,'error':traceback.format_exc()});raise


if __name__=='__main__':main()
