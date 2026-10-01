// 顺序执行冻结训练场景，逐条落盘；不创建模型进程或GPU容器。
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFile, writeFile, appendFile, unlink, rename } from 'node:fs/promises';
import path from 'node:path';
import { createSandbox, root, policy } from './pi_sandbox.mjs';
import { seed, judge, executeReference, digest } from './pi_tasks.mjs';

const python=path.join(root,'.local/venv-train/Scripts/python.exe');
function py(code,value){return execFileSync(python,['-c',"import json,sys;sys.path.insert(0,'scripts');from lab import ROOT,start_run,finish_run,write_json;"+code],
  {cwd:root,input:JSON.stringify(value),encoding:'utf8',windowsHide:true,env:{...process.env,PYTHONUTF8:'1'}}).trim();}
const arg=process.argv.indexOf('--config');
const config=JSON.parse(await readFile(path.join(root,arg<0?'configs/pi-reference-batch.json':process.argv[arg+1]),'utf8'));
const identity=JSON.parse(py("import psutil;p=psutil.Process(json.load(sys.stdin));print(json.dumps({'pid':p.pid,'created':p.create_time(),'command':p.cmdline()}))",process.pid));
const lock=path.join(root,'.local/pi-data-lock.json');
try{
  const previous=JSON.parse(await readFile(lock,'utf8'));
  const active=py("import psutil;v=json.load(sys.stdin);print(psutil.pid_exists(v['pid']) and abs(psutil.Process(v['pid']).create_time()-v['created'])<1)",previous);
  assert.notEqual(active,'True','Pi训练数据执行已有进程，不能重复启动');await unlink(lock);
}catch(error){if(error.code!=='ENOENT')throw error;}
await writeFile(lock,JSON.stringify(identity,null,2),{flag:'wx'});
let directory, finished=false;const began=performance.now();let lastReport=began;
const records=[];let calls=0,errors=0;
async function save(name,value){
  const file=path.join(directory,name),temporary=file+`.${process.pid}.${Date.now()}.tmp`;
  await writeFile(temporary,JSON.stringify(value,null,2),'utf8');
  for(let attempt=0;attempt<12;attempt++){
    try{await rename(temporary,file);return;}
    catch(error){if(!['EPERM','EACCES','EBUSY'].includes(error.code)||attempt===11)throw error;
      await new Promise(resolve=>setTimeout(resolve,Math.min(50*(attempt+1),200)));}
  }
}
function refresh(){
  for(const args of [['scripts/notes_pi_tasks.py'],['scripts/index.py'],['scripts/report.py','--volume','02']]){
    try{execFileSync(python,args,{cwd:root,stdio:'inherit',windowsHide:true,env:{...process.env,PYTHONUTF8:'1'}});}
    catch(error){console.error('文档更新失败，原始逐条记录保留：'+error.message);}
  }
}
try{
  const source=path.join(root,'.local/runs',config.source_scene_run);
  const manifest=JSON.parse(await readFile(path.join(source,'result.json'),'utf8'));
  assert.equal(manifest.status,'training_requests_frozen');
  const raw=await readFile(path.join(root,manifest.requests_file));assert.equal(digest(raw),manifest.requests_sha256);
  const requests=raw.toString('utf8').trim().split('\n').map(JSON.parse);
  assert.equal(requests.length,2000);assert.equal(config.target_references,1000);
  const tasks=requests.slice(0,config.target_references);assert(tasks.every(t=>t.split==='train'&&t.reference.length<=policy.max_tool_calls));
  directory=py("print(start_run('E14',json.load(sys.stdin)))",{...config,operation:'pi_reference_batch',
    command:identity.command,source_requests_sha256:manifest.requests_sha256,sandbox:policy,model:null});
  const runId=path.basename(directory);const saved=JSON.parse(await readFile(path.join(directory,'config.json'),'utf8'));
  saved.process_id=process.pid;saved.process_created=identity.created;await save('config.json',saved);
  await save('tasks.json',tasks);await writeFile(path.join(directory,'records.jsonl'),'','utf8');
  const progress=async()=>save('progress.json',{status:'running',completed:records.length,target:tasks.length,
    valid:records.filter(r=>r.reference.passed).length,failed:records.filter(r=>!r.reference.passed).length,
    reference_tool_calls:calls,reference_error_returns:errors,elapsed_seconds:(performance.now()-began)/1000});
  await progress();refresh();
  for(const task of tasks){
    const item={task_id:task.task_id,category:task.category,template_family:task.template_family,
      repository_family:task.repository_family,split:'train',scene_sha256:task.scene_sha256,started:new Date().toISOString()};
    const sandbox=await createSandbox();
    try{
      const inspection=await sandbox.inspect();const host=inspection.HostConfig;
      item.isolation={network:host.NetworkMode,readonly_root:host.ReadonlyRootfs,user:inspection.Config.User,
        binds:host.Binds??[],gpu_requests:host.DeviceRequests??[],cap_drop:host.CapDrop,security:host.SecurityOpt};
      assert.equal(item.isolation.network,'none');assert(item.isolation.readonly_root&&item.isolation.user==='1000:1000');
      assert(!item.isolation.binds.length&&!item.isolation.gpu_requests.length);
      await seed(sandbox,task.files);const events=[];
      item.reference_answer=await executeReference(task,sandbox,async(tool,input)=>{
        const event={call_id:`${runId}-${task.task_id}-${events.length+1}`,tool,input};const started=performance.now();
        try{event.result=await sandbox.tools.find(t=>t.name===tool).execute(event.call_id,input);event.is_error=Boolean(event.result.isError);}
        catch(error){event.error=error.message;event.is_error=true;}
        event.seconds=(performance.now()-started)/1000;events.push(event);return event;
      });
      item.reference_events=events;item.reference=await judge(task,sandbox,item.reference_answer,events);
      calls+=events.length;errors+=events.filter(e=>e.is_error).length;
    }finally{
      const removed=await sandbox.dispose();assert.equal(removed.code,0,removed.stderr);
      item.reference_removed={container:sandbox.name,container_id:sandbox.containerId,removed:true};
    }
    item.finished=new Date().toISOString();records.push(item);
    await appendFile(path.join(directory,'records.jsonl'),JSON.stringify(item)+'\n','utf8');await progress();
    if(records.length%25===0)console.log(JSON.stringify({run_id:runId,completed:records.length,valid:records.filter(r=>r.reference.passed).length,target:1000}));
    if(performance.now()-lastReport>=config.update_seconds*1000){refresh();lastReport=performance.now();}
  }
  await save('records.json',records);
  const valid=records.filter(r=>r.reference.passed).length;
  const result={status:valid===tasks.length?'reference_batch_verified':'criterion_not_met',exit_code:valid===tasks.length?0:1,
    operation:'pi_reference_batch',task_count:records.length,target_references:tasks.length,reference_passed:valid,
    reference_tool_calls:calls,reference_error_returns:errors,model_calls:0,dev_tasks:0,test_tasks:0,
    records_sha256:digest(await readFile(path.join(directory,'records.json'))),tasks_sha256:digest(await readFile(path.join(directory,'tasks.json'))),
    duration_seconds:(performance.now()-began)/1000,scope:config.scope};
  console.log(py("v=json.load(sys.stdin);print(json.dumps(finish_run(ROOT/'.local/runs'/v['id'],v['result']),ensure_ascii=False))",{id:runId,result}));
  finished=true;if(valid<tasks.length)process.exitCode=1;
  refresh();
  if(valid===tasks.length&&config.encode_after_success){
    const format=JSON.parse(await readFile(path.join(root,'configs/pi-reference-extension.json'),'utf8'));
    Object.assign(format,{source_run:runId,output_file:'train-reference.jsonl',scope:'1000个实际执行训练场景的完整表示与监督核验；模型迁移训练待执行'});
    const relative='.local/runtime-config/'+runId+'-encoding.json';
    py("v=json.load(sys.stdin);write_json(ROOT/v['path'],v['config'])",{path:relative,config:format});
    execFileSync(python,['scripts/pi_reference_data.py','--config',relative],{cwd:root,stdio:'inherit',windowsHide:true,env:{...process.env,PYTHONUTF8:'1'}});
  }
}catch(error){
  const failure=error.stack;console.error(failure);
  if(directory){await save(finished?'following-stage-error.json':'failure.json',{error:failure});
    if(!finished)py("v=json.load(sys.stdin);finish_run(ROOT/'.local/runs'/v['id'],v['result'])",{
      id:path.basename(directory),result:{status:'failed',exit_code:1,operation:'pi_reference_batch',error:failure,completed:records.length,target_references:config.target_references}});}
  process.exitCode=1;
}finally{await unlink(lock).catch(()=>{});}
