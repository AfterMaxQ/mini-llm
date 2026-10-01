// 用真实工具检查dev/test参考与判据；不运行模型或生成训练标签。
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile,writeFile,appendFile,rename,unlink,mkdir} from 'node:fs/promises';
import path from 'node:path';
import {createSandbox,root,policy} from './pi_sandbox.mjs';
import {seed,judge,executeReference,digest} from './pi_tasks.mjs';
import {devTasks} from './pi_tasks_dev.mjs';
import {testTasks} from './pi_tasks_test.mjs';

const python=path.join(root,'.local/venv-train/Scripts/python.exe');
const py=(code,value)=>execFileSync(python,['-c',"import json,sys;sys.path.insert(0,'scripts');from lab import ROOT,start_run,finish_run,write_json;"+code],
  {cwd:root,input:JSON.stringify(value),encoding:'utf8',windowsHide:true,env:{...process.env,PYTHONUTF8:'1'}}).trim();
const config=JSON.parse(await readFile(path.join(root,process.argv[2]??'configs/pi-dev-tasks.json'),'utf8'));
assert(['dev','test'].includes(config.split));
const split=config.split,target=config.task_count,familyCount=config.template_families??8;
const identity=JSON.parse(py("import psutil;p=psutil.Process(json.load(sys.stdin));print(json.dumps({'pid':p.pid,'created':p.create_time(),'command':p.cmdline()}))",process.pid));
const lock=path.join(root,`.local/pi-${split}-lock.json`);
try{
  const previous=JSON.parse(await readFile(lock,'utf8'));
  assert.notEqual(py("import psutil;v=json.load(sys.stdin);print(psutil.pid_exists(v['pid']) and abs(psutil.Process(v['pid']).create_time()-v['created'])<.01)",previous),'True',`已有${split}核验进程`);
  await unlink(lock);
}catch(error){if(error.code!=='ENOENT')throw error;}
await writeFile(lock,JSON.stringify(identity,null,2),{flag:'wx'});
let directory;const records=[];const started=performance.now();
const canonical=value=>Array.isArray(value)?value.map(canonical):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(k=>[k,canonical(value[k])])):value;
const scene=task=>digest(JSON.stringify(canonical({prompt:task.prompt,files:task.files})));
const count=key=>Object.fromEntries([...new Set(tasks.map(t=>t[key]))].map(value=>[value,tasks.filter(t=>t[key]===value).length]));
let tasks,failure,current;
async function save(name,value){
  const file=path.join(directory,name),temporary=file+`.${process.pid}.${Date.now()}.tmp`;
  await writeFile(temporary,JSON.stringify(value,null,2),'utf8');
  for(let n=0;n<12;n++){
    try{await rename(temporary,file);return;}catch(error){
      if(!['EPERM','EACCES','EBUSY'].includes(error.code)||n===11)throw error;
      await new Promise(resolve=>setTimeout(resolve,Math.min(50*(n+1),200)));
    }
  }
}
async function call(sandbox,events,tool,input){
  const event={call_id:`${path.basename(directory)}-${sandbox.name}-${events.length+1}`,tool,input};const began=performance.now();
  try{event.result=await sandbox.tools.find(t=>t.name===tool).execute(event.call_id,input);event.is_error=Boolean(event.result.isError);}
  catch(error){event.error=error.message;event.is_error=true;}
  event.seconds=(performance.now()-began)/1000;events.push(event);return event;
}
async function execute(task,kind,entry){
  const sandbox=await createSandbox();
  try{
    const inspection=await sandbox.inspect();const host=inspection.HostConfig;
    entry[kind+'_isolation']={network:host.NetworkMode,readonly_root:host.ReadonlyRootfs,user:inspection.Config.User,
      binds:host.Binds??[],gpu_requests:host.DeviceRequests??[]};
    assert.equal(host.NetworkMode,'none');assert(host.ReadonlyRootfs&&inspection.Config.User==='1000:1000');
    assert(!host.Binds?.length&&!host.DeviceRequests?.length);await seed(sandbox,task.files);
    const events=[];entry[kind+'_events']=events;
    if(kind==='control'){
      entry.initial=await judge(task,sandbox,task.wrong_answer,[]);assert(!entry.initial.passed,task.task_id+'初始错误被接受');
      for(const action of task.negative.actions)await call(sandbox,events,action.tool,action.input);
      entry.negative=await judge(task,sandbox,task.negative.answer,events);assert(!entry.negative.passed,task.task_id+'错误候选被接受');
    }else{
      entry.reference_answer=await executeReference(task,sandbox,(tool,input)=>call(sandbox,events,tool,input));
      entry.reference=await judge(task,sandbox,entry.reference_answer,events);
      assert(entry.reference.passed,task.task_id+':'+entry.reference.reasons.join(','));assert(events.length<=policy.max_tool_calls);
    }
  }finally{
    const removed=await sandbox.dispose();assert.equal(removed.code,0,removed.stderr);
    entry[kind+'_removed']={container:sandbox.name,container_id:sandbox.containerId,removed:true};
  }
}
try{
  const source=JSON.parse(await readFile(path.join(root,'.local/runs',config.training_scene_run,'result.json'),'utf8'));
  const raw=await readFile(path.join(root,source.requests_file));assert.equal(digest(raw),source.requests_sha256);
  const training=raw.toString('utf8').trim().split('\n').map(JSON.parse);
  tasks=(split==='dev'?devTasks():testTasks()).map(t=>({...t,scene_sha256:scene(t)}));
  directory=py("print(start_run('E14',json.load(sys.stdin)))",{...config,operation:`pi_${split}_probe`,command:identity.command,
    training_requests_sha256:source.requests_sha256,sandbox:policy,model:null});
  const saved=JSON.parse(await readFile(path.join(directory,'config.json'),'utf8'));
  saved.process_id=process.pid;saved.process_created=identity.created;await save('config.json',saved);await save('tasks.json',tasks);
  assert.equal(tasks.length,target);assert.equal(new Set(tasks.map(t=>t.task_id)).size,target);
  assert.equal(new Set(tasks.map(t=>t.scene_sha256)).size,target);assert.equal(new Set(tasks.map(t=>t.template_family)).size,familyCount);
  assert(tasks.every(t=>t.split===split&&t.reference.length<=12));
  if(config.category_quotas)assert.deepEqual(count('category'),config.category_quotas);
  assert(training.every(t=>scene(t)===t.scene_sha256),'训练区场景哈希表示不一致');
  const overlap={};
  const comparison={train:training};
  if(config.dev_scene_run){
    const devRoot=path.join(root,'.local/runs',config.dev_scene_run);
    const result=JSON.parse(await readFile(path.join(devRoot,'result.json'),'utf8'));
    assert.equal(result.status,'dev_reference_verified');
    const file=await readFile(path.join(devRoot,'tasks.json'));assert.equal(digest(file),result.tasks_sha256);
    comparison.dev=JSON.parse(file);saved.dev_tasks_sha256=result.tasks_sha256;await save('config.json',saved);
  }
  for(const [name,data] of Object.entries(comparison))for(const key of ['scene_sha256','template_family','repository_family']){
    const known=new Set(data.map(t=>t[key]));overlap[name+'_'+key]=tasks.filter(t=>known.has(t[key])).length;assert.equal(overlap[name+'_'+key],0,key);
  }
  await save('split-inspection.json',{overlap,categories:count('category'),families:count('template_family'),
    policy:`${familyCount}个${split}模板族共${target}个不同场景；族内相关性保留，不宣称每个场景都是新模板`});
  await writeFile(path.join(directory,'records.jsonl'),'','utf8');
  for(const task of tasks){
    current={task_id:task.task_id,category:task.category,split,template_family:task.template_family,
      repository_family:task.repository_family,scene_sha256:task.scene_sha256,started:new Date().toISOString()};
    await execute(task,'control',current);await execute(task,'reference',current);current.finished=new Date().toISOString();
    records.push(current);await appendFile(path.join(directory,'records.jsonl'),JSON.stringify(current)+'\n','utf8');
    await save('progress.json',{status:'running',completed:records.length,target,reference_passed:records.filter(r=>r.reference.passed).length});
    console.log(JSON.stringify({task:task.task_id,reference_calls:current.reference_events.length,reference_passed:true}));
    current=null;
  }
  const output=path.join(root,'.local/data/pi',path.basename(directory));await mkdir(output,{recursive:false});
  const taskFile=path.join(output,split+'.json');await writeFile(taskFile,JSON.stringify(tasks,null,2),'utf8');
  config[`frozen_${split}_file`]=path.relative(root,taskFile).replaceAll('\\','/');config[`frozen_${split}_sha256`]=digest(await readFile(taskFile));
}catch(error){failure=error.stack;console.error(failure);if(directory)await save('failure.json',{error:failure,partial_task:current});}
finally{
  if(directory){
    await save('records.json',records);
    const result={status:failure?'failed':`${split}_reference_verified`,exit_code:failure?1:0,operation:`pi_${split}_probe`,error:failure,
      target_tasks:target,task_count:records.length,template_families:familyCount,categories:count('category'),
      reference_passed:records.filter(r=>r.reference.passed).length,initial_rejected:records.filter(r=>!r.initial.passed).length,
      negative_rejected:records.filter(r=>!r.negative.passed).length,
      reference_tool_calls:records.reduce((n,r)=>n+r.reference_events.length,0),
      reference_error_returns:records.reduce((n,r)=>n+r.reference_events.filter(e=>e.is_error).length,0),
      diagnostic_tool_calls:records.reduce((n,r)=>n+r.control_events.length,0),
      records_sha256:digest(await readFile(path.join(directory,'records.json'))),tasks_sha256:digest(await readFile(path.join(directory,'tasks.json'))),
      [`frozen_${split}_file`]:config[`frozen_${split}_file`],[`frozen_${split}_sha256`]:config[`frozen_${split}_sha256`],
      duration_seconds:(performance.now()-started)/1000,model_calls:0,dev_tasks:!failure&&split==='dev'?target:0,test_tasks:!failure&&split==='test'?target:0,scope:config.scope};
    console.log(py("v=json.load(sys.stdin);print(json.dumps(finish_run(ROOT/'.local/runs'/v['id'],v['result']),ensure_ascii=False))",{id:path.basename(directory),result}));
  }
  await unlink(lock).catch(()=>{});if(failure)process.exitCode=1;
}
