// 使用冻结的 Pi dev 场景、原生 Pi 会话和隔离工具完成模型评测。
import assert from 'node:assert/strict';
import {spawn,spawnSync,execFileSync} from 'node:child_process';
import {existsSync,readFileSync,writeFileSync,unlinkSync} from 'node:fs';
import {appendFile,readFile,rename,unlink,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createSandbox,policy,root} from './pi_sandbox.mjs';
import {devTasks} from './pi_tasks_dev.mjs';
import {testTasks} from './pi_tasks_test.mjs';
import {seed,judge,digest} from './pi_tasks.mjs';

const configPath=process.argv[2]??'configs/pi-agent-eval.json';
const resumeArg=process.argv.indexOf('--resume-run');
const resumeRun=resumeArg>=0?process.argv[resumeArg+1]:null;
const config=JSON.parse(await readFile(path.join(root,configPath),'utf8'));
const experiment=config.experiment??'E13';
const split=config.split??'dev';
const sourceTasks=split==='dev'?devTasks:split==='test'?testTasks:null;
const packageRoot=path.join(root,'.local/pi/node_modules/@earendil-works/pi-coding-agent');
const pi=await import(pathToFileURL(path.join(packageRoot,'dist/index.js')).href);
const {createToolDefinitionFromAgentTool}=await import(pathToFileURL(path.join(packageRoot,'dist/core/tools/tool-definition-wrapper.js')).href);
const python=path.join(root,'.local/venv-train/Scripts/python.exe');
const lockPath=path.join(root,'.local/pi-agent-lock.json');
const scaleLockPath=path.join(root,'.local/scale-lock.json');
const zero='0'.repeat(64);

function py(code,value={}){
  return execFileSync(python,['-c',"import json,sys;sys.path.insert(0,'scripts');from lab import ROOT,start_run,finish_run,write_json,sha256,now;"+code],
    {cwd:root,input:JSON.stringify(value),encoding:'utf8',windowsHide:true,env:{...process.env,PYTHONUTF8:'1'}}).trim();
}
async function save(runDir,name,value){
  const file=path.join(runDir,name),temporary=file+`.${process.pid}.${Date.now()}.tmp`;
  await writeFile(temporary,JSON.stringify(value,null,2),'utf8');
  await rename(temporary,file);
}
async function saveProgress(runDir,progress){await save(runDir,'progress.json',progress);}
function identity(){
  return JSON.parse(py("import psutil;p=psutil.Process(json.load(sys.stdin)['pid']);print(json.dumps({'pid':p.pid,'created':p.create_time(),'command':p.cmdline()}))",{pid:process.pid}));
}
function checkGpuLock(){
  if(existsSync(scaleLockPath)){
    const previous=JSON.parse(readFileSync(scaleLockPath,'utf8'));
    const alive=py("import psutil;v=json.load(sys.stdin);print(psutil.pid_exists(v['pid']) and abs(psutil.Process(v['pid']).create_time()-v['created'])<1)",previous)==='True';
    throw new Error(alive?'已有GPU队列或任务正在运行':'发现失效GPU锁；核对原运行记录后再继续');
  }
  scaleLockIdentity=identity();
  writeFileSync(scaleLockPath,JSON.stringify(scaleLockIdentity,null,2),{flag:'wx'});
  scaleLockOwned=true;
}
function checkLock(){
  try{
    const old=JSON.parse(readFileSync(lockPath,'utf8'));
    const alive=py("import psutil;v=json.load(sys.stdin);print(psutil.pid_exists(v['pid']) and abs(psutil.Process(v['pid']).create_time()-v['created'])<.01)",old)==='True';
    assert(!alive,'已有 Pi 模型评测进程运行');
    if(!resumeRun)throw new Error('发现失效的 Pi 模型评测锁；请使用原运行号核对并恢复');
    unlinkSync(lockPath);
  }catch(error){if(error.code!=='ENOENT')throw error;}
  writeFileSync(lockPath,JSON.stringify(identity(),null,2),{flag:'wx'});
}
function transcriptState(messages){
  return messages.map(message=>{
    if(message.role==='assistant')return {role:message.role,content:message.content};
    if(message.role==='toolResult')return {role:message.role,toolCallId:message.toolCallId,
      toolName:message.toolName,isError:message.isError,content:message.content};
    return {role:message.role,content:message.content};
  });
}
function observedEvents(messages){
  const calls=messages.flatMap(message=>message.role==='assistant'?message.content.filter(block=>block.type==='toolCall'):[]);
  const results=messages.filter(message=>message.role==='toolResult');
  return calls.map(call=>{
    const result=results.find(item=>item.toolCallId===call.id);
    return {tool:call.name,input:call.arguments,is_error:Boolean(result?.isError),
      result:result?{content:result.content}:undefined};
  });
}
function finalAnswer(messages){
  for(const message of [...messages].reverse()){
    if(message.role!=='assistant')continue;
    const content=message.content.filter(block=>block.type==='text').map(block=>block.text).join('');
    if(content.trim())return content;
  }
  return '';
}
function counts(rows){
  return Object.fromEntries([...new Set(rows.map(row=>row.category))].sort().map(category=>[category,
    {tasks:rows.filter(row=>row.category===category).length,passed:rows.filter(row=>row.category===category&&row.passed).length}]));
}
async function launchApi(runDir){
  const trace=path.join(runDir,'model-api.jsonl');
  const child=spawn(python,[path.join(root,'scripts/pi_agent_api.py'),'--config',configPath,'--run-id',path.basename(runDir),'--trace',trace],
    {cwd:root,windowsHide:true,stdio:['ignore','pipe','pipe'],env:{...process.env,PYTHONUTF8:'1'}});
  const stdout=appendFile(path.join(runDir,'model-api.stdout.txt'),'','utf8');await stdout;
  child.stdout.on('data',data=>void appendFile(path.join(runDir,'model-api.stdout.txt'),data));
  child.stderr.on('data',data=>void appendFile(path.join(runDir,'model-api.stderr.txt'),data));
  const started=Date.now();
  while(Date.now()-started<600000){
    if(child.exitCode!==null||child.signalCode!==null)throw new Error(`本地模型服务提前退出，exit=${child.exitCode}, signal=${child.signalCode}`);
    try{
      const response=await fetch(`http://${config.api.host}:${config.api.port}/health`);
      const body=await response.json();
      if(body.status==='ready'&&body.run_id===path.basename(runDir)&&body.model===config.api.model)return child;
    }catch{}
    await new Promise(resolve=>setTimeout(resolve,1000));
  }
  child.kill();throw new Error('本地模型服务在十分钟内未就绪');
}
async function stopApi(child){
  if(!child||child.exitCode!==null)return;
  child.kill();
  await new Promise(resolve=>child.once('exit',resolve));
}

let runDir,api,rows=[],current=null,failure,lastNotes=Date.now(),lockOwned=false,scaleLockOwned=false,scaleLockIdentity;
const initialPromptHashes=new Map();
const started=Date.now();
function updateNotes(){
  if(experiment==='E15'){
    spawnSync(python,['scripts/notes_teacher.py','--run',path.basename(runDir)],{cwd:root,windowsHide:true});
    return;
  }
  const script=config.notes_script??(experiment==='E14'?'scripts/notes_domain.py':'scripts/notes_pi_agent.py');
  const args=experiment==='E14'?['--run',config.source_run,'--pi-run',path.basename(runDir)]:['--run',path.basename(runDir)];
  spawnSync(python,[script,...args],{cwd:root,windowsHide:true});
  spawnSync(python,['scripts/report.py','--volume','02B'],{cwd:root,windowsHide:true});
}
try{
  checkGpuLock();
  checkLock();lockOwned=true;
  assert.equal(policy.pi_version,'0.99.1');
  assert(['E13','E14','E15'].includes(experiment));
  assert(['dev','test'].includes(split));
  assert(sourceTasks);
  assert.equal(config.operation,experiment==='E15'?'pi_teacher_agent_dev':`pi_model_agent_${split}`);
  if(experiment==='E13'){
    assert.equal(split,'dev');
    assert.equal(config.source_run,'E09-R15');
    assert.equal(config.source_eval_run,'E09-R16');
  }else if(experiment==='E14'){
    const source=JSON.parse(await readFile(path.join(root,'.local/runs',config.source_run,'result.json'),'utf8'));
    assert.equal(source.status,'trained_pending_tool_eval');
    if(config.source_eval_run){
      const evaluation=JSON.parse(await readFile(path.join(root,'experiments/E14/runs',`${config.source_eval_run}.json`),'utf8'));
      assert.equal(evaluation.status,'completed');
      assert.equal(evaluation.config.source_train_run,config.source_run);
    }
  }else{
    const source=JSON.parse(await readFile(path.join(root,'.local/runs',config.source_run,'result.json'),'utf8'));
    const evaluation=JSON.parse(await readFile(path.join(root,'experiments/E14/runs',`${config.source_eval_run}.json`),'utf8'));
    assert.equal(config.split,'dev');
    assert.equal(config.frozen_subset,'pi_dev');
    assert.equal(source.status,'trained_pending_tool_eval');
    assert.equal(evaluation.status,'completed');
    assert.equal(evaluation.validity,'valid_harness');
    assert.equal(evaluation.input_delivery_verified,true);
    assert.equal(evaluation.config.source_run,config.source_run);
    assert.equal(evaluation.target_tasks,config.task_count);
    assert.equal(evaluation.config.ids_sha256,config.ids_sha256);
    assert.equal(config.quantization,'NF4');
    assert.equal(config.load_in_4bit,true);
    assert.equal(config.tokenizer_template_sha256,config.student_tokenizer_template_sha256);
    assert.equal(config.model_manifest_sha256,py("print(sha256(ROOT/json.load(sys.stdin)['model_manifest_path']))",config));
  }
  assert.equal(config.selected_prompt,'few_shot');
  assert.equal(config.prompt_sha256,py("print(sha256(ROOT/json.load(sys.stdin)['prompt_config']))",config));
  if(experiment!=='E15')assert.equal(config.adapter_sha256,py("import json,sys;print(sha256(ROOT/json.load(sys.stdin)['adapter']/'adapter_model.safetensors'))",config));
  const frozen=JSON.parse(await readFile(path.join(root,'configs/subsets-frozen.json'),'utf8'))[config.frozen_subset];
  assert.equal(frozen.selected_count,config.task_count);
  assert.equal(frozen.ids_sha256,config.ids_sha256);
  assert.equal(py("import hashlib;print(hashlib.sha256(json.dumps(json.load(sys.stdin),ensure_ascii=False,sort_keys=True).encode()).hexdigest())",frozen.ids),frozen.ids_sha256);
  assert.equal(config.frozen_subset,`pi_${split}`);
  const all=sourceTasks(),byId=new Map(all.map(task=>[task.task_id,task]));
  assert.equal(all.length,frozen.source_count);
  const tasks=frozen.ids.map(id=>byId.get(id));
  assert(tasks.every(Boolean));
  const quotas=tasks.reduce((out,task)=>(out[task.category]=(out[task.category]??0)+1,out),{});
  assert.deepEqual(quotas,frozen.strata);
  const taskFile=split==='dev'?'scripts/pi_tasks_dev.mjs':'scripts/pi_tasks_test.mjs';
  const runConfig={...config,command:[process.execPath,...process.argv.slice(1)],process_identity:identity(),
    pi_version:policy.pi_version,pi_sandbox_sha256:py("print(sha256(ROOT/'scripts/pi_sandbox.mjs'))"),
    frozen_manifest_sha256:py("print(sha256(ROOT/'configs/subsets-frozen.json'))"),
    task_ids:frozen.ids,category_quotas:quotas,source_task_file_sha256:py("import hashlib,pathlib; p=pathlib.Path(ROOT/json.load(sys.stdin)['file']);print(hashlib.sha256(p.read_bytes()).hexdigest())",{file:taskFile})};
  if(resumeRun){
    runDir=path.join(root,'.local/runs',resumeRun);
    const saved=JSON.parse(await readFile(path.join(runDir,'config.json'),'utf8'));
    for(const [key,value] of Object.entries(runConfig))if(!['command','process_identity'].includes(key))assert.deepEqual(saved[key],value,`恢复配置变化：${key}`);
    const progress=JSON.parse(await readFile(path.join(runDir,'progress.json'),'utf8'));
    rows=progress.rows??[];
    assert(rows.every((row,index)=>row.task_id===frozen.ids[index]));
  }else runDir=py("v=json.load(sys.stdin);print(start_run(v['experiment'],v))",runConfig);
  py("v=json.load(sys.stdin);write_json(ROOT/'.local/scale-state.json',{'status':'running','scope':v['scope'],'run_id':v['run_id'],'time':now(),'process_identity':v['process_identity']})",
    {run_id:path.basename(runDir),scope:`${experiment} local Pi ${split} evaluation`,process_identity:scaleLockIdentity});
  await save(runDir,'tasks.json',tasks);
  await appendFile(path.join(runDir,'records.jsonl'),'','utf8');
  const authPath=path.join(root,'.local/pi-agent-runtime','auth.json');
  const settings=pi.SettingsManager.inMemory({}, {projectTrusted:true});
  const modelRuntime=await pi.ModelRuntime.create({authPath,modelsPath:null,refreshOnCreate:false});
  modelRuntime.registerProvider(config.api.provider,{name:'MiniLLM local',api:'openai-completions',
    baseUrl:config.api.base_url,apiKey:'local',authHeader:true,models:[{id:config.api.model,name:config.api.display_name??'Qwen3-1.7B E09-R15',
      api:'openai-completions',baseUrl:config.api.base_url,reasoning:false,input:['text'],
      cost:{input:0,output:0,cacheRead:0,cacheWrite:0},contextWindow:config.context_window,maxTokens:config.max_new_tokens,
      compat:{maxTokensField:'max_tokens',requiresToolResultName:true,supportsTemperature:false,supportsDeveloperRole:false}}]});
  const model=modelRuntime.getModel(config.api.provider,config.api.model);
  assert(model);
  api=await launchApi(runDir);
  const referencePrompt=JSON.parse(await readFile(path.join(root,'configs/pi-reference-data.json'),'utf8')).system_prompt;
  const frozenPrompt=JSON.parse(await readFile(path.join(root,config.prompt_config),'utf8'));
  assert.equal(frozenPrompt.selected_prompt,config.selected_prompt);
  for(let index=rows.length;index<tasks.length;index++){
    const task=tasks[index],taskStart=Date.now();current={task_id:task.task_id,category:task.category,index:index+1};
    let sandbox,session,messages=[],timedOut=false,budgetExceeded=false,toolAttempts=0,entry={...current,started:new Date().toISOString()};
    const tracePath=path.join(runDir,'model-api.jsonl');
    const traceOffset=await readFile(tracePath,'utf8').then(text=>text.split('\n').filter(Boolean).length).catch(()=>0);
    try{
      sandbox=await createSandbox({task_timeout_seconds:config.task_timeout_seconds,max_tool_calls:config.max_tool_calls});
      const inspected=await sandbox.inspect(),host=inspected.HostConfig;
      entry.isolation={network:host.NetworkMode,readonly_root:host.ReadonlyRootfs,user:inspected.Config.User,
        binds:host.Binds??[],gpu_requests:host.DeviceRequests??[]};
      assert.equal(host.NetworkMode,'none');assert(inspected.HostConfig.ReadonlyRootfs&&inspected.Config.User==='1000:1000');
      assert(!host.Binds?.length&&!host.DeviceRequests?.length);
      await seed(sandbox,task.files);
      const loader=new pi.DefaultResourceLoader({cwd:root,agentDir:path.join(root,'.local/pi-agent-runtime'),
        settingsManager:settings,noExtensions:true,noSkills:true,noPromptTemplates:true,noThemes:true,noContextFiles:true,
        systemPrompt:referencePrompt+'\n\n'+frozenPrompt.suffix});
      await loader.reload();
      const sessionManager=pi.SessionManager.inMemory('/workspace');
      const tools=sandbox.tools.map(createToolDefinitionFromAgentTool);
      ({session}=await pi.createAgentSession({cwd:'/workspace',agentDir:path.join(root,'.local/pi-agent-runtime'),
        settingsManager:settings,sessionManager,resourceLoader:loader,modelRuntime,model,thinkingLevel:'off',
        noTools:'builtin',customTools:tools}));
      // 第13次请求已超出冻结预算，记录预算耗尽并结束会话。
      session.subscribe(event=>{
        if(event.type==='tool_execution_start'&&++toolAttempts>config.max_tool_calls){
          budgetExceeded=true;void session.abort();
        }
      });
      const taskTimeout=setTimeout(()=>{timedOut=true;void session.abort();},config.task_timeout_seconds*1000-500);
      try{await session.prompt(task.prompt,{expandPromptTemplates:false});await session.waitForIdle();}
      finally{clearTimeout(taskTimeout);}
      messages=transcriptState(session.state.messages);
      const events=observedEvents(messages),answer=finalAnswer(messages);
      const checked=await judge(task,sandbox,answer,events);
      if(budgetExceeded){checked.passed=false;checked.reasons.push('tool_call_budget_exhausted');}
      entry.agent_timed_out=timedOut;
      entry.tool_budget_exhausted=budgetExceeded;
      entry.model_requests=await readFile(tracePath,'utf8').then(text=>text.split('\n').filter(Boolean).slice(traceOffset).map(line=>JSON.parse(line))).catch(()=>[]);
      entry.truncated_generations=entry.model_requests.filter(request=>request.truncated).length;
      entry.agent_events=messages;entry.answer=answer;entry.observed_tool_calls=events;
      entry.judgement={passed:checked.passed,reasons:checked.reasons,execution:checked.execution};
      entry.final_files=checked.files.map(file=>({path:file.path,kind:file.kind??'file',bytes:file.bytes,sha256:file.sha256,target:file.target}));
      entry.final_files_sha256=digest(JSON.stringify(entry.final_files));entry.passed=checked.passed;
    }catch(error){
      entry.agent_timed_out=timedOut;
      entry.tool_budget_exhausted=budgetExceeded;
      entry.model_requests=await readFile(tracePath,'utf8').then(text=>text.split('\n').filter(Boolean).slice(traceOffset).map(line=>JSON.parse(line))).catch(()=>[]);
      entry.truncated_generations=entry.model_requests.filter(request=>request.truncated).length;
      entry.passed=false;entry.judgement={passed:false,reasons:[typeName(error)+':'+String(error.message??error).slice(0,300)]};
      entry.agent_events=messages;
    }finally{
      if(session)session.dispose();
      if(sandbox){
        const removed=await sandbox.dispose().catch(error=>({code:1,stderr:String(error)}));
        entry.container_removed=removed.code===0;entry.container_remove_error=removed.code===0?undefined:removed.stderr;
      }
    }
    const firstRequest=entry.model_requests?.[0];
    assert(firstRequest?.prompt_sha256,`任务 ${task.task_id} 没有本机模型请求轨迹`);
    for(const [previousPrompt,previousHash] of initialPromptHashes){
      if(previousPrompt!==task.prompt)assert.notEqual(firstRequest.prompt_sha256,previousHash,
        '不同冻结任务提示被渲染为相同的模型输入');
    }
    initialPromptHashes.set(task.prompt,firstRequest.prompt_sha256);
    entry.task_prompt_sha256=digest(task.prompt);
    entry.model_input_prompt_sha256=firstRequest.prompt_sha256;
    entry.finished=new Date().toISOString();entry.seconds=Math.round((Date.now()-taskStart)/1000*1000)/1000;
    rows.push(entry);await appendFile(path.join(runDir,'records.jsonl'),JSON.stringify(entry)+'\n','utf8');
    await saveProgress(runDir,{status:'running',completed:rows.length,target:tasks.length,passed:rows.filter(row=>row.passed).length,
      category_results:counts(rows),rows,last_task:entry.task_id,updated:entry.finished});
    console.log(JSON.stringify({task:entry.task_id,completed:rows.length,total:tasks.length,passed:entry.passed,reasons:entry.judgement.reasons}));
    current=null;
    if(Date.now()-lastNotes>=600000){
      updateNotes();
      lastNotes=Date.now();
    }
  }
  assert.equal(rows.length,config.task_count);
}catch(error){failure=error.stack??String(error);console.error(failure);
  if(runDir){await save(runDir,'failure.json',{error:failure,current,updated:new Date().toISOString()});}
}finally{
  try{
    await stopApi(api).catch(()=>{});
    if(runDir){
      const passed=rows.filter(row=>row.passed).length;
      const complete=rows.length===config.task_count&&!failure;
      const tasksPath=path.join(runDir,'tasks.json');
      const result={status:complete?'completed':'failed',exit_code:complete?0:1,operation:config.operation,
        source_run:config.source_run,source_eval_run:config.source_eval_run,split:config.split,
        target_tasks:config.task_count,evaluated_tasks:rows.length,passed_tasks:passed,category_results:counts(rows),
        records_sha256:rows.length?digest(await readFile(path.join(runDir,'records.jsonl'),'utf8')):zero,
        tasks_sha256:existsSync(tasksPath)?py("print(sha256(ROOT/'.local/runs'/json.load(sys.stdin)['run_id']/'tasks.json'))",{run_id:path.basename(runDir)}):zero,
        model_api_trace_sha256:await readFile(path.join(runDir,'model-api.jsonl')).then(data=>digest(data)).catch(()=>zero),
        elapsed_seconds:Math.round((Date.now()-started)/1000),error:failure,
        scope:`${config.task_count}个冻结Pi ${split}任务的实际Agent执行；不代表其他任务集或官方榜单表现`};
      const finished=py("v=json.load(sys.stdin);print(json.dumps(finish_run(ROOT/'.local/runs'/v['id'],v['result']),ensure_ascii=False))",
        {id:path.basename(runDir),result});
      console.log(finished);
      await saveProgress(runDir,{status:result.status,completed:rows.length,target:config.task_count,passed,
        category_results:counts(rows),rows,last_task:rows.at(-1)?.task_id??null,updated:new Date().toISOString(),error:failure});
      updateNotes();
      try{
        py("v=json.load(sys.stdin);write_json(ROOT/'.local/scale-state.json',{'status':v['status'],'scope':v['scope'],'run_id':v['run_id'],'time':now(),'evaluated_tasks':v['evaluated_tasks'],'passed_tasks':v['passed_tasks'],'error':v.get('error')})",
          {status:result.status,scope:`${experiment} local Pi ${split} evaluation`,run_id:path.basename(runDir),evaluated_tasks:result.evaluated_tasks,passed_tasks:result.passed_tasks,error:result.error});
      }catch(error){console.error('无法更新规模状态：'+String(error));process.exitCode=1;}
      if(!complete)process.exitCode=1;
    }else if(failure)process.exitCode=1;
  }finally{
    if(lockOwned){
      const lock=await readFile(lockPath,'utf8').then(JSON.parse).catch(()=>null);
      if(lock?.pid===process.pid)await unlink(lockPath).catch(()=>{});
    }
    if(scaleLockOwned){
      const lock=await readFile(scaleLockPath,'utf8').then(JSON.parse).catch(()=>null);
      if(lock?.pid===scaleLockIdentity.pid&&lock?.created===scaleLockIdentity.created)await unlink(scaleLockPath).catch(()=>{});
    }
  }
}

function typeName(error){return error?.name??'Error';}
