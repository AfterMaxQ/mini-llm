// 不启动模型服务，验证 Pi 会话目录与隔离文件工具共用容器路径。
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createSandbox,root,policy} from './pi_sandbox.mjs';

const packageRoot=path.join(root,'.local/pi/node_modules/@earendil-works/pi-coding-agent');
const pi=await import(pathToFileURL(path.join(packageRoot,'dist/index.js')).href);
const {createToolDefinitionFromAgentTool}=await import(pathToFileURL(path.join(packageRoot,'dist/core/tools/tool-definition-wrapper.js')).href);
const prompt=JSON.parse(await readFile(path.join(root,'configs/pi-reference-data.json'),'utf8'));
const frozen=JSON.parse(await readFile(path.join(root,'configs/prompt-frozen.json'),'utf8'));
const sandbox=await createSandbox({task_timeout_seconds:45,max_tool_calls:4});
let session,resultRecord;
try{
  const settings=pi.SettingsManager.inMemory({}, {projectTrusted:true});
  const modelRuntime=await pi.ModelRuntime.create({authPath:path.join(root,'.local/pi-agent-runtime/auth.json'),
    modelsPath:null,refreshOnCreate:false});
  modelRuntime.registerProvider('smoke',{name:'Local smoke',api:'openai-completions',
    baseUrl:'http://127.0.0.1:1/v1',apiKey:'local',authHeader:true,models:[{id:'unused',name:'unused',
      api:'openai-completions',baseUrl:'http://127.0.0.1:1/v1',reasoning:false,input:['text'],
      cost:{input:0,output:0,cacheRead:0,cacheWrite:0},contextWindow:8192,maxTokens:32,
      compat:{maxTokensField:'max_tokens',requiresToolResultName:true,supportsTemperature:false,supportsDeveloperRole:false}}]});
  const model=modelRuntime.getModel('smoke','unused');
  const loader=new pi.DefaultResourceLoader({cwd:root,agentDir:path.join(root,'.local/pi-agent-runtime'),
    settingsManager:settings,noExtensions:true,noSkills:true,noPromptTemplates:true,noThemes:true,noContextFiles:true,
    systemPrompt:prompt.system_prompt+'\n\n'+frozen.suffix});
  await loader.reload();
  const tools=sandbox.tools.map(createToolDefinitionFromAgentTool);
  ({session}=await pi.createAgentSession({cwd:'/workspace',agentDir:path.join(root,'.local/pi-agent-runtime'),
    settingsManager:settings,sessionManager:pi.SessionManager.inMemory('/workspace'),resourceLoader:loader,
    modelRuntime,model,thinkingLevel:'off',noTools:'builtin',customTools:tools}));
  const signal=new AbortController().signal;
  const piWorkspaceRoot=path.win32.resolve('/workspace').replaceAll('\\','/');
  const canonicalPath=piWorkspaceRoot+'/pi-path-smoke.txt';
  const invoke=(name,id,params)=>tools.find(tool=>tool.name===name).execute(
    id,params,signal,()=>{},{cwd:'/workspace'});
  await invoke('write','path-smoke-write',{path:canonicalPath,content:'workspace-ok'});
  const result=await invoke('read','path-smoke-read',{path:canonicalPath});
  const text=result.content.map(item=>item.text??'').join('\n');
  assert.equal(text,'workspace-ok');
  resultRecord={status:'harness_verified',pi_version:policy.pi_version,
    session_cwd:'/workspace',session_manager_cwd:'/workspace',canonical_path:canonicalPath,
    tool_calls:['write','read'],
    read_back_sha256:createHash('sha256').update(text).digest('hex'),model_requests:0,gpu_used:false};
}finally{
  await session?.dispose();
  const removed=await sandbox.dispose();
  assert.equal(removed.code,0,removed.stderr);
  if(resultRecord)resultRecord.container_removed=true;
}
console.log(JSON.stringify(resultRecord));
