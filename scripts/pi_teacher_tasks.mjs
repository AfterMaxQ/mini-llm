// 教师只读取冻结训练请求；接受轨迹必须同时通过格式、执行和最终状态检查。
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {root} from './pi_sandbox.mjs';
import {digest} from './pi_tasks.mjs';

function argumentsMatch(value,schema){
  if(schema.type==='object')return value!==null&&typeof value==='object'&&!Array.isArray(value)&&
    (schema.required??[]).every(key=>key in value)&&Object.entries(value).every(([key,item])=>
      key in schema.properties&&argumentsMatch(item,schema.properties[key]));
  if(schema.type==='array')return Array.isArray(value)&&value.every(item=>argumentsMatch(item,schema.items));
  return schema.type==='number'?typeof value==='number'&&Number.isFinite(value):typeof value===schema.type;
}

export async function teacherTasks(config){
  const manifest=JSON.parse(await readFile(path.join(root,config.request_manifest),'utf8'));
  assert.equal(digest(await readFile(path.join(root,config.request_manifest))),config.request_manifest_sha256);
  const raw=await readFile(path.join(root,manifest.source_file));
  assert.equal(digest(raw),manifest.source_sha256);
  const pool=raw.toString('utf8').trim().split('\n').map(JSON.parse);
  const byId=new Map(pool.map(task=>[task.task_id,task]));
  const tasks=manifest.ids.map(id=>byId.get(id));
  assert.equal(tasks.length,config.maximum_requests);
  assert(tasks.every(task=>task?.split==='train'));
  assert.equal(new Set(manifest.ids).size,tasks.length);
  assert.equal(manifest.initial_requests,config.initial_requests);
  assert.equal(manifest.extension_batch,config.extension_batch);
  assert.equal(manifest.valid_target,config.valid_target);
  return {tasks,manifest};
}

export function teacherFilter(row,definitions){
  const reasons=[];
  if(!row.passed)reasons.push('task_failed');
  if(row.agent_timed_out)reasons.push('timeout');
  if(row.tool_budget_exhausted)reasons.push('tool_budget_exhausted');
  if(!row.container_removed)reasons.push('container_cleanup_failed');
  const requests=row.model_requests??[];
  if(!requests.length)reasons.push('missing_model_trace');
  if(requests.some(request=>request.truncated||request.cancelled))reasons.push('truncated_or_cancelled');
  if(requests.some(request=>request.raw_response.includes('<tool_call>')&&
    (request.raw_response.match(/<tool_call>/g)?.length??0)!==request.tool_calls.length))reasons.push('malformed_tool_call');
  const messages=row.agent_events??[],calls=messages.flatMap(message=>message.role==='assistant'?
    message.content.filter(block=>block.type==='toolCall'):[]),results=messages.filter(message=>message.role==='toolResult');
  if(calls.length!==results.length||new Set(calls.map(call=>call.id)).size!==calls.length||
    calls.some(call=>results.filter(result=>result.toolCallId===call.id&&result.toolName===call.name).length!==1))
    reasons.push('unmatched_tool_result');
  if(calls.some(call=>!definitions[call.name]||!argumentsMatch(call.arguments,definitions[call.name].parameters)))
    reasons.push('invalid_schema_or_arguments');
  if(messages.some(message=>Array.isArray(message.content)&&message.content.some(block=>!['text','toolCall'].includes(block.type))))
    reasons.push('unsupported_message_block');
  return {accepted:reasons.length===0,reasons};
}
