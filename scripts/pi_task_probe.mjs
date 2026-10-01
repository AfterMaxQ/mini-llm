// 在本机真实容器检查任务判据，错误候选与参考操作使用不同容器。
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { createSandbox, root, policy } from './pi_sandbox.mjs';
import { pilotTasks, seed, judge, executeReference, digest } from './pi_tasks.mjs';
import { extendedTasks } from './pi_tasks_extended.mjs';

const python = path.join(root, '.local/venv-train/Scripts/python.exe');
function recordPython(code, value) {
  return execFileSync(python, ['-c', "import json,sys;sys.path.insert(0,'scripts');from lab import ROOT,start_run,finish_run;" + code],
    { cwd: root, input: JSON.stringify(value), encoding: 'utf8', windowsHide: true, env: { ...process.env, PYTHONUTF8: '1' } }).trim();
}
const catalog = JSON.parse(await readFile(path.join(root, 'configs/pi-task-catalog.json'), 'utf8'));
assert.equal(catalog.pilot_split, 'train');
const suite = process.argv.includes('--extended') ? 'structural_extension' : 'pilot';
const tasks = suite === 'pilot' ? pilotTasks() : extendedTasks();
assert.equal(tasks.length, 8);
assert(tasks.every(t => t.split === 'train'));
const scope = suite === 'pilot' ? '八个训练区任务的规则参考与判据核验；无模型调用，不满足正式E14规模'
  : '八个新增训练模板族的真实规则执行与判据核验；无模型调用，未满足正式轨迹规模';
const directory = recordPython("print(start_run('E14',json.load(sys.stdin)))", {
  operation: 'reference_task_probe', suite, command: process.argv, catalog, sandbox: policy, model: null, scope });
const runId = path.basename(directory);
const save = (name, value) => writeFile(path.join(directory, name), JSON.stringify(value, null, 2), 'utf8');
const config = JSON.parse(await readFile(path.join(directory, 'config.json'), 'utf8'));
config.process_id = process.pid; await save('config.json', config);
const sourceFiles = ['scripts/pi_tasks.mjs', 'scripts/pi_tasks_extended.mjs', 'scripts/pi_task_probe.mjs', 'scripts/pi_sandbox.mjs', 'configs/pi-task-catalog.json', 'configs/pi-tools.json', 'configs/pi-sandbox.json'];
const sources = [];
for (const file of sourceFiles) sources.push({ file, sha256: digest(await readFile(path.join(root, file))) });
await save('sources.json', sources); await save('tasks.json', tasks);
const records = [], live = new Set();
const started = performance.now();
async function sandbox() {
  const value = await createSandbox(); live.add(value); return value;
}
async function dispose(value) {
  const result = await value.dispose(); assert.equal(result.code, 0, result.stderr); live.delete(value);
  return { container: value.name, container_id: value.containerId, removed: true };
}
async function call(value, events, tool, input) {
  const begin = performance.now(), id = `${runId}-${value.name}-${events.length + 1}`;
  const event = { call_id: id, tool, input };
  try { event.result = await value.tools.find(x => x.name === tool).execute(id, input); event.is_error = Boolean(event.result.isError); }
  catch (error) { event.is_error = true; event.error = error.message; }
  event.seconds = (performance.now() - begin) / 1000; events.push(event); return event;
}
let failure;
try {
  for (const task of tasks) {
    const entry = { task_id: task.task_id, category: task.category, template_family: task.template_family,
      repository_family: task.repository_family, split: task.split, started: new Date().toISOString() };
    records.push(entry); await save('records.json', records);
    const control = await sandbox(); await seed(control, task.files);
    entry.control_container = await control.inspect();
    assert.equal(entry.control_container.HostConfig.NetworkMode, 'none');
    assert(!entry.control_container.HostConfig.DeviceRequests?.length);
    entry.initial = await judge(task, control, task.wrong_answer, []);
    assert.equal(entry.initial.passed, false, task.task_id + '：初始错误候选被误判成功');
    const controlEvents = [];
    let wrongAnswer = task.wrong_answer;
    if (task.negative) {
      for (const action of task.negative.actions) await call(control, controlEvents, action.tool, action.input);
      wrongAnswer = task.negative.answer;
    } else if (task.expected.check_output) {
      // 只改检查文件让它输出成功，不应骗过最终判据。
      await call(control, controlEvents, 'write', { path: 'checks.mjs', content: `console.log(${JSON.stringify(task.expected.check_output)});\n` });
    } else if (task.category === 'config_change') {
      const wrong = structuredClone(task.expected.json_value); wrong.profiles.admin.cache.maxItems = 21;
      await call(control, controlEvents, 'write', { path: 'settings.json', content: JSON.stringify(wrong, null, 2) + '\n' });
    } else if (task.category === 'missing_information') {
      await call(control, controlEvents, 'edit', { path: 'upload.json', edits: [{ oldText: '"timeoutSeconds": 30', newText: '"timeoutSeconds": 60' }] });
      wrongAnswer = task.expected.answer;
    } else if (task.category === 'no_tool') {
      await call(control, controlEvents, 'read', { path: 'settings.json' }); wrongAnswer = task.expected.answer;
    } else if (task.category === 'invalid_path_recovery') {
      await call(control, controlEvents, 'write', { path: 'data/inventory.csv', content: 'sku,quantity\nFAKE,23\n' });
      wrongAnswer = JSON.stringify(task.expected.answer);
    } else {
      wrongAnswer = JSON.stringify({ file: 'lib/legacy-quota.mjs', symbol: 'allowed', config: 'rate.windowRequests' });
    }
    entry.control_events = controlEvents;
    entry.negative = await judge(task, control, wrongAnswer, controlEvents);
    assert.equal(entry.negative.passed, false, task.task_id + '：明确错误候选被误判成功');
    entry.control_removed = await dispose(control); await save('records.json', records);

    const reference = await sandbox(); await seed(reference, task.files);
    const events = [];
    const answer = await executeReference(task, reference, (tool, input) => call(reference, events, tool, input));
    entry.reference_events = events; entry.reference_answer = answer;
    entry.reference = await judge(task, reference, answer, events);
    await save('records.json', records);
    assert.equal(entry.reference.passed, true, task.task_id + '：规则参考未通过：' + entry.reference.reasons.join(','));
    assert(events.length <= policy.max_tool_calls);
    entry.reference_removed = await dispose(reference); entry.finished = new Date().toISOString();
    await save('records.json', records);
    console.log(JSON.stringify({ task: task.task_id, reference_calls: events.length, error_returns: events.filter(x => x.is_error).length,
      initial_rejected: !entry.initial.passed, negative_rejected: !entry.negative.passed, reference_passed: entry.reference.passed }));
  }
} catch (error) { failure = error.stack; await save('failure.json', { error: failure }); }
finally {
  for (const value of live) { try { await dispose(value); } catch (error) { await save('cleanup-error.json', { error: error.stack }); } }
  await save('records.json', records);
  const result = { status: failure ? 'failed' : suite === 'pilot' ? 'reference_pilot_verified' : 'reference_extension_verified', exit_code: failure ? 1 : 0,
    operation: 'reference_task_probe', suite, error: failure, duration_seconds: (performance.now() - started) / 1000,
    task_count: records.length, reference_passed: records.filter(x => x.reference?.passed).length,
    initial_rejected: records.filter(x => x.initial && !x.initial.passed).length,
    negative_rejected: records.filter(x => x.negative && !x.negative.passed).length,
    reference_tool_calls: records.reduce((n, x) => n + (x.reference_events?.length ?? 0), 0),
    reference_error_returns: records.reduce((n, x) => n + (x.reference_events?.filter(e => e.is_error).length ?? 0), 0),
    diagnostic_tool_calls: records.reduce((n, x) => n + (x.control_events?.length ?? 0), 0),
    records_sha256: digest(await readFile(path.join(directory, 'records.json'))),
    tasks_sha256: digest(await readFile(path.join(directory, 'tasks.json'))), sources,
    model_calls: 0, dev_tasks: 0, test_tasks: 0, valid_reference_trajectories: records.filter(x => x.reference?.passed).length,
    per_task: records.map(x => ({ task_id: x.task_id, category: x.category, split: x.split,
      reference_passed: Boolean(x.reference?.passed), initial_rejected: Boolean(x.initial && !x.initial.passed),
      negative_rejected: Boolean(x.negative && !x.negative.passed), negative_reasons: x.negative?.reasons,
      reference_calls: x.reference_events?.length ?? 0, error_returns: x.reference_events?.filter(e => e.is_error).length ?? 0 })),
    scope };
  const response = recordPython("v=json.load(sys.stdin);print(json.dumps(finish_run(ROOT/'.local/runs'/v['run_id'],v['result']),ensure_ascii=False))", { run_id: runId, result });
  console.log(response); if (failure) process.exitCode = 1;
}
