// 实际执行工具与容器边界核验；不调用模型，不把结果算成 Agent 成功率。
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { createSandbox, root, packageInfo, policy } from './pi_sandbox.mjs';

const stamp = new Date().toISOString().replaceAll(':', '-');
const directory = path.join(root, '.local/preparation/pi-sandbox', stamp);
await mkdir(directory, { recursive: true });
const events = [], snapshots = [], containers = [];
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const save = (name, value) => writeFile(path.join(directory, name), JSON.stringify(value, null, 2), 'utf8');
for (const [source, target] of [['scripts/pi_sandbox.mjs', 'adapter-source.mjs'], ['scripts/pi_tools_probe.mjs', 'probe-source.mjs'], ['configs/pi-sandbox.json', 'config.json']]) {
  await writeFile(path.join(directory, target), await readFile(path.join(root, source)));
}
await save('execution.json', { started: new Date().toISOString(), command: process.argv, node_version: process.version,
  commit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8', windowsHide: true }).trim(),
  scope: '实际Pi工具、容器与故障边界核验，不进行模型推理' });
async function call(sandbox, name, input) {
  const id = 'probe-' + (events.length + 1), start = performance.now();
  const event = { container: sandbox.name, call_id: id, tool: name, input };
  try {
    event.result = await sandbox.tools.find(t => t.name === name).execute(id, input);
    event.is_error = Boolean(event.result.isError);
  } catch (e) { event.is_error = true; event.error = e.message; }
  event.seconds = (performance.now() - start) / 1000;
  events.push(event); await save('events.json', events);
  return event;
}
let status = 'failed', error;
try {
  const sandbox = await createSandbox(); containers.push(sandbox);
  const inspect = await sandbox.inspect(); await save('container-inspect.json', inspect);
  assert.equal(inspect.HostConfig.NetworkMode, 'none'); assert.equal(inspect.HostConfig.ReadonlyRootfs, true);
  assert.equal(inspect.Config.User, '1000:1000'); assert.deepEqual(inspect.HostConfig.CapDrop, ['ALL']);
  assert.equal(inspect.HostConfig.Memory, 1024 ** 3); assert.equal(inspect.HostConfig.PidsLimit, 128);
  assert(!inspect.Mounts.some(m => m.Type === 'bind' || m.Type === 'volume'));
  const schemas = sandbox.tools.map(t => ({ name: t.name, description: t.description, parameters: t.parameters }));
  const sourceFiles = ['package.json', 'dist/core/tools/read.js', 'dist/core/tools/edit.js', 'dist/core/tools/write.js', 'dist/core/tools/bash.js', 'dist/utils/mime.js'];
  const source = [];
  for (const file of sourceFiles) source.push({ file, sha256: digest(await readFile(path.join(root, '.local/pi/node_modules/@earendil-works/pi-coding-agent', file))) });
  const schemaRecord = { package: packageInfo.name, version: packageInfo.version, license: packageInfo.license, source, tools: schemas };
  await save('tool-schemas.json', schemaRecord);
  await writeFile(path.join(root, 'configs/pi-tools.json'), JSON.stringify(schemaRecord, null, 2), 'utf8');
  const original = 'export const sum = (a, b) => a - b;\n';
  assert(!(await call(sandbox, 'write', { path: 'src/sum.mjs', content: original })).is_error);
  const read = await call(sandbox, 'read', { path: '/workspace/src/sum.mjs' });
  assert.equal(read.result.content[0].text, original);
  assert(!(await call(sandbox, 'edit', { path: 'src/sum.mjs', edits: [{ oldText: 'a - b', newText: 'a + b' }] })).is_error);
  const executed = await call(sandbox, 'bash', { command: 'node --input-type=module -e "import {sum} from \'./src/sum.mjs\'; if(sum(2,3)!==5)process.exit(1); console.log(sum(2,3));console.log(process.version)"' });
  assert.equal(executed.result.structuredContent.exit_code, 0); assert.equal(executed.result.structuredContent.output.trim(), '5\nv24.14.0');
  assert((await call(sandbox, 'read', { path: 'missing.txt' })).is_error);
  assert((await call(sandbox, 'edit', { path: 'src/sum.mjs', edits: [{ oldText: 'nonexistent marker', newText: 'x' }] })).is_error);
  assert((await call(sandbox, 'read', { path: '../outside.txt' })).is_error);
  assert(!(await call(sandbox, 'bash', { command: 'ln -s /tmp escape' })).is_error);
  assert((await call(sandbox, 'write', { path: 'escape/probe.txt', content: 'outside' })).is_error);
  for (let i = 0; i < 3; i++) assert(!(await call(sandbox, 'read', { path: 'src/sum.mjs' })).is_error);
  const exceeded = await call(sandbox, 'read', { path: 'src/sum.mjs' });
  assert.match(exceeded.error, /调用预算/);
  const files = await sandbox.snapshot(); snapshots.push({ container: sandbox.name, files });
  assert.equal(Buffer.from(files.find(f => f.path === 'src/sum.mjs').base64, 'base64').toString('utf8'), original.replace('a - b', 'a + b'));
  const extra = await createSandbox(); containers.push(extra);
  const network = await call(extra, 'bash', { command: 'node -e "const net=require(\'net\');const s=net.connect({host:\'1.1.1.1\',port:443});s.setTimeout(2000);s.on(\'connect\',()=>process.exit(1));s.on(\'error\',e=>{console.log(e.code);process.exit(0)});s.on(\'timeout\',()=>process.exit(2));"', timeout: 3 });
  assert.equal(network.result.structuredContent.exit_code, 0); assert.match(network.result.structuredContent.output, /ENETUNREACH/);
  assert((await call(extra, 'bash', { command: 'node -e "require(\'fs\').writeFileSync(\'/probe-outside-workspace\',\'x\')"' })).is_error);
  const nonzero = await call(extra, 'bash', { command: 'exit 7' });
  assert(nonzero.is_error); assert.equal(nonzero.result.structuredContent.exit_code, 7);
  const timeout = await call(extra, 'bash', { command: 'sleep 3', timeout: 1 });
  assert(timeout.is_error); assert.match(timeout.error, /timed out after 1 seconds/);
  assert(timeout.seconds >= 1 && timeout.seconds < 4);
  // 固定的一像素 PNG 仅检查图片读取接口，不作为实验图或截图。
  const png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNQcEgAAAFEAMELjZZCAAAAAElFTkSuQmCC';
  assert(!(await call(extra, 'bash', { command: `node -e "require('fs').writeFileSync('fixture.png',Buffer.from('${png}','base64'))"` })).is_error);
  const image = await call(extra, 'read', { path: 'fixture.png' });
  assert(image.result.content.some(item => item.type === 'image' && item.mimeType === 'image/png'));
  snapshots.push({ container: extra.name, files: await extra.snapshot() });
  // 总时限分支用独立的 2 秒诊断条件核验；正式任务配置仍是 300 秒。
  const deadline = await createSandbox({ task_timeout_seconds: 2 }); containers.push(deadline);
  const expired = await call(deadline, 'bash', { command: 'sleep 5' });
  assert(expired.is_error); assert.match(expired.error, /任务超过总时限/);
  assert(expired.seconds < 5);
  status = 'tool_chain_verified';
} catch (e) { error = e.stack; await save('failure.json', { error }); }
finally {
  await save('snapshots.json', snapshots);
  const removed = [];
  for (const sandbox of containers) removed.push({ container: sandbox.name, config: sandbox.config, command: sandbox.command, ...(await sandbox.dispose()) });
  await save('containers.json', removed);
  const result = { status, time: new Date().toISOString(), event_count: events.length, error,
    schema_sha256: digest(await readFile(path.join(directory, 'tool-schemas.json')).catch(() => Buffer.alloc(0))),
    events_sha256: digest(await readFile(path.join(directory, 'events.json')).catch(() => Buffer.alloc(0))),
    adapter_sha256: digest(await readFile(path.join(root, 'scripts/pi_sandbox.mjs'))),
    probe_sha256: digest(await readFile(new URL(import.meta.url))), config_sha256: digest(await readFile(path.join(root, 'configs/pi-sandbox.json'))),
    task_policy: policy, scope: '真实Pi工具与本机容器的执行核验；无模型调用，不是E13 Agent任务成功率', directory: path.relative(root, directory).replaceAll('\\', '/') };
  await save('result.json', result); console.log(JSON.stringify(result, null, 2));
  if (status !== 'tool_chain_verified') process.exitCode = 1;
}
