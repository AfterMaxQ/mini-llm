// 八类训练区原型；参考操作来自任务规则，评判不使用模型自述。
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';

const json = value => JSON.stringify(value, null, 2) + '\n';
const digest = value => createHash('sha256').update(value).digest('hex');
const text = result => result.content.filter(x => x.type === 'text').map(x => x.text).join('\n');
const command = 'node checks.mjs';

export function pilotTasks() {
  const common = { 'package.json': json({ type: 'module', private: true }),
    'README.md': '这是一个可离线运行的小项目，代码位于 lib 或 src。使用 Node 运行 checks.mjs。\n',
    'notes/archive.txt': '旧方案仅供对照，不是当前运行入口。\n' };
  const task = (category, family, prompt, files, allowed, reference, expected, wrongAnswer = '已完成') => ({
    task_id: 'pilot-' + category, split: 'train', category, template_family: family, repository_family: family,
    prompt, files: { ...common, ...files }, allowed_changes: allowed, reference, expected, wrong_answer: wrongAnswer });
  const tasks = [];
  tasks.push(task('read_locate', 'train-router-window-budget',
    '请查看当前请求限流入口，找出决定窗口内是否允许请求的模块、导出函数和配置键。不要修改文件。最终用 JSON 返回 file、symbol、config 三个字段。', {
      'server.mjs': "import {route} from './lib/router.mjs';\nexport const serve = (request, settings) => route(request, settings);\n",
      'lib/router.mjs': "import {allowedInWindow} from './window-budget.mjs';\nexport function route(request, settings) {\n  return allowedInWindow(request.timestamps, settings.rate.windowRequests);\n}\n",
      'lib/window-budget.mjs': 'export function allowedInWindow(timestamps, limit) {\n  return timestamps.length < limit;\n}\n',
      'lib/legacy-quota.mjs': 'export const allowed = () => true; // 旧逻辑未被入口导入\n',
      'settings.json': json({ rate: { windowRequests: 18, windowSeconds: 60 } })
    }, [], [{ tool: 'read', input: { path: 'server.mjs' } }, { tool: 'read', input: { path: 'lib/router.mjs' } },
            { tool: 'read', input: { path: 'lib/window-budget.mjs' } }],
    { answer: { file: 'lib/window-budget.mjs', symbol: 'allowedInWindow', config: 'rate.windowRequests' } }));
  const settings = { profiles: { search: { cache: { maxItems: 53, ttlSeconds: 90 }, pageSize: 25 },
    admin: { cache: { maxItems: 20, ttlSeconds: 30 } } }, telemetry: { enabled: true } };
  const changed = structuredClone(settings); changed.profiles.search.cache.maxItems = 120;
  tasks.push(task('config_change', 'train-profile-cache-json',
    '请把 settings.json 中 search 配置的缓存容量 maxItems 调整为 120。其他配置保持原值，然后确认配置仍能解析。', {
      'settings.json': json(settings), 'lib/config.mjs': "import fs from 'node:fs';\nexport const settings = JSON.parse(fs.readFileSync('settings.json','utf8'));\n"
    }, ['settings.json'], [{ tool: 'read', input: { path: 'settings.json' } },
      { tool: 'edit', input: { path: 'settings.json', edits: [{ oldText: '"maxItems": 53', newText: '"maxItems": 120' }] } },
      { tool: 'bash', input: { command: "node --input-type=module -e \"import {settings} from './lib/config.mjs';console.log(settings.profiles.search.cache.maxItems)\"" } }],
    { json_file: 'settings.json', json_value: changed }));
  const intervalBad = 'export function overlaps(a, b) {\n  return a.start <= b.end && b.start <= a.end;\n}\n';
  const intervalGood = 'export function overlaps(a, b) {\n  return a.start < a.end && b.start < b.end &&\n    a.start < b.end && b.start < a.end;\n}\n';
  const intervalChecks = "import assert from 'node:assert/strict';\nimport {overlaps} from './src/interval.mjs';\nconst cases = [\n  [{start:0,end:3},{start:3,end:6},false],\n  [{start:0,end:4},{start:3,end:6},true],\n  [{start:4,end:4},{start:3,end:6},false],\n  [{start:9,end:2},{start:1,end:8},false],\n  [{start:-5,end:-1},{start:-2,end:0},true],\n  [{start:0,end:8},{start:2,end:4},true]\n];\nfor (const [a,b,expected] of cases) assert.equal(overlaps(a,b),expected);\nconsole.log('interval checks: 6 passed');\n";
  tasks.push(task('function_fix', 'train-half-open-booking',
    '预约冲突判断有边界错误。区间采用左闭右开形式：前一场结束恰好等于后一场开始时不冲突；空区间和反向区间也不冲突。请修复 src/interval.mjs，保留检查文件，运行现有检查。',
    { 'src/interval.mjs': intervalBad, 'checks.mjs': intervalChecks, 'src/format.mjs': 'export const label = item => `${item.start}-${item.end}`;\n' },
    ['src/interval.mjs'], [{ tool: 'read', input: { path: 'src/interval.mjs' } }, { tool: 'read', input: { path: 'checks.mjs' } },
      { tool: 'edit', input: { path: 'src/interval.mjs', edits: [{ oldText: intervalBad.trim(), newText: intervalGood.trim() }] } },
      { tool: 'bash', input: { command } }], { check_output: 'interval checks: 6 passed' }));
  const schema = { type: 'object', properties: { request_count: { type: 'integer' }, uptime: { type: 'number' } }, required: ['request_count', 'uptime'] };
  const schemaNew = structuredClone(schema); schemaNew.properties.served_requests = schemaNew.properties.request_count;
  delete schemaNew.properties.request_count; schemaNew.required[0] = 'served_requests';
  tasks.push(task('multi_file', 'train-metrics-contract-migration',
    '服务指标里的 request_count 要改成 served_requests。请同步修改生产指标的模块、展示指标的模块和 schema.json；uptime 的含义与值保持不变。现有检查不能改，最后运行检查。', {
      'src/metrics.mjs': 'export const metrics = count => ({request_count: count, uptime: 12});\n',
      'src/panel.mjs': 'export const format = stats => `requests=${stats.request_count};uptime=${stats.uptime}`;\n',
      'schema.json': json(schema),
      'checks.mjs': "import assert from 'node:assert/strict';\nimport fs from 'node:fs';\nimport {metrics} from './src/metrics.mjs';\nimport {format} from './src/panel.mjs';\nconst schema=JSON.parse(fs.readFileSync('schema.json','utf8'));\nassert.deepEqual(metrics(9),{served_requests:9,uptime:12});\nassert.equal(format(metrics(9)),'requests=9;uptime=12');\nassert('served_requests' in schema.properties && !('request_count' in schema.properties));\nassert(schema.required.includes('served_requests') && !schema.required.includes('request_count'));\nconsole.log('metrics contract: passed');\n"
    }, ['src/metrics.mjs', 'src/panel.mjs', 'schema.json'], [
      { tool: 'read', input: { path: 'src/metrics.mjs' } }, { tool: 'read', input: { path: 'src/panel.mjs' } },
      { tool: 'read', input: { path: 'schema.json' } },
      { tool: 'edit', input: { path: 'src/metrics.mjs', edits: [{ oldText: 'request_count', newText: 'served_requests' }] } },
      { tool: 'edit', input: { path: 'src/panel.mjs', edits: [{ oldText: 'request_count', newText: 'served_requests' }] } },
      { tool: 'write', input: { path: 'schema.json', content: json(schemaNew) } }, { tool: 'bash', input: { command } }
    ], { check_output: 'metrics contract: passed', json_file: 'schema.json', json_value: schemaNew }));
  const normalBad = "export const normalize = value => value.trim().replace(' ', '-').toLowerCase();\n";
  const normalGood = "export const normalize = value => value.trim().replace(/\\s+/g, '-').toLowerCase();\n";
  tasks.push(task('failure_recovery', 'train-slug-whitespace-regression',
    '最近 CI 的字符串规范化检查失败了。先运行 checks.mjs 看失败，再定位原因修复。空白包括连续空格、制表符和换行，结果用单个连字符连接。请保留检查文件。', {
      'src/slug.mjs': normalBad,
      'checks.mjs': "import assert from 'node:assert/strict';\nimport {normalize} from './src/slug.mjs';\nfor(const [x,y] of [[' A  B ','a-b'],['C\\tD','c-d'],['E\\nF','e-f'],['x','x'],['  ','']])assert.equal(normalize(x),y);\nconsole.log('slug regression: 5 passed');\n",
      'notes/ci.txt': '运行入口为 checks.mjs；上次失败发生在连续空白样例。\n'
    }, ['src/slug.mjs'], [{ tool: 'bash', input: { command } }, { tool: 'read', input: { path: 'src/slug.mjs' } },
      { tool: 'edit', input: { path: 'src/slug.mjs', edits: [{ oldText: normalBad.trim(), newText: normalGood.trim() }] } },
      { tool: 'bash', input: { command } }], { check_output: 'slug regression: 5 passed', must_observe_error: true }));
  tasks.push(task('missing_information', 'train-unset-upload-timeout',
    '请把上传服务的超时时间调大一些。当前配置在 upload.json，单位为秒。我还没有确定具体目标值，你先处理一下。', {
      'upload.json': json({ timeoutSeconds: 30, retries: 2 }),
      'lib/upload.mjs': "import fs from 'node:fs';\nexport const config = JSON.parse(fs.readFileSync('upload.json','utf8'));\n"
    }, [], [{ tool: 'read', input: { path: 'upload.json' } }],
    { answer_contains: ['超时', '多少'], answer: '目标超时时间需要设为多少秒？确认后再修改配置。' }, '已将超时时间改为60秒。'));
  tasks.push(task('no_tool', 'train-explain-request-rate',
    '只根据这句话回答，不需要查看文件：系统每秒允许 8 次请求。用户问这个限制表示什么。请用一句话说明每秒最多能请求多少次，不修改项目。',
    { 'settings.json': json({ unrelatedBatchSize: 32 }) }, [], [],
    { answer_contains: ['每秒', '8'], max_tool_calls: 0, answer: '这个限制表示每秒最多允许发起8次请求。' }, '每秒可以请求32次。'));
  tasks.push(task('invalid_path_recovery', 'train-stale-inventory-path',
    '请读取 data/inventory.csv，统计库存数量总和。如果这个旧路径不存在，请在项目中查找当前库存文件，按文件内容统计。不要新建文件或修改数据，最终用 JSON 返回 file 与 total。', {
      'README.md': '库存导出目录调整过，当前文件记录在 catalog/manifest.json。\n',
      'catalog/manifest.json': json({ currentInventory: 'exports/warehouse/stock.csv' }),
      'exports/warehouse/stock.csv': 'sku,quantity\nA001,7\nB002,12\nC003,4\n',
      'exports/old.csv': 'sku,quantity\nARCHIVED,999\n'
    }, [], [{ tool: 'read', input: { path: 'data/inventory.csv' } }, { tool: 'read', input: { path: 'README.md' } },
      { tool: 'read', input: { path: 'catalog/manifest.json' } }, { tool: 'read', input: { path: 'exports/warehouse/stock.csv' } }],
    { answer: { file: 'exports/warehouse/stock.csv', total: 23 }, must_observe_error: true }));
  assert.equal(tasks.length, 8);
  return tasks;
}

export function docker(args, input = '') {
  return new Promise((resolve, reject) => {
    const child = spawn('docker', args, { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
    let stdout = '', stderr = '';
    child.stdout.on('data', x => stdout += x); child.stderr.on('data', x => stderr += x);
    child.on('error', reject); child.on('close', code => resolve({ code, stdout, stderr }));
    child.stdin.on('error', e => { if (e.code !== 'EPIPE') reject(e); }); child.stdin.end(input);
  });
}

export async function seed(sandbox, files) {
  const worker = "const fs=require('node:fs'),path=require('node:path');let raw='';process.stdin.on('data',x=>raw+=x);process.stdin.on('end',()=>{for(const [name,text] of Object.entries(JSON.parse(raw))){const p=path.resolve('/workspace',name);if(!p.startsWith('/workspace/'))throw Error('invalid fixture path');fs.mkdirSync(path.dirname(p),{recursive:true});fs.writeFileSync(p,text);}});";
  const result = await docker(['exec', '-i', sandbox.containerId, 'node', '-e', worker], JSON.stringify(files));
  assert.equal(result.code, 0, result.stderr);
}

export async function judge(task, sandbox, answer, events) {
  const files = await sandbox.snapshot(), byPath = new Map(files.map(f => [f.path, f]));
  const reasons = [];
  for (const [name, value] of Object.entries(task.files)) {
    if (!task.allowed_changes.includes(name) && byPath.get(name)?.sha256 !== digest(value)) reasons.push('protected_file_changed:' + name);
  }
  for (const file of files) if (!(file.path in task.files)) reasons.push('unexpected_file:' + file.path);
  if (task.expected.json_file) {
    try {
      const value = JSON.parse(Buffer.from(byPath.get(task.expected.json_file).base64, 'base64').toString('utf8'));
      assert.deepEqual(value, task.expected.json_value);
    } catch { reasons.push('json_state_mismatch'); }
  }
  let execution = null;
  if (task.expected.check_output) {
    execution = await docker(['exec', '-i', sandbox.containerId, 'node', 'checks.mjs']);
    if (execution.code !== 0 || execution.stdout.trim() !== task.expected.check_output) reasons.push('execution_check_failed');
  }
  if (task.expected.answer && typeof task.expected.answer !== 'string') {
    try { assert.deepEqual(JSON.parse(answer.replace(/^```(?:json)?\s*|\s*```$/g, '')), task.expected.answer); }
    catch { reasons.push('answer_mismatch'); }
  }
  if (task.expected.answer_contains && !task.expected.answer_contains.every(x => answer.includes(x))) reasons.push('answer_missing_required_information');
  if (task.expected.max_tool_calls !== undefined && events.length > task.expected.max_tool_calls) reasons.push('unnecessary_tool_call');
  if (task.expected.must_observe_error && !events.some(e => e.is_error)) reasons.push('expected_error_not_observed');
  return { passed: reasons.length === 0, reasons, execution, files };
}

export async function executeReference(task, sandbox, call) {
  for (const action of task.reference) await call(action.tool, action.input);
  return typeof task.expected.answer === 'string' ? task.expected.answer : JSON.stringify(task.expected.answer ?? { status: 'completed' });
}

export { text, digest };
