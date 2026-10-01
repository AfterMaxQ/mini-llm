// 第二批训练模板改变数据结构与修复原因，全部留在训练区。
const json = value => JSON.stringify(value, null, 2) + '\n';
const read = path => ({ tool: 'read', input: { path } });
const write = (path, content) => ({ tool: 'write', input: { path, content } });
const edit = (path, oldText, newText) => ({ tool: 'edit', input: { path, edits: [{ oldText, newText }] } });
const check = { tool: 'bash', input: { command: 'node checks.mjs' } };

export function extendedTasks() {
  const common = { 'package.json': json({ type: 'module', private: true }),
    'README.md': '离线小项目，使用 Node 运行 checks.mjs。\n', 'notes/archive.txt': '旧记录不参与当前运行。\n' };
  const task = (category, family, prompt, files, allowed, reference, expected, negative) => ({
    task_id: 'structure-' + category, split: 'train', category, template_family: family, repository_family: family,
    prompt, files: { ...common, ...files }, allowed_changes: allowed, reference, expected,
    wrong_answer: '未完成', negative });
  const tasks = [];
  tasks.push(task('read_locate', 'train-selected-storage-registry',
    '读取配置与存储注册表，确定当前启用的存储实现、读取函数和用于选择实现的配置键。不要修改文件，用 JSON 返回 file、symbol、config。', {
      'config.json': json({ storage: { backend: 'disk' }, cache: { backend: 'memory' } }),
      'storage/registry.json': json({ disk: { module: 'storage/disk.mjs', read: 'readChunk' }, memory: { module: 'storage/memory.mjs', read: 'lookup' } }),
      'storage/disk.mjs': "export const readChunk = key => ({key, origin:'disk'});\n",
      'storage/memory.mjs': "export const lookup = key => ({key, origin:'memory'});\n"
    }, [], [read('config.json'), read('storage/registry.json'), read('storage/disk.mjs')],
    { answer: { file: 'storage/disk.mjs', symbol: 'readChunk', config: 'storage.backend' } },
    { actions: [], answer: JSON.stringify({ file: 'storage/memory.mjs', symbol: 'lookup', config: 'cache.backend' }) }));

  const defaults = { workers: 4, cache: { slots: 64, ttlSeconds: 90 } };
  const staging = { workers: 2, cache: { ttlSeconds: 15 } };
  const stagingNew = { workers: 2, cache: { ttlSeconds: 15, slots: 24 } };
  tasks.push(task('config_change', 'train-layered-environment-override',
    'staging 继承 defaults.json，再应用 staging.json。只把 staging 的缓存槽位改成24；生产环境仍使用默认64，staging的TTL仍为15。保留默认文件与检查，运行现有检查。', {
      'defaults.json': json(defaults), 'staging.json': json(staging),
      'lib/load.mjs': "import fs from 'node:fs';\nconst read=p=>JSON.parse(fs.readFileSync(p,'utf8'));\nexport function load(profile){const base=read('defaults.json');const extra=profile==='staging'?read('staging.json'):{};return {...base,...extra,cache:{...base.cache,...extra.cache}};}\n",
      'checks.mjs': "import assert from 'node:assert/strict';\nimport {load} from './lib/load.mjs';\nassert.deepEqual(load('staging'),{workers:2,cache:{slots:24,ttlSeconds:15}});\nassert.deepEqual(load('production'),{workers:4,cache:{slots:64,ttlSeconds:90}});\nconsole.log('layered config: passed');\n"
    }, ['staging.json'], [read('defaults.json'), read('staging.json'), write('staging.json', json(stagingNew)), check],
    { json_file: 'staging.json', json_value: stagingNew, check_output: 'layered config: passed' },
    { actions: [write('defaults.json', json({ ...defaults, cache: { ...defaults.cache, slots: 24 } }))], answer: '已改成24。' }));

  const bad = 'export function latest(items){return [...new Map(items.map(x=>[x.id,x])).values()];}\n';
  const good = 'export function latest(items){\n  const result=new Map();\n  for(const item of items){result.delete(item.id);result.set(item.id,item);}\n  return [...result.values()];\n}\n';
  const dedupCheck = "import assert from 'node:assert/strict';\nimport {latest} from './src/latest.mjs';\nconst oracle=items=>items.filter((x,i)=>!items.slice(i+1).some(y=>y.id===x.id));\nlet seed=73;\nconst next=()=>{seed=(seed*1664525+1013904223)>>>0;return seed;};\nconst examples=[[],[{id:'a',v:1},{id:'b',v:2},{id:'a',v:3}]];\nfor(let n=0;n<40;n++)examples.push(Array.from({length:next()%25},(_,i)=>({id:String(next()%7),v:i})));\nfor(const items of examples){const before=JSON.stringify(items);assert.deepEqual(latest(items),oracle(items));assert.equal(JSON.stringify(items),before);}\nconsole.log('latest order: 42 passed');\n";
  tasks.push(task('function_fix', 'train-last-occurrence-order',
    'latest 按id去重，保留最后一次出现的对象，并按这些最后出现的位置排序。例如a、b、a应返回b、a。当前内容对了但顺序不对。请修复函数，不修改输入数组或检查文件，运行现有检查。', {
      'src/latest.mjs': bad, 'checks.mjs': dedupCheck,
      'src/first.mjs': 'export const first = items => items.filter((x,i)=>items.findIndex(y=>y.id===x.id)===i);\n'
    }, ['src/latest.mjs'], [read('src/latest.mjs'), read('checks.mjs'), write('src/latest.mjs', good), check],
    { check_output: 'latest order: 42 passed' },
    { actions: [write('src/latest.mjs', 'export const latest=items=>items.reverse().filter((x,i,a)=>a.findIndex(y=>y.id===x.id)===i);\n')], answer: '已保留最后对象。' }));

  const cfg = { refreshMs: 2500, retries: 3 }, cfgNew = { refreshSeconds: 2.5, retries: 3 };
  const schema = { type: 'object', properties: { refreshMs: { type: 'number' }, retries: { type: 'integer' } }, required: ['refreshMs', 'retries'] };
  const schemaNew = { ...schema, properties: { refreshSeconds: { type: 'number' }, retries: { type: 'integer' } }, required: ['refreshSeconds', 'retries'] };
  const worker = 'export const delay = config => config.refreshMs;\n', workerNew = 'export const delay = config => config.refreshSeconds * 1000;\n';
  tasks.push(task('multi_file', 'train-duration-unit-migration',
    '刷新配置要从refreshMs迁移为refreshSeconds。同步config.json、schema.json和读取延时的模块，保持实际延时2500毫秒与retries=3，允许小数秒。不要只换字段名，保留检查与旧版归档。', {
      'config.json': json(cfg), 'schema.json': json(schema), 'src/worker.mjs': worker,
      'notes/old-config.json': json(cfg),
      'checks.mjs': "import assert from 'node:assert/strict';\nimport fs from 'node:fs';\nimport {delay} from './src/worker.mjs';\nconst config=JSON.parse(fs.readFileSync('config.json','utf8'));const schema=JSON.parse(fs.readFileSync('schema.json','utf8'));\nassert.deepEqual(config,{refreshSeconds:2.5,retries:3});assert.equal(delay(config),2500);assert.equal(delay({refreshSeconds:.125}),125);\nassert.deepEqual(schema.required,['refreshSeconds','retries']);assert.deepEqual(schema.properties,{refreshSeconds:{type:'number'},retries:{type:'integer'}});\nconsole.log('duration migration: passed');\n"
    }, ['config.json', 'schema.json', 'src/worker.mjs'], [read('config.json'), read('schema.json'), read('src/worker.mjs'),
      write('config.json', json(cfgNew)), write('schema.json', json(schemaNew)), edit('src/worker.mjs', worker.trim(), workerNew.trim()), check],
    { check_output: 'duration migration: passed', json_file: 'config.json', json_value: cfgNew },
    { actions: [write('config.json', json({ refreshSeconds: 2500, retries: 3 })), write('schema.json', json(schemaNew)),
      write('src/worker.mjs', 'export const delay = config => config.refreshSeconds;\n')], answer: '字段已迁移。' }));

  const importBad = "import {fee} from './fees.mjs';\nexport const total = amount => amount + fee(amount);\n";
  tasks.push(task('failure_recovery', 'train-moved-module-import',
    '费用模块移动目录后检查失败。先运行checks.mjs观察报错，再根据迁移说明修复入口导入。不能复制旧路径文件、改变计费规则或修改检查。', {
      'src/order.mjs': importBad, 'src/pricing/fees.mjs': 'export const fee = amount => amount >= 100 ? 0 : 5;\n',
      'notes/migration.md': '费用实现现位于 src/pricing/fees.mjs，order 入口需要更新相对导入。\n',
      'checks.mjs': "import assert from 'node:assert/strict';\nimport {total} from './src/order.mjs';\nfor(const [a,b] of [[0,5],[99,104],[100,100],[125,125]])assert.equal(total(a),b);\nconsole.log('moved import: 4 passed');\n"
    }, ['src/order.mjs'], [check, read('notes/migration.md'), read('src/order.mjs'),
      edit('src/order.mjs', "'./fees.mjs'", "'./pricing/fees.mjs'"), check],
    { check_output: 'moved import: 4 passed', must_observe_error: true },
    { actions: [write('src/fees.mjs', 'export const fee = () => 0;\n')], answer: '补回旧模块。' }));

  tasks.push(task('missing_information', 'train-unspecified-schedule-zone',
    '请把提醒设为2026-11-02早上9点。schedule.json里现用UTC，但这次我没有说明是按哪个时区。先看现有配置并确认缺少的信息，不要猜我所在的时区。', {
      'schedule.json': json({ localTime: '08:00', timezone: 'UTC', weekdays: [1, 2, 3, 4, 5] }),
      'lib/reminder.mjs': 'export const label = item => `${item.localTime} ${item.timezone}`;\n'
    }, [], [read('schedule.json')],
    { answer_contains: ['时区', '确认'], answer: '需要先确认这次早上9点按哪个时区，再更新提醒配置。' },
    { actions: [edit('schedule.json', '"timezone": "UTC"', '"timezone": "Asia/Shanghai"')],
      answer: '需要先确认时区。' }));

  const answer = { maxRequests: 315, seconds: 35, requestsPerSecond: 9 };
  tasks.push(task('no_tool', 'train-given-rate-budget-calculation',
    '仅根据已给信息计算，不查看文件也不运行命令：速率上限每秒9次，时间窗口35秒。忽略突发额度，窗口内最多请求多少次？用JSON返回maxRequests、seconds、requestsPerSecond。', {
      'config.json': json({ requestsPerSecond: 40, seconds: 60 })
    }, [], [], { answer, max_tool_calls: 0 },
    { actions: [read('config.json')], answer: JSON.stringify(answer) }));

  const stockAnswer = { file: 'reports/current-stock.json', total: 23, includedSkus: ['P1', 'P3'] };
  tasks.push(task('invalid_path_recovery', 'train-current-stock-active-join',
    '读取data/stock.csv统计可销售商品库存；旧路径无效时按项目说明寻找当前数据。用products.json的active标记筛选库存，忽略停售及未知商品。不要修改文件，用JSON返回file、total、includedSkus（按sku排序）。', {
      'README.md': '库存格式已迁移，当前路径见 reports/manifest.json，商品状态见 products.json。\n',
      'reports/manifest.json': json({ current: 'reports/current-stock.json', archived: 'reports/old-stock.json' }),
      'reports/current-stock.json': json([{ sku: 'P1', quantity: 8 }, { sku: 'P2', quantity: 400 }, { sku: 'P3', quantity: 15 }, { sku: 'X9', quantity: 999 }]),
      'products.json': json([{ sku: 'P1', active: true }, { sku: 'P2', active: false }, { sku: 'P3', active: true }]),
      'reports/old-stock.json': json([{ sku: 'P1', quantity: 777 }])
    }, [], [read('data/stock.csv'), read('README.md'), read('reports/manifest.json'), read('reports/current-stock.json'), read('products.json')],
    { answer: stockAnswer, must_observe_error: true },
    { actions: [read('data/stock.csv')], answer: JSON.stringify({ ...stockAnswer, total: 1422, includedSkus: ['P1', 'P2', 'P3', 'X9'] }) }));
  return tasks;
}
