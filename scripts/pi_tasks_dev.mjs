// dev任务使用不同的问题结构；同族场景只在dev内变化。
const json=value=>JSON.stringify(value,null,2)+'\n';
const read=path=>({tool:'read',input:{path}});
const write=(path,content)=>({tool:'write',input:{path,content}});
const edit=(path,oldText,newText)=>({tool:'edit',input:{path,edits:[{oldText,newText}]}});
const check={tool:'bash',input:{command:'node checks.mjs'}};

export function devTasks(){
  const tasks=[];
  const add=(n,category,family,prompt,files,allowed,reference,expected,negative,wrong='已完成')=>tasks.push({
    task_id:`dev-${family}-${n}`,split:'dev',category,template_family:`dev-${family}`,
    repository_family:`dev-repo-${family}`,variant:n,prompt,
    files:{'README.md':'这是离线任务项目。保留现有检查及无关文件，不安装外部依赖。\n',...files},
    allowed_changes:allowed,reference,expected,negative,wrong_answer:wrong});
  for(let n=1;n<=5;n++){
    const enhanced=n%2===0,enabled=n%3!==0;
    const chosen=enabled?(enhanced?'fast':'safe'):'offline';
    const mapping={fast:{module:`handlers/fast-${n}.mjs`,symbol:'sendFast'},safe:{module:`handlers/safe-${n}.mjs`,symbol:'sendSafe'},offline:{module:`handlers/offline-${n}.mjs`,symbol:'storePending'}};
    const files={'features.json':json({delivery:{enabled,enhanced},preview:{enabled:true}}),
      'routes.json':json(mapping),
      'app.mjs':"import {resolve} from './lib/resolve.mjs';\nexport const start=flags=>resolve(flags.delivery);\n",
      'lib/resolve.mjs':"import fs from 'node:fs';\nexport function resolve(flags){const route=!flags.enabled?'offline':flags.enhanced?'fast':'safe';return JSON.parse(fs.readFileSync('routes.json','utf8'))[route];}\n",
      'notes/old-route.json':json({module:'handlers/legacy.mjs',symbol:'dispatch'})};
    for(const [name,value] of Object.entries(mapping))files[value.module]=`export const ${value.symbol}=data=>({route:'${name}',data});\n`;
    const answer={file:mapping[chosen].module,symbol:mapping[chosen].symbol,route:chosen};
    add(n,'read_locate','conditional-delivery-route',
      '请沿app.mjs的当前调用链和features.json，确定投递使用哪个模块与具名函数。不要修改项目，用JSON返回file、symbol和route；notes中的旧说明可能不适用。',
      files,[],[read('app.mjs'),read('lib/resolve.mjs'),read('features.json'),read('routes.json'),read(answer.file)],
      {answer},{actions:[],answer:JSON.stringify({file:'handlers/legacy.mjs',symbol:'dispatch',route:'legacy'})});

    const target=`tenant-${n}`,peer=`peer-${n}`,period=12+n*3;
    const subscriptions=[{tenant:target,channel:'sms',periodMinutes:30,enabled:false},
      {tenant:peer,channel:'email',periodMinutes:45,enabled:true},
      {tenant:target,channel:'email',periodMinutes:60,enabled:true}];
    const changed=structuredClone(subscriptions);changed[2].periodMinutes=period;
    add(n,'config_change','subscription-row-selection',
      `只把subscriptions.json中租户${target}的email订阅周期改为${period}分钟。sms、其他租户、enabled字段和数组顺序保持原值，运行现有检查。`,
      {'subscriptions.json':json(subscriptions),'checks.mjs':`import assert from 'node:assert/strict';\nimport fs from 'node:fs';\nassert.deepEqual(JSON.parse(fs.readFileSync('subscriptions.json','utf8')),${JSON.stringify(changed)});\nconsole.log('subscription rows: passed');\n`},
      ['subscriptions.json'],[read('subscriptions.json'),write('subscriptions.json',json(changed)),check],
      {json_file:'subscriptions.json',json_value:changed,check_output:'subscription rows: passed'},
      {actions:[write('subscriptions.json',json(changed.map((row,i)=>i===1?{...row,periodMinutes:period}:row)))],answer:'已修改指定订阅。'});

    const csvPath=`src/csv-${n}.mjs`;
    const good="export function parseLine(line){\n  const result=[];let field='',quoted=false;\n  for(let i=0;i<line.length;i++){const c=line[i];\n    if(c==='\"'){if(quoted&&line[i+1]==='\"'){field+='\"';i++;}else quoted=!quoted;}\n    else if(c===','&&!quoted){result.push(field);field='';}else field+=c;\n  }\n  result.push(field);return result;\n}\n";
    const cases=[['', ['']],['a,b',['a','b']],['a,',['a','']],['"a,b",c',['a,b','c']],
      ['"a""b",x',['a"b','x']],['"",,',['','','']],['"中,文",z',['中,文','z']],
      [`"batch-${n},part",${n}`,[`batch-${n},part`,String(n)]]];
    add(n,'function_fix','quoted-csv-state',
      'parseLine当前直接按逗号拆分，带引号的CSV字段会错。修复单行解析：引号内逗号属于字段、连续两个引号表示一个引号，保留空字段和末尾空列。输入是合法CSV单行，不含字段内换行；运行原检查，不改检查或无关函数。',
      {[csvPath]:"export const parseLine=line=>line.split(',');\n",'src/tsv.mjs':"export const parseTSV=line=>line.split('\\t');\n",
        'checks.mjs':`import assert from 'node:assert/strict';\nimport {parseLine} from './${csvPath}';\nfor(const [line,expected] of ${JSON.stringify(cases)})assert.deepEqual(parseLine(line),expected);\nconsole.log('quoted csv: 8 passed');\n`},
      [csvPath],[read(csvPath),read('checks.mjs'),write(csvPath,good),check],
      {check_output:'quoted csv: 8 passed'},
      {actions:[write(csvPath,"export const parseLine=line=>line.replaceAll('\"','').split(',');\n")],answer:'已移除引号并解析。'});

    const source=`core/quantity-${n}.mjs`,consumer=`views/quantity-${n}.mjs`;
    const implementation='value=>Number.parseInt(value,10)';
    add(n,'multi_file','named-export-chain',
      '把当前公共API的normalizeQuantity改名为parseQuantity，同时对齐core实现、index.mjs转发和views里的使用方。保持解析行为，归档版本不改，运行现有检查。',
      {[source]:`export const normalizeQuantity=${implementation};\n`,'index.mjs':`export {normalizeQuantity} from './${source}';\n`,
        [consumer]:"import {normalizeQuantity} from '../index.mjs';\nexport const display=value=>`items=${normalizeQuantity(value)}`;\n",
        'archive/quantity.mjs':`export const normalizeQuantity=${implementation};\n`,
        'checks.mjs':`import assert from 'node:assert/strict';\nimport * as api from './index.mjs';\nimport * as core from './${source}';\nimport {display} from './${consumer}';\nassert(!('normalizeQuantity' in api));assert(!('normalizeQuantity' in core));\nassert.equal(api.parseQuantity('${n}'),${n});assert.equal(display('${n}'), 'items=${n}');\nassert.equal(api.parseQuantity('7suffix'),7);\nconsole.log('named export chain: passed');\n`},
      [source,'index.mjs',consumer],[read('index.mjs'),read(source),read(consumer),
        edit(source,'normalizeQuantity','parseQuantity'),edit('index.mjs','normalizeQuantity','parseQuantity'),
        {tool:'edit',input:{path:consumer,edits:[{oldText:'import {normalizeQuantity}',newText:'import {parseQuantity}'},
          {oldText:'${normalizeQuantity(value)}',newText:'${parseQuantity(value)}'}]}},check],
      {check_output:'named export chain: passed'},
      {actions:[edit(source,'normalizeQuantity','parseQuantity')],answer:'实现已改名。'});

    const route=`src/response-${n}.mjs`;
    const bad="import {format} from './format.mjs';\nexport async function respond(value){return `body=${format(value)}`;}\n";
    const fixed="import {format} from './format.mjs';\nexport async function respond(value){return `body=${await format(value)}`;}\n";
    add(n,'failure_recovery','await-format-result',
      '现有异步响应检查失败。先运行checks.mjs，再根据真实错误找到修复位置。返回正文应包含格式化后的值，格式化抛出的异常应继续传播。不要改检查或同步归档实现。',
      {[route]:bad,'src/format.mjs':"export async function format(value){if(value<0)throw Error('negative value');return `value:${value}`;}\n",
        'archive/sync-format.mjs':"export const format=value=>String(value);\n",
        'checks.mjs':`import assert from 'node:assert/strict';\nimport {respond} from './${route}';\nassert.equal(await respond(${n}),'body=value:${n}');\nassert.equal(await respond(0),'body=value:0');\nawait assert.rejects(()=>respond(-1),/negative value/);\nconsole.log('async response: 3 passed');\n`},
      [route],[check,read(route),read('src/format.mjs'),read('checks.mjs'),write(route,fixed),check],
      {check_output:'async response: 3 passed',must_observe_error:true},
      {actions:[write(route,"export async function respond(value){return `body=value:${value}`;}\n")],answer:'正常正文已经正确。'});

    const tenants=[{tenant:`east-${n}`,notifications:true},{tenant:`west-${n}`,notifications:true}];
    add(n,'missing_information','ambiguous-target-tenant',
      '请关掉该租户的通知。项目中可能有不止一个租户，请读取当前租户列表确认是否足以确定目标；缺信息先询问，不自行选择。',
      {'tenants.json':json(tenants),'settings.json':json({defaultTenant:null,environment:'production'})},[],
      [read('settings.json'),read('tenants.json')],
      {answer_contains:['租户','哪'],answer:`请确认要关闭哪个租户的通知：east-${n}还是west-${n}？`},
      {actions:[write('tenants.json',json(tenants.map((row,i)=>i===0?{...row,notifications:false}:row)))],answer:`请确认哪个租户？我先关闭east-${n}。`},'已关闭east租户通知。');

    const times=Array.from({length:20},(_,i)=>5+n*11+i*3);const shuffled=times.filter((_,i)=>i%2).concat(times.filter((_,i)=>!(i%2)).reverse());
    const answer95={rank:19,p95_ms:times[18]};
    add(n,'no_tool','nearest-rank-percentile',
      `只根据已给数据回答，不查看文件或执行命令。20次请求延迟（毫秒）为${JSON.stringify(shuffled)}。p95采用最近秩法：升序排列，取ceil(0.95*N)位置，位置从1计。返回JSON，字段rank和p95_ms。`,
      {'notes/other-latency.json':json({p95_ms:999})},[],[],{answer:answer95,max_tool_calls:0},
      {actions:[read('notes/other-latency.json')],answer:JSON.stringify(answer95)},JSON.stringify({rank:20,p95_ms:times[19]}));

    const current=`snapshots/report-${n}.json`;
    const observations=[{kind:'completed',milliseconds:10+n},{kind:'cancelled',milliseconds:800},{kind:'completed',milliseconds:20+n*2}];
    const total=30+n*3;
    add(n,'invalid_path_recovery','latest-successful-report',
      '先尝试读取reports/latest.json，若旧路径失效，沿项目说明找到不晚于2026-09-25的最近成功快照。只汇总completed记录的milliseconds，用JSON返回file、count、total_ms；不新增旧路径文件。',
      {'README.md':'旧reports/latest.json可能已移除。快照清单位于snapshots/manifest.json；按成功状态和日期选择，不按列表最后一项猜测。\n',
        'snapshots/manifest.json':json([{file:current,date:'2026-09-24',status:'success'},
          {file:'snapshots/future.json',date:'2026-09-26',status:'success'},
          {file:'snapshots/failed.json',date:'2026-09-25',status:'failed'},
          {file:'snapshots/old.json',date:'2026-09-23',status:'success'}]),
        [current]:json(observations),'snapshots/future.json':json([{kind:'completed',milliseconds:5000}]),
        'snapshots/failed.json':json([{kind:'completed',milliseconds:1000}]),'snapshots/old.json':json([{kind:'completed',milliseconds:1}])},[],
      [read('reports/latest.json'),read('README.md'),read('snapshots/manifest.json'),read(current)],
      {answer:{file:current,count:2,total_ms:total},must_observe_error:true},
      {actions:[write('reports/latest.json',json(observations))],answer:JSON.stringify({file:'reports/latest.json',count:2,total_ms:total})});
  }
  return tasks;
}
