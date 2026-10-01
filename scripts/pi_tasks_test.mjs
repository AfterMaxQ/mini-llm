// 最终测试使用16个新任务族；参数变体只留在test内。
const json=value=>JSON.stringify(value,null,2)+'\n';
const read=path=>({tool:'read',input:{path}});
const write=(path,content)=>({tool:'write',input:{path,content}});
const check={tool:'bash',input:{command:'node checks.mjs'}};
const checks=(body,label)=>"import assert from 'node:assert/strict';\nimport fs from 'node:fs';\n"+body+`\nconsole.log('${label}');\n`;
const gcd=(a,b)=>b?gcd(b,a%b):a;

export function testTasks(){
  const tasks=[];
  const add=(n,category,family,prompt,files,allowed,reference,expected,negative,wrong='未完成')=>tasks.push({
    task_id:`test-${family}-${n}`,split:'test',category,template_family:`test-${family}`,
    repository_family:`test-repo-${family}`,variant:n,prompt,
    files:{'README.md':'离线任务项目。保留检查和无关文件，不安装依赖。\n',...files},
    allowed_changes:allowed,reference,expected,negative,wrong_answer:wrong});

  for(let n=1;n<=6;n++){
    const leaf=`raw/input-${n}.json`,a=`build/core-${n}.json`,b=`build/view-${n}.json`,c=`build/page-${n}.json`;
    const graph={[leaf]:[],[a]:[leaf,'raw/settings.json'],[b]:[a],[c]:[b,'raw/banner.json'],
      'build/unrelated.json':['raw/banner.json'],'raw/settings.json':[],'raw/banner.json':[]};
    const answer={changed:leaf,invalidated:[a,c,b].sort()};
    add(n,'read_locate','transitive-build-invalidation',
      `构建图中每项的数组表示它直接依赖的文件。${leaf}已改变；从watch.json的当前图定位所有直接或间接受影响的构建产物，不包含改变的原始文件，按文件名升序返回JSON字段changed和invalidated。不改文件，归档图不参与。`,
      {'watch.json':json({graph:'build/dependencies.json',roots:[c,'build/unrelated.json']}),
        'build/dependencies.json':json(graph),'archive/dependencies.json':json({[a]:[leaf]})},[],
      [read('watch.json'),read('build/dependencies.json')],{answer},
      {actions:[],answer:JSON.stringify({changed:leaf,invalidated:[a]})});

    const request={method:n%2?'POST':'GET',path:'/private/export',token:n%3?'':'ok',bytes:120+n*15};
    const limit=n%2?100:250;
    const pipeline=[{name:'maintenance',enabled:false,rule:'maintenance'},
      {name:'authentication',enabled:true,rule:'authentication'},
      {name:'body-size',enabled:true,rule:'body-size'},
      {name:'read-only',enabled:true,rule:'read-only'}];
    let blocked_by=null,reason=null;
    if(!request.token){blocked_by='authentication';reason='missing-token';}
    else if(request.bytes>limit){blocked_by='body-size';reason='too-large';}
    else if(request.method!=='GET'){blocked_by='read-only';reason='write-disabled';}
    const ruleText="export const rules={maintenance:()=> 'maintenance',authentication:r=>r.token?null:'missing-token','body-size':(r,c)=>r.bytes>c.maxBytes?'too-large':null,'read-only':r=>r.method==='GET'?null:'write-disabled'};\n";
    add(n,'read_locate','ordered-request-veto',
      '读取request.json，并沿app.mjs和pipeline.json的实际执行顺序判断哪个已启用环节最先拦截请求。忽略未启用环节；没有拦截时两个值都用null。只返回JSON字段blocked_by和reason，不修改文件。',
      {'request.json':json(request),'pipeline.json':json(pipeline),'limits.json':json({maxBytes:limit}),
        'app.mjs':"import {rules} from './rules.mjs';\nexport function run(request,pipeline,limits){for(const stage of pipeline){if(!stage.enabled)continue;const reason=rules[stage.rule](request,limits);if(reason)return {blocked_by:stage.name,reason};}return {blocked_by:null,reason:null};}\n",
        'rules.mjs':ruleText},[],[read('request.json'),read('app.mjs'),read('pipeline.json'),read('rules.mjs'),read('limits.json')],
      {answer:{blocked_by,reason}},{actions:[],answer:JSON.stringify({blocked_by:'maintenance',reason:'maintenance'})});

    const tenant=`org-${n}`,base=[{id:'internal',tenant:'*',prefix:'/internal/',decision:'deny'},
      {id:'fallback',tenant:'*',prefix:'/',decision:'deny'}];
    const exception={id:`download-${n}`,tenant,prefix:`/downloads/${n}/`,decision:'allow'};
    const updated=[base[0],exception,base[1]];
    const engine="export const decision=(rules,tenant,url)=>rules.find(r=>(r.tenant==='*'||r.tenant===tenant)&&url.startsWith(r.prefix))?.decision??'deny';\n";
    add(n,'config_change','first-match-rule-insertion',
      `policy.json采用从上往下第一个匹配规则。为${tenant}的/downloads/${n}/前缀加allow例外，id为download-${n}，放在fallback之前。保留原规则、顺序和其他组织行为，运行现有检查。`,
      {'policy.json':json(base),'engine.mjs':engine,'checks.mjs':checks(`import {decision} from './engine.mjs';\nconst rules=JSON.parse(fs.readFileSync('policy.json','utf8'));assert.deepEqual(rules,${JSON.stringify(updated)});\nassert.equal(decision(rules,'${tenant}','/downloads/${n}/a'),'allow');assert.equal(decision(rules,'other','/downloads/${n}/a'),'deny');assert.equal(decision(rules,'${tenant}','/internal/a'),'deny');`, 'ordered policy: passed')},
      ['policy.json'],[read('policy.json'),read('engine.mjs'),write('policy.json',json(updated)),check],
      {json_file:'policy.json',json_value:updated,check_output:'ordered policy: passed'},
      {actions:[write('policy.json',json([...base,exception]))],answer:'例外规则已添加。'});

    const quota=30+n*4,name=`team-${n}`,newId=`dedicated-${n}`;
    const shared={teams:{[name]:{quota:'shared'},peer:{quota:'shared'}},quotas:{shared:{limit:80,windowSeconds:60}}};
    const detached=structuredClone(shared);detached.teams[name].quota=newId;detached.quotas[newId]={limit:quota,windowSeconds:60};
    add(n,'config_change','shared-quota-detach',
      `quota.json中${name}和peer引用同一配额。把${name}的limit改为${quota}，窗口仍60秒；新配额id用${newId}。peer和原shared配额保持原值，不修改读取器，运行检查。`,
      {'quota.json':json(shared),'load.mjs':"export const load=(config,team)=>config.quotas[config.teams[team].quota];\n",
        'checks.mjs':checks(`import {load} from './load.mjs';\nconst value=JSON.parse(fs.readFileSync('quota.json','utf8'));assert.deepEqual(value,${JSON.stringify(detached)});assert.deepEqual(load(value,'${name}'),{limit:${quota},windowSeconds:60});assert.deepEqual(load(value,'peer'),{limit:80,windowSeconds:60});`, 'detached quota: passed')},
      ['quota.json'],[read('quota.json'),read('load.mjs'),write('quota.json',json(detached)),check],
      {json_file:'quota.json',json_value:detached,check_output:'detached quota: passed'},
      {actions:[write('quota.json',json({...shared,quotas:{shared:{limit:quota,windowSeconds:60}}}))],answer:'已更新共享配额。'});
  }

  for(let n=1;n<=7;n++){
    const topo=`src/order-${n}.mjs`;
    const topoGood="export function order(nodes,edges){\n  const incoming=new Map(nodes.map(x=>[x,0])),next=new Map(nodes.map(x=>[x,[]]));\n  for(const [before,after] of edges){incoming.set(after,incoming.get(after)+1);next.get(before).push(after);}\n  const ready=nodes.filter(x=>incoming.get(x)===0).sort(),result=[];\n  while(ready.length){const current=ready.shift();result.push(current);for(const target of next.get(current)){incoming.set(target,incoming.get(target)-1);if(incoming.get(target)===0){ready.push(target);ready.sort();}}}\n  if(result.length!==nodes.length)throw Error('cycle');return result;\n}\n";
    const topoCases=[[[],[],[]],[['a','b'],[],['a','b']],[['a','b','c'],[['c','a']],['b','c','a']],
      [['a','b','c','d'],[['d','a'],['b','c']],['b','c','d','a']],
      [[`x${n}`,`a${n}`,`m${n}`],[[`x${n}`,`a${n}`]],[`m${n}`,`x${n}`,`a${n}`]]];
    add(n,'function_fix','stable-topological-order',
      '修复order(nodes,edges)：边[before,after]表示before必须先执行；每一步从当前入度为0的节点中选择字典序最小者。节点唯一，边不重复且只引用已有节点。有环抛出包含cycle的错误；不修改输入。运行原检查。',
      {[topo]:"export const order=(nodes,edges)=>nodes.sort();\n",'checks.mjs':checks(`import {order} from './${topo}';\nfor(const [nodes,edges,wanted] of ${JSON.stringify(topoCases)}){const copy=JSON.stringify([nodes,edges]);assert.deepEqual(order(nodes,edges),wanted);assert.equal(JSON.stringify([nodes,edges]),copy);}\nassert.throws(()=>order(['a','b'],[['a','b'],['b','a']]),/cycle/);assert.throws(()=>order(['z'],[['z','z']]),/cycle/);`, 'stable topology: 7 passed')},
      [topo],[read(topo),read('checks.mjs'),write(topo,topoGood),check],{check_output:'stable topology: 7 passed'},
      {actions:[write(topo,"export const order=(nodes,edges)=>[...nodes].sort();\n")],answer:'已避免修改输入。'});

    const merge=`src/merge-${n}.mjs`;
    const mergeGood="const object=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);\nexport function merge(base,patch){\n  const result=structuredClone(base);\n  for(const [key,value] of Object.entries(patch)){\n    if(value===null)delete result[key];\n    else if(object(value))result[key]=merge(object(result[key])?result[key]:{},value);\n    else result[key]=structuredClone(value);\n  }return result;\n}\n";
    const mergeCases=[[{a:{x:1,y:2},keep:true},{a:{x:n}},{a:{x:n,y:2},keep:true}],
      [{a:1,b:2},{a:null},{b:2}],[{a:[1,2]},{a:[n]},{a:[n]}],[{a:3},{a:{x:1}},{a:{x:1}}],
      [{a:{x:2}},{a:{x:null,y:false}},{a:{y:false}}],[{}, {items:[],enabled:false,text:''},{items:[],enabled:false,text:''}]];
    add(n,'function_fix','recursive-merge-delete',
      'merge(base,patch)目前只做浅合并。输入根节点都是普通JSON对象：对象字段递归合并，null删除键，数组和其他值整体替换；结果与两个输入不能共享可变对象。保留未提到的字段，不改变输入，运行检查。',
      {[merge]:"export const merge=(base,patch)=>({...base,...patch});\n",'checks.mjs':checks(`import {merge} from './${merge}';\nfor(const [base,patch,wanted] of ${JSON.stringify(mergeCases)}){const before=JSON.stringify([base,patch]);assert.deepEqual(merge(base,patch),wanted);assert.equal(JSON.stringify([base,patch]),before);}\nconst base={a:{x:1},b:[2]},patch={c:[3]};const value=merge(base,patch);value.a.x=9;value.b.push(9);value.c.push(9);assert.deepEqual(base,{a:{x:1},b:[2]});assert.deepEqual(patch,{c:[3]});`, 'recursive merge: 7 passed')},
      [merge],[read(merge),read('checks.mjs'),write(merge,mergeGood),check],{check_output:'recursive merge: 7 passed'},
      {actions:[write(merge,"export const merge=(base,patch)=>Object.fromEntries(Object.entries({...base,...patch}).filter(([,v])=>v!==null));\n")],answer:'null字段已删除。'});

    const payload={name:`sample-${n}`,count:n},schema={version:1,fields:['payload']};
    const updatedSchema={version:2,fields:['payload','checksum'],algorithm:'sha256',encoding:'hex',input:'JSON.stringify(payload)'};
    const encode="import {createHash} from 'node:crypto';\nexport function encode(payload){return {payload,checksum:createHash('sha256').update(JSON.stringify(payload)).digest('hex')};}\n";
    const decode="import {createHash} from 'node:crypto';\nexport function decode(message){const actual=createHash('sha256').update(JSON.stringify(message.payload)).digest('hex');if(actual!==message.checksum)throw Error('checksum mismatch');return message.payload;}\n";
    add(n,'multi_file','checksum-envelope-contract',
      '升级消息契约：sender.mjs输出payload和checksum，checksum为JSON.stringify(payload)的UTF-8字节SHA256小写hex；receiver.mjs验证校验和，不匹配抛出checksum错误，正确则返回payload。protocol.json改为version=2、fields=[payload,checksum]、algorithm=sha256、encoding=hex、input=JSON.stringify(payload)。归档不改，运行现有检查。',
      {'sender.mjs':"export const encode=payload=>({payload});\n",'receiver.mjs':"export const decode=message=>message.payload;\n",'protocol.json':json(schema),'archive/sender.mjs':"export const encode=payload=>({payload});\n",
        'checks.mjs':checks(`import {createHash} from 'node:crypto';import {encode} from './sender.mjs';import {decode} from './receiver.mjs';\nassert.deepEqual(JSON.parse(fs.readFileSync('protocol.json','utf8')),${JSON.stringify(updatedSchema)});\nfor(const payload of [${JSON.stringify(payload)},{text:'中文',items:[1,2]},{}]){const before=JSON.stringify(payload),message=encode(payload);assert.deepEqual(Object.keys(message).sort(),['checksum','payload']);assert.equal(message.checksum,createHash('sha256').update(before).digest('hex'));assert.deepEqual(decode(message),payload);assert.equal(JSON.stringify(payload),before);assert.throws(()=>decode({...message,checksum:'0'.repeat(64)}),/checksum/);assert.throws(()=>decode({payload:{changed:true},checksum:message.checksum}),/checksum/);}`, 'checksum contract: passed')},
      ['sender.mjs','receiver.mjs','protocol.json'],[read('sender.mjs'),read('receiver.mjs'),read('protocol.json'),write('sender.mjs',encode),write('receiver.mjs',decode),write('protocol.json',json(updatedSchema)),check],
      {check_output:'checksum contract: passed',json_file:'protocol.json',json_value:updatedSchema},
      {actions:[write('sender.mjs',"export const encode=payload=>({payload,checksum:'0'.repeat(64)});\n"),write('protocol.json',json(updatedSchema))],answer:'两个字段已补齐。'});

    const mask=n%7+1,permissions=['read','write','execute'].filter((_,i)=>mask&(1<<i));
    const config={role:`member-${n}`,mask,enabled:n%2===0},updated={role:config.role,permissions,enabled:config.enabled};
    const codec="const names=['read','write','execute'];\nexport const decode=mask=>names.filter((name,index)=>mask&(1<<index));\nexport function encode(permissions){let mask=0;for(const name of permissions){const index=names.indexOf(name);if(index<0)throw Error('unknown permission');mask|=1<<index;}return mask;}\n";
    const access="export const can=(config,action)=>config.enabled&&config.permissions.includes(action);\n";
    add(n,'multi_file','permission-mask-to-set',
      '权限配置从mask迁移为permissions数组，位1/2/4依次表示read/write/execute，数组按此顺序。同步access.json、codec.mjs和access.mjs：codec导出decode(mask)与encode(permissions)，未知权限抛出unknown错误；can仍受enabled控制。保持role和真实权限，不改归档，运行检查。',
      {'access.json':json(config),'codec.mjs':"export const decode=mask=>mask;\nexport const encode=mask=>mask;\n",'access.mjs':"const bits={read:1,write:2,execute:4};\nexport const can=(config,action)=>config.enabled&&Boolean(config.mask&bits[action]);\n",'archive/access.json':json(config),
        'checks.mjs':checks(`import {decode,encode} from './codec.mjs';import {can} from './access.mjs';\nconst config=JSON.parse(fs.readFileSync('access.json','utf8'));assert.deepEqual(config,${JSON.stringify(updated)});\nconst names=['read','write','execute'];for(let mask=0;mask<8;mask++){const list=names.filter((_,i)=>mask&(1<<i));assert.deepEqual(decode(mask),list);assert.equal(encode(list),mask);for(const action of names){assert.equal(can({permissions:list,enabled:true},action),list.includes(action));assert.equal(can({permissions:list,enabled:false},action),false);}}assert.throws(()=>encode(['owner']),/unknown/);assert.equal(encode(['read','read']),1);`, 'permission set: passed')},
      ['access.json','codec.mjs','access.mjs'],[read('access.json'),read('codec.mjs'),read('access.mjs'),write('access.json',json(updated)),write('codec.mjs',codec),write('access.mjs',access),check],
      {json_file:'access.json',json_value:updated,check_output:'permission set: passed'},
      {actions:[write('access.json',json({...updated,permissions:['read']}))],answer:'配置已换成数组。'});

    const validator=`src/code-${n}.mjs`;
    const regexBad="const pattern=/^[A-Z]{2}-[0-9]{3}$/g;\nexport const valid=value=>pattern.test(value);\n";
    const regexGood="const pattern=/^[A-Z]{2}-[0-9]{3}$/;\nexport const valid=value=>pattern.test(value);\n";
    const sample=`AB-${String(n).padStart(3,'0')}`;
    const codeCases=[[sample,true],[sample,true],['aa-123',false],['ZZ-999',true],['ZZ-999',true],
      ['A-123',false],['AB-１２３',false],['AB-123x',false],['AB-123\n',false]];
    add(n,'failure_recovery','global-regex-state',
      '编号校验在连续调用时偶尔失败。先运行checks.mjs，再修复valid，让每次结果只由当前输入决定。编号只能是两个ASCII大写字母、连字符和三个ASCII数字，不能放宽规则或改检查。',
      {[validator]:regexBad,'checks.mjs':checks(`import {valid} from './${validator}';\nfor(const [value,wanted] of ${JSON.stringify(codeCases)})assert.equal(valid(value),wanted);`, 'stateless code: 9 passed')},
      [validator],[check,read(validator),read('checks.mjs'),write(validator,regexGood),check],
      {check_output:'stateless code: 9 passed',must_observe_error:true},
      {actions:[write(validator,"export const valid=value=>value.includes('-');\n")],answer:'连续调用已成功。'});

    const options=`src/options-${n}.mjs`;
    const defaultOptions={retries:2,cache:{enabled:true,tags:['base']}};
    const optionsGood=`const defaults=${JSON.stringify(defaultOptions)};\nexport function options(overrides={}){const result=structuredClone(defaults);if(Object.hasOwn(overrides,'retries'))result.retries=overrides.retries;if(overrides.cache)result.cache={...result.cache,...structuredClone(overrides.cache)};return result;}\n`;
    add(n,'failure_recovery','nested-default-reference-sharing',
      'options返回值修改后会影响下一次调用。先运行检查定位问题，再修复：每次返回独立的嵌套对象和tags数组，cache覆盖保留未指定的默认字段，显式false和retries=0有效；不能修改传入覆盖对象。保留默认值和检查。',
      {[options]:`const defaults=${JSON.stringify(defaultOptions)};\nexport const options=(overrides={})=>Object.assign(defaults,overrides);\n`,
        'checks.mjs':checks(`import {options} from './${options}';\nconst first=options();first.cache.tags.push('changed-${n}');first.cache.enabled=false;assert.deepEqual(options(),${JSON.stringify(defaultOptions)});\nconst input={retries:0,cache:{enabled:false,tags:['custom-${n}']}};const result=options(input);assert.deepEqual(result,input);result.cache.tags.push('changed');assert.deepEqual(input,{retries:0,cache:{enabled:false,tags:['custom-${n}']}});assert.deepEqual(options({cache:{enabled:false}}),{retries:2,cache:{enabled:false,tags:['base']}});assert.deepEqual(options(),${JSON.stringify(defaultOptions)});`, 'independent defaults: passed')},
      [options],[check,read(options),read('checks.mjs'),write(options,optionsGood),check],
      {check_output:'independent defaults: passed',must_observe_error:true},
      {actions:[write(options,`const defaults=${JSON.stringify(defaultOptions)};\nexport const options=(overrides={})=>({...defaults,...overrides});\n`)],answer:'顶层已复制。'});

    const queryKey=`legacy-${n}`,currentKey=`current-${n}`;
    const rows=[{key:currentKey,id:`job-${n}-a`,status:'done',seconds:3+n},
      {key:currentKey,id:`job-${n}-b`,status:'cancelled',seconds:700},
      {key:currentKey,id:`job-${n}-c`,status:'done',seconds:7+n},
      {key:`current-other-${n}`,id:'other',status:'done',seconds:900}];
    const query="import fs from 'node:fs';\nconst rows=JSON.parse(fs.readFileSync('jobs/index.json','utf8'));console.log(JSON.stringify(rows.filter(row=>row.key===process.argv[2])));\n";
    const queryAnswer={key:currentKey,completed_ids:[`job-${n}-a`,`job-${n}-c`],total_seconds:10+2*n};
    add(n,'invalid_path_recovery','empty-query-alias-recovery',
      `先运行node query.mjs ${queryKey}查询；如果没有记录，沿README和别名说明找当前key，再重新查询。只汇总done的seconds，按id升序返回JSON字段key、completed_ids和total_seconds。不要新建或修改文件。`,
      {'README.md':'旧key可能已不使用，映射在jobs/aliases.json；当前查询入口仍为query.mjs。\n',
        'query.mjs':query,'jobs/index.json':json(rows),'jobs/aliases.json':json({[queryKey]:currentKey,unrelated:'archive'})},[],
      [{tool:'bash',input:{command:`node query.mjs ${queryKey}`}},read('README.md'),read('jobs/aliases.json'),{tool:'bash',input:{command:`node query.mjs ${currentKey}`}}],
      {answer:queryAnswer,must_observe_empty_result:true},{actions:[],answer:JSON.stringify(queryAnswer)},JSON.stringify({...queryAnswer,total_seconds:710+2*n}));

    const chunkA=`segments/part-${n}-a.jsonl`,chunkB=`segments/part-${n}-b.jsonl`;
    const eventsA=[{id:`a${n}`,seq:1,state:'pending'},{id:`b${n}`,seq:2,state:'done'},{id:`c${n}`,seq:3,state:'pending'}];
    const eventsB=[{id:`a${n}`,seq:4,state:'done'},{id:`b${n}`,seq:5,state:'cancelled'},{id:`c${n}`,seq:6,state:'done'},{id:`a${n}`,seq:4,state:'done'}];
    const logAnswer={done_ids:[`a${n}`,`c${n}`].sort(),count:2,distinct_events:3};
    add(n,'invalid_path_recovery','fragmented-event-latest-state',
      '先读取logs/events.json；旧路径失效时按README找到分片清单。记录按id归并，只采用该id最大seq的状态，重复记录不能重复计算；返回JSON字段done_ids（升序）、count和distinct_events。不修改或补建旧文件。',
      {'README.md':'事件改为分片JSONL，清单位于segments/list.json。清单顺序不是事件顺序；seq是事件版本号。\n',
        'segments/list.json':json({files:[chunkB,chunkA]}),[chunkA]:eventsA.map(JSON.stringify).join('\n')+'\n',[chunkB]:eventsB.map(JSON.stringify).join('\n')+'\n','archive/events.json':json([{id:'obsolete',seq:99,state:'done'}])},[],
      [read('logs/events.json'),read('README.md'),read('segments/list.json'),read(chunkB),read(chunkA)],
      {answer:logAnswer,must_observe_error:true},{actions:[read('logs/events.json')],answer:JSON.stringify({done_ids:[`a${n}`,`a${n}`,`b${n}`,`c${n}`],count:4,distinct_events:3})});
  }

  for(let n=1;n<=5;n++){
    add(n,'missing_information','unspecified-page-base',
      `把查询设置为第${n+2}页，每页20项。请先看pagination.json和说明，若不知道页码从0还是1开始，先询问约定，不替我猜offset。`,
      {'pagination.json':json({offset:0,limit:20}),'docs/pagination.md':'接口接收offset和limit，调用方的page编号约定尚未写明。\n'},[],
      [read('pagination.json'),read('docs/pagination.md')],
      {answer_contains:['页码','0','1','确认'],answer:'请确认页码从0还是1开始，确认后才能计算offset。'},
      {actions:[write('pagination.json',json({offset:(n+1)*20,limit:20}))],answer:'请确认页码从0还是1开始，我先按1计算了。'});

    const records=[{id:`entry-${n}-active`,active:true},{id:`entry-${n}-old`,active:false}];
    add(n,'missing_information','unspecified-retention-action',
      '清理records.json中inactive记录。请先看当前数据和retention.json；我没有说明清理是删除还是归档。缺少约定先确认，不把归档目录存在当成已授权删除。',
      {'records.json':json(records),'retention.json':json({inactiveAction:null,archivePath:`archive/${n}.json`}),[`archive/${n}.json`]:json([])},[],
      [read('records.json'),read('retention.json')],
      {answer_contains:['删除','归档','确认'],answer:'请确认inactive记录要删除还是归档，确认前保留现有数据。'},
      {actions:[write('records.json',json(records.filter(r=>r.active)))],answer:'请确认删除还是归档，我先删除了inactive。'});

    const countA=2+n,countB=countA*3,meanA=10+n,meanB=30+n;
    const weighted={requests:countA+countB,mean_ms:25+n};
    add(n,'no_tool','unequal-group-weighted-mean',
      `只根据给定数据计算，不读文件、不执行命令。A组${countA}次请求，平均${meanA}毫秒；B组${countB}次请求，平均${meanB}毫秒。合并均值按每组请求数加权，用JSON返回requests和mean_ms。`,
      {'other-metrics.json':json({mean_ms:999})},[],[],{answer:weighted,max_tool_calls:0},
      {actions:[read('other-metrics.json')],answer:JSON.stringify(weighted)},JSON.stringify({requests:countA+countB,mean_ms:(meanA+meanB)/2}));

    const tp=6+n,fp=2+n,fn=3,tn=10+n,num=2*tp,den=2*tp+fp+fn,divisor=gcd(num,den);
    const f1={numerator:num/divisor,denominator:den/divisor};
    add(n,'no_tool','given-f1-reduced-fraction',
      `只用已给值回答，不查看文件或运行命令。TP=${tp}、FP=${fp}、FN=${fn}、TN=${tn}；F1定义为2*TP/(2*TP+FP+FN)，TN不进入公式。用约分后的正整数分数返回JSON字段numerator和denominator。`,
      {'other-model.json':json({numerator:1,denominator:1})},[],[],{answer:f1,max_tool_calls:0},
      {actions:[read('other-model.json')],answer:JSON.stringify(f1)},JSON.stringify({numerator:tp+tn,denominator:tp+fp+fn+tn}));
  }
  return tasks;
}
