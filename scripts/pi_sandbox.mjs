// 使用 Pi 的原始工具，只把文件和 bash operations 换为本机容器。
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
export const policy = JSON.parse(await readFile(path.join(root, 'configs/pi-sandbox.json'), 'utf8'));
const packageRoot = path.join(root, '.local/pi/node_modules/@earendil-works/pi-coding-agent');
export const packageInfo = JSON.parse(await readFile(path.join(packageRoot, 'package.json'), 'utf8'));
assert.equal(packageInfo.version, policy.pi_version);
const pi = await import(pathToFileURL(path.join(packageRoot, 'dist/index.js')).href);
const mime = await import(pathToFileURL(path.join(packageRoot, 'dist/utils/mime.js')).href);

function docker(args, input = '', onData) {
  return new Promise((resolve, reject) => {
    const child = spawn('docker', args, { cwd: root, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
    const output = [], errors = [];
    child.stdout.on('data', data => { if (!onData) output.push(data); onData?.(data); });
    child.stderr.on('data', data => { if (!onData) errors.push(data); onData?.(data); });
    child.on('error', reject);
    child.on('close', code => resolve({ code, stdout: Buffer.concat(output).toString('utf8'), stderr: Buffer.concat(errors).toString('utf8') }));
    child.stdin.on('error', error => { if (error.code !== 'EPIPE') reject(error); });
    child.stdin.end(input);
  });
}

// 路径检查在容器内完成，包含现有文件及最近存在的父目录。
const fileWorker = String.raw`
const fs = require('node:fs/promises'), path = require('node:path'), crypto = require('node:crypto');
let raw = ''; process.stdin.setEncoding('utf8'); process.stdin.on('data', x => raw += x);
process.stdin.on('end', async () => {
  try {
    const request = JSON.parse(raw), target = path.resolve('/workspace', request.path);
    const inside = p => p === '/workspace' || p.startsWith('/workspace/');
    if (!inside(target)) throw new Error('文件路径超出任务工作区');
    let parent = target;
    for (;;) {
      try { if (!inside(await fs.realpath(parent))) throw new Error('符号链接超出任务工作区'); break; }
      catch (e) { if (e.code !== 'ENOENT') throw e; parent = path.dirname(parent); }
    }
    let value = null;
    if (request.op === 'read') value = (await fs.readFile(target)).toString('base64');
    else if (request.op === 'write') await fs.writeFile(target, request.content, 'utf8');
    else if (request.op === 'mkdir') await fs.mkdir(target, { recursive: true });
    else if (request.op === 'access') await fs.access(target, request.mode);
    else if (request.op === 'snapshot') {
      value = [];
      async function visit(dir) {
        for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
          const name = path.join(dir, entry.name);
          if (entry.isDirectory()) await visit(name);
          else if (entry.isSymbolicLink()) value.push({ path: path.relative('/workspace', name), kind: 'symlink', target: await fs.readlink(name) });
          else { const data = await fs.readFile(name); value.push({ path: path.relative('/workspace', name), bytes: data.length, sha256: crypto.createHash('sha256').update(data).digest('hex'), base64: data.toString('base64') }); }
        }
      }
      await visit(target);
    } else throw new Error('未知文件操作');
    process.stdout.write(JSON.stringify({ value }));
  } catch (e) { process.stdout.write(JSON.stringify({ error: e.message, code: e.code })); }
});`;

export async function createSandbox(overrides = {}) {
  const config = { ...policy, ...overrides };
  const name = 'minillm-task-' + randomUUID();
  const cwd = path.join(root, '.local/virtual-pi-tasks', name);
  const args = ['run', '-d', '--name', name, '--network', config.network, '--read-only', '--user', config.user,
    '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', String(config.cpus), '--memory', config.memory,
    '--pids-limit', String(config.pids_limit), '--tmpfs', '/workspace:' + config.workspace_tmpfs,
    '--tmpfs', '/tmp:' + config.tmp_tmpfs, '--workdir', config.workdir, config.image, 'sleep', 'infinity'];
  const started = await docker(args);
  if (started.code !== 0) throw new Error(started.stderr);
  const containerId = started.stdout.trim();
  assert.match(containerId, /^[a-f0-9]{64}$/);
  let calls = 0, expired = false;
  const dispose = () => docker(['rm', '-f', containerId]);
  const timer = setTimeout(() => { expired = true; void dispose(); }, config.task_timeout_seconds * 1000);
  timer.unref();
  function virtualPath(input) {
    assert.equal(typeof input, 'string', '文件路径必须是字符串');
    const requested = input === '/workspace' || input.startsWith('/workspace/') ? input.slice('/workspace'.length).replace(/^\//, '') : input;
    const resolved = path.resolve(cwd, requested);
    const relative = path.relative(cwd, resolved);
    if (relative === '..' || relative.startsWith('..' + path.sep) || path.isAbsolute(relative)) throw new Error('文件路径超出任务工作区');
    return resolved;
  }
  function remotePath(absolute) {
    return '/workspace/' + path.relative(cwd, virtualPath(absolute)).split(path.sep).join('/');
  }
  async function fileOperation(op, absolute, extra = {}) {
    const result = await docker(['exec', '-i', containerId, 'node', '-e', fileWorker], JSON.stringify({ op, path: remotePath(absolute), ...extra }));
    if (result.code !== 0) throw new Error(result.stderr);
    const response = JSON.parse(result.stdout);
    if (response.error) throw new Error(response.error);
    return response.value;
  }
  const read = { readFile: async p => Buffer.from(await fileOperation('read', p), 'base64'), access: p => fileOperation('access', p, { mode: 4 }) };
  read.detectImageMimeType = async p => mime.detectSupportedImageMimeType((await read.readFile(p)).subarray(0, 4100));
  const write = { writeFile: (p, content) => fileOperation('write', p, { content }), mkdir: p => fileOperation('mkdir', p) };
  const edit = { readFile: read.readFile, writeFile: write.writeFile, access: p => fileOperation('access', p, { mode: 6 }) };
  const bash = { exec: async (command, _cwd, options) => {
    const seconds = Math.min(config.command_timeout_seconds, options.timeout ?? config.command_timeout_seconds);
    if (!Number.isFinite(seconds) || seconds <= 0) throw new Error('命令时限必须大于零');
    const onAbort = () => { void dispose(); };
    if (options.signal?.aborted) throw new Error('aborted');
    options.signal?.addEventListener('abort', onAbort, { once: true });
    try {
      const result = await docker(['exec', '-i', '--workdir', config.workdir, containerId,
        'timeout', '--kill-after=2s', String(seconds) + 's', 'bash', '--noprofile', '--norc', '-c', command], '', options.onData);
      if (options.signal?.aborted) throw new Error('aborted');
      if (result.code === 124) throw new Error('timeout:' + seconds);
      return { exitCode: result.code };
    } finally { options.signal?.removeEventListener('abort', onAbort); }
  } };
  const tools = [pi.createReadTool(cwd, { operations: read }), pi.createEditTool(cwd, { operations: edit }),
    pi.createWriteTool(cwd, { operations: write }), pi.createBashTool(cwd, { operations: bash, exposeSessionEnvironment: false })].map(tool => ({
    ...tool, execute: async (id, params, signal, onUpdate, context) => {
      if (expired) throw new Error('任务超过总时限');
      if (calls >= config.max_tool_calls) throw new Error('任务超过工具调用预算');
      calls += 1;
      const input = tool.name === 'bash' ? params : { ...params, path: virtualPath(params.path) };
      // 返回正文、补丁和错误中的文件位置统一使用容器路径。
      function visible(value) {
        if (typeof value === 'string') {
          if (input.path) value = value.replaceAll(input.path, remotePath(input.path));
          return value.replaceAll(cwd, '/workspace');
        }
        if (Array.isArray(value)) return value.map(visible);
        if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, visible(item)]));
        return value;
      }
      try {
        const result = await tool.execute(id, input, signal, onUpdate, { ...context, cwd });
        if (expired) throw new Error('任务超过总时限');
        return visible(result);
      } catch (error) { error.message = visible(error.message); throw error; }
    }
  }));
  return { name, containerId, config, tools, command: ['docker', ...args],
    inspect: async () => { const r = await docker(['inspect', containerId]); if (r.code !== 0) throw new Error(r.stderr); return JSON.parse(r.stdout)[0]; },
    snapshot: () => fileOperation('snapshot', cwd),
    dispose: async () => { clearTimeout(timer); return dispose(); } };
}
