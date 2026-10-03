"""顺序完成短提示基线、专项微调和同提示评测，不并行占用 GPU。"""
import json
import os
import subprocess
import time

import psutil

from lab import ROOT, now, sha256, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def alive(identity):
    try:
        return abs(psutil.Process(identity['pid']).create_time() - identity['created']) < 1
    except psutil.Error:
        return False


def main():
    queue_lock = ROOT / '.local/focused-tools-queue.lock.json'
    identity = {'pid': os.getpid(), 'created': psutil.Process().create_time(), 'command': psutil.Process().cmdline()}
    with queue_lock.open('x', encoding='utf-8') as handle:
        json.dump(identity, handle)
    state_path = ROOT / '.local/focused-tools-state.json'
    gpu_lock = ROOT / '.local/scale-lock.json'
    python = ROOT / '.local/venv-train/Scripts/python.exe'
    node = 'G:/program files/Node_js/node.exe'

    def state(status, **details):
        write_json(state_path, {'status': status, 'time': now(), 'process_identity': identity, **details})

    def wait_gpu():
        for lock in (gpu_lock, ROOT / '.local/pi-agent-lock.json'):
            while lock.exists():
                if not alive(read(lock)):
                    raise RuntimeError(f'发现失效锁，须先核对原运行：{lock.name}')
                time.sleep(2)

    def launch(arguments, label, training=False):
        wait_gpu()
        if training:
            with gpu_lock.open('x', encoding='utf-8') as handle:
                json.dump(identity, handle)
        try:
            log = ROOT / '.local/logs' / f'focused-{label}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('x', encoding='utf-8') as handle:
                process = subprocess.Popen(arguments, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                           env={**os.environ, 'PYTHONUTF8': '1'}, creationflags=subprocess.CREATE_NO_WINDOW)
                if process.wait() != 0:
                    raise RuntimeError(f'{label}失败，原记录保留；日志：{log.name}')
        finally:
            if training and gpu_lock.exists() and read(gpu_lock) == identity:
                gpu_lock.unlink()

    try:
        baseline = ROOT / '.local/runs/E14-R31'
        state('waiting_baseline', baseline_run='E14-R31')
        while not (baseline / 'result.json').exists():
            lock = ROOT / '.local/pi-agent-lock.json'
            if lock.exists() and not alive(read(lock)):
                raise RuntimeError('短提示基线进程已结束，尚无完整结果')
            time.sleep(2)
        baseline_result = read(baseline / 'result.json')
        assert baseline_result['status'] == 'completed' and baseline_result['evaluated_tasks'] == 16
        state('training', baseline_run='E14-R31', baseline_passed=baseline_result['passed_tasks'])
        launch([str(python), 'scripts/sft.py', '--config', 'configs/sft-pi-focused.json', '--experiment', 'E14'],
               'train', training=True)
        candidates = [p.parent for p in (ROOT / '.local/runs').glob('E14-R*/config.json')
                      if read(p).get('data_run') == 'E14-focused-tools' and (p.parent / 'result.json').exists()]
        assert len(candidates) == 1
        training = candidates[0]
        trained = read(training / 'result.json')
        assert trained['status'] == 'trained_pending_tool_eval' and trained['steps'] > 0
        config = read(ROOT / 'configs/pi-agent-focused-baseline-dev.json')
        config.update(source_run=training.name, source_eval_run=None, condition='pi_focused_sft',
                      adapter=f'.local/runs/{training.name}/selected-adapter',
                      adapter_sha256=sha256(training / 'selected-adapter/adapter_model.safetensors'),
                      notes_script='scripts/notes_focused.py')
        config['api']['model'] = 'qwen3-1.7b-' + training.name.lower()
        write_json(ROOT / 'configs/pi-agent-focused-trained-dev.json', config)
        state('evaluating', baseline_run='E14-R31', training_run=training.name)
        launch([node, 'scripts/pi_agent_eval.mjs', 'configs/pi-agent-focused-trained-dev.json'], 'dev')
        evaluations = [read(p) for p in (ROOT / 'experiments/E14/runs').glob('*.json')
                       if read(p).get('config', {}).get('condition') == 'pi_focused_sft']
        assert len(evaluations) == 1 and evaluations[0]['status'] == 'completed' and evaluations[0]['evaluated_tasks'] == 16
        evaluation = evaluations[0]
        state('completed', baseline_run='E14-R31', training_run=training.name, evaluation_run=evaluation['run_id'],
              baseline_passed=baseline_result['passed_tasks'], trained_passed=evaluation['passed_tasks'], tasks=16)
    except Exception as error:
        state('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if queue_lock.exists() and read(queue_lock) == identity:
            queue_lock.unlink()


if __name__ == '__main__':
    main()
