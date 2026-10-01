"""实验共用的少量文件操作：运行号、来源快照与原始记录。"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows 读句柄可能短暂阻止替换；临时文件独立，关闭句柄后重试。
    for attempt in range(12):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            if attempt == 0:
                print(f"记录暂时被占用，重试替换：{path.name}", file=sys.stderr, flush=True)
            time.sleep(min(0.05 * (attempt + 1), 0.2))


def command(args, run_dir, name):
    started = now()
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (run_dir / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (run_dir / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    return {"command": args, "started": started, "finished": now(), "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def start_run(experiment, config, run_dir=None):
    runs = ROOT / ".local/runs"
    runs.mkdir(parents=True, exist_ok=True)
    if run_dir is None:
        number = 1
        while (runs / f"{experiment}-R{number:02d}").exists() or (ROOT / "experiments" / experiment / "runs" / f"{experiment}-R{number:02d}.json").exists():
            number += 1
        run_dir = runs / f"{experiment}-R{number:02d}"
    run_dir = Path(run_dir).resolve()
    public = ROOT / "experiments" / experiment / "runs" / f"{run_dir.name}.json"
    if public.exists():
        raise FileExistsError(f"运行号已有公开结果，不能覆盖：{run_dir.name}")
    run_dir.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_files = sorted(p for folder in ["scripts", "configs"] for p in (ROOT / folder).rglob("*")
                          if p.is_file() and "__pycache__" not in p.parts and "node_modules" not in p.parts)
    with zipfile.ZipFile(run_dir / "code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files:
            archive.write(path, path.relative_to(ROOT))
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    (run_dir / "code.diff").write_bytes(diff)
    write_json(run_dir / "config.json", {**config, "run_id": run_dir.name, "started": now(), "commit": commit,
                                        "process_id": os.getpid(), "code_snapshot_sha256": sha256(run_dir / "code.zip")})
    return run_dir


def finish_run(run_dir, result):
    previous=[]
    if (run_dir/'result.json').exists():
        number=1
        while (run_dir/f'previous-result-{number:02d}.json').exists():number+=1
        shutil.copy2(run_dir/'result.json',run_dir/f'previous-result-{number:02d}.json')
    for path in sorted(run_dir.glob('previous-result-*.json')):
        previous.append({'file':path.name,'sha256':sha256(path)})
    if previous:result={**result,'previous_results':previous}
    result = {**result, "run_id": run_dir.name, "finished": now()}
    write_json(run_dir / "result.json", result)
    experiment = run_dir.name.split("-")[0]
    public = ROOT / "experiments" / experiment / "runs" / f"{run_dir.name}.json"
    write_json(public, {**result, "config": json.loads((run_dir / "config.json").read_text(encoding="utf-8")),
                        "raw_result_sha256": sha256(run_dir / "result.json")})
    return result
