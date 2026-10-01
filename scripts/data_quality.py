"""固定规则筛选训练区，再按来源和类别匹配；不改写公开标签。"""
import argparse
import json
import math
import random
import re
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation

from data import canonical, hash_value, save_jsonl
from lab import ROOT, finish_run, sha256, start_run, write_json


def words(value):
    return tuple(re.findall(r"\w+", value.casefold()))


def leaves(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f"{path}.{key}".lstrip("."))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{path}[{index}]")
    else:
        yield path, value


def numbers(text):
    # 千位逗号与小数保留数值含义；日期拆成的数字不能证明日期转换正确。
    pattern = r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)"
    result = set()
    for match in re.finditer(pattern, text):
        try:
            result.add(Decimal(match.group().replace(",", "")))
        except InvalidOperation:
            pass
    return result


def inspect_record(record, config):
    evidence, decisions, issues = [], [], []
    exact = set(config["placeholder_exact"])
    placeholder = re.compile(config["placeholder_pattern"], re.I)
    for index, message in enumerate(record["messages"]):
        if message["role"] in ["user", "tool"]:
            text = message.get("content", "")
            evidence.append({"message_index": index, "role": message["role"], "words": words(text),
                             "numbers": numbers(text)})
            if message["role"] == "tool":
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    value = text
                for path, item in leaves(value):
                    if isinstance(item, str) and (item.strip().casefold() in exact or placeholder.search(item)):
                        issues.append({"reason": "placeholder_tool_return", "message_index": index,
                                       "path": path, "value": item})
        if message["role"] != "assistant":
            continue
        for call in message.get("tool_calls", []):
            for path, value in leaves(call["function"]["arguments"]):
                decision = {"message_index": index, "call_id": call["id"],
                            "tool": call["function"]["name"], "path": path, "value": value}
                if value is None or isinstance(value, bool):
                    decision.update({"status": "not_checked_type", "evidence_messages": []})
                elif isinstance(value, str):
                    target = words(value)
                    matches = [e["message_index"] for e in evidence if target and any(
                        e["words"][j:j + len(target)] == target for j in range(len(e["words"]) - len(target) + 1))]
                    decision.update({"status": "surface_supported" if matches else "unverified_string",
                                     "evidence_messages": matches})
                elif isinstance(value, (int, float)):
                    matches = [e["message_index"] for e in evidence
                               if math.isfinite(value) and Decimal(str(value)) in e["numbers"]]
                    decision.update({"status": "surface_supported" if matches else "unverified_number",
                                     "evidence_messages": matches})
                else:
                    raise ValueError(f"未处理的参数类型：{type(value).__name__}")
                decisions.append(decision)
                if decision["status"].startswith("unverified"):
                    issues.append({"reason": decision["status"], **decision})
    return {"sample_id": record["sample_id"], "group_id": record["group_id"],
            "source": record["source"], "category": record["category"],
            "accepted": not issues, "issues": issues, "argument_checks": decisions}


def train_representatives(base_run, config):
    manifest = [json.loads(line) for line in (base_run / "pool-manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assignment = {}
    candidates = defaultdict(list)
    for row in manifest:
        group, split = row["group_id"], row["assigned_split"]
        if group in assignment:
            assert assignment[group] == split
        assignment[group] = split
        if split == "train" and row["tokens"] <= config["training_context"]:
            candidates[group].append(row["sample_id"])
    selected_ids = {sorted(ids, key=lambda sample: hash_value([2026, sample]))[0]
                    for ids in candidates.values()}
    selected = {}
    with (base_run / "normalized-pool.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_id"] in selected_ids:
                assert assignment[row["group_id"]] == "train"
                row["split"] = "train"
                selected[row["group_id"]] = row
    assert len(selected) == len(candidates)
    return [selected[group] for group in sorted(selected, key=lambda group: hash_value([2026, group]))], assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data-quality.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "configs/data-frozen.json").read_text(encoding="utf-8"))
    base_run = ROOT / ".local/runs" / config["base_data_run"]
    data = ROOT / ".local/data/processed" / config["base_data_run"]
    inputs = [base_run / "pool-manifest.jsonl", base_run / "normalized-pool.jsonl",
              data / "train-5000.jsonl", data / "train-10000.jsonl", data / "dev.jsonl"]
    run = start_run("E12", {**config, "operation": "data_selection", "command": [sys.executable, *sys.argv],
                            "inputs": [{"file": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for p in inputs]})
    started = time.perf_counter()
    try:
        assert config["reference_size"] == 5000 and frozen["run_id"] == config["base_data_run"]
        for name in ["train-5000.jsonl", "train-10000.jsonl", "dev.jsonl"]:
            assert sha256(data / name) == frozen["outputs"][name]["sha256"]
        pool, assignment = train_representatives(base_run, config)
        original = [json.loads(line) for line in (data / "train-10000.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [canonical(r) for r in pool[:10000]] == [canonical(r) for r in original]
        reference = pool[:config["reference_size"]]
        quota = Counter((r["source"], r["category"]) for r in reference)
        checks = [inspect_record(r, config) for r in pool]
        save_jsonl(run / "screening.jsonl", checks)
        available = Counter((r["source"], r["category"]) for r, check in zip(pool, checks) if check["accepted"])
        shortages = {f"{source}/{category}": count - available[(source, category)]
                     for (source, category), count in quota.items() if available[(source, category)] < count}
        if shortages:
            raise RuntimeError(f"筛选后匹配配额不足，规则与类别不缩减：{shortages}")
        selected, used = [], Counter()
        for record, check in zip(pool, checks):
            key = (record["source"], record["category"])
            if check["accepted"] and used[key] < quota[key]:
                selected.append(record)
                used[key] += 1
        assert used == quota and len(selected) == len(reference) == 5000
        assert len({r["group_id"] for r in selected}) == 5000
        assert all(assignment[r["group_id"]] == "train" for r in selected)
        output = ROOT / ".local/data/processed" / run.name
        output.mkdir(parents=True, exist_ok=False)
        save_jsonl(output / "train-5000.jsonl", selected)
        shutil.copy2(data / "dev.jsonl", output / "dev.jsonl")
        assert sha256(output / "dev.jsonl") == frozen["outputs"]["dev.jsonl"]["sha256"]
        accepted = [i for i, check in enumerate(checks) if check["accepted"]]
        rejected = [i for i, check in enumerate(checks) if not check["accepted"]]
        randomizer = random.Random(config["inspection_seed"])
        review = [{"arm": arm, "record": pool[i], "screening": checks[i]} for arm, indices in
                  [("accepted", accepted), ("rejected", rejected)] for i in
                  randomizer.sample(indices, config["inspection_per_arm"])]
        save_jsonl(run / "inspection-50.jsonl", review)
        reference_ids = {r["sample_id"] for r in reference}
        selected_ids = {r["sample_id"] for r in selected}
        save_jsonl(run / "selection-manifest.jsonl", [{"sample_id": r["sample_id"], "group_id": r["group_id"],
                    "source": r["source"], "category": r["category"], "baseline": r["sample_id"] in reference_ids,
                    "filtered": r["sample_id"] in selected_ids} for r in pool
                    if r["sample_id"] in reference_ids | selected_ids])
        summary = {"status": "data_selected_pending_review", "exit_code": 0, "operation": "data_selection",
                   "policy": config["policy_version"], "candidate_train_groups": len(pool),
                   "accepted": len(accepted), "rejected": len(rejected), "selected": len(selected),
                   "baseline_retained": len(reference_ids & selected_ids),
                   "baseline_replaced": len(reference_ids - selected_ids),
                   "matched_strata": [{"source": source, "category": category, "reference": count,
                                        "filtered": used[(source, category)], "available": available[(source, category)]}
                                       for (source, category), count in sorted(quota.items())],
                   "reason_counts": dict(Counter(reason for check in checks for reason in
                                                  {issue["reason"] for issue in check["issues"]})),
                   "argument_status": dict(Counter(d["status"] for c in checks for d in c["argument_checks"])),
                   "screening_seconds": time.perf_counter() - started,
                   "outputs": {name: {"sha256": sha256(output / name), "samples": 5000 if name.startswith("train") else 500}
                               for name in ["train-5000.jsonl", "dev.jsonl"]},
                   "evidence": [{"file": p.name, "sha256": sha256(p)} for p in
                                [run / "screening.jsonl", run / "inspection-50.jsonl", run / "selection-manifest.jsonl"]],
                   "frozen_prefix_identical": True, "evaluation_samples_screened": 0,
                   "gpu_model_calls": 0, "inspection_status": "pending"}
        finish_run(run, summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception:
        (run / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finish_run(run, {"status": "failed", "exit_code": 1, "operation": "data_selection",
                         "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
