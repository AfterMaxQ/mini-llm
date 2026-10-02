"""在推理前冻结样本；只按数据属性抽取，不读取模型成绩。"""
import csv
import hashlib
import json
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from lab import ROOT, sha256, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def choose(items, count, strata, identifier, salt):
    groups = defaultdict(list)
    for item in items:
        groups[strata(item)].append(item)
    keys = sorted(groups, key=str)
    allocation = {key: int(count >= len(keys)) for key in keys}
    assert 0 < count <= len(items)
    while sum(allocation.values()) < count:
        eligible = [key for key in keys if allocation[key] < len(groups[key])]
        key = max(eligible, key=lambda key: len(groups[key]) * count / len(items) - allocation[key])
        allocation[key] += 1
    selected = []
    for key in keys:
        ordered = sorted(groups[key], key=lambda item: digest([2026, salt, identifier(item)]))
        selected.extend(ordered[:allocation[key]])
    selected.sort(key=identifier)
    assert len(selected) == len({identifier(item) for item in selected}) == count
    return selected


def length_bin(length):
    return sum(length >= threshold for threshold in [512, 1024, 1536])


def manifest(path, original, selected, identifier, strata):
    return {"source_sha256": sha256(path), "source_count": len(original), "selected_count": len(selected),
            "ids": [identifier(item) for item in selected],
            "ids_sha256": digest([identifier(item) for item in selected]),
            "strata": dict(Counter(str(strata(item)) for item in selected))}


def save_rows(path, records):
    content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    if path.exists():
        assert path.read_text(encoding="utf-8") == content, "已冻结文件不同，不能覆盖"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return sha256(path)


def decision_count(records):
    return sum(message["role"] == "assistant" and
               bool(message.get("tool_calls") or record["messages"][index - 1]["role"] == "user")
               for record in records for index, message in enumerate(record["messages"]))


def main():
    budget = read(ROOT / "configs/experiment-budget.json")
    public = read(ROOT / "configs/data-frozen.json")
    quality = read(ROOT / "configs/data-quality-frozen.json")
    processed = ROOT / ".local/data/processed"
    source = processed / public["run_id"]
    result = {"sampling_seed": budget["sampling_seed"], "budget_sha256": sha256(ROOT / "configs/experiment-budget.json"),
              "selection": "按来源、类别、长度或任务族分层，层内按seed与id哈希排序；不读取模型结果"}
    identity = lambda record: record["sample_id"]
    attributes = lambda record: (record["source"], length_bin(record["token_count"]))
    dev_path = source / "dev.jsonl"
    dev = rows(dev_path)
    assert sha256(dev_path) == public["outputs"]["dev.jsonl"]["sha256"]
    selected_dev = []
    for category, count in budget["public_dev_quotas"].items():
        selected_dev.extend(choose([r for r in dev if r["category"] == category], count, attributes, identity, "public-dev"))
    selected_dev.sort(key=identity)
    result["public_dev"] = manifest(dev_path, dev, selected_dev, identity, lambda r: (r["source"], r["category"], attributes(r)[1]))
    result["public_dev"]["decision_turns"] = decision_count(selected_dev)
    test_path = source / "test.jsonl"
    test = rows(test_path)
    assert sha256(test_path) == public["outputs"]["test.jsonl"]["sha256"]
    selected_test = choose(test, budget["public_test_trajectories"],
                           lambda r: (r["category"], r["source"], length_bin(r["token_count"])), identity, "public-test")
    result["public_test"] = manifest(test_path, test, selected_test, identity, lambda r: (r["source"], r["category"]))
    result["public_test"]["decision_turns"] = decision_count(selected_test)
    train_path = source / "train-1000.jsonl"
    baseline = rows(train_path)
    assert sha256(train_path) == public["outputs"]["train-1000.jsonl"]["sha256"]
    quality_path = processed / quality["run_id"] / "train-5000.jsonl"
    assert sha256(quality_path) == quality["outputs"]["train-5000.jsonl"]["sha256"]
    pool = rows(quality_path)
    matched = []
    quotas = Counter((r["source"], r["category"]) for r in baseline)
    for key, count in sorted(quotas.items()):
        matched.extend(choose([r for r in pool if (r["source"], r["category"]) == key], count,
                              lambda r: length_bin(r["token_count"]), identity, "quality-train"))
    matched.sort(key=identity)
    assert Counter((r["source"], r["category"]) for r in matched) == quotas
    assert not {identity(r) for r in baseline + matched} & {identity(r) for r in dev + test}
    result["quality_train"] = manifest(quality_path, pool, matched, identity, lambda r: (r["source"], r["category"]))
    result["datasets"] = {}
    for name, training in [("focused-public", baseline), ("focused-quality", matched)]:
        files = {"train-1000.jsonl": training, "dev.jsonl": selected_dev, "test.jsonl": selected_test}
        result["datasets"][name] = {"files": {file: save_rows(processed / name / file, data) for file, data in files.items()},
                                    "train_count": len(training), "dev_count": len(selected_dev), "test_count": len(selected_test)}
    for split, per_category in [("dev", budget["pi"]["dev_per_category"]), ("test", budget["pi"]["test_per_category"])]:
        path = ROOT / f"experiments/E14/{split}-reference.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            tasks = list(csv.DictReader(handle))
        selected = []
        for category in sorted({r["category"] for r in tasks}):
            selected.extend(choose([r for r in tasks if r["category"] == category], per_category,
                                   lambda r: r["template_family"], lambda r: r["task_id"], "pi-" + split))
        selected.sort(key=lambda r: r["task_id"])
        assert {r["template_family"] for r in selected} == {r["template_family"] for r in tasks}
        result["pi_" + split] = manifest(path, tasks, selected, lambda r: r["task_id"], lambda r: r["category"])
        if split == "test":
            repeat = []
            for category in sorted({r["category"] for r in selected}):
                repeat.extend(choose([r for r in selected if r["category"] == category], budget["pi"]["repeat_per_category"],
                                     lambda r: r["template_family"], lambda r: r["task_id"], "pi-repeat"))
            repeat.sort(key=lambda r: r["task_id"])
            result["pi_repeat"] = manifest(path, tasks, repeat, lambda r: r["task_id"], lambda r: r["category"])
    bfcl = read(ROOT / "configs/bfcl-frozen.json")
    result["bfcl"] = {"source_manifest_sha256": sha256(ROOT / "configs/bfcl-frozen.json"), "categories": {}}
    for category in bfcl["categories"]:
        name = category["category"]
        path = ROOT / ".local/bfcl/data" / bfcl["run_id"] / f"{bfcl['dataset_version']}_{name}.json"
        assert sha256(path) == category["question_sha256"]
        questions = rows(path)
        selected = choose(questions, min(budget["bfcl_per_category"], len(questions)),
                          lambda q: (min(len(q.get("function", [])), 3), len(str(q.get("question", ""))) // 1000),
                          lambda q: q["id"], "bfcl-" + name)
        result["bfcl"]["categories"][name] = manifest(path, questions, selected, lambda q: q["id"], lambda q: len(q.get("function", [])))
    result["bfcl"]["selected_count"] = sum(c["selected_count"] for c in result["bfcl"]["categories"].values())
    arc = read(ROOT / "configs/arc-eval.json")
    path = ROOT / ".local/data/arc-challenge/test-00000-of-00001.parquet"
    assert sha256(path) == arc["dataset"]["test_file_sha256"]
    questions = pq.read_table(path).to_pylist()
    strata = lambda q: (len(q["choices"]["text"]), len(q["question"]) // 200)
    selected = choose(questions, budget["arc_test_questions"], strata, lambda q: q["id"], "arc-test")
    result["arc_test"] = manifest(path, questions, selected, lambda q: q["id"], strata)
    output = ROOT / "configs/subsets-frozen.json"
    if output.exists():
        assert read(output) == result, "抽样清单已冻结，不能静默改变"
    else:
        write_json(output, result)
    print(json.dumps({key: value.get("selected_count", len(value.get("ids", []))) for key, value in result.items()
                      if isinstance(value, dict) and ("ids" in value or "selected_count" in value)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
