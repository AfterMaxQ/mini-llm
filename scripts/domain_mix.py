"""按冻结规则组合公开工具轨迹和 Pi 参考轨迹。"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from archive import archive_run
from lab import ROOT, finish_run, sha256, start_run


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def choose_proportional(records, count, keys, seed, salt):
    groups = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in keys)].append(record)
    total = len(records)
    quotas = {key: count * len(group) // total for key, group in groups.items()}
    remainder = count - sum(quotas.values())
    order = sorted(groups, key=lambda key: (-(count * len(groups[key]) % total), key))
    for key in order[:remainder]:
        quotas[key] += 1
    selected = []
    for key, group in groups.items():
        ordered = sorted(group, key=lambda row: digest([seed, salt, row["sample_id"]]))
        selected.extend(ordered[:quotas[key]])
    assert len(selected) == count
    assert len({row["sample_id"] for row in selected}) == count
    return sorted(selected, key=lambda row: row["sample_id"]), quotas


def choose_per_family(records, family_key, per_family, seed):
    groups = defaultdict(list)
    for record in records:
        groups[record[family_key]].append(record)
    assert len(groups) == 16 and all(len(group) >= per_family for group in groups.values())
    selected = []
    for family, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: digest([seed, "pi-domain", family, row["sample_id"]]))
        selected.extend(ordered[:per_family])
    assert len(selected) == per_family * 16
    assert len({row["sample_id"] for row in selected}) == len(selected)
    return sorted(selected, key=lambda row: row["sample_id"]), groups


def ids_sha(records):
    return digest([row["sample_id"] for row in sorted(records, key=lambda row: row["sample_id"])])


def save_immutable(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        assert path.read_text(encoding="utf-8") == encoded, "冻结清单已存在且内容不同，不能覆盖"
    else:
        path.write_text(encoded, encoding="utf-8")


def save_rows(path, records):
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    if path.exists():
        assert path.read_text(encoding="utf-8") == encoded, "已冻结训练集不同，不能覆盖"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    return sha256(path)


def write_run_configs(config, frozen_sha, train_sha, dev_sha):
    focused_train = json.loads((ROOT / "configs/sft-focused.json").read_text(encoding="utf-8"))
    train = {**focused_train, "data_run": config["output_data_run"], "train_size": 1024,
             "data_file_hashes": {"train-1024.jsonl": train_sha, "dev.jsonl": dev_sha},
             "domain_manifest_sha256": frozen_sha}
    train.pop("sampling_manifest_sha256", None)
    save_immutable(ROOT / "configs/sft-domain.json", train)

    evaluation = json.loads((ROOT / "configs/offline-eval-focused.json").read_text(encoding="utf-8"))
    evaluation.update({"data_run": config["output_data_run"],
                       "data_file_hashes": {"dev.jsonl": dev_sha},
                       "domain_manifest_sha256": frozen_sha})
    save_immutable(ROOT / "configs/offline-eval-domain.json", evaluation)

    scale = {"experiment": "E14", "training_config": "configs/sft-domain.json",
             "evaluation_config": "configs/offline-eval-domain.json",
             "jobs": [{"size": 1024, "seed": 17}]}
    save_immutable(ROOT / "configs/scale-domain.json", scale)


def main():
    config_path = ROOT / "configs/domain-mix.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    budget = json.loads((ROOT / "configs/experiment-budget.json").read_text(encoding="utf-8"))
    assert config["sampling_seed"] == budget["sampling_seed"]
    assert config["public_count"] == budget["training"]["domain_public_trajectories"]
    assert config["pi_per_family"] * 16 == budget["training"]["domain_pi_trajectories"]
    processed = ROOT / ".local/data/processed"
    public_path = processed / config["public_data_run"] / config["public_train_file"]
    pi_path = processed / config["pi_data_run"] / config["pi_train_file"]
    dev_path = processed / config["dev_data_run"] / config["dev_file"]
    expected_sources = {"public": config["public_source_sha256"], "pi": config["pi_source_sha256"],
                        "dev": config["dev_source_sha256"]}
    for name, path in [("public", public_path), ("pi", pi_path), ("dev", dev_path)]:
        assert sha256(path) == expected_sources[name], f"{name} 来源文件哈希不符"

    public = read_rows(public_path)
    pi = read_rows(pi_path)
    dev = read_rows(dev_path)
    assert len(public) == 1000 and len(pi) == 1000 and len(dev) == 100
    assert all(row["split"] == "train" for row in public + pi)
    assert all(row["template_family"].startswith("train-") and
               row["repository_family"].startswith("train-") for row in pi)

    public_selected, quotas = choose_proportional(public, config["public_count"],
                                                   config["public_strata"], config["sampling_seed"], "domain-public")
    pi_selected, families = choose_per_family(pi, config["pi_family_key"],
                                              config["pi_per_family"], config["sampling_seed"])
    combined = public_selected + pi_selected
    assert len(combined) == 1024 and len({row["sample_id"] for row in combined}) == len(combined)
    combined.sort(key=lambda row: digest([config["sampling_seed"], "domain-order", row["sample_id"]]))
    train_ids = {row["sample_id"] for row in combined}
    assert len(train_ids) == 1024 and not (train_ids & {row["sample_id"] for row in dev})

    manifest = {
        "sampling_seed": config["sampling_seed"],
        "domain_config_sha256": sha256(config_path),
        "experiment_budget_sha256": sha256(ROOT / "configs/experiment-budget.json"),
        "selection": "公开数据按来源和类别比例分层，Pi 数据按 16 个训练模板族各抽 32 条；只按 id 哈希排序，不读取模型结果",
        "sources": {
            "public": {"path": public_path.relative_to(ROOT).as_posix(), "source_sha256": sha256(public_path),
                       "source_count": len(public), "selected_count": len(public_selected),
                       "ids": [row["sample_id"] for row in public_selected], "ids_sha256": ids_sha(public_selected),
                       "strata": {str(key): value for key, value in sorted(quotas.items())}},
            "pi": {"path": pi_path.relative_to(ROOT).as_posix(), "source_sha256": sha256(pi_path),
                   "source_count": len(pi), "selected_count": len(pi_selected),
                   "ids": [row["sample_id"] for row in pi_selected], "ids_sha256": ids_sha(pi_selected),
                   "families": {family: len([r for r in pi_selected if r[config["pi_family_key"]] == family])
                                for family in sorted(families)}},
            "dev": {"path": dev_path.relative_to(ROOT).as_posix(), "source_sha256": sha256(dev_path),
                    "selected_count": len(dev), "ids_sha256": ids_sha(dev)}
        }
    }
    frozen_path = ROOT / "configs/domain-mix-frozen.json"
    save_immutable(frozen_path, manifest)
    output = processed / config["output_data_run"]
    output.mkdir(parents=True, exist_ok=True)
    train_hash = save_rows(output / "train-1024.jsonl", combined)
    dev_hash = save_rows(output / "dev.jsonl", dev)
    frozen_hash = sha256(frozen_path)
    write_run_configs(config, frozen_hash, train_hash, dev_hash)
    result = {"status": "domain_mix_frozen", "exit_code": 0, "train_size": len(combined),
              "public_trajectories": len(public_selected), "pi_trajectories": len(pi_selected),
              "public_strata": {str(key): value for key, value in sorted(quotas.items())},
              "pi_families": len(families), "pi_per_family": config["pi_per_family"],
              "dev_trajectories": len(dev), "files": {"train-1024.jsonl": train_hash, "dev.jsonl": dev_hash},
              "domain_manifest_sha256": frozen_hash,
              "scope": "冻结512条公开工具轨迹与512条Pi训练参考轨迹；沿用100条固定公开dev，不读取模型成绩"}
    run = start_run("E14", {**config, **result})
    result["files"]["domain-mix-frozen.json"] = sha256(frozen_path)
    result = finish_run(run, result)
    archive_run(run.name)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
