"""冻结 ARC-Challenge 官方测试集与 harness 请求，不加载模型或计算分数。"""
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

import truststore

truststore.inject_into_ssl()
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as parquet
import requests
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks._yaml_loader import load_yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from lab import finish_run, sha256, start_run, write_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def prepare():
    config = read_json(ROOT / "configs/arc-eval.json")
    run = start_run("E24", config)
    try:
        dataset = config["dataset"]
        harness = config["harness"]
        vendor = ROOT / ".local/vendor/lm-evaluation-harness"
        commit = subprocess.check_output(
            ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != harness["commit"]:
            raise RuntimeError(f"harness commit 不匹配：{commit}")

        task_file = vendor / harness["task_yaml"]
        base_file = vendor / harness["base_yaml"]
        installed_arc = Path(sys.prefix) / "Lib/site-packages/lm_eval/tasks/arc"
        task_hashes = {
            "task_yaml_sha256": sha256(task_file),
            "base_yaml_sha256": sha256(base_file),
            "installed_task_yaml_sha256": sha256(installed_arc / "arc_challenge.yaml"),
            "installed_base_yaml_sha256": sha256(installed_arc / "arc_easy.yaml"),
        }
        if task_hashes["task_yaml_sha256"] != task_hashes["installed_task_yaml_sha256"]:
            raise RuntimeError("安装版 task yaml 与锁定源码不一致")
        if task_hashes["base_yaml_sha256"] != task_hashes["installed_base_yaml_sha256"]:
            raise RuntimeError("安装版基础 yaml 与锁定源码不一致")

        api_url = f"https://huggingface.co/api/datasets/{dataset['repo_id']}/revision/{dataset['revision']}"
        response = requests.get(api_url, params={"blobs": "true"}, timeout=(30, 60))
        response.raise_for_status()
        metadata = response.json()
        if metadata["sha"] != dataset["revision"]:
            raise RuntimeError(f"数据集 revision 不匹配：{metadata['sha']}")
        file_info = next((f for f in metadata["siblings"] if f["rfilename"] == dataset["test_file"]), None)
        file_lfs = file_info.get("lfs") if file_info else None
        source_sha = file_lfs.get("sha256") if file_lfs else None
        if source_sha != dataset["test_file_sha256"] or file_info["size"] != dataset["test_file_bytes"]:
            raise RuntimeError(f"官方 ARC test 元数据不匹配：{source_sha}")

        raw_dir = ROOT / ".local/data/arc-challenge"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / "test-00000-of-00001.parquet"
        if not raw_file.exists():
            file_url = (
                f"https://huggingface.co/datasets/{dataset['repo_id']}/resolve/"
                f"{dataset['revision']}/{dataset['test_file']}"
            )
            response = requests.get(file_url, stream=True, timeout=(30, 120))
            response.raise_for_status()
            partial = raw_file.with_suffix(raw_file.suffix + ".part")
            with partial.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        output.write(block)
            response.close()
            if partial.stat().st_size != dataset["test_file_bytes"] or sha256(partial) != source_sha:
                raise RuntimeError("下载后的 ARC test Parquet 与官方源哈希不符")
            partial.replace(raw_file)
        raw_sha = sha256(raw_file)
        if raw_sha != source_sha or raw_file.stat().st_size != dataset["test_file_bytes"]:
            raise RuntimeError(f"本地 ARC test 校验失败：{raw_sha}")

        table = parquet.read_table(raw_file)
        raw_rows = table.to_pylist()
        required = {"id", "question", "choices", "answerKey"}
        if not required.issubset(table.column_names):
            raise RuntimeError(f"ARC test 字段不齐：{table.column_names}")
        raw_ids = [str(row["id"]) for row in raw_rows]
        if len(set(raw_ids)) != len(raw_ids):
            raise RuntimeError("ARC test split 中出现重复 id")
        if any(not isinstance(row["question"], str) or not row["question"].strip() for row in raw_rows):
            raise RuntimeError("ARC test split 存在空问题")

        task_config = load_yaml(task_file, recursive=True)
        task_config.update({
            "dataset_path": "parquet",
            "dataset_name": None,
            "dataset_kwargs": {"data_files": {"test": str(raw_file)}},
            "training_split": None,
            "validation_split": None,
            "num_fewshot": 0,
        })
        task = ConfigurableTask(config=task_config)
        docs = list(task.test_docs())
        if len(docs) != len(raw_rows) or {str(doc["id"]) for doc in docs} != set(raw_ids):
            raise RuntimeError("harness 实际 test split 与锁定 parquet 的行数或 id 不一致")

        choice_counts = Counter()
        question_chars = []
        rendered_hashes = []
        expected_requests = 0
        for doc in docs:
            choices = task.doc_to_choice(doc)
            target = task.doc_to_target(doc)
            if not choices or not isinstance(target, int) or not 0 <= target < len(choices):
                raise RuntimeError(f"选项或目标索引无效：{doc['id']}")
            context = task.fewshot_context(doc, num_fewshot=0)
            if not context.startswith("Question: ") or not context.endswith("Answer:"):
                raise RuntimeError(f"ARC prompt 模板渲染异常：{doc['id']}")
            choice_counts[len(choices)] += 1
            question_chars.append(len(doc["question"]))
            rendered_hashes.append(digest(context.encode("utf-8")))
            expected_requests += len(choices)

        task.build_all_requests(limit=None, cache_requests=False, apply_chat_template=False)
        instances = task.instances
        request_types = Counter(instance.request_type for instance in instances)
        if len(instances) != expected_requests or set(request_types) != {"loglikelihood"}:
            raise RuntimeError(f"评分请求数量或类型异常：{len(instances)} {request_types}")
        if {str(instance.doc["id"]) for instance in instances} != set(raw_ids):
            raise RuntimeError("评分请求未覆盖完整 test split")

        result = {
            "status": "preparation_verified_not_scored",
            "exit_code": 0,
            "scope": "官方 test 数据、task 模板与全量评分请求已核对；未加载模型，未计算模型指标",
            "dataset": {
                "repo_id": dataset["repo_id"],
                "revision": metadata["sha"],
                "config": dataset["config"],
                "split": config["evaluation"]["split"],
                "file": dataset["test_file"],
                "local_file": ".local/data/arc-challenge/test-00000-of-00001.parquet",
                "file_bytes": raw_file.stat().st_size,
                "file_sha256": raw_sha,
                "documents": len(docs),
                "unique_ids": len(set(raw_ids)),
                "ids_sha256": digest("\n".join(sorted(raw_ids)).encode("utf-8")),
                "fields": sorted(table.column_names),
                "choice_count_distribution": {str(k): choice_counts[k] for k in sorted(choice_counts)},
                "question_character_count": {
                    "min": min(question_chars),
                    "median": statistics.median(question_chars),
                    "max": max(question_chars),
                },
            },
            "harness": {
                **harness,
                "repository_commit": commit,
                **task_hashes,
                "num_fewshot": task.config.num_fewshot,
                "prompt_template": config["evaluation"]["prompt_template"],
                "choice_template": config["evaluation"]["choice_template"],
                "target_template": config["evaluation"]["target_template"],
                "prompt_template_sha256": digest(config["evaluation"]["prompt_template"].encode("utf-8")),
                "rendered_prompts_sha256": digest("\n".join(rendered_hashes).encode("ascii")),
                "apply_chat_template": False,
                "limit": None,
                "scoring_backend": config["evaluation"]["scoring_backend"],
                "metrics": config["evaluation"]["metrics"],
                "dataset_loader": "parquet with the fixed local test file; train/validation split disabled",
            },
            "scoring_requests": {
                "documents": len(docs),
                "choices": len(instances),
                "request_types": dict(request_types),
            },
            "model_inference": {"performed": False, "scores": None},
        }

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
        keys = sorted(choice_counts)
        axes[0].bar([str(k) for k in keys], [choice_counts[k] for k in keys], color="#376f9f", width=0.68)
        axes[0].set_title(f"测试题选项数分布（n={len(docs)}）")
        axes[0].set_xlabel("每题选项数")
        axes[0].set_ylabel("题目数")
        axes[0].grid(axis="y", alpha=0.22)
        axes[1].hist(question_chars, bins=18, color="#d7883b", edgecolor="white")
        axes[1].set_title("题干长度分布")
        axes[1].set_xlabel("英文字符数")
        axes[1].set_ylabel("题目数")
        axes[1].grid(axis="y", alpha=0.22)
        figure = ROOT / "experiments/E24/figures/arc-test-structure.png"
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=180)
        plt.close(fig)

        source = {
            "source_file_sha256": raw_sha,
            "dataset_revision": metadata["sha"],
            "documents": len(docs),
            "choice_count_distribution": result["dataset"]["choice_count_distribution"],
            "question_character_count": result["dataset"]["question_character_count"],
            "figure_sha256": sha256(figure),
            "note": "汇总题干长度与选项数，不使用 test 答案分布。",
        }
        source_path = ROOT / "experiments/E24/figures/arc-test-structure.source.json"
        write_json(source_path, source)
        result["chart"] = {
            "file": "experiments/E24/figures/arc-test-structure.png",
            "source": "experiments/E24/figures/arc-test-structure.source.json",
            "sha256": sha256(figure),
            "source_json_sha256": sha256(source_path),
        }
        final = finish_run(run, result)
        print(json.dumps(final, ensure_ascii=False, indent=2))
    except Exception as error:
        finish_run(run, {
            "status": "preparation_failed",
            "exit_code": 1,
            "error_type": type(error).__name__,
            "error": str(error),
            "scope": "评测器与测试集准备失败；没有加载模型或计算分数",
        })
        raise


if __name__ == "__main__":
    prepare()
