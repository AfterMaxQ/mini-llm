"""展开冻结训练数据，检查每个当前回复的真实 token 与监督位置。"""
import argparse
import hashlib
import json
from collections import Counter

from datasets import Dataset, load_from_disk
from lab import ROOT, sha256, write_json
from templates import action_records, encode_action, read_records, resolve_data_run, tokenizer_and_template


def prepare(name, data_run):
    tokenizer, original, marked = tokenizer_and_template()
    source = ROOT / ".local/data/processed" / data_run / name
    signature = hashlib.sha256((sha256(source) + original + marked + "assistant_action_v1").encode()).hexdigest()[:16]
    directory = ROOT / ".local/data/encoded" / f"{name.removesuffix('.jsonl')}-{signature}"
    if (directory / "manifest.json").exists():
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert all(sha256(directory / p["file"]) == p["sha256"] for p in manifest["files"])
        return directory, manifest
    records = read_records(name, data_run)
    encoded, units = [], []
    for record in records:
        for action in action_records(record):
            item = encode_action(tokenizer, marked, action)
            text = tokenizer.apply_chat_template(action["messages"], tools=action["tools"], tokenize=False,
                                enable_thinking=False, add_generation_prompt=False)
            tagged = tokenizer.apply_chat_template(action["messages"], tools=action["tools"], chat_template=marked,
                                tokenize=False, enable_thinking=False, add_generation_prompt=False)
            prefix = tokenizer.apply_chat_template(action["messages"][:-1], tools=action["tools"], tokenize=False,
                                enable_thinking=False, add_generation_prompt=True)
            assert text == tagged and text.startswith(prefix)
            assert tokenizer(text, add_special_tokens=False)["input_ids"] == item["input_ids"]
            targets = sum(x != -100 for x in item["labels"][1:])
            assert targets > 0 and tokenizer.eos_token_id in item["labels"]
            item.update({"sample_id": record["sample_id"], "message_index": action["target_message_index"],
                         "messages": [{"role": m["role"], "content": m.get("content", "")} for m in action["messages"]]})
            encoded.append(item)
            units.append({"sample_id": record["sample_id"], "message_index": action["target_message_index"],
                          "input_tokens": len(item["input_ids"]), "target_tokens": targets,
                          "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
        if len(units) % 250 < len(action_records(record)):
            print(f"{name}：已展开 {len(units)} 个回复单元", flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(encoded).save_to_disk(directory / "dataset")
    with (directory / "units.jsonl").open("w", encoding="utf-8") as handle:
        for item in units: handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {"data_run": data_run, "data_file": name, "source_sha256": sha256(source),
                "independent_trajectories": len(records), "assistant_units": len(encoded),
                "input_tokens": sum(x["input_tokens"] for x in units),
                "supervised_tokens": sum(x["target_tokens"] for x in units),
                "max_sequence_length": max(x["input_tokens"] for x in units),
                "max_target_length": max(x["target_tokens"] for x in units),
                "categories": dict(Counter(r["category"] for r in records)),
                "template_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "marked_template_sha256": hashlib.sha256(marked.encode()).hexdigest(),
                "prefix_checks": len(units), "packing": False,
                "files": [{"file": str(p.relative_to(directory)).replace("\\", "/"), "sha256": sha256(p)}
                          for p in directory.rglob("*") if p.is_file()]}
    write_json(directory / "manifest.json", manifest)
    return directory, manifest


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-run", default="E02-R02"); args = parser.parse_args()
    data_run = resolve_data_run(args.data_run)
    manifests = {}
    for name in ["train-1000.jsonl", "train-5000.jsonl", "train-10000.jsonl", "dev.jsonl"]:
        directory, manifest = prepare(name, data_run)
        manifests[name] = {**manifest, "cache": str(directory.relative_to(ROOT)).replace("\\", "/")}
        print(json.dumps(manifests[name], ensure_ascii=False), flush=True)
    write_json(ROOT / ".local/data/encoded/index.json", manifests)


if __name__ == "__main__": main()
