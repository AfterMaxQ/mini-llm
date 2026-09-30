"""把公开工具数据转换、校验、分组，再冻结独立划分。"""
import argparse
import ast
import csv
import hashlib
import json
import random
import re
import sys
import traceback
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from jsonschema import Draft202012Validator
from transformers import AutoTokenizer

from lab import ROOT, finish_run, now, sha256, start_run, write_json

SYSTEM = "You are a helpful assistant. Use the provided tools when needed. Ask for missing required information instead of guessing."


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_value(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def parse_call(text, flags):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Glaive 原标注混用 JSON 与 Python 字符串引号；只做安全反序列化。
        value = ast.literal_eval(text)
        flags.append("python_literal_call")
    if isinstance(value.get("arguments"), str):
        value["arguments"] = json.loads(value["arguments"])
        flags.append("string_arguments")
    if not isinstance(value.get("arguments"), dict):
        raise ValueError("arguments_not_object")
    return value


def glaive_tools(text):
    decoder = json.JSONDecoder()
    tools, offset = [], 0
    while (begin := text.find("{", offset)) >= 0:
        value, size = decoder.raw_decode(text[begin:])
        if "name" not in value:
            raise ValueError("tool_definition_missing_name")
        tools.append({"type": "function", "function": value})
        offset = begin + size
    return tools


def convert(row, source, source_id, metadata):
    flags, pending, messages = [], [], [{"role": "system", "content": SYSTEM}]
    if source == "hermes":
        tools = row["tools"]
        tools = json.loads(tools) if isinstance(tools, str) else tools
        turns = [(x["from"], x["value"]) for x in row["conversations"] if x["from"] != "system"]
    else:
        tools = glaive_tools(row["system"])
        chunks = re.split(r"(?:^|\n)\s*(USER|ASSISTANT|FUNCTION RESPONSE):\s*", row["chat"])
        if chunks[0].strip() or len(chunks) < 3:
            raise ValueError("chat_role_parse")
        turns = list(zip(chunks[1::2], chunks[2::2]))
    if not tools:
        raise ValueError("no_tool_schema")
    tools = [t if t.get("type") == "function" else {"type": "function", "function": t} for t in tools]
    schemas = {}
    for tool in tools:
        f = tool["function"]
        if not isinstance(f.get("name"), str) or not f["name"] or f["name"] in schemas:
            raise ValueError("invalid_or_duplicate_tool_name")
        schema = f["parameters"]
        Draft202012Validator.check_schema(schema)
        if '"$ref"' in canonical(schema) and "http" in canonical(schema):
            raise ValueError("external_schema_reference")
        schemas[f["name"]] = Draft202012Validator(schema)
    call_count, call_turns, parallel = 0, 0, False
    for role, content in turns:
        content = content.replace("<|endoftext|>", "").strip()
        if role in ["human", "USER"]:
            if pending:
                raise ValueError("missing_tool_response")
            if not content:
                raise ValueError("empty_user")
            messages.append({"role": "user", "content": content})
        elif role in ["gpt", "ASSISTANT"]:
            if pending:
                raise ValueError("missing_tool_response")
            raw_calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.S)
            rest = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.S).strip()
            if "<functioncall>" in content:
                before, after = content.split("<functioncall>", 1)
                raw_calls, rest = [after.strip()], before.strip()
            if ("<tool_call>" in rest) or ("</tool_call>" in rest):
                raise ValueError("incomplete_tool_call")
            calls = []
            for index, raw in enumerate(raw_calls):
                call = parse_call(raw, flags)
                name = call.get("name")
                if name not in schemas:
                    raise ValueError("unknown_tool")
                errors = list(schemas[name].iter_errors(call["arguments"]))
                if errors:
                    raise ValueError("argument_schema")
                call_id = f"call_{len(messages)}_{index}"
                calls.append({"id": call_id, "type": "function", "function": call})
                pending.append((call_id, name))
            message = {"role": "assistant", "content": rest}
            if calls:
                message["tool_calls"] = calls
                call_turns += 1
                call_count += len(calls)
                parallel |= len(calls) > 1
            elif not rest:
                raise ValueError("empty_assistant")
            messages.append(message)
        elif role in ["tool", "FUNCTION RESPONSE"]:
            if role == "tool":
                responses = re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", content, re.S)
                if not responses:
                    raise ValueError("tool_response_parse")
                for raw in responses:
                    response = json.loads(raw)
                    name = response["name"]
                    matches = [x for x in pending if x[1] == name]
                    if not matches:
                        raise ValueError("unmatched_tool_response")
                    call_id, name = matches[0]
                    pending.remove(matches[0])
                    messages.append({"role": "tool", "name": name, "tool_call_id": call_id,
                                     "content": canonical(response.get("content"))})
            else:
                if not pending:
                    raise ValueError("unexpected_tool_response")
                call_id, name = pending.pop(0)
                response = json.loads(content)
                messages.append({"role": "tool", "name": name, "tool_call_id": call_id, "content": canonical(response)})
        else:
            raise ValueError("unknown_role")
    # 单轮数据可以结束于完整调用；已有返回的多轮记录必须完整，不能丢一半。
    if pending and not messages[-1].get("tool_calls"):
        raise ValueError("missing_tool_response")
    if len(messages) < 3 or messages[1]["role"] != "user" or messages[-1]["role"] != "assistant":
        raise ValueError("incomplete_conversation")
    combined = []
    for message in messages:
        if combined and message["role"] == combined[-1]["role"] == "assistant":
            previous = combined[-1]
            previous["content"] = "\n".join(x for x in [previous["content"], message["content"]] if x)
            if message.get("tool_calls"):
                previous.setdefault("tool_calls", []).extend(message["tool_calls"])
            flags.append("merged_consecutive_assistant")
        else:
            combined.append(message)
    messages = combined
    category = "parallel" if parallel else "multi_turn" if call_turns > 1 else "single_call" if call_count else "no_call"
    return {"sample_id": f"{source}-{source_id}", "source": source, "source_id": str(source_id),
            "source_revision": metadata["revision"], "license": metadata["license"], "messages": messages,
            "tools": tools, "category": category, "validation": {"format": True, "arguments": True,
            "execution": "not_executed_public_annotations", "normalization": sorted(set(flags))},
            "teacher_run_id": None, "target_token_count": None, "rejection_reason": None}


def prompt_template(record):
    query = record["messages"][1]["content"].lower()
    query = re.sub(r'"[^"\n]+"|\x27[^\x27\n]+\x27', " <quoted> ", query)
    query = re.sub(r"\b\d+(?:[./:-]\d+)*\b", " <number> ", query)
    query = re.sub(r"\s+", " ", query).strip()
    signature = []
    for t in record["tools"]:
        f = t["function"]
        signature.append([f["name"], sorted(f["parameters"].get("properties", {})), sorted(f["parameters"].get("required", []))])
    return hash_value(sorted(signature)), query


def save_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data-cleaning.json")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8-sig"))
    sources = json.loads((ROOT / "configs/data-sources.json").read_text(encoding="utf-8"))
    run = start_run("E02", {**config, "command": [sys.executable, *sys.argv], "sources": sources}, args.run_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(ROOT / ".local/models/Qwen3-1.7B", local_files_only=True)
        counts, rejected, duplicates, seen, groups = Counter(), Counter(), Counter(), set(), defaultdict(list)
        buckets = defaultdict(list)
        all_lengths, reasons, normalization = [], [], Counter()
        accepted = []
        inputs = [("hermes", path) for path in sorted((ROOT / ".local/data/raw/hermes").glob("*.json"))
                  if path.name not in ["source-metadata.json", "download-manifest.json"]]
        inputs += [("glaive", ROOT / ".local/data/raw/glaive/glaive-function-calling-v2.json")]
        files = []
        for source, path in inputs:
            rows = json.loads(path.read_text(encoding="utf-8"))
            files.append({"source": source, "file": path.name, "rows": len(rows), "sha256": sha256(path)})
            for index, row in enumerate(rows):
                counts[source] += 1
                source_id = row.get("id", index)
                try:
                    record = convert(row, source, source_id, sources[source])
                    record["source_file"] = path.name
                    digest = hash_value([record["messages"], record["tools"]])
                    if digest in seen:
                        duplicates["exact_normalized"] += 1
                        raise ValueError("duplicate_exact")
                    seen.add(digest)
                    text = tokenizer.apply_chat_template(record["messages"], tools=record["tools"], tokenize=False,
                                                         add_generation_prompt=False, enable_thinking=False)
                    length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                    record["token_count"] = length
                    all_lengths.append(length)
                    if length > config["evaluation_context"]:
                        raise ValueError("over_evaluation_context")
                    signature, query = prompt_template(record)
                    bucket = (signature, query[:config["near_duplicate_prefix"]])
                    group_id = None
                    for previous, existing in buckets[bucket]:
                        if query == previous:
                            duplicates["same_prompt_template"] += 1
                            group_id = existing
                            break
                        if SequenceMatcher(None, query, previous, autojunk=False).ratio() >= config["near_duplicate_threshold"]:
                            duplicates["near_prompt_template"] += 1
                            group_id = existing
                            break
                    if group_id is None:
                        group_id = hash_value([signature, query])
                        buckets[bucket].append((query, group_id))
                    record["group_id"] = group_id
                    normalization.update(record["validation"]["normalization"])
                    accepted.append(record)
                    groups[group_id].append(record)
                except Exception as error:
                    reason = str(error) if isinstance(error, ValueError) and len(str(error)) < 80 else type(error).__name__
                    rejected[reason] += 1
                    reasons.append({"source": source, "source_file": path.name, "source_id": str(source_id), "reason": reason})
                if sum(counts.values()) % 10000 == 0:
                    print(canonical({"at": now(), "read": sum(counts.values()), "accepted": len(accepted), "groups": len(groups), "rejected": sum(rejected.values())}), flush=True)
        ordered = sorted(groups, key=lambda g: hash_value([config["split_seed"], g]))
        borders = [int(len(ordered) * 0.8), int(len(ordered) * 0.9)]
        assignment = {g: "train" if i < borders[0] else "dev" if i < borders[1] else "test" for i, g in enumerate(ordered)}
        selected = defaultdict(list)
        train_long = 0
        for group in ordered:
            split = assignment[group]
            candidates = [r for r in groups[group] if split != "train" or r["token_count"] <= config["training_context"]]
            if not candidates:
                train_long += 1
                continue
            # 每组只选一条完整记录，避免把任务变体算成独立样本。
            record = sorted(candidates, key=lambda r: hash_value([config["split_seed"], r["sample_id"]]))[0]
            record["split"] = split
            selected[split].append(record)
        needed = {"train": config["training_sizes"][-1], "dev": config["dev_size"], "test": config["test_size"]}
        for split, size in needed.items():
            if len(selected[split]) < size:
                raise RuntimeError(f"独立有效 {split} 样本不足：{len(selected[split])} < {size}")
        output = ROOT / ".local/data/processed" / run.name
        output.mkdir(parents=True, exist_ok=False)
        outputs = {}
        for size in config["training_sizes"]:
            path = output / f"train-{size}.jsonl"
            save_jsonl(path, selected["train"][:size])
            outputs[path.name] = {"samples": size, "sha256": sha256(path)}
        for split in ["dev", "test"]:
            path = output / f"{split}.jsonl"
            save_jsonl(path, selected[split][:needed[split]])
            outputs[path.name] = {"samples": needed[split], "sha256": sha256(path)}
        save_jsonl(run / "rejected.jsonl", reasons)
        save_jsonl(run / "normalized-pool.jsonl", accepted)
        pool_manifest = [{"sample_id": r["sample_id"], "group_id": r["group_id"], "assigned_split": assignment[r["group_id"]],
                          "source": r["source"], "source_id": r["source_id"], "tokens": r["token_count"]} for r in accepted]
        save_jsonl(run / "pool-manifest.jsonl", pool_manifest)
        directory = ROOT / "experiments/E02"
        directory.mkdir(parents=True, exist_ok=True)
        frozen = [r for split in needed for r in selected[split][:needed[split]]]
        with (directory / "split-manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["sample_id", "source", "source_id", "source_file", "source_revision", "group_id", "split", "token_count", "category"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(frozen)
        (run / "split-manifest.csv").write_bytes((directory / "split-manifest.csv").read_bytes())
        randomizer = random.Random(config["inspection_seed"])
        review = randomizer.sample(frozen, config["inspection_samples"])
        save_jsonl(run / "inspection-50.jsonl", review)
        with (directory / "inspection-50.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "category", "split", "tokens", "first_user", "calls", "format", "schema", "semantic_review"])
            for r in review:
                calls = [m["tool_calls"] for m in r["messages"] if m.get("tool_calls")]
                writer.writerow([r["sample_id"], r["category"], r["split"], r["token_count"], r["messages"][1]["content"], canonical(calls), True, True, "pending"])
        categories = {split: dict(Counter(r["category"] for r in selected[split][:needed[split]])) for split in needed}
        length_hist = Counter((n // 256) * 256 for n in all_lengths)
        statistics = {"raw_sources": dict(counts), "files": files, "rejected": dict(rejected),
                   "duplicates": dict(duplicates), "accepted_records": len(accepted), "independent_groups": len(groups),
                   "normalization": dict(normalization), "train_groups_over_2048": train_long,
                   "available_groups": {s: len(selected[s]) for s in needed}, "selected_categories": categories,
                   "length_histogram_256": dict(sorted(length_hist.items())), "inspection_status": "pending_semantic_review"}
        write_json(directory / "statistics.json", statistics)
        write_json(run / "statistics.json", statistics)
        (run / "inspection-50.csv").write_bytes((directory / "inspection-50.csv").read_bytes())
        sets = {s: {r["group_id"] for r in selected[s][:needed[s]]} for s in needed}
        intersections = {f"{a}_{b}": len(sets[a] & sets[b]) for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]}
        assert not any(intersections.values()), "任务组发生跨划分泄漏"
        result = finish_run(run, {"status": "awaiting_sample_review", "exit_code": 0, "sources": sources,
                "raw_rows": sum(counts.values()), "accepted_records": len(accepted), "independent_groups": len(groups),
                "outputs": outputs, "split_manifest_sha256": sha256(directory / "split-manifest.csv"),
                "statistics_sha256": sha256(directory / "statistics.json"), "inspection_samples": len(review),
                "group_intersections": intersections})
        print(canonical(result), flush=True)
    except Exception:
        (run / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finish_run(run, {"status": "failed", "exit_code": 1, "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
