"""E06：32 条完整轨迹的过拟合检查，保存可恢复的真实训练状态。"""
import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
import time
import traceback

import torch

from lab import ROOT, finish_run, now, sha256, start_run, write_json
from model_utils import batch, load_model
from templates import action_records, encode_action, encode_record, read_records, resolve_data_run, tokenizer_and_template


def evaluate(model, tokenizer, records, path, max_new_tokens):
    model.eval()
    results = []
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            turns = []
            for index, message in enumerate(record["messages"]):
                if not message.get("tool_calls"):
                    continue
                inputs = tokenizer.apply_chat_template(record["messages"][:index], tools=record["tools"],
                          tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True)
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                            temperature=None, top_p=None, top_k=None, use_cache=True)
                tokens = output[0, inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(tokens, skip_special_tokens=True)
                expected = [c["function"] for c in message["tool_calls"]]
                parsed, error = [], None
                try:
                    parsed = [json.loads(x) for x in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.S)]
                    if not parsed:
                        error = "no_complete_tool_call"
                except Exception as failure:
                    error = type(failure).__name__
                exact = not error and parsed == expected
                turns.append({"message_index": index, "expected": expected, "parsed": parsed, "response": response,
                              "generated_tokens": len(tokens), "truncated": len(tokens) >= max_new_tokens,
                              "exact": bool(exact), "error": error})
            result = {"sample_id": record["sample_id"], "category": record["category"], "turns": turns,
                      "passed": bool(turns) and all(t["exact"] and not t["truncated"] for t in turns)}
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n"); handle.flush()
            print(f"评估 {len(results)}/{len(records)}，通过 {sum(r['passed'] for r in results)}", flush=True)
    model.train()
    return {"samples": len(results), "passed": sum(r["passed"] for r in results),
            "tool_turns": sum(len(r["turns"]) for r in results), "result_sha256": sha256(path)}


def save_checkpoint(model, tokenizer, optimizer, step, order, cursor, rng, directory):
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    torch.save({"optimizer": optimizer.state_dict(), "step": step, "order": order, "cursor": cursor,
                "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
                directory / "training-state.pt")
    write_json(directory / "checkpoint.json", {"step": step, "saved": now(),
        "files": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in directory.iterdir() if p.is_file()]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny-train.json")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    config["data_run"] = resolve_data_run(config["data_run"])
    config.setdefault("unit_policy", "whole_conversation")
    run = start_run("E06", {**config, "command": [sys.executable, *sys.argv]}, args.run_dir)
    try:
        torch.manual_seed(config["seed"])
        rng = random.Random(config["seed"])
        tokenizer, _, marked = tokenizer_and_template()
        pool = read_records(data_run=config["data_run"])
        records = []
        for category, count in config["category_samples"].items():
            eligible = [r for r in pool if r["category"] == category and r["messages"][2].get("tool_calls")]
            records.extend(rng.sample(eligible, count))
        assert len(records) == 32 and len({r["group_id"] for r in records}) == 32
        write_json(run / "samples.json", records)
        actions = [action for record in records for action in action_records(record)]
        data = [encode_action(tokenizer, marked, action) for action in actions] if config["unit_policy"] == "assistant_action" else [encode_record(tokenizer, marked, r) for r in records]
        write_json(run / "training-units.json", {"policy": config["unit_policy"], "independent_trajectories": len(records),
            "assistant_units": len(data), "sample_indices": [{"sample_id": a["sample_id"], "target_message_index": a["target_message_index"]} for a in actions]})
        assert all(len(r["input_ids"]) <= config["max_sequence_length"] for r in data)
        model = load_model(adapter=True, config=config)
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"], weight_decay=0)
        baseline = evaluate(model, tokenizer, records, run / "eval-step-000000.jsonl", config["max_new_tokens"])
        write_json(run / "baseline.json", baseline)
        order, cursor = list(range(len(data))), 0
        rng.shuffle(order)
        metrics = run / "metrics.csv"
        last_report = time.monotonic()
        final = baseline
        with metrics.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "time", "train_loss", "learning_rate", "step_seconds", "supervised_tokens", "peak_allocated_mib", "train_passed", "train_samples"])
            writer.writeheader(); handle.flush()
            for step in range(1, config["max_steps"] + 1):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                losses, tokens = [], 0
                for _ in range(config["gradient_accumulation"]):
                    if cursor == len(order):
                        rng.shuffle(order); cursor = 0
                    encoded = data[order[cursor]]; cursor += 1
                    inputs = batch([encoded], tokenizer.pad_token_id)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(**inputs).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("训练 loss 出现非有限值")
                    (loss / config["gradient_accumulation"]).backward()
                    losses.append(loss.item()); tokens += sum(x != -100 for x in encoded["labels"][1:])
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step(); torch.cuda.synchronize()
                row = {"step": step, "time": now(), "train_loss": sum(losses) / len(losses), "learning_rate": config["learning_rate"],
                       "step_seconds": time.perf_counter() - started, "supervised_tokens": tokens,
                       "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2, "train_passed": "", "train_samples": 32}
                if step % config["evaluate_every"] == 0 or step == config["max_steps"]:
                    final = evaluate(model, tokenizer, records, run / f"eval-step-{step:06d}.jsonl", config["max_new_tokens"])
                    row["train_passed"] = final["passed"]
                    save_checkpoint(model, tokenizer, optimizer, step, order, cursor, rng, run / f"checkpoint-{step:06d}")
                writer.writerow(row); handle.flush()
                write_json(run / "progress.json", {"status": "running", "latest": row, "baseline": baseline, "latest_eval": final})
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if row["train_passed"] != "" or time.monotonic() - last_report >= 600:
                    subprocess.run([sys.executable, "scripts/notes_training.py", "--run", run.name], cwd=ROOT, check=True)
                    subprocess.run([sys.executable, "scripts/report.py", "--volume", "01"], cwd=ROOT, check=True)
                    last_report = time.monotonic()
                if step >= config["min_steps"] and row["train_passed"] != "" and final["passed"] >= config["required_passed"]:
                    break
        result = {"status": "completed" if final["passed"] >= config["required_passed"] else "criterion_not_met", "exit_code": 0,
                  "steps": step, "baseline": baseline, "final": final, "criterion": "至少29/32条完整轨迹的全部标注调用通过工具名与参数严格检查",
                  "samples_sha256": sha256(run / "samples.json"), "metrics_sha256": sha256(metrics),
                  "data_run": config["data_run"], "resume_checkpoint": f"checkpoint-{step:06d}",
                  "scope": "训练样本过拟合检查；评估逐轮使用真实标注历史，不是真实 Agent 执行或泛化成绩"}
        finish_run(run, result)
        subprocess.run([sys.executable, "scripts/notes_training.py", "--run", run.name], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "scripts/report.py", "--volume", "01"], cwd=ROOT, check=True)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception:
        (run / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finish_run(run, {"status": "failed", "exit_code": 1, "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
