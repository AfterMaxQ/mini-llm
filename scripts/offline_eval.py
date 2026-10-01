"""完整公开工具划分的逐轮评估：保留错误，按原始轨迹统计。"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import traceback
from collections import Counter

import torch
from jsonschema import Draft202012Validator
from peft import PeftModel, prepare_model_for_kbit_training

from lab import ROOT, finish_run, now, sha256, start_run, write_json
from model_utils import load_model
from templates import read_records, resolve_data_run, tokenizer_and_template


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def prompt_candidates():
    train = read_records("train-1000.jsonl")
    examples = []
    for category in ["single_call", "parallel", "no_call"]:
        record = next(r for r in train if r["category"] == category)
        message = record["messages"][2]
        examples.append({"sample_id": record["sample_id"], "tools": record["tools"],
                         "user": record["messages"][1]["content"], "reply": message})
    instruction = ("\nThe following three training examples illustrate when to call tools and when to respond without a call. "
                   "The tools inside examples are not available for the current task. Use only the functions in the current tool list. "
                   "Write each call as <tool_call>{\"name\":\"function_name\",\"arguments\":{...}}</tool_call>. "
                   "Do not invent missing required information.\n" + canonical(examples))
    return {"zero_shot": "", "few_shot": instruction}, examples


def decision_turns(records):
    result = []
    for record in records:
        for index, message in enumerate(record["messages"]):
            if message["role"] == "assistant" and (message.get("tool_calls") or record["messages"][index-1]["role"] == "user"):
                result.append({"record": record, "message_index": index,
                               "expected": [c["function"] for c in message.get("tool_calls", [])]})
    return result


def judge(response, expected, tools, truncated=False, runtime_error=None):
    calls, error = [], runtime_error
    if not response.strip():
        error = error or 'empty_response'
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.S)
    if response.count("<tool_call>") != len(blocks) or response.count("</tool_call>") != len(blocks):
        error = error or "incomplete_tool_call"
    try:
        calls = [json.loads(block) for block in blocks]
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
                raise ValueError("invalid_call_object")
            if call["name"] not in schemas:
                raise ValueError("unknown_tool")
            Draft202012Validator(schemas[call["name"]]).validate(call["arguments"])
    except Exception as failure:
        error = error or type(failure).__name__ + ":" + str(failure).splitlines()[0][:150]
    if expected and not calls and not error:
        error = "no_tool_call"
    valid = not error and not truncated
    actual_names = Counter(c.get("name") for c in calls if isinstance(c, dict))
    expected_names = Counter(c["name"] for c in expected)
    return {"parsed": calls, "error": error, "truncated": truncated,
            "name_exact": bool(valid and actual_names == expected_names),
            "exact": bool(valid and sorted(map(canonical, calls)) == sorted(map(canonical, expected))),
            "expected_call": bool(expected), "predicted_call": bool(calls),
            "format_valid": bool(not error or error == "no_tool_call")}


def summarize(rows, records):
    by_id = {r["sample_id"]: [] for r in records}
    for row in rows:
        by_id[row["sample_id"]].append(row)
    complete = {r["sample_id"]: bool(by_id[r["sample_id"]]) and all(x["exact"] for x in by_id[r["sample_id"]]) for r in records}
    call = [r for r in rows if r["expected_call"]]
    no_call = [r for r in rows if not r["expected_call"]]
    return {"trajectories": len(records), "evaluated_trajectories": sum(bool(v) for v in by_id.values()),
            "trajectory_passed": sum(complete.values()), "decision_turns": len(rows),
            "turn_passed": sum(r["exact"] for r in rows), "call_turns": len(call),
            "call_turn_passed": sum(r["exact"] for r in call), "no_call_turns": len(no_call),
            "no_call_turn_passed": sum(r["exact"] for r in no_call),
            "name_exact_turns": sum(r["name_exact"] for r in call),
            "truncated": sum(r["truncated"] for r in rows), "errors": dict(Counter(r["error"] for r in rows if r["error"])),
            "categories": {category: {"samples": sum(r["category"] == category for r in records),
                                      "passed": sum(complete[r["sample_id"]] for r in records if r["category"] == category)}
                           for category in sorted({r["category"] for r in records})}}


def evaluate(model, tokenizer, records, config, suffix, path, progress=None):
    turns = decision_turns(records)
    rows = []
    # 接续只读取已落盘的完整批次，不用已有汇总代替逐条结果。
    if path.exists():
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert all((r["sample_id"], r["message_index"]) == (t["record"]["sample_id"], t["message_index"])
                   for r, t in zip(rows, turns)) and len(rows) <= len(turns)
    tokenizer.padding_side = "left"
    model.eval()
    with path.open("a", encoding="utf-8") as handle:
        for offset in range(len(rows), len(turns), config["batch_size"]):
            group = turns[offset:offset + config["batch_size"]]
            prompts = []
            for turn in group:
                record = turn["record"]
                messages = [dict(m) for m in record["messages"][:turn["message_index"]]]
                messages[0]["content"] += suffix
                prompts.append(tokenizer.apply_chat_template(messages, tools=record["tools"], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False))
            tokenized = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
            lengths = tokenized["attention_mask"].sum(dim=1).tolist()
            eligible = [i for i, n in enumerate(lengths) if n + config["max_new_tokens"] <= config["context_length"]]
            outputs = {}
            if eligible:
                # 排除的长度超限条目仍写入失败结果，不进入生成批次。
                inputs = tokenizer([prompts[i] for i in eligible], padding=True, return_tensors="pt", add_special_tokens=False)
                inputs = {k: v.to("cuda") for k,v in inputs.items()}
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
                try:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        generated = model.generate(**inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                              max_new_tokens=config["max_new_tokens"], max_time=config["max_batch_seconds"], use_cache=True,
                              pad_token_id=tokenizer.pad_token_id)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-started
                    eos = model.generation_config.eos_token_id
                    eos = set(eos if isinstance(eos,list) else [eos])
                    for position, index in enumerate(eligible):
                        tokens = generated[position,inputs["input_ids"].shape[1]:].tolist()
                        endings = [j for j, value in enumerate(tokens) if value in eos]
                        if endings: tokens = tokens[:endings[0]+1]
                        outputs[index] = {"response":tokenizer.decode(tokens,skip_special_tokens=True),
                            "generated_tokens":len(tokens), "truncated":not bool(endings),
                            "runtime_error": "batch_timeout" if not endings and elapsed >= config["max_batch_seconds"] else None,
                            "batch_seconds":elapsed,"peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2}
                except torch.cuda.OutOfMemoryError:
                    for index in eligible: outputs[index]={"response":"","generated_tokens":0,"truncated":False,"runtime_error":"cuda_oom"}
                    torch.cuda.empty_cache()
            for index, turn in enumerate(group):
                output = outputs.get(index,{"response":"","generated_tokens":0,"truncated":False,"runtime_error":"context_overflow"})
                record=turn["record"]
                row={"sample_id":record["sample_id"],"category":record["category"],"message_index":turn["message_index"],
                     "expected":turn["expected"],"prompt_tokens":lengths[index],**output,
                     **judge(output["response"],turn["expected"],record["tools"],output["truncated"],output.get("runtime_error"))}
                rows.append(row);handle.write(json.dumps(row,ensure_ascii=False)+"\n")
            handle.flush()
            summary=summarize(rows,records)
            print(json.dumps({"time":now(),"evaluated_turns":len(rows),"total_turns":len(turns),
                              "trajectory_passed_so_far":summary["trajectory_passed"]},ensure_ascii=False),flush=True)
            if progress: progress(summary)
    assert len(rows)==len(turns)
    result=summarize(rows,records)
    result["rows_sha256"]=sha256(path)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/offline-eval.json")
    parser.add_argument("--adapter")
    parser.add_argument("--source-run")
    parser.add_argument("--experiment",default="E08")
    parser.add_argument("--resume-run")
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding="utf-8"))
    config["data_run"]=resolve_data_run(config["data_run"])
    config["adapter"]=args.adapter
    config["source_train_run"]=args.source_run
    config["kind"]="tool_eval" if args.adapter else "prompt_baseline"
    if args.adapter:
        frozen=json.loads((ROOT/"configs/prompt-frozen.json").read_text(encoding="utf-8"))
        config["prompts"]=[frozen["selected_prompt"]]
        config["frozen_prompt_sha256"]=sha256(ROOT/"configs/prompt-frozen.json")
        config["adapter_sha256"]=sha256(ROOT/args.adapter/"adapter_model.safetensors")
    config["command"]=[sys.executable,*sys.argv]
    run=ROOT/".local/runs"/args.resume_run if args.resume_run else start_run(args.experiment,config)
    if args.resume_run:
        saved=json.loads((run/"config.json").read_text(encoding="utf-8"))
        assert all(config[k]==saved[k] for k in config if k!="command"),"续跑配置发生变化，需另开运行"
    try:
        torch.manual_seed(config["seed"])
        records=read_records(config["split"]+".jsonl",config["data_run"])
        tokenizer,_,_=tokenizer_and_template()
        if args.adapter:
            prompts={frozen['selected_prompt']:frozen['suffix']};examples=[]
        else:
            prompts,examples=prompt_candidates()
        if args.adapter:prompts[frozen["selected_prompt"]]=frozen["suffix"]
        write_json(run/"prompt-candidates.json",{"prompts":prompts,"examples":examples})
        model=prepare_model_for_kbit_training(load_model(),use_gradient_checkpointing=False)
        if args.adapter:model=PeftModel.from_pretrained(model,ROOT/args.adapter)
        summaries={};last_report=0
        def progress(summary):
            nonlocal last_report
            write_json(run/"progress.json",{"status":"running","current_prompt":name,"summary":summary,"completed_prompts":summaries})
            if time.monotonic()-last_report>=600:
                script={'E10':'scripts/notes_precision.py','E11':'scripts/notes_tuning.py','E12':'scripts/notes_quality.py'}.get(args.experiment,'scripts/notes_sft.py')
                note_command=[sys.executable,"scripts/notes_baseline.py","--run",run.name] if args.experiment=="E08" else [sys.executable,script]
                subprocess.run(note_command,cwd=ROOT,check=True)
                subprocess.run([sys.executable,"scripts/report.py","--volume","02"],cwd=ROOT,check=True)
                last_report=time.monotonic()
        for name in config["prompts"]:
            summaries[name]=evaluate(model,tokenizer,records,config,prompts[name],run/f"{name}.jsonl",progress)
            write_json(run/"summaries.json",summaries)
        # 平局使用较短的零样本提示；只有 dev 决定选择。
        selected=max(config["prompts"],key=lambda x:summaries[x]["trajectory_passed"])
        result=finish_run(run,{"status":"completed","exit_code":0,"summaries":summaries,"selected_prompt":selected,
                      "prompt_candidates_sha256":sha256(run/"prompt-candidates.json"),
                      "scope":"完整冻结划分的工具决策；使用标注历史，不代表工具执行或最终任务状态"})
        if args.experiment=="E08" and config["split"]=="dev" and not args.adapter:
            write_json(ROOT/"configs/prompt-frozen.json",{"source_run":run.name,"selection_split":"dev","selected_prompt":selected,
                      "suffix":prompts[selected],"source_sha256":result["prompt_candidates_sha256"],"tie_break":"zero_shot"})
        script={'E10':'scripts/notes_precision.py','E11':'scripts/notes_tuning.py','E12':'scripts/notes_quality.py'}.get(args.experiment,'scripts/notes_sft.py')
        note_command=[sys.executable,"scripts/notes_baseline.py","--run",run.name] if args.experiment=="E08" else [sys.executable,script]
        subprocess.run(note_command,cwd=ROOT,check=True)
        subprocess.run([sys.executable,"scripts/report.py","--volume","02"],cwd=ROOT,check=True)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception:
        (run/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        finish_run(run,{"status":"failed","exit_code":1,"error":traceback.format_exc()})
        raise


if __name__=="__main__":main()
