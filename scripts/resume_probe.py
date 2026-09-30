"""E07：加载真实 checkpoint，核对输出，并恢复优化器和随机状态继续一步。"""
import argparse
import gc
import json
import random
import re
import sys
import traceback

import torch
from peft import PeftModel, prepare_model_for_kbit_training

from lab import ROOT, finish_run, sha256, start_run, write_json
from mechanisms import fingerprint
from model_utils import batch, load_model
from templates import action_records, encode_action, tokenizer_and_template
from tiny_train import save_checkpoint


def load_adapter(checkpoint):
    base = load_model()
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    return PeftModel.from_pretrained(base, checkpoint, is_trainable=True)


def continue_step(model, optimizer, state, data, pad_id, accumulation):
    rng = random.Random(); rng.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
    order, cursor = list(state["order"]), state["cursor"]
    model.train(); optimizer.zero_grad(set_to_none=True)
    losses, used = [], []
    for _ in range(accumulation):
        if cursor == len(order): rng.shuffle(order); cursor = 0
        index = order[cursor]; cursor += 1; used.append(index)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch([data[index]], pad_id)).loss
        (loss / accumulation).backward(); losses.append(loss.item())
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step(); torch.cuda.synchronize()
    return {"step": state["step"] + 1, "loss": sum(losses) / len(losses), "unit_indices": used,
            "order": order, "cursor": cursor, "rng": rng}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-run")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    if args.from_run is None:
        candidates = [p.parent for p in sorted((ROOT / ".local/runs").glob("E06-R*/result.json"))
                      if json.loads(p.read_text(encoding="utf-8")).get("status") == "completed"]
        if not candidates:
            raise FileNotFoundError("先完成本机 32 条过拟合实验，再恢复其 checkpoint")
        args.from_run = candidates[-1].name
    source = ROOT / ".local/runs" / args.from_run
    result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed", "32 条实验尚未达到门槛，不开始恢复验收"
    checkpoint = source / result["resume_checkpoint"]
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    run = start_run("E07", {"source_run": args.from_run, "checkpoint": checkpoint.name,
                    "checkpoint_manifest_sha256": sha256(checkpoint / "checkpoint.json"), "command": [sys.executable, *sys.argv]}, args.run_dir)
    try:
        tokenizer, _, marked = tokenizer_and_template()
        records = json.loads((source / "samples.json").read_text(encoding="utf-8"))
        actions = [action for record in records for action in action_records(record)]
        data = [encode_action(tokenizer, marked, action) for action in actions]
        previous = [json.loads(x) for x in (source / f"eval-step-{result['steps']:06d}.jsonl").read_text(encoding="utf-8").splitlines()]
        model = load_adapter(checkpoint)
        before = fingerprint(model, True)
        first = records[0]; index = previous[0]["turns"][0]["message_index"]
        inputs = tokenizer.apply_chat_template(first["messages"][:index], tools=first["tools"], tokenize=True,
                    enable_thinking=False, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.generate(**inputs, max_new_tokens=config["max_new_tokens"], do_sample=False,
                                    temperature=None, top_p=None, top_k=None, use_cache=True)
        text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        assert text == previous[0]["turns"][0]["response"], "加载后的首条调用与保存前不同"
        write_json(run / "output-comparison.json", {"sample_id": first["sample_id"], "saved_response": previous[0]["turns"][0]["response"],
                   "loaded_response": text, "identical": True})
        state = torch.load(checkpoint / "training-state.pt", map_location="cpu", weights_only=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"], weight_decay=0)
        optimizer.load_state_dict(state["optimizer"])
        expected = continue_step(model, optimizer, state, data, tokenizer.pad_token_id, config["gradient_accumulation"])
        expected_weights = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
        after = fingerprint(model, True)
        assert before != after
        save_checkpoint(model, tokenizer, optimizer, expected["step"], expected["order"], expected["cursor"], expected["rng"], run / "continued-checkpoint")
        del model, optimizer, output, inputs, state
        gc.collect(); torch.cuda.empty_cache()
        model = load_adapter(checkpoint)
        assert fingerprint(model, True) == before
        state = torch.load(checkpoint / "training-state.pt", map_location="cpu", weights_only=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"], weight_decay=0)
        optimizer.load_state_dict(state["optimizer"])
        actual = continue_step(model, optimizer, state, data, tokenizer.pad_token_id, config["gradient_accumulation"])
        differences = [(p.detach().cpu() - expected_weights[name]).abs().max().item() for name, p in model.named_parameters() if p.requires_grad]
        assert expected["unit_indices"] == actual["unit_indices"]
        assert max(differences) <= 1e-6 and abs(expected["loss"] - actual["loss"]) <= 1e-4
        finish_run(run, {"status": "completed", "exit_code": 0, "source_run": args.from_run,
              "saved_step": state["step"], "continued_step": actual["step"], "generation_identical": True,
              "adapter_before_sha256": before, "adapter_after_sha256": after, "expected_loss": expected["loss"],
              "replayed_loss": actual["loss"], "max_parameter_difference": max(differences),
              "next_unit_indices": actual["unit_indices"], "optimizer_restored": True, "random_states_restored": True})
        print(json.dumps({"status": "completed", "saved_step": state["step"], "continued_step": actual["step"], "max_difference": max(differences)}, ensure_ascii=False))
    except Exception:
        (run / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finish_run(run, {"status": "failed", "exit_code": 1, "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
