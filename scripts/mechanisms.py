"""E04 的交叉熵核对、E05 的冻结参数和 LoRA 实际更新。"""
import argparse
import hashlib
import json
import sys
import traceback

import torch
import torch.nn.functional as F

from lab import ROOT, finish_run, start_run
from model_utils import batch, load_model
from templates import action_records, encode_action, encode_record, read_records, resolve_data_run, tokenizer_and_template


def fingerprint(model, trainable):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == trainable:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["E04", "E05"], required=True)
    parser.add_argument("--config", default="configs/mechanisms.json")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    config["data_run"] = resolve_data_run(config["data_run"])
    run = start_run(args.experiment, {**config, "command": [sys.executable, *sys.argv]}, args.run_dir)
    try:
        torch.manual_seed(config["seed"])
        tokenizer, _, marked = tokenizer_and_template()
        records = read_records(data_run=config["data_run"])[:2]
        encoded = [encode_action(tokenizer, marked, action_records(r)[-1]) for r in records] if config.get("unit_policy") == "assistant_action" else [encode_record(tokenizer, marked, r) for r in records]
        inputs = batch(encoded, tokenizer.pad_token_id)
        torch.cuda.reset_peak_memory_stats()
        model = load_model(adapter=args.experiment == "E05", config=config)
        if args.experiment == "E04":
            model.eval()
            with torch.no_grad():
                output = model(**inputs)
                logits = output.logits[:, :-1].float()
                labels = inputs["labels"][:, 1:]
                manual = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
                full_labels = inputs["input_ids"].clone()
                full_labels[inputs["attention_mask"] == 0] = -100
                full = model(**{**inputs, "labels": full_labels})
                manual_full = F.cross_entropy(full.logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                             full_labels[:, 1:].reshape(-1), ignore_index=-100)
            differences = {"assistant": abs(manual.item() - output.loss.item()), "full": abs(manual_full.item() - full.loss.item())}
            assert max(differences.values()) < 1e-5
            assert (inputs["labels"][inputs["attention_mask"] == 0] == -100).all()
            result = {"status": "completed", "exit_code": 0, "sample_ids": [r["sample_id"] for r in records],
                      "input_shape": list(inputs["input_ids"].shape), "assistant_tokens": int((labels != -100).sum()),
                      "full_tokens": int((full_labels[:, 1:] != -100).sum()), "padding_tokens": int((inputs["attention_mask"] == 0).sum()),
                      "assistant_framework_loss": output.loss.item(), "assistant_manual_loss": manual.item(),
                      "full_framework_loss": full.loss.item(), "full_manual_loss": manual_full.item(), "absolute_differences": differences}
        else:
            model.train()
            trainable = [p for p in model.parameters() if p.requires_grad]
            assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
            frozen_before = fingerprint(model, False)
            adapter_before = fingerprint(model, True)
            optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"], weight_decay=0)
            steps = []
            for step in range(1, 3):
                optimizer.zero_grad(set_to_none=True)
                loss = model(**inputs).loss
                assert torch.isfinite(loss)
                loss.backward()
                gradients = {"A": [], "B": []}
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        assert p.grad is not None and torch.isfinite(p.grad).all()
                        gradients["A" if "lora_A" in name else "B"].append(p.grad.float().norm().item())
                steps.append({"step": step, "loss": loss.item(), **{k: {"tensors": len(v), "nonzero_tensors": sum(x > 0 for x in v),
                                  "norm_sum": sum(v), "norm_max": max(v)} for k, v in gradients.items()}})
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
            frozen_after = fingerprint(model, False)
            adapter_after = fingerprint(model, True)
            assert frozen_before == frozen_after and adapter_before != adapter_after
            assert steps[0]["A"]["nonzero_tensors"] == 0 and steps[0]["B"]["nonzero_tensors"] > 0
            assert steps[1]["A"]["nonzero_tensors"] > 0
            result = {"status": "completed", "exit_code": 0, "sample_ids": [r["sample_id"] for r in records],
                      "trainable_parameters": sum(p.numel() for p in trainable), "steps": steps,
                      "frozen_before_sha256": frozen_before, "frozen_after_sha256": frozen_after,
                      "adapter_before_sha256": adapter_before, "adapter_after_sha256": adapter_after,
                      "weight_decay": 0, "explanation": "默认 B 初始化为零，第一步 A 梯度为零；B 更新后第二步 A 获得非零梯度"}
        torch.cuda.synchronize()
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        finish_run(run, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        (run / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finish_run(run, {"status": "failed", "exit_code": 1, "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
