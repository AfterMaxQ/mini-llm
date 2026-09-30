"""E03：实际消息的渲染、token、EOS 与监督遮罩对应。"""
import argparse
import copy
import csv
import json
import random
import re
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from lab import ROOT, finish_run, sha256, start_run, write_json
from templates import action_records, encode_action, encode_record, read_records, resolve_data_run, tokenizer_and_template


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/template-probe.json")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    config["data_run"] = resolve_data_run(config["data_run"])
    run = start_run("E03", {**config, "command": [sys.executable, *sys.argv]}, args.run_dir)
    try:
        tokenizer, original, marked = tokenizer_and_template()
        (run / "original.jinja").write_text(original, encoding="utf-8")
        (run / "assistant-mask.jinja").write_text(marked, encoding="utf-8")
        records = read_records(data_run=config["data_run"])
        rng = random.Random(42)
        selected = []
        for category, size in [("single_call", 7), ("multi_turn", 4), ("parallel", 4), ("no_call", 3)]:
            selected.extend(rng.sample([r for r in records if r["category"] == category], size))
        selected.append(max(records, key=lambda r: r["token_count"]))
        empty = copy.deepcopy(next(r for r in records if any(m["role"] == "tool" for m in r["messages"])))
        empty["fixture_origin"] = empty["sample_id"]
        empty["sample_id"] += "-empty-tool-fixture"
        next(m for m in empty["messages"] if m["role"] == "tool")["content"] = "[]"
        empty["fixture_change"] = "只把第一条工具返回改为空列表，用于遮罩检查，不作为真实轨迹或能力评测样本"
        selected.append(empty)
        assert len(selected) == config["samples"]
        results, example = [], None
        for number, record in enumerate(selected):
            keywords = {"tools": record["tools"], "tokenize": False, "add_generation_prompt": False, "enable_thinking": False}
            rendered = tokenizer.apply_chat_template(record["messages"], chat_template=original, **keywords)
            patched = tokenizer.apply_chat_template(record["messages"], chat_template=marked, **keywords)
            assert rendered == patched, "增加遮罩标记改变了实际渲染文本"
            original_tokens = tokenizer.apply_chat_template(record["messages"], tools=record["tools"], chat_template=original,
                                                           enable_thinking=False, add_generation_prompt=False)
            encoded = encode_record(tokenizer, marked, record)
            assert original_tokens == encoded["input_ids"], "增加标记改变了 token"
            mask = encoded["assistant_masks"]
            assert len(mask) == len(original_tokens) and 0 < sum(mask) < len(mask)
            offsets = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            spans = [(m.start(), m.end(), m.group(1)) for m in re.finditer(r"<\|im_start\|>(system|user|assistant)\n.*?<\|im_end\|>\n", rendered, re.S)]
            roles = []
            for index, (start, end) in enumerate(offsets):
                role = next((role for begin, stop, role in spans if begin <= start < stop), "other")
                roles.append(role)
                assert not mask[index] or role == "assistant", "监督包含了用户、系统或工具返回"
            eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
            assistant_eos = [i for i, token in enumerate(original_tokens) if token == eos and roles[i] == "assistant"]
            assert assistant_eos and all(mask[i] for i in assistant_eos), "assistant EOS 未被监督"
            prefix = next(i for i, m in enumerate(record["messages"]) if m["role"] == "assistant")
            prompt = tokenizer.apply_chat_template(record["messages"][:prefix], tools=record["tools"], chat_template=original,
                            tokenize=False, enable_thinking=False, add_generation_prompt=True)
            patched_prompt = tokenizer.apply_chat_template(record["messages"][:prefix], tools=record["tools"], chat_template=marked,
                            tokenize=False, enable_thinking=False, add_generation_prompt=True)
            assert prompt == patched_prompt and prompt.endswith("<think>\n\n</think>\n\n"), "非思考部署提示不一致"
            action_checks = []
            for action in action_records(record):
                action_text = tokenizer.apply_chat_template(action["messages"], chat_template=original, **keywords)
                action_prompt = tokenizer.apply_chat_template(action["messages"][:-1], tools=action["tools"], chat_template=original,
                             tokenize=False, enable_thinking=False, add_generation_prompt=True)
                assert action_text.startswith(action_prompt), "当前回复的训练前缀与推理不一致"
                action_encoded = encode_action(tokenizer, marked, action)
                prompt_length = len(tokenizer(action_prompt, add_special_tokens=False)["input_ids"])
                # 当前回复的完整 assistant 区间可含角色前缀，旧轮次不能重复监督。
                starts = [i for i, value in enumerate(action_encoded["assistant_masks"]) if value and (i == 0 or not action_encoded["assistant_masks"][i - 1])]
                assert len(starts) == 1 and sum(action_encoded["assistant_masks"][prompt_length:]) > 0
                action_checks.append({"target_message_index": action["target_message_index"], "prefix_identical": True,
                                      "tokens": len(action_encoded["input_ids"]), "supervised_tokens": sum(action_encoded["assistant_masks"])})
            sample_dir = run / f"sample-{number + 1:02d}"
            sample_dir.mkdir()
            write_json(sample_dir / "record.json", record)
            (sample_dir / "rendered.txt").write_text(rendered, encoding="utf-8")
            with (sample_dir / "tokens.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle); writer.writerow(["index", "token_id", "text", "role", "assistant_mask", "label"])
                writer.writerows([i, token, repr(tokenizer.decode([token])), roles[i], mask[i], encoded["labels"][i]] for i, token in enumerate(original_tokens))
            result = {"sample_id": record["sample_id"], "category": record["category"], "tokens": len(mask),
                      "supervised_tokens": sum(mask), "assistant_eos": len(assistant_eos), "render_identical": True,
                      "tokens_identical": True, "non_assistant_supervised": 0,
                      "action_checks": action_checks,
                      "fixture": record.get("fixture_change"), "record_sha256": sha256(sample_dir / "record.json")}
            results.append(result)
            if example is None and record["category"] == "single_call":
                example = (record, encoded, roles)
        (ROOT / "configs/qwen3-assistant-mask.jinja").write_text(marked, encoding="utf-8")
        directory = ROOT / "experiments/E03"
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "samples.json", results)
        figures = directory / "figures"; figures.mkdir(exist_ok=True)
        record, encoded, roles = example
        focus = next(i for i, value in enumerate(encoded["assistant_masks"]) if value)
        begin, stop = max(0, focus - 30), min(len(roles), focus + 100)
        indices = list(range(begin, stop))
        font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
        fig, axis = plt.subplots(figsize=(8, 2.8), layout="constrained")
        colors = {"system": "#6B7280", "user": "#2563EB", "assistant": "#059669", "other": "#D97706"}
        axis.bar(indices, [1] * len(indices), color=[colors[roles[i]] for i in indices], width=1)
        axis.scatter(indices, [1.25 if encoded["assistant_masks"][i] else 0 for i in indices], marker="|", s=55, color="#1F2937")
        axis.set_yticks([0, 0.5, 1.25], ["mask = 0", "消息角色", "mask = 1"], fontproperties=font)
        axis.set_xlabel("真实 token 序号（完整样本的局部窗口）", fontproperties=font)
        axis.set_title("蓝色为用户，绿色为 assistant；黑色刻度表示是否参与 loss", fontproperties=font, fontsize=11)
        axis.set_ylim(-0.2, 1.6)
        for side in ["top", "right", "left"]: axis.spines[side].set_visible(False)
        image = figures / "assistant-mask.png"; fig.savefig(image, dpi=300); plt.close(fig)
        write_json(figures / "assistant-mask.source.json", {"run_id": run.name, "sample_id": record["sample_id"],
            "window": [begin, stop], "source": "sample-01/tokens.csv", "image_sha256": sha256(image)})
        finish_run(run, {"status": "completed", "exit_code": 0, "samples": len(results), "results": results,
            "original_template_sha256": sha256(run / "original.jinja"), "mask_template_sha256": sha256(run / "assistant-mask.jinja"),
            "mask_policy": "完整 assistant 消息含角色前缀及 EOS；系统、用户、工具返回、padding 不参与 loss",
            "thinking": False, "packing": False})
        print(json.dumps({"status": "completed", "samples": len(results), "max_tokens": max(r['tokens'] for r in results)}, ensure_ascii=False), flush=True)
    except Exception:
        (run / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finish_run(run, {"status": "failed", "exit_code": 1, "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
