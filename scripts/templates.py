"""保持官方渲染不变，只给 assistant 分支增加监督区间标记。"""
import json

from transformers import AutoTokenizer

from lab import ROOT, sha256


def tokenizer_and_template():
    tokenizer = AutoTokenizer.from_pretrained(ROOT / ".local/models/Qwen3-1.7B", local_files_only=True)
    original = tokenizer.chat_template
    begin = '{%- elif message.role == "assistant" %}'
    end = '{%- elif message.role == "tool" %}'
    assert original.count(begin) == original.count(end) == 1
    marked = original.replace(begin, begin + "\n        {%- generation %}")
    marked = marked.replace(end, "{%- endgeneration %}\n    " + end)
    return tokenizer, original, marked


def encode_record(tokenizer, template, record):
    encoded = tokenizer.apply_chat_template(record["messages"], tools=record["tools"], chat_template=template,
                    tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
                    add_generation_prompt=False, enable_thinking=False)
    mask = encoded["assistant_masks"]
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
            "labels": [token if supervised else -100 for token, supervised in zip(encoded["input_ids"], mask)],
            "assistant_masks": mask}


def action_records(record):
    return [{**record, "messages": record["messages"][:index + 1], "target_message_index": index}
            for index, message in enumerate(record["messages"]) if message["role"] == "assistant"]


def encode_action(tokenizer, template, action):
    encoded = encode_record(tokenizer, template, action)
    mask = encoded["assistant_masks"]
    starts = [i for i, value in enumerate(mask) if value and (i == 0 or not mask[i - 1])]
    assert starts
    begin = starts[-1]
    mask[:begin] = [0] * begin
    encoded["labels"] = [token if target else -100 for token, target in zip(encoded["input_ids"], mask)]
    return encoded


def resolve_data_run(data_run="E02-R02"):
    if (ROOT / ".local/data/processed" / data_run).exists():
        return data_run
    frozen = json.loads((ROOT / "configs/data-frozen.json").read_text(encoding="utf-8"))
    for directory in sorted((ROOT / ".local/data/processed").glob("E02-R*"), reverse=True):
        if all((directory / name).exists() and sha256(directory / name) == item["sha256"] for name, item in frozen["outputs"].items()):
            return directory.name
    raise FileNotFoundError("没有找到冻结划分或与其全部哈希相同的本机复现数据；先运行 data.py")


def read_records(name="train-10000.jsonl", data_run="E02-R02"):
    data_run = resolve_data_run(data_run)
    path = ROOT / ".local/data/processed" / data_run / name
    frozen = json.loads((ROOT / "configs/data-frozen.json").read_text(encoding="utf-8"))
    if data_run == frozen["run_id"]:
        assert sha256(path) == frozen["outputs"][name]["sha256"], "冻结数据文件已改变"
    subsets = ROOT / "configs/subsets-frozen.json"
    if subsets.exists():
        datasets = json.loads(subsets.read_text(encoding="utf-8"))["datasets"]
        if data_run in datasets:
            assert sha256(path) == datasets[data_run]["files"][name], "固定抽样数据已改变"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
