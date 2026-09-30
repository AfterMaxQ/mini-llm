"""本机模型加载、右侧 padding 与 LoRA 的共用小函数。"""
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from lab import ROOT


def load_model(adapter=False, config=None):
    config = config or {}
    model = AutoModelForCausalLM.from_pretrained(ROOT / ".local/models/Qwen3-1.7B", local_files_only=True,
              torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
              quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model.config.use_cache = False
    if adapter:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(r=config.get("rank", 16), lora_alpha=config.get("alpha", 32),
                    lora_dropout=config.get("dropout", 0.05), bias="none", task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    return model


def batch(records, pad_id):
    length = max(len(r["input_ids"]) for r in records)
    padded = {}
    for field, value in [("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)]:
        padded[field] = torch.tensor([r[field] + [value] * (length - len(r[field])) for r in records],
                                     device="cuda", dtype=torch.long)
    return padded
