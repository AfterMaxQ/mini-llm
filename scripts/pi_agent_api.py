"""将本地 Transformers + LoRA 模型包装为 Pi 可调用的 OpenAI 兼容接口。"""
import argparse
import hashlib
import json
import re
import select
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from lab import sha256
from model_utils import load_model, model_loading_kwargs
from templates import tokenizer_and_template


def normalize_messages(messages):
    result = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") not in {"text", "thinking"}:
                    raise ValueError("当前本机评测只支持文本消息块")
                if block["type"] == "text":
                    parts.append(block.get("text") or "")
            content = "".join(parts)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            raise TypeError("消息 content 必须是字符串或文本块列表")
        item = {"role": message["role"], "content": content}
        if message.get("name"):
            item["name"] = message["name"]
        if message.get("tool_calls"):
            calls = []
            for raw in message["tool_calls"]:
                function = raw.get("function", {})
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                calls.append({"id": raw.get("id", ""), "type": "function",
                              "function": {"name": function["name"], "arguments": arguments}})
            item["tool_calls"] = calls
        result.append(item)
    return result


def parse_calls(text):
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if text.count("<tool_call>") != len(blocks) or text.count("</tool_call>") != len(blocks):
        return text, []
    try:
        calls = [json.loads(block) for block in blocks]
        if any(not isinstance(call, dict) or not isinstance(call.get("name"), str) or
               not isinstance(call.get("arguments"), dict) for call in calls):
            return text, []
    except Exception:
        return text, []
    content = re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", text, flags=re.S).strip()
    return content, calls


class Handler(BaseHTTPRequestHandler):
    server_version = "MiniLLM-local/1"

    def log_message(self, *_):
        return

    def send_json(self, status, value):
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ready", "run_id": self.server.run_id,
                                 "model": self.server.model_id})
        elif self.path == "/v1/models":
            self.send_json(200, {"object": "list", "data": [{"id": self.server.model_id, "object": "model"}]})
        else:
            self.send_json(404, {"error": {"message": "not_found"}})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": "not_found"}})
            return
        request_id = "chatcmpl-" + uuid.uuid4().hex
        started = time.time()
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            with self.server.generate_lock:
                messages = normalize_messages(payload["messages"])
                tools = payload.get("tools") or None
                encoded = self.server.tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, return_tensors="pt")
                input_ids = encoded.to(self.server.model.device)
                prompt_tokens = int(input_ids.shape[-1])
                tool_names = sorted(tool.get("function", {}).get("name", "") for tool in tools or [])
                message_roles = [message["role"] for message in messages]
                prompt_hash = hashlib.sha256(json.dumps(encoded[0].tolist(), separators=(",", ":")).encode()).hexdigest()
                messages_hash = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True,
                                                           separators=(",", ":")).encode("utf-8")).hexdigest()
                if prompt_tokens >= self.server.context_window:
                    raise ValueError("context_overflow")
                max_new = min(int(payload.get("max_tokens") or self.server.max_new_tokens),
                              self.server.max_new_tokens, self.server.context_window - prompt_tokens)
                attention_mask = torch.ones_like(input_ids)
                torch.manual_seed(self.server.inference_seed)
                torch.cuda.manual_seed_all(self.server.inference_seed)
                disconnected = DisconnectStop(self.connection)
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    generated = self.server.model.generate(
                        input_ids=input_ids, attention_mask=attention_mask,
                        max_new_tokens=max_new, do_sample=True, temperature=self.server.temperature, use_cache=True,
                        stopping_criteria=StoppingCriteriaList([disconnected]),
                        pad_token_id=self.server.tokenizer.pad_token_id)
                new_tokens = generated[0, input_ids.shape[-1]:]
                text = self.server.tokenizer.decode(new_tokens, skip_special_tokens=True)
                content, calls = parse_calls(text)
                eos = self.server.model.generation_config.eos_token_id
                eos = set(eos if isinstance(eos, list) else [eos])
                truncated = not any(int(token) in eos for token in new_tokens.tolist())
                reserved_before = round(torch.cuda.memory_reserved() / 1024 ** 2)
                peak_allocated = round(torch.cuda.max_memory_allocated() / 1024 ** 2)
                # 释放不同长度请求留下的空闲缓存，控制 reserved 显存。
                torch.cuda.empty_cache()
                trace = {"time": started, "run_id": self.server.run_id, "request_id": request_id,
                         "prompt_tokens": prompt_tokens, "generated_tokens": len(new_tokens),
                         "seconds": round(time.time() - started, 3),
                         "temperature": self.server.temperature, "inference_seed": self.server.inference_seed,
                         "truncated": truncated, "cancelled": disconnected.disconnected,
                         "prompt_sha256": prompt_hash, "messages_sha256": messages_hash,
                         "message_roles": message_roles, "tool_names": tool_names,
                         "tool_calls": calls, "raw_response": text,
                         "peak_allocated_mib": peak_allocated, "reserved_mib_before_cleanup": reserved_before,
                         "reserved_mib_after_cleanup": round(torch.cuda.memory_reserved() / 1024 ** 2)}
                with self.server.trace_lock, self.server.trace_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
            if disconnected.disconnected:
                return
            if payload.get("stream"):
                self.send_sse(request_id, content, calls, truncated)
            else:
                message = {"role": "assistant", "content": content or None}
                if calls:
                    message["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex,
                        "type": "function", "function": {"name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False)}} for call in calls]
                self.send_json(200, {"id": request_id, "object": "chat.completion", "created": int(started),
                    "model": self.server.model_id, "choices": [{"index": 0, "message": message,
                    "finish_reason": "length" if truncated else "tool_calls" if calls else "stop"}]})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception as error:
            self.send_json(500, {"error": {"message": str(error)[:300], "type": type(error).__name__}})
            with self.server.trace_lock, self.server.trace_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"time": started, "run_id": self.server.run_id,
                    "error": type(error).__name__ + ":" + str(error)[:300],
                    "seconds": round(time.time() - started, 3)}, ensure_ascii=False) + "\n")

    def send_sse(self, request_id, content, calls, truncated):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(value):
            self.wfile.write(b"data: " + json.dumps(value, ensure_ascii=False).encode("utf-8") + b"\n\n")
            self.wfile.flush()

        base = {"id": request_id, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": self.server.model_id}
        emit({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        if content:
            emit({**base, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
        for index, call in enumerate(calls):
            delta = {"tool_calls": [{"index": index, "id": "call_" + uuid.uuid4().hex,
                "type": "function", "function": {"name": call["name"],
                "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}]}
            emit({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
        finish_reason = "length" if truncated else "tool_calls" if calls else "stop"
        emit({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class DisconnectStop(StoppingCriteria):
    """Pi 中止请求时同步停止本机生成，不让旧请求继续占用 GPU。"""
    def __init__(self, connection):
        self.connection = connection
        self.disconnected = False

    def __call__(self, input_ids, scores, **kwargs):
        try:
            if select.select([self.connection], [], [], 0)[0]:
                self.disconnected = not self.connection.recv(1, socket.MSG_PEEK)
        except OSError:
            self.disconnected = True
        return self.disconnected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pi-agent-eval.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trace", required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if config.get("base_model_path"):
        manifest_path = ROOT / config["model_manifest_path"]
        assert sha256(manifest_path) == config["model_manifest_sha256"], "教师模型清单哈希不匹配"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["revision"] == config["model_revision"]
        assert manifest["repo"] == "Qwen/Qwen3-4B"
        assert config["quantization"] == "NF4" and config["load_in_4bit"]
        model_path = ROOT / config["base_model_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        template_hash = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
        assert template_hash == config["tokenizer_template_sha256"], "教师 tokenizer 模板哈希不匹配"
        model = AutoModelForCausalLM.from_pretrained(
            model_path, **model_loading_kwargs(config.get("load_in_4bit", True)))
        model.config.use_cache = True
        model_details = {"base_model": config["base_model_path"],
                         "model_revision": config["model_revision"],
                         "model_manifest_sha256": config["model_manifest_sha256"],
                         "tokenizer_template_sha256": template_hash,
                         "quantization": config["quantization"]}
    else:
        tokenizer, _, _ = tokenizer_and_template()
        model = PeftModel.from_pretrained(load_model(), ROOT / config["adapter"])
        model_details = {"adapter_sha256": sha256(ROOT / config["adapter"] / "adapter_model.safetensors")}
    model.config.use_cache = True
    model.eval()
    server = ThreadingHTTPServer((config["api"]["host"], config["api"]["port"]), Handler)
    server.model, server.tokenizer = model, tokenizer
    server.model_id, server.run_id = config["api"]["model"], args.run_id
    server.context_window, server.max_new_tokens = config["context_window"], config["max_new_tokens"]
    server.temperature, server.inference_seed = config["temperature"], config["inference_seed"]
    server.trace_file = Path(args.trace)
    server.generate_lock, server.trace_lock = threading.Lock(), threading.Lock()
    print(json.dumps({"status": "ready", "run_id": args.run_id, **model_details,
                      "model": config["api"]["model"], "gpu": torch.cuda.get_device_name(0),
                      "reserved_mib": round(torch.cuda.memory_reserved() / 1024 ** 2)}, ensure_ascii=False), flush=True)
    server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
