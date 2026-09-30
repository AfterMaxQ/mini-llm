"""E00 环境记录与 E01 实际 CUDA/NF4 运算。"""
import argparse
import importlib.metadata
import json
import platform
import sys
import time
import traceback

from lab import ROOT, command, finish_run, start_run, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["E00", "E01"], required=True)
    parser.add_argument("--config")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    config = json.loads(open(args.config, encoding="utf-8").read()) if args.config else {}
    run = start_run(args.experiment, {"experiment": args.experiment, "command": [sys.executable, *sys.argv], **config}, args.run_dir)
    try:
        import torch
        gpu = command(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                       "--format=csv,noheader,nounits"], run, "nvidia-smi")
        if gpu["exit_code"] != 0:
            raise RuntimeError(gpu["stderr"])
        name, total, used, free, driver = [x.strip() for x in gpu["stdout"].splitlines()[0].split(",")]
        result = {"status": "completed", "python": sys.version, "platform": platform.platform(),
                  "gpu": {"name": name, "total_mib": int(total), "used_mib": int(used), "free_mib": int(free), "driver": driver},
                  "torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                  "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate", "numpy"]}}
        if args.experiment == "E00":
            global_probe = command(["C:/Users/ray/AppData/Local/Programs/Python/Python311/python.exe", "-c",
                                    "import torch,json; print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available()}))"], run, "global-python")
            result["global_python"] = json.loads(global_probe["stdout"])
        else:
            assert torch.cuda.is_available(), "PyTorch 没有识别 CUDA"
            torch.manual_seed(42)
            torch.cuda.reset_peak_memory_stats()
            a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            b = torch.randn_like(a)
            for _ in range(5):
                (a @ b).float().square().mean().backward()
                a.grad = None
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(20):
                loss = (a @ b).float().square().mean()
                loss.backward()
                assert torch.isfinite(loss) and torch.isfinite(a.grad).all()
                a.grad = None
            torch.cuda.synchronize()
            result["cuda_probe"] = {"dtype": "bfloat16", "shape": [1024, 1024], "warmup": 5, "iterations": 20,
                                     "total_seconds": time.perf_counter() - started, "last_loss": loss.item(),
                                     "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                                     "bf16_supported": torch.cuda.is_bf16_supported()}
            import bitsandbytes as bnb
            weights = torch.randn(32, 64, dtype=torch.bfloat16)
            layer = bnb.nn.Linear4bit(64, 32, bias=False, compute_dtype=torch.bfloat16,
                                     quant_type="nf4", compress_statistics=True)
            layer.weight = bnb.nn.Params4bit(weights.clone(), requires_grad=False, quant_type="nf4", compress_statistics=True)
            layer = layer.to("cuda")
            x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            output = layer(x)
            output.float().square().mean().backward()
            reference = torch.nn.functional.linear(x.detach().float(), weights.to("cuda").float())
            relative_error = ((output.float() - reference).norm() / reference.norm()).item()
            assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()
            assert relative_error < 0.25, "NF4 相对误差异常"
            result["nf4_probe"] = {"input_shape": list(x.shape), "output_shape": list(output.shape),
                                    "output_finite": bool(torch.isfinite(output).all()), "input_gradient_finite": bool(torch.isfinite(x.grad).all()),
                                    "relative_l2_error": relative_error, "packed_weight_dtype": str(layer.weight.dtype)}
        finish_run(run, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        (run / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finish_run(run, {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
