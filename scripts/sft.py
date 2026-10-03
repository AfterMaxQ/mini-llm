"""TRL 正式微调；完整验证、逐步指标和可恢复 checkpoint 都来自真实运行。"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from lab import ROOT, finish_run, now, sha256, start_run, write_json
from model_utils import batch, load_model
from prepare_sft import prepare
from templates import resolve_data_run, tokenizer_and_template


def valid_checkpoints(run):
    valid=[]
    for checkpoint in sorted(run.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1])):
        manifest_path=checkpoint/'manifest.json'
        state_path=checkpoint/'trainer_state.json'
        if not manifest_path.exists() or not state_path.exists():continue
        try:
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
            state=json.loads(state_path.read_text(encoding='utf-8'))
            step=int(checkpoint.name.split('-')[-1])
            files=manifest['files']
            intact=(manifest['step']==step and int(state['global_step'])==step and bool(files)
                    and all(Path(item['file']).name==item['file'] and (checkpoint/item['file']).is_file()
                            and sha256(checkpoint/item['file'])==item['sha256'] for item in files))
        except (OSError,KeyError,ValueError,TypeError,json.JSONDecodeError):
            continue
        if intact:valid.append(checkpoint)
    return valid


class RecordedTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if model.training:
            self.supervised_tokens += int((inputs["labels"][..., 1:] != -100).sum())
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        prior_mode = self.model.training
        self.model.eval()
        total, tokens, samples = 0.0, 0, set()
        stem = f"validation-step-{self.state.global_step:06d}"
        path = self.run_directory / f"{stem}.jsonl"
        number = 2
        while path.exists(): path = self.run_directory / f"{stem}-{number}.jsonl"; number += 1
        torch.cuda.synchronize(); started = time.perf_counter()
        with path.open("w", encoding="utf-8") as handle, torch.no_grad():
            for index, item in enumerate(dataset):
                count = sum(x != -100 for x in item["labels"][1:])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = self.model(**batch([item], self.processing_class.pad_token_id), use_cache=False).loss
                assert torch.isfinite(loss), "dev 验证出现非有限 loss"
                value=loss.item()
                total += value * count; tokens += count; samples.add(item["sample_id"])
                allocated=torch.cuda.memory_allocated()/1024**2;reserved=torch.cuda.memory_reserved()/1024**2
                del loss
                if self.resource_policy.get('empty_cuda_cache_each_validation_unit',False):
                    torch.cuda.empty_cache()
                handle.write(json.dumps({"sample_id": item["sample_id"], "message_index": item["message_index"],
                                        "loss": value, "target_tokens": count,
                                        "allocated_mib":allocated,"reserved_before_release_mib":reserved,
                                        "reserved_after_release_mib":torch.cuda.memory_reserved()/1024**2}, ensure_ascii=False)+"\n")
                if (index+1) % 200 == 0:
                    handle.flush(); print(f"dev loss：{index+1}/{len(dataset)} 个回复", flush=True)
                    write_json(self.run_directory/'validation-progress.json',{'time':now(),'step':self.state.global_step,
                               'units':index+1,'total_units':len(dataset),'seconds':time.perf_counter()-started,
                               'reserved_mib':torch.cuda.memory_reserved()/1024**2})
        assert samples == self.expected_validation_ids, "验证轨迹与冻结清单不一致"
        torch.cuda.synchronize()
        metrics = {metric_key_prefix+"_loss": total/tokens, metric_key_prefix+"_target_tokens": tokens,
                   metric_key_prefix+"_units": len(dataset), metric_key_prefix+"_trajectories": len(samples),
                   metric_key_prefix+"_runtime": time.perf_counter()-started}
        write_json(path.with_suffix(".summary.json"), {"step":self.state.global_step, **metrics, "rows_sha256":sha256(path)})
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self.model.train(prior_mode)
        return metrics


class RecordCallback(TrainerCallback):
    def __init__(self, run, trainer):
        self.run, self.trainer, self.last_report = run, trainer, 0

    def refresh(self):
        experiment=self.run.name.split('-')[0]
        script=self.trainer.resource_policy.get('notes_script') or {'E10':'scripts/notes_precision.py','E11':'scripts/notes_tuning.py','E12':'scripts/notes_quality.py',
                'E14':'scripts/notes_domain.py'}.get(experiment,'scripts/notes_sft.py')
        volume='02B' if experiment=='E14' else '02'
        commands=[[sys.executable, script, '--run', self.run.name] if experiment=='E14' else [sys.executable, script],
                  [sys.executable, 'scripts/report.py', '--volume', volume]]
        for command in commands:
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
            if result.returncode:
                write_json(self.run/f'report-error-{time.time_ns()}.json',{'time':now(),'command':command,
                           'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
                print('文档更新失败，证据已保存；训练继续，稍后重试。',flush=True)
                self.last_report=time.monotonic()-540
                return
        self.last_report = time.monotonic()

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize(); self.started = time.perf_counter()
        self.reserved_before = torch.cuda.memory_reserved()/1024**2
        if self.trainer.resource_policy.get('empty_cuda_cache_each_step', False):
            # 只释放没有张量使用的缓存，不删除模型、梯度或优化器状态。
            torch.cuda.empty_cache()
        self.reserved_after = torch.cuda.memory_reserved()/1024**2
        torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        # 末步也验证并保存，不能漏掉不足一个验证间隔的尾段。
        if state.global_step == state.max_steps: control.should_evaluate = control.should_save = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        torch.cuda.synchronize()
        row = {"time":now(), "step":state.global_step, **logs, "supervised_tokens":self.trainer.supervised_tokens,
               "peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,
               "peak_reserved_mib":torch.cuda.max_memory_reserved()/1024**2,
               "allocated_mib":torch.cuda.memory_allocated()/1024**2,
               "reserved_mib":torch.cuda.memory_reserved()/1024**2,
               "device_free_mib":torch.cuda.mem_get_info()[0]/1024**2}
        if "loss" in logs:
            row.update(step_seconds=time.perf_counter()-self.started,
                       reserved_before_step_mib=self.reserved_before,
                       reserved_after_release_mib=self.reserved_after)
        with (self.run/"metrics.jsonl").open("a",encoding="utf-8") as handle: handle.write(json.dumps(row,ensure_ascii=False)+"\n")
        write_json(self.run/"progress.json", {"status":"running", "latest":row, "max_steps":state.max_steps,
                   "best_metric":state.best_metric, "best_checkpoint":state.best_model_checkpoint})
        print(json.dumps(row,ensure_ascii=False),flush=True)
        if time.monotonic()-self.last_report>=600 or "eval_loss" in logs: self.refresh()

    def on_save(self, args, state, control, **kwargs):
        directory=self.run/f"checkpoint-{state.global_step}"
        write_json(directory/"lab-state.json", {"supervised_tokens":self.trainer.supervised_tokens,
                    "input_tokens":self.trainer._total_train_tokens})
        write_json(directory/"manifest.json", {"step":state.global_step, "saved":now(),
                    "files":[{"file":p.name,"bytes":p.stat().st_size,"sha256":sha256(p)} for p in directory.iterdir() if p.is_file()]})
        progress=json.loads((self.run/'progress.json').read_text(encoding='utf-8'))
        progress.update(best_metric=state.best_metric,best_checkpoint=state.best_model_checkpoint,
                        checkpoint_saved=directory.name)
        write_json(self.run/'progress.json',progress)
        self.refresh()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/sft.json")
    parser.add_argument("--size",type=int)
    parser.add_argument("--seed",type=int)
    parser.add_argument("--experiment",default="E09")
    parser.add_argument("--resume-run")
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding="utf-8"))
    if args.size is not None: config["train_size"]=args.size
    if args.seed is not None: config["seed"]=args.seed
    config["data_run"]=resolve_data_run(config["data_run"])
    frozen=ROOT/config.get('evaluation_prompt_config', 'configs/prompt-frozen.json')
    if not frozen.exists(): raise FileNotFoundError("先完成 E08，冻结 dev 提示")
    config["evaluation_prompt_sha256"]=sha256(frozen)
    config["command"]=[sys.executable,*sys.argv]
    run=ROOT/".local/runs"/args.resume_run if args.resume_run else start_run(args.experiment,config)
    resume=None
    if args.resume_run:
        saved=json.loads((run/"config.json").read_text(encoding="utf-8"))
        assert all(config[k]==saved[k] for k in config if k!="command")
        checkpoints=valid_checkpoints(run)
        if not checkpoints: raise FileNotFoundError("该运行没有通过文件哈希核验的 checkpoint，保留失败证据，不覆盖原运行")
        resume=checkpoints[-1]
        interruptions=sorted(run.glob('interruption-*.json'))
        history=json.loads(interruptions[-1].read_text(encoding='utf-8')) if interruptions else None
        started=now()
        write_json(run/f"resume-{time.time_ns()}.json",{"time":started,"started":started,"process_id":os.getpid(),
                   "command":config["command"],"checkpoint":resume.name,
                   "commit":subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   "interruption":history})
    try:
        set_seed(config["seed"])
        if config.get("domain_manifest_sha256"):
            assert sha256(ROOT / "configs/domain-mix-frozen.json") == config["domain_manifest_sha256"], "领域混合抽样清单已改变"
        if config.get("sampling_manifest_sha256"):
            assert sha256(ROOT / "configs/subsets-frozen.json") == config["sampling_manifest_sha256"]
        for name, expected_hash in config.get("data_file_hashes", {}).items():
            assert sha256(ROOT / ".local/data/processed" / config["data_run"] / name) == expected_hash, "对照数据文件已改变"
        train_cache,train_manifest=prepare(f"train-{config['train_size']}.jsonl",config["data_run"])
        dev_cache,dev_manifest=prepare("dev.jsonl",config["data_run"])
        assert train_manifest["max_sequence_length"]<=config["max_sequence_length"],"完整训练单元超过当前上下文，不允许截断"
        assert dev_manifest["max_sequence_length"]<=config.get("validation_context_length",8192),"验证单元超过验证上下文，不允许截断"
        write_json(run/"training-data.json",{"train":train_manifest,"dev":dev_manifest})
        tokenizer,_,_=tokenizer_and_template();tokenizer.padding_side="right"
        train_dataset=load_from_disk(train_cache/"dataset");dev_dataset=load_from_disk(dev_cache/"dataset")
        # 裸模型交给 TRL 准备后添加 LoRA，避免重复准备冻结适配器。
        model=load_model(load_in_4bit=config.get('load_in_4bit',True))
        storage=Counter()
        for parameter in model.parameters():storage[str(parameter.dtype)]+=parameter.numel()
        write_json(run/'base-model-storage.json',{'is_loaded_in_4bit':bool(getattr(model,'is_loaded_in_4bit',False)),
                   'stored_elements_by_dtype':dict(storage),
                   'allocated_mib':torch.cuda.memory_allocated()/1024**2})
        peft_config=LoraConfig(r=config['rank'],lora_alpha=config['alpha'],lora_dropout=config['dropout'],
            bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
        training_args=SFTConfig(output_dir=str(run),num_train_epochs=config["epochs"],max_steps=config.get("max_steps",-1),
            per_device_train_batch_size=config["micro_batch"],per_device_eval_batch_size=1,
            gradient_accumulation_steps=config["gradient_accumulation"],learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],adam_beta1=config["adam_beta1"],adam_beta2=config["adam_beta2"],adam_epsilon=config["adam_epsilon"],
            warmup_ratio=config["warmup_ratio"],max_grad_norm=config["max_grad_norm"],lr_scheduler_type=config["lr_scheduler_type"],
            bf16=True,fp16=False,tf32=False,seed=config["seed"],data_seed=config["seed"],
            gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},
            eval_strategy="steps",eval_steps=config["eval_steps"],save_strategy="steps",save_steps=config["eval_steps"],
            logging_steps=1,logging_first_step=True,load_best_model_at_end=True,metric_for_best_model="eval_loss",greater_is_better=False,
            save_only_model=False,save_total_limit=None,optim="adamw_torch",report_to=[],push_to_hub=False,
            dataloader_num_workers=0,dataloader_pin_memory=False,remove_unused_columns=False,disable_tqdm=True,
            max_length=None,packing=False,assistant_only_loss=True,dataset_kwargs={"skip_prepare_dataset":True})
        trainer=RecordedTrainer(model=model,args=training_args,processing_class=tokenizer,train_dataset=train_dataset,eval_dataset=dev_dataset,peft_config=peft_config)
        if config.get('initial_adapter'):
            # TRL 创建可训练 LoRA 后加载旧权重，避免再次准备量化模型时冻结适配器。
            adapter_path=ROOT/config['initial_adapter']/'adapter_model.safetensors'
            assert sha256(adapter_path)==config['initial_adapter_sha256']
            loaded=set_peft_model_state_dict(trainer.model,load_file(str(adapter_path)),adapter_name='default')
            assert not loaded.unexpected_keys and not any('lora_' in key for key in loaded.missing_keys)
        trainer.run_directory=run;trainer.supervised_tokens=0;trainer.resource_policy=config
        trainer.expected_validation_ids=set(dev_dataset["sample_id"])
        assert len(trainer.expected_validation_ids)==dev_manifest["independent_trajectories"]
        trainable=[(name,p) for name,p in trainer.model.named_parameters() if p.requires_grad]
        assert trainable and all('lora_' in name for name,p in trainable), 'TRL 接入后 LoRA 没有保持可训练'
        write_json(run/'trainable-parameters.json',{'parameters':sum(p.numel() for name,p in trainable),
                   'tensors':len(trainable),'only_lora':True,'dtypes':sorted({str(p.dtype) for name,p in trainable}),
                   'base_dtypes':sorted({str(p.dtype) for p in trainer.model.parameters() if not p.requires_grad}),
                   'is_loaded_in_4bit':bool(getattr(trainer.model,'is_loaded_in_4bit',False)),
                   'preparation':'TRL 准备裸模型后创建 LoRA'})
        if resume:
            state=json.loads((resume/"lab-state.json").read_text(encoding="utf-8"))
            trainer.supervised_tokens=state["supervised_tokens"];trainer._total_train_tokens=state["input_tokens"]
        callback=RecordCallback(run,trainer);trainer.add_callback(callback)
        write_json(run/"effective-training-arguments.json",training_args.to_dict())
        if not resume: trainer.evaluate()
        trained=trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        assert trainer.state.best_model_checkpoint and trainer.state.best_metric is not None
        trainer.save_model(str(run/"selected-adapter"))
        tokenizer.save_pretrained(run/"selected-adapter")
        trainer.state.save_to_json(str(run/"trainer_state.json"))
        result=finish_run(run,{"status":"trained_pending_tool_eval","exit_code":0,"train_size":config["train_size"],"seed":config["seed"],
                  "steps":trainer.state.global_step,"train":train_manifest,"dev":dev_manifest,
                  "selected_checkpoint":str(__import__('pathlib').Path(trainer.state.best_model_checkpoint).relative_to(run)),
                  "selected_dev_loss":trainer.state.best_metric,"train_metrics":trained.metrics,
                  "resume_history":[json.loads(p.read_text(encoding='utf-8')) for p in sorted(run.glob('resume-*.json'))],
                  "metrics_sha256":sha256(run/"metrics.jsonl"),
                  "scope":f"完成配置规定的训练预算；{dev_manifest['independent_trajectories']}条冻结dev用于loss选择，工具泛化由独立生成评估补齐"})
        callback.refresh();print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception:
        (run/f"failure-{time.time_ns()}.txt").write_text(traceback.format_exc(),encoding="utf-8")
        finish_run(run,{"status":"failed","exit_code":1,"error":traceback.format_exc(),
                   'allocated_mib_at_failure':torch.cuda.memory_allocated()/1024**2,
                   'reserved_mib_at_failure':torch.cuda.memory_reserved()/1024**2})
        script=config.get('notes_script') or {'E10':'scripts/notes_precision.py','E11':'scripts/notes_tuning.py','E12':'scripts/notes_quality.py',
                'E14':'scripts/notes_domain.py'}.get(args.experiment,'scripts/notes_sft.py')
        volume='02B' if args.experiment=='E14' else '02'
        command=[sys.executable,script,'--run',run.name] if args.experiment=='E14' else [sys.executable,script]
        subprocess.run(command,cwd=ROOT,check=True)
        subprocess.run([sys.executable,"scripts/report.py","--volume",volume],cwd=ROOT,check=True)
        raise


if __name__=="__main__":main()
