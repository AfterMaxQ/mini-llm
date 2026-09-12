# AGENTS.md

## Project purpose

`mini-llm` is a project-driven LLM systems learning repository. It covers two equal main tracks:

- training systems: model implementation, training correctness, memory, precision, profiling, kernels, and optimization;
- inference systems: autoregressive decoding, KV cache, prefill/decode, batching, scheduling, paged/prefix cache, speculative decoding, and later vLLM/SGLang study and tuning.

## Startup order

Before substantive work:

1. Read `PROJECT_MEMORY.md`.
2. Read `skills/mini-llm-learning/SKILL.md` for teaching, implementation, benchmarking, profiling, or optimization work.
3. Read the relevant current code and experiment files before making assumptions.

Repository contents are the source of truth for current code state. `PROJECT_MEMORY.md` is the source of truth for concise session continuity.

## Rules vs memory

- Put durable operating rules here or in the skill.
- Put only current project state in `PROJECT_MEMORY.md`.
- Do not turn `PROJECT_MEMORY.md` into a changelog, specification, or duplicate of this file.
- Update memory only after meaningful learning, implementation, or experimental progress; rewrite stale state instead of appending forever.

## Teaching rules

Preserve technical depth while controlling cognitive load.

- Prefer one main concept or implementation goal at a time.
- Explain intuition first, then the minimum useful math, tensor shapes, code, and engineering meaning.
- Do not remove important knowledge just to make an explanation shorter; sequence it into smaller pieces instead.
- When the user says they are confused, simplify the presentation rather than replacing the concept with a shallow summary.
- Course-adjacent side topics should get the shortest explanation needed to unblock the current work, then return to the main line. Do not pause the project for long detours unless the topic is truly blocking or the user explicitly asks to explore it.
- Revisit earlier concepts naturally when the project exposes them again.

## Engineering rules

- Prefer small, testable changes and simple module boundaries.
- Avoid premature abstractions, framework-building, and generated boilerplate.
- Do not hide learning-critical first implementations behind `nn.Transformer` or similarly opaque wrappers.
- Preserve user changes. Do not use destructive Git operations such as force reset/clean unless explicitly requested.
- Treat first implementations of Attention, RoPE, RMSNorm, SwiGLU, training loops, KV cache, schedulers, and Triton kernels as learning-critical: explain the design and data flow before or while implementing them.
- Repetitive, low-learning-value engineering work may be completed directly.

## Execution boundaries

GitHub-side edits are not proof of local execution.

Never claim that CUDA, Triton, GPU training, benchmarks, profilers, or local commands ran unless actual output was produced in an environment that can run them or supplied by the user.

Use these labels when relevant:

- **Observed:** measured or executed result.
- **Hypothesis:** plausible explanation not yet proven.
- **Evidence:** benchmark/profile/log data used to test a hypothesis.
- **Conclusion:** explanation supported by evidence.

## Optimization discipline

For both training and inference:

1. establish correctness;
2. establish a baseline;
3. change as few variables as practical;
4. benchmark under recorded conditions;
5. profile when the cause is uncertain;
6. keep an optimization only when evidence supports it.

Training measurements may include loss, tokens/s, latency, peak memory, MFU, bandwidth, and kernel time.

Inference measurements may include TTFT, TPOT/inter-token latency, throughput, concurrency, prompt/generated lengths, KV-cache memory, cache hit rate, and scheduler behavior.

Store important comparisons under `experiments/`; do not create an experiment record for every trivial run.
