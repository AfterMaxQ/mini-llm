---
name: mini-llm-learning
description: Use when teaching, implementing, benchmarking, profiling, or optimizing the mini-llm project across training or inference systems.
---

# Mini LLM Learning Workflow

## Core principle

Use the repository as a laboratory: advance the real project, teach only what the current step needs, verify correctness, then measure before optimizing.

## Default loop

```text
current repository state
→ smallest useful goal
→ just-in-time explanation
→ implementation
→ correctness check
→ local experiment when required
→ evidence-based analysis
→ concise memory update
```

## Teaching style

Keep full technical content, but reduce how much the learner must hold at once.

- Prose first; formulas only when they clarify the mechanism.
- Introduce one main concept or engineering decision at a time.
- For important operations, connect intuition → tensor shapes → minimal math → code → systems meaning.
- If confusion appears, reduce presentation complexity, not conceptual depth.
- Side topics get a brief answer and return to the main path unless they block the current task or the learner explicitly wants a detour.
- Prefer concrete examples and actual project tensors over abstract surveys.

## What the learner should implement deliberately

First implementations of these deserve explanation and active learner involvement:

- Attention, RoPE, RMSNorm, SwiGLU, Decoder Block;
- training loop and generation loop;
- KV cache, prefill/decode split, batching, scheduler;
- Triton kernels and performance-critical data movement.

Boilerplate, repetitive configuration, file cleanup, and routine test scaffolding may be handled directly.

## Dual project tracks

### Training systems

Progress from a correct decoder-only Mini LLM to reproducible training, then study precision, activation memory, checkpointing, profiling, SDPA/FlashAttention, Triton, kernel behavior, throughput, and memory efficiency.

### Inference systems

Evolve the same model from naive autoregressive decoding through KV cache, prefill/decode, static and continuous batching, request scheduling, paged KV cache, prefix cache, chunked prefill, speculative decoding, and finally vLLM/SGLang comparison and tuning.

## Correctness before performance

Do not optimize an unverified implementation. Use small tests and, for training, a tiny overfit test when appropriate. For inference, compare optimized paths against a simple reference implementation.

## Performance reasoning

Never infer a cause from a timing change alone. Separate:

1. **Observation** — what was measured;
2. **Hypothesis** — possible cause;
3. **Evidence** — benchmark/profile data;
4. **Conclusion** — what the evidence supports.

Record enough conditions to reproduce useful experiments: hardware, commit, dtype, shapes, batch/concurrency, sequence or prompt/generated lengths, warmup, iterations, and relevant memory/cache settings.

## Memory maintenance

After meaningful progress, update `PROJECT_MEMORY.md` by replacing stale state with a concise current snapshot. Do not copy rules, long explanations, or a historical changelog into memory.
