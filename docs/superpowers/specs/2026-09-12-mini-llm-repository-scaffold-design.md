# Mini LLM Repository Scaffold Design

## Goal

Initialize `AfterMaxQ/mini-llm` as a long-lived, project-driven LLM learning repository that supports both **training systems** and **inference systems** learning through real code, experiments, profiling, and optimization.

The repository supports three cooperating actors:

- the local machine for real training, inference, CUDA/Triton execution, benchmarking, and profiling;
- GitHub as the canonical code/documentation state;
- ChatGPT/Codex-style agents as mentor, engineering assistant, reviewer, and experiment analyst.

The repository should stay small, understandable, and easy to resume from a fresh conversation.

## Core design principle: rules and memory are separate

The repository will use three distinct layers:

1. `AGENTS.md` — repository-wide operating rules for coding agents.
2. `skills/mini-llm-learning/SKILL.md` — the reusable teaching, implementation, and performance-experiment workflow for both training and inference.
3. `PROJECT_MEMORY.md` — the single concise continuity file for current project state.

`PROJECT_MEMORY.md` is intentionally not a second specification or changelog. It should remain short enough to read at the beginning of every session.

## Learning scope: training and inference are both first-class

The project grows in two connected stages using the same model codebase.

### Training systems track

Build and understand a small modern decoder-only LLM, then measure and optimize its training behavior. Topics introduced when the project reaches them include:

- tokenizer/data pipeline;
- Embedding, RMSNorm, RoPE, causal Attention/GQA, SwiGLU, residual stream, LM Head;
- next-token loss, AdamW, checkpointing, validation, generation;
- FP32/BF16, activation memory, checkpointing, batch/sequence-length trade-offs;
- profiler-driven optimization, SDPA, `torch.compile`, Triton kernels, FlashAttention, GEMM/data-movement reasoning.

### Inference systems track

After a correct model baseline exists, evolve the same project into a mini inference engine rather than treating inference as a separate unrelated tutorial. Topics include:

- naive autoregressive decoding;
- KV Cache;
- prefill versus decode;
- static and continuous batching;
- request scheduling;
- paged KV/cache management;
- prefix cache;
- chunked prefill;
- speculative decoding;
- inference quantization when appropriate;
- TTFT, TPOT, throughput, latency, memory, cache hit rate, batch/concurrency measurements;
- profiling and tuning of inference bottlenecks;
- later comparison with and source study of vLLM and SGLang.

The repository skill must treat training optimization and inference optimization with the same evidence discipline: establish a baseline, form a hypothesis, measure, profile, and only then conclude.

## Initial repository structure

```text
mini-llm/
├── AGENTS.md
├── PROJECT_MEMORY.md
├── README.md
├── .gitignore
├── pyproject.toml
├── configs/
│   └── .gitkeep
├── experiments/
│   └── .gitkeep
├── skills/
│   └── mini-llm-learning/
│       └── SKILL.md
├── src/
│   └── minillm/
│       └── __init__.py
├── tests/
│   └── .gitkeep
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-09-12-mini-llm-repository-scaffold-design.md
```

No model or inference-engine implementation files are created yet. Files such as `attention.py`, `rope.py`, `train.py`, KV-cache/scheduler modules, or benchmark scripts should appear only when the corresponding learning step begins.

## `AGENTS.md`

`AGENTS.md` defines durable repository rules. It should instruct agents to:

- read `PROJECT_MEMORY.md` before starting substantive work;
- read `skills/mini-llm-learning/SKILL.md` when doing teaching, implementation, benchmarking, profiling, or performance analysis;
- treat GitHub/current repository contents as the source of truth for code state;
- distinguish local-machine execution from GitHub-side editing;
- never claim that CUDA/GPU training, inference, benchmarks, or profilers ran unless actual output was supplied or executed in an environment that can run them;
- prefer small, testable changes over large rewrites;
- avoid overengineering;
- preserve user changes and avoid destructive Git operations;
- update `PROJECT_MEMORY.md` only after meaningful progress;
- keep project rules out of `PROJECT_MEMORY.md` to avoid duplication.

## `skills/mini-llm-learning/SKILL.md`

The skill defines how an agent acts as both mentor and engineering assistant across model training and inference systems work.

Its workflow is project-first:

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

The skill should encode these teaching principles:

- prose-first explanations with formulas only when useful;
- low cognitive load and small increments;
- explain tensor shapes and data flow for important model and inference operations;
- do not hide learning-critical modules behind `nn.Transformer` or an opaque inference framework;
- let the user understand first implementations of Attention, RoPE, RMSNorm, SwiGLU, training loops, KV Cache, prefill/decode, schedulers, paged cache concepts, and Triton kernels;
- handle repetitive engineering work directly when it does not carry much learning value.

The skill should encode two connected implementation arcs:

```text
Mini LLM training baseline
→ training correctness and profiling
→ training optimization experiments
→ naive inference baseline
→ KV Cache / prefill / decode
→ batching and scheduling
→ cache-management and serving optimizations
→ vLLM / SGLang comparison and source study
```

The skill should also encode performance-analysis discipline:

- baseline before optimization;
- separate observation, hypothesis, evidence, and conclusion;
- use real benchmark/profile output before asserting causes;
- for training preserve conditions such as GPU, dtype, shapes, batch size, sequence length, warmup, and iteration count;
- for inference additionally preserve concurrency, prompt length, generated length, cache state, TTFT, TPOT, throughput, and memory conditions when relevant;
- allow experiments to be stored under `experiments/` when they are worth preserving.

## `PROJECT_MEMORY.md`

This is the only persistent session-resume memory file. It should be deliberately compact, using headings such as:

```text
# Project Memory

## Goal
## Current Stage
## Completed / Verified
## Current Understanding
## Open Questions
## Current Work
## Next Step
```

Rules for maintenance:

- keep it concise;
- record only current, useful state;
- rewrite stale information instead of appending indefinitely;
- do not duplicate agent rules from `AGENTS.md`;
- do not duplicate the entire README;
- distinguish verified results from plans;
- update only after meaningful project or learning progress.

The Goal should explicitly state that the long-term project covers both training and inference systems, including performance tuning and later vLLM/SGLang study.

## Code scaffold

The initial Python package will contain only `src/minillm/__init__.py`. The repository establishes packaging and test locations without prematurely implementing model or inference-engine components.

`pyproject.toml` should provide a minimal Python project using a `src/` layout and basic development/test dependencies suitable for later PyTorch work, while avoiding unnecessary framework dependencies at initialization.

## Experiments

`experiments/` exists as a stable location but starts empty except for `.gitkeep`. A future experiment should be created only when there is a real comparison or result worth preserving.

A typical future experiment may contain:

```text
experiments/exp_001_name/
├── config.json
├── metrics.json
└── notes.md
```

Training and inference experiments share the same evidence format, but inference experiments may additionally record TTFT, TPOT, request concurrency, prompt/generated lengths, KV-cache usage, cache hit rate, and scheduler/batching configuration.

Large checkpoints, datasets, profiler traces, caches, and virtual environments must remain outside normal Git tracking.

## README

The initial README should describe:

- the project's purpose: learn LLM training, inference, serving, and GPU systems by building and measuring a real mini LLM and then evolving it into a mini inference engine;
- the local/GitHub/agent workflow;
- the current early stage;
- how continuity works through `PROJECT_MEMORY.md`, `AGENTS.md`, and the learning skill;
- a short note that the project will grow incrementally rather than starting with a large generated codebase.

## Validation

Repository scaffold validation is intentionally lightweight:

- all expected files exist;
- `pyproject.toml` is valid TOML;
- `python -c "import pathlib; print(pathlib.Path('PROJECT_MEMORY.md').exists())"` can verify local pull state;
- no GPU, training, inference, or model test is claimed during scaffold initialization because no model code exists yet.

## Non-goals for initialization

This scaffold does not yet implement:

- tokenizer;
- Transformer layers;
- training loop;
- generation;
- benchmark scripts;
- Triton kernels;
- KV Cache;
- inference scheduler;
- paged/prefix cache;
- serving engine;
- vLLM/SGLang integration;
- CI/CD.

Those are introduced incrementally as learning tasks begin.
