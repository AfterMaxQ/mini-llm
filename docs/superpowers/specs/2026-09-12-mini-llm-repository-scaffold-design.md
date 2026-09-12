# Mini LLM Repository Scaffold Design

## Goal

Initialize `AfterMaxQ/mini-llm` as a long-lived, project-driven LLM learning repository that supports three cooperating actors:

- the local machine for real training, CUDA/Triton execution, benchmarking, and profiling;
- GitHub as the canonical code/documentation state;
- ChatGPT/Codex-style agents as mentor, engineering assistant, reviewer, and experiment analyst.

The repository should stay small, understandable, and easy to resume from a fresh conversation.

## Core design principle: rules and memory are separate

The repository will use three distinct layers:

1. `AGENTS.md` — repository-wide operating rules for coding agents.
2. `skills/mini-llm-learning/SKILL.md` — the reusable teaching, implementation, and performance-experiment workflow.
3. `PROJECT_MEMORY.md` — the single concise continuity file for current project state.

`PROJECT_MEMORY.md` is intentionally not a second specification or changelog. It should remain short enough to read at the beginning of every session.

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

No model implementation files are created yet. Files such as `attention.py`, `rope.py`, `train.py`, or benchmark scripts should appear only when the corresponding learning step begins.

## `AGENTS.md`

`AGENTS.md` defines durable repository rules. It should instruct agents to:

- read `PROJECT_MEMORY.md` before starting substantive work;
- read `skills/mini-llm-learning/SKILL.md` when doing teaching, implementation, benchmarking, profiling, or performance analysis;
- treat GitHub/current repository contents as the source of truth for code state;
- distinguish local-machine execution from GitHub-side editing;
- never claim that CUDA/GPU training, benchmarks, or profilers ran unless actual output was supplied or executed in an environment that can run them;
- prefer small, testable changes over large rewrites;
- avoid overengineering;
- preserve user changes and avoid destructive Git operations;
- update `PROJECT_MEMORY.md` only after meaningful progress;
- keep project rules out of `PROJECT_MEMORY.md` to avoid duplication.

## `skills/mini-llm-learning/SKILL.md`

The skill defines how an agent acts as both mentor and engineering assistant.

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
- explain tensor shapes and data flow for important model operations;
- do not hide learning-critical modules behind `nn.Transformer`;
- let the user understand first implementations of Attention, RoPE, RMSNorm, SwiGLU, training loops, KV cache, schedulers, and Triton kernels;
- handle repetitive engineering work directly when it does not carry much learning value.

The skill should also encode performance-analysis discipline:

- baseline before optimization;
- separate observation, hypothesis, evidence, and conclusion;
- use real benchmark/profile output before asserting causes;
- preserve experiment conditions such as GPU, dtype, shapes, batch size, sequence length, warmup, and iteration count;
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

## Code scaffold

The initial Python package will contain only `src/minillm/__init__.py`. The repository establishes packaging and test locations without prematurely implementing model components.

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

Large checkpoints, datasets, profiler traces, caches, and virtual environments must remain outside normal Git tracking.

## README

The initial README should describe:

- the project's purpose: learn LLM training/inference/systems by building and measuring a real mini LLM;
- the local/GitHub/agent workflow;
- the current early stage;
- how continuity works through `PROJECT_MEMORY.md`, `AGENTS.md`, and the learning skill;
- a short note that the project will grow incrementally rather than starting with a large generated codebase.

## Validation

Repository scaffold validation is intentionally lightweight:

- all expected files exist;
- `pyproject.toml` is valid TOML;
- `python -c "import pathlib; print(pathlib.Path('PROJECT_MEMORY.md').exists())"` can verify local pull state;
- no GPU or model test is claimed during scaffold initialization because no model code exists yet.

## Non-goals for initialization

This scaffold does not yet implement:

- tokenizer;
- Transformer layers;
- training loop;
- generation;
- benchmark scripts;
- Triton kernels;
- inference engine;
- CI/CD.

Those are introduced incrementally as learning tasks begin.
