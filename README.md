# mini-llm

A project-driven lab for learning **LLM training systems and inference systems** by building, running, measuring, and optimizing one small decoder-only language model over time.

The goal is not to copy a tutorial GPT. The repository should gradually become a reproducible learning artifact with real code, correctness checks, benchmark results, profiler evidence, and explainable engineering decisions.

## Two equal learning tracks

### Training systems

Build the model and training path from first principles, then study and tune topics such as mixed precision, activation memory, checkpointing, SDPA/FlashAttention, Triton kernels, profiling, throughput, and memory efficiency.

### Inference systems

Evolve the same model from naive autoregressive generation into a small inference engine: KV cache, prefill/decode, batching, scheduling, paged/prefix cache, chunked prefill, speculative decoding, and later comparison/tuning with vLLM and SGLang.

## How the project works

```text
ChatGPT / coding agent
  mentor + engineer + reviewer + experiment analyst
                ↕
             GitHub
      code + rules + memory + results
                ↕
           local machine
 training + CUDA/Triton + benchmark + profiler
```

GitHub is the canonical shared state. Local execution is required for claims about GPU training, CUDA/Triton performance, benchmarks, or profiler results.

## Continuity

The repository deliberately separates three concerns:

- `AGENTS.md` — durable repository rules for agents.
- `skills/mini-llm-learning/SKILL.md` — the teaching, implementation, and performance-experiment workflow.
- `PROJECT_MEMORY.md` — a short, continuously maintained snapshot of the current project state.

A new session should read `PROJECT_MEMORY.md` first, then the skill and relevant code. The memory file stays concise; rules belong in `AGENTS.md` or the skill.

## Learning style

The project keeps technical depth but controls cognitive load: one meaningful concept or implementation step at a time, intuition before formulas, tensor shapes and data flow when useful, and short treatment of side topics unless they truly block the current work.

## Repository shape

```text
mini-llm/
├── AGENTS.md
├── PROJECT_MEMORY.md
├── skills/mini-llm-learning/SKILL.md
├── src/minillm/
├── configs/
├── experiments/
├── tests/
└── docs/superpowers/
```

Model modules, training scripts, inference-engine components, benchmark tools, and Triton kernels are intentionally added only when their learning step begins.

## Local bootstrap

After pulling the repository:

```bash
git pull
python -m pip install -e ".[dev]"
python -c "import minillm; print(minillm.__name__)"
```

At the initialization stage there is no claim that model training or GPU benchmarks have already run.
