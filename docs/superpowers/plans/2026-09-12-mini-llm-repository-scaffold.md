# Mini LLM Repository Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Initialize `AfterMaxQ/mini-llm` with the smallest durable repository scaffold for project-driven LLM learning across both training systems and inference systems, with rules, learning workflow, concise project memory, Python package structure, and experiment locations.

**Architecture:** Keep durable rules in `AGENTS.md`, reusable mentor/engineering workflow in `skills/mini-llm-learning/SKILL.md`, and resumable current state in `PROJECT_MEMORY.md`. The skill must cover a connected arc from a correct Mini LLM training baseline through training optimization, then into a Mini Inference Engine with KV Cache, prefill/decode, batching/scheduling, cache-management optimizations, and later vLLM/SGLang study. Keep code intentionally minimal during initialization.

**Tech Stack:** Markdown, Python packaging via `pyproject.toml`, pytest as the initial development/test dependency, Git/GitHub.

**Spec:** `docs/superpowers/specs/2026-09-12-mini-llm-repository-scaffold-design.md`

## Global Constraints

- GitHub/current repository contents are the source of truth for code state.
- The long-term learning scope includes both LLM training optimization and inference/serving optimization.
- Do not claim local CUDA/GPU execution, training, inference, benchmarks, or profiling occurred unless actual output is available.
- Keep `PROJECT_MEMORY.md` concise and state-focused; do not duplicate repository rules there.
- Do not create tokenizer, Transformer, training, generation, benchmark, Triton, KV-cache, scheduler, inference-engine, vLLM/SGLang integration, or CI implementation during initialization.
- Avoid overengineering and unnecessary dependencies.
- Preserve user changes and avoid destructive Git operations.

---

### Task 1: Establish durable agent rules and the training+inference learning workflow

**Files:**
- Create: `AGENTS.md`
- Create: `skills/mini-llm-learning/SKILL.md`

**Interfaces:**
- Consumes: repository design spec.
- Produces: durable operating rules and reusable project-first teaching/engineering workflow used by later training and inference sessions.

- [ ] **Step 1: Create `AGENTS.md`**

Write repository rules that require agents to read `PROJECT_MEMORY.md`, invoke the repository learning skill for teaching/implementation/performance work, treat repository contents as current truth, distinguish GitHub edits from local GPU execution, prefer small testable changes, preserve user work, and update memory only after meaningful progress. Explicitly state that the project covers both training systems and inference systems rather than treating inference as an optional appendix.

- [ ] **Step 2: Create `skills/mini-llm-learning/SKILL.md`**

Define the workflow:

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

Include low-cognitive-load teaching, tensor-shape/data-flow explanations, learning-critical module guidance, baseline-before-optimization, and observation/hypothesis/evidence/conclusion discipline.

Encode both learning arcs:

```text
Mini LLM training baseline
→ training correctness/profiling
→ training optimization
→ naive inference baseline
→ KV Cache
→ prefill/decode
→ batching / continuous batching
→ scheduler
→ paged/prefix cache
→ chunked prefill
→ speculative decoding
→ vLLM / SGLang comparison and source study
```

For inference experiments, require relevant metrics such as TTFT, TPOT, throughput, latency, concurrency, prompt/generated lengths, memory/KV-cache usage, and cache state when applicable.

- [ ] **Step 3: Verify both files can be fetched from `main`**

Expected: both paths exist and contain the required rule/workflow sections, including explicit training and inference scopes.

---

### Task 2: Create the single persistent project memory

**Files:**
- Create: `PROJECT_MEMORY.md`

**Interfaces:**
- Consumes: project goal and current initialization state.
- Produces: the single short session-resume state document read at the beginning of future work.

- [ ] **Step 1: Create initial memory**

Use exactly these state-oriented headings:

```markdown
# Project Memory

## Goal
## Current Stage
## Completed / Verified
## Current Understanding
## Open Questions
## Current Work
## Next Step
```

The Goal must state that this repository will build a Mini LLM, optimize its training, then evolve the same model into a Mini Inference Engine for inference/serving optimization and later vLLM/SGLang study.

Record that repository scaffolding is being established, no model implementation or GPU experiment has been verified yet, and the next learning milestone is defining Mini LLM v0.1 before implementing the first minimal module.

- [ ] **Step 2: Keep memory concise**

Verify it contains current state only, with no duplicated agent rules and no long-term changelog.

---

### Task 3: Create the minimal Python/package scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/minillm/__init__.py`
- Create: `configs/.gitkeep`
- Create: `experiments/.gitkeep`
- Create: `tests/.gitkeep`

**Interfaces:**
- Consumes: the repository layout from the spec.
- Produces: a minimal installable Python package layout and tracked locations for future configs, experiments, and tests.

- [ ] **Step 1: Create `pyproject.toml`**

Use a minimal setuptools `src/` layout, Python `>=3.11`, package name `mini-llm`, and a small `dev` optional dependency group containing `pytest>=8`. Do not add PyTorch yet; it should be added when the first model implementation requires it.

- [ ] **Step 2: Create `.gitignore`**

Ignore Python caches, virtual environments, build artifacts, common editor files, datasets, checkpoints/model weights, local artifacts, large profiler outputs, and large local serving traces while leaving `experiments/` tracked.

- [ ] **Step 3: Create package and tracked empty directories**

Create `src/minillm/__init__.py` with only a package docstring and create `.gitkeep` files in `configs/`, `experiments/`, and `tests/`.

- [ ] **Step 4: Validate TOML structure by inspection**

Expected sections include `[build-system]`, `[project]`, `[project.optional-dependencies]`, and setuptools package discovery under `src`.

---

### Task 4: Rewrite README around the actual training+inference workflow

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: rule/memory/package structure from Tasks 1–3.
- Produces: a concise human-facing entry point for the project.

- [ ] **Step 1: Replace placeholder README**

Describe:

- project purpose: learn LLM training, inference, serving, and GPU systems by building and measuring a real mini LLM, then evolving it into a mini inference engine;
- training-side optimization topics at a high level;
- inference-side optimization topics at a high level, including KV Cache, prefill/decode, batching/scheduling, cache management, speculative decoding, and later vLLM/SGLang;
- local machine / GitHub / agent responsibilities;
- current initialization stage;
- continuity via `PROJECT_MEMORY.md`, `AGENTS.md`, and `skills/mini-llm-learning/SKILL.md`;
- incremental-growth principle rather than generating a large codebase upfront.

- [ ] **Step 2: Add minimal local bootstrap commands**

Include only repository sync and editable development install commands that are valid at this stage; do not imply GPU/model/inference execution exists yet.

---

### Task 5: Verify the scaffold against the spec

**Files:**
- Read: all files created/modified above.

**Interfaces:**
- Consumes: completed scaffold.
- Produces: evidence that repository initialization matches the approved design.

- [ ] **Step 1: Verify expected repository paths exist**

Expected tracked paths:

```text
AGENTS.md
PROJECT_MEMORY.md
README.md
.gitignore
pyproject.toml
configs/.gitkeep
experiments/.gitkeep
skills/mini-llm-learning/SKILL.md
src/minillm/__init__.py
tests/.gitkeep
```

- [ ] **Step 2: Verify prohibited premature implementation is absent**

Confirm no `attention.py`, `rope.py`, `train.py`, Triton kernel, benchmark script, KV-cache module, inference scheduler, serving engine, vLLM/SGLang integration, or CI file was created by this initialization.

- [ ] **Step 3: Verify continuity wiring**

Confirm `AGENTS.md` points to `PROJECT_MEMORY.md` and the learning skill, and README explains the same separation without duplicating their full contents. Confirm the learning skill explicitly covers both training and inference optimization.

- [ ] **Step 4: Report local validation commands**

After the user pulls locally, run:

```bash
python -c "import pathlib; print(pathlib.Path('PROJECT_MEMORY.md').exists())"
python -m pip install -e ".[dev]"
python -c "import minillm; print(minillm.__name__)"
```

Expected: `True`, editable install succeeds, and final command prints `minillm`. Do not claim these local commands ran from GitHub-side editing.
