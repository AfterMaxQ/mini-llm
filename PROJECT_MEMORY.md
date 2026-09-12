# Project Memory

## Goal
Build one small but real decoder-only LLM project that supports both training-systems learning and inference-systems learning, with code, experiments, benchmarks, and profiler evidence rather than tutorial-only understanding.

## Current Stage
Repository scaffold is established and locally verified. The project is now defining Mini LLM v0.1 before implementing model code.

## Completed / Verified
- Repository rules exist in `AGENTS.md`.
- Teaching/engineering workflow exists in `skills/mini-llm-learning/SKILL.md`.
- Minimal Python/package, config, test, and experiment locations exist on GitHub.
- Training and inference are defined as equal project tracks.
- Local `git pull` completed successfully on `F:\mini-llm`.
- Local editable install `python -m pip install -e ".[dev]"` completed successfully with Python 3.11.
- Local import check `python -c "import minillm; print(minillm.__name__)"` returned `minillm`.
- No model implementation, training run, CUDA/Triton run, GPU benchmark, or profiler result has been verified yet.

## Current Understanding
The project should advance through small, absorbable, testable steps. Theory is introduced just in time for the current code or experiment. Training builds the model and training systems; the same model later evolves into a Mini Inference Engine for inference/serving optimization.

## Open Questions
- Mini LLM v0.1 model size and configuration.
- First learning-critical module to implement after the configuration is fixed.

## Current Work
Define Mini LLM v0.1's smallest useful decoder-only architecture and configuration.

## Next Step
Settle the initial `vocab_size`, `d_model`, layer count, attention head layout, KV-head layout, and context length, then implement the first minimal verifiable model module.
