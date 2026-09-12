# Project Memory

## Goal
Build one small but real decoder-only LLM project that supports both training-systems learning and inference-systems learning, with code, experiments, benchmarks, and profiler evidence rather than tutorial-only understanding.

## Current Stage
Repository scaffold is locally verified. Mini LLM v0.1's baseline configuration is fixed, and the project is ready to begin the first learning-critical model module after local config tests are verified.

## Completed / Verified
- Repository rules exist in `AGENTS.md`.
- Teaching/engineering workflow exists in `skills/mini-llm-learning/SKILL.md`.
- Minimal Python/package, config, test, and experiment locations exist on GitHub.
- Training and inference are defined as equal project tracks.
- Local `git pull` completed successfully on `F:\mini-llm`.
- Local editable install `python -m pip install -e ".[dev]"` completed successfully with Python 3.11.
- Local import check `python -c "import minillm; print(minillm.__name__)"` returned `minillm`.
- Mini LLM v0.1 baseline is `vocab_size=8192`, `d_model=512`, `n_layers=8`, `n_heads=8`, `n_kv_heads=2`, `context_length=1024`.
- `ModelConfig` and its initial contract tests are committed on GitHub; an isolated CPU sandbox run passed all 3 config tests.
- No model forward path, training run, CUDA/Triton run, GPU benchmark, or profiler result has been verified yet.

## Current Understanding
The shared model width is 512. With 8 query heads, `head_dim=64`; with 2 KV heads, each KV head serves 4 query heads. This GQA choice is intentional so the same architecture later exposes meaningful KV-cache and serving trade-offs.

## Open Questions
- Local Windows verification of the new config tests.
- First learning-critical model module to implement after config verification.

## Current Work
Verify the committed Mini LLM v0.1 configuration on the user's local repository.

## Next Step
Pull the config commit and run the config tests locally. If they pass, begin one model component at a time, starting with the simplest component that establishes the `[B, T] -> [B, T, d_model]` model data path.
