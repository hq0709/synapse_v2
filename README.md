# Synapse 2.0

Memory-driven scientific exploration CLI inspired by EverMemOS and OpenScholar.

## Quick start

```bash
# one-click setup + run
bash run.sh
```

## Configuration

Secrets are loaded from `.env` (git-ignored).

```bash
cp .env.example .env
# edit GEMINI_API_KEY in .env
```

Environment variables:
- `GEMINI_API_KEY`: enables Gemini generation and embeddings.
- `GEMINI_MODEL`: model name (default `gemini-3-pro-preview`).
- `GEMINI_TEMPERATURE`: generation temperature (default `0.0` for reproducibility).
- `GEMINI_TOP_P`: nucleus sampling parameter.
- `GEMINI_TOP_K`: top-k sampling parameter.
- `GEMINI_MAX_OUTPUT_TOKENS`: output budget per call.
- `GEMINI_EMBEDDING_MODEL`: fixed embedding model for all runs (default `gemini-embedding-001`).
- `GEMINI_EMBEDDING_DIM`: embedding output dimensionality (default `768`).
- `SYNAPSE_AUTO_BACKFILL_EMBEDDINGS`: auto-embed legacy memories missing vectors at startup.
- `SYNAPSE_EXPLORER_MODE`: exploration mode (`epistemic_tools` or `legacy`).
- `SYNAPSE_EPISTEMIC_LAMBDA_COST`: cost penalty in epistemic tool utility.
- `SYNAPSE_EPISTEMIC_MU_RISK`: risk penalty in epistemic tool utility.
- `SYNAPSE_EPISTEMIC_STOP_ENTROPY`: stopping threshold for posterior entropy.
- `SYNAPSE_EPISTEMIC_COST_BUDGET`: average per-step cost budget in constrained optimization.
- `SYNAPSE_EPISTEMIC_RISK_BUDGET`: average per-step risk budget in constrained optimization.
- `SYNAPSE_EPISTEMIC_DUAL_STEP`: dual ascent step size for lambda/mu updates.
- `SYNAPSE_FALSIFICATION_WEIGHT`: relative weight for falsification-first hypothesis/experiment ordering.
- `SYNAPSE_PROJECT_HORIZON`: number of milestones tracked in long-horizon project progression.
- `SYNAPSE_COUNTERFACTUAL_BRANCHES`: max counterfactual scenarios generated per exploration step.
- `SYNAPSE_MEMORY_DIR`: persistent memory directory (default `.synapse_memory`).

## LLM requirement

Synapse now runs in strict model mode for scientific workflows:
- `upload`, `ask`, `explore`, and `trace` require a working LLM connection.
- Embeddings are required and pinned to one configured model (`GEMINI_EMBEDDING_MODEL`).
- No offline fallback is used for extraction, reasoning, retrieval planning, or exploration.

## Audit logs

Every run writes structured JSONL events to:
- `.synapse_memory/runs/run_<timestamp>.jsonl`

Each event contains query/topic metadata, citation coverage, hypothesis counts,
and runtime info for reproducibility and paper traceability.

## Core commands

```text
/memory upload <file>   Upload and extract memory
/memory trace <query>   Trace relevant memory graph
/explore <topic>        Multi-step scientific exploration
/status                 Runtime and memory stats
/selfcheck              Dependency and storage preflight checks
```

## Reliability checks

```bash
# no-network unit checks
python -m unittest tests/test_reliability.py
python -m unittest tests/test_evaluation_metrics.py

# online end-to-end smoke check (requires GEMINI_API_KEY)
python scripts/reliability_smoke.py
```

## Evaluation and ablation

```bash
python scripts/run_evaluation.py \
  --dataset evaluation/datasets/sample_eval.jsonl \
  --profiles full,no_memory,no_agentic,no_purity,no_hier_summary \
  --outdir evaluation/results
```

Fast iteration on existing memory:

```bash
python scripts/run_evaluation.py \
  --dataset evaluation/datasets/sample_eval.jsonl \
  --profiles full,no_agentic \
  --skip-setup-upload \
  --no-isolate-memory
```

Details: `evaluation/README.md`

## Notes

- QA evidence retrieval is now purity-filtered to `ATOMIC_FACT` and `EPISODE` only.
- Long documents are processed with chunked extraction and hierarchical summarization.
- FAISS is used by default for vector retrieval speed; if unavailable at runtime, retrieval falls back to in-memory cosine search.
- Scientific exploration now defaults to an epistemic tool policy that uses constrained optimization with dual ascent and KL-based information gain tracking.
- Exploration includes integrated project progression, counterfactual lab, causal graph updates, and protocol-grade experiment outputs.
- Experiment planning is falsification-first and emits structured protocol fields (null hypothesis, confound mitigation, analysis/stopping rules).
- Theory and formal objective: `EPISTEMIC_TOOL_POLICY.md`.
- Extended derivations and proof sketches: `THEORY_APPENDIX.md`.
