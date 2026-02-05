# Nature Communications Readiness Checklist (Synapse 2.0)

This checklist tracks engineering and scientific evidence required for a credible submission.

## A. Core system claims

- [x] Strict LLM requirement for scientific workflows (`upload`, `ask`, `explore`)
- [x] Memory purity constraints for QA evidence retrieval
- [x] Long-document chunked extraction and hierarchical synthesis
- [x] Evidence contract outputs (assumptions/risks/citation coverage)
- [x] Hypothesis lifecycle outputs (ranked hypotheses + experiment plans)

## B. Reproducibility and traceability

- [x] `.env`-based secret management
- [x] Configurable model and generation controls
- [x] Structured run audit logs in JSONL
- [ ] Full environment lockfile and exact dependency pinning for paper artifact
- [ ] Frozen benchmark prompts and fixed judge model config for all experiments
- [ ] Reproducibility script that regenerates all table/figure metrics end-to-end

## C. Evaluation package (required for publication quality)

- [ ] Automatic benchmark runner (LoCoMo/LongMemEval/ScholarQABench-style tasks)
- [ ] Claim-level faithfulness metric (answer sentence -> supporting memory evidence)
- [ ] Citation precision/recall metrics and error taxonomy
- [ ] Contradiction detection benchmark (synthetic + real conflict sets)
- [ ] Hypothesis quality benchmark (testability/falsifiability judged rubric)
- [ ] Ablation suite:
  - [ ] no memory purity filtering
  - [ ] no evidence contract
  - [ ] no lifecycle scoring
  - [ ] no retrieval refinement

## D. Engineering quality

- [ ] Unit tests for parser robustness and retrieval filters
- [ ] Integration tests for upload -> ask -> explore workflows
- [ ] Regression tests for schema compatibility and migration
- [ ] Exception hardening (replace broad except blocks)
- [ ] JSON extraction hardening (replace greedy regex parsing with robust parser)

## E. Paper artifact completeness

- [ ] Dataset and split disclosure
- [ ] Failure cases + qualitative error analysis appendix
- [ ] Statistical significance and confidence intervals
- [ ] Human evaluation protocol for scientific usefulness
- [ ] Public reproducibility package (scripts, configs, logs, seed policy)

## Suggested milestone order

1. Build evaluation runner and metrics (Section C)
2. Add tests + parser hardening (Section D)
3. Freeze paper configs and produce reproducible result bundle (Section B/E)
4. Draft method and experiment sections from tracked artifacts
