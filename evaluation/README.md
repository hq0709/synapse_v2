# Synapse Evaluation

This folder provides a reproducible evaluation pipeline for Synapse with ablation profiles.

## Dataset format (JSONL)

Each line is one evaluation sample.

Required fields:
- `id`: unique sample id
- `question`: user question to evaluate

Optional fields:
- `setup_upload_docs`: list of documents uploaded before evaluation
- `retrieval_query`: query used for retrieval metrics (defaults to `question`)
- `gold_answer_keywords`: expected answer keywords for keyword-recall metric
- `gold_evidence_substrings`: evidence strings expected to appear in top-k retrieval
- `expect_refusal`: whether the system should refuse due to insufficient evidence
- `expect_contradiction`: whether contradiction memories should be retrieved

Example dataset:
- `evaluation/datasets/sample_eval.jsonl`

## Profiles

- `full`: all modules enabled
- `no_memory`: retrieval disabled
- `no_agentic`: agentic retrieval disabled (hybrid only)
- `no_purity`: disables ATOMIC_FACT/EPISODE purity filtering
- `no_hier_summary`: disables hierarchical long-document summarization

## Run

```bash
python scripts/run_evaluation.py \
  --dataset evaluation/datasets/sample_eval.jsonl \
  --profiles full,no_memory,no_agentic,no_purity,no_hier_summary \
  --outdir evaluation/results
```

If your memory is already prepared and you want fast iteration:

```bash
python scripts/run_evaluation.py \
  --dataset evaluation/datasets/sample_eval.jsonl \
  --profiles full,no_agentic \
  --skip-setup-upload \
  --no-isolate-memory
```

## Outputs

- `evaluation/results/<profile>/items.jsonl`: per-sample metrics
- `evaluation/results/<profile>/summary.json`: profile summary + setup metadata
- `evaluation/results/<profile>/summary.csv`: single-row profile metrics
- `evaluation/results/summary_all.csv`: merged table across profiles
- `evaluation/results/summary_all.json`: full run payload

## Current metrics

- Retrieval: `Recall@k`, `MRR`
- Citation quality: sentence citation coverage, evidence citation coverage
- Answer quality: keyword recall
- Robustness: refusal accuracy, contradiction hit rate
- Runtime: per-question latency
