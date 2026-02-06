#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from .runner import EvaluationConfig, ensure_example_profiles, run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Synapse evaluation and ablation experiments.")
    parser.add_argument(
        "--dataset",
        default="evaluation/datasets/sample_eval.jsonl",
        help="Path to JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--outdir",
        default="evaluation/results",
        help="Directory for JSONL/JSON/CSV outputs.",
    )
    parser.add_argument(
        "--profiles",
        default="full",
        help="Comma-separated profiles: full,no_memory,no_agentic,no_purity,no_hier_summary",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for retrieval metrics.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap on evaluated items.")
    parser.add_argument(
        "--no-isolate-memory",
        action="store_true",
        help="Reuse current SYNAPSE_MEMORY_DIR instead of creating profile-local memory dirs.",
    )
    parser.add_argument(
        "--skip-setup-upload",
        action="store_true",
        help="Skip setup_upload_docs from dataset (useful when memory is already prepared).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    ensure_example_profiles(profiles)

    config = EvaluationConfig(
        dataset_path=args.dataset,
        outdir=args.outdir,
        profiles=profiles,
        top_k=args.top_k,
        max_items=args.max_items,
        isolate_memory=not args.no_isolate_memory,
        skip_setup_upload=args.skip_setup_upload,
    )

    result = run_evaluation(config)

    print(json.dumps(result, ensure_ascii=True, indent=2))
    print(f"Summary CSV: {Path(result['summary_csv']).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
