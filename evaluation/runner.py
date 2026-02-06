import csv
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import core.synapse_brain as synapse_module
from core.synapse_brain import SynapseBrain

from .ablation import AblationProfile
from .data import load_jsonl_dataset
from .metrics import (
    citation_metrics,
    contradiction_hit_rate,
    keyword_recall,
    refusal_accuracy,
    retrieval_metrics,
    safe_mean,
)


@dataclass
class EvaluationConfig:
    dataset_path: str
    outdir: str = "evaluation/results"
    profiles: Optional[List[str]] = None
    top_k: int = 10
    max_items: Optional[int] = None
    isolate_memory: bool = True
    skip_setup_upload: bool = False


def _resolve_doc_path(doc: str, dataset_path: Path) -> Path:
    doc_path = Path(doc)
    if doc_path.is_absolute() and doc_path.exists():
        return doc_path

    candidates = [
        Path.cwd() / doc,
        dataset_path.parent / doc,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Upload document not found: {doc}")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    contradiction_vals = [i["contradiction_hit_rate"] for i in items if i["contradiction_hit_rate"] is not None]

    return {
        "samples_total": len(items),
        "samples_with_error": sum(1 for i in items if i.get("error")),
        "retrieval_recall_at_k_mean": safe_mean([i["retrieval_recall_at_k"] for i in items]),
        "retrieval_mrr_mean": safe_mean([i["retrieval_mrr"] for i in items]),
        "answer_keyword_recall_mean": safe_mean([i["answer_keyword_recall"] for i in items]),
        "sentence_citation_coverage_mean": safe_mean([i["sentence_citation_coverage"] for i in items]),
        "evidence_citation_coverage_mean": safe_mean([i["evidence_citation_coverage"] for i in items]),
        "refusal_accuracy_mean": safe_mean([i["refusal_accuracy"] for i in items]),
        "contradiction_hit_rate_mean": safe_mean(contradiction_vals),
        "latency_seconds_mean": safe_mean([i["latency_seconds"] for i in items]),
    }


def _run_single_profile(
    config: EvaluationConfig,
    dataset: List[Dict[str, Any]],
    dataset_path: Path,
    profile: str,
    outdir: Path,
) -> Dict[str, Any]:
    profile_dir = outdir / profile
    profile_dir.mkdir(parents=True, exist_ok=True)

    memory_dir = profile_dir / "memory"
    if config.isolate_memory:
        if memory_dir.exists():
            shutil.rmtree(memory_dir)
        memory_dir.mkdir(parents=True, exist_ok=True)

    synapse_module.MEMORY_DIR = str(memory_dir if config.isolate_memory else synapse_module.MEMORY_DIR)

    items: List[Dict[str, Any]] = []
    item_path = profile_dir / "items.jsonl"

    brain = SynapseBrain()
    try:
        preflight = brain.preflight_check()

        with AblationProfile(brain, profile):
            upload_docs: List[Path] = []
            seen = set()
            for sample in dataset:
                for doc in sample.get("setup_upload_docs", []):
                    resolved = _resolve_doc_path(doc, dataset_path)
                    if str(resolved) not in seen:
                        seen.add(str(resolved))
                        upload_docs.append(resolved)

            setup_uploads = []
            if config.skip_setup_upload:
                for doc in upload_docs:
                    setup_uploads.append({
                        "doc": str(doc),
                        "skipped": True,
                        "error": None,
                        "memcells_created": 0,
                        "episodes_created": 0,
                    })
            else:
                for doc in upload_docs:
                    res = brain.upload(str(doc))
                    setup_uploads.append({
                        "doc": str(doc),
                        "skipped": False,
                        "error": res.get("error"),
                        "memcells_created": res.get("memcells_created", 0),
                        "episodes_created": res.get("episodes_created", 0),
                    })

            with item_path.open("w", encoding="utf-8") as item_file:
                for sample in dataset:
                    q = sample["question"]
                    retrieval_query = sample.get("retrieval_query") or q
                    top_k = int(config.top_k)

                    retrieval_error = None
                    retrieved = []
                    retrieval_start = time.time()
                    try:
                        retrieved = brain.memory.retrieve(
                            retrieval_query,
                            top_k=top_k,
                            strategy="agentic",
                            allowed_types=brain.qa.EVIDENCE_TYPES,
                        )
                    except Exception as e:
                        retrieval_error = str(e)
                    retrieval_elapsed = time.time() - retrieval_start

                    ret = retrieval_metrics(
                        retrieved=retrieved,
                        gold_substrings=sample.get("gold_evidence_substrings", []),
                        top_k=top_k,
                    )

                    ask_start = time.time()
                    ask_result = brain.ask(q)
                    ask_elapsed = time.time() - ask_start

                    answer = ask_result.get("answer", "") if isinstance(ask_result, dict) else ""
                    ask_error = ask_result.get("error") if isinstance(ask_result, dict) else "ask_result_not_dict"

                    citation = citation_metrics((ask_result.get("evidence_contract") or {}) if isinstance(ask_result, dict) else {})

                    contradiction_val: Optional[float] = None
                    if sample.get("expect_contradiction", False):
                        contradiction_retrieved = brain.memory.retrieve(
                            retrieval_query,
                            top_k=top_k,
                            strategy="hybrid",
                            allowed_types=None,
                        )
                        contradiction_val = contradiction_hit_rate(contradiction_retrieved, top_k=top_k)

                    item_result = {
                        "id": sample.get("id"),
                        "profile": profile,
                        "question": q,
                        "retrieval_query": retrieval_query,
                        "error": ask_error or retrieval_error,
                        "retrieval_error": retrieval_error,
                        "ask_error": ask_error,
                        "memories_used": int((ask_result or {}).get("memories_used", 0) or 0),
                        "mode": (ask_result or {}).get("mode", ""),
                        "retrieval_recall_at_k": ret["recall_at_k"],
                        "retrieval_mrr": ret["mrr"],
                        "answer_keyword_recall": keyword_recall(answer, sample.get("gold_answer_keywords", [])),
                        "sentence_citation_coverage": citation["sentence_citation_coverage"],
                        "evidence_citation_coverage": citation["evidence_citation_coverage"],
                        "refusal_accuracy": refusal_accuracy(sample.get("expect_refusal", False), answer),
                        "contradiction_hit_rate": contradiction_val,
                        "latency_seconds": float((ask_result or {}).get("elapsed", ask_elapsed) or ask_elapsed),
                        "retrieval_latency_seconds": retrieval_elapsed,
                        "timestamp": datetime.now().isoformat(),
                    }
                    items.append(item_result)
                    item_file.write(json.dumps(item_result, ensure_ascii=True) + "\n")

        aggregate = _aggregate_items(items)
        summary = {
            "profile": profile,
            "timestamp": datetime.now().isoformat(),
            "dataset_path": str(dataset_path),
            "top_k": config.top_k,
            "max_items": config.max_items,
            "isolate_memory": config.isolate_memory,
            "skip_setup_upload": config.skip_setup_upload,
            "memory_dir": str(memory_dir if config.isolate_memory else synapse_module.MEMORY_DIR),
            "model": brain.llm.model_name,
            "embedding_model": brain.embedder.model_name,
            "preflight_ok": preflight.get("ok", False),
            "setup_uploads": setup_uploads,
            "aggregate": aggregate,
        }
        _write_json(profile_dir / "summary.json", summary)
        _write_summary_csv(profile_dir / "summary.csv", [{"profile": profile, **aggregate}])
        return summary
    finally:
        brain.close()


def run_evaluation(config: EvaluationConfig) -> Dict[str, Any]:
    dataset_path = Path(config.dataset_path).resolve()
    outdir = Path(config.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = load_jsonl_dataset(str(dataset_path), max_items=config.max_items)
    profiles = config.profiles or ["full"]

    summaries: List[Dict[str, Any]] = []
    for profile in profiles:
        summary = _run_single_profile(
            config=config,
            dataset=dataset,
            dataset_path=dataset_path,
            profile=profile,
            outdir=outdir,
        )
        summaries.append(summary)

    compact_rows = []
    for s in summaries:
        compact_rows.append({
            "profile": s["profile"],
            **s["aggregate"],
            "dataset_path": s["dataset_path"],
            "model": s["model"],
            "embedding_model": s["embedding_model"],
            "preflight_ok": s["preflight_ok"],
        })

    _write_summary_csv(outdir / "summary_all.csv", compact_rows)
    _write_json(outdir / "summary_all.json", {"runs": summaries})

    return {"runs": summaries, "summary_csv": str(outdir / "summary_all.csv")}


def example_profiles() -> List[str]:
    return ["full", "no_memory", "no_agentic", "no_purity", "no_hier_summary"]


def ensure_example_profiles(names: Sequence[str]) -> None:
    allowed = set(example_profiles())
    unknown = [n for n in names if n not in allowed]
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}. Allowed: {sorted(allowed)}")


__all__ = ["EvaluationConfig", "run_evaluation", "example_profiles", "ensure_example_profiles"]
