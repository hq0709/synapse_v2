import re
from typing import Any, Dict, List, Sequence, Tuple


REFUSAL_PATTERNS = [
    r"\bI (do not|don't) know\b",
    r"\binsufficient (evidence|information)\b",
    r"\bnot enough (evidence|information|data)\b",
    r"\bcannot determine\b",
    r"\b无法确定\b",
    r"\b证据不足\b",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def keyword_recall(answer: str, gold_keywords: Sequence[str]) -> float:
    if not gold_keywords:
        return 0.0
    ans = _normalize(answer)
    hits = 0
    for kw in gold_keywords:
        if _normalize(str(kw)) in ans:
            hits += 1
    return hits / len(gold_keywords)


def detect_refusal(answer: str) -> bool:
    text = answer or ""
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


def refusal_accuracy(expected_refusal: bool, answer: str) -> float:
    predicted = detect_refusal(answer)
    return 1.0 if predicted == expected_refusal else 0.0


def retrieval_metrics(
    retrieved: Sequence[Tuple[Any, float]],
    gold_substrings: Sequence[str],
    top_k: int,
) -> Dict[str, float]:
    top = list(retrieved[:top_k])
    contents = [_normalize(getattr(mem, "content", "")) for mem, _ in top]

    if not gold_substrings:
        return {
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "gold_hits": 0,
            "gold_total": 0,
            "first_hit_rank": 0,
        }

    normalized_gold = [_normalize(str(g)) for g in gold_substrings if str(g).strip()]
    if not normalized_gold:
        return {
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "gold_hits": 0,
            "gold_total": 0,
            "first_hit_rank": 0,
        }

    hit_ranks: List[int] = []
    gold_hits = 0
    for gold in normalized_gold:
        matched_rank = 0
        for idx, content in enumerate(contents, start=1):
            if gold in content:
                matched_rank = idx
                break
        if matched_rank > 0:
            gold_hits += 1
            hit_ranks.append(matched_rank)

    first_hit_rank = min(hit_ranks) if hit_ranks else 0
    mrr = (1.0 / first_hit_rank) if first_hit_rank else 0.0
    return {
        "recall_at_k": gold_hits / len(normalized_gold),
        "mrr": mrr,
        "gold_hits": float(gold_hits),
        "gold_total": float(len(normalized_gold)),
        "first_hit_rank": float(first_hit_rank),
    }


def citation_metrics(evidence_contract: Dict[str, Any]) -> Dict[str, float]:
    coverage = (evidence_contract or {}).get("citation_coverage") or {}
    sent_cited = float(coverage.get("sentences_with_citation", 0) or 0)
    sent_total = float(coverage.get("sentences_total", 0) or 0)
    ev_cited = float(coverage.get("evidence_items_cited", 0) or 0)
    ev_total = float(coverage.get("evidence_items_total", 0) or 0)

    sentence_coverage = (sent_cited / sent_total) if sent_total > 0 else 0.0
    evidence_coverage = (ev_cited / ev_total) if ev_total > 0 else 0.0

    return {
        "sentence_citation_coverage": sentence_coverage,
        "evidence_citation_coverage": evidence_coverage,
        "sentences_with_citation": sent_cited,
        "sentences_total": sent_total,
        "evidence_items_cited": ev_cited,
        "evidence_items_total": ev_total,
    }


def contradiction_hit_rate(retrieved: Sequence[Tuple[Any, float]], top_k: int) -> float:
    for mem, _ in list(retrieved[:top_k]):
        mem_type = getattr(getattr(mem, "memory_type", None), "value", "")
        if mem_type == "contradiction":
            return 1.0
    return 0.0


def safe_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
