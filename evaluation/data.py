import json
from pathlib import Path
from typing import Any, Dict, List, Optional


REQUIRED_FIELDS = {"id", "question"}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def load_jsonl_dataset(path: str, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {lineno}: {e}") from e

            missing = [k for k in REQUIRED_FIELDS if k not in item]
            if missing:
                raise ValueError(f"Line {lineno}: missing required fields {missing}")

            item.setdefault("retrieval_query", item.get("question", ""))
            item["setup_upload_docs"] = _as_list(item.get("setup_upload_docs"))
            item["gold_answer_keywords"] = [str(x) for x in _as_list(item.get("gold_answer_keywords"))]
            item["gold_evidence_substrings"] = [str(x) for x in _as_list(item.get("gold_evidence_substrings"))]
            item["expect_refusal"] = bool(item.get("expect_refusal", False))
            item["expect_contradiction"] = bool(item.get("expect_contradiction", False))
            rows.append(item)

            if max_items is not None and len(rows) >= max_items:
                break

    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    return rows
