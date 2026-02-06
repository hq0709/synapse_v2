from contextlib import AbstractContextManager
from types import MethodType
from typing import Any, Dict, Optional


class AblationProfile(AbstractContextManager):
    """Applies temporary evaluation ablations to a SynapseBrain instance."""

    def __init__(self, brain: Any, profile: str):
        self.brain = brain
        self.profile = profile
        self._attrs: Dict[Any, Dict[str, Any]] = {}
        self._missing = object()

    def __enter__(self):
        if self.profile == "full":
            return self
        if self.profile == "no_memory":
            self._patch_method(
                self.brain.memory,
                "retrieve",
                lambda _self, *_args, **_kwargs: [],
            )
            self._patch_method(
                self.brain.memory,
                "trace",
                lambda _self, query: {
                    "found": False,
                    "query": query,
                    "message": "Disabled by no_memory ablation",
                    "traces": [],
                    "related_episodes": [],
                },
            )
        elif self.profile == "no_agentic":
            def hybrid_only(memory_self, query, query_keywords, top_k, allowed_types=None):
                return memory_self._hybrid_retrieval(query, query_keywords, top_k, allowed_types)

            self._patch_method(self.brain.memory, "_agentic_retrieval", hybrid_only)
        elif self.profile == "no_purity":
            self._patch_attr(self.brain.qa, "EVIDENCE_TYPES", None)
            self._patch_attr(self.brain.explorer, "EVIDENCE_TYPES", None)
        elif self.profile == "no_hier_summary":
            def simple_summary(memory_self, chunks):
                return memory_self._build_coverage_excerpt(
                    chunks,
                    max_chunks=4,
                    chunk_chars=1200,
                    include_headers=True,
                )

            self._patch_method(
                self.brain.memory,
                "_build_hierarchical_document_summary",
                simple_summary,
            )
        else:
            raise ValueError(f"Unknown ablation profile: {self.profile}")
        return self

    def __exit__(self, exc_type, exc, tb):
        for obj, attrs in self._attrs.items():
            for key, old in attrs.items():
                if old is self._missing:
                    if hasattr(obj, key):
                        delattr(obj, key)
                else:
                    setattr(obj, key, old)
        self._attrs.clear()
        return False

    def _patch_attr(self, obj: Any, name: str, value: Any):
        old = getattr(obj, name) if hasattr(obj, name) else self._missing
        self._attrs.setdefault(obj, {})[name] = old
        setattr(obj, name, value)

    def _patch_method(self, obj: Any, name: str, fn):
        old = getattr(obj, name) if hasattr(obj, name) else self._missing
        self._attrs.setdefault(obj, {})[name] = old
        setattr(obj, name, MethodType(fn, obj))


def supported_profiles() -> Dict[str, str]:
    return {
        "full": "All modules enabled",
        "no_memory": "Disable retrieval memory access",
        "no_agentic": "Disable multi-round agentic retrieval",
        "no_purity": "Disable evidence type purity filtering",
        "no_hier_summary": "Disable hierarchical long-document summarization",
    }


def parse_profiles(raw: str) -> Optional[list]:
    if not raw:
        return None
    profiles = [p.strip() for p in raw.split(",") if p.strip()]
    if not profiles:
        return None
    valid = set(supported_profiles().keys())
    bad = [p for p in profiles if p not in valid]
    if bad:
        raise ValueError(f"Unknown profiles: {bad}. Valid: {sorted(valid)}")
    return profiles
