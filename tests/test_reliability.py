import tempfile
import unittest
from types import SimpleNamespace

from core.synapse_brain import (
    BM25,
    EMBEDDINGS_REQUIRED_ERROR,
    EmbeddingProvider,
    LLM_REQUIRED_ERROR,
    SynapseBrain,
    _extract_first_json,
)


class JsonExtractionTests(unittest.TestCase):
    def test_extract_json_from_fenced_block(self):
        text = """```json
{"a": 1, "b": [2, 3]}
```"""
        data = _extract_first_json(text, dict)
        self.assertEqual(data, {"a": 1, "b": [2, 3]})

    def test_extract_first_json_from_noisy_text(self):
        text = "prefix noise {not json} and then [1, 2, 3] trailing"
        data = _extract_first_json(text, list)
        self.assertEqual(data, [1, 2, 3])

    def test_invalid_json_returns_none(self):
        self.assertIsNone(_extract_first_json("not-json-at-all", dict))


class ServiceGateTests(unittest.TestCase):
    @staticmethod
    def _make_brain_stub(llm_available=True, emb_available=True, emb_error=None):
        brain = SynapseBrain.__new__(SynapseBrain)
        brain.llm = SimpleNamespace(is_available=llm_available, model_name="gemini-test", last_error=None)
        brain.embedder = SimpleNamespace(
            available=emb_available,
            model_name="gemini-embedding-001",
            error_message=emb_error,
        )
        brain.memory = SimpleNamespace(
            faiss_vectors=SimpleNamespace(index=None, size=0),
            vectors=SimpleNamespace(embeddings={}),
        )
        brain.explorer = SimpleNamespace(
            mode="epistemic_tools",
            cost_budget=0.28,
            risk_budget=0.12,
            dual_step=0.08,
            falsification_weight=0.68,
            project_horizon=3,
            counterfactual_branches=3,
        )
        return brain

    def test_service_gate_rejects_missing_llm(self):
        brain = self._make_brain_stub(llm_available=False, emb_available=True)
        err = brain._ensure_services_available()
        self.assertEqual(err, {"error": LLM_REQUIRED_ERROR})

    def test_service_gate_rejects_missing_embedding(self):
        brain = self._make_brain_stub(llm_available=True, emb_available=False, emb_error="quota exceeded")
        err = brain._ensure_services_available()
        self.assertIsNotNone(err)
        self.assertIn(EMBEDDINGS_REQUIRED_ERROR, err["error"])
        self.assertIn("quota exceeded", err["error"])

    def test_preflight_ok_when_deps_and_paths_ready(self):
        brain = self._make_brain_stub(llm_available=True, emb_available=True)
        with tempfile.TemporaryDirectory() as td:
            run_dir = f"{td}/runs"
            import os
            os.makedirs(run_dir, exist_ok=True)
            brain.memory_dir = td
            brain.run_log_path = f"{run_dir}/run_test.jsonl"
            checks = brain.preflight_check()
        self.assertTrue(checks["ok"])


class RetrievalPrimitiveTests(unittest.TestCase):
    def test_bm25_remove_document(self):
        bm = BM25()
        bm.add_document("a", ["alpha", "beta", "beta"])
        bm.add_document("b", ["alpha", "gamma"])
        self.assertEqual(bm.corpus_size, 2)
        bm.remove_document("a")
        self.assertEqual(bm.corpus_size, 1)
        self.assertNotIn("a", bm.doc_keywords)


class EmbeddingErrorPolicyTests(unittest.TestCase):
    def test_fatal_embedding_error_detection(self):
        self.assertTrue(EmbeddingProvider._is_fatal_error("404 NOT_FOUND model is not found"))
        self.assertTrue(EmbeddingProvider._is_fatal_error("401 unauthorized api key"))

    def test_transient_embedding_error_detection(self):
        self.assertFalse(EmbeddingProvider._is_fatal_error("deadline exceeded timeout"))
        self.assertFalse(EmbeddingProvider._is_fatal_error("temporary network error"))


if __name__ == "__main__":
    unittest.main()
