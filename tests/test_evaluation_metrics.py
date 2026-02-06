import unittest
from types import SimpleNamespace

from evaluation.metrics import (
    citation_metrics,
    contradiction_hit_rate,
    keyword_recall,
    refusal_accuracy,
    retrieval_metrics,
)


class EvaluationMetricsTests(unittest.TestCase):
    def test_keyword_recall(self):
        answer = "Sulfide electrolytes reached 12.4 mS/cm conductivity."
        score = keyword_recall(answer, ["sulfide", "12.4 mS/cm", "oxide"])
        self.assertAlmostEqual(score, 2 / 3)

    def test_retrieval_metrics(self):
        mem1 = SimpleNamespace(content="interface resistance remains the primary bottleneck")
        mem2 = SimpleNamespace(content="sulfide electrolytes demonstrated ionic conductivity of 12.4 mS/cm")
        retrieved = [(mem1, 0.9), (mem2, 0.8)]
        m = retrieval_metrics(
            retrieved,
            ["12.4 mS/cm", "primary bottleneck"],
            top_k=2,
        )
        self.assertAlmostEqual(m["recall_at_k"], 1.0)
        self.assertAlmostEqual(m["mrr"], 1.0)

    def test_citation_metrics(self):
        contract = {
            "citation_coverage": {
                "sentences_with_citation": 3,
                "sentences_total": 4,
                "evidence_items_cited": 2,
                "evidence_items_total": 5,
            }
        }
        m = citation_metrics(contract)
        self.assertAlmostEqual(m["sentence_citation_coverage"], 0.75)
        self.assertAlmostEqual(m["evidence_citation_coverage"], 0.4)

    def test_refusal_accuracy(self):
        self.assertEqual(refusal_accuracy(True, "I do not know based on current evidence."), 1.0)
        self.assertEqual(refusal_accuracy(False, "I do not know."), 0.0)

    def test_contradiction_hit_rate(self):
        contradiction = SimpleNamespace(memory_type=SimpleNamespace(value="contradiction"))
        fact = SimpleNamespace(memory_type=SimpleNamespace(value="atomic_fact"))
        self.assertEqual(contradiction_hit_rate([(fact, 0.7), (contradiction, 0.6)], top_k=2), 1.0)
        self.assertEqual(contradiction_hit_rate([(fact, 0.7)], top_k=1), 0.0)


if __name__ == "__main__":
    unittest.main()
