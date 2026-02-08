import unittest
from types import SimpleNamespace

from core.synapse_brain import ScientificExplorer


class _MemoryStub:
    def retrieve(self, *args, **kwargs):
        return []


class _LLMStub:
    def generate(self, prompt: str):
        return None


class EpistemicPolicyTests(unittest.TestCase):
    def setUp(self):
        self.explorer = ScientificExplorer(memory=_MemoryStub(), llm=_LLMStub())

    def test_entropy_bounds(self):
        high_entropy = self.explorer._belief_entropy([
            {"belief": 0.5},
            {"belief": 0.5},
        ])
        low_entropy = self.explorer._belief_entropy([
            {"belief": 0.95},
            {"belief": 0.05},
        ])
        self.assertGreaterEqual(high_entropy, 0.0)
        self.assertLessEqual(high_entropy, 1.0)
        self.assertGreater(high_entropy, low_entropy)

    def test_force_propose_with_small_hypothesis_space(self):
        state = {
            "hypotheses": [{"hypothesis": "h1", "belief": 0.5, "uncertainty": 0.7}],
            "knowledge": "",
            "experiments": [],
        }
        action, info = self.explorer._select_epistemic_tool(state, step=1, max_steps=4)
        self.assertEqual(action, "propose_hypotheses")
        self.assertIn("utility", info)

    def test_apply_relative_updates(self):
        hypotheses = [{"hypothesis": "A causes B", "belief": 0.5, "uncertainty": 0.6}]
        updates = [{"hypothesis": "A causes B", "support_delta": 0.2, "uncertainty_delta": -0.1}]
        self.explorer._apply_hypothesis_updates(hypotheses, updates)
        self.assertAlmostEqual(hypotheses[0]["belief"], 0.7)
        self.assertAlmostEqual(hypotheses[0]["uncertainty"], 0.5)

    def test_apply_absolute_updates(self):
        hypotheses = [{"hypothesis": "A causes B", "belief": 0.3, "uncertainty": 0.8}]
        updates = [{"hypothesis": "A causes B", "support_delta": 0.75, "uncertainty_delta": 0.2, "absolute": True}]
        self.explorer._apply_hypothesis_updates(hypotheses, updates, absolute=True)
        self.assertAlmostEqual(hypotheses[0]["belief"], 0.75)
        self.assertAlmostEqual(hypotheses[0]["uncertainty"], 0.2)


if __name__ == "__main__":
    unittest.main()
