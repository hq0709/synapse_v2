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
        hypotheses = [
            {"hypothesis": "A causes B", "belief": 0.5, "uncertainty": 0.6},
            {"hypothesis": "C causes D", "belief": 0.5, "uncertainty": 0.6},
        ]
        before = hypotheses[0]["belief"]
        updates = [{"hypothesis": "A causes B", "support_delta": 0.2, "uncertainty_delta": -0.1}]
        self.explorer._apply_hypothesis_updates(hypotheses, updates)
        self.assertGreater(hypotheses[0]["belief"], before)
        self.assertAlmostEqual(hypotheses[0]["uncertainty"], 0.5)
        self.assertAlmostEqual(sum(h["belief"] for h in hypotheses), 1.0, places=6)

    def test_apply_absolute_updates(self):
        hypotheses = [{"hypothesis": "A causes B", "belief": 0.3, "uncertainty": 0.8}]
        updates = [{"hypothesis": "A causes B", "support_delta": 0.75, "uncertainty_delta": 0.2, "absolute": True}]
        self.explorer._apply_hypothesis_updates(hypotheses, updates, absolute=True)
        self.assertAlmostEqual(hypotheses[0]["belief"], 1.0)
        self.assertAlmostEqual(hypotheses[0]["uncertainty"], 0.2)

    def test_kl_divergence_non_negative(self):
        prior = [0.5, 0.5]
        posterior = [0.8, 0.2]
        self.assertGreaterEqual(self.explorer._kl_divergence(posterior, prior), 0.0)

    def test_bayes_update_monotonic(self):
        up = self.explorer._bayes_update_from_delta(0.5, 0.2)
        down = self.explorer._bayes_update_from_delta(0.5, -0.2)
        self.assertGreater(up, 0.5)
        self.assertLess(down, 0.5)

    def test_dual_variables_affect_utility(self):
        state_low_penalty = {
            "hypotheses": [
                {"hypothesis": "h1", "belief": 0.5, "uncertainty": 0.6},
                {"hypothesis": "h2", "belief": 0.5, "uncertainty": 0.6},
            ],
            "experiments": [],
            "dual_lambda": 0.1,
            "dual_mu": 0.1,
        }
        state_high_penalty = {
            **state_low_penalty,
            "dual_lambda": 2.0,
            "dual_mu": 2.0,
        }
        u_low = self.explorer._estimate_tool_utility(state_low_penalty, "retrieve_evidence", step=0, max_steps=5)["utility"]
        u_high = self.explorer._estimate_tool_utility(state_high_penalty, "retrieve_evidence", step=0, max_steps=5)["utility"]
        self.assertGreater(u_low, u_high)


if __name__ == "__main__":
    unittest.main()
