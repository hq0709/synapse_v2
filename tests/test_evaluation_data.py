import tempfile
import unittest
from pathlib import Path

from evaluation.data import load_jsonl_dataset


class EvaluationDataTests(unittest.TestCase):
    def test_load_jsonl_dataset_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dataset.jsonl"
            p.write_text('{"id":"a","question":"q"}\n', encoding="utf-8")
            rows = load_jsonl_dataset(str(p))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["retrieval_query"], "q")
            self.assertEqual(row["setup_upload_docs"], [])
            self.assertEqual(row["gold_answer_keywords"], [])
            self.assertEqual(row["gold_evidence_substrings"], [])
            self.assertFalse(row["expect_refusal"])
            self.assertFalse(row["expect_contradiction"])


if __name__ == "__main__":
    unittest.main()
