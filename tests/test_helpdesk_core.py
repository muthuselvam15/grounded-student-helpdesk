import tempfile
import unittest
from pathlib import Path

from helpdesk_core import JsonTicketStore, cosine_similarity, retrieve_policy


class CoreTests(unittest.TestCase):
    def test_cosine_similarity(self):
        scores = cosine_similarity([1, 0], [[1, 0], [0, 1], [0, 0]])
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 0.0)
        self.assertAlmostEqual(scores[2], 0.0)

    def test_dimension_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1, 0], [[1]])

    def test_retrieve_policy_returns_grounded_match(self):
        policies = [{"title": "Attendance", "text": "Approved attendance wording."}]
        result = retrieve_policy("attendance", policies, [[1.0, 0.0]],
                                 lambda _query: [1.0, 0.0], threshold=0.7)
        self.assertTrue(result["found"])
        self.assertEqual(result["policy"]["title"], "Attendance")

    def test_retrieve_policy_abstains_below_threshold(self):
        policies = [{"title": "Attendance", "text": "Approved attendance wording."}]
        result = retrieve_policy("unrelated", policies, [[1.0, 0.0]],
                                 lambda _query: [0.0, 1.0], threshold=0.7)
        self.assertFalse(result["found"])
        self.assertIsNone(result["policy"])

    def test_ticket_persists_between_store_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tickets.json"
            ticket = JsonTicketStore(path).create("Question?", "No policy match")
            loaded = JsonTicketStore(path).list_all()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["id"], ticket["id"])
            self.assertEqual(loaded[0]["status"], "open")

    def test_ticket_rejects_empty_question(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                JsonTicketStore(Path(directory) / "tickets.json").create(" ", "reason")


if __name__ == "__main__":
    unittest.main()
