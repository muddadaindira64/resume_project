import unittest
from pathlib import Path

from src.rag_chain import ResumeRagPipeline
from src.evaluation import evaluate_answer


class RagAppTests(unittest.TestCase):
    def test_pipeline_loads_existing_vector_store(self) -> None:
        pipeline = ResumeRagPipeline(vector_db_dir="vector_db")
        self.assertGreaterEqual(len(pipeline.documents), 1)
        self.assertIsNotNone(pipeline.index)

    def test_evaluation_metrics_return_expected_shape(self) -> None:
        metrics = evaluate_answer("Python and SQL", "Python, SQL, and ML")
        self.assertIn("similarity_score", metrics)
        self.assertIn("answer_relevance", metrics)
        self.assertIn("retrieval_accuracy", metrics)


if __name__ == "__main__":
    unittest.main()
