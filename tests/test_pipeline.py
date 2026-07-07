import unittest

from src.document_loader import load_resume_documents
from src.preprocess import clean_resume_text


class PipelineTests(unittest.TestCase):
    def test_clean_resume_text_removes_noise(self) -> None:
        noisy_text = """
        [Image]
        Page 1 of 1
        John Doe
        Skills
        Python, SQL, AWS
        """

        cleaned_text = clean_resume_text(noisy_text)

        self.assertNotIn("[Image]", cleaned_text)
        self.assertNotIn("Page 1 of 1", cleaned_text)
        self.assertIn("John Doe", cleaned_text)
        self.assertIn("Skills", cleaned_text)

    def test_load_resume_documents_adds_required_metadata(self) -> None:
        documents = load_resume_documents("resumes")

        self.assertGreaterEqual(len(documents), 1)
        first_document = documents[0]
        self.assertIn("person_name", first_document.metadata)
        self.assertIn("source", first_document.metadata)
        self.assertEqual(first_document.metadata["document_type"], "resume")


if __name__ == "__main__":
    unittest.main()
