"""
Fast, local, offline tests for src/parser/pdf_parser.py.
No PDFs required for most tests - text-level functions are tested directly.
A couple of tests generate a throwaway PDF in-memory via PyMuPDF to cover
the extraction path end-to-end.

Run with:
    python -m pytest tests/test_parser.py -v
or:
    python -m unittest tests.test_parser -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.parser.pdf_parser import (
    clean_text,
    detect_sections,
    extract_skills,
    extract_contact_info,
    extract_text,
    process_resume,
    process_all_resumes,
    process_job_description,
    PROTECTED_TERMS_SAMPLE,
)

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


class TestCleanText(unittest.TestCase):
    def test_collapses_excess_whitespace(self):
        raw = "Hello    world\n\n\n\n\nBye"
        cleaned = clean_text(raw)
        self.assertNotIn("    ", cleaned)
        self.assertNotIn("\n\n\n", cleaned)

    def test_strips_control_characters(self):
        raw = "Hello\x00World\x0b"
        cleaned = clean_text(raw)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\x0b", cleaned)

    def test_preserves_technical_terms(self):
        raw = "Skills: " + ", ".join(PROTECTED_TERMS_SAMPLE)
        cleaned = clean_text(raw)
        for term in PROTECTED_TERMS_SAMPLE:
            self.assertIn(term, cleaned)

    def test_empty_input(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")


class TestDetectSections(unittest.TestCase):
    def test_basic_sections(self):
        text = (
            "Skills\n"
            "Python, React, SQL\n"
            "Experience\n"
            "Software Engineer at Acme\n"
            "Education\n"
            "B.Tech Computer Science\n"
        )
        sections = detect_sections(text)
        self.assertIn("Python", sections["skills"])
        self.assertIn("Acme", sections["experience"])
        self.assertIn("B.Tech", sections["education"])

    def test_missing_sections_return_empty_string(self):
        sections = detect_sections("Just some text with no headings.")
        for value in sections.values():
            self.assertEqual(value, "")

    def test_alias_variations(self):
        text = "Technical Skills\nDocker, Kubernetes\nProfessional Experience\nDid stuff\n"
        sections = detect_sections(text)
        self.assertIn("Docker", sections["skills"])
        self.assertIn("Did stuff", sections["experience"])

    def test_empty_input_does_not_crash(self):
        sections = detect_sections("")
        self.assertEqual(sections["skills"], "")


class TestExtractSkills(unittest.TestCase):
    def test_finds_known_skills(self):
        text = "Experienced in Python, React.js, Node.js and AWS."
        skills = extract_skills(text)
        self.assertIn("Python", skills)
        self.assertIn("React", skills)
        self.assertIn("Node.js", skills)
        self.assertIn("AWS", skills)

    def test_normalizes_aliases(self):
        text = "Built apps with NodeJS and ReactJS, deployed on GCP."
        skills = extract_skills(text)
        self.assertIn("Node.js", skills)
        self.assertIn("React", skills)
        self.assertIn("GCP", skills)
        # Aliases should not appear as separate entries.
        self.assertNotIn("NodeJS", skills)
        self.assertNotIn("ReactJS", skills)

    def test_avoids_false_positive_substring_matches(self):
        # "c" must not match inside "vaccine" / "certificate"; "r" (not even
        # a tracked skill) and similar single-letter traps are covered by
        # the word-boundary logic used for "C".
        text = "Received a vaccine and a certificate for accessibility training."
        skills = extract_skills(text)
        self.assertNotIn("C", skills)

    def test_c_plus_plus_and_c_sharp_detected_distinctly(self):
        text = "Proficient in C++ and C# as well as plain C."
        skills = extract_skills(text)
        self.assertIn("C++", skills)
        self.assertIn("C#", skills)
        self.assertIn("C", skills)

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(extract_skills(""), [])


class TestExtractContactInfo(unittest.TestCase):
    def test_extracts_email(self):
        text = "Jane Doe\nEmail: jane.doe@example.com\nSkills\nPython"
        contact = extract_contact_info(text)
        self.assertEqual(contact["email"], "jane.doe@example.com")

    def test_extracts_indian_phone_number(self):
        text = "Jane Doe\nPhone: +91 98765 43210\nSkills\nPython"
        contact = extract_contact_info(text)
        self.assertIsNotNone(contact["phone"])
        digits = "".join(ch for ch in contact["phone"] if ch.isdigit())
        self.assertGreaterEqual(len(digits), 10)

    def test_missing_fields_return_none(self):
        text = "Skills\nPython, SQL"
        contact = extract_contact_info(text)
        self.assertIsNone(contact["email"])

    def test_empty_input_does_not_crash(self):
        contact = extract_contact_info("")
        self.assertIsNone(contact["name"])
        self.assertIsNone(contact["email"])
        self.assertIsNone(contact["phone"])


@unittest.skipUnless(HAS_FITZ, "PyMuPDF not installed")
class TestExtractTextAndProcessResume(unittest.TestCase):
    def _make_pdf(self, text, path):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), text)
        doc.save(path)
        doc.close()

    def test_extract_text_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = os.path.join(tmp, "sample.pdf")
            self._make_pdf("Jane Doe\njane@example.com\nSkills\nPython, AWS", pdf_path)
            result = extract_text(pdf_path)
            self.assertEqual(result["status"], "success")
            self.assertIn("Python", result["text"])

    def test_extract_text_missing_file(self):
        result = extract_text("/nonexistent/path/does_not_exist.pdf")
        self.assertEqual(result["status"], "error")
        self.assertIsNotNone(result["error"])

    def test_extract_text_blank_page_reports_no_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = os.path.join(tmp, "blank.pdf")
            doc = fitz.open()
            doc.new_page()
            doc.save(pdf_path)
            doc.close()
            result = extract_text(pdf_path)
            self.assertEqual(result["status"], "no_text")

    def test_process_resume_success_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = os.path.join(tmp, "resume_001.pdf")
            self._make_pdf(
                "Jane Doe\njane.doe@example.com\n"
                "Skills\nPython, React, AWS\n"
                "Experience\nSoftware Engineer\n"
                "Education\nB.Tech CS\n",
                pdf_path,
            )
            result = process_resume(pdf_path)
            self.assertEqual(result["parsing_status"], "success")
            self.assertEqual(result["filename"], "resume_001.pdf")
            self.assertIn("Python", result["skills"])
            self.assertIn("raw_text", result)
            self.assertIn("sections", result)
            self.assertIsNone(result["error"])

    def test_process_resume_missing_file_reports_error_not_crash(self):
        result = process_resume("/nonexistent/resume.pdf")
        self.assertEqual(result["parsing_status"], "error")
        self.assertIsNotNone(result["error"])

    def test_process_all_resumes_continues_after_bad_pdf(self):
        with tempfile.TemporaryDirectory() as resumes_dir:
            good_path = os.path.join(resumes_dir, "good.pdf")
            self._make_pdf("Python, AWS, Docker", good_path)

            bad_path = os.path.join(resumes_dir, "bad.pdf")
            with open(bad_path, "w") as f:
                f.write("this is not a real pdf")

            with tempfile.TemporaryDirectory() as out_dir:
                output_path = os.path.join(out_dir, "resumes.json")
                results = process_all_resumes(resumes_dir=resumes_dir, output_path=output_path)

                statuses = {r["filename"]: r["parsing_status"] for r in results}
                self.assertEqual(statuses["good.pdf"], "success")
                self.assertEqual(statuses["bad.pdf"], "error")
                self.assertTrue(os.path.isfile(output_path))


class TestProcessJobDescription(unittest.TestCase):
    def test_missing_file_reports_error_not_crash(self):
        result = process_job_description("/nonexistent/job_description.txt")
        self.assertEqual(result["raw_text"], "")
        self.assertEqual(result["skills"], [])
        self.assertIsNotNone(result["error"])

    def test_reads_and_extracts_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            jd_path = os.path.join(tmp, "jd.txt")
            with open(jd_path, "w", encoding="utf-8") as f:
                f.write("Looking for a Python developer with React and AWS experience.")
            result = process_job_description(jd_path)
            self.assertIsNone(result["error"])
            self.assertIn("Python", result["skills"])
            self.assertIn("React", result["skills"])
            self.assertIn("AWS", result["skills"])


if __name__ == "__main__":
    unittest.main()
