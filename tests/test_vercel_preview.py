"""Tests for the dependency-free Vercel Python API."""
import unittest

from web.api.generate import generate_site


class VercelPreviewTests(unittest.TestCase):
    def test_generates_portfolio_files(self):
        result = generate_site("Create a developer portfolio")
        self.assertEqual(result["kind"], "portfolio")
        self.assertIn("index.html", result["files"])
        self.assertIn("styles.css", result["files"])
        self.assertIn("script.js", result["files"])
        self.assertIn("Build a portfolio", result["files"]["index.html"])

    def test_escapes_prompt_html(self):
        result = generate_site("<script>alert(1)</script>")
        self.assertNotIn("<script>alert(1)</script>", result["files"]["index.html"])
        self.assertIn("&lt;script&gt;", result["files"]["index.html"])

    def test_rejects_empty_prompt(self):
        with self.assertRaises(ValueError):
            generate_site("  ")

    def test_rejects_long_prompt(self):
        with self.assertRaises(ValueError):
            generate_site("x" * 1201)


if __name__ == "__main__":
    unittest.main()
