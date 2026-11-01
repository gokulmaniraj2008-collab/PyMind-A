from pathlib import Path

import pytest

from pymind.website_builder import generate_website, write_website


def test_generates_responsive_business_site():
    project = generate_website("Create a website for my local design business")
    assert project.kind == "business"
    assert {"index.html", "styles.css", "script.js", "README.txt"} == set(project.files)
    assert "viewport" in project.files["index.html"]
    assert "@media" in project.files["styles.css"]


def test_detects_portfolio_and_3d_site():
    assert generate_website("Build a developer portfolio").kind == "portfolio"
    project = generate_website("Create a 3D car model viewer")
    assert project.kind == "3d"
    assert "THREE.WebGLRenderer" in project.files["script.js"]
    assert "three.js" in project.files["index.html"]


def test_escapes_user_content_and_rejects_empty_prompt():
    project = generate_website('<script>alert("x")</script>')
    assert "<script>alert" not in project.files["index.html"]
    assert "&lt;script&gt;" in project.files["index.html"]
    with pytest.raises(ValueError):
        generate_website("  ")


def test_writes_expected_files(tmp_path: Path):
    project = generate_website("Create a shop website")
    target = write_website(project, tmp_path / "generated")
    assert (target / "index.html").is_file()
    assert (target / "styles.css").is_file()
    assert (target / "script.js").is_file()
