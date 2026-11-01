"""Dependency-free prompt-to-website API for Vercel Python Functions."""
from http.server import BaseHTTPRequestHandler
from html import escape
import json
import re
import zipfile
from io import BytesIO


def generate_site(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Please enter a website description.")
    if len(prompt) > 1200:
        raise ValueError("Website description must be 1,200 characters or fewer.")
    text = prompt.strip()
    lowered = text.lower()
    if any(word in lowered for word in ("3d", "three-dimensional", "car", "model viewer")):
        kind = "3D viewer"
        heading = "Explore a 3D scene in your browser."
        cards = [("Interactive preview", "Drag to rotate and scroll to zoom."),
                 ("Custom assets", "Replace the sample with your own GLB model."),
                 ("Next step", "Connect Blender exports from PyMind-A.")]
    elif any(word in lowered for word in ("shop", "store", "product", "ecommerce")):
        kind = "product page"
        heading = "A better way to discover your next favourite."
        cards = [("Featured product", "Add product details, images and pricing."),
                 ("Why choose us", "Explain the value your product provides."),
                 ("Get in touch", "Connect customers with your team.")]
    elif any(word in lowered for word in ("portfolio", "resume", "personal", "developer")):
        kind = "portfolio"
        heading = "Build a portfolio that tells your story."
        cards = [("Selected work", "Showcase your strongest projects."),
                 ("Skills", "Highlight the tools and technologies you use."),
                 ("Contact", "Make it easy for people to reach you.")]
    else:
        kind = "business landing page"
        heading = "A clear, modern home for your business."
        cards = [("What we do", "Describe your services in a few simple sentences."),
                 ("Why it matters", "Explain what makes your work valuable."),
                 ("Contact", "Give visitors a clear way to reach you.")]
    title = escape(text.splitlines()[0][:48])
    safe_prompt = escape(text)
    card_markup = "".join(
        '<article class="card"><small>0%d</small><h2>%s</h2><p>%s</p></article>'
        % (index, escape(card_title), escape(description))
        for index, (card_title, description) in enumerate(cards, 1)
    )
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>%s</title>
<link rel="stylesheet" href="styles.css"></head><body>
<header><b>%s</b><a href="#details">Explore ↗</a></header>
<main><section class="hero"><small>MADE WITH PYMIND-A</small><h1>%s</h1>
<p>%s</p><p class="brief">%s</p><a class="button" href="#details">Get started</a></section>
<section id="details"><small>THE FOUNDATION</small><h2>Made to grow with your idea.</h2>
<div class="grid">%s</div></section></main>
<footer>Built with PyMind-A · <span id="year"></span></footer>
<script src="script.js"></script></body></html>""" % (title, title, heading, "A clean, responsive starter page you can customize.", safe_prompt, card_markup)
    css = """*{box-sizing:border-box}body{margin:0;background:#f5f7fb;color:#142033;font-family:system-ui,sans-serif}header{height:68px;padding:0 7vw;display:flex;align-items:center;justify-content:space-between;background:#fff;border-bottom:1px solid #e5eaf2}header a{color:#2457e6;text-decoration:none;font-weight:700}.hero{padding:clamp(65px,12vw,130px) 10vw;background:radial-gradient(circle at 85% 20%,#dce7ff,transparent 38%),#fff}.hero h1{font-size:clamp(2.7rem,7vw,5.8rem);letter-spacing:-.06em;line-height:1;margin:18px 0;max-width:850px}.hero p{color:#53627a;max-width:680px;line-height:1.7}.brief{font-size:13px}.button{display:inline-block;margin-top:14px;padding:13px 19px;border-radius:30px;background:#2457e6;color:white;text-decoration:none;font-weight:700}section#details{padding:75px 8vw}section small{color:#58709a;font-weight:800;letter-spacing:.14em}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin-top:25px}.card{padding:25px;border:1px solid #e5eaf2;border-radius:18px;background:#fff}.card small{color:#2457e6}.card p{color:#64748b;line-height:1.6}footer{text-align:center;padding:25px;color:#718096;border-top:1px solid #e5eaf2}@media(max-width:600px){header{padding:0 20px}}"""
    js = "document.querySelector('#year').textContent=new Date().getFullYear();"
    return {"kind": kind, "files": {"index.html": html, "styles.css": css, "script.js": js}}


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 10000:
                return self._json({"error": "Request is too large."}, 413)
            body = json.loads(self.rfile.read(length) or b"{}")
            result = generate_site(body.get("prompt"))
            if self.path.endswith("format=zip"):
                archive = BytesIO()
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                    for name, content in result["files"].items():
                        bundle.writestr("pymind-website/" + name, content)
                data = archive.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", 'attachment; filename="pymind-website.zip"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self._json(result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception:
            self._json({"error": "Website generation failed. Please try again."}, 500)

    def do_GET(self):
        self._json({"error": "Use POST with a JSON prompt."}, 405)

    def _json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return
