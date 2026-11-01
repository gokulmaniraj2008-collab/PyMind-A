"""Minimal health endpoint for the PyMind-A Vercel preview."""
from http.server import BaseHTTPRequestHandler
import json


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps({"status": "ok", "service": "pymind-a-python-api"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        self.send_error(405, "Method not allowed")

    def log_message(self, format, *args):
        return
