"""Live health endpoint for the PyMind-A Vercel preview."""
from http.server import BaseHTTPRequestHandler
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
import os


class handler(BaseHTTPRequestHandler):
    def _json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        base = os.environ.get("PYMIND_INFERENCE_URL", "").strip().rstrip("/")
        if not base:
            return self._json({"status": "disconnected", "model": "PyMind-A TinyGPT"}, 503)

        headers = {"Accept": "application/json"}
        token = os.environ.get("PYMIND_INFERENCE_TOKEN", "")
        if token:
            headers["Authorization"] = "Bearer " + token
        request = Request(base + "/health", headers=headers, method="GET")
        try:
            with urlopen(request, timeout=8) as response:
                data = json.loads(response.read(16_000).decode("utf-8"))
            if isinstance(data, dict) and data.get("status") == "ok":
                data["service"] = "pymind-a-inference"
                return self._json(data, 200)
            return self._json({"status": "loading", "model": "PyMind-A TinyGPT"}, 503)
        except HTTPError as exc:
            return self._json({"status": "unavailable", "error": "Inference service HTTP %d" % exc.code}, 503)
        except (URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
            return self._json({"status": "unavailable", "model": "PyMind-A TinyGPT"}, 503)

    def do_POST(self):
        self._json({"error": "Method not allowed"}, 405)

    def log_message(self, format, *args):
        return
