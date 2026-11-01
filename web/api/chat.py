"""Vercel chat proxy for a separately hosted PyMind-A inference server."""
from http.server import BaseHTTPRequestHandler
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import json
import os


MAX_BODY = 12_000
MAX_MESSAGE = 1_200
MAX_HISTORY = 8
UPSTREAM_TIMEOUT = 25


def prepare_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object.")
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please enter a message.")
    message = message.strip()
    if len(message) > MAX_MESSAGE:
        raise ValueError("Message must be 1,200 characters or fewer.")
    history = payload.get("history", [])
    if not isinstance(history, list):
        raise ValueError("Conversation history must be a list.")
    safe_history = []
    for item in history[-MAX_HISTORY:]:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            safe_history.append({"role": role, "content": content.strip()[:MAX_MESSAGE]})
    return {"message": message, "history": safe_history}


def inference_url():
    base = os.environ.get("PYMIND_INFERENCE_URL", "").strip().rstrip("/")
    if not base:
        raise RuntimeError("TinyGPT inference service is not connected yet.")
    parsed = urlparse(base)
    if parsed.scheme not in ("https", "http") or not parsed.netloc:
        raise RuntimeError("PYMIND_INFERENCE_URL must be a valid HTTP(S) URL.")
    return base + "/chat"


def ask_inference(payload):
    url = inference_url()
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = os.environ.get("PYMIND_INFERENCE_TOKEN", "")
    if token:
        headers["Authorization"] = "Bearer " + token
    request = Request(url, data=data, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=UPSTREAM_TIMEOUT) as response:
            result = json.loads(response.read(64_000).decode("utf-8"))
    except HTTPError as exc:
        if exc.code in (400, 401, 413, 503):
            raise RuntimeError("TinyGPT service rejected the request (HTTP %d)." % exc.code) from exc
        raise RuntimeError("TinyGPT service is temporarily unavailable.") from exc
    except (URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("Could not reach the TinyGPT inference service.") from exc
    reply = result.get("reply") if isinstance(result, dict) else None
    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError("TinyGPT service returned an invalid response.")
    return {"reply": reply.strip(), "model": result.get("model", "PyMind-A TinyGPT")}


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_BODY:
                return self._json({"error": "Request size must be between 1 byte and 12 KB."}, 413)
            payload = prepare_payload(json.loads(self.rfile.read(length)))
            result = ask_inference(payload)
            self._json(result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)
        except RuntimeError as exc:
            status = 503 if "not connected" in str(exc).lower() or "unavailable" in str(exc).lower() or "reach" in str(exc).lower() else 502
            self._json({"error": str(exc), "code": "model_unavailable"}, status)
        except Exception:
            self._json({"error": "Chat request failed. Please try again."}, 500)

    def do_GET(self):
        self._json({"error": "Use POST with a JSON message."}, 405)

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
