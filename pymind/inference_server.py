"""Small HTTP server that runs the real PyMind-A TinyGPT checkpoint."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os

from pymind.chat import LocalChatEngine


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object.")
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please enter a message.")
    if len(message) > 1200:
        raise ValueError("Message must be 1,200 characters or fewer.")
    history = payload.get("history", [])
    if not isinstance(history, list):
        raise ValueError("Conversation history must be a list.")
    safe_history = []
    for item in history[-8:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                safe_history.append({"role": item["role"], "content": content.strip()[:1200]})
    return message.strip(), safe_history


def make_handler(engine, auth_token=""):
    class InferenceHandler(BaseHTTPRequestHandler):
        def _send(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                return self._send({"error": "Not found."}, 404)
            self._send({
                "status": "ok" if Path(engine.checkpoint_path).is_file() else "checkpoint_missing",
                "model": "PyMind-A TinyGPT",
            })

        def do_POST(self):
            if self.path != "/chat":
                return self._send({"error": "Not found."}, 404)
            if auth_token and self.headers.get("Authorization") != "Bearer " + auth_token:
                return self._send({"error": "Unauthorized."}, 401)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 12000:
                    return self._send({"error": "Request size must be between 1 byte and 12 KB."}, 413)
                payload = json.loads(self.rfile.read(length))
                message, history = validate_payload(payload)
                reply = engine.reply(message, history=history, max_new_tokens=120)
                self._send({"reply": reply, "model": "PyMind-A TinyGPT"})
            except (ValueError, json.JSONDecodeError) as exc:
                self._send({"error": str(exc)}, 400)
            except FileNotFoundError:
                self._send({"error": "TinyGPT checkpoint is missing. Train the model first."}, 503)
            except Exception:
                self._send({"error": "TinyGPT inference failed."}, 500)

        def log_message(self, format, *args):
            return

    return InferenceHandler


def main():
    checkpoint = os.environ.get("PYMIND_CHECKPOINT", "checkpoints/tiny_gpt.pt")
    token = os.environ.get("PYMIND_INFERENCE_TOKEN", "")
    engine = LocalChatEngine(checkpoint)
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(engine, token))
    print("PyMind-A TinyGPT inference server listening on port %d" % port)
    server.serve_forever()


if __name__ == "__main__":
    main()
