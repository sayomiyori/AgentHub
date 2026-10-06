"""Explicit controlled receiver CLI; never imported by normal application startup."""

import hashlib
import hmac
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

records: dict[str, dict] = {}
lock = threading.Lock()
calls = 0


class Receiver(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body):
        content = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        with lock:
            self.reply(200, {"calls": calls, "records": len(records)})

    def do_POST(self):
        global calls
        assert self.path == "/internal/v1/telegram/answers"
        size = int(self.headers["Content-Length"])
        assert 0 < size <= 1048576
        body = self.rfile.read(size)
        expected = "sha256=" + hmac.new(os.environ["AGENT_WEBHOOK_REPLY_KEY"].encode(),
                                      body, hashlib.sha256).hexdigest()
        assert hmac.compare_digest(self.headers["X-Webhook-Signature"], expected)
        envelope = json.loads(body)
        assert "chat_id" not in envelope["payload"] and "token" not in envelope["payload"]
        with lock:
            calls += 1
            if calls == 1:
                self.reply(409, {"detail": "ingress_publication_not_ready"})
                return
            event = envelope["event_id"]
            if event not in records:
                records[event] = {"body": body, "receipt": {
                    "event_id": event, "delivery_id": os.environ["VERIFICATION_DELIVERY_ID"], "state": "pending"}}
                # Persisted remote admission with a lost HTTP receipt.
                self.close_connection = True
                return
            assert records[event]["body"] == body
            self.reply(200, records[event]["receipt"])


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8000), Receiver).serve_forever()
