"""Run against the isolated built image, never against a public service."""
import json
import time
from urllib.request import Request, urlopen
from uuid import uuid4

BASE = "http://127.0.0.1:38082"


def request(path, payload=None, headers=None):
    data = json.dumps(payload).encode() if isinstance(payload, dict) else payload
    request_headers = {"Content-Type": "application/json"} if isinstance(payload, dict) else {}
    request_headers.update(headers or {})
    with urlopen(Request(BASE + path, data=data, headers=request_headers), timeout=10) as response:
        return json.load(response)


def main():
    assert request("/health")["status"] == "ok"
    question = f"Evidence {uuid4()}"
    boundary = "verification-boundary"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="evidence.txt"\r\n'
            f'Content-Type: text/plain\r\n\r\n{question}\r\n--{boundary}--\r\n').encode()
    document = request("/api/v1/documents", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = request(f"/api/v1/documents/{document['document_id']}")
        if status["upload_status"] == "ready":
            break
        if status["upload_status"] == "failed":
            raise AssertionError("Embedding worker failed")
        time.sleep(0.2)
    assert status["upload_status"] == "ready" and status["chunk_count"] > 0
    before = request("/api/v1/usage/stats")["total_tokens"]
    first = request("/api/v1/query", {"question": question, "use_agent": False})
    second = request("/api/v1/query", {"question": question, "use_agent": False})
    assert first["answer"] == "Verification answer" and first["tokens_used"] == 15
    assert first["sources"] and second["semantic_cache_hit"]
    assert second["tokens_used"] == 0 and second["cost_usd"] == 0
    assert request("/api/v1/usage/stats")["total_tokens"] - before == 15
    assert len(request(f"/api/v1/conversations/{first['conversation_id']}/messages")) == 2
    failed = request("/api/v1/query", {"question": "provider-failure " + str(uuid4()), "use_agent": False})
    assert failed["tokens_used"] == 0 and "temporarily unavailable" in failed["answer"]
    print("PASS: built HTTP upload -> Celery -> pgvector -> boundary LLM -> usage -> Redis cache; provider timeout")


if __name__ == "__main__":
    main()
