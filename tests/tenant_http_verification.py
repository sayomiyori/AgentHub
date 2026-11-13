"""Controlled issuer and real built HTTP smoke; never use against public services."""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

TENANT = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"
USER = "33333333-3333-3333-3333-333333333333"
BASE = os.getenv("AGENTHUB_VERIFICATION_URL", "http://127.0.0.1:58147")


class Issuer(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        permissions = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        token = self.headers.get("Authorization", "")
        tenant = OTHER if token == "Bearer synthetic-other" else TENANT
        role = "member" if token == "Bearer synthetic-member" else "owner"
        status = 200
        if token not in {"Bearer synthetic-owner", "Bearer synthetic-other", "Bearer synthetic-member"}:
            status = 401
        elif self.path != f"/api/v1/tenants/{tenant}/authorize":
            status = 404
        elif permissions["permission"] == "ai.configure" and role != "owner":
            status = 403
        body = json.dumps(dict(user_id=USER, tenant_id=tenant, role=role,
                               permission=permissions["permission"], allowed=status == 200)).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def request(method, path, payload=None, *, token="synthetic-owner", headers=None):
    data = json.dumps(payload).encode() if isinstance(payload, dict) else payload
    h = {"Authorization": "Bearer " + token} if token else {}
    if isinstance(payload, dict):
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    try:
        response = urlopen(Request(BASE + path, data=data, headers=h, method=method), timeout=10)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


def upload(tenant, token, name):
    boundary = "controlled-upload"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}.txt"\r\n'
            f'Content-Type: text/plain\r\n\r\n{name} knowledge\r\n--{boundary}--\r\n').encode()
    status, receipt = request("POST", f"/api/v1/tenants/{tenant}/documents", body, token=token,
                              headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert status == 202, (status, receipt)
    deadline = time.monotonic() + 30
    path = f"/api/v1/tenants/{tenant}/documents/{receipt['document_id']}"
    while time.monotonic() < deadline:
        status, document = request("GET", path, token=token)
        assert status == 200 and document["upload_status"] != "failed"
        if document["upload_status"] == "ready":
            assert document["chunk_count"] > 0
            return receipt["document_id"]
        time.sleep(0.2)
    raise AssertionError("Document worker did not finish")


def smoke():
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if request("GET", "/health", token="")[0] == 200:
                break
        except OSError:
            pass
        time.sleep(0.2)
    assert request("GET", "/health", token="")[0] == 200
    for path in ("/api/v1/documents", "/api/v1/query", "/api/v1/conversations",
                 "/api/v1/usage/stats", "/mcp/sse"):
        assert request("GET", path)[0] == 404
    base = f"/api/v1/tenants/{TENANT}"
    assert request("GET", base + "/documents", token="")[0] == 401
    assert request("GET", base + "/documents", token="synthetic-revoked")[0] == 401
    assert request("GET", f"/api/v1/tenants/{OTHER}/documents")[0] == 404
    assert request("POST", base + "/documents", token="synthetic-member")[0] == 403
    own = upload(TENANT, "synthetic-owner", "Own")
    foreign = upload(OTHER, "synthetic-other", "Foreign")
    assert request("GET", base + f"/documents/{foreign}")[0] == 404
    assert request("DELETE", base + f"/documents/{foreign}")[0] == 404
    assert request("GET", base + "/documents", token="synthetic-member")[1][0]["id"] == own
    before = request("GET", base + "/usage/stats")[1]["total_tokens"]
    payload = {"question": "Own knowledge?", "use_agent": False}
    status, first = request("POST", base + "/query", payload)
    assert status == 200 and first["answer"] == "Verification answer" and first["tokens_used"] == 15
    assert {x["document_title"] for x in first["sources"]} == {"Own.txt"}
    status, second = request("POST", base + "/query", payload)
    assert status == 200 and second["semantic_cache_hit"] and second["tokens_used"] == 0
    assert request("GET", base + "/usage/stats")[1]["total_tokens"] - before == 15
    assert len(request("GET", base + f"/conversations/{first['conversation_id']}/messages")[1]) == 2
    assert request("DELETE", base + f"/documents/{own}")[0] == 200
    status, after_delete = request("POST", base + "/query", payload)
    assert status == 200 and after_delete["sources"] == [] and not after_delete["semantic_cache_hit"]
    assert request("DELETE", f"/api/v1/tenants/{OTHER}/documents/{foreign}", token="synthetic-other")[0] == 200
    print("PASS: built startup; fresh issuer authorization; owner/member/foreign/revoked; "
          "upload -> Celery -> pgvector -> boundary LLM -> usage/history/cache; delete invalidation")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["issuer"]:
        ThreadingHTTPServer(("0.0.0.0", 8000), Issuer).serve_forever()
    else:
        smoke()
