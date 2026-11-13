"""Real isolated AuthFortress JWTs stay in memory; provider boundary is controlled."""

import json
import os
import secrets
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from tests import tenant_http_verification as smoke

AUTH = os.getenv("AUTHFORTRESS_VERIFICATION_URL", "http://127.0.0.1:55921")
smoke.BASE = os.getenv("AGENTHUB_VERIFICATION_URL", "http://127.0.0.1:58148")


def auth(path, payload=None, token=None):
    headers = {"Authorization": "Bearer " + token} if token else {}
    data = json.dumps(payload).encode() if payload is not None else None
    if data:
        headers["Content-Type"] = "application/json"
    try:
        response = urlopen(Request(AUTH + path, data=data, headers=headers, method="POST"), timeout=10)
    except HTTPError as error:
        response = error
    with response:
        body = response.read()
        return response.status, json.loads(body) if body else {}


def identity():
    name = "agenthub-" + uuid4().hex
    email = name + "@example.com"
    password = secrets.token_urlsafe(32) + "Ab9!"
    assert auth("/api/v1/auth/register", dict(email=email, password=password, username=name))[0] == 200
    status, session = auth("/api/v1/auth/login", dict(email=email, password=password))
    assert status == 200
    token = session["access_token"]
    status, tenant = auth("/api/v1/tenants", {"name": name}, token)
    assert status == 201
    return token, tenant["id"]


def main():
    token, tenant = identity()
    foreign_token, foreign_tenant = identity()
    base = f"/api/v1/tenants/{tenant}"
    foreign_base = f"/api/v1/tenants/{foreign_tenant}"
    assert smoke.request("GET", base + "/documents", token=token)[0] == 200
    assert smoke.request("GET", foreign_base + "/documents", token=token)[0] == 404
    assert smoke.request("GET", base + "/documents", token="not-a-jwt")[0] == 401
    own = smoke.upload(tenant, token, "Real issuer own")
    foreign = smoke.upload(foreign_tenant, foreign_token, "Real issuer foreign")
    assert smoke.request("GET", base + f"/documents/{foreign}", token=token)[0] == 404
    status, answer = smoke.request("POST", base + "/query",
                                  {"question": "Real issuer own knowledge?", "use_agent": False}, token=token)
    assert status == 200 and answer["answer"] == "Verification answer"
    assert {x["document_title"] for x in answer["sources"]} == {"Real issuer own.txt"}
    assert smoke.request("DELETE", base + f"/documents/{own}", token=token)[0] == 200
    assert smoke.request("DELETE", foreign_base + f"/documents/{foreign}", token=foreign_token)[0] == 200
    assert auth("/api/v1/auth/logout", token=token)[0] == 204
    assert smoke.request("GET", base + "/documents", token=token)[0] == 401
    assert auth("/api/v1/auth/logout", token=foreign_token)[0] == 204
    print("PASS: real AuthFortress register/login/tenant JWT; scoped upload/worker/query; "
          "foreign tenant/resource 404; invalid and revoked JWT 401; tokens never persisted")


if __name__ == "__main__":
    main()
