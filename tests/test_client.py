"""Full request path with a stubbed transport: no network, no key, no quota."""

from __future__ import annotations

import pytest

from bench.callog import CallLog
from bench.config import resolve
from bench.client import BenchError, Client, NotConfigured


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else _ok("hello")
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._payload


def _ok(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


class FakeHTTP:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json, headers):
        self.calls.append({"url": url, "json": json, "headers": headers})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def close(self):
        pass


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")


MSG = [{"role": "user", "content": "ping"}]


def test_happy_path_logs_and_caches(keyed):
    c = Client()
    c._http = FakeHTTP([FakeResponse()])
    out = c.complete("implementer", MSG, max_tokens=16)
    assert c.text(out) == "hello"

    # Second identical call must not touch the transport at all.
    again = c.complete("implementer", MSG, max_tokens=16)
    assert c.text(again) == "hello"
    assert len(c._http.calls) == 1

    records = CallLog().read()
    assert [r["cached"] for r in records] == [False, True]
    assert records[0]["model"] == resolve("implementer").id
    assert records[0]["prompt_tokens"] == 11


def test_routes_to_the_right_model_and_sends_auth(keyed):
    c = Client()
    c._http = FakeHTTP([FakeResponse()])
    c.complete("investigator", MSG)
    call = c._http.calls[0]
    assert call["json"]["model"] == resolve("investigator").id
    assert call["url"].startswith("https://integrate.api.nvidia.com/v1")
    assert call["headers"]["Authorization"] == "Bearer test-key-not-real"


def test_retries_transient_status_then_succeeds(keyed, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    c = Client()
    c._http = FakeHTTP([
        FakeResponse(status_code=429, headers={"retry-after": "0"}),
        FakeResponse(status_code=503),
        FakeResponse(payload=_ok("recovered")),
    ])
    assert c.text(c.complete("implementer", MSG)) == "recovered"
    assert len(c._http.calls) == 3


def test_client_error_is_not_retried_and_is_logged(keyed):
    c = Client()
    c._http = FakeHTTP([FakeResponse(status_code=400, text="bad request")])
    with pytest.raises(BenchError, match="HTTP 400"):
        c.complete("implementer", MSG)
    assert len(c._http.calls) == 1
    record = CallLog().read()[-1]
    assert record["ok"] is False and "HTTP 400" in record["error"]


def test_missing_key_raises_before_any_call():
    c = Client()
    c._http = FakeHTTP([])
    with pytest.raises(NotConfigured, match="NVIDIA_API_KEY_IMPLEMENTER"):
        c.complete("implementer", MSG)


def test_max_tokens_above_model_ceiling_is_refused(keyed):
    c = Client()
    with pytest.raises(BenchError, match="caps output"):
        c.complete("implementer", MSG, max_tokens=500_000)


def test_failed_call_still_consumed_budget(keyed):
    """A request that reached the provider and failed has spent a token."""
    from bench.config import credential_for, resolve
    from bench.ratelimit import bucket_for

    cred = credential_for(resolve("implementer"))
    c = Client()
    c._http = FakeHTTP([FakeResponse(status_code=400, text="nope")])
    before = bucket_for(cred.bucket_key, cred.rpm).peek()
    with pytest.raises(BenchError):
        c.complete("implementer", MSG)
    assert bucket_for(cred.bucket_key, cred.rpm).peek() < before


def test_key_file_is_used_when_env_is_absent(tmp_path, monkeypatch):
    """The MCP server cannot see a variable exported after Claude Code started."""
    keyfile = tmp_path / "env"
    keyfile.write_text("# comment\nNVIDIA_API_KEY='from-file'\nOTHER=x\n")
    monkeypatch.setenv("BENCH_KEY_FILE", str(keyfile))

    c = Client()
    c._http = FakeHTTP([FakeResponse()])
    c.complete("implementer", MSG)
    assert c._http.calls[0]["headers"]["Authorization"] == "Bearer from-file"


def test_environment_wins_over_key_file(tmp_path, monkeypatch):
    keyfile = tmp_path / "env"
    keyfile.write_text("NVIDIA_API_KEY=from-file\n")
    monkeypatch.setenv("BENCH_KEY_FILE", str(keyfile))
    monkeypatch.setenv("NVIDIA_API_KEY", "from-env")

    c = Client()
    c._http = FakeHTTP([FakeResponse()])
    c.complete("implementer", MSG)
    assert c._http.calls[0]["headers"]["Authorization"] == "Bearer from-env"


def test_missing_key_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_KEY_FILE", str(tmp_path / "nope"))
    c = Client()
    with pytest.raises(NotConfigured):
        c.complete("implementer", MSG)


def _truncated(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 32,
                  "completion_tokens_details": {"reasoning_tokens": 29}},
    }


def test_truncated_generation_raises_rather_than_returning_a_stump(keyed):
    """Reasoning models eat max_tokens before the visible answer starts."""
    from bench.client import Truncated
    c = Client()
    c._http = FakeHTTP([FakeResponse(payload=_truncated("KIM"))])
    payload = c.complete("investigator", MSG, max_tokens=32)
    with pytest.raises(Truncated, match="reasoning"):
        c.text(payload)
    assert c.text(payload, allow_truncated=True) == "KIM"


def test_truncation_is_recorded_in_the_call_log(keyed):
    c = Client()
    c._http = FakeHTTP([FakeResponse(payload=_truncated("KIM"))])
    c.complete("investigator", MSG, max_tokens=32)
    assert CallLog().read()[-1]["finish_reason"] == "length"


def test_reasoning_with_no_answer_is_an_error(keyed):
    c = Client()
    c._http = FakeHTTP([FakeResponse(payload={
        "choices": [{"message": {"content": "", "reasoning_content": "hmm..."},
                     "finish_reason": "stop"}],
        "usage": {},
    })])
    payload = c.complete("investigator", MSG)
    with pytest.raises(BenchError, match="reasoning but no answer"):
        c.text(payload)


def test_truncation_detected_without_finish_reason(keyed):
    """NIM returned a plainly-truncated body with finish_reason absent."""
    from bench.client import Truncated
    c = Client()
    c._http = FakeHTTP([FakeResponse(payload={
        "choices": [{"message": {"content": "KIM"}}],   # no finish_reason
        "usage": {"prompt_tokens": 94, "completion_tokens": 32},
    })])
    payload = c.complete("investigator", MSG, max_tokens=32)
    with pytest.raises(Truncated):
        c.text(payload)


def test_completion_below_ceiling_is_not_truncated(keyed):
    c = Client()
    c._http = FakeHTTP([FakeResponse(payload={
        "choices": [{"message": {"content": "OK"}}],
        "usage": {"prompt_tokens": 94, "completion_tokens": 5},
    })])
    payload = c.complete("investigator", MSG, max_tokens=32)
    assert c.text(payload) == "OK"
