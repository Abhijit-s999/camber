"""One client for every OpenAI-compatible endpoint the bench talks to.

NVIDIA NIM, OpenRouter, DeepSeek direct and LM Studio all speak the same
chat-completions dialect, so one implementation covers all of them and swapping
provider is a base-URL change. Anthropic is deliberately absent: it needs a
different request shape, and the bench does not need it -- Opus 5 participates
by running Claude Code, not by being called from inside it.

Every request passes through the shared bucket, then the cache, then the wire,
and every outcome lands in the call log whether it succeeded or not.
"""

from __future__ import annotations

import random
import time
from typing import Any, Iterable

from . import config
from .cache import Cache, request_key
from .concurrency import slots_for
from .callog import CallLog, CallRecord
from .ratelimit import bucket_for

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class BenchError(RuntimeError):
    pass


class NotConfigured(BenchError):
    """The provider needs an API key that is not in the environment."""


class Truncated(BenchError):
    """Generation hit max_tokens before the model finished.

    Worth its own exception because the failure is silent otherwise: the
    response looks like a normal answer and is simply missing its end. For
    `implement` that means a source file that stops mid-function but still
    parses as a string, which is far worse than an error.
    """


class Client:
    def __init__(
        self,
        *,
        cache: Cache | None = None,
        callog: CallLog | None = None,
        max_retries: int = 4,
        timeout: float = 300.0,
    ) -> None:
        self.cache = cache if cache is not None else Cache()
        self.callog = callog if callog is not None else CallLog()
        self.max_retries = max_retries
        self.timeout = timeout
        self._http: Any = None

    # httpx is imported lazily so that `import bench` works in an environment
    # that only installed the product dependencies.
    def _client(self):
        if self._http is None:
            try:
                import httpx
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise BenchError(
                    "the bench needs httpx: uv pip install -e '.[bench]'"
                ) from exc
            self._http = httpx.Client(timeout=self.timeout)
        return self._http

    def complete(
        self,
        role: str,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        use_cache: bool = True,
        wait_timeout: float | None = None,
    ) -> dict:
        """Send a chat completion. Returns the provider's parsed JSON body."""
        model = config.resolve(role)
        provider = config.provider_for(model)
        cred = config.credential_for(model)

        if not cred.configured:
            raise NotConfigured(
                f"role {role!r} ({model.id}) needs ${cred.source}. "
                f"Put it in ~/.config/camber/env or export it."
            )
        if max_tokens and model.max_output and max_tokens > model.max_output:
            raise BenchError(
                f"{model.id} caps output at {model.max_output} tokens, "
                f"asked for {max_tokens}"
            )

        params: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if response_format is not None:
            params["response_format"] = response_format

        key = request_key(model.id, messages, **params)

        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                self.callog.write(CallRecord(
                    role=role, model=model.id, provider=provider.name,
                    prompt_hash=key, cached=True, ok=True, latency_s=0.0,
                ))
                return hit

        bucket = bucket_for(cred.bucket_key, cred.rpm)
        slots = slots_for(cred.bucket_key, cred.max_concurrency)

        body = {"model": model.id, "messages": messages, **params}
        headers = {"Content-Type": "application/json"}
        if cred.key:
            headers["Authorization"] = f"Bearer {cred.key}"
        url = f"{provider.base_url.rstrip('/')}/chat/completions"

        # The slot is held for the whole request, the token only to start it.
        # These requests run for minutes, so a second one launched while the
        # first is in flight is rejected instantly regardless of rate.
        waited = 0.0
        started = time.monotonic()
        payload, error = None, None
        slot_wait = time.monotonic()
        with slots.hold(timeout=wait_timeout):
            waited += time.monotonic() - slot_wait
            waited += bucket.acquire(timeout=wait_timeout)
            payload, error, waited = self._send(
                url, body, headers, bucket, waited, wait_timeout
            )
        latency = time.monotonic() - started
        usage = (payload or {}).get("usage") or {}
        finish = None
        try:
            finish = (payload or {})["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError):
            pass
        self.callog.write(CallRecord(
            role=role, model=model.id, provider=provider.name, prompt_hash=key,
            cached=False, ok=payload is not None, latency_s=round(latency, 3),
            waited_s=round(waited, 3),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=finish, error=error,
        ))

        if payload is None:
            raise BenchError(f"{model.id} call failed: {error}")

        # text() needs to know what ceiling produced this body; the provider
        # does not reliably say so itself.
        if max_tokens is not None:
            payload["_bench_max_tokens"] = max_tokens

        if use_cache:
            self.cache.put(key, model.id, body, payload)
        return payload

    def _send(self, url, body, headers, bucket, waited, wait_timeout):
        """Post with retries. Runs while holding a concurrency slot."""
        payload, error = None, None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client().post(url, json=body, headers=headers)
                if resp.status_code in RETRY_STATUS and attempt < self.max_retries:
                    # Every retry re-enters the bucket. A 429 means the account
                    # is over its limit, so retrying without paying again is
                    # exactly how you stay over it.
                    delay = self._backoff(attempt, resp.headers.get("retry-after"))
                    time.sleep(delay)
                    waited += delay + bucket.acquire(timeout=wait_timeout)
                    continue
                if resp.status_code >= 400:
                    error = f"HTTP {resp.status_code}: {resp.text[:400]}"
                    break
                payload = resp.json()
                break
            except Exception as exc:  # noqa: BLE001 - logged, then re-raised
                error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.max_retries:
                    break
                delay = self._backoff(attempt, None)
                time.sleep(delay)
                waited += delay + bucket.acquire(timeout=wait_timeout)
        return payload, error, waited

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(60.0, float(retry_after))
            except ValueError:
                pass
        return min(30.0, (2 ** attempt)) * (0.5 + random.random() / 2)

    @staticmethod
    def text(payload: dict, *, allow_truncated: bool = False) -> str:
        """Pull the assistant message out of a completion body.

        Raises on a truncated generation rather than returning the fragment.
        Both models here are reasoning models: they spend tokens thinking before
        emitting anything visible, so a max_tokens sized for the answer alone
        gets consumed by the reasoning and returns a stump. Asking for "OK" with
        max_tokens=32 returned "KIM" -- which is exactly the kind of result that
        looks like an answer and is not.
        """
        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BenchError(f"unexpected response shape: {payload!r}") from exc

        content = message.get("content") or ""
        finish = choice.get("finish_reason")
        usage_now = payload.get("usage") or {}

        # NIM does not always populate finish_reason. The observed Kimi
        # response had completion_tokens exactly equal to the requested
        # max_tokens and no finish_reason at all -- plainly truncated, but
        # invisible to a finish_reason check alone. Treat hitting the ceiling
        # exactly as truncation regardless of what the provider claims.
        requested = (payload.get("_bench_max_tokens")
                     or usage_now.get("_requested_max_tokens"))
        hit_ceiling = (
            requested is not None
            and usage_now.get("completion_tokens") == requested
        )

        if (finish == "length" or hit_ceiling) and not allow_truncated:
            usage = usage_now
            reasoning = (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            )
            hint = f", {reasoning} of them on reasoning" if reasoning else ""
            raise Truncated(
                f"generation hit max_tokens "
                f"({usage.get('completion_tokens', '?')} completion tokens{hint}). "
                f"Raise max_tokens; these are reasoning models and need headroom "
                f"well beyond the visible answer. Got: {content[:80]!r}"
            )

        if not content and message.get("reasoning_content"):
            raise BenchError(
                "model returned reasoning but no answer -- usually too small a "
                "max_tokens, or a prompt it declined to answer"
            )
        return content

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
        self.cache.close()
