"""Providers, models and where the bench keeps its state.

The important idea here is `Provider.bucket_key`. Rate limits are imposed per
*account*, not per model, so every model reached through one account must draw
from one shared token bucket. Kimi K3 and DeepSeek V4-Pro are both NVIDIA NIM
models: two clients with independent limiters would happily starve each other
and neither would understand why. See docs/DECISIONS.md AI-01.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def state_dir() -> Path:
    """Machine-local, git-ignored: cache, call log and rate-limiter state.

    Resolved on every call rather than captured at import. A module-level
    constant is read once, which makes the location impossible to redirect
    afterwards -- including from a test fixture, which is how this was caught.
    """
    return Path(os.environ.get("BENCH_STATE_DIR", ROOT / ".bench"))


def _key_file() -> Path:
    return Path(
        os.environ.get("BENCH_KEY_FILE", Path.home() / ".config" / "camber" / "env")
    )


def _key_from_file(name: str) -> str | None:
    """Read `NAME=value` from the key file, if it exists.

    The MCP server is launched as a subprocess by Claude Code and inherits the
    environment Claude Code itself started with, so exporting a variable in a
    live shell does not reach it. A file the server reads at call time does,
    without requiring a restart of the editor.
    """
    path = _key_file()
    try:
        raw = path.read_text()
    except (OSError, UnicodeDecodeError):
        return None
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == name:
            return value.strip().strip("'\"") or None
    return None


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key_env: str | None
    #: Requests per minute for the whole account. None means unmetered (local).
    rpm: int | None
    #: Models sharing this key share one token bucket.
    bucket_key: str
    #: Maximum requests in flight at once. 0 means unlimited. Measured, not
    #: documented: NIM 429s a second concurrent request to the same model.
    max_concurrency: int = 1

    @property
    def api_key(self) -> str | None:
        """Environment first, then the key file. Neither is ever logged."""
        if self.api_key_env is None:
            return None
        return _normalise_key(
            os.environ.get(self.api_key_env) or _key_from_file(self.api_key_env)
        )

    @property
    def configured(self) -> bool:
        return self.api_key_env is None or self.api_key is not None


PROVIDERS: dict[str, Provider] = {
    "nvidia": Provider(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        rpm=40,
        max_concurrency=int(os.environ.get("BENCH_NVIDIA_CONCURRENCY", "1")),
        bucket_key="nvidia",
    ),
    # LM Studio's default server. Unmetered because it is your own silicon.
    "local": Provider(
        name="local",
        base_url=os.environ.get("BENCH_LOCAL_URL", "http://127.0.0.1:1234/v1"),
        api_key_env=None,
        rpm=None,
        max_concurrency=0,  # your own GPU; queue it yourself if you must
        bucket_key="local",
    ),
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        rpm=None,
        max_concurrency=0,
        bucket_key="openrouter",
    ),
}


def _normalise_key(raw: str | None) -> str | None:
    """Tolerate a key pasted with its Authorization scheme attached.

    Credentials are copied out of dashboards and docs, and often arrive as
    "Bearer nvapi-...". The client adds the scheme itself, so leaving it on
    produces "Bearer Bearer ..." and a 401 that says nothing useful.
    """
    if raw is None:
        return None
    key = raw.strip().strip("'\"")
    for scheme in ("bearer ", "token "):
        if key.lower().startswith(scheme):
            key = key[len(scheme):].strip()
    return key or None


@dataclass(frozen=True)
class Model:
    #: The id the provider expects on the wire.
    id: str
    provider: str
    vision: bool = False
    #: Provider-documented output ceiling, for sanity-checking max_tokens.
    max_output: int | None = None
    note: str = ""
    #: Overrides the provider's key variable. Distinct credentials on the same
    #: provider usually mean distinct accounts, and so distinct rate limits.
    api_key_env: str | None = None


#: Roles, not model names, are what callers ask for. Swapping the model behind
#: a role must not touch call sites.
MODELS: dict[str, Model] = {
    "implementer": Model(
        id="deepseek-ai/deepseek-v4-pro-0813",
        provider="nvidia",
        api_key_env="NVIDIA_API_KEY_IMPLEMENTER",
        max_output=384_000,
        note="Best SWE-bench (80.6%) and #1 LiveCodeBench (93.5%). Give it a "
             "settled spec and failing tests; do not ask it to plan.",
    ),
    "investigator": Model(
        id="moonshotai/kimi-k3",
        provider="nvidia",
        api_key_env="NVIDIA_API_KEY_INVESTIGATOR",
        vision=True,
        note="Agentic 89.5 vs DeepSeek's 59.1, and native vision. The only "
             "model here that can read a residual plot or a mesh screenshot.",
    ),
    "janitor": Model(
        id=os.environ.get("BENCH_LOCAL_MODEL", "bitnet-b1.58-2b-4t"),
        provider="local",
        note="Ternary weights, unmetered, always on. Drive it with constrained "
             "decoding rather than trusting it to emit valid JSON. See AI-10.",
    ),
}


def resolve(role_or_id: str) -> Model:
    """Accept a role name, a capability, or a raw model id.

    Capabilities are resolved against measured availability (see roster.py), so
    `resolve("code")` returns whichever code-capable model this account can
    actually reach today, not a name hardcoded months ago.
    """
    if role_or_id in MODELS:
        return MODELS[role_or_id]
    for m in MODELS.values():
        if m.id == role_or_id:
            return m

    from .roster import CAPABILITIES, resolve_capability
    if role_or_id in CAPABILITIES:
        chosen = resolve_capability(role_or_id)
        return Model(
            id=chosen.model_id,
            provider="nvidia",
            api_key_env="NVIDIA_API_KEY_IMPLEMENTER",
            vision=(role_or_id == "vision"),
            note=chosen.why,
        )

    raise KeyError(
        f"unknown model, role or capability {role_or_id!r}; "
        f"roles: {', '.join(sorted(MODELS))}; "
        f"capabilities: {', '.join(sorted(CAPABILITIES))}"
    )


def provider_for(model: Model) -> Provider:
    return PROVIDERS[model.provider]


@dataclass(frozen=True)
class Credential:
    """A key, where it came from, and which rate-limit bucket it draws on."""

    key: str | None
    source: str
    bucket_key: str
    rpm: int | None
    max_concurrency: int = 1

    @property
    def configured(self) -> bool:
        return self.key is not None or self.source == "none"

    def __repr__(self) -> str:
        """Masked. A Credential can reach a traceback or a log line."""
        if self.key is None:
            shown = "None"
        else:
            shown = f"'{self.key[:9]}...{self.key[-4:]}'"
        return (
            f"Credential(key={shown}, source={self.source!r}, "
            f"bucket_key={self.bucket_key!r}, rpm={self.rpm}, "
            f"max_concurrency={self.max_concurrency})"
        )


def credential_for(model: Model) -> Credential:
    """Resolve the key for a model, and the bucket that key spends from.

    Rate limits are imposed per *account*, and several keys can belong to one
    account -- which is the case here. So the default is one bucket per
    provider, shared by every credential on it.

    The asymmetry decides the default. Under-issuing merely runs slower than
    necessary; over-issuing produces sustained 429s, and retries against an
    exhausted account make that worse rather than better. So splitting is
    opt-in: set BENCH_SPLIT_BUCKETS=1 only when the keys really are separate
    accounts with separate allowances.
    """
    provider = provider_for(model)

    if provider.api_key_env is None:  # local, unmetered
        return Credential(None, "none", provider.bucket_key, provider.rpm,
                          provider.max_concurrency)

    source = model.api_key_env or provider.api_key_env
    key = _normalise_key(os.environ.get(source) or _key_from_file(source))

    if key is None and model.api_key_env:  # fall back to the provider-wide key
        source = provider.api_key_env
        key = _normalise_key(os.environ.get(source) or _key_from_file(source))

    if key is None:
        return Credential(None, model.api_key_env or provider.api_key_env,
                          provider.bucket_key, provider.rpm,
                          provider.max_concurrency)

    if os.environ.get("BENCH_SPLIT_BUCKETS") == "1":
        digest = hashlib.sha256(key.encode()).hexdigest()[:12]
        bucket = f"{provider.bucket_key}:{digest}"
    else:
        bucket = provider.bucket_key
    return Credential(key, source, bucket, provider.rpm,
                      provider.max_concurrency)
