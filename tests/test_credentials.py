"""Key normalisation, and one bucket per credential."""

import pytest

from bench.config import MODELS, credential_for, resolve


def test_bearer_prefix_is_stripped(tmp_path, monkeypatch):
    """Keys are copied out of docs and dashboards with the scheme attached."""
    monkeypatch.setenv("NVIDIA_API_KEY_IMPLEMENTER", "Bearer nvapi-abc123")
    assert credential_for(resolve("implementer")).key == "nvapi-abc123"

    monkeypatch.setenv("NVIDIA_API_KEY_IMPLEMENTER", "  'nvapi-quoted'  ")
    assert credential_for(resolve("implementer")).key == "nvapi-quoted"


def test_distinct_keys_share_one_bucket_by_default(monkeypatch):
    """Several keys usually belong to one account, so assume they do.

    Under-issuing runs slow; over-issuing produces 429 storms. Default to the
    recoverable failure.
    """
    monkeypatch.setenv("NVIDIA_API_KEY_IMPLEMENTER", "nvapi-one")
    monkeypatch.setenv("NVIDIA_API_KEY_INVESTIGATOR", "nvapi-two")
    a = credential_for(resolve("implementer"))
    b = credential_for(resolve("investigator"))
    assert a.bucket_key == b.bucket_key == "nvidia"
    assert a.rpm == b.rpm == 40


def test_identical_keys_share_a_bucket_even_when_split(monkeypatch):
    """Normalisation must happen before hashing, or the same key splits."""
    monkeypatch.setenv("NVIDIA_API_KEY_IMPLEMENTER", "nvapi-same")
    monkeypatch.setenv("NVIDIA_API_KEY_INVESTIGATOR", "Bearer nvapi-same")
    monkeypatch.setenv("BENCH_SPLIT_BUCKETS", "1")
    a = credential_for(resolve("implementer"))
    b = credential_for(resolve("investigator"))
    assert a.bucket_key == b.bucket_key


def test_split_buckets_is_opt_in(monkeypatch):
    """Only for genuinely separate accounts with separate allowances."""
    monkeypatch.setenv("NVIDIA_API_KEY_IMPLEMENTER", "nvapi-one")
    monkeypatch.setenv("NVIDIA_API_KEY_INVESTIGATOR", "nvapi-two")
    monkeypatch.setenv("BENCH_SPLIT_BUCKETS", "1")
    a = credential_for(resolve("implementer"))
    b = credential_for(resolve("investigator"))
    assert a.bucket_key != b.bucket_key


def test_falls_back_to_provider_wide_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-shared")
    cred = credential_for(resolve("implementer"))
    assert cred.key == "nvapi-shared"
    assert cred.source == "NVIDIA_API_KEY"


def test_key_is_masked_in_repr():
    """A Credential lands in tracebacks and log lines; the key must not."""
    from bench.config import Credential
    c = Credential(key="nvapi-supersecretvalue", source="X", bucket_key="b", rpm=40)
    assert "supersecretvalue" not in repr(c)
    assert "nvapi-sup" in repr(c), "a masked prefix is still useful for triage"
    assert repr(Credential(None, "none", "local", None)).count("None") >= 1


def test_hosted_model_ids_are_namespaced():
    """NIM ids are `publisher/model`; the web catalogue shows the bare name.

    Using the bare name returns a 404 whose body is literally "404 page not
    found", which gives no hint that the id is the problem. Cost one debugging
    round, so it gets a test.
    """
    for role, model in MODELS.items():
        if model.provider == "local":
            continue
        assert "/" in model.id, (
            f"{role} -> {model.id!r} is missing its publisher prefix"
        )
