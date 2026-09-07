import pytest


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Keep tests away from real state and, especially, real credentials.

    `bench.config` reads the environment on every call, so setting these is
    enough -- no module reloading. Pointing BENCH_KEY_FILE at a path that does
    not exist matters: without it the suite picks up ~/.config/camber/env and
    starts exercising live API keys, which is how this fixture earned its
    second half.
    """
    monkeypatch.setenv("BENCH_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("BENCH_KEY_FILE", str(tmp_path / "no-such-key-file"))
    for var in (
        "NVIDIA_API_KEY",
        "NVIDIA_API_KEY_IMPLEMENTER",
        "NVIDIA_API_KEY_INVESTIGATOR",
        "OPENROUTER_API_KEY",
        "BENCH_SHARED_BUCKET",
    ):
        monkeypatch.delenv(var, raising=False)

    from bench import concurrency, ratelimit
    ratelimit._reset_buckets()
    concurrency._reset_slots()
    yield
    ratelimit._reset_buckets()
    concurrency._reset_slots()
