import os
import subprocess
import sys
import time

from bench.ratelimit import RateLimitTimeout, TokenBucket


def test_burst_up_to_capacity_then_throttles():
    b = TokenBucket(key="t", rpm=60)  # 1/s, capacity 60
    for _ in range(60):
        assert b.acquire(timeout=0.01) == 0.0
    # 61st must wait, because the bucket started full and is now empty.
    start = time.monotonic()
    b.acquire()
    assert time.monotonic() - start > 0.5


def test_refills_over_time():
    b = TokenBucket(key="refill", rpm=600, capacity=2)  # 10/s
    b.acquire(); b.acquire()
    time.sleep(0.25)
    assert b.acquire(timeout=0.05) == 0.0  # ~2.5 tokens accrued


def test_timeout_raises_rather_than_blocking_forever():
    b = TokenBucket(key="to", rpm=1, capacity=1)
    b.acquire()
    try:
        b.acquire(timeout=0.1)
    except RateLimitTimeout:
        return
    raise AssertionError("expected RateLimitTimeout")


def test_bucket_is_shared_across_processes():
    """Two processes must not each believe they own the full allowance."""
    state = os.environ["BENCH_STATE_DIR"]
    # acquire() returns 0.0 or raises; it never reports a non-zero wait for a
    # token it did not get. So drain by counting successful acquisitions.
    prog = (
        "from bench.ratelimit import TokenBucket, RateLimitTimeout\n"
        "b = TokenBucket(key='shared', rpm=60)\n"
        "n = 0\n"
        "try:\n"
        "    while True:\n"
        "        b.acquire(timeout=0.01)\n"
        "        n += 1\n"
        "except RateLimitTimeout:\n"
        "    pass\n"
        "print(n)"
    )
    env = {**os.environ, "BENCH_STATE_DIR": state, "PYTHONPATH": os.getcwd()}
    outs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                           text=True, env=env, timeout=60)
        assert r.returncode == 0, r.stderr
        outs.append(int(r.stdout.strip()))
    # Capacity is 60. If the bucket were per-process both would drain ~60.
    assert sum(outs) <= 62, f"bucket not shared across processes: {outs}"
    assert outs[1] < outs[0], f"second process should find it nearly empty: {outs}"


def test_unmetered_bucket_never_waits():
    b = TokenBucket(key="local", rpm=None)
    assert b.unmetered
    for _ in range(500):
        assert b.acquire() == 0.0
