"""Slots, because requests-per-minute did not describe the real limit."""

import os
import subprocess
import sys
import threading
import time

import pytest

from bench.concurrency import ConcurrencySlots, ConcurrencyTimeout


def test_second_holder_waits_for_the_first():
    s = ConcurrencySlots(key="k", limit=1)
    order = []
    with s.hold():
        order.append("first-in")

        def second():
            with s.hold():
                order.append("second-in")

        t = threading.Thread(target=second)
        t.start()
        time.sleep(0.4)
        order.append("first-still-holding")
    t.join(timeout=5)
    assert order == ["first-in", "first-still-holding", "second-in"]


def test_limit_allows_that_many_at_once():
    s = ConcurrencySlots(key="k2", limit=3)
    with s.hold(), s.hold(), s.hold():
        assert s.in_use() == 3
        with pytest.raises(ConcurrencyTimeout):
            with s.hold(timeout=0.2):
                pass


def test_slot_is_released_on_exception():
    s = ConcurrencySlots(key="k3", limit=1)
    with pytest.raises(ValueError):
        with s.hold():
            raise ValueError("boom")
    with s.hold(timeout=0.5):  # must not deadlock
        pass


def test_zero_limit_means_unlimited():
    s = ConcurrencySlots(key="local", limit=0)
    with s.hold(), s.hold(), s.hold():
        pass


def test_slots_are_shared_across_processes():
    """A slot held by another process must block this one."""
    state = os.environ["BENCH_STATE_DIR"]
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import time\n"
         "from bench.concurrency import ConcurrencySlots\n"
         "s = ConcurrencySlots(key='xp', limit=1)\n"
         "with s.hold():\n"
         "    print('held', flush=True)\n"
         "    time.sleep(3)\n"],
        stdout=subprocess.PIPE, text=True,
        env={**os.environ, "BENCH_STATE_DIR": state, "PYTHONPATH": os.getcwd()},
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        s = ConcurrencySlots(key="xp", limit=1)
        with pytest.raises(ConcurrencyTimeout):
            with s.hold(timeout=0.6):
                pass
    finally:
        holder.kill()
        holder.wait(timeout=5)

    # The kernel releases the flock when the killed process dies.
    s = ConcurrencySlots(key="xp", limit=1)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with s.hold(timeout=0.3):
                return
        except ConcurrencyTimeout:
            continue
    raise AssertionError("slot never freed after holder was killed")
