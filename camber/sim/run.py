"""Fly a controller: headless, in the interactive viewer, or to a video."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Flight:
    """What happened, for plotting and for tests."""

    t: list = field(default_factory=list)
    position: list = field(default_factory=list)
    velocity: list = field(default_factory=list)
    euler: list = field(default_factory=list)
    thrust: list = field(default_factory=list)
    crashed: bool = False

    def arrays(self):
        return {k: np.asarray(getattr(self, k)) for k in
                ("t", "position", "velocity", "euler", "thrust")}

    @property
    def top_speed(self) -> float:
        v = np.asarray(self.velocity)
        return float(np.linalg.norm(v, axis=1).max()) if len(v) else 0.0


class _FFmpeg:
    """Raw RGB frames piped to the system ffmpeg as H.264.

    Not imageio: without its separate ffmpeg plugin it picks a writer by file
    extension, and for `.mp4` it chose TIFF -- eight bytes of output and an
    error about an `fps` keyword a TIFF has no use for.
    """

    def __init__(self, path: str, width: int, height: int, fps: int) -> None:
        import shutil
        import subprocess
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("recording needs ffmpeg on the PATH")
        self.proc = subprocess.Popen(
            [exe, "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
             str(path)], stdin=subprocess.PIPE)

    def write(self, frame) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed to write the recording")


def fly(sim, controller, duration: float, *, viewer: bool = False,
        record: str | None = None, camera: str = "chase", fps: int = 30,
        width: int = 1280, height: int = 720, realtime: bool = True,
        log_every: int = 10) -> Flight:
    """Run `controller` on `sim` for `duration` seconds.

    `viewer` opens MuJoCo's interactive window and paces the simulation to
    wall-clock time; `record` writes an MP4 from the named camera instead. The
    physics is identical either way -- the viewer only watches.
    """
    import mujoco

    state = sim.reset()
    controller.reset(sim)
    log = Flight()
    steps = int(round(duration / sim.dt))
    frame_every = max(1, int(round(1.0 / (fps * sim.dt))))
    renderer = writer = None
    if record:
        renderer = mujoco.Renderer(sim.model, height, width)
        writer = _FFmpeg(record, width, height, fps)
    handle = mujoco.viewer.launch_passive(sim.model, sim.data) if viewer else None
    try:
        wall0 = time.perf_counter()
        for i in range(steps):
            state = sim.step(controller.update(state, sim))
            if i % log_every == 0:
                log.t.append(state.t)
                log.position.append(state.position)
                log.velocity.append(state.velocity)
                log.euler.append(state.euler)
                log.thrust.append(state.thrust)
            if sim.crashed:
                log.crashed = True
                break
            if writer is not None and i % frame_every == 0:
                renderer.update_scene(sim.data, camera=camera)
                writer.write(renderer.render())
            if handle is not None:
                if not handle.is_running():
                    break
                handle.sync()
                if realtime:
                    lag = state.t - (time.perf_counter() - wall0)
                    if lag > 0:
                        time.sleep(lag)
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if handle is not None:
            handle.close()
    return log
