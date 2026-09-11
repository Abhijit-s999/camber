"""Background jobs: the long runs, outside the UI's lifetime.

A topology run takes two hours and a CFD sweep twenty minutes. Neither can live
inside a request, and neither should die because the window was closed. Each
job is the `camber` CLI in its own process group, writing a log in the project,
with a small JSON record beside it. The record is the truth: on restart the
manager reads the records, checks whether each process is still alive, and
carries on reporting -- or marks it interrupted if it is not.

Progress comes from the log, because the CLI already prints it. A parser per
kind turns lines into a fraction and a sentence a person would want to read.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Job:
    id: str
    kind: str
    label: str
    params: dict
    argv: list
    outputs: dict
    status: str = "queued"          # running, done, failed, cancelled, interrupted
    pid: int | None = None
    started: float | None = None
    finished: float | None = None
    exit_code: int | None = None
    progress: float | None = None
    detail: str = ""

    def public(self) -> dict:
        d = asdict(self)
        d.pop("argv")
        d["elapsed"] = ((self.finished or time.time()) - self.started
                        if self.started else 0.0)
        return d


def _args(kind: str, spec: str, params: dict, out: dict) -> list[str]:
    """CLI arguments for a kind of job. Every value is validated into a string
    here, so nothing a UI sends ends up interpreted by a shell."""
    f = lambda k, d: str(float(params.get(k, d)))       # noqa: E731
    i = lambda k, d: str(int(params.get(k, d)))         # noqa: E731
    if kind == "optimise":
        return ["optimise", spec, "--population", i("population", 40),
                "--generations", i("generations", 30), "--seed", i("seed", 1),
                "--verbose", "--front-json", out["front"], "--out", out["best"]]
    if kind == "generate":
        a = ["generate", spec, "--element", f("element", 0.003),
             "--thickness", f("thickness", 0.012),
             "--iterations", i("iterations", 80),
             "--min-feature", f("min_feature", 3.0), "--verbose",
             "--out", out["stl"], "--density-out", out["density"],
             "--into", out["vehicle"]]
        if params.get("mode", "price") == "mass":
            a += ["--mass", f("mass", 40.0)]
        else:
            a += ["--price", f("price", 1.0)]
        return a
    if kind == "aero":
        return ["aero", spec, "--out", out["dir"], "--n", i("n", 6),
                "--iterations", i("iterations", 600)]
    if kind == "record":
        a = ["sim", spec, "--pilot", str(params.get("pilot", "gates")),
             "--duration", f("duration", 20.0), "--speed", f("speed", 14.0),
             "--record", out["video"]]
        if params.get("aero"):
            a += ["--aero", str(params["aero"])]
        return a
    raise ValueError(f"unknown job kind {kind!r}")


OUTPUTS = {
    "optimise": lambda r: {"front": f"{r}/front.json", "best": f"{r}/best.json"},
    "generate": lambda r: {"stl": f"{r}/frame.stl", "density": f"{r}/density.npy",
                           "vehicle": f"{r}/framed.json"},
    "aero": lambda r: {"dir": r, "table": f"{r}/aero.json"},
    "record": lambda r: {"video": f"{r}/flight.mp4"},
}

LABELS = {"optimise": "Optimise layout", "generate": "Grow frame",
          "aero": "Aerodynamics", "record": "Record flight"}


def _progress(kind: str, params: dict, log: str) -> tuple[float | None, str]:
    lines = log.splitlines()
    if kind == "generate":
        total = int(params.get("iterations", 80))
        its = [ln for ln in lines if re.match(r"\s+it\s+\d+", ln)]
        if its:
            m = re.match(r"\s+it\s+(\d+)\s+p\s+([\d.]+)(?:\s+b\s+([\d.]+))?.*vol\s+([\d.]+)",
                         its[-1])
            if m:
                n = int(m.group(1))
                return n / total, (f"iteration {n} of {total}"
                                   + (" -- sharpening" if m.group(3) and float(m.group(3)) < 16 else ""))
        if any("mesh " in ln for ln in lines):
            return 1.0, "meshing and drilling"
        return 0.0, "setting up the finite-element model"
    if kind == "aero":
        n = int(params.get("n", 6))
        done = sum(1 for ln in lines if " cells, " in ln)
        started = [ln.strip() for ln in lines if re.match(r"\s{2}\S+: alpha", ln)]
        now = started[-1].split(",")[0] if started else "building the airframe surface"
        return done / n, (f"case {min(done + 1, n)} of {n}: {now}" if done < n
                          else "writing the table")
    if kind == "optimise":
        total = int(params.get("generations", 30))
        gens = [ln for ln in lines if re.match(r"\s+gen\s+\d+", ln)]
        if gens:
            g = int(re.match(r"\s+gen\s+(\d+)", gens[-1]).group(1))
            return (g + 1) / total, f"generation {g + 1} of {total}"
        return 0.0, "starting"
    if kind == "record":
        return None, "flying and recording"
    return None, ""


class JobManager:
    def __init__(self, project) -> None:
        self.project = project
        self.dir = project.root / "jobs"
        self.dir.mkdir(exist_ok=True)

    # --- records -----------------------------------------------------------
    def _record(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def _log(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.log"

    def _save(self, job: Job) -> None:
        self._record(job.id).write_text(json.dumps(asdict(job), indent=1))

    def _load(self, path: Path) -> Job:
        return Job(**json.loads(path.read_text()))

    # --- control -------------------------------------------------------------
    def start(self, kind: str, params: dict | None = None,
              spec: str | None = None) -> Job:
        params = dict(params or {})
        if kind not in OUTPUTS:
            raise ValueError(f"unknown job kind {kind!r}")
        running = [j for j in self.list() if j.kind == kind and j.status == "running"]
        if running and kind in ("generate", "aero"):
            # Two of these at once share memory bandwidth and both slow to a
            # crawl (measured: CFD concurrency was 0.34x the throughput).
            raise RuntimeError(f"a {LABELS[kind].lower()} job is already running")
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        rel = f"results/{kind}-{job_id}"
        (self.project.root / rel).mkdir(parents=True, exist_ok=True)
        outputs = OUTPUTS[kind](rel)
        # Jobs run on a *frozen copy* of the design: editing on while a two-hour
        # run is in flight must not change what it is computing.
        frozen = self.project.root / rel / "vehicle.json"
        from ..spec import io
        io.save_json(self.project.vehicle, frozen)
        argv = [sys.executable, "-m", "camber.cli",
                *_args(kind, spec or str(frozen), params, outputs)]
        job = Job(job_id, kind, LABELS[kind], params, argv, outputs)
        env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "glfw"),
                   PYTHONUNBUFFERED="1")
        log = open(self._log(job_id), "w")
        proc = subprocess.Popen(argv, cwd=self.project.root, stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, env=env)
        job.pid, job.started, job.status = proc.pid, time.time(), "running"
        self._save(job)
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status == "running" and job.pid:
            try:
                os.killpg(job.pid, signal.SIGTERM)       # the whole group: MPI too
            except ProcessLookupError:
                pass
            job.status, job.finished = "cancelled", time.time()
            self._save(job)
        return job

    # --- reading ---------------------------------------------------------------
    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return False
        try:                                          # reaped zombie of ours?
            pid_done, status = os.waitpid(pid, os.WNOHANG)
            if pid_done == pid:
                return False
        except ChildProcessError:
            pass
        stat = Path(f"/proc/{pid}/stat")
        return not (stat.exists() and ") Z" in stat.read_text())

    def get(self, job_id: str) -> Job:
        job = self._load(self._record(job_id))
        if job.status == "running":
            text = self.log(job_id, tail=None)
            job.progress, job.detail = _progress(job.kind, job.params, text)
            if not job.pid or not self._alive(job.pid):
                job.finished = time.time()
                ok = self._succeeded(job, text)
                job.status = "done" if ok else ("failed" if job.started else "interrupted")
                if ok:
                    job.progress, job.detail = 1.0, "finished"
                else:
                    job.detail = self._failure(text)
                self._save(job)
        return job

    def _succeeded(self, job: Job, log: str) -> bool:
        key = {"optimise": "front", "generate": "vehicle", "aero": "table",
               "record": "video"}[job.kind]
        out = self.project.root / job.outputs[key]
        return out.exists() and "Traceback" not in log

    @staticmethod
    def _failure(log: str) -> str:
        lines = [ln for ln in log.splitlines() if ln.strip()]
        for ln in reversed(lines):
            if "Error" in ln or "error" in ln or "WARNING" in ln:
                return ln.strip()[:240]
        return (lines[-1].strip()[:240] if lines else "the job stopped without output")

    def list(self) -> list[Job]:
        jobs = [self.get(p.stem) for p in self.dir.glob("*.json")]
        return sorted(jobs, key=lambda j: j.started or 0, reverse=True)

    def log(self, job_id: str, tail: int | None = 60) -> str:
        path = self._log(job_id)
        if not path.exists():
            return ""
        text = path.read_text(errors="replace")
        if tail is None:
            return text
        return "\n".join(text.splitlines()[-tail:])
