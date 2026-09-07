"""Drive OpenFOAM: mesh, solve, extract. Resumable by construction.

Every OpenFOAM call goes through `scripts/foam-env.sh`, a bash subshell that
sources the installation, because the login shell is fish and cannot source
OpenFOAM's bashrc. Nothing here depends on the interactive environment.

The pattern -- and the resumability -- is vendored from an earlier CFD study,
where a sweep running the better part of a day had to survive a closed laptop
lid. Camber's campaigns are longer. State is written after every completed
step, and a step whose marker exists is skipped rather than repeated.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FOAM_ENV = ROOT / "scripts" / "foam-env.sh"


class FoamError(RuntimeError):
    pass


@dataclass
class StepResult:
    name: str
    ok: bool
    seconds: float
    log: str
    message: str = ""


def foam(case: Path, command: str, log_name: str | None = None,
         timeout: float = 36000.0, parallel: bool = False,
         n_proc: int = 14) -> StepResult:
    """Run one OpenFOAM command inside a case directory."""
    case = Path(case)
    app = command.split()[0]
    log_name = log_name or f"log.{app}"
    log_path = case / log_name

    if parallel:
        argv = ["mpirun", "-np", str(n_proc), *command.split(), "-parallel"]
    else:
        argv = command.split()

    started = time.monotonic()
    with log_path.open("w") as log:
        proc = subprocess.run(
            ["bash", str(FOAM_ENV), *argv],
            cwd=case, stdout=log, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
    elapsed = time.monotonic() - started
    text = log_path.read_text(errors="ignore")

    # OpenFOAM sometimes exits 0 having printed a fatal error, so the log is
    # checked as well as the status. Trusting the exit code alone lets a failed
    # mesh flow downstream and fail later, somewhere less informative.
    fatal = "FOAM FATAL" in text
    ok = proc.returncode == 0 and not fatal
    message = ""
    if not ok:
        tail = [ln for ln in text.splitlines() if ln.strip()][-6:]
        message = " | ".join(tail)[:400]
    return StepResult(app, ok, elapsed, log_name, message)


@dataclass
class CaseState:
    """Which steps have completed. Written after each one."""

    steps: dict[str, dict]

    @classmethod
    def load(cls, case: Path) -> "CaseState":
        path = Path(case) / ".camber_state.json"
        if path.exists():
            return cls(json.loads(path.read_text()))
        return cls({})

    def save(self, case: Path) -> None:
        (Path(case) / ".camber_state.json").write_text(
            json.dumps(self.steps, indent=2))

    def done(self, name: str) -> bool:
        return bool(self.steps.get(name, {}).get("ok"))

    def record(self, case: Path, result: StepResult) -> None:
        self.steps[result.name] = asdict(result)
        self.save(case)


def mesh(case: Path, n_proc: int = 14, force: bool = False) -> list[StepResult]:
    """blockMesh -> snappyHexMesh -> topoSet -> checkMesh, resumably."""
    case = Path(case)
    state = CaseState.load(case)
    results: list[StepResult] = []

    plan = [
        ("blockMesh", "blockMesh", False),
        ("snappyHexMesh", "snappyHexMesh -overwrite", False),
        ("topoSet", "topoSet", False),
        ("checkMesh", "checkMesh -constant", False),
    ]
    for name, command, parallel in plan:
        if state.done(name) and not force:
            results.append(StepResult(name, True, 0.0, f"log.{name}", "skipped"))
            continue
        result = foam(case, command, parallel=parallel, n_proc=n_proc)
        state.record(case, result)
        results.append(result)
        if not result.ok:
            raise FoamError(f"{name} failed in {case}: {result.message}")
    return results


def n_subdomains(case: Path) -> int:
    """How many ranks this case is decomposed into, from the case itself.

    Read rather than passed in. `decomposePar` uses the dictionary while
    `mpirun -np` uses whatever the caller said, and when the two disagree
    OpenFOAM aborts with an MPI message that names neither number. One source
    of truth removes the whole class of mistake.
    """
    text = (Path(case) / "system" / "decomposeParDict").read_text()
    m = re.search(r"numberOfSubdomains\s+(\d+)", text)
    if not m:
        raise FoamError(f"{case}: decomposeParDict has no numberOfSubdomains")
    return int(m.group(1))


def solve(case: Path, parallel: bool = True) -> StepResult:
    case = Path(case)
    state = CaseState.load(case)
    if state.done("simpleFoam"):
        return StepResult("simpleFoam", True, 0.0, "log.simpleFoam", "skipped")
    n_proc = n_subdomains(case) if parallel else 1
    if parallel:
        foam(case, "decomposePar -force")
    result = foam(case, "simpleFoam", parallel=parallel, n_proc=n_proc)
    state.record(case, result)
    if not result.ok:
        raise FoamError(f"solve failed in {case}: {result.message}")
    if parallel:
        foam(case, "reconstructPar -latestTime")
    return result


_CELLS = re.compile(r"cells:\s+(\d+)")
_FORCE_LINE = re.compile(r"^\s*(\d+)\s+\((.*)\)\s*$")


def cell_count(case: Path) -> int | None:
    log = Path(case) / "log.checkMesh"
    if not log.exists():
        return None
    m = _CELLS.search(log.read_text(errors="ignore"))
    return int(m.group(1)) if m else None


def read_body_force(case: Path) -> dict[str, float] | None:
    """Total force on the airframe from the `forces` function object.

    This is the download: the airframe being pushed by its own rotor wash. It
    is the whole point of the L2 tier -- installed thrust is the sum of the
    prescribed disc thrusts minus this.
    """
    candidates = sorted(Path(case).glob("postProcessing/bodyForces/*/force*.dat"))
    if not candidates:
        return None
    rows = [ln for ln in candidates[-1].read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    if not rows:
        return None
    parts = rows[-1].replace("(", " ").replace(")", " ").split()
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    # time, then (total) (pressure) (viscous) as x y z triples.
    if len(values) < 4:
        return None
    return {"time": values[0], "fx": values[1], "fy": values[2], "fz": values[3]}
