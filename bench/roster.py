"""Which model does which job, and how that gets decided.

The first version of this bench had three hardcoded roles pointing at two
models, both of them the largest available. That was a mistake with two parts.

The first is routing by prestige rather than by fit. Writing a file parser or a
CLI argument block against a settled interface does not need a 1.6-trillion
parameter reasoning model; a 20B code model does it as well and returns in
seconds instead of queueing for minutes. Most of the latency that made
delegation look pointless was a consequence of that choice, not of the free
tier.

The second is ignoring specialists entirely. The provider hosts document
parsers, chart-to-table models, embedding models and reward models. Several map
directly onto work this project actually needs -- turning datasheets into
catalogue rows, reading performance curves published only as images, matching
part numbers -- and none of that is a job for a general chat model.

So capability is the unit here, not a fixed role, and availability is
*measured* rather than assumed: `bench probe` records what this account can
actually invoke and how fast, because the catalogue endpoint lists models the
account cannot call.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import state_dir


@dataclass(frozen=True)
class Candidate:
    """One model considered for a capability, with why it is a candidate."""

    model_id: str
    #: Rough scale, for the cost/latency intuition the router needs.
    size_hint: str
    why: str


#: Capabilities, each with an ordered preference list. The router walks the
#: list and takes the first candidate the probe found usable, so an unavailable
#: or slow model degrades to the next rather than failing the job.
CAPABILITIES: dict[str, list[Candidate]] = {
    # Bounded implementation from a settled spec plus failing tests. Wants
    # speed and correctness, not judgement -- the design is already done.
    "code": [
        Candidate("poolside/laguna-xs-2.1", "xs",
                  "code-specialised, measured at 0.4s -- the default for bounded work"),
        Candidate("openai/gpt-oss-20b", "20B",
                  "open-weight general model, strong on code, 0.4s"),
        Candidate("nvidia/nemotron-3.5-lightning-30b-a3b", "30B MoE",
                  "few active parameters, 0.9s"),
        Candidate("deepseek-ai/deepseek-v4-pro-0813", "1.6T",
                  "best SWE-bench of the set, but queues for minutes; "
                  "escalation only, never the default"),
    ],
    # Problems where the answer is not already known: tricky numerics, a design
    # trade-off, an argument that has to hold together.
    "reason": [
        Candidate("nvidia/nemotron-3.5-lightning-30b-a3b", "30B MoE",
                  "few active parameters, built for latency, 0.9s"),
        Candidate("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "30B MoE",
                  "reasoning-tuned sibling, 1.0s"),
        Candidate("minimaxai/minimax-m3", "-", "fast general model, 0.7s"),
        Candidate("moonshotai/kimi-k3", "2.8T",
                  "strongest available; queues for minutes, so escalation only"),
    ],
    # Anything requiring sight: residual plots, mesh screenshots, ParaView
    # renders, a photograph of a part.
    "vision": [
        Candidate("meta/llama-3.2-11b-vision-instruct", "11B",
                  "the only vision model that answered promptly, 8.7s"),
        Candidate("meta/llama-3.2-90b-vision-instruct", "90B",
                  "larger sibling, queues"),
        Candidate("moonshotai/kimi-k3", "2.8T", "native vision, but queues"),
    ],
    # Turning a published curve into numbers. The UIUC archive alone holds
    # 1450 plot images, and manufacturer data is often only ever a picture.
    "chart": [
        Candidate("google/deplot", "-", "plot image to data table, specifically"),
    ],
    # Datasheets and tables into catalogue rows. This is the tool that unblocks
    # the motor catalogue, which P3 needs and does not have.
    "parse": [
        Candidate("nvidia/nemotron-parse", "-", "document and table extraction"),
    ],
    # Fuzzy part-number and description matching inside the catalogue. An
    # embedding model is the right instrument; a chat model is not.
    "embed": [
        Candidate("nvidia/nv-embedqa-mistral-7b-v2", "7B", "retrieval embeddings"),
        Candidate("snowflake/arctic-embed-l", "-", "compact retrieval embeddings"),
        Candidate("nvidia/embed-qa-4", "-", "NVIDIA QA embeddings"),
    ],
    # Ranking several candidate answers instead of trusting one.
    "rank": [
        Candidate("nvidia/nemotron-4-340b-reward", "340B", "reward model, scores outputs"),
    ],
    # Screening text before it is trusted or stored. Sub-second, and the right
    # instrument for a job a general chat model does slowly and less reliably.
    "safety": [
        Candidate("nvidia/nemotron-3.5-content-safety", "-", "0.2s"),
        Candidate("nvidia/llama-3.1-nemoguard-8b-content-safety", "8B", "0.2s"),
    ],
}


@dataclass
class Availability:
    """What `bench probe` learned. Measured, not assumed.

    Two numbers, because they answer different questions and the first one
    alone is misleading. `latency_s` with a one-token reply measures *queue
    admission* -- whether the provider takes the request now. `tokens_per_s`
    measures generation, which is what a real task actually waits on.

    A model can be admitted in 0.3 s and still need minutes for a six-thousand
    token file. Reporting only the first number produced exactly that mistake:
    seven models were called "fast" on the strength of a metric that did not
    describe the work they were about to be given.
    """

    model_id: str
    status: str          # "fast" | "queued" | "unavailable"
    latency_s: float | None = None
    detail: str = ""
    #: Generation throughput, when a depth probe was run. None if not measured.
    tokens_per_s: float | None = None
    #: Wall time of the depth probe, which is what a caller actually waits.
    generate_s: float | None = None


def probe_path() -> Path:
    return state_dir() / "availability.json"


def save(results: list[Availability]) -> Path:
    path = probe_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    return path


def load() -> dict[str, Availability]:
    path = probe_path()
    if not path.exists():
        return {}
    return {d["model_id"]: Availability(**d) for d in json.loads(path.read_text())}


def resolve_capability(capability: str, allow_queued: bool = True) -> Candidate:
    """First candidate for this capability that the probe found usable.

    Falls back to the head of the list when no probe has been run, so the
    roster still works on a fresh checkout -- it is simply guessing rather than
    knowing, which `bench status` says plainly.
    """
    if capability not in CAPABILITIES:
        raise KeyError(
            f"unknown capability {capability!r}; have {sorted(CAPABILITIES)}"
        )
    candidates = CAPABILITIES[capability]
    seen = load()
    if not seen:
        return candidates[0]

    ok = {"fast"} | ({"queued"} if allow_queued else set())
    for c in candidates:
        if seen.get(c.model_id) and seen[c.model_id].status in ok:
            return c
    raise RuntimeError(
        f"no usable model for capability {capability!r}. Tried: "
        + ", ".join(f"{c.model_id} ({seen.get(c.model_id).status if seen.get(c.model_id) else 'unprobed'})"
                    for c in candidates)
    )
