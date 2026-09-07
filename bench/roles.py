"""The three jobs the bench delegates, and the prompts that keep them honest.

Each role exists because a model is measurably better at it than the others,
and each carries instructions that narrow it to that job. The implementer is
told not to redesign; the investigator is told to distinguish what it observed
from what it inferred. Those constraints are the difference between delegation
and hoping.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from .client import Client

IMPLEMENT_SYSTEM = """\
You implement a settled specification. You are not the architect.

Rules:
- The interface in the spec is fixed. If it looks wrong, implement it anyway \
and say so in a trailing NOTE block. Do not silently change signatures.
- Make the provided tests pass without weakening them. Editing a test to fit \
your implementation is a failure, not a solution.
- No new dependencies unless the spec names them.
- For physics or numerics: no invented constants, no unexplained magic \
numbers. Every empirical coefficient gets a comment naming its source, or a \
TODO(source) if the spec did not supply one.
- Match the style of the surrounding code you are shown.
- Output only the file contents unless asked otherwise.
"""

INVESTIGATE_SYSTEM = """\
You investigate and report. You do not make changes.

Rules:
- Separate what you OBSERVED from what you INFER. Label the two.
- If evidence is insufficient, say what specific artifact would settle it \
(a log line, a plot, a field dump) rather than guessing.
- When given images, describe what is actually visible before interpreting it.
- Be concrete: file paths, line numbers, numeric values.
- Lead with the single most likely cause, then the alternatives you ruled out.
"""

REVIEW_SYSTEM = """\
You review a diff for defects. Report only what you can justify.

Rules:
- Correctness first: wrong results, edge cases, silent failures, resource \
leaks. Style is out of scope unless it hides a bug.
- For every finding give a concrete failure scenario: inputs, then wrong output.
- If you are unsure whether something is a real defect, mark it UNCERTAIN \
rather than asserting it.
- An empty report is a valid outcome. Do not invent findings to seem useful.
"""


def _image_part(path: str | Path) -> dict:
    """Inline a local image as a data URI for an OpenAI-style vision message."""
    p = Path(path)
    data = p.read_bytes()
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def implement(
    client: Client,
    spec: str,
    *,
    tests: str | None = None,
    context: str | None = None,
    max_tokens: int = 32_000,
    route: str = "code",
) -> str:
    """Write code from a settled spec.

    Routed by *capability*, not by a fixed model name. `route="code"` resolves
    through the roster to whichever code model this account can actually reach
    quickly today; pass `route="implementer"` to force the heavyweight model
    when a task genuinely warrants minutes of waiting.

    The default used to be the legacy role name, which silently kept sending
    work to a model that takes ~280s per call long after a 27 tok/s
    alternative had been measured and wired up. Defaulting to the capability
    is what makes the roster actually load-bearing rather than decorative.
    """
    parts = [f"# Specification\n\n{spec}"]
    if tests:
        parts.append(f"# Tests that must pass\n\n```python\n{tests}\n```")
    if context:
        parts.append(f"# Surrounding code, for style and interfaces\n\n{context}")
    payload = client.complete(
        route,
        [
            {"role": "system", "content": IMPLEMENT_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return client.text(payload)


def investigate(
    client: Client,
    question: str,
    *,
    context: str | None = None,
    images: list[str | Path] | None = None,
    max_tokens: int = 16_000,
    route: str | None = None,
) -> str:
    """Diagnose something.

    Routes to the `vision` capability when images are supplied and `reason`
    otherwise, because the vision-capable models are slower and there is no
    reason to pay that when there is nothing to look at.
    """
    content: list[dict] = [{"type": "text", "text": question}]
    if context:
        content.append({"type": "text", "text": f"\n\n# Context\n\n{context}"})
    for img in images or []:
        content.append(_image_part(img))

    payload = client.complete(
        route or ("vision" if images else "reason"),
        [
            {"role": "system", "content": INVESTIGATE_SYSTEM},
            {"role": "user", "content": content},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return client.text(payload)


def review(
    client: Client,
    diff: str,
    *,
    focus: str | None = None,
    role: str = "code",
    max_tokens: int = 16_000,
) -> str:
    """Review a diff. Routed to the code capability by default."""
    prompt = f"# Diff\n\n```diff\n{diff}\n```"
    if focus:
        prompt = f"Focus: {focus}\n\n{prompt}"
    payload = client.complete(
        role,
        [
            {"role": "system", "content": REVIEW_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return client.text(payload)
