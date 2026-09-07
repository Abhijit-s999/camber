"""The bench as an MCP server, so Claude Code can delegate natively.

This is the point of the whole package. Rather than piping text between CLIs,
Opus 5 stays the orchestrator and calls `implement`, `investigate` and `review`
as ordinary tools -- while the tokens those calls consume are spent on NVIDIA's
GPUs instead of the orchestrating session's budget.

Written against mcp 2.x, where FastMCP became MCPServer.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.mcpserver import MCPServer

from . import roles
from .cache import Cache
from .callog import CallLog
from .client import Client, NotConfigured
from .config import MODELS, PROVIDERS, credential_for, provider_for, resolve
from .ratelimit import bucket_for

server = MCPServer(
    name="bench",
    instructions=(
        "Delegation to open-weight models. Use `implement` for code you have "
        "already specified -- it is routed to the strongest pure coder and "
        "expects a settled interface plus tests. Use `investigate` for "
        "diagnosis, especially anything involving images (plots, meshes, "
        "screenshots): it is the only route with vision. Use `review` for a "
        "second opinion on a diff. Check `budget` before a burst of calls."
    ),
    version="0.0.1",
)

_client: Client | None = None


def client() -> Client:
    global _client
    if _client is None:
        _client = Client(cache=Cache(), callog=CallLog())
    return _client


@server.tool(
    description=(
        "Implement a settled specification. Routed to deepseek-v4-pro-0813, "
        "which leads SWE-bench and LiveCodeBench. Give it a fixed interface "
        "and failing tests; it is told not to redesign. Returns file contents."
    )
)
def implement(spec: str, tests: str = "", context: str = "") -> str:
    try:
        return roles.implement(
            client(), spec, tests=tests or None, context=context or None
        )
    except NotConfigured as exc:
        return f"NOT CONFIGURED: {exc}"


@server.tool(
    description=(
        "Investigate or diagnose something. Routed to kimi-k3, which has the "
        "strongest agentic scores here and native vision. `images` takes paths "
        "to local PNG/JPEG files -- residual plots, mesh screenshots, ParaView "
        "renders -- and is the reason to prefer this over `review` for "
        "anything visual. Reports; does not change files."
    )
)
def investigate(question: str, context: str = "", images: list[str] | None = None) -> str:
    paths: list[str] = []
    for p in images or []:
        if not Path(p).is_file():
            return f"ERROR: no such image: {p}"
        paths.append(p)
    try:
        return roles.investigate(
            client(), question, context=context or None, images=paths or None
        )
    except NotConfigured as exc:
        return f"NOT CONFIGURED: {exc}"


@server.tool(
    description=(
        "Review a unified diff for correctness defects. Returns findings with "
        "concrete failure scenarios, or an empty report. Set role to "
        "'investigator' for a second opinion from the other model."
    )
)
def review(diff: str, focus: str = "", role: str = "implementer") -> str:
    if role not in MODELS:
        return f"ERROR: unknown role {role!r}; known: {', '.join(sorted(MODELS))}"
    try:
        return roles.review(client(), diff, focus=focus or None, role=role)
    except NotConfigured as exc:
        return f"NOT CONFIGURED: {exc}"


@server.tool(
    description=(
        "Report remaining rate-limit budget, per-provider configuration, and "
        "call-log totals. Cheap and local -- it makes no model call. Worth "
        "checking before dispatching a batch."
    )
)
def budget() -> str:
    lines: list[str] = []
    for role, model in MODELS.items():
        cred = credential_for(model)
        if cred.rpm is None:
            state = "unmetered"
        else:
            state = f"{bucket_for(cred.bucket_key, cred.rpm).peek():.1f}/{cred.rpm} tokens"
        ready = "ready" if cred.configured else f"needs ${cred.source}"
        lines.append(
            f"{role:<14} {model.id:<24} {state:<20} {ready}  "
            f"[bucket {cred.bucket_key}]"
        )

    summary = CallLog().summary()
    lines.append("")
    lines.append(f"calls logged: {summary.get('calls', 0)}")
    for model, s in (summary.get("by_model") or {}).items():
        lines.append(
            f"  {model:<24} {s['calls']:>4} calls, {s['cached']:>4} cached, "
            f"{s['failed']:>3} failed, {s['in']:>8} in, {s['out']:>8} out"
        )
    return "\n".join(lines)


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
