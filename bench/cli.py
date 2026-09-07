"""`bench` -- a small CLI for the things you want without an MCP round trip."""

from __future__ import annotations

import argparse
import sys

from .cache import Cache
from .callog import CallLog
from .client import Client, NotConfigured
from .config import MODELS, PROVIDERS, credential_for, provider_for
from .ratelimit import bucket_for


def cmd_status(_: argparse.Namespace) -> int:
    for role, model in MODELS.items():
        cred = credential_for(model)
        if cred.rpm is None:
            budget = "unmetered"
        else:
            budget = f"{bucket_for(cred.bucket_key, cred.rpm).peek():.1f}/{cred.rpm}"
        mark = "ok " if cred.configured else "!! "
        need = "" if cred.configured else f"  (missing ${cred.source})"
        print(f"{mark}{role:<14} {model.id:<24} {budget:<12} "
              f"bucket={cred.bucket_key}{need}")

    print()
    for name, provider in PROVIDERS.items():
        print(f"   {name:<12} {provider.base_url}")

    print()
    summary = CallLog().summary()
    print(f"   calls logged: {summary.get('calls', 0)}")
    for model, s in (summary.get("by_model") or {}).items():
        print(f"     {model:<24} {s['calls']:>4} calls  {s['cached']:>4} cached  "
              f"{s['failed']:>3} failed")
    cached = Cache().stats()
    if cached:
        print(f"   cache: {sum(cached.values())} responses "
              f"({', '.join(f'{k}={v}' for k, v in cached.items())})")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Smallest possible live check that a provider actually answers."""
    client = Client()
    try:
        payload = client.complete(
            args.role,
            [{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
            use_cache=not args.no_cache,
        )
    except NotConfigured as exc:
        print(f"not configured: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"failed: {exc}", file=sys.stderr)
        return 1
    print(client.text(payload))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """List model ids the provider actually serves.

    NIM namespaces ids as `publisher/model`, while its web catalogue shows the
    bare name. Taking the short name produces a 404 whose body is the string
    "404 page not found", which says nothing about why. This is the check.
    """
    import httpx

    from .config import PROVIDERS, credential_for, resolve
    cred = credential_for(resolve(args.role))
    provider = PROVIDERS[resolve(args.role).provider]
    headers = {"Authorization": f"Bearer {cred.key}"} if cred.key else {}
    try:
        r = httpx.get(f"{provider.base_url.rstrip('/')}/models",
                      headers=headers, timeout=60)
    except Exception as exc:  # noqa: BLE001
        print(f"failed: {exc}", file=sys.stderr)
        return 1
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
        return 1
    ids = sorted(m["id"] for m in r.json().get("data", []))
    if args.pattern:
        ids = [i for i in ids if args.pattern.lower() in i.lower()]
    for i in ids:
        print(i)
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Measure which models this account can actually invoke, and how fast.

    The catalogue endpoint lists models the account cannot call -- most of them
    return "Not found for account" -- and among those it can call, latency
    spans three orders of magnitude. Neither fact is discoverable from the
    listing, so the roster is built from measurement.

    Classification is cheap because failure is fast: errors return in about
    0.1s while real inference queues for minutes, so a short timeout separates
    "unavailable" from "provisioned but slow" without waiting for either.
    """
    import time

    import httpx

    from .config import PROVIDERS, credential_for, resolve
    from .roster import CAPABILITIES, Availability, save

    cred = credential_for(resolve("investigator"))
    provider = PROVIDERS["nvidia"]
    headers = {"Authorization": f"Bearer {cred.key}", "Content-Type": "application/json"}
    base = provider.base_url.rstrip("/")

    wanted = args.models or sorted({c.model_id for v in CAPABILITIES.values() for c in v})
    results: list[Availability] = []
    for model_id in wanted:
        started = time.time()
        try:
            r = httpx.post(f"{base}/chat/completions", headers=headers, json={
                "model": model_id, "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1, "temperature": 0}, timeout=args.timeout)
            elapsed = time.time() - started
            if r.status_code == 200:
                a = Availability(model_id, "fast", round(elapsed, 2))
            elif r.status_code == 429:
                a = Availability(model_id, "queued", round(elapsed, 2), "rate limited")
            else:
                a = Availability(model_id, "unavailable", round(elapsed, 2),
                                 f"HTTP {r.status_code}")
        except httpx.TimeoutException:
            a = Availability(model_id, "queued", args.timeout, "no reply within timeout")
        except Exception as exc:  # noqa: BLE001
            a = Availability(model_id, "unavailable", None, type(exc).__name__)

        # Admission is not throughput. Measure generation too when asked,
        # because that is what a real task waits on.
        if a.status == "fast" and args.tokens > 1:
            started = time.time()
            try:
                r = httpx.post(f"{base}/chat/completions", headers=headers, json={
                    "model": model_id,
                    "messages": [{"role": "user",
                                  "content": "Count from 1 to 200, comma separated."}],
                    "max_tokens": args.tokens, "temperature": 0},
                    timeout=args.generate_timeout)
                elapsed = time.time() - started
                if r.status_code == 200:
                    used = (r.json().get("usage") or {}).get("completion_tokens") or 0
                    a.generate_s = round(elapsed, 1)
                    a.tokens_per_s = round(used / elapsed, 1) if elapsed > 0 else None
                else:
                    a.detail = (a.detail + f" gen HTTP {r.status_code}").strip()
            except httpx.TimeoutException:
                a.status = "queued"
                a.generate_s = args.generate_timeout
                a.detail = "admitted fast but generation timed out"
            except Exception as exc:  # noqa: BLE001
                a.detail = (a.detail + f" gen {type(exc).__name__}").strip()

        results.append(a)
        mark = {"fast": "FAST", "queued": "QUEUE", "unavailable": "----"}[a.status]
        lat = f"{a.latency_s:5.1f}s" if a.latency_s is not None else "     -"
        thr = (f" {a.tokens_per_s:6.1f} tok/s in {a.generate_s:5.1f}s"
               if a.tokens_per_s else "")
        print(f"{mark:<5} admit {lat}{thr}  {model_id}  {a.detail}", flush=True)

    path = save(results)
    fast = sum(1 for r in results if r.status == "fast")
    print(f"\n{fast} fast, "
          f"{sum(1 for r in results if r.status == 'queued')} queued, "
          f"{sum(1 for r in results if r.status == 'unavailable')} unavailable")
    print(f"saved to {path}")
    return 0


def cmd_roster(_: argparse.Namespace) -> int:
    from .roster import CAPABILITIES, load, resolve_capability
    seen = load()
    if not seen:
        print("no probe data yet -- run `bench probe`. Showing first choices only.\n")
    for cap, candidates in CAPABILITIES.items():
        try:
            chosen = resolve_capability(cap).model_id
        except RuntimeError:
            chosen = None
        print(f"{cap}:")
        for c in candidates:
            a = seen.get(c.model_id)
            state = a.status if a else "unprobed"
            lat = f" {a.latency_s:.1f}s" if a and a.latency_s is not None else ""
            mark = "->" if c.model_id == chosen else "  "
            print(f"  {mark} {c.model_id:<45} {state:<12}{lat:<8} {c.why}")
        print()
    return 0


def cmd_serve(_: argparse.Namespace) -> int:
    from .mcp_server import main as serve
    serve()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="providers, budget and call totals").set_defaults(
        func=cmd_status
    )
    sub.add_parser("serve", help="run the MCP server on stdio").set_defaults(
        func=cmd_serve
    )
    sub.add_parser("roster", help="capabilities and which model serves each").set_defaults(
        func=cmd_roster
    )

    probe = sub.add_parser("probe", help="measure model availability and latency")
    probe.add_argument("models", nargs="*", help="default: everything in the roster")
    probe.add_argument("--timeout", type=float, default=9.0,
                       help="admission probe timeout")
    probe.add_argument("--tokens", type=int, default=1,
                       help=">1 also measures generation throughput, which is "
                            "what a real task waits on")
    probe.add_argument("--generate-timeout", type=float, default=90.0)
    probe.set_defaults(func=cmd_probe)

    models = sub.add_parser("models", help="list model ids the provider serves")
    models.add_argument("--role", default="implementer", choices=sorted(MODELS))
    models.add_argument("pattern", nargs="?", help="substring filter")
    models.set_defaults(func=cmd_models)

    ask = sub.add_parser("ask", help="one-shot prompt to a role")
    ask.add_argument("role", choices=sorted(MODELS))
    ask.add_argument("prompt")
    ask.add_argument("--max-tokens", type=int, default=4096,
                 help="reasoning models spend most of this before "
                      "emitting anything visible")
    ask.add_argument("--no-cache", action="store_true")
    ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
