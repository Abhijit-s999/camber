# Technical decisions

Every non-obvious choice in Camber, with the value, the reasoning, and the
source. Tagged so that code comments and this document can refer to each other.
The format is lifted from an earlier CFD study, where it earned its keep.

**Status key**
`FIXED` — decided and implemented.
`OPEN` — needs a decision, or a study before it can be settled.
`PROVISIONAL` — implemented with a defensible default, worth revisiting.

---

## 0. The bench

### AI-01 · One token bucket per account, not per key — `FIXED`

**Value:** a single cross-process `TokenBucket` keyed by
`Provider.bucket_key`. Kimi K3 and DeepSeek V4-Pro both map to `nvidia`, and
share one 40 rpm allowance. `BENCH_SPLIT_BUCKETS=1` opts into one bucket per
distinct credential.

**Revised 2026-09-07.** Two separate keys arrived, one per model, and the
bucket was briefly split per credential on the assumption that separate keys
meant separate accounts. They do not here — both belong to one account, so
splitting would have issued 80 rpm against a 40 rpm limit.

The default is now the conservative one, and the asymmetry is why:
under-issuing merely runs slower than necessary, while over-issuing produces
sustained 429s, and retries against an exhausted account make that worse rather
than better. Prefer the recoverable failure.

**Reasoning:** NVIDIA's free-tier limit is imposed per account. Two clients with
independent limiters would each believe they owned the full 40 rpm, and the
account would start returning 429 to whichever request arrived third. The bucket
is file-backed and `flock`-guarded rather than in-process because the MCP
server, the overnight daemon and an interactive call can all run at once.

The lock is never held across a sleep — a waiter computes its deficit, releases,
sleeps, and re-competes. Holding it would serialise every other process behind
the unluckiest caller.

**Source:** NVIDIA build.nvidia.com free tier, ~40 req/min for Developer
Program members. Verify against your own account before trusting the number.

### AI-14 · The binding limit is concurrency, not requests per minute — `FIXED`

**Value:** `bench/concurrency.py`, a cross-process semaphore of flock'd slot
files. `Provider.max_concurrency` defaults to **1** for NVIDIA. The slot is
held for the whole request; the rate token is only spent to start it.

**Reasoning — measured, not documented.** Four concurrent requests on
2026-09-07 gave:

```
[0] deepseek-v4-pro   HTTP 504 in 302.1s
[1] kimi-k3           HTTP 429 in   0.1s
[2] deepseek-v4-pro   HTTP 504 in 302.1s
[3] kimi-k3           ok      in 171.3s
```

The 429 arrived in 0.1 s with the account nowhere near 40 requests in the
preceding minute, while another request to the same model was in flight. A
token bucket cannot express this: it counts starts and forgets them, whereas a
request that runs for three minutes occupies capacity for all three. Both
limiters now apply, and the rate bucket is no longer the binding one.

Slots are files rather than a counter because a crashed process must release
its slot. The kernel drops an flock when the holder's descriptors close; a
counter would leak until someone noticed.

**Consequence for the daemon:** parallel dispatch does not help on this
provider. Overnight campaigns are a single-file queue, not a fan-out.

### AI-15 · DeepSeek V4-Pro is not currently usable on the NIM free tier — `OPEN`

**Observed:** every request to `deepseek-ai/deepseek-v4-pro-0813` returned
**HTTP 504 at 302 s**, twice, consistently — a gateway timeout from NVIDIA's
side rather than a client timeout. `moonshotai/kimi-k3` on the same account and
key succeeded at 154 s and 171 s.

**Consequence:** the implementer role, which the whole three-model split rests
on, has no working model on this provider right now. Options, in the order I
would try them: `deepseek-v4-flash-0731` (same family, 13B active, should be
far quicker); a different time of day, since free-tier queue depth varies; or
OpenRouter / DeepSeek direct for latency-sensitive work, which `AI-09` already
supports as a base-URL change.

**Not yet established:** whether the 504 is load-dependent or a persistent
problem with that model's endpoint. Re-test before concluding.

### AI-16 · Route by capability and measured latency, not by model size — `FIXED`

**Value:** `bench/roster.py`. Capabilities (`code`, `reason`, `vision`,
`safety`, ...) each hold an ordered candidate list; `bench probe` measures what
this account can actually reach and how fast; the router takes the first
candidate that works.

**Reasoning — this was a real mistake, and it went unnoticed for two phases.**
The first bench hardcoded three roles onto the two largest models available,
and both queue for minutes. From that I concluded the free tier was too slow
for interactive delegation and quietly stopped delegating at all, writing P1
and P2 by hand. The call log records the whole story: five calls ever, three
successful, none after P0.

The conclusion was wrong. Probing the catalogue found seven models answering in
under a second, including `openai/gpt-oss-20b` at **0.3 s** -- a capable code
model that was available the entire time. The latency was a consequence of
asking a 1.6-trillion-parameter reasoning model to write a file parser, not a
property of the provider. **Routing by prestige is the error; a 20B model
returning in 0.3 s beats a 1.6T model that never returns.**

Two further facts the catalogue listing does not tell you, both of which is why
availability is measured rather than assumed:

* `GET /v1/models` lists models the account **cannot invoke**. Most return
  `"Not found for account"` -- every `mistralai/*` entry and every dedicated
  code model (Codestral, StarCoder2, granite-code, CodeLlama, deepseek-coder)
  among them.
* Among those it can invoke, latency spans three orders of magnitude, and the
  fast ones are not the ones whose names suggest capability.

Specialists that looked promising and are **not** provisioned, recorded so they
are not proposed again: `google/deplot` (charts to tables, 404),
`nvidia/nemotron-4-340b-reward` (404), all embedding endpoints (404), and
`nvidia/nemotron-parse`, which is not a chat-completions endpoint (400).

**How to apply:** re-run `bench probe` when calls start failing or crawling;
availability moves. Escalate to Kimi or DeepSeek only for work that genuinely
needs them, and expect to wait.

### AI-12 · Hosted model ids carry a publisher prefix — `FIXED`

**Value:** `deepseek-ai/deepseek-v4-pro-0813`, `moonshotai/kimi-k3`. Guarded by
a test that rejects any non-local model id without a `/`.

**Reasoning:** the build.nvidia.com catalogue displays the bare name, and the
bare name returns `HTTP 404: 404 page not found` — a body that says nothing
about the id being wrong, and reads like a broken endpoint. `bench models`
lists what the provider actually serves, which is the fastest way to settle it.

### AI-13 · A truncated generation is an error, not a result — `FIXED`

**Value:** `Client.text()` raises `Truncated` on `finish_reason == "length"`
unless `allow_truncated=True`. `finish_reason` is recorded in the call log.

**Reasoning:** both hosted models are reasoning models -- they spend completion
tokens thinking before emitting anything visible. Asking Kimi to "reply with
exactly: KIMI OK" at `max_tokens=32` returned `"KIM"`: a plausible-looking
answer that was actually a stump, with the budget consumed by reasoning. For
`implement` the same failure yields a source file that stops mid-function and
still parses as a string, which is far worse than an exception. Role defaults
were raised to 16k accordingly, and `bench ask` now defaults to 4096 rather
than 512.

### AI-02 · Integration is an MCP server, not a pipe between CLIs — `FIXED`

**Value:** `bench/mcp_server.py`, exposing `implement`, `investigate`, `review`
and `budget` over stdio.

**Reasoning:** the orchestrating model stays in Claude Code and delegates
natively, so the heavy token spend lands on NVIDIA's GPUs rather than the
orchestrating session's budget. A text-piping harness would need its own
conversation state, error handling and retry logic — all of which MCP already
has. Written against `mcp` 2.x, where `FastMCP` was renamed `MCPServer`.

### AI-04 · The bench never ships — `FIXED`

**Value:** `camber/` must not import `bench/`; `bench` is excluded from the
wheel; no HTTP client appears in runtime dependencies. Enforced by
`tests/test_isolation.py`.

**Reasoning:** two separate claims depend on this. *Open source* — both model
weights are open (DeepSeek V4-Pro is MIT) but NIM is proprietary hosted
infrastructure, and a 2.8T-parameter model is not something a user can
self-host as a fallback; a Camber that phones NVIDIA to pick a motor would be
open source wrapped around a rented brain. *Offline* — the tool has to work on
a plane, deterministically, without an account.

### AI-05 · Tests precede implementation, and the implementer does not write them — `FIXED`

**Reasoning:** a model that writes its own tests proves only that it is
self-consistent. For physics modules the tests are not unit tests in the usual
sense but *analytic limits and published data*: momentum theory in the ideal
limit, BEMT against UIUC wind-tunnel curves, a rocket burn against its own
certified total impulse.

### AI-06 · Models populate and explain; the optimiser decides — `FIXED`

**Reasoning:** ask a language model for "the optimal 6-inch prop" and you get a
fluent, plausible, unfalsifiable answer. The model's job is retrieval and
structured extraction into catalogue rows carrying a source URL and a
measurement method. The numerical optimiser then searches measured data.

### AI-07 · Zero models loaded is the default path — `FIXED`

**Reasoning:** if a local model sits on the critical path, the reliability
problem has been moved rather than removed. Selection, optimisation and physics
are all computed, not generated. A local model adds only convenience —
plain-language mission intake, fuzzy SKU matching, unit normalisation, prose
explanation. The test: remove every model and Camber still designs the aircraft.

### AI-08 · "Go online and find a motor" is answered by shipping the catalogue — `FIXED`

**Reasoning:** the going-online happens once, at build time, on the maintainer's
machine. The user receives a versioned SQLite catalogue and searches it offline
and deterministically. `camber catalogue update` fetches a newer signed
catalogue rather than turning a model loose on the web.

### AI-09 · Optional keys through one provider-agnostic adapter — `FIXED`

**Value:** one OpenAI-compatible client covers NIM, OpenRouter, DeepSeek direct
and LM Studio; base URL is the only difference. Anthropic is deliberately
absent — Opus 5 participates by *running* Claude Code, not by being called from
inside it.

### AI-10 · Ternary local model, with constrained decoding — `PROVISIONAL`

**Value:** BitNet b1.58-2B-4T (native ternary, ~0.4 GB non-embedding weights);
`bitnet.cpp` also runs a 1.58-bit Llama3-8B if more judgement is needed.

**Reasoning:** a 2B model is weak at reliable tool calling, which is exactly
what both of its jobs need. The fix is not a bigger model but **constrained
decoding** — a GBNF grammar or XGrammar makes malformed JSON unrepresentable,
because the sampler cannot emit a token the schema forbids. Schema conformance
then stops depending on model quality.

**Open:** during an overnight CFD campaign OpenFOAM saturates all 16 cores, and
a CPU-resident ternary model competes for the cores running the solve. Pin the
janitor to the GPU, or accept that it is slowest when it is busiest.

### AI-11 · `state_dir()` is a function, not a constant — `FIXED`

**Reasoning:** found by a failing test. A module-level `STATE_DIR` is captured
at first import, so every module that did `from .config import STATE_DIR` held
a path that could never afterwards be redirected — including by a test fixture,
which is how three tests ended up sharing one cache file. Resolve the
environment on every call.

---

## 1. Scope and environment

### SCOPE-01 · Electric multirotor is the v1 validation class — `FIXED`

**Reasoning:** best open data by a wide margin (UIUC and Tyto both centre on
it), so validation gates are cheap to build and genuinely meaningful. The
thruster interface stays polymorphic from the first commit regardless — rockets
bring time-varying mass, EDF brings duct coupling, and retrofitting either into
a hard-coded rotor model means rewriting the flight simulation.

### MFG-01 · 3-axis CNC first, process user-selectable — `FIXED`

**Reasoning:** manufacturability is a pluggable `Process` interface. 3-axis
milling **cannot cut undercuts** — every surface must be reachable straight down
from one of the setups — which is a far harder geometric constraint than FDM's
overhang angle. Satisfy it first and the additive processes are relaxations;
do it in the other order and half the CNC designs are unmakeable.

### LIC-02 · The repository stays private — `FIXED`

**Value:** local git only. **No remote, no push, no GitHub repository**, and no
publishing of any kind until Abhijit says otherwise. Stated 2026-09-07 as
"make the repo private for now. Keep it that way."

**Reasoning:** the licence chosen in LIC-01 governs what recipients may do once
the code is shared; it says nothing about *when* to share. Those are separate
decisions and only the author makes the second one. GPL-3.0 remains the
intended licence for an eventual release.

There is also a concrete reason to be careful before any first push. LIC-01
leaves catalogue redistribution **open**: `data/raw/` is git-ignored so the
UIUC archive is not in the tree today, but a future catalogue build could put
derived measurements there, and those terms must be settled before anything
leaves this machine.

**How to apply:** do not run `git remote add`, `gh repo create`, or `git push`.
If a remote is ever wanted, it is created private, by Abhijit, deliberately.

### LIC-01 · GPL-3.0 for code; catalogue licensing is separate — `FIXED` / `OPEN`

**Value:** GPL-3.0-or-later for Camber. Catalogue redistribution: **OPEN**.

**Reasoning:** GPL matches the ecosystem Camber orchestrates (OpenFOAM is
GPL-3.0; CalculiX and XFOIL are GPL-2.0) and removes any argument about whether
tight OpenFOAM coupling creates a derivative work.

The code licence says nothing about whether third-party *measurements* may be
redistributed. UIUC, APC and Tyto Robotics each set their own terms, and AI-08
has us shipping a compiled database of them. Verify per source before any
catalogue ships; where redistribution is not permitted, ship the build script
and fetch on first run. ThrustCurve is the encouraging precedent — OpenRocket
already redistributes a compiled database built from it.

### ENV-01 · Other projects are off-limits, including the shared OpenFOAM install — `FIXED`

**Value:** no writes to another project's tree; no refactor that extracts
shared code out of it; `~/.local/opt/OpenFOAM-v2606` is read-only and pinned.

**Reasoning:** an earlier CFD study shares this machine, and its results must
stay reproducible. Copying code duplicates
maintenance, which is the correct trade here.

The non-obvious hazard is the **shared OpenFOAM install** — 2.4 GB, hand-built,
and used by both. Upgrading it, patching a solver, or rebuilding it after a
system library moves would silently change the tool those results were produced
with, with no error to warn anyone. If Camber ever needs a different version it
gets its own prefix, and the two never meet.

---

## 2. Architecture

### ARCH-01 · Three-tier fidelity ladder; CFD never in the inner loop — `FIXED`

**Value:** L0 analytic (µs, 10⁶ evals) → L1 BEMT/VLM (ms, 10⁴ evals) → L2
OpenFOAM (hours, 10¹ evals). Plus L3, transient 6-DOF, alongside rather than
above.

**Reasoning:** a RANS rotor case takes minutes to hours; an optimiser needs
10⁴–10⁶ evaluations. CFD's product is not a verdict on candidates but a
correction to L1, fitted as a Gaussian process over the residual and folded back
in — multi-fidelity trust-region model management. L3 is not optional: three of
the four mission objectives are properties of a trajectory, not a hover point.

### ARCH-04 · The ladder is closed: the optimiser searches corrected L1 — `FIXED`

**Value:** `DesignProblem(interference=...)`. The calibrated model lives in
`data/interference.json` and is loaded, not refitted.

**Reasoning:** ARCH-01 has said since the plan that CFD corrects the cheap model
and the optimiser searches the correction. Until now that was only half true --
the model was measured, validated and never connected, so the optimiser had
been searching an uncorrected L1 the whole time. A validated correction nobody
consumes is a result, not a feature.

The correction enters through the physics rather than as a fudge on the answer.
Interference means the rotors must produce `weight / (1 - loss)` of *isolated*
thrust to hold the aircraft up, because the airframe gives some of it back as
download. That extra thrust costs current, which costs endurance. Above 15%
loss the layout also takes a constraint penalty, since at that point the
geometry is bad for reasons beyond efficiency.

**What it changes, measured.** On the unconstrained design space, almost
nothing -- both fronts converge on ~170 mm arms, where the discs clear the
airframe entirely and interference is 0.05%. That is the right answer and worth
stating plainly: given freedom, the optimiser avoids the interference regime
without being told to.

It bites when the envelope is constrained. Forced to arms of 85-125 mm:

```
uncorrected : arm 115.5 mm,  7.75 min,  interference not modelled
corrected   : arm 124.7 mm,  8.02 min,  interference 0.76%
```

The corrected search pushes the arms to the edge of the allowed envelope to
escape download. The uncorrected one has no reason to and stops short.

### ARCH-02 · Constraints are declarative data in the vehicle spec — `PROVISIONAL`

**Value:** `fixed`, `relative(a, b, offset)`, `min_distance(a, b, d)`,
`view_cone(camera, half_angle, unobstructed_by=[...])`, `cg_box`,
`symmetry(plane)`.

**Reasoning:** the optimiser reads them, the GUI draws them, and the validator
rejects an over- or under-constrained spec before anything expensive runs. As
code they would be none of those things.

### ARCH-03 · Controllability is a constraint, not a post-check — `FIXED`

**Reasoning:** a design that is aerodynamically excellent and dynamically
uncontrollable is a failed design, and an unconstrained optimiser will find
those enthusiastically. Every candidate gets its allocation matrix built from
thruster positions and orientations, then checked for rank and control-authority
margin about all three axes. It is also what makes the output plug-and-play: the
mixer is part of the deliverable.

---

## 3. The vehicle model (P1)

### GEOM-01 · SI everywhere; convert once, at the edge — `FIXED`

**Value:** metres, kilograms, seconds, radians, newtons, volts, amperes
throughout. Inches, grams, mAh and RPM appear only in catalogue parsing and
spec files (`horizontal_fov_deg`, `rotation_deg`), converted at the boundary.

**Reasoning:** mixed units survive unit tests. Every value passes an assertion
about its type and none about its dimension, so a diameter in millimetres flows
through the whole stack and emerges as a plausible, wrong answer. That is the
one failure mode this project cannot tolerate, because nothing downstream
would flag it.

### GEOM-02 · NED body frame: x forward, y right, **z down** — `FIXED`

**Reasoning:** it is what ArduPilot, PX4 and the flight-dynamics literature
assume, and Camber must eventually emit a mixer for one of them. The
consequence to remember is that lift is **negative** z, so `UP = (0, 0, -1)`.

### GEOM-03 · Mass properties are analytic, not tessellated — `FIXED`

**Value:** `camber/spec/shapes.py` returns closed-form inertia tensors. The CAD
kernel is used only for STEP output, booleans and imported geometry.

**Reasoning:** a box's inertia has been a closed form since Euler; obtaining it
by tessellating a B-rep and summing tetrahedra is slower, less accurate, and
makes the product depend on a 200 MB wheel for arithmetic. Keeping the boundary
sharp is what lets the optimiser evaluate mass properties millions of times
(AI-07), and build123d stays an optional extra.

Bodies are modelled as solids rather than point masses because a 4S LiPo
treated as a point understates roll inertia badly, and roll inertia is what
sets control authority. Verified against closed forms, the thin-rod limit, and
the triangle inequality `I1 + I2 >= I3`, which no real mass distribution can
violate.

### GEOM-04 · Millimetres exist only in the CAD exporter — `FIXED`

**Reasoning:** STEP and every CAM package expect millimetres. The conversion
lives in `camber/geometry/cad.py` and nowhere else; a mismatch between the
geometry's units and the STEP header produces a part correct in shape and wrong
by 1000x in size, which a machinist discovers expensively.

### CTRL-01 · An axis is controllable only if it lies *in* the range of A — `FIXED`

**Value:** `controllability()` tests whether the unit wrench `e_i` lies in
`range(A)`, not whether it has non-zero projection onto it.

**Reasoning:** found by a failing test. Canting a quadrotor's rotors gives every
axis a non-zero projection while the range stays four-dimensional, so a
projection test reports such a design as fully controllable when lateral force
is still unavailable without disturbing something else. `rank(A) <= min(6, N)`,
so four thrusters can never be fully actuated whatever their angles — six
canted rotors can, which is why fully-actuated multirotors are built with six.

The SVD's smallest singular value is reported as `margin` because rank alone is
a discontinuous yes/no that an optimiser cannot follow downhill. The margin
shrinks continuously as arms shorten, so a design can be pushed away from the
cliff before it falls off.

### CTRL-02 · Hover trim respects the sign of thrust — `FIXED`

**Reasoning:** the least-squares solution to `A u = [0,0,-W,0,0,0]` will
happily return negative commands, and a propeller cannot pull. Checking the
residual and the bounds separately is what distinguishes "this hovers" from
"an unconstrained solver found numbers". Confirmed against hand arithmetic: a
symmetric quad splits weight exactly four ways, and four rotors spinning the
same way cannot hold a heading because nothing cancels their reaction torque.

### SPEC-01 · An unknown key in a spec is an error — `FIXED`

**Reasoning:** `mas = 0.25` must stop the load. Silently substituting a default
mass produces a confidently wrong endurance figure with nothing to flag it.
Unknown keys are reported *before* missing ones, so the message names the typo
rather than the symptom it caused.

---

## 4. The physics tier (P2)

### PHYS-01 · BEMT solves a bracketed residual in phi, not a fixed point — `FIXED`

**Reasoning:** the obvious scheme -- guess the induction factors, recompute,
repeat -- diverges for heavily loaded stations, which is exactly where a
propeller operates at low advance ratio. Reformulating as a sign-changing
residual in the inflow angle and bracketing it either converges or reports
honestly that it could not. Convergence is recorded per station and a
non-converged point is discarded from validation rather than averaged in.

### PHYS-02 · The blade section is the dominant error, and it is unknown — `PROVISIONAL`

**Value:** default section `e63`.

**Reasoning — measured, by sensitivity study over the whole database.** With
everything else fixed, changing only the assumed aerofoil moves predicted CT
across the 115-propeller set from a median of **-15.9%** (Clark-Y) to **+7.1%**
(E63). Tip and hub loss, by comparison, account for three to four per cent.

UIUC publishes chord and twist distributions but not the aerofoil, and real
blades vary section along the span -- high-camber inboard, thin outboard. A
single section for the whole blade is therefore a named approximation, and it
is the largest single source of error in the tier. E63 is chosen because it is
a high-lift low-Reynolds section of the kind thin electric propellers actually
use, and because it minimises bias across the set. It is not a fitted
parameter: it was selected from four candidates on a metric fixed in advance.

**Open:** a radially blended section is physically better motivated and would
probably reduce the scatter. It also adds a knob, so it should be introduced
only with a held-out set to check it is not being tuned.

### PHYS-03 · Validation error is scaled by peak CT, never by the local value — `FIXED`

**Reasoning:** CT falls to zero as a propeller approaches its zero-thrust
advance ratio, so a relative error taken against the local value diverges there
for arithmetic reasons that say nothing about the model. Measured directly:
points below CT = 0.01 showed a mean relative error of **799%** while points
above CT = 0.08 sat at **10.3%**, from the same predictions. Quoting the first
number would be misleading in both directions -- it makes a decent model look
worthless, and it would let a genuinely bad model hide behind "the tail
dominates".

Scaling by each run's peak CT asks the question that matters: how many newtons
out, against how many the propeller makes. Both metrics are reported; the
scaled one is the one to quote.

### PHYS-04 · Small propellers are not Reynolds-independent — `FIXED`

**Value:** the model reproduces the direction and order of magnitude of the
effect while over-stating it roughly twofold. Recorded, not corrected.

**Reasoning:** CT and CP remove rpm by construction, so a textbook reading
expects them invariant at fixed J. At chord Reynolds numbers near 10^5 they are
not. The database settles it -- doubling rpm at fixed J raises measured CT by
about **+25%** (apce_10x5) to **+42%** (apce_11x7). The model gives roughly
+70% over the same change.

This matters for design: a propeller characterised on a bench at one rpm cannot
be assumed to hold its coefficients elsewhere, and an optimiser comparing
designs at different disc loadings is comparing different Reynolds regimes.

### PHYS-05 · Propellers are tabulated before they are coupled — `FIXED`

**Value:** `PropellerMap` samples CT and CP against J once (about 0.5 s), after
which a lookup costs **2.8 microseconds** against **5000 microseconds** for a
direct BEMT solve -- a factor of 1800.

**Reasoning:** the coupled motor-propeller match is a root-find that evaluates
propeller torque twenty or thirty times. Calling BEMT inside it makes a single
operating point cost 100 ms, and an optimiser needs thousands. This is the
fidelity-ladder move (ARCH-01) applied one level down: pay for the physics
once, then search a cheap surrogate. CT and CP are smooth in J and the sampling
is finer than the data the model is validated against, so it is a speed
optimisation rather than an additional approximation.

### PHYS-06 · Viterna's post-stall model needs two guards — `FIXED`

**Reasoning:** a blade element near the hub at low advance ratio genuinely
visits the whole circle of angles, and the published Viterna formula does not
survive there. Its lift correction carries `cos^2(a)/sin(a)`, singular at 0 and
180 degrees -- unguarded it returned CL of 2.4x10^5. Its drag correction goes
*negative* beyond 90 degrees, which would let a BEMT solve extract energy from
nothing. The first is damped by a window vanishing at 90 degrees and a floored
denominator; the second is clipped at zero with a reversed-flow floor. Tested
across the full circle.

### PHYS-07 · Kt and Kv are one constant, not two inputs — `FIXED`

**Reasoning:** Kv is quoted in RPM per volt and Kt in newton-metres per amp,
which look independent until both are in SI, where Kt = 1/Kv_rad exactly.
Accepting them as separate parameters is a standard way to build a motor model
that violates conservation of energy while appearing well-calibrated.

### PHYS-08 · A pack can fail by sag as well as by capacity — `FIXED`

**Reasoning:** endurance is coulomb-counted against a voltage cutoff rather
than obtained by dividing capacity by current, because the two answers differ
whenever internal resistance binds first. A 1 Ah pack at 80 A browns out with
charge remaining; `Ah/A` reports a flight it will not complete. Sizing on
energy alone finds designs that fail on the first aggressive manoeuvre.

---

## 5. The optimiser (P3)

### OPT-01 · Hard gates for flyability, penalties for geometry — `FIXED`

**Value:** controllability rank, control margin and hover feasibility reject a
candidate outright. Clearance, view cones and the CG box become a summed
violation instead.

**Reasoning:** the two kinds of failure are not alike. A design whose rotors
cannot produce a roll moment is not a worse design, it is not a design, and its
objective values are meaningless -- letting it compete on endurance drags the
population into a region where the numbers mean nothing. A geometric breach is
different: it is continuous, so a penalty gives the search a gradient to walk
back out along. Getting the split backwards is expensive in both directions --
penalise controllability and the budget is spent among unflyable candidates;
harden the view cone and the search cannot find its way back in.

### OPT-02 · Constrained domination, not a weighted sum — `FIXED`

**Reasoning:** feasible beats infeasible always; between infeasible candidates
the smaller violation wins; between feasible ones the ordinary Pareto rule
applies. A weighted sum would hide both the trade-off and the weighting -- an
arbitrary choice by whoever wrote the line -- inside the answer. The front hands
the trade-off back to the person who should be making it.

### OPT-03 · The momentum-theory endurance model is a placeholder, and a bad one — `FIXED`

**Value:** `physics_endurance_model()` wires the validated blade-element tier
in. The analytic fallback remains only for runs with no propeller selected, and
the CLI warns when it is in use.

**Reasoning:** the placeholder assumes an ideal rotor and ignores motor and ESC
losses. Its docstring estimated it would overstate endurance "by roughly a
factor of two". Measured against the validated tier on the same airframe it
reported **69.4 minutes against 4.2** -- out by 16.5x, and the first
optimisation run duly converged on a fantasy. A surrogate whose error is
unmeasured is not a surrogate, it is a guess with a docstring.

### OPT-04 · Mass must be a decision, or the second objective is decorative — `FIXED`

**Reasoning:** the first run produced a "front" of forty identical points.
Nothing in the variable set changed mass, so there was no trade-off to find and
every candidate tied. Pack capacity is now a variable with mass following from
energy at 180 Wh/kg, which is what makes endurance-versus-mass a real axis.

A multi-objective front that collapses to a point is not a converged search; it
is a sign that one objective is not connected to any variable.

### OPT-05 · One propeller map is scaled across radii — `PROVISIONAL`

**Reasoning:** CT and CP are functions of blade *shape* and Reynolds number,
not absolute size, so a single map serves every candidate radius with thrust
scaling as rho n^2 D^4. Rebuilding a map per candidate would cost about 0.5 s
against roughly 0.4 ms for the whole evaluation.

The approximation is the Reynolds part, and PHYS-04 records that it is not
negligible at this scale, so comparing radii far apart carries a known error.
Worth revisiting with a small family of maps if radius ranges widen.

### OPT-06 · Delegated code is reviewed, not merely tested — `FIXED`

**Reasoning:** the Pareto module was generated by the bench and passed all
fifteen tests written for it beforehand. Reading it found that
`crowding_distance` computed a *backward* difference, `f[i] - f[i-1]`, where
NSGA-II specifies the central difference `f[i+1] - f[i-1]` -- understating
crowding and biasing selection toward one end of the front.

The tests were the problem: checking that boundaries are infinite and interiors
positive cannot distinguish the two formulas. `TestCrowdingDistanceExactValues`
now pins the arithmetic. The lesson generalises past this bug -- a green suite
licenses delegated code only as far as the suite actually constrains it, and
tests written to describe a shape will accept any implementation of that shape.

---

## 6. The CFD tier and the closed loop (P4)

### CFD-01 · CFD calibrates the cheap model; it never evaluates candidates — `FIXED`

**Value:** prescribed-thrust actuator discs plus a resolved airframe, steady
RANS. The measured quantity is the *download* on the body, not rotor
performance.

**Reasoning:** isolated rotor accuracy was settled in P2 against 115
wind-tunnel propellers, and no simulation improves on measuring the thing
itself. CFD earns its cost only where BEMT is structurally blind -- airframe
download, rotor-rotor interaction, blockage -- all of which vary with *layout*,
which is exactly what the optimiser changes.

Blade-resolved rotors were rejected for a second reason beyond cost: feeding a
BEMT-derived loading into a rotor model and then using the result to correct
BEMT is circular. Prescribing the thrust L1 predicts and measuring what the
airframe feels isolates the one term L1 cannot produce.
`vectorSemiImplicitSource` is used rather than `rotorDiskSource` for precisely
that reason -- the latter computes its own blade-element loading.

### CFD-02 · The CAD exporter needs an explicit unit — `FIXED`

**Value:** `cad.export(..., metres=True)` for CFD; millimetres remains the
default for STEP and printing.

**Reasoning:** found by reading a log that reported success. GEOM-04 confines
millimetres to the CAD exporter, and the first CFD case then fed that
millimetre STL straight into a metre-based domain -- making a 180 mm airframe
180 m across and swallowing the entire domain. snappyHexMesh reported
`Selected for refinement : 0 cells`, exited zero, and produced a mesh with no
`body` patch at all; the forces object would have integrated over nothing.

This is exactly the failure GEOM-04 was written to warn about, walked into by
the person who wrote the warning. The lesson is that a unit convention stated
in a docstring is not enforcement: the boundary now takes an argument, and a
test compares the two files' coordinate magnitudes.

### CFD-03 · Coarse background, refined near field — `FIXED`

**Reasoning:** the actuator discs need cells small enough to resolve a momentum
source; applying that size to the whole domain costs about four million cells
before snappy refines anything, which does not fit in 22 GB. The background is
therefore coarser by 2^levels with a near-field box refined back down.

The first attempt got this backwards by dividing the *base* cell by four -- the
same arithmetic applied to the wrong quantity, coarsening the discs themselves
to **three cells each**. A momentum source spread over three cells is a point
force with a misleading name. `cells_per_disc()` now guards it and a test
requires more than fifty.

### CFD-04 · Trust the log, not the exit code — `FIXED`

**Reasoning:** OpenFOAM will exit zero having printed a fatal error, and worse,
exit zero having done nothing useful -- both of which happened here within one
session. `foam()` scans for `FOAM FATAL` as well as checking the status, and
the mesh step is verified by asserting the `body` patch exists and the cell
zones are populated, rather than by the absence of a crash.

### CFD-05 · Momentum sources are kinematic, not newtons — `FIXED`

**Value:** `vectorSemiImplicitSource` values are `thrust / rho`.

**Reasoning:** incompressible OpenFOAM divides the momentum equation by
density, so a source in the U equation carries units of m^4/s^2, not newtons.
Passing newtons over-thrusts every disc by a factor of rho -- 22.5% here.

Nothing in the run reported it. Residuals converged, the solver exited zero,
the log looked clean. It surfaced only from an independent check: momentum
theory fixes the far-wake velocity at `2 sqrt(T / 2 rho A)` = 12.06 m/s for
this disc loading, and the field showed **13.56**, which is what 5.53 N
produces rather than 4.51 N. After the correction the field gives **12.25 m/s**,
within 1.6% of theory.

The general lesson is the one PHYS-03 and the probe metric taught in different
costumes: a simulation that runs cleanly is not a simulation that is right, and
the only defence is an independent quantity the code cannot fudge. Every future
CFD case should be checked against its own momentum-theory wake velocity before
its forces are believed.

### CFD-06 · The momentum source is the reaction, so the wake goes down — `FIXED`

**Value:** the `vectorSemiImplicitSource` vector is `-direction * thrust / rho`.

**Reasoning:** a rotor lifting a vehicle pushes air *downward*; the source acts
on the fluid and is the reaction to the thrust, not the thrust. Applying
`+direction` blew the wake upward through the airframe and lifted it, which
reported download of **-6% to -17%** -- a body gaining thrust by sitting under
its own rotors, which is impossible.

Two things make this worth recording. First, every diagnostic looked healthy:
residuals converged, the solver exited zero, the disc zones were populated, the
sources were listed in the log. Second, and worse, the momentum-theory check
introduced in CFD-05 **passed** -- because it compared only the *magnitude* of
the far-wake velocity, and a wake blowing the wrong way is exactly as fast as
one blowing the right way.

A validation that checks magnitude but not direction is half a validation. The
check now asserts the fastest cell moves in +z, and a test asserts the source
vector's sign directly.

### CFD-07 · Disc thickness is capped by clearance to the airframe — `FIXED`

**Value:** thickness `min(R/6, 1.6 * clearance)`, and `DiscGeometryError` if a
disc still intersects a body.

**Reasoning:** thickness was `3 * cell`, and the cell scales with radius, so a
large rotor produced a disc **0.75 R thick** -- taller than the vehicle. Every
layout in the first campaign had all four discs intersecting all four bodies,
so the "force on the body" was reading the momentum source injected into the
cells around it. The results were smooth, converged and entirely meaningless.

The example spec was also wrong and the check found it: rotors sat at
z = -14 mm with the camera top at -13.5 mm, half a millimetre of clearance. A
2207 motor stands about 32 mm, so the disc belongs at -45 mm. A geometric
assertion caught a modelling error in the input, which is the argument for
making it fatal rather than a warning.

### CAL-02 · The loop closes; the correction is usable — `RESOLVED`

**Value:** 25 cases, all usable, zero failures. Leave-one-out mean error
**1.34%**, R2 **0.952**, pairwise ordering **95.5%**, no false clears at a 2%
threshold. `as_penalty()` remains the conservative interface; `correction()`
and `installed_thrust()` are now defensible as quantitative outputs.

**The history matters more than the numbers.** This decision was resolved once
before, on data that was wrong, and the earlier conclusion -- "accurate where it
is boring, weak where it matters" -- was an artifact of CFD-08. Two-thirds of a
cell of disc thickness produced 2-cell momentum sources, point-force jets, and
downloads reported as 66-70%. Re-run with the meshing fixed:

```
regime                   n   LOO mean    worst      contaminated
flat  (<1%)              2     0.54%     0.84%      0.41% /  2.67%
transition (1-15%)      15     0.90%     2.31%      1.28% /  2.27%
steep (>15%)             8     2.35%     3.85%      9.25% / 48.77%
aggregate                     1.34%                 3.06%
R2                            0.952                 0.527
```

The steep band improved fourfold in mean and thirteenfold in worst case. The
regime I had described as intrinsically hard to model was not hard; the
measurements were broken. The true maximum download is **26%**, and the 66-70%
figures never existed.

Three other things the fix bought: **zero solver failures** where the previous
campaign lost 4 of 20 to floating-point exceptions, **zero cases dropped** by
the mesh-resolution guard where 3 more had gone, and meshes of 137k to 379k
cells instead of 37k to 90k.

**What this cost.** Two campaigns and roughly three hours of compute produced
data that had to be discarded. The tell was visible early -- a cell count of
exactly 37,034 recurring across three different geometries -- and I noted it
twice before investigating. An identical derived quantity across different
inputs is not a curiosity; it is proof that something ignored the input.

**Remaining honest limits.** 25 cases, one airframe family, hover only. The
model has never seen forward flight, a different rotor count, or a body shape
that is not a stack of boxes. It should be re-fitted, not extrapolated, for any
of those.

### CAL-01 · A Gaussian process over four physical features — `FIXED`

**Value:** `camber/calibrate.py`. Features: blockage (body planform / disc
area), rotor spacing in diameters, body-to-disc offset in radii, and disc
loading. The correction is stored as a *fraction* of isolated thrust.

**Reasoning:** with ten or twenty CFD runs the data is tiny, and a GP is
well-behaved there where a flexible model overfits and then extrapolates
confidently. More importantly a GP reports its own uncertainty, which turns
"where should the next expensive run go?" into arithmetic rather than taste --
`next_case()` samples where the correction is least known. Spending hours
re-confirming what the model already knows is the most expensive way to learn
nothing.

A fraction rather than an absolute force because interference scales with
thrust: a design twice as heavy loses roughly twice as much to download, and a
model fitted in newtons would relearn that at every scale. The correction is
clamped outside the training range so an extrapolating fit cannot invent thrust.

An unfitted model returns exactly `(0.0, 0.0)` and is a transparent no-op --
never a silent guess.

---

## 7. Generated structure (P5)

### STR-01 · Hover is almost never the governing load — `FIXED`

**Value:** three cases, the largest governs: hover, manoeuvre (one arm at full
thrust, as a hard roll commands), and a landing at 6 g shared across the arms.

**Reasoning:** a frame sized for the thrust needed to hang still is a frame
that breaks. An optimiser makes this mistake far more enthusiastically than a
person would, because minimising mass drives every section to its limit, so
whichever load case is omitted is exactly the one that fails.

Measured, and the result is more interesting than expected. On the 460 g
example the distinction does **not** bite: the smallest purchasable tube is
already 3.5x over-strength, which is why hobbyists get away with ignoring it.
From roughly 1.8 kg upward it bites hard -- a hover-sized 8 mm arm on a 10-inch
quad fails under **landing**, not manoeuvre, and needs 14 mm. At 18 kg no stock
section in the ladder passes at all, which the sizer reports rather than
silently returning the strongest thing it has.

### STR-02 · Sections come from a stock ladder, not a closed-form optimum — `FIXED`

**Reasoning:** tube is bought in sizes. A continuous optimum of 11.3 mm outer
diameter is not purchasable, and rounding it afterwards either wastes mass or
quietly invalidates the margin it was chosen for. The search is over the
ladder, so the answer is orderable.

Deflection is checked alongside stress, and not for comfort: a visibly flexing
arm tilts its rotor off vertical, which both steals thrust and feeds the flight
controller a moment it did not command. Slender arms fail this limit long
before they approach yield.

### MFG-02 · Manufacturability is checked inside the loop — `FIXED`

**Value:** `camber/manufacture.py`, a `Process` interface with `CNC3Axis` and
`FDM`. Violations carry a continuous magnitude, not a boolean.

**Reasoning:** an unconstrained search converges on a beautiful shape nobody
can produce, and finding that out afterwards means discarding the answer rather
than steering away from it. Continuous violations let the optimiser walk back
out of a mild breach.

MFG-01's ordering is confirmed by the checks themselves: on the same part CNC
raises five violations to FDM's two. The one worth naming is that a round
cutter cannot produce an internal corner sharper than its own radius -- a
geometric impossibility rather than a tolerance, and the constraint most often
missed by people designing for milling for the first time.

### GEN-01 · Straight tubular arms, deliberately — `PROVISIONAL`

**Value:** the generated frame is a centre plate sized to the payload envelope
plus one straight tube per rotor, each sized by STR-01.

**Reasoning:** topology optimisation would produce something lighter and
something no closed-form beam model can check. That would mean trusting a
generated airframe on a generated analysis with nothing measured anywhere in
the chain. A shape the structural model sizes *exactly* is worth more than a
lighter one it can only approximate.

**Known optimism.** The generated frame comes out at 30.8 g against the 110 g
chassis in the example, and that gap is not all savings: it is a plate and four
tubes with no standoffs, fasteners, camera mount or crash margin. Treat it as a
structural lower bound, not a bill of materials.

---

## 8. GPU acceleration

### GPU-01 · The bottleneck is memory bandwidth, and it is measured — `FIXED`

**Hardware:** RTX 5060 Laptop, 8 GiB GDDR7, compute capability 12.0
(Blackwell), driver 610.57. Host is an Ultra 7 255H with LPDDR5-5600.

**Measured on this machine**, not quoted from a spec sheet:

```
stream triad          CPU (1 thread)   18.0 GB/s
                      GPU             202.5 GB/s        11x
SpMV, 636k rows        CPU              6.84 ms
                      GPU              0.24 ms          28x
pressure CG, 300 it    CPU            3562.5 ms
                      GPU              93.4 ms          38x
```

Abhijit's diagnosis was right: incompressible RANS on unstructured meshes is
bandwidth bound, not compute bound, which is why earlier CFD runs were capped
by RAM speed and why a 5060's GDDR7 is the relevant resource rather than its
shader count.

**Two honest qualifications.** The 38x is CG against CG -- the same algorithm on
both devices -- so it measures the hardware gap, not the gain in practice.
OpenFOAM does not use plain CG on the CPU; it uses GAMG, which is far better per
iteration. The fair comparison is GPU algebraic multigrid (AmgX) against CPU
GAMG, and the published range there is **4-8x on the pressure solve and about
2x end to end**. Second, this is a *laptop* 5060: 128-bit GDDR7 at roughly
448 GB/s peak, against about 90 GB/s for dual-channel LPDDR5. A desktop card
would widen the gap considerably.

**VRAM is the real ceiling.** A 636k-cell pressure system occupied 2.55 GiB, so
8 GiB caps this route at roughly 4-6M cells. Adequate for actuator-disc cases;
not adequate for blade-resolved AMI, which was already the expensive tier.

### GPU-02 · The integration needs a system CUDA toolkit — `BLOCKED`

**Value:** the intended route is PETSc built with CUDA plus `petsc4Foam`,
installed into `FOAM_USER_LIBBIN`. That path is specifically chosen because the
plugin lives in the *user* tree and never touches
`~/.local/opt/OpenFOAM-v2606`, so ENV-01 holds: the install the other study depends on
stays untouched. `fvSolution` then selects `solver petsc;` for `p` and nothing
else about the case changes.

**Blocked on:** `pip install nvidia-cuda-nvcc-cu12` ships only `ptxas`, not the
`nvcc` driver, so native CUDA code cannot be compiled in userland. CuPy works
regardless -- it compiles kernels through NVRTC at runtime, which is how the
benchmarks above ran -- but CuPy cannot accelerate OpenFOAM.

Building PETSc with CUDA, or AmgX, therefore needs `sudo pacman -S cuda`. That
is the one step requiring elevated privileges; everything after it is a
userland build.

**Explicitly rejected: FluidX3D.** It is the fastest GPU CFD code available and
would suit an actuator disc in a box, but it is licensed *free for
non-commercial use*, which is not open source and is incompatible with the
GPL-3.0 chosen in LIC-01. If a GPU-native rewrite is ever wanted, OpenLB (GPL,
CUDA) is the licence-compatible option.

---

## 9. Screening before solving

### SCR-01 · A geometric screen decides which layouts get a CFD run — `FIXED`

**Value:** `camber/cfd/screen.py`. Project each rotor disc straight down, measure
the fraction of body planform beneath it, and skip the run below 5% overlap.
Roughly 7 ms per layout, no flow solved.

**Reasoning:** Abhijit's suggestion, and it is better than the GPU work it
replaced. His framing was to use cheap compute where the quality difference does
not matter. The question those cheap runs answer -- *is any part of the airframe
underneath a disc at all* -- turns out not to need a flow solver, because it is
geometry.

**Validated against the 30 labelled CFD cases** from the 40-case campaign:

```
correlation, overlap x disc loading vs measured download :  +0.991
threshold overlap < 5%  ->  22 of 30 cases skipped
                            0 interference cases missed
                            73% of CFD budget freed
```

This directly addresses CAL-02. That campaign spent 22 of 30 runs, about
45 minutes, establishing that widely spaced rotors do not interfere -- runs that
landed in the flat region, taught the model nothing, and inflated its aggregate
accuracy. Sampling now draws only from the region the screen says can
interfere, so the whole budget lands where download actually varies.

**The threshold is deliberately asymmetric.** 5% sits well below the smallest
overlap of any case that showed measurable download, because the two errors do
not cost the same: a needless CFD run wastes a minute, while a missed
interference case silently poisons the calibration the optimiser then trusts.

**What the screen is not.** It says where interference *cannot* be, which is a
weak and reliable claim. It does not say how much there is, and
`estimated_loss()` raises rather than returning a number for any layout it did
not rule out.

### GPU-04 · The GPU path works, is validated, and is slower — `FIXED`

**Built:** CUDA 13.3, PETSc 3.25.5 configured `--with-cuda --with-cuda-arch=120`
(Blackwell), and `petsc4Foam` compiled against v2606 into
`FOAM_USER_LIBBIN`. ENV-01 verified afterwards: nothing under
`~/.local/opt/OpenFOAM-v2606` was modified.

**Correct.** CPU GAMG and GPU PETSc give an identical answer -- max |U| of
15.335 m/s from both, agreeing to 0.00%. The integration is sound.

**And slower.** On a 126k-cell case, 200 outer iterations, 4 ranks:

```
OpenFOAM GAMG (CPU)    30.5 s     <- best
PETSc jacobi  (GPU)    59.2 s     1.9x slower
PETSc gamg    (GPU)    83.2 s     2.7x slower
PETSc bjacobi (GPU)   200.4 s     did not finish
```

Single rank, whole mesh on one device: GPU 93.5 s against CPU 97.1 s -- a tie.
GPU utilisation sat at **4%**, so the device was idle rather than saturated.

**Why, given the pressure solve is 87% of runtime.** That fraction was measured
directly, by slackening the pressure tolerance until the solve was nearly free
and seeing 29.2 s fall to 3.7 s. Amdahl's law therefore permits a large win --
up to 6.6x if the linear solve were 38x faster, as the synthetic CuPy benchmark
in GPU-01 suggested it could be.

It is not realised because the problem is too small. At 126k cells over 4 ranks
each device solve is 31k unknowns, where kernel launch and host-device transfer
dominate, and the pressure equation converges in 1 to 7 iterations from a good
SIMPLE initial guess -- there is little iteration to accelerate. OpenFOAM's
GAMG is also extremely well tuned, so the honest comparison was never CG
against CG.

**Kept anyway, off by default.** The stack is built, validated and costs
nothing when unused: `libs (petscFoam);` plus `solver petsc;` in a case turns
it on. It should be revisited for blade-resolved AMI, which is the tier that
would produce meshes in the millions of cells where published 4-8x results
were obtained -- though 8 GiB of VRAM caps that at roughly 4-6M cells.

**What this cost and what it bought.** About an hour of build time for a
negative result, and the negative result is worth having: it replaces "GPU
should help, published results say 4-8x" with a measurement on this machine, at
this problem size, with the answer validated against the CPU baseline. The
synthetic 38x in GPU-01 was real and irrelevant -- it measured a regime this
workload never enters.

### GPU-03 · GPU acceleration is deferred, not rejected — `SUPERSEDED by GPU-04`

**Reasoning:** the measurements in GPU-01 stand -- the solve is bandwidth bound
and VRAM is 11 to 38x faster depending on what is measured. But the realistic
end-to-end gain from AmgX in OpenFOAM is about **2x**, it is blocked on a system
CUDA install (GPU-02), and the screen in SCR-01 delivers roughly **4x** on
campaign throughput for free, today, with no new dependency.

Cheaper and larger, so it goes first. GPU work becomes worthwhile again when
the screen has already removed the wasted runs and the remaining cases are the
bottleneck -- or when meshes grow past what 16 CPU cores can chew, which
blade-resolved AMI certainly would.

FP32 is worth noting for that eventual work: on this card it halves wall clock
for a given cell count (3.4 ms against 7.0 ms on the triad) and doubles the
capacity ceiling, and screening-quality accuracy does not need FP64.

---

## 9b. Machine manners

### RUN-02 · One wide case beats several narrow ones — `FIXED`

**Value:** run cases sequentially at the full rank count. Do not run campaign
cases concurrently.

**Reasoning — measured, and it refuted the hypothesis it was testing.** The
solve is memory-bandwidth bound (GPU-01), and an earlier strong-scaling run
showed only 1.40x going from 2 ranks to 4 -- 70% efficiency. From that I
expected extra ranks on one case to be nearly worthless and concurrent cases to
win, since each would at least keep its own working set.

The opposite, by a wide margin, on identical 150-iteration runs:

```
1 case  x 16 ranks : 37.9 s   0.026 cases/s
2 cases x  8 ranks : 220.2 s  0.009 cases/s   0.34x
4 cases x  4 ranks : 494.9 s  0.008 cases/s   0.30x
```

Checked before believing it: all runs completed exactly 150 iterations with no
early convergence, so the comparison is fair rather than an artifact of
different decompositions converging at different points.

The lesson is about the earlier measurement, not this one. That 70% figure came
from a different and much larger case, and I generalised a scaling curve from
one mesh to a workload it did not describe. Concurrent cases evidently contend
for cache and memory far worse than extra ranks on a single case do.

### RUN-01 · Simulations take 80% of the machine, not all of it — `SUPERSEDED`

**Value:** `CPU_BUDGET = 0.8` (12 of 16 ranks here) and `nice +10` on every
OpenFOAM command. Both overridable with `CAMBER_CPU_BUDGET` and `CAMBER_NICE`.

**Reasoning:** a campaign that saturates every core makes the laptop unusable
and the fans intolerable, and it was doing exactly that -- load average 17.2 on
16 cores.

The last cores were nearly free to give up anyway, which is the part worth
recording. The solve is memory-bandwidth bound (GPU-01), so ranks beyond about
half the cores contend for the same DRAM channels rather than adding
throughput: measured here, going from 2 ranks to 4 gave 1.40x, already only 70%
efficient. Dropping 14 ranks to 12 costs a few per cent of throughput and gives
back a quarter of the machine.

`nice` matters as much as the rank count and is easy to forget. It does not
reduce heat, but it means an interactive process preempts the solver
immediately, so the desktop stops stuttering even while the run continues. It
is applied to meshing as well: `snappyHexMesh` is single threaded but pegs a
core for minutes, which is just as noticeable on a laptop as the parallel
solve.

Measured after the change: load average 6.65, with 72.9% of CPU time in the
niced bucket and cores left idle.

## 10. The viewport (P6)

### UI-01 · The browser owns the view and nothing else — `FIXED`

**Value:** every number the panel displays is computed by the same Python the
CLI and the optimiser call. The JavaScript positions meshes and formats text.

**Reasoning:** the usual way a GUI and its engine drift apart is that the GUI
grows a convenient local copy of some calculation -- a mass sum here, a centre
of gravity there -- and the two answers diverge quietly until someone trusts
the wrong one. There is no second implementation of mass properties in
JavaScript, and a test asserts the payload matches `mass_properties()`,
`controllability()` and `solve_hover()` exactly.

The cost is a round trip per drag. It is worth paying: dragging a rotor
outboard re-solves hover trim, the allocation matrix and the interference
screen, and watching those three numbers move together is the entire point of
the tool.

### UI-02 · Standard library only on the server side — `FIXED`

**Value:** `http.server`, bound to loopback. No Flask, no FastAPI.

**Reasoning:** AI-07 says Camber must work with nothing installed, and a GUI is
exactly where a framework dependency creeps in and quietly becomes mandatory.
A single user on one machine does not need routing, templating or an ASGI
server. three.js from a CDN is the only external piece, and it is confined to
the browser.

There is no authentication or CSRF protection, deliberately: the server binds
to 127.0.0.1 and refuses to bind elsewhere, so those would be theatre around a
tool that never leaves the machine.

### UI-03 · Dragging happens in the horizontal plane — `PROVISIONAL`

**Reasoning:** a pointer supplies two numbers and a position needs three, so
something has to be assumed. Height is the right thing to fix: payload layout
is overwhelmingly a plan-view problem, and vertical dragging is both rarely
intended and impossible to aim. Height stays editable in the spec file.

The camera frames itself from the vehicle's own span rather than a fixed
distance, because a 5-inch quad and a heavy-lift airframe differ by an order of
magnitude and a fixed camera renders one of them as a speck.
