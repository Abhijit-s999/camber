# Technical decisions

Every non-obvious choice in Camber, with the value, the reasoning, and the
source. Tagged so that code comments and this document can refer to each other.
The format is lifted from an earlier CFD study, where it earned its keep.

**Status key**
`FIXED` — decided and implemented.
`OPEN` — needs a decision, or a study before it can be settled.
`PROVISIONAL` — implemented with a defensible default, worth revisiting.
`PARKED` — built and tested, then taken out of the release because the rest of
the app could not use it yet. The code lives on a branch; the record stays here.

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

### LIC-02 · The repository stays private — `SUPERSEDED by REL-01`

**Superseded 2026-10-15:** Abhijit asked for Camber to be pushed to GitHub and
released as an app. The reasoning below was why it waited; LIC-01 had already
cleared the data question it raised.

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

### LIC-01 · GPL-3.0 for code; no source data is redistributed — `RESOLVED`

**Value:** GPL-3.0-or-later for Camber. **No third-party measurements ship with
it.** The catalogue is built on the user's machine from the originals;
`camber catalogue fetch` downloads them, `camber catalogue sources` prints the
terms, and `NOTICE` carries the citations.

**Checked, 2026-09-09, and the answer is the same for all four:**

| source | terms as published | redistributable |
|---|---|---|
| UIUC Propeller Database | (c) 1994-2026 UIUC, citation requested, **no licence** | no |
| APC performance files | no licence statement at all | no |
| Tyto Robotics database | (c) 2026; terms of service silent on reuse | no |
| ThrustCurve.org | licences declared **per file**, some restricted | no |

None grants a redistribution right, and silence is not permission -- under
copyright the default is reserved. Raw measurements may not be copyrightable
everywhere, but a *compiled* database attracts protection for its selection and
arrangement in most jurisdictions, and more strongly under a sweat-of-the-brow
doctrine than under US law. So AI-08's original plan -- ship a signed, compiled
catalogue -- cannot be done.

**What ships instead is the builder.** That costs a network connection once, at
setup, which is the single place AI-07's offline guarantee does not hold. The
distinction that keeps it honest is setup versus runtime: fetching is a one-off
step the user initiates, and everything afterwards is offline and
deterministic.

Three of the four have no bulk download, and Camber does **not** scrape them.
It names the source and asks the user to gather it, because writing a scraper
for data whose terms forbid redistribution would be doing indirectly what the
licence forbids directly.

**Enforced, not merely intended.** `tests/test_licensing.py` fails the build if
any source data becomes tracked, if a large data blob appears, if a source is
marked redistributable without someone checking, or if `NOTICE` drifts from the
recorded terms. A licence obligation that depends on remembering is one a
`git add -A` will eventually break.

**What is safe to ship, and why:** `data/interference.json` holds 25 rows of
four *derived* design descriptors -- blockage, spacing, offset, disc loading --
fitted from Camber's own OpenFOAM runs. No third-party measurement is in it.
`data/validation_summary.json` is aggregate error statistics.

**Consequence: the repository can be made public whenever Abhijit wants**,
which was the only thing this decision was blocking. It remains private by
LIC-02, which is a separate choice and his alone.

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


### ARCH-05 · The generated frame goes back into the vehicle — `PROVISIONAL`

**The gap.** `camber generate` wrote an STL and stopped. The vehicle model went
on believing in its placeholder -- on the racing quad an 85 g box,
150 x 70 x 24 mm, under `[[frame]]` -- so hover, thrust-to-weight, inertia and
the interference estimate all described an airframe nobody was going to build.
For a tool whose name has "closed loop" in it, the loop was open at the one
point where the design becomes real.

**Value:** `camber generate --into vehicle.json` measures the finished mesh --
volume, centroid, inertia tensor, extents, footprint -- and writes the vehicle
back out with that body as its frame, then prints what changed. The body is a
`Tabulated` shape: the numbers are computed once from the mesh and carried as
numbers from then on, so evaluating it costs what a Box costs and the core
still imports only numpy (GEOM-03, AI-07). The STL is recorded as a source, not
required. Mass properties are measured while the mesh is still in metres and
NED, before the manufacturing export turns it into millimetres and Z-up.

**Blockage is a footprint, and the calibration had to survive that.** The
interference model's first feature is total body planform over disc area, and
it was trained on 25 CFD cases computed from bounding rectangles. For a box
that is exact. For a topology-optimised X-frame it is nonsense: the box is 400
cm^2 and the material in it is 119 cm^2, so the old formula would have put
blockage near 0.8 -- far outside the 0.19 to 0.61 the model has seen -- for a
part that is mostly air. `Shape.planform_area()` now defaults to *exactly* the
old formula, so every existing body reports what it always did and the fitted
model's inputs are unchanged (asserted for both examples to 1e-12); a
`Tabulated` shape reports its real shadow instead.

**The model now says when it is extrapolating.** `InterferenceModel.outside()`
names any feature outside the fitted envelope. A Gaussian process off its data
reverts to its prior mean and widens its sigma, which is honest to anyone
reading the sigma and invisible to anyone reading only the mean; generated
frames are exactly the layouts that can leave a 25-case envelope, so the check
is made and printed rather than left to the uncertainty.

**Found on the way:** `to_dict` serialised any shape it did not recognise as
`{"kind": "point"}`. Saving a vehicle with a new shape kept the mass and
silently deleted the geometry and inertia. It now raises.

**On a coarse smoke run** (25 iterations, 4 mm, not a design): mass 391 g to
430.5 g, T:W 12.83 to 11.65, yaw inertia up 12%, blockage 0.28 to 0.31,
interference 0.6% to 0.8%, inside the envelope. The numbers that matter will
come from the 1.5 mm run.

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

### OPT-07 · MAX_SPEED trades thrust margin against endurance, not mass — `FIXED`

**The bug:** the objectives were `[-thrust_margin, mass_kg]`. Installed thrust
is fixed by the chosen motors, so thrust margin is `T / (m g)` -- a function of
mass and nothing else. The pair states the same preference twice: every lighter
design is also higher-margin, no design ever wins on one axis and loses on the
other, and the front is one point however long the search runs.

This is OPT-04's failure wearing a different hat. There it was a variable set
that could not move an objective; here it is two objectives that are the same
axis. Both look identical from outside -- a degenerate front -- and neither
raises anything.

**Value:** `[-thrust_margin, -endurance_s]`, which is the trade a racer
actually makes: a bigger pack flies longer and accelerates worse.
`TestMaxSpeedObjectives` demonstrates the old degeneracy rather than asserting
it, by evaluating eight pack sizes and showing the old pairing admits exactly
one non-dominated point while the new one admits several.

**Worth stating generally:** before pairing two objectives, check they are not
functions of the same variable. It is easy to write down two things you care
about and not notice one determines the other.

### OPT-09 · Infeasible designs are ranked by violation, one front per level — `FIXED`

**The bug:** constrained domination says the smaller violation wins between two
infeasible designs, and the module docstring says so. `_constrained_ranks`
sorted the infeasible designs by violation and then returned them as *one*
front. Every one of them had the same rank, so the tournament fell through to
crowding distance, and the survival cut, which breaks an overflowing front by
crowding, threw the sort away. A population that started infeasible was pushed
toward spread in objective space and not toward feasibility at all.

**Found by** the clean-install check before announcing the project
(2026-10-26). `camber optimise examples/racer5.toml`, at the CLI's default seed
0 and 32 x 20, printed "feasible 0/32" and then a table of those 32 designs as
if they were the front. Over seeds 0-19 at that budget, 4 of 20 runs ended with
nothing feasible, and seed 0 stayed at 0/32 at 40 generations.

**Value:** each violation level is its own front; exact ties share one. The
same sweep afterwards: 0 of 20 runs without a feasible design, and racer5
averages 32.0/32 feasible against 25.6 before. quad5, feasible from the start,
barely moves: the median best placeholder endurance over 20 seeds went from
89.1 to 87.7 min, inside the seed-to-seed spread (interquartile range 83-94
before, 86-95 after). `test_an_infeasible_start_is_pulled_to_feasibility` pins
the racer5 case.

**And an infeasible front is no longer presented as an answer.** With nothing
feasible, `camber optimise` exits 1, names the constraint the closest design
still breaks, and writes neither `--out` nor `--front-json`. The app shows that
line as the job's failure. Before, `--out` handed the least-bad infeasible
design on to `generate`. The app test's optimiser job ran 8 x 2 on racer5 and
passed only because of that; it now runs the app's own budget, 40 x 30.

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

### GEN-01 · Straight tubular arms, deliberately — `SUPERSEDED by GEN-02`

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

**Why it was superseded.** The reasoning above is sound and the conclusion did
not follow from it. "No closed-form model can check the result" is an argument
for putting a model that *can* check it in the loop, not for declining to
generate the shape. The honest description of what GEN-01 produces is that a
human -- me -- decided a frame is a plate and four tubes, and the optimiser was
allowed to choose their diameter. That is sizing, not generation. GEN-02
replaces it. GEN-01's output remains available and remains the right default
when a beam-checkable frame is what is wanted.

### GEN-02 · The frame is grown by topology optimisation — `PROVISIONAL`

**Value:** `camber/generate/topology.py`. SIMP -- Solid Isotropic Material with
Penalisation -- over a voxelised design domain, with an 8-node hexahedral
finite-element solve inside every iteration. `camber generate` runs it.

Nothing in it decides what a frame looks like. The inputs are where load enters
(the rotor mounts), where the structure is held (the stack and battery
mounting), how much material there is to spend, and the mission's element size.
Everything between those is chosen by the compliance sensitivities.

**On the racing quad**, 40 g of carbon over a 91x91x3 domain at 3 mm elements
(24,843 elements, volume fraction 0.038), 60 iterations: compliance fell from
9.561e8 to 5.466e6, a factor of 175. The shape that came out is a central hub
with four tapered diagonal struts, each of which *forks into a Y* approaching
the rotor mount. Nobody drew that fork. It is what a single strut becomes when
the load it carries has to reach a mount wider than the strut is.

**Why the element matrix is integrated, not transcribed.** `hex8_stiffness`
does 2x2x2 Gauss quadrature. Published closed-form 24x24 tables exist and would
have been faster to type; two attempts to transcribe one produced a matrix that
was still square, still symmetric, still assembled without complaint, and had
negative eigenvalues -- a different physics problem, failing silently. The
quadrature version is checkable against the property that defines the element:
six zero eigenvalues, the rigid-body modes of an unconstrained brick, and
nothing else. `tests/test_topology.py` asserts exactly that.

**What this costs.** The FEA is a direct sparse solve at 126,960 degrees of
freedom, about 19 s per iteration on this machine after GEN-05, so a run is
half an hour rather than the seconds GEN-01 takes. It is not in the NSGA-II
inner loop and must not be: it runs once, on a layout the optimiser has already
chosen.

### GEN-03 · Continuation on the penalty, and connectivity is reported — `PROVISIONAL`

**The failure.** The first GEN-02 run looked like a success -- compliance down
175x, density field 96% binary, a plausible organic shape -- and had scattered
**35% of its material budget into 106 disconnected islands**. At threshold 0.4
the largest connected component held 65.3% of the material. The only symptom
downstream was that the meshed part weighed 26.3 g against a 40 g budget.
Pruning the islands is correct, since a disconnected island carries no load and
cannot be manufactured, but pruning a third of the structure means a third of
the budget was never spent on structure at all.

**Why it happens.** At `p = 3` an element at density 0.3 is worth 2.7% of solid
stiffness. Joining two members through a thin grey neck is therefore priced
almost as dearly as leaving them apart, and at a volume fraction of 0.038 there
is very little material to form necks with. The optimiser is free to park mass
where it does nothing.

**Value:** the penalty ramps from 1.0 to 3.0 in quarter steps over the first
half of the iteration budget. At `p = 1` the problem is convex and grey is
honestly priced, so a connected layout forms first; raising `p` then hardens
that layout rather than choosing a new one. `Result.connectivity()` reports the
island count and the fraction of material in the largest, and `camber generate`
warns when that fraction drops below 95%.

**The reporting matters as much as the fix.** This failure did not raise, did
not warn, and did not look wrong in the viewer. It was found by counting
connected components on a hunch. Anything that can be lost that quietly should
be measured every run.

### GEN-04 · Convergence needs the objective, not just the density field — `FIXED`

**The bug, found by running a control.** The standard SIMP stopping rule is
"the largest density change fell below 1%". Run without continuation, this
optimiser satisfied that rule at iteration 4 and reported `converged=True` with
compliance at 3.922e10 against the 9.958e8 it started from -- **forty times
worse than the uniform field**, declared converged.

The rule is wrong at low volume fractions for a specific reason: a few hundred
solid elements spread over thirty thousand can connect or sever a load path
while no single element moves much. Compliance is at its most violent precisely
where the density criterion goes quiet.

**Value:** `_has_settled`. Convergence requires both that the field has stopped
moving *and* that compliance has been flat over the last five iterations. The
window is what catches oscillation -- consecutive iterations differing little
while the run flips between two layouts, which is what the history
`[9.958e8, 1.3067e11, 2.6754e10, 3.9221e10]` actually was. That history is now
a test case.

Convergence is also not tested during the continuation ramp, since every
penalty increase moves the field on purpose.

### GEN-05 · The stiffness matrix is assembled once and refilled — `FIXED`

**Measured on the racing frame**, 33,124 elements and 126,960 DOF:

```
                          before      after
assemble K (per iter)     0.47 s      0.17 s
linear solve              37.04 s     18.67 s
                          -------     -------
per iteration             37.5 s      18.8 s        2.0x
structure (once)              --      1.7 s
```

Same displacement field to 4.4e-9 relative, which is the point: this is a
speed change, not a modelling change, and it is only worth having if it can be
shown to compute the same thing. `TestAssembly` asserts the fast path produces
a matrix identical to the obvious COO-and-slice one.

**Two separate things were wasteful.**

*Assembly.* The mesh does not move between iterations, so the sparsity pattern
of K never changes -- only the values do. The naive loop built a COO matrix of
19.1 million entries and converted it every iteration, which re-sorts all 19.1
million and re-sums duplicates for an answer that was already known. Then it
took `K[free, :][:, free]`: two fancy-index passes over a 127,000-square
sparse matrix, each allocating a full copy, to remove 1,014 rows. Now the sort
happens once, fixed DOFs are dropped at the index level rather than sliced out
afterwards, and each iteration is one gather-multiply and a `reduceat`.

*The solve.* `spsolve` defaults to COLAMD ordering and an unsymmetric
factorisation. K is symmetric positive definite, and saying so --
`permc_spec="MMD_AT_PLUS_A"` with `SymmetricMode` -- halves the time by cutting
fill-in, which is what dominates a 3D elasticity factorisation.

**Not moved to the GPU.** CuPy is installed and GPU-01 measured 11x the memory
bandwidth, which would suit a Krylov solver. It does not suit this one: SIMP
spans nine orders of magnitude of stiffness between void and solid, and an
unpreconditioned CG on that conditioning does not converge in any useful number
of iterations. A GPU path here needs algebraic multigrid first, not a port.

### GEN-06 · Mounts are bolt pads, not billets — `FIXED`

**The bug that made GEN-02 mostly not generative.** `frame_problem` forced a
solid block through the full domain thickness under the whole stack footprint,
39 x 39 x 12 mm, plus a column at each rotor. On the racing frame:

```
mass budget              956 elements    40.0 g
forced solid (keep)      820 elements    34.3 g
the optimiser chose      136 elements     5.7 g     14.2% of the budget
```

So 86% of the airframe was a brick I had specified, and the "generated" part
was what could be arranged with the remaining 5.7 g. After the fix the same
budget forces 72 elements, 3.0 g, and the optimiser places 92.5% of the mass.

**How it presented.** Not as an error. Compliance fell, the field went binary,
and the shape looked like load paths. The three symptoms were:

- material stranded in islands -- GEN-03 -- because what was left over could
  not span anything, so continuation improved it (65.3% to 73.3% of material in
  the largest component) without ever fixing it;
- a meshed part of *exactly* 1,088 facets and 17.85 cm^3 from two runs that
  had nothing in common, one of 80 iterations with continuation and one of 4
  without, because the block dominated both;
- a per-iteration density change of 0.01 to 0.04, which read as "settling" and
  was actually "nearly everything is frozen". It is 0.14 now.

The identical-derived-quantity signature is the one that gave it away, and it
is the second time on this project: CFD-02 was found the same way, by a cell
count that did not move when the input did. **An identical derived quantity
across different inputs means something is ignoring the input.**

**Value:** the domain thickness is the envelope the structure may occupy, not
the plate. A flight stack bolts on a 30.5 mm square of M3 and a motor on its
own boss, so `keep` is five 9 mm pads on a single layer each -- material at the
fasteners, which is what a frame actually needs -- and the supports are those
four bolt pads rather than the whole hub plan area. Fixing the hub made it
rigid by decree and hid whatever the optimiser would have built there.

`TestMountsAreNotBillets` asserts the forced-solid fraction stays under 15% of
the budget, that pads occupy at most two layers, and that the support does not
span the hub.

### GEN-07 · A flying airframe is a free body — `FIXED`

**Value:** thrust up at the motor pads, the payload's weight down at the stack
pads, and a 3-2-1 restraint -- one node pinned, a second held against two axes,
a third against one -- which removes the six rigid-body modes and nothing else.

**What the grounded version did.** Holding all four stack pads rigidly makes
each bolt an independent, infinitely stiff support, and the optimiser answers
that question correctly: four separate arms, each running from its motor to its
own bolt, never meeting in the middle because nothing requires them to. The
connectivity report called it fragmentation -- 28 islands, largest holding
22.7% -- and the structure was fine. The supports were wrong.

A self-equilibrated load set makes the restraint carry nothing: the measured
reactions are 1e-10 of the applied load. That is what makes it legitimate to
restrain a node that is also loaded, which would be a bug under a grounded
formulation. `TestTheFrameIsAFreeBody` measures the reactions rather than
assuming them.

The reaction check caught its own bug on the way, worth recording because it is
easy to repeat: **reaction is `Ku - f`, not `Ku`.** The restrained nodes carry
applied load too, and without the subtraction the test reports the applied load
back as if it were a reaction -- three z-reactions of exactly 0.76875 N, which
is exactly 49.2 N over 64 loaded nodes.

### GEN-08 · Pruning used the wrong connectivity and was deleting the arms — `FIXED`

**The bug.** `largest_connected` labelled with `scipy.ndimage`'s default
structuring element, which is 6-connectivity: two voxels are joined only if
they share a *face*. An airframe's arms run diagonally, and a diagonal in a
Cartesian grid is a staircase whose voxels touch at their edges. So every voxel
of every arm was its own island, pruning kept whatever happened to be
blockiest, and the meshed part came out at **1.3 g of a requested 40 g**.

```
coarse racing frame, 172 solid elements
  face-only (6)    96 components, largest holds   6.3%
  edge     (18)     2 components, largest holds  56.1%
  corner   (26)     1 component,  largest holds 100.0%
```

The same field is 96 pieces or one piece depending only on which rule you ask
with. This is the second time the connectivity report has accused a sound
structure: GEN-07's four arms were real too.

**Value:** pruning and `connectivity()` default to 26-connectivity, which is
what the element formulation itself uses -- elements meeting at an edge share
two nodes and transmit load through them, so it is the right rule for *is this
one structure*.

**But the face count still means something,** so it is still reported, as
`Result.hinges()`: the number of joins inside the structure that hold only at
an edge. Zero means every join is a shared face and the part can be made. 95,
which is what the coarse run scores, means members are about one element wide
and the discrete model is stiffer there than carbon would be -- a known SIMP
artefact, and a real manufacturability defect. It calls for more material or a
coarser grid, not more iterations, which is why it is a separate warning from
the stranded-material one.

`camber generate` also now reports the meshed part's shell count, since a
staircase that survives pruning can still reach the slicer as ninety-three
separate objects.

### GEN-09 · Two scale errors that a relative objective cannot see — `FIXED`

Both of these were in the code from the start, both are *global* factors on
compliance, and minimising compliance at a fixed volume does not depend on a
global factor at all -- so the optimal shape was right and every absolute
number was wrong. Neither could surface until GEN-10 needed compliance in
joules.

**The modulus was applied twice.** `KE = hex8_stiffness(E0, nu)` scales by the
modulus, and so does the SIMP interpolation `Emin + x^p (E0 - Emin)`. Assembled
stiffness therefore went as `E0^2`. At `E0 = 1`, which is what the solve ran at,
squaring is the identity -- so it passed every test in the file. Moved to a real
70 GPa it produced a compliance of 1e-17 J.

**The element was one metre across.** `hex8_stiffness(size=1.0)` is the
default and nothing ever passed `domain.element`. Element stiffness goes as
its edge, so at 5 mm elements the structure was 200 times too stiff.

Together they showed up as a cantilever **205 times stiffer** than
`wL^3 / 3EI`. Fixed, the same block lands at 0.0349 mm against the beam
formula's 0.0357 mm -- 2%, in the right direction, since a 5:1 stubby beam
carries real shear the Euler-Bernoulli formula ignores. That check is now a
test, and it is the first absolute validation this module could have: while
the solve ran in arbitrary units there was nothing to compare against.

**The general point.** A quantity that only ever appears in a ratio is a
quantity nothing is checking. Both errors survived 80-odd tests because every
one of them compared a compliance to another compliance.

### GEN-10 · The mass is a choice: a budget, or a price — `PROVISIONAL`

**Value:** `camber generate --mass 40` keeps the original behaviour, a hard
volume constraint the optimiser must meet exactly. `--price w` drops the
constraint and charges for material instead, so the mass is an outcome:

```
J = c / c_scale  +  w * m / m_scale
```

Both terms are dimensionless *and* O(1) at a design anyone would build,
because both scales are physical rather than invented:

- `c_scale` is the strain energy at the largest deflection the airframe may
  have -- applied load times that deflection, `--deflection`, default 1 mm.
- `m_scale` is the aircraft's own all-up mass, because a gram of airframe
  costs whatever any other gram costs.

So `w` reads directly as an exchange rate, and below 1 says material is
cheaper than compliance.

**Measured on the racing quad**, 60 iterations at 5.5 mm:

```
price      mass      mean deflection
0.05     167.7 g       0.01 mm
0.2       84.1 g       0.03 mm
1.0       40.1 g       0.10 mm
5.0       29.0 g       0.31 mm
20.0      23.3 g       0.89 mm
```

**`w = 1` independently picks 40 g**, which is the budget that had been
hardcoded, and which is what a 5-inch racing frame weighs. Nothing told it
that; it falls out of charging material at par against the airframe it lifts.
Treat the coincidence as encouraging rather than as evidence -- one vehicle,
one load case -- but it is the right kind of coincidence.

**Two earlier normalisations were wrong and are worth recording**, because both
were dimensionally impeccable and practically useless. `J = c/c_ref + w V/V_box`
against the starting compliance gave `w = 4` a 591 g frame and `w = 0.5` an
1151 g one: numbers with no scale cannot be quoted to anyone. Anchoring `w`
to the Lagrange multiplier a short constrained warm-up converges to was better
reasoned and still failed, because that multiplier climbs by orders of
magnitude as the design sharpens, so freezing it early leaves material cheap
forever -- `w = 1` gave 489 g. The scale had to come from the physics, not from
the run.

### GEN-11 · Mounts are bolt circles, and the holes are drilled — `PROVISIONAL`

**Value:** every fastener is modelled: a motor on its own 16 mm square of M3, a
flight stack on 30.5 mm. Each bolt contributes a solid boss -- material to bear
on, with edge distance -- and a voided hole through the thickness, and load
enters at the bosses, four per motor.

**What a single pad produced.** Applying thrust at one 9 mm pad per motor asks
for a load path that reaches the middle of the mount, and that is exactly what
came back: arms tapering to a point, with the mount a separate floating body
beyond them. Thirteen shells in the meshed part. The structure was answering
the question asked.

**Holes are a boolean, not a grid feature.** An M3 clearance hole is 3.4 mm
across -- under two cells at a 2 mm element, and no rounder than the element is
small. Asking a voxel grid for fastener tolerance is asking the wrong part of
the pipeline. `drill()` subtracts exact cylinders from the finished mesh, so
the hole lands on the bolt circle at full precision whatever the isosurface
did. The void mask remains, but only as a hint that keeps the optimiser from
filling a hole it will lose anyway.

Where the element cannot separate a 9 mm boss from a 3.4 mm hole, the boss is
kept whole and the hole waits for `drill`. Voiding it there would leave the
mount with no material and nowhere to apply load, which is worse than a
coarsely-modelled hole.

**Clearance is 1.7 mm radius, not 1.5.** A nominal-diameter hole does not admit
the bolt.

### GEN-12 · Multigrid, because the direct solver's fill-in is the resolution limit — `FIXED`

**Measured on this problem**, one solve of the same matrix:

```
element   elements    free dof     direct       AMG
2.5 mm      59,405     217,800       45 s        --
2.0 mm     110,976     394,143      376 s      23.1 s      16x
1.5 mm     264,992     904,197        --       45.5 s
```

The direct factorisation takes 8 times longer for 1.8 times the unknowns:
fill-in, not physics, is what put fine grids out of reach. Smoothed-aggregation
AMG preconditioning CG is linear instead -- 2.3 times the unknowns for 2.0
times the time -- and 1.5 mm elements now cost 55 s an iteration at 8.9 GB,
where the direct path could not run at all.

**The near-nullspace is the whole trick.** Aggregation coarsens by grouping
strongly-connected unknowns, and a hierarchy that does not know a rigid
rotation costs no energy builds coarse spaces that cannot represent one --
leaving the smoother to resolve bending on the finest grid, which is barely
better than plain CG. `rigid_body_modes` hands over the three translations and
three rotations. `TestMultigrid` asserts they store no strain energy, that
there are six independent ones, and that AMG reproduces the direct
factorisation's displacement field to 1e-6.

Above 150,000 unknowns `solver="auto"` switches over; the threshold is
measured, and `--solver` forces either.

### GEN-13 · The faceting is the geometry, not the render — `FIXED`

Marching cubes takes the isosurface of a *continuous* field, so in principle
the boundary carries sub-element information. SIMP is built to destroy exactly
that: the penalty exists to drive every element to solid or void, and by the
end the racing frame's field is 99.3% binary. With no intermediate values left
to interpolate the isosurface has nowhere to go but along the voxel boundary,
and the part comes out faceted at the element scale -- which reads as a
rendering artefact and is really the shape.

**Value:** resample, blur a little, and re-level. Cubic-spline zoom puts the
surface on a finer lattice than the elements; a small Gaussian restores the
intermediate band the penalty removed; and the isolevel is then chosen to
enclose the *design* volume rather than fixed at 0.5. On the 1.5 mm frame the
raw isosurface came out **11% light** -- 49.3 g of a 54.5 g design, mass the
optimiser had paid for and the mesher threw away. Resampled, it lands at 53.1 g.

**The blur is deliberately small, at 0.5.** Holding the volume fixed means a
stronger blur is paid for with a higher isolevel, and a higher level cuts the
thinnest material first: at 0.9 this part went from one solid body to nine,
purely in meshing. Taubin passes are nearly free by comparison -- 80 of them
cost 1% of volume and take 2% off the surface area -- so the smoothing budget
is spent there instead.

**Three mesh-handling bugs, all of which produced surfaces that were not
watertight and, being open, could not even be tested for inverted winding:**

- Padding was one element, which a blurred field spills straight through. The
  surface then reaches the edge of the array and marching cubes returns an
  open sheet. The margin now scales as three sigma of the blur.
- The `Trimesh` constructor was handed per-vertex normals, which makes its
  merge treat two coincident vertices with different normals as distinct -- so
  the merge silently did nothing.
- Vertices were merged *after* scaling to metres. `merge_vertices` welds by
  rounding to an absolute tolerance, and at metre scale on a 1.5 mm grid that
  collapses genuinely distinct vertices into each other, making non-manifold
  edges out of a surface that arrived closed. Cleanup now happens in voxel
  coordinates, where neighbours are order 1 apart, and the mesh is scaled
  afterwards.

Order matters in the cleanup too: welding has to precede dropping degenerate
faces, because a zero-area triangle between unwelded vertices is still holding
two patches together and deleting it punctures the surface.

### GEN-14 · The density filter was half-applied — `FIXED`

**The bug.** Sensitivities were transformed by `dc = H @ (dc / Hs)`. That is
the chain rule *for a density filter* -- and the density field was never
filtered. Nothing filtered anything. The optimiser was told about a smoothing
that was not happening: it could place a one-element-wide web, and the
sensitivity it read for doing so was the one it would have had if webs were
smoothed.

It was found by looking at a side view of an arm, which was a ragged lattice
of single-element skins and stringy webs -- structure with no length scale at
all, because nothing in the formulation had one.

**Value:** a real density filter, `xPhys = H x / Hs`, with the analysis run on
`xPhys` rather than on the design variables, plus a smoothed Heaviside
projection to put the edges back. A density filter of radius R cannot produce a
member much thinner than R, which is the mechanism that gives a minimum
thickness; the projection restores the crisp boundary the filter costs.
`--min-feature` quotes the radius in millimetres so the number stays meaningful
when the grid changes.

**Measured on a test cantilever**, 60 iterations, same problem both ways:

```
                 stiffness     grey elements
no projection      3.5x            32.5%
beta = 16          5.5x            10.0%
```

Crisper *and* stiffer, which is the expected direction: grey material is
structurally inefficient by construction, so a formulation that avoids it
spends the same volume better.

**Sharpening has to wait for the field to settle.** On a fixed schedule this
oscillates: every doubling of beta multiplies the projection gradient, the
optimality-criteria step saturates at the move limit, and the design flips
between two states -- compliance alternating between 4.09e7 and 1.89e12 on
consecutive iterations, forever. `should_sharpen` doubles beta only once the
design has stopped moving, or after a fixed number of iterations if it never
does. The move limit is also damped as `move / sqrt(beta)`, to keep each step
inside the range where the linearised sensitivity is still a guide.

That is the third time on this project that a schedule ignoring what the run
was actually doing drove it somewhere useless: GEN-03's penalty ramp and
GEN-04's stopping rule were the same lesson.

**The projection gradient is easy to omit** and its absence is not an error --
it is a run that never settles. `TestProjection` checks it against finite
differences at three sharpnesses.

### GEN-15 · Minimum thickness is enforced, not encouraged — `FIXED`

GEN-14 made stringy webs rarer and did not remove them. Two reasons, which
compound:

**Plain projection does not guarantee a length scale.** Projecting at
`eta = 0.5` pushes the filtered field to solid or void, but a member thinner
than the filter can still project to solid if its filtered value clears the
threshold. So the analysis now runs on the design projected at `eta + 0.2` --
*eroded*, every boundary pulled in -- while the part that is built and weighed
is projected at `eta`. A web too thin to survive erosion carries no load in the
analysis, its sensitivity says so, and the optimiser thickens or deletes it.
For compliance the eroded design is always the weakest of the
eroded/blueprint/dilated set, so this is the full robust formulation at the
cost of one solve per iteration rather than three.

**The filter was flattering every surface.** The usual normalisation divides by
the weights of the neighbours that *exist*, which is a mirror boundary: an
element on the top face sees its own layer reflected above it, and a one-layer
skin reads to the filter as twice as thick as it is. Erosion could never bite
at the faces -- and the faces are where a plate in bending puts its skins. The
normalisation is now the weight of the whole stencil, so outside the domain
counts as void, which it is: the faces of this domain are the surfaces of the
airframe.

**Measured** on a 48 x 12 x 6 cantilever, 90 iterations, fixed volume, the
fraction of material narrower than the minimum feature:

```
                        thin      blueprint compliance
before (GEN-14)        83.0%          6.14e7
erosion only           32.5%          9.37e7
void boundary only     18.6%          1.02e8
both                    8.0%          1.26e8
```

The two stack. **The cost is stated rather than hidden:** at a fixed volume
the blueprint is about twice as compliant, because the thin members were doing
real structural work that the constraint now forbids. Under `--price` the
optimiser answers that by buying more material, so expect heavier frames
rather than floppier ones.

**The diagnostic needed fixing before it could be trusted.**
`Result.thin_fraction` measures by morphological opening -- what a ball of the
feature size cannot fit inside is thin -- and a ball centred on a voxel can only
have an odd diameter. Asking for "narrower than 2 cells" that way silently
tested "narrower than 3", and condemned every member that met the spec. The
ball is now built by diameter, and calibrated on slabs of known thickness in
the tests.

**A domain can be too thin to host a length scale.** With air on both faces a
member needs a filter radius of clearance from each, so a domain fewer than
`2R + 1` elements thick has no interior and the void boundary greys the whole
field -- 57.5% grey on a two-element-thick test. `camber generate` warns rather
than returning mush that looks like a convergence problem. The 1.5 mm racing
frame is 8 elements thick against a need of 5; coarse 5.5 mm runs are 2 and
will warn.

Also removed: the dead `elif False:` branch left from GEN-10's abandoned
warm-start normalisation, and its unused `warmup` parameter. The reasoning it
carried is recorded in GEN-10; the code was just in the way.

### GEOM-05 · The motor bosses were on the ground side, and Z-up hid it — `FIXED`

**The bug.** GEOM-02 is NED, z down, so layer 0 of the topology domain -- the
most negative z -- is the *top* face. `frame_problem` put the motor bosses on
layer `nz - 1`, under a comment saying "top face". The rotors sit at
z = -42 mm, above the frame; the bosses were at z = +6 mm, the bottom. So the
reinforcement faced the ground and the face each motor actually bolts to had
no material guaranteed at all -- for the feature added specifically so the
motors could be mounted.

**Why it looked right.** The STL was written in NED and every viewer and
slicer is Z-up, which turns the frame over and put the bosses visually on top.
One convention error hiding another. It was found by reading the code against
GEOM-02 while planning something else, not by looking at a render -- the
render was the thing that was lying.

**Value:** the boss face is derived from where the rotors are, not asserted:
layer 0 if the rotors are above, `nz - 1` if below. The stack mounts on the
same face, as a racing stack sits on standoffs on the prop side. Tests pin
both, including that hanging the rotors below moves the bosses.

**Manufacturing exports are Z-up; the model stays NED.** `camber generate`
(millimetres) and `cad.export` without `metres=True` turn the part over by a
half-turn about x. Not a sign flip on z: that is a reflection, and a reflected
airframe is its own mirror image. The CFD path, `metres=True`, stays in NED,
because the CFD case is built in NED with the disc momentum pointing +z
(CFD-06). The unit flag and the orientation already meant the same thing --
"this is for CFD" -- so they travel together.

The assembly was trimmed to suit: storing which entry of the 24x24 element
matrix each non-zero carries, as int16, rather than the value as float64. At
265,000 elements that is 152 million entries, so it is 300 MB instead of
1.2 GB.

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

---

## 11. Flying it: simulation, ROS 2, and aero across the envelope (P7)

### SIM-01 · MuJoCo, driven by Camber's own allocation matrix — `PROVISIONAL`

**Value:** `camber/sim`. MuJoCo 3 is the physics engine: research-grade
rigid-body dynamics and contact, a Python API, a viewer, and it loads the
generated frame's mesh directly -- so the part that flies is the part that was
grown. It installs natively on this machine's Python 3.14. Gazebo, the usual
ROS partner, has no clean install path on a rolling Arch system.

**The model is built from the vehicle, not beside it.** Mass, CG and the full
inertia tensor are Camber's, stated explicitly (`inertiafromgeom="false"`), so
MuJoCo never re-derives them from geometry -- a generated frame's convex hull
is a solid slab and would make the aircraft several times heavier. Rotor
wrenches come from `control.allocation_matrix`, the matrix the optimiser
designed against. A second, hand-written model of the aircraft would be a
second thing to get wrong, and would let the simulator fly a better aircraft
than the one that gets built.

**Forces go in as a world wrench at the CG, not through MuJoCo actuators.**
Actuators are linear in their control; a rotor is not. Speed follows command
through a first-order lag (exact discrete solution, stable for any dt) and
thrust goes as speed squared, so half throttle is a quarter of full thrust.

**Checked against physics computable by hand, not against itself:**

- mass and inertia eigenvalues match Camber exactly;
- free fall matches the integrator's own discrete sum, g dt^2 N(N-1)/2, to
  1e-6 -- not 1/2 g t^2, which it misses by 0.4% for a reason that is not a bug;
- flying `solve_hover`'s trim -- rear rotors at 0.978 N, front at 0.939,
  because the CG sits 1.8 mm aft -- holds zero velocity, zero rates and zero
  drift for two seconds. A first version of this test used equal thrust on
  every rotor, "failed", and was the physics being right and the test wrong.

**Frames:** Camber is NED (GEOM-02); MuJoCo and ROS (REP-103) are FLU. One
matrix, a half-turn about x, converts positions, forces, moments and tensors;
it is a rotation, never a z sign flip, which would be a reflection.

**Crash means contact.** A crash is touching something while inverted or above
3 m/s. The first rule counted "upside down" as a crash and failed every flip at
90 degrees of roll.

### SIM-02 · Pilots are programmed at whichever layer the user cares about — `PROVISIONAL`

Position -> attitude -> rates -> moments -> mixer -> motors, and a pilot can
enter at any layer: `Controller.update` returns motor commands, and `Cascade`
exposes `attitude()` and `rates()` so an acro pilot need not write a position
loop and a trajectory pilot need not write a rate loop. Gains are stated as
accelerations and multiplied by the designed mass and inertia, so one set of
gains flies a 390 g racer and a 2 kg lifter. The mixer inverts the allocation
matrix and, out of reach, gives up yaw first, then collective, and holds roll
and pitch longest -- losing roll authority is how a quad flips over, losing
yaw is how it drifts.

### ROS-01 · ROS 2 Jazzy from RoboStack, in its own environment — `PROVISIONAL`

**Value:** `ros/`, a pixi workspace of conda-forge ROS 2 builds with their own
Python 3.12, per project, no root and no container. Jazzy targets 3.12; this
system runs 3.14 and updates continuously, so apt was never available and an
AUR build would break on the next system update. Camber is an editable
dependency, so the bridge imports the same simulator as the CLI. pixi was
installed with `PIXI_NO_PATH_UPDATE`: its installer otherwise appends to the
user's fish config.

**Four command levels, last one wins:** motors, a wrench (collective and body
moments, allocated by the mixer), a pose, and a `MultiDOFJointTrajectoryPoint`
setpoint with velocity and acceleration feed-forward. The last exists because a
pose can only say "go here and stop": a carrot-chasing pilot on poses settles at
about 2 m/s and never finished a lap in 26 s. With feed-forward, two laps in
under 30 s. A command stream silent for 0.5 s is treated as lost; the bridge
holds position and clears the feed-forward, so a dead pilot does not leave the
vehicle accelerating toward wherever it was going.

**The bridge refuses to be quiet about a second clock.** Testing left three
orphaned simulations publishing onto the same topics; sim-time timers fired at
400 Hz and odometry interleaved unrelated vehicles, with no error anywhere. The
bridge now reports any other `/clock` publisher on its domain.

### AERO-01 · The test conditions come from the vehicle — `PROVISIONAL`

**Value:** `camber/cfd/envelope.py`. A fixed test matrix tests a racer where it
never flies and misses where it lives. A multirotor has no angle of attack of
its own: it tilts to fly, and in steady level flight the tilt fixes the speed,
because thrust must hold the weight up (T cos = mg) and its horizontal part must
balance drag (T sin = D). So test points are tilts, each paired with the speed
the vehicle would be flying at there, and the range follows the mission: a
MAX_SPEED racer out to 65 degrees nose down (capped where thrust-to-weight runs
out, with 15% held back for control), an endurance vehicle to about 22. A
vehicle without rotors gets an alpha sweep through stall.

Angles use the aircraft convention everywhere, alpha = atan2(w, u): a multirotor
flying forward is nose down, air on its top, alpha *negative*. The first draft
of the aero table's docstring had that sign backwards.

**Rotors are in the CFD** as actuator discs at each condition's trim thrust. A
body in rotor wash is not a body in clean air, and testing the bare airframe
would measure a vehicle that does not exist. What is tabulated is the force on
the airframe patch only; the rotor thrust is the simulator's to apply, so
nothing is counted twice.

### CFD-09 · Forward flight needed a freestream, and the right sign on it — `FIXED`

Two bugs that hover could never reveal, because hover's freestream is zero.

**Sign.** The case wrote the freestream as (+V, 0, 0): air arriving from
behind, which for a vehicle flying forward is flying backwards. The relative
wind is now a vector from the test point's attitude, -V(cos a, 0, sin a) in NED.

**Far field.** Every boundary held *total* pressure at zero -- right when the
only flow is rotor-induced, wrong in a 31 m/s freestream, where it pins static
pressure at -U^2/2 wherever air enters and 0 wherever it leaves, imposing a
pressure drop that fights the flow. On the first forward-flight case the
pressure residual stalled at 0.19 and never fell, and the body force came out
at 14.9 N: a normal-force Cd near 3.8, about twice what a flat plate can do.

```
                     p residual     body force (NED)            time
totalPressure        0.19, stuck    (-1.04, +0.13, +14.87) N     303 s
freestream BCs       1.5e-4         (-0.39, -0.00, +5.49) N       87 s
```

`freestreamVelocity`/`freestreamPressure` switch between inflow and outflow on
the flux and hold static pressure at freestream. The case now converges and
stops itself, the normal-force Cd is 1.4, and the side force is zero as symmetry
demands instead of an invented 0.13 N. **The old boundaries inflated airframe
drag 2.7x**, which would have flown the simulated racer well below its real top
speed. Hover keeps the open boundaries, which are right there.

---

## 12. The app (P8)

### UI-04 · Camber is an app: projects, a pipeline, background jobs — `PROVISIONAL`

**Supersedes** the single-page viewport of UI-01 to UI-03, which is removed; its
camera field of view and interference readout were carried over first.

**Value:** `camber app`, or the Camber entry in the desktop launcher. A project is
a folder in `~/Camber` -- the vehicle as JSON, an undo history of one snapshot per
edit, results, and job logs -- so a project can be read, copied or backed up
without the app. The workspace is the pipeline in order -- Design, Optimise,
Structure, Aerodynamics, Fly -- around a 3D view that stays on screen throughout,
because the vehicle is the thing every step is about.

Long runs are the CLI in its own process group, recorded as JSON beside a log,
so a two-hour topology run survives the window closing and the app picks it up
again on reopening. Every job runs on a frozen copy of the design: editing on
while a run is in flight must not change what it is computing. Progress is
parsed from the log the CLI already writes. Only one topology or CFD job runs at
a time, because two share memory bandwidth and both crawl (CFD concurrency was
measured at 0.34x).

**The window is Qt WebEngine, the logic is a local server.** The window runs
under whichever Python has Qt -- on Arch the system's, from pacman -- and the
server under Camber's, which has the numerical stack; they talk over loopback
exactly as in one process. Without Qt it falls back to the browser.

### UI-05 · Symmetry is measured, not eyeballed — `PROVISIONAL`

The requirement, from use: during layout there was no way to tell whether rotors
and payload were actually symmetric, and a vehicle a few millimetres off flies --
the mixer hides it -- on an uneven thrust split, spending control authority to
hold itself level, with a lopsided frame grown for it.

So the design step gives every way of checking it that engineering drawings do:
orthographic Top, Front and Side views with the centreline drawn; *Mirror L-R*
and *Mirror 4-way*, which move a part's twins with it live during the drag and
keep centred parts on their plane; grid snap; exact numeric entry in mm; a
readout of each part's distance from the centreline; dimension lines to the
centreline for a selected part *and its twin*, so 88.0 beside 85.0 is visible
whatever the eye says; a symmetry report in millimetres, part by part, with a
button that splits the difference; a measure tool between any two parts; and
colour that means something -- amber for off-symmetric, red for clashing props.

Balance is shown beside it -- the CG against the thrust centre, and each rotor's
share of the hover thrust from `solve_hover` -- because that is what asymmetry
actually costs.

### UI-06 · The server needs a key now — `FIXED`

UI-02 left the old viewport unauthenticated because it only moved parts. This
server starts processes: CFD, topology runs, the simulator, and user pilot
scripts. Any web page in any browser on the machine can send requests to
127.0.0.1, so every API call carries a token generated at launch (handed to the
window in its URL, kept in session storage), and the Host header must be the
loopback address -- which is what stops a DNS-rebinding page reaching it by a
hostname that resolves to 127.0.0.1. Files are served only from inside the
project folder.

### SPEC-02 · Saving a vehicle dropped its constraints — `FIXED`

`to_dict` had no case for constraints and omitted them, so every JSON save
silently deleted the CG box, the camera's view cone and every minimum distance
a design declared: the optimiser's `--out`, the framed vehicle from
`camber generate --into`, and every app project, since projects are JSON. A
vehicle with no constraints is a valid vehicle, so nothing complained; it was
found because the app's rules panel showed none for a design known to have
three. Constraints now serialise as the exact inverse of their parser, and the
test is the whole document -- save, load, save again, and the TOML's every
top-level section present -- not the one field that was caught. The two example
JSON files had theirs restored.

The first thing the rules panel then showed: **racer5 fails its own camera
constraint**, the front props blocking the view by 1.4 degrees. The check uses
bounding spheres and is conservative, but it is exactly what hand placement
misses.

### PKG-01 · On Arch, an editable install with system Qt — `PROVISIONAL`

**Value:** `packaging/arch/install.sh`. A venv in `~/.local/share/camber` made
with `--system-site-packages`, so it sees `python-pyqt6-webengine` from pacman,
with Camber and its numerical stack (MuJoCo, trimesh, pyamg, build123d)
installed into the venv itself, where a system update cannot break them. A
desktop entry and icon, `camber` and `camber-app` on `~/.local/bin`. Nothing
needs root beyond the one pacman line the script prints if Qt or ffmpeg is
missing; `uninstall.sh` removes it all and leaves projects alone.

Editable, from the checkout: the app reads its templates, example pilots and
fitted interference model from the repository. That is the honest state for a
single-machine install and the obvious thing to change for distribution, which
wants a PKGBUILD and package data.

**Not in the installer:** OpenFOAM (hand-built, see the project notes) and ROS 2
(its own pixi environment in `ros/`). The app works without either; the
Aerodynamics step says so if OpenFOAM is missing.

### SIM-03 · The controller earned its integrators on the measured table — `SETTLED`

Flying the framed racer on its own CFD table, instead of on an estimate, broke
the flight controller three ways. Each failure was real, and none of them
showed up on estimated drag because the estimate is roughly half the true load
at cruising tilt.

**Height: the position loop had no integral.** Aerodynamic down-load grows with
the square of speed, so it is a disturbance that does not go away, and a P-D
loop can only answer it by moving: it needed about 7 m of altitude error to
balance 7 N. From 3 m, it flew into the ground at 38 m/s. There is now an
integral on the velocity loop (clamped, and frozen on the horizontal axes while
the tilt limit is saturated, or it winds up against a limit it cannot beat).
Height also gets its own, stiffer gains, as it does on PX4 and Betaflight,
because it faces the fastest-growing disturbance.

**Attitude: the rate integrator's clamp was a constant.** 0.5, which on this
vehicle caps the moment it can hold at 0.005 N m. The table's nose-down
pitching moment at 36 m/s is 0.14 N m, thirty times that, so proportional
action carried it at the price of a 25-degree attitude error: tilt ran from
the 65-degree limit to 82, and past there the vertical component of thrust
cannot hold the vehicle up. The clamp is now 30% of the moment the vehicle can
actually produce about each axis, computed from its own allocation matrix and
maximum thrusts. The gain went from 6 to 40 (against a proportional 22) for the
same reason: at 6 the integrator needed 7 s to wind up against a moment that
builds in 1.5 s.

**Collective: the projection went negative.** Thrust along body z is the
desired force projected onto the body's current z axis, which is negative when
the attitude must swing more than 90 degrees -- as it does from 65 degrees of
bank one way to 65 the other, coming out of a hairpin. Clamped at zero, the
vehicle lost thrust *and* attitude authority together, because rotors cannot
spin backwards and a mixer at zero collective cannot make a moment either. It
fell 1.2 m through every reversal. The collective now has a floor of 40% of
the demanded force.

**Tests:** steady force and moment loads applied directly, so the behaviour is
checked at hover instead of by flying fast and hoping. Including the negative
one -- with the integrator disabled the vehicle sags, and the test asserts it.

### AERO-02 · What the CFD sweep changed about the racer — `SETTLED`

Six cases, 10 to 65 degrees nose-down, on the grown airframe. Body force along
z, measured against the estimate the optimiser and simulator had been using:

| tilt | speed | CFD | estimate |
|---|---|---|---|
| 65° | 30.6 m/s | 9.00 N | 9.75 N |
| 43° | 27.8 m/s | 5.95 N | 4.56 N |
| 32° | 27.3 m/s | 4.66 N | 2.65 N |
| 21° | 24.1 m/s | 2.39 N | 0.95 N |

The estimate is close at the steep end and wrong by a factor of two and a half
in the middle, which is where a racer spends its lap. It is a flat-plate
projection and cannot see the rotor wash over the frame or the shape the
optimiser grew.

**What it costs in the air:** on the measured table the framed racer's best lap
is at the 14 m/s setting, 6.64 s. Asking for 18 or 25 is *slower*, 9.3 s,
because the drag it did not know about makes the hairpins harder. On the
estimate the same vehicle laps 6.5 s at any setting above 14 and never learns
this. A vehicle that is faster on the estimate than on its own aerodynamics is
the entire reason the CFD step exists.

**Also fixed here:** the printed table paired each angle with the speed from a
different case. Rows sort by angle; the speed column was left in run order, so
-65 degrees was listed beside the speed measured at -10. The coefficients were
right -- each used its own dynamic pressure -- but the table was unreadable.


### SIM-04 · Flying it by hand, and watching it live — `SETTLED`

**Value:** the simulation runs in the app, in a thread paced to the wall
clock, and the browser sends sticks and receives telemetry in one round trip
per animation frame. No second channel, so what you see and what you are
steering cannot drift apart.

**Why not a stream plus a separate input channel:** two channels need keeping
in step, and the stdlib server this app is built on does not do WebSockets.
One POST carrying sticks in and state out is simpler and, measured, faster
than it needs to be: 0.19 ms median round trip, against the 16 ms a 60 Hz
frame allows.

**Two bugs this turned up, both from turning on keep-alive** (at 60 requests a
second, HTTP/1.0's connection-per-request leaves tens of thousands of sockets
in TIME_WAIT):

- *Nagle's algorithm.* Small request, small response, one after another, is
  the exact pattern where it waits for the other end's delayed ACK: 41 ms a
  round trip, which would have capped the view at 24 Hz and looked like slow
  physics. `TCP_NODELAY`, and it went to 0.19 ms.
- *An unread body poisons the connection.* A request refused before its body
  is read -- an unauthorised POST -- leaves the body in the socket, and the
  next request on that connection parses `{"do": "start"}` as its request
  line. Bodies are now read before anything can refuse them.

**Modes:** assist, angle, acro, and a throttle curve with hover at mid-stick
on any vehicle. In assist, the hold point rides with the vehicle until it has
actually stopped -- latching it where the stick was released made it brake
from 12 m/s, overshoot by 11 m and then fly *backwards* to the release point.

**In the 3D view**, three cameras: chase, FPV at the nose (at the centre of
mass you are inside the battery), and orbit. Two things the layout editor left
behind had to be turned off for flight: the camera's field-of-view cone, which
is a layout aid, and `OrbitControls.update()`, which re-aims the camera at its
own target *even when disabled* and silently undid the chase camera every
frame -- the vehicle sat 2.8 m away, just outside the frame, while the camera
looked wherever the editor had last been pointed.

### SIM-05 · What the plant models, and what it was missing — `SETTLED`

Asked directly what the simulator accounts for. Three of four were already
there; the fourth was not, and it is the one that bites hardest on a vehicle
whose rotors do not cancel.

**Already modelled:** aerodynamic drag (estimated, or the measured CFD table),
first-order motor lag with the exact discrete solution, thrust as speed
*squared*, and the steady reaction torque that lets a multirotor yaw at all.

**Added — the rotors' angular momentum.** They are not massless discs that
make thrust appear:

- *Reaction to spinning up*, −J dω/dt about each rotor axis. On a snap yaw
  command every rotor changes speed at once and this arrives **before** the
  aerodynamic yaw torque, which cannot change until the new speed exists.
- *Gyroscopic precession*, −ω_body × h. Pitch a spinning rotor and it answers
  in roll. A balanced quad cancels this in hover; three rotors, five, or a
  tilting VTOL never do — which is exactly where this work is heading.

Checked by conservation, not by eye: with thrust suppressed, spinning the
rotors up turns the airframe the other way and the total angular momentum
holds to within a percent.

**Added — latency.** Everything between deciding and acting: the flight
controller's loop, the link, the ESC. A pure transport delay, not a filter —
the command sequence comes out unchanged, only later. Default 6 ms. Measured
on the gate course: 0 ms laps in 6.62 s, 6 ms in 6.58, and 20 ms in 8.07,
which is the real reason a laggy link cannot carry a hot rate loop.

**Rotor speed** comes from static thrust by momentum theory, with the
coefficient scaled by blade count (0.055 a blade). A 5-inch three-blade rated
12.3 N lands at 29,000 rpm, against high-twenties on a bench; with a single
blade's coefficient it read 37,000, which no 5-inch propeller survives. It is
used only for the momentum terms — thrust itself still comes from the
catalogue or the blade-element tier, never from a guess.

### SIM-06 · Wings, so an aeroplane can be a first-class vehicle — `PARKED`

**Parked for v0.1.0 (REL-01).** Built in `861674b` and reverted for the
release: wings are in neither the saved format nor the app, so no user could
reach them. It comes back with that work.

A multirotor's body aerodynamics are drag: a force that only opposes motion.
A wing turns forward speed into a force *across* the flight path, and that is
what holds a plane up and lets it turn. `camber/sim/wing.py` adds it, as a
strip model -- one panel per surface, the way flight simulators have done it
for decades:

- The air a surface meets is the body velocity plus the spin carried to where
  it sits, `v + omega x r`. A tail behind the CG feels the pitch rate as a
  change in angle of attack, which is most of what damps a plane's pitch;
  without it an aircraft porpoises. Verified: a tail opposes a pitch rate with
  no deflection.
- Lift grows linearly with angle of attack, matching thin-wing theory to a
  percent, then **stalls** -- folded to the flat-plate curve past the stall
  angle, so a controller cannot buy unlimited force by pulling harder.
- A control surface adds camber: its deflection shifts the lift curve.
  Elevator makes a clean pitching moment, ailerons a clean roll with no net
  lift change, a fin yaws.

**Wired into the plant** through one generalisation: a controller may return a
`Command` carrying motor commands *and* control-surface deflections, and an
aero model offering `wrench(v, omega)` is given the body rate as well as the
velocity. A multirotor pilot that returns a bare motor array still works
unchanged -- the plant splits whatever it is handed.

**A fixed-wing autopilot** (`FixedWing`) flies the standard cascade: speed to
throttle, height to a pitch attitude to elevator, heading to a bank angle to
ailerons, with rudder to coordinate. Flown on a 1.25 kg pusher aeroplane it
holds 50 m and 18 m/s, beats the single prop's reaction torque to stay
wings-level, and turns to a commanded heading -- banking 35 degrees into the
turn and rolling out on the new heading, not skidding round flat.

**What this is not:** the strip model has no compressibility, no Reynolds
effects, no wing-body interference, and no downwash from the wing onto the
tail. Those matter to a real design and belong in the CFD tier, not in the
loop that flies the plane a thousand times a second. Wings are not yet in the
saved vehicle format or the app; this is the physics and the controller,
tested, with the plane built in code.

### SIM-07 · Thrust vectoring is a moving configuration, not a magic force — `PARKED`

**Parked for v0.1.0 (REL-01).** Built in `8074057` and reverted for the
release: no controller uses a tilt yet and the app cannot place one. It comes
back with the wing work.

`tilt_rotor` is a saved vehicle component: it records a neutral thrust axis,
hinge axis, physical stops and a servo time constant.  It deliberately does
not mutate the design specification during a flight.  The simulator owns the
instantaneous servo angle, advances it with the same exact first-order update
used for motor lag, then rebuilds the force-and-moment allocation at that
angle.

That last detail is non-negotiable.  At ninety degrees a rotor produces
forward force rather than lift; retaining the allocation matrix from take-off
would let the controller claim it was commanding one wrench while the plant
received another.  A `Command` can therefore carry optional `tilts` alongside
motors and control-surface deflections.  Existing fixed-rotor controllers are
unchanged, and a tilt target has no effect until a vehicle actually declares a
tilting rotor.

This is the plant foundation for civilian tilt-rotor and VTOL transition
controllers.  It intentionally does not prescribe a transition schedule:
that depends on the aircraft's measured wing and prop-wash aerodynamics, and
is a controller/design problem rather than an invented universal curve.

---

## 13. Release

### REL-01 · v0.1.0 ships what a user can finish — `FIXED`

**Value:** public on GitHub, GPL-3.0-or-later, tagged `v0.1.0` with a wheel.
Before the tag, everything a user could reach but not complete was taken out,
and everything a user could not reach was parked:

| what | why it went | where it is |
|---|---|---|
| wings and the fixed-wing autopilot (SIM-06) | not in the saved format or the app | `861674b`, reverted |
| tilting rotors (SIM-07) | no controller commands a tilt; the app cannot place one | `8074057`, reverted |
| the tricopter preset | three fixed rotors cannot hold heading without SIM-07 | removed; the app now stops at four rotors |
| the `min_time_to_target` and `min_stable_speed` missions | both scored a constant, so the search optimised mass alone without saying so | removed; a test now checks every mission's first objective responds |
| APC, Tyto and ThrustCurve as sources | listed but never fetched or read; `catalogue fetch` always exited 1 | removed; they return with component selection |
| the PETSc hook in `foam-env.sh` | the GPU route it served is blocked (GPU-02) | removed |

**Fixed on the way, because a clean install found them:**

- `aerosandbox` and `neuralfoil` were imported by the blade-element tier and
  declared nowhere. The dev environment happened to have them; a fresh install
  failed `camber validate`. They are the `physics` extra, which `app` includes.
- `camber catalogue fetch` downloaded the UIUC zip and left it zipped, while
  every reader takes the extracted tree. It now extracts, refusing any entry
  that would land outside the data directory.
- The app's Optimise chart printed momentum-theory endurance as "Endurance min"
  with no caveat. OPT-03 measured it 16.5x too long. The chart now says so.

**Reasoning:** a first release is read as a claim about everything in it. Code
that is tested but unreachable, or reachable but silently wrong, costs more
trust than it adds features, and git keeps it either way.

**Also before publishing:** the history was rewritten, author addresses
included, so commit hashes differ from any noted elsewhere before 2026-10-15.

### PKG-02 · The wheel carries what the app reads from disk — `FIXED`

**Value:** `pyproject.toml` force-includes the interference model, the
validation summary, the two template vehicles, the example pilots and
`foam-env.sh` into the package. `camber/paths.py` prefers a checkout's copy and
falls back to the package's, so a development tree and an installed wheel run
the same code. `tests/test_packaging.py` fails if a template or pilot is added
without being shipped.

**Reasoning:** PKG-01's editable install read all of these from the repository,
which is honest for one machine and broken for anyone else: an installed wheel
would have had no templates to start a project from and no OpenFOAM wrapper.
Verified by installing the built wheel with `[app]` into an empty environment
and running `validate`, `optimise --prop`, `sim`, `generate`, `cad` and the app
server from outside any checkout.

**Not shipped:** the `bench` console script. It pointed into a module the wheel
leaves out (AI-04), so every wheel install had a `bench` command that could only
crash. In a checkout it is `python -m bench.cli`.


### PKG-03 · The README's commands work from an installed wheel — `FIXED`

**Value:** every vehicle argument goes through `cli._example`. A path that
exists is used as given. One that does not, such as `examples/quad5.toml` or
plain `quad5`, falls back to the copy PKG-02 ships inside the package, and the
CLI says which file it used. Pilots (`--pilot examples/pilots/flip.py`) resolve
the same way. A file that really is missing is now `spec error: ... no such
file or directory`, exit 2, where it used to be a Python traceback.

`camber aero` asks `cfd.run.missing()` before it writes anything. Without
OpenFOAM it says the step needs it, that it is optional, and how to point
Camber at one, and exits 3, the code `sim` uses for a missing MuJoCo. Before,
it wrote a case folder and died inside blockMesh with a traceback.

**Reasoning:** PKG-02 was verified by running each command against files that
happened to be there. The 2026-10-26 check went the way a stranger would:
the release wheel installed with `uv tool install` into an empty `HOME`, run
from an empty folder, following the README line by line. The first line of
its command-line section failed, because `examples/` exists only in a checkout.
`tests/test_cli.py` runs those commands from an empty working directory.

### PKG-04 · The mesh boolean is a declared dependency — `FIXED`

**Value:** `manifold3d` is in the `generate` extra, so in `app` too.

**Reasoning:** `drill()` (GEN-11) and the CFD surface union go through
trimesh's booleans, and trimesh has no engine of its own: it uses manifold3d if
it is installed, and failing that a `blender` on `PATH`. The development
machine has Blender, so every local run, including the clean-install check in
PKG-03, drilled holes without anyone noticing which engine did it. The first CI
run on a bare Ubuntu runner (2026-10-26) had neither and failed all five
drilling tests with "No boolean backend". A user without Blender got, after
the half-hour topology solve, a frame with no bolt holes and one warning line
that `camber generate` printed and carried on past.

With manifold3d installed trimesh prefers it to Blender, so a full install now
runs the engine CI tests. The whole suite passes on it, drilling included.
