import { Viewport } from "./viewport.js";
import { Live } from "./flying.js";

// ---- plumbing ----------------------------------------------------------------
const params = new URLSearchParams(location.search);
const TOKEN = params.get("t") || sessionStorage.getItem("camber-token") || "";
if (TOKEN) sessionStorage.setItem("camber-token", TOKEN);
if (params.has("t")) history.replaceState(null, "", location.pathname + (params.get("p") ? `?p=${params.get("p")}` : ""));

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 1) => (x == null || !isFinite(x) ? "–" : (Math.abs(x) < 0.5 * 10 ** -d ? 0 : x).toFixed(d));
const nice = (n) => n.replace(/_/g, " ");
// "12 Sep 2026, 05:04": day and month spelled out, never 12/9 or 9/12.
const when = (t) => new Date(t * 1000).toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });

async function api(method, path, body) {
  const res = await fetch(path, {
    method, headers: { "X-Camber-Token": TOKEN, "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status}`);
  return data;
}

let toastTimer;
function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (err ? " err" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), err ? 6000 : 2600);
}
window.camberToast = (m) => toast(m);
const guard = (fn) => async (...a) => { try { return await fn(...a); } catch (e) { toast(e.message, true); } };

// ---- state -------------------------------------------------------------------
const S = { pid: null, v: null, step: "design", jobs: [], sel: null, measure: null,
            dragging: false, pilots: [], pilot: "gates", flyAero: false, envelope: null,
            live: null, flyMode: "assist",
            gen: { mode: "price", price: 1.0, mass: 40, min_feature: 3, quality: "standard" },
            opt: { population: 40, generations: 30 } };
const P = () => `/api/p/${S.pid}`;
let live = null;

// ---- home --------------------------------------------------------------------
const TEMPLATES = {
  racer5: ["Racing quad, 5-inch", "Built for speed. Four rotors, 390 g."],
  quad5: ["Endurance quad, 5-inch", "Built to stay up. Gentle, efficient."],
};

async function home() {
  S.pid = null;
  history.replaceState(null, "", location.pathname);
  $("#work").hidden = true;
  $("#home").hidden = false;
  const r = await api("GET", "/api/projects");
  $("#home-where").textContent = `Projects are folders in ${r.home}`;
  let tpl = S.tpl || r.templates[0];
  const draw = () => {
    $("#templates").innerHTML = r.templates.map((t) => `
      <button class="tpl ${t === tpl ? "on" : ""}" data-tpl="${esc(t)}">
        <b>${esc(TEMPLATES[t]?.[0] ?? t)}</b><small>${esc(TEMPLATES[t]?.[1] ?? "")}</small></button>`).join("");
  };
  draw();
  $("#templates").onclick = (e) => { const b = e.target.closest("[data-tpl]"); if (b) { tpl = S.tpl = b.dataset.tpl; draw(); } };
  $("#projects").innerHTML = r.projects.map((p) => `
    <li data-pid="${esc(p.id)}"><b>${esc(p.name)}</b><span>${when(p.modified)}</span></li>`).join("");
  $("#no-projects").hidden = r.projects.length > 0;
  $("#projects").onclick = (e) => { const li = e.target.closest("[data-pid]"); if (li) open(li.dataset.pid); };
  $("#create").onclick = guard(async () => {
    const name = $("#new-name").value.trim();
    if (!name) { $("#new-name").focus(); return toast("Give the project a name first"); }
    const { id } = await api("POST", "/api/projects", { name, template: tpl });
    open(id);
  });
  $("#import").onclick = guard(async () => {
    const path = $("#import-path").value.trim();
    if (!path) return;
    const { id } = await api("POST", "/api/projects", { import: path });
    open(id);
  });
}

// ---- workspace ---------------------------------------------------------------
let vp;
async function open(pid) {
  S.pid = pid;
  S.sel = null;
  history.replaceState(null, "", `?p=${encodeURIComponent(pid)}`);
  $("#home").hidden = true;
  $("#work").hidden = false;
  if (!vp) {
    live = new Live(null, (m, path, body) => api(m, P() + path, body), $("#hud"),
                    { exit: () => land(), lost: (e) => { land(); toast(e.message, true); } });
    // Handles for the console, and for automated checks of the 3D view.
    window.__camber = { S, vp: () => vp, live: () => live };
    vp = new Viewport($("#view"), $("#labels"), {
      select: (n) => { S.sel = n; if (S.step === "design") panel(); },
      moved: guard(async (name, pos) => { await setV(await api("POST", `${P()}/move`, { name, pos, link: vp.link })); }),
      measured: (m) => { S.measure = m; if (S.step === "design") panel(); },
      dragging: (d) => (S.dragging = d),
    });
  }
  vp._framed = false;
  await setV(await api("GET", `${P()}/vehicle`));
  S.pilots = (await api("GET", `${P()}/pilots`)).pilots;
  await pollJobs();
}

async function setV(v) {
  S.v = v;
  $("#project-name").textContent = v.name;
  $("#undo").disabled = !v.can_undo;
  $("#redo").disabled = !v.can_redo;
  await vp.setVehicle(v, `${P()}/frame.stl`, TOKEN);
  rail();
  panel();
}

const edit = (path, body, msg) => guard(async () => {
  await setV(await api("POST", `${P()}/${path}`, body));
  if (msg) toast(msg);
});

// ---- the rail: the pipeline, in order ----------------------------------------
function latest(kind) { return S.jobs.find((j) => j.kind === kind); }
function jobLine(j) {
  if (!j) return "";
  if (j.status === "running") return j.progress != null ? `running · ${Math.round(j.progress * 100)}%` : "running";
  return { done: "finished", failed: "failed", cancelled: "cancelled", interrupted: "interrupted" }[j.status];
}
function rail() {
  const v = S.v;
  if (!v) return;
  const off = v.symmetry.xz.max_error_mm;
  $("#s-design").textContent = `${v.parts.length} parts · ${off < 0.05 ? "symmetric" : fmt(off) + " mm off"}`;
  $("#s-optimise").textContent = jobLine(latest("optimise")) || "not run";
  $("#s-structure").textContent = v.has_generated_frame ? "grown frame" : (jobLine(latest("generate")) || "placeholder frame");
  $("#s-aero").textContent = v.aero_table ? "CFD table" : (jobLine(latest("aero")) || "estimated");
  $("#s-fly").textContent = jobLine(latest("record")) || "";
  $$("#steps button").forEach((b) => b.classList.toggle("on", b.dataset.step === S.step));
}

// ---- panels -------------------------------------------------------------------
function panel() {
  if (!S.v) return;
  const el = $("#panel");
  const scroll = el.scrollTop;
  el.innerHTML = { design, optimise, structure, aero, fly }[S.step]();
  el.scrollTop = scroll;
  bindPanel(el);
}

function status(ok, label, value, note = "", level = null) {
  const cls = level || (ok ? "ok" : "warn");
  const mark = cls === "ok" ? "✓" : cls === "bad" ? "✕" : "!";
  return `<div class="status ${cls}"><i>${mark}</i><span>${label}</span><span class="v">${value}</span>${note ? `<small>${note}</small>` : ""}</div>`;
}

function design() {
  const v = S.v;
  const sel = v.parts.find((p) => p.name === S.sel);
  let html = "";
  if (sel) html += partSection(sel);
  else html += `<section class="sec"><h3>Layout</h3><p class="lead">Click a part to place it exactly. Drag the arrows to move it; with <b>Mirror L–R</b> on, its twin on the other side follows. Use <b>Top</b> view to check symmetry by eye, and the numbers below to check it for real.</p></section>`;

  // Symmetry: the thing that is easy to get wrong by eye.
  const lr = v.symmetry.xz, fb = v.symmetry.yz;
  const lrOk = lr.max_error_mm < 0.05 && !lr.unmatched.length;
  const rows = lr.rows.map((r) => `${esc(nice(r.a))}${r.b ? " ↔ " + esc(nice(r.b)) : " (centre)"} ${fmt(r.error_mm)} mm`).join("<br>");
  const yaw = Math.abs(v.yaw_balance) < 1e-9;
  html += `<section class="sec"><h3>Symmetry</h3>
    ${status(lrOk, "Left–right", lrOk ? "exact" : fmt(lr.max_error_mm, 2) + " mm off", lrOk ? "" : rows + (lr.unmatched.length ? `<br>No twin: ${lr.unmatched.map(nice).map(esc).join(", ")}` : ""))}
    ${vp.link === "4way" ? (() => {
      const fbOk = fb.max_error_mm < 0.05 && !fb.unmatched.length;
      return status(fbOk, "Front–back", fbOk ? "exact" : fb.unmatched.length ? `${fb.unmatched.length} without a twin` : fmt(fb.max_error_mm, 1) + " mm off",
        fbOk ? "" : [...fb.rows.map((r) => `${esc(nice(r.a))}${r.b ? " ↔ " + esc(nice(r.b)) : ""} ${fmt(r.error_mm)} mm`),
                     ...(fb.unmatched.length ? [`No twin: ${fb.unmatched.map(nice).map(esc).join(", ")}`] : [])].join("<br>"));
    })() : ""}
    ${status(yaw, "Prop spins", yaw ? "balanced" : "unbalanced", yaw ? "" : "Reaction torques do not cancel: it will yaw in hover unless the rotors run unevenly.", yaw ? "ok" : "bad")}
    <div class="actions">
      <button data-act="sym-lr" ${lrOk && yaw ? "disabled" : ""}>Make left–right symmetric</button>
      <button data-act="sym-4" class="ghost">4-way</button>
    </div></section>`;

  // Balance: where the weight is against where the thrust is.
  const b = v.balance;
  const [cf, cr] = b.cg_offset_mm;
  const off = Math.hypot(cf, cr);
  const split = b.hover_split ? Object.entries(b.hover_split) : [];
  const mx = Math.max(...split.map(([, s]) => s), 0.0001);
  html += `<section class="sec"><h3>Balance <span class="sub">${fmt(v.mass_g, 0)} g</span></h3>
    ${status(off < 1, "CG over thrust centre", off < 0.05 ? "centred" : `${fmt(Math.abs(cf))} mm ${cf > 0 ? "fwd" : "aft"}${Math.abs(cr) > 0.05 ? `, ${fmt(Math.abs(cr))} ${cr > 0 ? "right" : "left"}` : ""}`,
             off < 1 ? "" : "The rotors nearest the CG work harder to hold it level.")}
    ${b.hover_split ? `<div class="bars">${split.map(([n, s]) => `
      <div class="bar-row"><span>${esc(nice(n))}</span><span class="track"><b style="width:${(s / mx) * 100}%"></b></span><span class="n">${fmt(s * 100)}%</span></div>`).join("")}</div>
      <p class="hint">Share of hover thrust per rotor. ${fmt(b.hover_imbalance * 100)}% between hardest- and easiest-working.</p>`
      : `<p class="bad">${esc(b.hover_reason || "Cannot hover.")}</p>`}
    <dl class="kv"><dt>Thrust to weight</dt><dd>${fmt(b.thrust_to_weight, 2)} : 1</dd>
      <dt>Wheelbase</dt><dd>${fmt(v.wheelbase_mm)} mm</dd>
      <dt>Can command</dt><dd>${esc(b.controllable.join(" "))}</dd></dl></section>`;

  // Clearance: the other thing hand placement gets wrong.
  const c = v.clearance;
  html += `<section class="sec"><h3>Clearance</h3>
    ${status(!c.conflicts.length, "Prop tip to tip", c.min_tip_gap_mm == null ? "–" : `${fmt(c.min_tip_gap_mm)} mm`,
             c.conflicts.map((x) => `${esc(nice(x.a))} ${esc(x.what)} ${esc(nice(x.b))}`).join("<br>"),
             c.conflicts.length ? "bad" : "ok")}</section>`;

  if (v.constraints.length || v.wash) {
    const w = v.wash;
    html += `<section class="sec"><h3>Rules <span class="sub">from the design file</span></h3>
      ${v.constraints.map((c) => status(c.ok, esc(c.label), c.ok ? "met" : "broken", esc(c.note), c.ok ? "ok" : "bad")).join("")}
      ${w ? status(!w.loss || w.loss < 0.02, "Thrust lost to the airframe", w.loss == null ? `${fmt(w.overlap * 100, 0)}% in wash`
               : `${fmt(w.loss * 100)}%`, `${fmt(w.overlap * 100, 0)}% of the airframe sits under a prop.`
               + (w.outside.length ? ` Outside what the model was calibrated on (${esc(w.outside.join(", "))}); treat as a guess.` : ""),
               w.loss != null && w.loss >= 0.05 ? "bad" : null) : ""}
    </section>`;
  }

  if (S.measure) {
    const m = S.measure;
    html += `<section class="sec"><h3>Measurement</h3>
      <dl class="kv"><dt>${esc(nice(m.a))} → ${esc(nice(m.b))}</dt><dd>${fmt(m.d)} mm</dd>
      <dt>Forward</dt><dd>${fmt(m.dx)} mm</dd><dt>Right</dt><dd>${fmt(m.dy)} mm</dd><dt>Up</dt><dd>${fmt(m.dz)} mm</dd></dl></section>`;
  }

  html += `<section class="sec"><h3>Rotor layout</h3>
    <p class="lead">Place rotors exactly by wheelbase — the diagonal motor-to-motor distance a frame is sold by.</p>
    <div class="row3">
      <label class="lab"><span>Layout</span><select id="lay-kind">${v.presets.map((k) => `<option>${k}</option>`).join("")}</select></label>
      <label class="lab"><span>Wheelbase mm</span><input id="lay-wb" type="number" value="${Math.round(v.wheelbase_mm)}" min="20" step="5"></label>
      <label class="lab"><span>Height mm</span><input id="lay-h" type="number" value="${Math.round(v.parts.find((p) => p.kind === "rotor")?.pos[2] ?? 30)}" step="1"></label>
    </div><div class="actions"><button data-act="layout">Place rotors</button></div></section>`;

  html += `<section class="sec"><h3>Add a part</h3>
    <div class="row2"><label class="lab"><span>Kind</span><select id="add-kind"><option value="body">Payload</option><option value="battery">Battery</option><option value="camera">Camera</option></select></label>
    <label class="lab"><span>Name</span><input id="add-name" type="text" placeholder="gps"></label></div>
    <div class="row3"><label class="lab"><span>Length mm</span><input id="add-l" type="number" value="30"></label>
      <label class="lab"><span>Width mm</span><input id="add-w" type="number" value="30"></label>
      <label class="lab"><span>Height mm</span><input id="add-hh" type="number" value="10"></label></div>
    <div class="row2"><label class="lab"><span>Mass g</span><input id="add-m" type="number" value="20"></label>
      <div class="lab"><span>&nbsp;</span><button data-act="add">Add on the centreline</button></div></div></section>`;

  html += `<section class="sec"><h3>Parts</h3><div class="list">${v.parts.map((p) => `
    <button data-sel="${esc(p.name)}" class="${p.name === S.sel ? "on" : ""}">${esc(nice(p.name))}<span>${fmt(p.mass_g, 1)} g</span></button>`).join("")}</div></section>`;
  return html;
}

function partSection(p) {
  const twin = p.twin.xz;
  const rotor = p.kind === "rotor";
  const box = p.shape?.type === "box";
  return `<section class="sec"><h3>${esc(nice(p.name))} <span class="sub">${p.kind === "body" ? "payload" : p.kind}</span></h3>
    ${p.editable ? `
    <div class="row3">
      <label class="lab"><span>Forward mm</span><input type="number" step="${vp.snap || 1}" data-pos="0" value="${fmt(p.pos[0], 2)}"></label>
      <label class="lab"><span>Right mm</span><input type="number" step="${vp.snap || 1}" data-pos="1" value="${fmt(p.pos[1], 2)}"></label>
      <label class="lab"><span>Up mm</span><input type="number" step="${vp.snap || 1}" data-pos="2" value="${fmt(p.pos[2], 2)}"></label>
    </div>
    <p class="hint">${twin ? `Mirrored by <b>${esc(nice(twin))}</b>${vp.link !== "off" ? " — it moves with this one." : " — mirroring is off."}`
      : p.centred.xz ? "On the centreline." : "No left–right twin."}</p>` : `<p class="hint">The airframe is placed by the Structure step.</p>`}
    <div class="row2">
      <label class="lab"><span>Mass g</span><input type="number" data-field="mass_g" value="${p.mass_g}" step="1"></label>
      ${rotor ? `<label class="lab"><span>Prop Ø mm</span><input type="number" data-field="radius_mm" data-scale="0.5" value="${fmt(p.radius * 2, 1)}" step="1"></label>` : "<span></span>"}
    </div>
    ${box ? `<div class="row3">${["Length", "Width", "Height"].map((l, i) => `
      <label class="lab"><span>${l} mm</span><input type="number" data-size="${i}" value="${fmt(p.shape.size[i], 1)}" step="1"></label>`).join("")}</div>` : ""}
    ${rotor ? `<div class="lab"><span>Spin, seen from above</span>
      <div class="seg full"><button data-spin="ccw" class="${p.spin === "ccw" ? "on" : ""}">Anticlockwise</button><button data-spin="cw" class="${p.spin === "cw" ? "on" : ""}">Clockwise</button></div></div>
      <label class="lab"><span>Max thrust N</span><input type="number" data-field="thrust_n" value="${p.thrust_n ?? ""}" step="0.1"></label>` : ""}
    ${p.editable ? `<div class="actions"><button data-act="remove" class="ghost">Remove</button></div>` : ""}
  </section>`;
}

function optimise() {
  const j = latest("optimise");
  const running = j?.status === "running";
  const est = Math.max(1, Math.round(S.opt.population * S.opt.generations * 0.004 / 60));
  let html = `<section class="sec"><h3>Optimise the layout</h3>
    <p class="lead">Searches payload positions, arm length, prop size and battery for the ${esc(nice(S.v.mission))} mission, and returns the whole trade-off rather than one answer.</p>
    <div class="row2"><label class="lab"><span>Designs per generation</span><input type="number" id="o-pop" value="${S.opt.population}" min="8" step="4"></label>
    <label class="lab"><span>Generations</span><input type="number" id="o-gen" value="${S.opt.generations}" min="2"></label></div>
    <p class="hint">About ${est} min.</p>
    <div class="actions"><button class="primary" data-act="run-opt" ${running ? "disabled" : ""}>${running ? "Running…" : "Run"}</button></div></section>`;
  if (j?.status === "done") html += `<section class="sec" id="front-sec"><h3>Trade-off <span class="sub">click a design to use it</span></h3><div id="front">Loading…</div></section>`;
  else if (j?.status === "failed") html += failed(j);
  return html;
}

function structure() {
  const g = S.gen;
  const j = latest("generate");
  const running = j?.status === "running";
  const Q = { draft: ["Draft", "4 mm", "about 3 min"], standard: ["Standard", "3 mm", "about 25 min"], fine: ["Fine", "1.5 mm", "about 2 hours"] };
  let html = `<section class="sec"><h3>Grow the frame</h3>
    <p class="lead">Topology optimisation: the frame is grown from where the motors pull and where the payload bolts on. Nothing is drawn by hand.</p>
    <div class="seg"><button data-gmode="price" class="${g.mode === "price" ? "on" : ""}">Let it choose the mass</button><button data-gmode="mass" class="${g.mode === "mass" ? "on" : ""}">Fixed mass</button></div>
    ${g.mode === "price" ? `<label class="lab"><span>Stiffer ← → Lighter</span>
      <input type="range" id="g-price" min="-1.3" max="1.3" step="0.05" value="${Math.log10(g.price)}"></label>
      <p class="hint">Carbon priced at <b class="num">${fmt(g.price, 2)}</b> against the aircraft's own mass. At 1 the racing quad chooses about 40 g; lower buys a stiffer, heavier frame.</p>`
    : `<label class="lab"><span>Frame mass g</span><input type="number" id="g-mass" value="${g.mass}" min="5"></label>`}
    <div class="row2"><label class="lab"><span>Thinnest member mm</span><input type="number" id="g-min" value="${g.min_feature}" min="1" step="0.5"></label>
    <label class="lab"><span>Quality</span><select id="g-q">${Object.entries(Q).map(([k, [n, e]]) => `<option value="${k}" ${k === g.quality ? "selected" : ""}>${n} (${e})</option>`).join("")}</select></label></div>
    <p class="hint">${Q[g.quality][2]}. Runs in the background; you can keep working.</p>
    <div class="actions"><button class="primary" data-act="run-gen" ${running ? "disabled" : ""}>${running ? "Growing…" : "Grow"}</button></div></section>`;
  if (j?.status === "done") {
    html += `<section class="sec"><h3>Result</h3><div id="gen-report">Loading…</div>
      <div class="actions"><button class="primary" data-act="apply" data-job="${j.id}">Use this frame</button>
      <button data-file="${esc(j.outputs.stl)}">Download STL</button></div></section>`;
  } else if (j?.status === "failed") html += failed(j);
  return html;
}

function aero() {
  const j = latest("aero");
  const running = j?.status === "running";
  let html = `<section class="sec"><h3>Test the aerodynamics</h3>
    <p class="lead">Runs the vehicle through OpenFOAM at the conditions it actually flies in — worked out from what it is and what it is for — with the rotors spinning at the thrust each condition needs.</p>
    <div id="env">Working out the conditions…</div>
    <div class="actions"><button class="primary" data-act="run-aero" ${running ? "disabled" : ""}>${running ? "Testing…" : "Run"}</button></div>
    <p class="hint">About two minutes per condition. Needs OpenFOAM.</p></section>`;
  if (j?.status === "done") html += `<section class="sec"><h3>Measured <span class="sub">airframe force vs angle</span></h3><div id="aero-chart">Loading…</div>
    <div class="actions"><button class="primary" data-act="apply" data-job="${j.id}" ${S.v.aero_table === j.outputs.table ? "disabled" : ""}>${S.v.aero_table === j.outputs.table ? "Flying with this" : "Fly with this"}</button></div></section>`;
  else if (j?.status === "failed") html += failed(j);
  return html;
}

function fly() {
  const rec = latest("record");
  const flying = !!S.live;
  const pilots = [{ id: "manual", name: "You", about: "Fly it yourself, keyboard or gamepad" },
                  ...S.pilots];
  let html = `<section class="sec"><h3>Who flies</h3><div class="cards">${pilots.map((p) => `
    <button class="card ${p.id === S.pilot ? "on" : ""}" data-pilot="${esc(p.id)}" ${flying ? "disabled" : ""}><b>${esc(p.name)}</b><small>${esc(p.about)}</small></button>`).join("")}</div>
    ${S.pilot === "manual" ? "" : `<div class="actions"><button data-act="new-pilot" ${flying ? "disabled" : ""}>Write a new pilot…</button><button class="ghost" data-open="pilots">Open pilots folder</button></div>`}</section>`;

  if (S.pilot === "manual") {
    html += `<section class="sec"><h3>Mode</h3>
      <div class="seg wide" id="fly-mode">${[["assist", "Assist"], ["angle", "Angle"], ["acro", "Acro"]].map(([m, label]) =>
        `<button data-mode="${m}" class="${S.flyMode === m ? "on" : ""}">${label}</button>`).join("")}</div>
      <p class="hint">${{
        assist: "Sticks ask for a speed; let go and it stops and holds its height. Start here.",
        angle: "Sticks ask for a bank angle, throttle is yours. Let go and it levels.",
        acro: "Sticks ask for rotation rates. Nothing levels it — this is how racers fly, and the only mode that can fly inverted.",
      }[S.flyMode]}</p>
      <table class="keys"><tbody>
        <tr><td><kbd>W</kbd><kbd>S</kbd></td><td>pitch — forward, back</td></tr>
        <tr><td><kbd>A</kbd><kbd>D</kbd></td><td>roll — left, right</td></tr>
        <tr><td><kbd>Q</kbd><kbd>E</kbd></td><td>yaw — turn left, right</td></tr>
        <tr><td><kbd>Space</kbd><kbd>Z</kbd></td><td>throttle up, down</td></tr>
        <tr><td><kbd>R</kbd></td><td>reset where you started</td></tr>
        <tr><td><kbd>G</kbd></td><td>arm / disarm the motors</td></tr>
        <tr><td><kbd>1</kbd><kbd>2</kbd><kbd>3</kbd></td><td>chase, FPV, orbit view</td></tr>
      </tbody></table>
      <p class="hint">A gamepad works too — plug it in and move a stick. Mode 2: left is throttle and yaw, right is pitch and roll.</p></section>`;
  } else if (S.pilot === "gates") {
    html += `<section class="sec"><h3>Flight</h3><label class="lab"><span>Target speed m/s</span>
      <input type="number" id="f-speed" value="${S.speed ?? 14}" min="1" max="40" ${flying ? "disabled" : ""}></label></section>`;
  }

  html += `<section class="sec"><h3>Air</h3>
    <div class="seg"><button data-aero="0" class="${!S.flyAero ? "on" : ""}" ${flying ? "disabled" : ""}>Estimated drag</button><button data-aero="1" class="${S.flyAero ? "on" : ""}" ${S.v.aero_table && !flying ? "" : "disabled"}>CFD aerodynamics</button></div>
    ${S.v.aero_table ? "" : `<p class="hint">Run step 4 to measure this airframe's real aerodynamics.</p>`}
    <div class="actions">${flying
      ? `<button class="primary" data-act="land">Land and stop</button><button data-act="reset-live">Reset</button>`
      : `<button class="primary" data-act="fly-here">Fly it here</button><button data-act="record">Record a video</button>`}</div>
    ${flying ? `<p class="hint">Click the view first, then fly. <kbd>Esc</kbd> stops.</p>`
             : `<p class="hint">It flies right here, in real time. <button class="linky" data-act="fly-live">Open the physics engine window</button> instead for MuJoCo's own view.</p>`}</section>`;

  if (rec?.status === "done") html += `<section class="sec"><h3>Last recording</h3>
    <video controls src="${P()}/file?path=${encodeURIComponent(rec.outputs.video)}&t=${encodeURIComponent(TOKEN)}"></video></section>`;
  else if (rec?.status === "failed") html += failed(rec);
  html += `<section class="sec"><details class="more"><summary>Program it over ROS 2</summary>
    <p class="hint" style="margin-top:8px">The same simulator, as a ROS 2 node. Publishes <code>/camber/odom</code>, <code>/camber/imu</code>; accepts motors, a wrench, a pose or a trajectory setpoint.</p>
    <p class="hint">From a clone of the repository, with <a href="https://pixi.sh" target="_blank" rel="noopener">pixi</a>:</p>
    <pre class="log">pixi run -m ros/pixi.toml fly</pre><p class="hint"><a href="https://github.com/Abhijit-s999/camber/blob/main/docs/FLYING.md#3-program-it-over-ros-2" target="_blank" rel="noopener">The topics, and how to run your own node</a>.</p></details></section>`;
  return html;
}

async function takeOff() {
  live.vp = vp;
  await live.start({ pilot: S.pilot, mode: S.flyMode, aero: S.flyAero,
                     speed: S.speed ?? 14, course: true });
  S.live = true;
  $("#vtools").hidden = true;
  $(".axes").hidden = true;                 // a layout legend, not a flight one
  $("#view").focus();
  panel();
}

async function land() {
  if (!S.live) return;
  S.live = null;
  await live.stop();
  $("#vtools").hidden = false;
  $(".axes").hidden = false;
  panel();
}

function failed(j) {
  return `<section class="sec"><h3>Last run failed</h3><p class="bad">${esc(j.detail)}</p>
    <details class="more"><summary>Log</summary><pre class="log" data-log="${j.id}">…</pre></details></section>`;
}

// ---- panel events -------------------------------------------------------------
function bindPanel(el) {
  el.onclick = async (e) => {
    const t = e.target.closest("button, [data-sel]");
    if (!t) return;
    if (t.dataset.sel) { vp.select(t.dataset.sel); return; }
    if (t.dataset.spin) return edit("update", { name: S.sel, fields: { spin: t.dataset.spin } })();
    if (t.dataset.gmode) { S.gen.mode = t.dataset.gmode; return panel(); }
    if (t.dataset.pilot) { S.pilot = t.dataset.pilot; return panel(); }
    if (t.dataset.aero) { S.flyAero = t.dataset.aero === "1"; return panel(); }
    if (t.dataset.mode) {
      S.flyMode = t.dataset.mode;
      if (S.live) await live.setMode(S.flyMode);
      return panel();
    }
    if (t.dataset.file) return download(`${P()}/file?path=${encodeURIComponent(t.dataset.file)}&download=1&t=${encodeURIComponent(TOKEN)}`);
    if (t.dataset.open) return api("POST", `${P()}/open`, { path: t.dataset.open }).catch((x) => toast(x.message, true));
    const act = t.dataset.act;
    if (!act) return;
    const run = async (kind, params) => { await api("POST", `${P()}/jobs`, { kind, params }); await pollJobs(); panel(); };
    const actions = {
      "sym-lr": () => edit("symmetrise", { link: "lr" }, "Made left–right symmetric")(),
      "sym-4": () => edit("symmetrise", { link: "4way" }, "Made 4-way symmetric")(),
      // Lazy, all of them: this object is rebuilt on every click in every
      // panel, so an action that reads a field must not read it until it is
      // the action being run. Eagerly, `$("#lay-kind").value` threw on every
      // click outside the design panel and swallowed the click with it.
      layout: () => edit("layout", { kind: $("#lay-kind").value, wheelbase_mm: +$("#lay-wb").value,
                                     height_mm: +$("#lay-h").value }, "Rotors placed")(),
      add: () => edit("add", { kind: $("#add-kind")?.value, name: $("#add-name")?.value, mass_g: +$("#add-m")?.value,
                               size_mm: [+$("#add-l")?.value, +$("#add-w")?.value, +$("#add-hh")?.value], pos: [0, 0, 20] })(),
      remove: () => edit("remove", { name: S.sel })(),
      "run-opt": guard(() => run("optimise", { population: +$("#o-pop").value, generations: +$("#o-gen").value })),
      "run-gen": guard(() => {
        const q = { draft: [0.004, 40, 0.016], standard: [0.003, 80, 0.012], fine: [0.0015, 110, 0.012] }[S.gen.quality];
        return run("generate", { mode: S.gen.mode, price: S.gen.price, mass: S.gen.mass, min_feature: S.gen.min_feature,
                                 element: q[0], iterations: q[1], thickness: q[2] });
      }),
      "run-aero": guard(() => run("aero", { n: 6 })),
      apply: () => edit("apply", { job: t.dataset.job }, "Applied — undo to go back")(),
      "fly-here": guard(takeOff),
      land: guard(land),
      "reset-live": guard(() => live.send({ do: "reset" })),
      "fly-live": guard(async () => {
        await api("POST", `${P()}/fly`, { pilot: S.pilot, speed: S.speed ?? 14, aero: S.flyAero });
        toast("Opening the physics engine…");
      }),
      record: guard(() => run("record", { pilot: S.pilot === "gates" || S.pilot === "hover" ? S.pilot : S.pilot,
                                          speed: S.speed ?? 14, duration: 20,
                                          aero: S.flyAero ? S.v.aero_table : "" })),
      "new-pilot": guard(async () => {
        const name = prompt("Name your pilot", "My pilot");
        if (!name) return;
        const r = await api("POST", `${P()}/pilot`, { name });
        S.pilots = (await api("GET", `${P()}/pilots`)).pilots;
        S.pilot = r.id;
        panel();
        toast("Opened in your editor");
      }),
    };
    actions[act]?.();
  };
  el.onchange = (e) => {
    const t = e.target;
    if (t.dataset.pos !== undefined) {
      const p = S.v.parts.find((x) => x.name === S.sel);
      const pos = [...p.pos];
      pos[+t.dataset.pos] = +t.value;
      edit("move", { name: S.sel, pos, link: vp.link })();
    } else if (t.dataset.field) {
      const val = +t.value * (+t.dataset.scale || 1);
      edit("update", { name: S.sel, fields: { [t.dataset.field]: val } })();
    } else if (t.dataset.size !== undefined) {
      const p = S.v.parts.find((x) => x.name === S.sel);
      const size = [...p.shape.size];
      size[+t.dataset.size] = +t.value;
      edit("update", { name: S.sel, fields: { size_mm: size } })();
    } else if (t.id === "g-q") { S.gen.quality = t.value; panel(); }
    else if (t.id === "g-mass") S.gen.mass = +t.value;
    else if (t.id === "g-min") S.gen.min_feature = +t.value;
    else if (t.id === "o-pop") { S.opt.population = +t.value; panel(); }
    else if (t.id === "o-gen") { S.opt.generations = +t.value; panel(); }
    else if (t.id === "f-speed") S.speed = +t.value;
  };
  el.oninput = (e) => {
    if (e.target.id === "g-price") {
      S.gen.price = +(10 ** +e.target.value).toFixed(3);
      const b = el.querySelector("#g-price")?.closest(".sec")?.querySelector("b.num");
      if (b) b.textContent = fmt(S.gen.price, 2);
    }
  };
  // Things that load after the panel draws.
  const logEl = el.querySelector("[data-log]");
  if (logEl) api("GET", `${P()}/jobs/${logEl.dataset.log}/log`).then((r) => (logEl.textContent = r.log)).catch(() => {});
  if (S.step === "optimise" && $("#front", el)) drawFront(latest("optimise"));
  if (S.step === "structure" && $("#gen-report", el)) drawGenReport(latest("generate"));
  if (S.step === "aero") { drawEnvelope(); if ($("#aero-chart", el)) drawAero(latest("aero")); }
}

function download(url) {
  const a = document.createElement("a");
  a.href = url;
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// ---- results -------------------------------------------------------------------
function scatter(pts, xl, yl, onPick) {
  const W = 300, H = 210, m = { l: 44, r: 10, t: 10, b: 32 };
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const sx = (x) => m.l + ((x - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const sy = (y) => H - m.b - ((y - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const ticks = (a, b) => [a, (a + b) / 2, b];
  const svg = `<svg class="chart" viewBox="0 0 ${W} ${H}">
    ${ticks(y0, y1).map((y) => `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${sy(y)}" y2="${sy(y)}"/><text x="${m.l - 4}" y="${sy(y) + 3}" text-anchor="end">${fmt(y, y1 - y0 < 5 ? 1 : 0)}</text>`).join("")}
    ${ticks(x0, x1).map((x) => `<text x="${sx(x)}" y="${H - m.b + 14}" text-anchor="middle">${fmt(x, x1 - x0 < 5 ? 1 : 0)}</text>`).join("")}
    <text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle">${esc(xl)}</text>
    <text x="10" y="${(m.t + H - m.b) / 2}" text-anchor="middle" transform="rotate(-90 10 ${(m.t + H - m.b) / 2})">${esc(yl)}</text>
    ${pts.map((p, i) => `<circle data-i="${i}" cx="${sx(p[0])}" cy="${sy(p[1])}" r="4.5" fill="#8c9298" style="cursor:pointer"/>`).join("")}
  </svg>`;
  return svg;
}

async function drawFront(j) {
  const box = $("#front");
  const f = await api("GET", `${P()}/front?job=${j.id}`);
  // Fronts from before UI-08 carry no model name: they were all momentum theory.
  const rough = f.endurance_model !== "blade-element";
  const lab = { "-endurance_s": [rough ? "Endurance min (rough)" : "Endurance min", (v) => -v / 60], "mass_kg": ["Mass g", (v) => v * 1000],
                "-thrust_margin": ["Thrust : weight", (v) => -v] };
  const [a, b] = f.objective_names;
  const pts = f.points.map((p) => [lab[a]?.[1](p.objectives[0]) ?? p.objectives[0], lab[b]?.[1](p.objectives[1]) ?? p.objectives[1]]);
  box.innerHTML = scatter(pts, lab[a]?.[0] ?? a, lab[b]?.[0] ?? b)
    + (f.objective_names.includes("-endurance_s") ? (rough
      ? `<div class="hint">Endurance on this chart is a momentum-theory estimate with no motor or ESC losses. Checked against the blade-element model it read 16.5× too long (OPT-03), so treat it as a comparison, not a flight time. Fetch the propeller data with <code>camber catalogue fetch</code> and run again for the validated model.</div>`
      : `<div class="hint">Hover endurance from the blade-element model (${esc(f.endurance_detail)}), validated against 115 UIUC wind-tunnel propellers. One propeller is scaled across sizes and the motor is a default until components are chosen, so read it as a fair comparison and a rough flight time.</div>`) : "")
    + `<div id="front-pick" class="hint">${f.points.length} designs, none better than another on both at once.</div>`;
  box.querySelectorAll("circle").forEach((c) => {
    c.onclick = () => {
      const i = +c.dataset.i, p = f.points[i];
      box.querySelectorAll("circle").forEach((x) => x.setAttribute("fill", "#8c9298"));
      c.setAttribute("fill", "#ff5a1f");
      const vars = Object.entries(p.variables).filter(([k]) => /length|radius|capacity/.test(k));
      $("#front-pick").innerHTML = `<dl class="kv">${vars.map(([k, v]) => `<dt>${esc(nice(k))}</dt><dd>${fmt(v, 3)}</dd>`).join("")}</dl>
        <div class="actions" style="margin-top:8px"><button class="primary" id="use-pt">Use this design</button></div>`;
      $("#use-pt").onclick = edit("apply", { job: j.id, index: i }, "Design applied — undo to go back");
    };
  });
}

async function drawGenReport(j) {
  const r = await api("GET", `${P()}/jobs/${j.id}/log`);
  const grab = (re) => (r.log.match(re) || [])[1];
  const chose = grab(/it chose\s+([\d.]+) g/), defl = grab(/deflection\s+([\d.]+) mm/),
        thin = grab(/thin\s+([\d.]+)%/), mass = grab(/= ([\d.]+) g\n/), tw = r.log.match(/T:W\s+([\d.]+)\s+([\d.]+)/);
  $("#gen-report").innerHTML = `<dl class="kv">
    ${chose ? `<dt>Material it chose</dt><dd>${chose} g</dd>` : ""}
    ${mass ? `<dt>Printed part</dt><dd>${mass} g</dd>` : ""}
    ${defl ? `<dt>Flex at the motors</dt><dd>${defl} mm</dd>` : ""}
    ${thin ? `<dt>Thinner than the minimum</dt><dd>${thin}%</dd>` : ""}
    ${tw ? `<dt>Thrust : weight</dt><dd>${tw[1]} → ${tw[2]}</dd>` : ""}</dl>
    <details class="more"><summary>Full report</summary><pre class="log">${esc(r.log.split("\n").filter((l) => !/^\s+it\s/.test(l)).join("\n"))}</pre></details>`;
}

async function drawEnvelope() {
  const box = $("#env");
  if (!box) return;
  try {
    const e = S.envelope?.pid === S.pid ? S.envelope : { pid: S.pid, ...(await api("GET", `${P()}/envelope`)) };
    S.envelope = e;
    box.innerHTML = `<table class="t"><tr><th>#</th><th>Tilt</th><th>Speed</th><th>Thrust</th></tr>
      ${e.points.map((p, i) => `<tr><td>${i + 1}</td><td>${fmt(-p.alpha_deg, 0)}°</td><td>${fmt(p.speed)} m/s</td><td>${fmt(p.thrust_n, 2)} N</td></tr>`).join("")}</table>
      <p class="hint">${e.kind === "multirotor" ? "Level flight at each tilt: the tilt sets the speed, because thrust must hold the weight up and balance the drag." : "An angle-of-attack sweep through stall."}</p>`;
  } catch (x) { box.innerHTML = `<p class="bad">${esc(x.message)}</p>`; }
}

async function drawAero(j) {
  const t = await api("GET", `${P()}/json?path=${encodeURIComponent(j.outputs.table)}`);
  const W = 300, H = 190, m = { l: 36, r: 10, t: 10, b: 30 };
  const a = t.alpha_deg, series = [["CX", t.CX, "#eceeef"], ["CZ", t.CZ, "#ff5a1f"], ["Cm", t.Cm, "#8c9298"]];
  const all = series.flatMap((s) => s[1]);
  const [y0, y1] = [Math.min(...all, 0), Math.max(...all, 0)], [x0, x1] = [Math.min(...a), Math.max(...a)];
  const sx = (x) => m.l + ((x - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const sy = (y) => H - m.b - ((y - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  $("#aero-chart").innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}">
    <line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${sy(0)}" y2="${sy(0)}"/>
    ${[y0, y1].map((y) => `<text x="${m.l - 4}" y="${sy(y) + 3}" text-anchor="end">${fmt(y, 2)}</text>`).join("")}
    ${[x0, x1].map((x) => `<text x="${sx(x)}" y="${H - m.b + 13}" text-anchor="middle">${fmt(x, 0)}°</text>`).join("")}
    <text x="${(m.l + W - m.r) / 2}" y="${H - 3}" text-anchor="middle">angle of attack (nose down is negative)</text>
    ${series.map(([n, ys, c]) => `<polyline fill="none" stroke="${c}" stroke-width="1.6" points="${a.map((x, i) => `${sx(x)},${sy(ys[i])}`).join(" ")}"/>
      <text x="${sx(a[a.length - 1]) - 2}" y="${sy(ys[ys.length - 1]) - 5}" text-anchor="end" style="fill:${c}">${n}</text>`).join("")}
  </svg><p class="hint">CX drag (negative is backwards), CZ up-force, Cm pitching moment — on the airframe only; the rotors' thrust is applied separately.</p>`;
}

// ---- jobs -------------------------------------------------------------------------
let pollTimer;
async function pollJobs() {
  clearTimeout(pollTimer);
  if (!S.pid) return;
  try {
    const before = JSON.stringify(S.jobs.map((j) => [j.id, j.status]));
    S.jobs = (await api("GET", `${P()}/jobs`)).jobs;
    drawJobs();
    rail();
    const after = JSON.stringify(S.jobs.map((j) => [j.id, j.status]));
    if (before !== after && !S.dragging) {
      const finished = S.jobs.find((j) => j.status === "done" && !before.includes(`"${j.id}","done"`) && before.includes(j.id));
      if (finished) toast(`${finished.label} finished`);
      panel();
    }
  } catch (e) { /* the server may be restarting */ }
  pollTimer = setTimeout(pollJobs, S.jobs.some((j) => j.status === "running") ? 1000 : 5000);
}

function drawJobs() {
  const el = $("#jobs");
  const show = S.jobs.filter((j) => j.status === "running" || Date.now() / 1000 - (j.finished || 0) < 3600).slice(0, 5);
  el.hidden = !show.length;
  const t = (s) => (s < 60 ? `${Math.round(s)} s` : s < 3600 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);
  el.innerHTML = show.map((j) => `<div class="job ${j.status}">
    <b>${esc(j.label)}</b><span class="t">${t(j.elapsed)}</span>
    ${j.status === "running" ? `<button data-cancel="${j.id}">Stop</button>` : `<span class="t">${esc(j.status)}</span>`}
    <span class="d">${esc(j.detail || "")}</span>
    <span class="p"><b style="width:${Math.round((j.progress ?? (j.status === "done" ? 1 : 0.02)) * 100)}%"></b></span></div>`).join("");
  el.onclick = guard(async (e) => {
    const b = e.target.closest("[data-cancel]");
    if (!b) return;
    await api("POST", `${P()}/jobs/${b.dataset.cancel}/cancel`, {});
    pollJobs();
  });
}

// ---- chrome -----------------------------------------------------------------------
$("#to-home").onclick = () => { land(); home(); };
$("#undo").onclick = edit("undo", {});
$("#redo").onclick = edit("redo", {});
$("#steps").onclick = (e) => {
  const b = e.target.closest("[data-step]");
  if (!b) return;
  if (S.live && b.dataset.step !== "fly") land();
  S.step = b.dataset.step;
  rail();
  panel();
};
$("#views").onclick = (e) => {
  const b = e.target.closest("[data-view]");
  if (!b) return;
  $$("#views button").forEach((x) => x.classList.toggle("on", x === b));
  vp.setView(b.dataset.view);
};
$("#tool").onclick = (e) => {
  const b = e.target.closest("[data-tool]");
  if (!b) return;
  $$("#tool button").forEach((x) => x.classList.toggle("on", x === b));
  vp.setTool(b.dataset.tool);
  S.measure = null;
  panel();
};
$("#link").onclick = (e) => {
  const b = e.target.closest("[data-link]");
  if (!b) return;
  $$("#link button").forEach((x) => x.classList.toggle("on", x === b));
  vp.setLink(b.dataset.link);
  panel();
};
$("#snap").onchange = (e) => vp.setSnap(+e.target.value);
$("#export-btn").onclick = () => ($("#export-menu").hidden = !$("#export-menu").hidden);
$("#export-menu").onclick = (e) => {
  const a = e.target.closest("a");
  if (!a) return;
  $("#export-menu").hidden = true;
  if (a.dataset.export) download(`${P()}/export?what=${a.dataset.export}&t=${encodeURIComponent(TOKEN)}`);
  if (a.dataset.open) api("POST", `${P()}/open`, { path: a.dataset.open }).catch((x) => toast(x.message, true));
};
document.addEventListener("click", (e) => { if (!e.target.closest(".menu")) $("#export-menu").hidden = true; });
document.addEventListener("keydown", (e) => {
  if (!S.pid || e.target.matches("input, select, textarea")) return;
  if (S.live) return;               // the sticks have the keyboard
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); (e.shiftKey ? edit("redo", {}) : edit("undo", {}))(); }
  else if ((e.ctrlKey || e.metaKey) && k === "y") { e.preventDefault(); edit("redo", {})(); }
  else if (k === "escape") { vp.measure = []; vp._drawMeasure(); vp.select(null); }
  else if (k === "m") $('#tool [data-tool="measure"]').click();
  else if (k === "v") $('#tool [data-tool="move"]').click();
  else if ("1234".includes(k) && k) $$("#views button")[+k - 1]?.click();
  else if ((k === "delete" || k === "backspace") && S.sel) {
    const p = S.v.parts.find((x) => x.name === S.sel);
    if (p?.editable && confirm(`Remove ${nice(S.sel)}?`)) edit("remove", { name: S.sel })();
  }
});

// ---- start ------------------------------------------------------------------------
if (!TOKEN) {
  document.body.innerHTML = `<main class="home"><p class="lead">Open Camber from its launcher, or with <code>camber app</code>. This page needs the key the app gives it.</p></main>`;
} else if (params.get("p")) {
  open(params.get("p")).catch((e) => { toast(e.message, true); home(); });
} else {
  home().catch((e) => toast(e.message, true));
}
