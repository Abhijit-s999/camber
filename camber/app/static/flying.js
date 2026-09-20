// Flying it yourself: sticks from the keyboard or a gamepad, telemetry back,
// once per animation frame. One round trip carries both, so what you see and
// what you are steering can never drift apart.
//
// The simulation runs in the app, paced to the wall clock -- this file does no
// physics. If a round trip is still in the air when the next frame comes, that
// frame's input is skipped rather than queued: stale sticks are worse than
// slightly coarse ones.

const DEAD = 0.08;                 // gamepad deadzone
const RAMP = 4.0;                  // keyboard stick travel per second
const THR = 0.9;                   // keyboard throttle travel per second

const KEYS = {
  KeyW: ["pitch", 1], KeyS: ["pitch", -1], ArrowUp: ["pitch", 1], ArrowDown: ["pitch", -1],
  KeyA: ["roll", -1], KeyD: ["roll", 1], ArrowLeft: ["roll", -1], ArrowRight: ["roll", 1],
  KeyQ: ["yaw", -1], KeyE: ["yaw", 1],
  ShiftLeft: ["throttle", 1], Space: ["throttle", 1],
  ControlLeft: ["throttle", -1], KeyZ: ["throttle", -1],
};

export class Live {
  constructor(vp, api, hud, on = {}) {
    this.vp = vp;
    this.api = api;
    this.hud = hud;
    this.on = on;
    this.sticks = { roll: 0, pitch: 0, yaw: 0, throttle: 0.5 };
    this.held = new Set();
    this.on_ = false;
    this.pad = null;
    this._key = (e) => this._onKey(e, true);
    this._keyUp = (e) => this._onKey(e, false);
    this._blur = () => this.held.clear();
  }

  get manual() { return this.pilot === "manual"; }

  async start(opts) {
    const r = await this.api("POST", "/live", { do: "start", ...opts });
    this.pilot = opts.pilot;
    this.mode = r.state.mode;
    this.scene = r.scene;
    this.state = r.state;
    this.on_ = true;
    this.sticks = { roll: 0, pitch: 0, yaw: 0, throttle: 0.5 };
    this.vp.enterFlight(r.scene);
    this.vp.telemetry(r.state);
    this.hud.hidden = false;
    this.hud.innerHTML = TEMPLATE;
    this._bindHud();
    addEventListener("keydown", this._key);
    addEventListener("keyup", this._keyUp);
    addEventListener("blur", this._blur);
    this.last = performance.now();
    this._loop();
    return r;
  }

  async stop() {
    if (!this.on_) return;
    this.on_ = false;
    removeEventListener("keydown", this._key);
    removeEventListener("keyup", this._keyUp);
    removeEventListener("blur", this._blur);
    this.hud.hidden = true;
    this.vp.exitFlight();
    try { await this.api("POST", "/live", { do: "stop" }); } catch (e) { /* already gone */ }
  }

  // ---- input ---------------------------------------------------------------
  _onKey(e, down) {
    if (e.repeat) return;
    if (down && (e.code === "KeyR" || e.code === "Backspace")) { this.send({ do: "reset" }); return e.preventDefault(); }
    if (down && e.code === "KeyG") { this.send({ do: "arm", on: !this.state?.armed }); return e.preventDefault(); }
    if (down && e.code === "Escape") { this.on.exit?.(); return; }
    if (down && ["Digit1", "Digit2", "Digit3"].includes(e.code)) {
      this.camera(["chase", "fpv", "orbit"][+e.code.slice(-1) - 1]);
      return e.preventDefault();
    }
    if (!KEYS[e.code]) return;
    e.preventDefault();
    if (down) this.held.add(e.code); else this.held.delete(e.code);
  }

  _gamepad() {
    const pads = navigator.getGamepads?.() || [];
    const p = [...pads].find((g) => g && g.connected);
    this.pad = p ? p.id : null;
    if (!p) return null;
    const ax = (i) => { const v = p.axes[i] ?? 0; return Math.abs(v) < DEAD ? 0 : v; };
    // Mode 2, as sold: left stick is throttle and yaw, right stick is the
    // attitude. Pitch is inverted -- pushing the stick forward reads negative.
    const pressed = p.buttons.some((b, i) => i < 2 && b.pressed);
    return {
      roll: ax(2), pitch: -ax(3), yaw: ax(0),
      throttle: (1 - (p.axes[1] ?? 0)) / 2,
      arm: p.buttons[8]?.pressed || p.buttons[9]?.pressed,
      reset: pressed && p.buttons[1]?.pressed,
    };
  }

  _input(dt) {
    const pad = this._gamepad();
    if (pad && (Math.abs(pad.roll) + Math.abs(pad.pitch) + Math.abs(pad.yaw) > 0
                || this._padThrottle !== undefined)) {
      this._padThrottle = pad.throttle;
      this.sticks = { roll: pad.roll, pitch: pad.pitch, yaw: pad.yaw, throttle: pad.throttle };
      if (pad.arm && !this._armHeld) this.send({ do: "arm", on: !this.state?.armed });
      this._armHeld = pad.arm;
      return;
    }
    const want = { roll: 0, pitch: 0, yaw: 0 };
    let thr = 0;
    for (const code of this.held) {
      const [axis, dir] = KEYS[code];
      if (axis === "throttle") thr += dir; else want[axis] += dir;
    }
    for (const axis of ["roll", "pitch", "yaw"]) {
      const target = Math.max(-1, Math.min(1, want[axis]));
      const step = RAMP * dt * (target === 0 ? 1.6 : 1);   // springs back faster
      this.sticks[axis] += Math.max(-step, Math.min(step, target - this.sticks[axis]));
    }
    if (thr) this.sticks.throttle = Math.max(0, Math.min(1, this.sticks.throttle + thr * THR * dt));
    else if (this.mode === "assist") {
      // Mid-stick holds height, so letting go should mean "stay there".
      const back = 1.2 * dt;
      this.sticks.throttle += Math.max(-back, Math.min(back, 0.5 - this.sticks.throttle));
    }
  }

  // ---- the loop ------------------------------------------------------------
  _loop() {
    if (!this.on_) return;
    requestAnimationFrame(() => this._loop());
    const now = performance.now();
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    if (this.manual) this._input(dt);
    if (!this._busy) {
      this._busy = true;
      const body = this.manual
        ? { do: "input", sticks: [this.sticks.roll, this.sticks.pitch, this.sticks.yaw, this.sticks.throttle] }
        : { do: "poll" };
      this.api("POST", "/live", body)
        .then((r) => { this.state = r.state; this.vp.telemetry(r.state); this._draw(r.state); })
        .catch((e) => { this.on.lost?.(e); this.on_ = false; })
        .finally(() => { this._busy = false; });
    }
  }

  send(body) {
    return this.api("POST", "/live", body)
      .then((r) => { if (r.state) { this.state = r.state; this.mode = r.state.mode; this._draw(r.state); } })
      .catch((e) => this.on.lost?.(e));
  }

  setMode(mode) { this.mode = mode; return this.send({ do: "mode", mode }); }

  camera(mode) {
    this.vp.setFlightCamera(mode);
    this.hud.querySelectorAll("[data-cam]").forEach((b) =>
      b.classList.toggle("on", b.dataset.cam === mode));
  }

  // ---- the head-up display -------------------------------------------------
  _bindHud() {
    this.el = {};
    for (const k of ["speed", "alt", "climb", "tilt", "thr", "mode", "gates", "lap",
                     "best", "rate", "armed", "crash", "pad", "bars", "stickL", "stickR"]) {
      this.el[k] = this.hud.querySelector(`[data-hud="${k}"]`);
    }
    this.hud.onclick = (e) => {
      const b = e.target.closest("[data-cam], [data-hud-act]");
      if (!b) return;
      if (b.dataset.cam) return this.camera(b.dataset.cam);
      if (b.dataset.hudAct === "reset") return this.send({ do: "reset" });
      if (b.dataset.hudAct === "arm") return this.send({ do: "arm", on: !this.state?.armed });
      if (b.dataset.hudAct === "stop") return this.on.exit?.();
    };
    this.camera("chase");
    this.hud.classList.toggle("manual", this.manual);
  }

  _draw(s) {
    const n = (x, d = 1) => (x == null ? "–" : x.toFixed(d));
    const e = this.el;
    e.speed.textContent = n(s.ground_speed);
    e.alt.textContent = n(s.altitude, 2);
    e.climb.textContent = (s.climb >= 0 ? "+" : "") + n(s.climb);
    e.tilt.textContent = n(s.tilt_deg, 0);
    e.thr.style.setProperty("--v", `${Math.round(100 * s.throttle)}%`);
    e.mode.textContent = s.mode ? s.mode : s.pilot;
    e.gates.textContent = s.gates_total ? `${s.gates_passed % s.gates_total || (s.gates_passed ? s.gates_total : 0)}/${s.gates_total}` : "–";
    e.lap.textContent = s.lap ? `${s.lap.toFixed(2)}s` : "–";
    e.best.textContent = s.best_lap ? `${s.best_lap.toFixed(2)}s` : "–";
    // Real time is the claim; headroom is what is left over. A flight that
    // cannot keep up says so rather than quietly running slow.
    e.rate.textContent = s.rate < 0.95 ? `${s.rate.toFixed(2)}× real time`
                                       : `real time · ${s.headroom.toFixed(1)}× spare`;
    e.rate.classList.toggle("slow", s.rate < 0.9);
    e.armed.textContent = s.armed ? "armed" : "disarmed";
    e.armed.classList.toggle("off", !s.armed);
    e.crash.hidden = !s.crashed;
    e.pad.textContent = this.pad ? "gamepad" : "keyboard";
    const max = Math.max(...(this.scene?.max_thrust || [1]));
    e.bars.querySelectorAll("i").forEach((bar, i) =>
      bar.style.setProperty("--v", `${Math.round(100 * Math.min(1, (s.thrust[i] || 0) / max))}%`));
    if (this.manual && s.sticks) {
      const [roll, pitch, yaw, thr] = s.sticks;
      e.stickL.style.setProperty("--x", `${50 + 45 * yaw}%`);
      e.stickL.style.setProperty("--y", `${100 - 95 * thr}%`);
      e.stickR.style.setProperty("--x", `${50 + 45 * roll}%`);
      e.stickR.style.setProperty("--y", `${50 - 45 * pitch}%`);
    }
  }
}

const TEMPLATE = `
<div class="hud-top">
  <span class="hud-chip" data-hud="mode">assist</span>
  <span class="hud-chip" data-hud="armed">armed</span>
  <span class="hud-chip dim" data-hud="pad">keyboard</span>
  <span class="hud-chip dim" data-hud="rate">real time</span>
  <span class="hud-grow"></span>
  <span class="hud-chip">Gates <b data-hud="gates">–</b></span>
  <span class="hud-chip">Lap <b data-hud="lap">–</b></span>
  <span class="hud-chip">Best <b data-hud="best">–</b></span>
  <span class="seg hud-seg">
    <button data-cam="chase" class="on">Chase</button><button data-cam="fpv">FPV</button><button data-cam="orbit">Orbit</button>
  </span>
  <button class="hud-btn" data-hud-act="stop">Land</button>
</div>
<div class="hud-mid">
  <div class="hud-side">
    <div class="hud-read"><b data-hud="speed">0.0</b><span>m/s</span></div>
    <div class="hud-read"><b data-hud="alt">0.00</b><span>m</span></div>
    <div class="hud-read"><b data-hud="climb">+0.0</b><span>m/s climb</span></div>
    <div class="hud-read"><b data-hud="tilt">0</b><span>° tilt</span></div>
    <div class="hud-bar" data-hud="thr"><span>throttle</span></div>
    <div class="hud-motors" data-hud="bars"><i></i><i></i><i></i><i></i></div>
  </div>
</div>
<div class="hud-sticks">
  <div class="hud-stick" data-hud="stickL"><span>throttle / yaw</span></div>
  <div class="hud-stick" data-hud="stickR"><span>pitch / roll</span></div>
</div>
<div class="hud-crash" data-hud="crash" hidden>Crashed — <button data-hud-act="reset">reset (R)</button></div>`;
