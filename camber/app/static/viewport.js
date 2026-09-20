// The layout editor's 3D view.
//
// Axes. The API speaks (forward, right, up) in millimetres. three.js is y-up with
// the camera looking down -z, so a part at UI (f, r, u) is drawn at three
// (r, u, -f). That is a proper rotation of Camber's NED frame, so orientations
// convert cleanly (the server sends them already converted), and forward points
// into the screen from the default camera -- you look at the vehicle from behind.

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { TransformControls } from "three/addons/TransformControls.js";
import { STLLoader } from "three/addons/STLLoader.js";

const C = {
  ground: 0x0b0c0d, fill: 0x5c646b, fillBattery: 0x7a8288, edge: 0xa9b1b7,
  accent: 0xff5a1f, warn: 0xe9b43c, bad: 0xff5252, frame: 0x2b2f33,
  gridMinor: 0x16191b, gridMajor: 0x23272b, centre: 0x4a5157, disc: 0xffffff,
  gate: 0x7c848a,
};

const M = 1000;            // scene units per metre

export const toThree = ([f, r, u]) => new THREE.Vector3(r, u, -f);
export const fromThree = (v) => [-v.z, v.x, v.y];

export class Viewport {
  constructor(canvas, labels, on) {
    this.canvas = canvas;
    this.labelsEl = labels;
    this.on = on;                               // {select, moved, measured, dragging}
    this.tool = "move";
    this.link = "lr";
    this.snap = 5;
    this.view = "3d";
    this.parts = new Map();                     // name -> {obj, data, edges}
    this.labels = [];
    this.measure = [];
    this.selected = null;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(C.ground);
    this.scene.add(new THREE.HemisphereLight(0xdfe6ec, 0x0b0c0d, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(300, 600, 400);
    this.scene.add(key);

    this.persp = new THREE.PerspectiveCamera(38, 1, 1, 20000);
    this.ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, -20000, 20000);
    this.camera = this.persp;
    this.controls = new OrbitControls(this.persp, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;

    this.gizmo = new TransformControls(this.persp, canvas);
    this.gizmo.setSize(0.8);
    this.gizmo.addEventListener("dragging-changed", (e) => {
      this.controls.enabled = !e.value;
      this.on.dragging?.(e.value);
      if (!e.value && this.selected) this._commitDrag();
    });
    this.gizmo.addEventListener("objectChange", () => this._liveMirror());
    this.scene.add(this.gizmo);

    this.world = new THREE.Group();
    this.scene.add(this.world);
    this.helpers = new THREE.Group();
    this.scene.add(this.helpers);

    this.ray = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this._down = null;
    canvas.addEventListener("pointerdown", (e) => { this._down = [e.clientX, e.clientY]; });
    canvas.addEventListener("pointerup", (e) => this._click(e));
    canvas.addEventListener("pointermove", (e) => this._hover(e));

    new ResizeObserver(() => this._resize()).observe(canvas.parentElement);
    this._resize();
    const loop = () => {
      requestAnimationFrame(loop);
      // OrbitControls.update() re-aims the camera at its own target even when
      // it is disabled, which silently undid the chase and FPV cameras every
      // frame: the vehicle sat 2.8 m away, just outside the frame, while the
      // camera looked at wherever the layout editor had last been pointed.
      if (!this.flying || this.camMode === "orbit") this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this._placeLabels();
    };
    loop();
  }

  // ---- scene --------------------------------------------------------------
  async setVehicle(v, frameMeshUrl, token) {
    this.v = v;
    const keep = this.selected;
    this.gizmo.detach();
    this.world.clear();
    this.parts.clear();

    for (const p of v.parts) {
      const obj = p.kind === "rotor" ? this._rotor(p) : this._body(p);
      if (!obj) continue;
      obj.userData.name = p.name;
      this.world.add(obj);
      this.parts.set(p.name, { obj, data: p });
    }
    if (v.has_generated_frame && frameMeshUrl) {
      try {
        const res = await fetch(frameMeshUrl, { headers: { "X-Camber-Token": token } });
        if (res.ok) {
          const geom = new STLLoader().parse(await res.arrayBuffer());
          geom.computeVertexNormals();
          const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial(
            { color: C.frame, metalness: 0.25, roughness: 0.55 }));
          const holder = this.parts.get(v.parts.find((p) => p.kind === "frame")?.name);
          if (holder) {
            holder.obj.clear();
            holder.obj.position.set(0, 0, 0);
            holder.obj.quaternion.identity();
            holder.obj.add(mesh);
          }
        }
      } catch (e) { /* the box stand-in stays */ }
    }
    this._decorate();
    this._helpers();
    if (!this._framed) { this.fit(); this._framed = true; }
    if (keep && this.parts.has(keep)) this.select(keep, false); else this.select(null, false);
    this._drawMeasure();
  }

  _body(p) {
    const s = p.shape;
    let geom;
    if (s.type === "box") geom = new THREE.BoxGeometry(s.size[1], s.size[2], s.size[0]);
    else if (s.type === "cylinder") geom = new THREE.CylinderGeometry(s.radius, s.radius, s.length, 32);
    else if (s.type === "sphere") geom = new THREE.SphereGeometry(s.radius, 24, 16);
    else if (s.type === "mesh") geom = new THREE.BoxGeometry(s.extents[1], s.extents[2], s.extents[0]);
    else return null;
    const mat = new THREE.MeshStandardMaterial({
      color: p.kind === "battery" ? C.fillBattery : C.fill, roughness: 0.85, metalness: 0.05,
      transparent: s.type === "mesh", opacity: s.type === "mesh" ? 0.25 : 1 });
    const mesh = new THREE.Mesh(geom, mat);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geom),
      new THREE.LineBasicMaterial({ color: C.edge }));
    const g = new THREE.Group();
    g.add(mesh, edges);
    g.position.copy(toThree(p.pos));
    g.quaternion.set(...p.quat);
    g.userData.edges = edges;
    if (p.boresight && p.fov_deg) this._fov(p, g);
    return g;
  }

  // A camera's field of view, as a short pyramid along its boresight. The
  // boresight arrives in world axes, so it is taken into the part's frame.
  _fov(p, g) {
    const dir = new THREE.Vector3(...p.boresight).normalize()
      .applyQuaternion(g.quaternion.clone().invert());
    const L = 90, half = THREE.MathUtils.degToRad(p.fov_deg / 2);
    const w = Math.tan(half) * L, h = w * 0.5625;              // 16:9
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, -1), dir);
    const pts = [[w, h], [-w, h], [-w, -h], [w, -h]].map(([x, y]) =>
      new THREE.Vector3(x, y, -L).applyQuaternion(q));
    const seg = [];
    for (let i = 0; i < 4; i++) seg.push(new THREE.Vector3(), pts[i], pts[i], pts[(i + 1) % 4]);
    const cone = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(seg),
      new THREE.LineBasicMaterial({ color: 0x6f777d, transparent: true, opacity: 0.8 }));
    cone.userData.designOnly = true;          // a layout aid, not part of the vehicle
    g.add(cone);
  }

  _rotor(p) {
    const g = new THREE.Group();
    const axis = new THREE.Vector3(p.axis[1], -p.axis[2], -p.axis[0]).normalize();
    g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
    g.position.copy(toThree(p.pos));
    const disc = new THREE.Mesh(new THREE.CircleGeometry(p.radius, 64),
      new THREE.MeshBasicMaterial({ color: C.disc, transparent: true, opacity: 0.05,
                                    side: THREE.DoubleSide, depthWrite: false }));
    disc.rotation.x = -Math.PI / 2;
    const ringPts = [];
    for (let i = 0; i <= 96; i++) {
      const a = (i / 96) * Math.PI * 2;
      ringPts.push(new THREE.Vector3(Math.cos(a) * p.radius, 0, Math.sin(a) * p.radius));
    }
    const ring = new THREE.Line(new THREE.BufferGeometry().setFromPoints(ringPts),
      new THREE.LineBasicMaterial({ color: C.edge }));
    // Spin arrow, seen from above: counter-clockwise runs from +x (right) to -z
    // (forward), so at the ring's right-hand point the tangent is -z for CCW.
    const dir = p.spin === "ccw" ? -1 : 1;
    const tip = new THREE.Mesh(new THREE.ConeGeometry(p.radius * 0.07, p.radius * 0.2, 12),
      new THREE.MeshBasicMaterial({ color: C.edge }));
    tip.position.set(p.radius, 0, 0);
    tip.rotation.x = dir * Math.PI / 2;
    const bell = new THREE.Mesh(new THREE.CylinderGeometry(p.radius * 0.18, p.radius * 0.18,
      p.radius * 0.2, 24), new THREE.MeshStandardMaterial({ color: 0x8a9298, roughness: 0.4 }));
    bell.position.y = -p.radius * 0.12;
    g.add(disc, ring, tip, bell);
    g.userData.edges = ring;
    g.userData.tip = tip;
    return g;
  }

  // Colour by meaning: selected (accent), off-symmetric (amber), clashing (red).
  _decorate() {
    const v = this.v;
    const off = new Set();
    for (const pl of this._planes()) {
      for (const r of v.symmetry[pl].rows) if (r.error_mm > 0.05) { off.add(r.a); if (r.b) off.add(r.b); }
    }
    const clash = new Set(v.clearance.conflicts.flatMap((c) => [c.a, c.b]));
    for (const [name, { obj }] of this.parts) {
      const e = obj.userData.edges;
      if (!e) continue;
      const c = name === this.selected ? C.accent : clash.has(name) ? C.bad
              : off.has(name) ? C.warn : C.edge;
      e.material.color.setHex(c);
      obj.userData.tip?.material.color.setHex(c);
    }
  }

  _planes() { return this.link === "4way" ? ["xz", "yz"] : ["xz"]; }

  _helpers() {
    this.helpers.clear();
    this.labels = [];
    const v = this.v;
    const box = new THREE.Box3().setFromObject(this.world);
    const floor = box.isEmpty() ? 0 : box.min.y - 4;
    const half = Math.max(150, Math.ceil(Math.max(Math.abs(box.min.x), Math.abs(box.max.x),
      Math.abs(box.min.z), Math.abs(box.max.z)) * 1.4 / 50) * 50);
    this.floor = floor;
    const lines = (step, color) => {
      const pts = [];
      for (let x = -half; x <= half + 1e-6; x += step) {
        pts.push(x, floor, -half, x, floor, half, -half, floor, x, half, floor, x);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
      return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color }));
    };
    this.helpers.add(lines(10, C.gridMinor), lines(50, C.gridMajor));
    // The centreline: what left/right symmetry is measured against.
    const cl = new THREE.BufferGeometry().setFromPoints(
      [new THREE.Vector3(0, floor + 0.2, -half), new THREE.Vector3(0, floor + 0.2, half)]);
    this.helpers.add(new THREE.Line(cl, new THREE.LineBasicMaterial({ color: C.centre })));
    this.labels.push({ at: new THREE.Vector3(0, floor, half), text: "centreline", cls: "dim" });
    if (this.link === "4way") {
      const lat = new THREE.BufferGeometry().setFromPoints(
        [new THREE.Vector3(-half, floor + 0.2, 0), new THREE.Vector3(half, floor + 0.2, 0)]);
      this.helpers.add(new THREE.Line(lat, new THREE.LineBasicMaterial({ color: C.centre })));
    }
    if (this.link !== "off") {
      const h = Math.max(80, box.max.y - floor + 40);
      const plane = new THREE.Mesh(new THREE.PlaneGeometry(half * 2, h),
        new THREE.MeshBasicMaterial({ color: C.accent, transparent: true, opacity: 0.018,
                                      side: THREE.DoubleSide, depthWrite: false }));
      plane.rotation.y = Math.PI / 2;
      plane.position.set(0, floor + h / 2, 0);
      this.helpers.add(plane);
    }
    // Centre of gravity, and the thrust centre it should sit over.
    const cg = toThree(v.cg);
    const cross = [];
    for (const d of [[8, 0, 0], [0, 8, 0], [0, 0, 8]]) {
      const a = new THREE.Vector3(...d);
      cross.push(cg.clone().sub(a), cg.clone().add(a));
    }
    this.helpers.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(cross),
      new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false })));
    const drop = new THREE.BufferGeometry().setFromPoints([cg, new THREE.Vector3(cg.x, floor, cg.z)]);
    this.helpers.add(new THREE.Line(drop, new THREE.LineDashedMaterial(
      { color: 0x6a7278, dashSize: 3, gapSize: 3 })).computeLineDistances());
    this.labels.push({ at: cg, text: "CG", cls: "" });
    const tc = v.balance.thrust_centroid;
    if (tc) {
      const t = toThree(tc);
      t.y = floor;
      const ring = new THREE.Mesh(new THREE.RingGeometry(5, 6.5, 32),
        new THREE.MeshBasicMaterial({ color: 0x9aa1a7, side: THREE.DoubleSide }));
      ring.rotation.x = -Math.PI / 2;
      ring.position.copy(t).setY(floor + 0.3);
      this.helpers.add(ring);
      const off = Math.hypot(...v.balance.cg_offset_mm);
      if (off > 0.2) {
        const seg = new THREE.BufferGeometry().setFromPoints(
          [new THREE.Vector3(cg.x, floor + 0.3, cg.z), t.clone().setY(floor + 0.3)]);
        this.helpers.add(new THREE.Line(seg, new THREE.LineBasicMaterial({ color: C.warn })));
      }
    }
  }

  // ---- interaction ------------------------------------------------------------
  _pick(e) {
    const r = this.canvas.getBoundingClientRect();
    this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(this.pointer, this.camera);
    const hits = this.ray.intersectObjects(this.world.children, true);
    for (const h of hits) {
      let o = h.object;
      while (o && !o.userData.name) o = o.parent;
      if (o) return o.userData.name;
    }
    return null;
  }

  _click(e) {
    if (!this._down || Math.hypot(e.clientX - this._down[0], e.clientY - this._down[1]) > 4) return;
    if (this.gizmo.dragging || this.gizmo.axis) return;
    const name = this._pick(e);
    if (this.tool === "measure") {
      if (!name) { this.measure = []; this._drawMeasure(); return; }
      this.measure = this.measure.length >= 2 ? [name] : [...this.measure, name];
      this._drawMeasure();
      return;
    }
    this.select(name);
  }

  _hover(e) {
    if (this.gizmo.dragging) return;
    const name = this._pick(e);
    this.canvas.style.cursor = name ? "pointer" : "";
    this._hovered = name;
  }

  select(name, notify = true) {
    this.selected = name && this.parts.has(name) ? name : null;
    this.gizmo.detach();
    const p = this.selected && this.parts.get(this.selected);
    if (p && this.tool === "move" && p.data.editable) {
      this.gizmo.attach(p.obj);
      this._dragStart = p.obj.position.clone();
    }
    this._decorate();
    this._readout();
    this._dimensions();
    if (notify) this.on.select?.(this.selected);
  }

  // Distance to the centreline for the selected part and its twin, drawn in the
  // plane they sit in. Two numbers side by side are how symmetry is checked:
  // 88.0 and 88.0 is symmetric; 88.0 and 85.0 is not, whatever the eye says.
  _dimensions() {
    this.helpers.children.filter((c) => c.userData.dim).forEach((c) => this.helpers.remove(c));
    this.labels = this.labels.filter((l) => !l.dim);
    const p = this.parts.get(this.selected);
    if (!p || p.data.kind === "frame") return;
    const who = [p];
    const twin = this.parts.get(p.data.twin.xz);
    if (twin) who.push(twin);
    for (const q of who) {
      const a = q.obj.position.clone();
      const b = new THREE.Vector3(0, a.y, a.z);
      if (Math.abs(a.x) < 0.05) continue;
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]),
        new THREE.LineDashedMaterial({ color: q === p ? C.accent : 0x9aa1a7, dashSize: 4,
                                       gapSize: 3, depthTest: false }));
      line.computeLineDistances();
      line.userData.dim = true;
      this.helpers.add(line);
      this.labels.push({ at: a.clone().add(b).multiplyScalar(0.5), dim: true,
                         text: `${Math.abs(a.x).toFixed(1)}`, cls: q === p ? "accent" : "" });
    }
  }

  // While dragging, twins follow immediately; the server confirms on release.
  _liveMirror() {
    const p = this.parts.get(this.selected);
    if (!p) return;
    const o = p.obj.position;
    const s = this.snap;
    if (s > 0) { o.x = Math.round(o.x / s) * s; o.y = Math.round(o.y / s) * s; o.z = Math.round(o.z / s) * s; }
    const planes = this.link === "off" ? [] : this._planes();
    if (planes.includes("xz") && p.data.centred.xz) o.x = 0;
    if (planes.includes("yz") && p.data.centred.yz) o.z = 0;
    const put = (n, x, z) => { const t = this.parts.get(n); if (t) t.obj.position.set(x, o.y, z); };
    const lr = planes.includes("xz") ? p.data.twin.xz : null;
    const fb = planes.includes("yz") ? p.data.twin.yz : null;
    if (lr) put(lr, -o.x, o.z);
    if (fb) put(fb, o.x, -o.z);
    if (lr && fb) {
      const diag = this.parts.get(lr)?.data.twin.yz;
      if (diag && diag !== this.selected) put(diag, -o.x, -o.z);
    }
    this._readout();
    this._dimensions();
  }

  _commitDrag() {
    const p = this.parts.get(this.selected);
    if (!p || !this._dragStart || p.obj.position.distanceTo(this._dragStart) < 1e-6) return;
    this.on.moved?.(this.selected, fromThree(p.obj.position).map((x) => Math.round(x * 1000) / 1000));
  }

  _readout() {
    const el = document.getElementById("readout");
    const p = this.parts.get(this.selected);
    if (!p) { el.textContent = ""; return; }
    const [f, r, u] = fromThree(p.obj.position);
    const fmt = (x) => (Math.abs(x) < 0.05 ? 0 : x).toFixed(1).padStart(7);
    el.textContent = `${p.data.name.replace(/_/g, " ")}   fwd ${fmt(f)}   right ${fmt(r)}   up ${fmt(u)} mm`
      + `   ·   ${Math.abs(r) < 0.05 ? "on the centreline" : Math.abs(r).toFixed(1) + " mm from centreline"}`;
  }

  _drawMeasure() {
    this.helpers.children.filter((c) => c.userData.measure).forEach((c) => this.helpers.remove(c));
    this.labels = this.labels.filter((l) => !l.measure);
    const pts = this.measure.map((n) => this.parts.get(n)).filter(Boolean);
    for (const p of pts) this.labels.push({ at: p.obj.position, text: p.data.name, cls: "accent", measure: true });
    if (pts.length === 2) {
      const [a, b] = pts.map((p) => p.obj.position.clone());
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]),
        new THREE.LineBasicMaterial({ color: C.accent, depthTest: false }));
      line.userData.measure = true;
      this.helpers.add(line);
      const [fa, ra, ua] = fromThree(a), [fb, rb, ub] = fromThree(b);
      const d = Math.hypot(fb - fa, rb - ra, ub - ua);
      this.labels.push({ at: a.clone().add(b).multiplyScalar(0.5), cls: "accent", measure: true,
        text: `${d.toFixed(1)} mm   Δfwd ${(fb - fa).toFixed(1)}  Δright ${(rb - ra).toFixed(1)}  Δup ${(ub - ua).toFixed(1)}` });
      this.on.measured?.({ a: pts[0].data.name, b: pts[1].data.name, d,
                           dx: fb - fa, dy: rb - ra, dz: ub - ua });
    } else this.on.measured?.(null);
  }

  // ---- views ------------------------------------------------------------------
  setTool(t) { this.tool = t; this.measure = []; this._drawMeasure(); this.select(this.selected, false); }
  setLink(l) { this.link = l; if (this.v) { this._decorate(); this._helpers(); this._drawMeasure(); } }
  setSnap(s) { this.snap = s; this.gizmo.setTranslationSnap(s > 0 ? s : null); }

  // ---- flight -------------------------------------------------------------
  // The simulator speaks FLU metres (x forward, y left, z up); this view is
  // millimetres in three's (right, up, back). One rotation and one scale, and
  // the vehicle built for the layout editor flies without being rebuilt.
  //
  // C: sim -> three is (x, y, z) -> (-y, z, -x). A proper rotation (det +1),
  // so attitudes convert by conjugation and nothing turns inside out.
  static SIM = new THREE.Matrix4().set(0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 1);

  static toScene([x, y, z]) { return new THREE.Vector3(-y * M, z * M, -x * M); }

  static toSceneQuat([w, x, y, z]) {
    const c = new THREE.Quaternion().setFromRotationMatrix(Viewport.SIM);
    return c.clone().multiply(new THREE.Quaternion(x, y, z, w))
      .multiply(c.clone().invert());
  }

  enterFlight(scene) {
    this.flying = true;
    this.gizmo.detach();
    this.gizmo.visible = false;
    this.helpers.visible = false;
    this.labels = [];
    this.labelsEl.innerHTML = "";
    this.craft = new THREE.Group();
    this.craft.add(this.world);                 // the design, now airborne
    this.world.traverse((o) => { if (o.userData.designOnly) o.visible = false; });
    this.world.position.set(0, 0, 0);
    this.world.quaternion.identity();
    this.air = new THREE.Group();
    this.air.add(this.craft);
    this._buildAir(scene);
    this.scene.add(this.air);
    // Where an FPV camera actually sits: at the nose, just clear of the
    // airframe. At the centre of mass you look at the inside of the battery.
    const box = new THREE.Box3().setFromObject(this.world);
    this._nose = new THREE.Vector3(0, box.max.y * 0.35, box.min.z - 30);
    this.camera = this.persp;
    this.controls.object = this.persp;
    this.persp.near = 20;
    this.persp.far = 600000;
    this.persp.updateProjectionMatrix();
    this.setFlightCamera("chase");
  }

  exitFlight() {
    if (!this.flying) return;
    this.flying = false;
    this.scene.add(this.world);
    this.world.position.set(0, 0, 0);
    this.world.quaternion.identity();
    this.world.traverse((o) => { if (o.userData.designOnly) o.visible = true; });
    this.air?.removeFromParent();
    this.air = this.craft = null;
    this.gizmo.visible = true;
    this.helpers.visible = true;
    this.controls.enabled = true;
    this.persp.near = 1;
    this.persp.far = 20000;
    this.persp.fov = 38;
    this.persp.updateProjectionMatrix();
    this.setView(this.view);
  }

  setFlightCamera(mode) {
    this.camMode = mode;
    this.controls.enabled = mode === "orbit";
    this.persp.fov = mode === "fpv" ? 95 : 38;
    this.persp.updateProjectionMatrix();
    this._chase = null;                          // re-place, do not glide there
  }

  _buildAir(scene) {
    const g = new THREE.Group();
    const half = 40 * M, minor = 1 * M, major = 5 * M;
    const grid = (step, color) => {
      const pts = [];
      for (let x = -half; x <= half + 1e-6; x += step) {
        pts.push(x, 0, -half, x, 0, half, -half, 0, x, half, 0, x);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
      return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color }));
    };
    g.add(grid(minor, C.gridMinor), grid(major, C.gridMajor), this._sky());
    this.gates = [];
    for (const gate of scene.gates || []) {
      const frame = this._gate(gate);
      g.add(frame);
      this.gates.push(frame);
    }
    this.air.add(g);
  }

  // A horizon. Without one there is no up: in the FPV view the sky and the
  // unlit ground are the same black, and a vehicle rolling has nothing to roll
  // against. Two greys either side of y = 0, inside a dome.
  _sky() {
    const mat = new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false,
      uniforms: { sky: { value: new THREE.Color(0x141719) },
                  ground: { value: new THREE.Color(0x090a0b) },
                  line: { value: new THREE.Color(0x2c3236) } },
      vertexShader: `varying float h;
        void main() {
          vec4 w = modelMatrix * vec4(position, 1.0);
          h = normalize(w.xyz).y;
          gl_Position = projectionMatrix * viewMatrix * w;
        }`,
      fragmentShader: `uniform vec3 sky; uniform vec3 ground; uniform vec3 line;
        varying float h;
        void main() {
          vec3 c = mix(ground, sky, smoothstep(-0.08, 0.35, h));
          c = mix(c, line, smoothstep(0.02, 0.0, abs(h)) * 0.5);   // the horizon itself
          gl_FragColor = vec4(c, 1.0);
        }`,
    });
    const dome = new THREE.Mesh(new THREE.SphereGeometry(300 * M, 32, 16), mat);
    dome.renderOrder = -1;
    return dome;
  }

  // A gate, built the way the simulator builds it: four bars round a 1.5 m
  // opening, standing on a post.
  _gate(gate) {
    const grp = new THREE.Group();
    const s = gate.size * M, t = 0.05 * M, h = s / 2 + t;
    const mat = new THREE.MeshStandardMaterial({ color: C.gate, roughness: 0.6 });
    const bar = (x, y, sx, sy) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, t), mat);
      m.position.set(x, y, 0);
      return m;
    };
    // The frame turns to face along the gate's normal; the post does not --
    // it holds the gate up, and gravity is not part of the racing line. Making
    // the post a child of the turned frame laid every gate's post on its side.
    const frame = new THREE.Group();
    frame.add(bar(-h, 0, t, s + 2 * t), bar(h, 0, t, s + 2 * t),
              bar(0, h, s, t), bar(0, -h, s, t));
    const centre = Viewport.toScene(gate.centre);
    const n = Viewport.toScene(gate.normal).normalize();
    frame.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
    const stand = Math.max(centre.y - h, 1);
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.06 * M, stand, 0.06 * M),
      new THREE.MeshStandardMaterial({ color: 0x4d5257, roughness: 0.9 }));
    post.position.y = -(h + stand / 2);
    grp.add(frame, post);
    grp.position.copy(centre);
    grp.userData.bars = mat;
    return grp;
  }

  telemetry(s) {
    if (!this.flying || !this.craft) return;
    this.craft.position.copy(Viewport.toScene(s.pos));
    this.craft.quaternion.copy(Viewport.toSceneQuat(s.quat));
    const next = s.gates_total ? s.gates_passed % s.gates_total : -1;
    this.gates?.forEach((g, i) => {
      g.userData.bars.color.setHex(i === next ? C.accent : C.gate);
    });
    this._camera(s);
  }

  _camera(s) {
    const p = this.craft.position, q = this.craft.quaternion;
    if (this.camMode === "orbit") {
      this.controls.target.copy(p);
      return;
    }
    if (this.camMode === "fpv") {
      // Where the pilot's camera sits: at the vehicle, tilted up 25 degrees,
      // which is what a racer flies with -- level in cruise, nose down.
      this.persp.position.copy(p).add(this._nose.clone().applyQuaternion(q));
      this.persp.quaternion.copy(q).multiply(
        new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0),
          THREE.MathUtils.degToRad(25)));
      return;
    }
    // Chase: behind and above, following the heading but not the roll, so the
    // horizon stays level and you can see what the vehicle is doing.
    const yaw = Math.atan2(2 * (q.w * q.y + q.x * q.z),
                           1 - 2 * (q.y * q.y + q.x * q.x));
    const back = new THREE.Vector3(Math.sin(yaw) * 2.6 * M, 0.85 * M,
                                   Math.cos(yaw) * 2.6 * M);
    const want = p.clone().add(back);
    this._chase = this._chase ? this._chase.lerp(want, 0.10) : want;
    this.persp.position.copy(this._chase);
    // Aim ahead of the vehicle, not at it: aiming at the vehicle tips the view
    // down at the ground and puts everything it is flying towards -- the next
    // gate -- just above the top of the frame.
    this.persp.lookAt(p.clone().add(new THREE.Vector3(
      -Math.sin(yaw) * 3.5 * M, 0.25 * M, -Math.cos(yaw) * 3.5 * M)));
  }

  setView(view) {
    this.view = view;
    const box = new THREE.Box3().setFromObject(this.world);
    const c = box.isEmpty() ? new THREE.Vector3() : box.getCenter(new THREE.Vector3());
    const size = box.isEmpty() ? 300 : box.getSize(new THREE.Vector3()).length();
    if (view === "3d") {
      this.camera = this.persp;
      this.controls.object = this.persp;
      this.controls.enableRotate = true;
      this.fit();
    } else {
      this.camera = this.ortho;
      this.controls.object = this.ortho;
      this.controls.enableRotate = false;
      const d = size * 2;
      const pos = { top: [c.x, c.y + d, c.z], front: [c.x, c.y, c.z - d], side: [c.x + d, c.y, c.z] }[view];
      this.ortho.up.set(...(view === "top" ? [0, 0, -1] : [0, 1, 0]));
      this.ortho.position.set(...pos);
      this.ortho.zoom = 1;
      this._orthoSize = size * 0.62;
      this._resize();
      this.controls.target.copy(c);
      this.ortho.lookAt(c);
    }
    this.gizmo.camera = this.camera;
    this.controls.update();
  }

  fit() {
    const box = new THREE.Box3().setFromObject(this.world);
    if (box.isEmpty()) return;
    const c = box.getCenter(new THREE.Vector3());
    const r = box.getSize(new THREE.Vector3()).length() * 0.5;
    const dist = r / Math.sin(THREE.MathUtils.degToRad(this.persp.fov / 2)) * 1.15;
    this.persp.position.set(c.x + dist * 0.55, c.y + dist * 0.55, c.z + dist * 0.62);
    this.controls.target.copy(c);
    this.persp.lookAt(c);
  }

  _resize() {
    const el = this.canvas.parentElement;
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.persp.aspect = w / h;
    this.persp.updateProjectionMatrix();
    const s = this._orthoSize || 300, a = w / h;
    Object.assign(this.ortho, { left: -s * a, right: s * a, top: s, bottom: -s });
    this.ortho.updateProjectionMatrix();
  }

  _placeLabels() {
    const el = this.labelsEl;
    if (!el) return;
    const w = el.clientWidth, h = el.clientHeight;
    while (el.children.length < this.labels.length) el.appendChild(document.createElement("div"));
    [...el.children].forEach((node, i) => {
      const l = this.labels[i];
      if (!l) { node.style.display = "none"; return; }
      const p = l.at.clone().project(this.camera);
      const behind = p.z > 1;
      node.style.display = behind ? "none" : "";
      node.className = "lbl " + (l.cls || "");
      node.textContent = l.text;
      node.style.left = `${(p.x * 0.5 + 0.5) * w}px`;
      node.style.top = `${(-p.y * 0.5 + 0.5) * h}px`;
    });
  }
}
