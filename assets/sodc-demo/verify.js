import { e as P, c as J, a as O, t as W, l as q, h as G } from "./weights.js";
function H(a) {
  const t = atob(a),
    e = new Uint8Array(t.length);
  for (let s = 0; s < t.length; s++) e[s] = t.charCodeAt(s);
  return e;
}
function u(a) {
  const t = H(a.data_b64),
    e = new ArrayBuffer(t.byteLength);
  return new Uint8Array(e).set(t), new Float32Array(e);
}
function K(a) {
  const t = H(a.data_b64),
    e = new ArrayBuffer(t.byteLength);
  return new Uint8Array(e).set(t), new Int32Array(e);
}
const $ = 0.05;
function Y(a, t) {
  return Q(a, t);
}
function Q(a, t) {
  const e = t.header,
    s = t.wires.map(K),
    r = t.wires.map((h) => [h.shape[0], h.shape[1]]),
    i = t.layer_sizes.map((h) => [h[0], h[1]]),
    c = P(i, s, r, { arity: e.arity, hiddenDim: e.circuit_hidden_dim, maxNeighbors: e.max_neighbors }),
    o = {
      logits: u(t.initial_state.logits),
      hidden: u(t.initial_state.hidden),
      loss: u(t.initial_state.loss),
      gateMask: u(t.initial_state.gate_mask),
    },
    n = c.nNodes * (1 << e.arity);
  if (o.logits.length !== n) throw new Error(`initial_state.logits length ${o.logits.length} ≠ N×lutDim ${n}`);
  const p = u(t.task_data.x),
    m = u(t.task_data.y),
    x = e.case_n,
    I = J(o, c, p, m, x, e.arity),
    j = new Float32Array(I.predHard),
    M = O(a, c.nNodes),
    L = [];
  let f = 0;
  for (const h of t.ticks) {
    const y = W(o, c, a, M, p, m, x, e.arity),
      N = u(h.logits),
      z = u(h.hidden),
      U = u(h.loss);
    let k = 0;
    for (let l = 0; l < o.logits.length; l++) {
      const g = Math.abs(o.logits[l] - N[l]);
      g > k && (k = g);
    }
    let A = 0;
    for (let l = 0; l < o.hidden.length; l++) {
      const g = Math.abs(o.hidden[l] - z[l]);
      g > A && (A = g);
    }
    let _ = 0;
    for (let l = 0; l < o.loss.length; l++) {
      const g = Math.abs(o.loss[l] - U[l]);
      g > _ && (_ = g);
    }
    const w = Math.abs(y.hardAccuracy - h.hard_accuracy);
    w > f && (f = w),
      L.push({
        step: h.step,
        hardAccJax: h.hard_accuracy,
        hardAccTs: y.hardAccuracy,
        hardAccDelta: w,
        maxAbsLogitsDelta: k,
        maxAbsHiddenDelta: A,
        maxAbsLossDelta: _,
        predHard: new Float32Array(y.predHard),
      });
  }
  const D = f < $,
    B = D
      ? `TS-side parity PASS: max hard_acc Δ over ${t.ticks.length} ticks = ${f.toFixed(4)} (tol ${$})`
      : `TS-side parity FAIL: max hard_acc Δ = ${f.toFixed(4)} > tol ${$}`;
  return {
    pass: D,
    nTicks: t.ticks.length,
    maxHardAccDelta: f,
    perTick: L,
    message: B,
    initialPredHard: j,
    taskInputBits: p,
    taskTargetBits: m,
    caseN: x,
    inputBits: e.input_bits,
    outputBits: e.output_bits,
    taskStyle: e.task_style ?? "sequential",
    text: e.text ?? null,
  };
}
const T = "sodc-demo",
  R = "/assets/sodc-demo/",
  F = `${R}weights/reverse_random_damage.json`,
  V = `${R}weights/reverse_trajectory.json`;
function d(a, t, e) {
  const s = document.createElement(a);
  return t && (s.className = t), e !== void 0 && (s.textContent = e), s;
}
function E(a) {
  for (; a.firstChild; ) a.removeChild(a.firstChild);
}
function X(a) {
  E(a);
  const t = d("div", "sodc-wrap"),
    e = d("div", "sodc-status", "Loading weights ..."),
    s = d("pre", "sodc-log"),
    r = d("div", "sodc-image-panel"),
    i = d("div", "sodc-tick-label", "tick 0"),
    c = v("input  (x)"),
    o = v("current TMT output"),
    n = v("expected (y)");
  r.append(i, c.row, o.row, n.row);
  const p = d("table", "sodc-table"),
    m = document.createElement("style");
  return (
    (m.textContent = `
    .sodc-wrap { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .sodc-status { font-size: 1.1em; padding: 0.5em 0; font-weight: 600; }
    .sodc-status.pass { color: #1a7f37; }
    .sodc-status.fail { color: #cf222e; }
    .sodc-log { background: #f6f8fa; padding: 0.75em 1em; border-radius: 6px;
                font-size: 0.85em; line-height: 1.4; max-height: 14em; overflow: auto; }
    .sodc-image-panel { margin: 1em 0; padding: 1em; background: #0d1117;
                        border-radius: 8px; color: #d0d7de; }
    .sodc-tick-label { font-size: 0.9em; opacity: 0.85; margin-bottom: 0.5em;
                       font-variant-numeric: tabular-nums; }
    .sodc-image-row { display: grid; grid-template-columns: 12em 1fr; align-items: center;
                      gap: 1em; margin: 0.4em 0; }
    .sodc-image-label { font-size: 0.85em; opacity: 0.85; text-align: right; }
    .sodc-image-canvas { width: 100%; image-rendering: pixelated;
                         border: 1px solid #30363d; background: #1117; }
    .sodc-table { border-collapse: collapse; margin-top: 0.6em; font-size: 0.85em; }
    .sodc-table th, .sodc-table td { padding: 2px 10px; text-align: right;
                                     border-bottom: 1px solid #eaecef;
                                     font-variant-numeric: tabular-nums; }
    .sodc-table th { text-align: right; background: #f6f8fa; }
    .sodc-table th:first-child, .sodc-table td:first-child { text-align: left; }
  `),
    t.append(m, e, s, r, p),
    a.append(t),
    { status: e, log: s, imagePanel: r, inputCanvas: c.canvas, currentCanvas: o.canvas, expectedCanvas: n.canvas, tickLabel: i, table: p }
  );
}
function v(a) {
  const t = d("div", "sodc-image-row"),
    e = d("div", "sodc-image-label", a),
    s = d("canvas", "sodc-image-canvas");
  return t.append(e, s), { row: t, canvas: s };
}
function C(a, t, e, s, r = 0.5) {
  (a.width !== e || a.height !== s) &&
    ((a.width = e), (a.height = s), (a.style.aspectRatio = `${e} / ${s}`), (a.style.height = `${Math.max(40, s * 4)}px`));
  const i = a.getContext("2d");
  if (!i) return;
  const c = i.createImageData(e, s);
  for (let o = 0; o < s; o++)
    for (let n = 0; n < e; n++) {
      const p = t[n * s + o] >= r ? 255 : 32,
        m = (o * e + n) * 4;
      (c.data[m] = p), (c.data[m + 1] = p), (c.data[m + 2] = p), (c.data[m + 3] = 255);
    }
  i.putImageData(c, 0, 0);
}
function Z(a, t) {
  E(a);
  const e = d("thead"),
    s = d("tr");
  for (const i of ["step", "hard_acc (jax)", "hard_acc (ts)", "Δ", "Δ logits", "Δ hidden", "Δ loss"]) s.append(d("th", void 0, i));
  e.append(s);
  const r = d("tbody");
  for (const i of t) {
    const c = d("tr");
    for (const o of [
      String(i.step),
      i.hardAccJax.toFixed(4),
      i.hardAccTs.toFixed(4),
      i.hardAccDelta.toFixed(4),
      i.maxAbsLogitsDelta.toExponential(2),
      i.maxAbsHiddenDelta.toExponential(2),
      i.maxAbsLossDelta.toExponential(2),
    ])
      c.append(d("td", void 0, o));
    r.append(c);
  }
  a.append(e, r);
}
function tt(a, t, e = 6) {
  const s = [{ step: 0, bits: t.initialPredHard, hardAcc: NaN }, ...t.perTick.map((o) => ({ step: o.step, bits: o.predHard, hardAcc: o.hardAccTs }))];
  let r = 0;
  const i = 1e3 / e,
    c = () => {
      const o = s[r % s.length];
      C(a.currentCanvas, o.bits, t.caseN, t.outputBits);
      const n = Number.isFinite(o.hardAcc) ? `, hard_acc=${o.hardAcc.toFixed(4)}` : "";
      (a.tickLabel.textContent = `tick ${o.step}${n}`), r++;
    };
  c(), setInterval(c, i);
}
function b(a, t) {
  a.log.textContent +=
    t +
    `
`;
}
async function S() {
  const a = document.getElementById(T);
  if (!a) {
    console.error(`[sodc-demo] mount target #${T} not found`);
    return;
  }
  const t = X(a);
  try {
    t.status.textContent = "Loading weights ...";
    const e = performance.now(),
      s = await q(F),
      r = performance.now() - e;
    b(
      t,
      `weights:  ${F}
  arch:    ${s.header.modelKind} (gathered) · D=${s.header.attentionDim}, H=${s.header.numHeads}, arity=${s.header.arity}, hidden=${s.header.circuitHiddenDim}
  params:  ${G(s).toLocaleString()}
  dtype:   ${s.header.tensorDtype}
  loaded:  ${r.toFixed(0)} ms`
    ),
      (t.status.textContent = "Fetching reference trajectory ...");
    const i = await fetch(V);
    if (!i.ok) throw new Error(`Failed to fetch trajectory: ${i.status}`);
    const c = await i.json();
    t.status.textContent = "Running parity replay ...";
    const o = performance.now(),
      n = Y(s, c),
      p = performance.now() - o;
    b(t, `replay:   ${n.nTicks} ticks in ${p.toFixed(0)} ms`),
      n.text ? b(t, `task:     ${n.taskStyle} (text="${n.text}", n_cases=${n.caseN})`) : b(t, `task:     ${n.taskStyle} (n_cases=${n.caseN})`),
      b(t, n.message),
      (t.status.textContent = n.message),
      t.status.classList.add(n.pass ? "pass" : "fail"),
      C(t.inputCanvas, n.taskInputBits, n.caseN, n.inputBits),
      C(t.expectedCanvas, n.taskTargetBits, n.caseN, n.outputBits),
      tt(t, n),
      Z(t.table, n.perTick);
  } catch (e) {
    console.error(e), (t.status.textContent = `Error: ${e.message}`), t.status.classList.add("fail");
  }
}
document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", () => void S()) : S();
