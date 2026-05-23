(function () {
  const e = document.createElement("link").relList;
  if (e && e.supports && e.supports("modulepreload")) return;
  for (const i of document.querySelectorAll('link[rel="modulepreload"]')) r(i);
  new MutationObserver((i) => {
    for (const n of i) if (n.type === "childList") for (const f of n.addedNodes) f.tagName === "LINK" && f.rel === "modulepreload" && r(f);
  }).observe(document, { childList: !0, subtree: !0 });
  function o(i) {
    const n = {};
    return (
      i.integrity && (n.integrity = i.integrity),
      i.referrerPolicy && (n.referrerPolicy = i.referrerPolicy),
      i.crossOrigin === "use-credentials"
        ? (n.credentials = "include")
        : i.crossOrigin === "anonymous"
          ? (n.credentials = "omit")
          : (n.credentials = "same-origin"),
      n
    );
  }
  function r(i) {
    if (i.ep) return;
    i.ep = !0;
    const n = o(i);
    fetch(i.href, n);
  }
})();
function M(t, e, o, r, i, n, f) {
  for (let s = 0; s < i; s++)
    for (let a = 0; a < f; a++) {
      let l = r ? r[a] : 0;
      const d = s * n;
      for (let c = 0; c < n; c++) l += e[d + c] * o[c * f + a];
      t[s * f + a] = l;
    }
}
function S(t, e, o, r, i, n, f = 1e-6) {
  for (let s = 0; s < i; s++) {
    const a = s * n;
    let l = 0;
    for (let m = 0; m < n; m++) l += e[a + m];
    l /= n;
    let d = 0;
    for (let m = 0; m < n; m++) {
      const u = e[a + m] - l;
      d += u * u;
    }
    const c = 1 / Math.sqrt(d / n + f);
    for (let m = 0; m < n; m++) {
      const u = (e[a + m] - l) * c * o[m];
      t[a + m] = r ? u + r[m] : u;
    }
  }
}
function j(t, e, o, r) {
  for (let i = 0; i < o; i++) {
    const n = i * r;
    let f = -1 / 0;
    for (let l = 0; l < r; l++) {
      const d = e[n + l];
      d > f && (f = d);
    }
    let s = 0;
    for (let l = 0; l < r; l++) {
      const d = Math.exp(e[n + l] - f);
      (t[n + l] = d), (s += d);
    }
    const a = 1 / s;
    for (let l = 0; l < r; l++) t[n + l] *= a;
  }
}
function B(t) {
  if (t >= 0) return 1 / (1 + Math.exp(-t));
  const e = Math.exp(t);
  return e / (1 + e);
}
const x = Math.sqrt(2 / Math.PI);
function z(t) {
  return 0.5 * t * (1 + Math.tanh(x * (t + 0.044715 * t * t * t)));
}
function E(t) {
  for (let e = 0; e < t.length; e++) t[e] = z(t[e]);
}
function W(t, e, o, r = 1e4) {
  if (o & 1) throw new Error(`sinusoidalPE: dim must be even, got ${o}`);
  const i = e.length,
    n = o >>> 1,
    f = -Math.log(r) / o,
    s = new Float32Array(n);
  for (let a = 0; a < n; a++) s[a] = Math.exp(2 * a * f);
  for (let a = 0; a < i; a++) {
    const l = e[a],
      d = a * o;
    for (let c = 0; c < n; c++) {
      const m = l * s[c];
      (t[d + 2 * c] = Math.sin(m)), (t[d + 2 * c + 1] = Math.cos(m));
    }
  }
}
function H(t, e) {
  const o = new Int32Array(e * e);
  for (let r = 0; r < e; r++) {
    let i = 0;
    for (let n = 0; n < e; n++) t[r * e + n] && (o[r * e + i++] = n);
    for (let n = 0; n < e; n++) t[r * e + n] || (o[r * e + i++] = n);
  }
  return o;
}
const L = -10,
  w = 1e4;
function I(t, e, o, r, i) {
  let n = 0;
  for (let l = 1; l < t.length; l++) n += t[l][0] * i;
  const f = new Int32Array(n),
    s = new Int32Array(n);
  let a = 0;
  for (let l = 1; l < t.length; l++) {
    const [d, c] = t[l],
      m = d / c,
      u = o[l - 1],
      [h, g] = r[l - 1];
    if (h !== i || g !== m) throw new Error(`wire shape mismatch: layer ${l} expected (${i}, ${m}) got (${h}, ${g})`);
    const y = e[l],
      b = e[l - 1];
    for (let p = 0; p < d; p++) {
      const A = y + p,
        _ = (p / c) | 0;
      for (let k = 0; k < i; k++) {
        const P = u[k * m + _];
        (f[a] = b + P), (s[a] = A), a++;
      }
    }
  }
  return { senders: f, receivers: s };
}
function q(t, e, o) {
  const r = new Uint8Array(o * o);
  for (let i = 0; i < o; i++) r[i * o + i] = 1;
  for (let i = 0; i < t.length; i++) {
    const n = t[i],
      f = e[i];
    (r[f * o + n] = 1), (r[n * o + f] = 1);
  }
  return r;
}
function $(t, e) {
  let o = 0;
  const r = [];
  for (const [s] of t) r.push(o), (o += s);
  const i = new Float32Array(o * e),
    n = new Float32Array(o * e),
    f = t.length - 1;
  for (let s = 0; s < t.length; s++) {
    const [a] = t[s],
      l = r[s],
      c = (f > 0 ? s / f : 0) * w,
      m = new Float32Array(a);
    for (let h = 0; h < a; h++) m[h] = c;
    W(i.subarray(l * e, (l + a) * e), m, e);
    const u = new Float32Array(a);
    for (let h = 0; h < a; h++) u[h] = h;
    W(n.subarray(l * e, (l + a) * e), u, e);
  }
  return { layerPe: i, intraLayerPe: n, nNodes: o, layerStart: r };
}
function O(t, e, o) {
  const r = Math.min(o, e),
    i = H(t, e),
    n = new Int32Array(e * r),
    f = new Uint8Array(e * r);
  for (let s = 0; s < e; s++) {
    let a = 0;
    for (let l = 0; l < e; l++) t[s * e + l] && a++;
    for (let l = 0; l < r; l++) (n[s * r + l] = i[s * e + l]), (f[s * r + l] = l < a ? 1 : 0);
  }
  return { neighborIndices: n, neighborMask: f, maxNeighbors: r };
}
function N(t, e, o, r) {
  const { arity: i, hiddenDim: n, maxNeighbors: f } = r,
    { layerPe: s, intraLayerPe: a, nNodes: l, layerStart: d } = $(t, n),
    { senders: c, receivers: m } = I(t, d, e, o, i),
    u = q(c, m, l),
    h = O(u, l, f),
    g = d[t.length - 1],
    y = g + t[t.length - 1][0];
  return {
    layerSizes: [...t],
    wires: e,
    wiresShape: o,
    arity: i,
    nNodes: l,
    layerStart: d,
    outputStart: g,
    outputEnd: y,
    attentionMask: u,
    layerPe: s,
    intraLayerPe: a,
    neighborIndices: h.neighborIndices,
    neighborMask: h.neighborMask,
    maxNeighbors: h.maxNeighbors,
  };
}
function tt(t) {
  let e = t >>> 0;
  return {
    next() {
      e = (e + 1831565813) >>> 0;
      let o = e;
      return (o = Math.imul(o ^ (o >>> 15), o | 1)), (o ^= o + Math.imul(o ^ (o >>> 7), o | 61)), ((o ^ (o >>> 14)) >>> 0) / 4294967296;
    },
    normal() {
      const o = Math.max(this.next(), 1e-12),
        r = this.next();
      return Math.sqrt(-2 * Math.log(o)) * Math.cos(2 * Math.PI * r);
    },
  };
}
function R(t, e, o, r, i) {
  const n = (o * r) / i,
    f = Math.max(e, n),
    s = new Int32Array(f);
  for (let d = 0; d < f; d++) s[d] = d;
  for (let d = f - 1; d > 0; d--) {
    const c = (t.next() * (d + 1)) | 0,
      m = s[d];
    (s[d] = s[c]), (s[c] = m);
  }
  const a = new Int32Array(r * (n / r));
  for (let d = 0; d < n; d++) a[d] = s[d] % e;
  const l = n / r;
  return { data: a, shape: [r, l] };
}
function G(t, e, o, r, i, n) {
  const f = 1 << o;
  for (let s = 0; s < e; s++) {
    const a = s % o;
    for (let l = 0; l < f; l++) {
      const c = (2 * ((l >> a) & 1) - 1) * 3,
        m = t.normal() * r;
      i[n + s * f + l] = c + m;
    }
  }
}
function U(t, e, o) {
  const { hiddenDim: r, arity: i, noiseScale: n = 0.1 } = e,
    f = 1 << i,
    s = t.nNodes,
    a = new Float32Array(s * f);
  for (let l = 1; l < t.layerSizes.length; l++) {
    const [d] = t.layerSizes[l],
      c = t.layerStart[l] * f;
    G(o, d, i, n, a, c);
  }
  return { logits: a, hidden: new Float32Array(s * r), loss: new Float32Array(s), gateMask: new Float32Array(s).fill(1) };
}
function K(t, e, o, r, i, n, f, s, a, l) {
  const d = 1 << i,
    c = new Float32Array(d);
  for (let m = 0; m < n; m++)
    for (let u = 0; u < f; u++) {
      let h = u * s * d;
      for (let g = 0; g < s; g++) {
        for (let b = 0; b < d; b++) {
          let p = B(t[h + b]);
          l && (p = p >= 0.5 ? 1 : 0), (c[b] = p);
        }
        let y = d;
        for (let b = 0; b < i; b++) {
          const p = r[b * f + u],
            A = e[m * o + p],
            _ = y >> 1;
          for (let k = 0; k < _; k++) c[k] = (1 - A) * c[2 * k] + A * c[2 * k + 1];
          y = _;
        }
        (a[m * (f * s) + u * s + g] = c[0]), (h += d);
      }
    }
}
function F(t, e, o, r, i, n) {
  const f = 1 << n,
    s = [],
    a = t.layerSizes[0][0],
    l = e.gateMask.subarray(t.layerStart[0], t.layerStart[0] + a);
  let d = new Float32Array(r * a);
  for (let m = 0; m < r; m++) for (let u = 0; u < a; u++) d[m * a + u] = o[m * a + u] * l[u];
  s.push(d);
  let c = a;
  for (let m = 1; m < t.layerSizes.length; m++) {
    const [u, h] = t.layerSizes[m],
      g = u / h,
      y = t.wires[m - 1],
      b = e.logits.subarray(t.layerStart[m] * f, (t.layerStart[m] + u) * f),
      p = e.gateMask.subarray(t.layerStart[m], t.layerStart[m] + u),
      A = new Float32Array(r * u);
    K(b, d, c, y, n, r, g, h, A, i);
    for (let _ = 0; _ < r; _++) for (let k = 0; k < u; k++) A[_ * u + k] *= p[k];
    s.push(A), (d = A), (c = u);
  }
  return s;
}
function et(t, e, o, r, i = 0.1) {
  return U(t, { arity: o, hiddenDim: r, noiseScale: i }, e);
}
function nt(t, e, o, r) {
  const i = e.layerSizes[0][0],
    n = e.layerSizes[e.layerSizes.length - 1][0],
    f = e.nNodes;
  if (o < i || o >= f - n || t.gateMask[o] === 0) return !1;
  t.gateMask[o] = 0;
  for (let s = 0; s < r; s++) t.logits[o * r + s] = L;
  return !0;
}
function rt(t, e, o, r, i) {
  const n = [];
  for (let a = 1; a < e.layerSizes.length - 1; a++) {
    const [l] = e.layerSizes[a],
      d = e.layerStart[a];
    for (let c = 0; c < l; c++) t.gateMask[d + c] === 1 && n.push(d + c);
  }
  const f = Math.min(o, n.length);
  for (let a = 0; a < f; a++) {
    const l = a + ((r.next() * (n.length - a)) | 0),
      d = n[a];
    (n[a] = n[l]), (n[l] = d);
  }
  const s = new Int32Array(f);
  for (let a = 0; a < f; a++) {
    (s[a] = n[a]), (t.gateMask[s[a]] = 0);
    for (let l = 0; l < i; l++) t.logits[s[a] * i + l] = L;
  }
  return s;
}
function ot(t, e, o, r, i) {
  const n = [],
    f = [];
  for (let s = 1; s < t.layerSizes.length; s++) {
    const [a, l] = t.layerSizes[s],
      d = t.layerSizes[s - 1][0],
      c = R(e, d, a, o, l);
    n.push(c.data), f.push(c.shape);
  }
  return N(t.layerSizes, n, f, { arity: o, hiddenDim: r, maxNeighbors: i });
}
function it(t) {
  t.hidden.fill(0);
}
function at(t, e) {
  const o = t.header.attentionDim,
    r = t.header.featureDim,
    i = t.header.numHeads,
    n = t.header.maxNeighbors,
    f = t.block.ffnW1.length / o,
    s = t.header.logitDim,
    a = t.header.circuitHiddenDim;
  return {
    features: new Float32Array(e * r),
    inputNormed: new Float32Array(e * r),
    proj: new Float32Array(e * o),
    postAttn: new Float32Array(e * o),
    finalNormed: new Float32Array(e * o),
    xNorm: new Float32Array(e * o),
    Q: new Float32Array(e * o),
    K: new Float32Array(e * o),
    V: new Float32Array(e * o),
    scores: new Float32Array(e * i * n),
    attnW: new Float32Array(e * i * n),
    attnOut: new Float32Array(e * o),
    attnOutProj: new Float32Array(e * o),
    ffnNorm: new Float32Array(e * o),
    ffnHidden: new Float32Array(e * f),
    ffnOut: new Float32Array(e * o),
    dLogit: new Float32Array(e * s),
    dHidden: new Float32Array(e * a),
  };
}
function V(t, e, o, r) {
  const { logitDim: i, circuitHiddenDim: n, useLayerPE: f, useIntraLayerPE: s, useNodeLoss: a, featureDim: l } = r,
    d = o.nNodes;
  for (let c = 0; c < d; c++) {
    const m = c * l;
    let u = 0;
    for (let h = 0; h < i; h++) t[m + u + h] = e.logits[c * i + h];
    u += i;
    for (let h = 0; h < n; h++) t[m + u + h] = e.hidden[c * n + h];
    if (((u += n), s)) {
      for (let h = 0; h < n; h++) t[m + u + h] = o.intraLayerPe[c * n + h];
      u += n;
    }
    if (f) {
      for (let h = 0; h < n; h++) t[m + u + h] = o.layerPe[c * n + h];
      u += n;
    }
    if ((a && ((t[m + u] = e.loss[c]), (u += 1)), u !== l)) throw new Error(`extractFeatures cursor=${u} ≠ featureDim=${l}`);
  }
}
function T(t, e, o, r, i, n) {
  const f = e.numHeads,
    s = i / f,
    a = o.maxNeighbors,
    l = n.ffnHidden.length / r;
  S(n.xNorm, t, e.attnNormScale, e.attnNormBias, r, i),
    M(n.Q, n.xNorm, e.Wq, e.bq, r, i, i),
    M(n.K, n.xNorm, e.Wk, e.bk, r, i, i),
    M(n.V, n.xNorm, e.Wv, e.bv, r, i, i),
    S(n.Q, n.Q, e.qLnGamma, null, r * f, s),
    S(n.K, n.K, e.kLnGamma, null, r * f, s);
  const d = Math.sqrt(s);
  for (let u = 0; u < r; u++) {
    const h = u * i;
    for (let g = 0; g < f; g++) {
      const y = h + g * s;
      for (let b = 0; b < a; b++) {
        const A = o.neighborIndices[u * a + b] * i + g * s;
        let _ = 0;
        for (let P = 0; P < s; P++) _ += n.Q[y + P] * n.K[A + P];
        let k = _ / d;
        o.neighborMask[u * a + b] || (k = -1e30), (n.scores[u * f * a + g * a + b] = k);
      }
    }
  }
  j(n.attnW, n.scores, r * f, a);
  for (let u = 0; u < r; u++)
    for (let h = 0; h < f; h++)
      for (let g = 0; g < s; g++) {
        let y = 0;
        for (let b = 0; b < a; b++) {
          const p = o.neighborIndices[u * a + b];
          y += n.attnW[u * f * a + h * a + b] * n.V[p * i + h * s + g];
        }
        n.attnOut[u * i + h * s + g] = y;
      }
  M(n.attnOutProj, n.attnOut, e.Wo, e.bo, r, i, i);
  const c = e.attnRezero;
  for (let u = 0; u < r * i; u++) t[u] += c * n.attnOutProj[u];
  S(n.ffnNorm, t, e.ffnLnScale, e.ffnLnBias, r, i),
    M(n.ffnHidden, n.ffnNorm, e.ffnW1, e.ffnB1, r, i, l),
    E(n.ffnHidden),
    M(n.ffnOut, n.ffnHidden, e.ffnW2, e.ffnB2, r, l, i);
  const m = e.ffnRezero;
  for (let u = 0; u < r * i; u++) t[u] += m * n.ffnOut[u];
}
function Q(t, e, o, r) {
  const { attentionDim: i, featureDim: n, logitDim: f, circuitHiddenDim: s } = o.header,
    a = e.nNodes;
  V(r.features, t, e, o.header),
    S(r.inputNormed, r.features, o.inputNormScale, o.inputNormBias, a, n),
    M(r.proj, r.inputNormed, o.featureProjW, o.featureProjB, a, n, i),
    T(r.proj, o.block, e, a, i, r),
    S(r.finalNormed, r.proj, o.finalNormScale, o.finalNormBias, a, i),
    M(r.dLogit, r.finalNormed, o.logitProjW, o.logitProjB, a, i, f),
    M(r.dHidden, r.finalNormed, o.hiddenProjW, o.hiddenProjB, a, i, s);
  for (let c = 0; c < a; c++)
    if (t.gateMask[c] !== 1) {
      for (let m = 0; m < f; m++) r.dLogit[c * f + m] = 0;
      for (let m = 0; m < s; m++) r.dHidden[c * s + m] = 0;
    }
  const l = o.logitRezero,
    d = o.hiddenRezero;
  for (let c = 0; c < a * f; c++) t.logits[c] += l * r.dLogit[c];
  for (let c = 0; c < a * s; c++) t.hidden[c] += d * r.dHidden[c];
}
function J(t, e, o, r, i, n) {
  const f = F(e, t, o, i, !1, n),
    s = F(e, t, o, i, !0, n),
    a = f[f.length - 1],
    l = s[s.length - 1],
    d = e.layerSizes[e.layerSizes.length - 1][0];
  t.loss.fill(0);
  let c = 0,
    m = 0;
  const u = i * d;
  for (let h = 0; h < d; h++) {
    let g = 0;
    for (let y = 0; y < i; y++) {
      const b = y * d + h,
        p = a[b],
        A = l[b],
        _ = r[b];
      (g += Math.abs(p - _)), Math.abs(p - _) < 0.5 && c++, A === _ && m++;
    }
    t.loss[e.outputStart + h] = g / i;
  }
  return { predSoft: a, predHard: l, softAccuracy: c / u, hardAccuracy: m / u };
}
function st(t, e, o, r, i, n, f, s) {
  return Q(t, e, o, r), J(t, e, i, n, f, s);
}
function C(t) {
  const e = atob(t),
    o = new Uint8Array(e.length);
  for (let r = 0; r < e.length; r++) o[r] = e.charCodeAt(r);
  return o;
}
function X(t) {
  const e = (t & 32768) >>> 15,
    o = (t & 31744) >>> 10,
    r = t & 1023;
  if (o === 0) {
    if (r === 0) return e ? -0 : 0;
    const n = r * Math.pow(2, -24);
    return e ? -n : n;
  }
  if (o === 31) return r === 0 ? (e ? -1 / 0 : 1 / 0) : NaN;
  const i = (1 + r / 1024) * Math.pow(2, o - 15);
  return e ? -i : i;
}
function v(t) {
  const e = C(t.data_b64),
    o = t.shape.reduce((r, i) => r * i, 1);
  if (t.dtype === "fp16") {
    if (e.length !== o * 2) throw new Error(`fp16 byte length mismatch: have ${e.length}, expected ${o * 2}`);
    const r = new Uint16Array(e.buffer, e.byteOffset, o),
      i = new Float32Array(o);
    for (let n = 0; n < o; n++) i[n] = X(r[n]);
    return i;
  }
  if (t.dtype === "uint8") {
    if (t.scale === void 0 || t.zero_point === void 0) throw new Error("uint8 tensor missing scale or zero_point");
    if (e.length !== o) throw new Error(`uint8 byte length mismatch: have ${e.length}, expected ${o}`);
    const r = t.scale,
      i = t.zero_point,
      n = new Float32Array(o);
    for (let f = 0; f < o; f++) n[f] = (e[f] - i) * r;
    return n;
  }
  throw new Error(`Unknown tensor dtype: ${t.dtype}`);
}
function Y(t) {
  if (t.model_kind !== "gathered_attention") throw new Error(`Unsupported model_kind '${t.model_kind}'; only gathered_attention is wired up.`);
  if (t.tensor_dtype !== "fp16" && t.tensor_dtype !== "uint8") throw new Error(`Unsupported tensor_dtype ${t.tensor_dtype}`);
  return {
    schemaVersion: t.schema_version,
    modelKind: "gathered_attention",
    arity: t.arity,
    circuitHiddenDim: t.circuit_hidden_dim,
    attentionDim: t.attention_dim,
    numHeads: t.num_heads,
    numAttnLayers: t.num_attn_layers,
    useLayerPE: t.use_layer_PE,
    useIntraLayerPE: t.use_intra_layer_PE,
    useNodeLoss: t.use_node_loss,
    maxNeighbors: t.max_neighbors,
    logitDim: t.logit_dim,
    featureDim: t.feature_dim,
    useGeluApprox: t.use_gelu_approx,
    sourceRunId: t.source_run_id,
    tensorDtype: t.tensor_dtype,
  };
}
function Z(t, e, o) {
  const r = (n) => {
      const f = t[n];
      if (!f) throw new Error(`Missing tensor ${n} in weights JSON`);
      return v(f);
    },
    i = (n) => {
      const f = e[n];
      if (f === void 0) throw new Error(`Missing scalar ${n} in weights JSON`);
      return f;
    };
  return {
    attnNormScale: r("block.attn_norm.scale"),
    attnNormBias: r("block.attn_norm.bias"),
    Wq: r("block.Wq"),
    bq: r("block.bq"),
    Wk: r("block.Wk"),
    bk: r("block.bk"),
    Wv: r("block.Wv"),
    bv: r("block.bv"),
    Wo: r("block.Wo"),
    bo: r("block.bo"),
    qLnGamma: r("block.q_ln_gamma"),
    kLnGamma: r("block.k_ln_gamma"),
    ffnLnScale: r("block.ffn_ln.scale"),
    ffnLnBias: r("block.ffn_ln.bias"),
    ffnW1: r("block.ffn_W1"),
    ffnB1: r("block.ffn_b1"),
    ffnW2: r("block.ffn_W2"),
    ffnB2: r("block.ffn_b2"),
    attnRezero: i("block.attn_rezero"),
    ffnRezero: i("block.ffn_rezero"),
    numHeads: o,
  };
}
function D(t) {
  const e = t,
    o = Y(e.header),
    r = e.tensors,
    i = e.scalars,
    n = (s) => v(r[s]),
    f = (s) => {
      const a = i[s];
      if (a === void 0) throw new Error(`Missing scalar ${s} in weights JSON`);
      return a;
    };
  return {
    header: o,
    inputNormScale: n("input_norm.scale"),
    inputNormBias: n("input_norm.bias"),
    featureProjW: n("feature_proj.kernel"),
    featureProjB: n("feature_proj.bias"),
    block: Z(r, i, o.numHeads),
    finalNormScale: n("final_norm.scale"),
    finalNormBias: n("final_norm.bias"),
    logitProjW: n("logit_proj.kernel"),
    logitProjB: n("logit_proj.bias"),
    hiddenProjW: n("hidden_proj.kernel"),
    hiddenProjB: n("hidden_proj.bias"),
    logitRezero: f("logit_rezero"),
    hiddenRezero: f("hidden_rezero"),
  };
}
async function lt(t) {
  const e = await fetch(t);
  if (!e.ok) throw new Error(`Failed to fetch weights at ${t}: ${e.status} ${e.statusText}`);
  return D(await e.json());
}
function ft(t) {
  return (
    [
      t.inputNormScale,
      t.inputNormBias,
      t.featureProjW,
      t.featureProjB,
      t.block.attnNormScale,
      t.block.attnNormBias,
      t.block.Wq,
      t.block.bq,
      t.block.Wk,
      t.block.bk,
      t.block.Wv,
      t.block.bv,
      t.block.Wo,
      t.block.bo,
      t.block.qLnGamma,
      t.block.kLnGamma,
      t.block.ffnLnScale,
      t.block.ffnLnBias,
      t.block.ffnW1,
      t.block.ffnB1,
      t.block.ffnW2,
      t.block.ffnB2,
      t.finalNormScale,
      t.finalNormBias,
      t.logitProjW,
      t.logitProjB,
      t.hiddenProjW,
      t.hiddenProjB,
    ].reduce((o, r) => o + r.length, 0) + 4
  );
}
export { at as a, rt as b, J as c, nt as d, N as e, F as f, R as g, ft as h, U as i, lt as l, tt as m, et as r, ot as s, st as t, it as z };
