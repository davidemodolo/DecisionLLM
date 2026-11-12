// Static demo for GitHub Pages: replaces the server API with answers recorded by pages/record.py.
// Aggregation, metrics and conformal calibration mirror core.py.
(() => {
  const LETTERS = "ABCDEFGHIJ";
  const json = path => fetch(path).then(r => {
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r.json();
  });
  const once = {};
  const cached = path => (once[path] ||= json(path));
  const answersFor = model => (once[model] ||= json(`data/answers-${model}.json`).catch(() => ({answers: {}, prompt: null})));
  const calibrations = {};
  let loaded = null;

  const key = (q, state) => JSON.stringify([q.type, state || "", q.question, q.options, q.none_option]);
  const allOptions = (q, cfg) => q.type === "yesno" ? ["Sì", "No"] : q.type === "choice" && q.none_option ? [...q.options, cfg.none_option] : [...q.options];
  const calKey = b => `${b.model}|${b.prompt_style}|s${b.scramble}|${b.scramble > 1 ? b.aggregation : "single"}${b.set_attention ? "|set" : ""}`;

  function aggregate(vp, method, mass) {
    const n = vp[0].length, mean = j => vp.reduce((a, v) => a + v[j], 0) / vp.length;
    if (method === "mean") return vp[0].map((_, j) => mean(j));
    if (method === "weighted") {
      const w = mass.map(m => Math.max(m, 1e-6)), tot = w.reduce((a, b) => a + b, 0);
      return vp[0].map((_, j) => vp.reduce((a, v, i) => a + w[i] * v[j], 0) / tot);
    }
    if (method === "geomean") {
      const logs = vp[0].map((_, j) => vp.reduce((a, v) => a + Math.log(Math.max(v[j], 1e-12)), 0) / vp.length);
      const m = Math.max(...logs), e = logs.map(l => Math.exp(l - m)), s = e.reduce((a, b) => a + b, 0);
      return e.map(x => x / s);
    }
    const votes = Array(n).fill(0);
    vp.forEach(v => votes[v.indexOf(Math.max(...v))]++);
    const scores = votes.map((c, j) => c + 1e-3 * mean(j)), s = scores.reduce((a, b) => a + b, 0);
    return scores.map(x => x / s);
  }

  function rng(seed) {
    let h = 2166136261;
    for (const c of seed) h = Math.imul(h ^ c.charCodeAt(0), 16777619);
    return () => {
      h = Math.imul(h ^ (h >>> 15), 2246822507);
      h = Math.imul(h ^ (h >>> 13), 3266489909);
      return ((h ^= h >>> 16) >>> 0) / 4294967296;
    };
  }
  const softmax = xs => { const m = Math.max(...xs), e = xs.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };

  function simulated(seed, n, k) {
    const rand = rng(seed), gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
    const base = Array.from({length: n}, () => 1.6 * gauss());
    const perms = n > 4 ? k : Math.min(k, [1, 1, 2, 6, 24][n]);
    return Array.from({length: perms}, (_, i) => {
      const order = Array.from({length: n}, (_, j) => (j + i) % n);
      const probs = softmax(base.map(b => b + 0.6 * gauss()));
      const position = order.map(o => probs[o]);
      return [order, position, probs, 0.97 + 0.03 * rand(), " " + LETTERS[position.indexOf(Math.max(...position))]];
    });
  }

  function permutations(n, k, seed) {
    const base = [...Array(n).keys()], perms = [];
    for (let s = 0; s < Math.min(k, n); s++) perms.push([...base.slice(s), ...base.slice(0, s)]);
    const rand = rng(String(seed)), seen = new Set(perms.map(p => p.join()));
    const fact = [...Array(n).keys()].reduce((a, i) => a * (i + 1), 1);
    while (perms.length < k && seen.size < fact) {
      const p = [...base];
      for (let i = n - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; }
      if (!seen.has(p.join())) { seen.add(p.join()); perms.push(p); }
    }
    return perms;
  }

  // The template's own placeholders are filled first and the user's text last, through functions, so "$" patterns or
  // an "(A o B)" written in a question stay as typed.
  function buildPrompt(tpl, question, options, state) {
    const l = LETTERS.slice(0, options.length).split("");
    const ctx = state ? `Contesto:\n${state}\n\n` : "";
    const [head, rest] = tpl.split("@@Q@@");
    const tail = rest.replace("(A o B)", () => `(${l.slice(0, -1).join(", ")} o ${l[l.length - 1]})`);
    const [before, after] = tail.split("A) @@O1@@\nB) @@O2@@");
    return head + ctx + question + before + options.map((o, i) => `${l[i]}) ${o}`).join("\n") + after;
  }
  const DEFAULT_TEMPLATE = "@@Q@@\n\nA) @@O1@@\nB) @@O2@@\n\nQuale opzione è corretta? Rispondi con una sola lettera (A o B), senza altro testo né formattazione.\nRisposta:";

  // Real answers computed in the browser: one forward pass per permutation, as core.Engine.run_variants.
  async function browserVariants(q, state, opts, k, qi, tpl) {
    const vs = [];
    for (const order of permutations(opts.length, k, qi)) {
      const r = await window.browserEngine.letters(buildPrompt(tpl, q.question, order.map(o => opts[o]), state), opts.length);
      const probs = Array(opts.length);
      order.forEach((o, pos) => { probs[o] = r.probs[pos]; });
      vs.push([order, r.probs, probs, r.mass, r.top]);
    }
    return vs;
  }

  function item(q, vs, method, cal, cfg) {
    const opts = allOptions(q, cfg);
    const vp = vs.map(v => v[2]), mass = vs.map(v => v[3]);
    const final = vs.length > 1 ? aggregate(vp, method, mass) : vp[0];
    const pred = final.indexOf(Math.max(...final));
    const argmaxes = vp.map(v => v.indexOf(Math.max(...v)));
    const r = {
      id: q.id, type: q.type, question: q.question, options: opts, probs: final, argmax: LETTERS[pred],
      confidence: final[pred], entropy: -final.reduce((a, p) => a + p * Math.log(Math.max(p, 1e-12)), 0),
      consistency: argmaxes.filter(a => a === pred).length / vs.length,
      letter_mass: mass.reduce((a, b) => a + b, 0) / vs.length,
      answer: q.answer || null, correct: q.answer ? q.answer === LETTERS[pred] : null,
      variants: vs.map(v => ({order: v[0], position_probs: v[1], probs: v[2], letter_mass: v[3], unconstrained_top: v[4]})),
    };
    if (q.type === "score") {
      r.score = final.reduce((a, p, i) => a + p * (i + 1), 0);
      r.score_norm = (r.score - 1) / (final.length - 1);
    } else if (q.type === "yesno") r.p_yes = final[0];
    if (cal) {
      const set = final.map((p, i) => p >= 1 - cal.qhat ? LETTERS[i] : null).filter(Boolean);
      r.prediction_set = set.length ? set : [LETTERS[pred]];
      r.decided = r.prediction_set.length === 1;
    }
    return r;
  }

  function metrics(graded, bins = 10) {
    if (!graded.length) return {ece: null, brier: null, nll: null, reliability: []};
    let ece = 0, brier = 0, nll = 0;
    const reliability = [];
    for (let b = 0; b < bins; b++) {
      const lo = b / bins, hi = (b + 1) / bins;
      const rows = graded.filter(r => (lo < r.confidence && r.confidence <= hi) || (b === 0 && r.confidence === 0));
      if (!rows.length) continue;
      const conf = rows.reduce((a, r) => a + r.confidence, 0) / rows.length;
      const acc = rows.filter(r => r.correct).length / rows.length;
      ece += rows.length / graded.length * Math.abs(acc - conf);
      reliability.push({lo, hi, confidence: conf, accuracy: acc, n: rows.length});
    }
    for (const r of graded) {
      const t = LETTERS.indexOf(r.answer);
      brier += r.probs.reduce((a, p, i) => a + (p - (i === t ? 1 : 0)) ** 2, 0);
      nll -= Math.log(Math.max(r.probs[t], 1e-12));
    }
    return {ece, brier: brier / graded.length, nll: nll / graded.length, reliability};
  }

  async function score(b, cfg) {
    const spec = cfg.models.find(m => m.key === b.model);
    if (spec.browser) return browserScore(b, cfg, spec);
    const data = await answersFor(b.model);
    const decision = spec.kind === "decision";
    const exact = decision || (b.prompt_style === cfg.default_prompt && !b.set_attention);
    const method = b.scramble > 1 ? b.aggregation : "mean";
    const cal = calibrations[calKey(b)];
    const recorded = Object.values(data.answers);
    const avgMs = recorded.length ? recorded.reduce((a, r) => a + r.t, 0) / recorded.length : 50;
    let forward = 0, prompts = 0;
    const posSum = [], posCnt = [];
    const results = b.questions.map((q, i) => {
      q = {...q, id: q.id ?? String(i), options: q.options || []};
      const state = (q.state || b.state || "").trim();
      const rec = exact ? data.answers[key(q, state)] : null;
      const n = allOptions(q, cfg).length;
      const k = decision ? 1 : b.scramble;
      const vs = rec ? rec.v.slice(0, k) : simulated(`${b.model}|${key(q, state)}`, n, k);
      forward += vs.length * (rec ? rec.t : avgMs) / 1000;
      prompts += vs.length;
      vs.forEach(v => v[1].forEach((p, j) => { posSum[j] = (posSum[j] || 0) + p; posCnt[j] = (posCnt[j] || 0) + 1; }));
      return Object.assign(item(q, vs, method, cal, cfg), rec ? {} : {simulated: true});
    });
    const tpl = data.prompt || DEFAULT_TEMPLATE;
    return finish(b, cfg, spec, results, prompts, forward, posSum, posCnt, cal, tpl,
      {loaded: b.model, label: spec.label, repo: spec.repo, device: "demo", vram_gb: null});
  }

  function finish(b, cfg, spec, results, prompts, forward, posSum, posCnt, cal, tpl, model) {
    const graded = results.filter(r => r.answer);
    const summary = {
      questions: results.length, graded: graded.length,
      accuracy: graded.length ? graded.filter(r => r.correct).length / graded.length : null,
      ...metrics(graded),
      avg_letter_mass: results.reduce((a, r) => a + r.letter_mass, 0) / results.length,
      avg_confidence: results.reduce((a, r) => a + r.confidence, 0) / results.length,
      avg_consistency: results.reduce((a, r) => a + r.consistency, 0) / results.length,
      prompts, forward_s: forward, total_s: forward, prompts_per_s: prompts / forward,
      position_bias: posSum.map((s, j) => s / posCnt[j]), tokens_saved: 0, peak_vram_gb: 0,
    };
    if (cal) {
      const decided = results.filter(r => r.decided), dg = decided.filter(r => r.answer);
      summary.conformal = {
        epsilon: cal.epsilon, qhat: cal.qhat, threshold: 1 - cal.qhat, calibration_n: cal.n,
        decided_rate: decided.length / results.length,
        avg_set_size: results.reduce((a, r) => a + r.prediction_set.length, 0) / results.length,
        coverage: graded.length ? graded.filter(r => r.prediction_set.includes(r.answer)).length / graded.length : null,
        accuracy_when_decided: dg.length ? dg.filter(r => r.correct).length / dg.length : null,
      };
    }
    const q0 = b.questions[0], opts0 = allOptions({...q0, options: q0.options || []}, cfg);
    const sample = buildPrompt(tpl, q0.question, opts0, (q0.state || b.state || "").trim());
    return {summary, results, sample_prompt: sample, model};
  }

  const BROWSER_MODELS = [{key: "browser", label: "LFM2.5 1.2B nel browser", quant: "q8"}];
  async function configWithBrowser() {
    const cfg = await cached("data/config.json");
    if (!window.browserEngine || cfg.models.some(m => m.browser)) return cfg;
    cfg.models.push(...BROWSER_MODELS.map(m => ({...m, repo: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX", params_b: 1.17, browser: true})));
    return cfg;
  }

  async function browserScore(b, cfg, spec) {
    if (window.browserEngine.info().loaded !== b.model) await window.browserEngine.load(b.model);
    const tpl = (await answersFor(window.browserEngine.RECORDED)).prompt || DEFAULT_TEMPLATE;
    const method = b.scramble > 1 ? b.aggregation : "mean";
    const cal = calibrations[calKey(b)];
    const t0 = performance.now();
    let prompts = 0;
    const posSum = [], posCnt = [], results = [];
    for (const [i, raw] of b.questions.entries()) {
      const q = {...raw, id: raw.id ?? String(i), options: raw.options || []};
      const opts = allOptions(q, cfg);
      const vs = await browserVariants(q, (q.state || b.state || "").trim(), opts, b.scramble, i, tpl);
      prompts += vs.length;
      vs.forEach(v => v[1].forEach((p, j) => { posSum[j] = (posSum[j] || 0) + p; posCnt[j] = (posCnt[j] || 0) + 1; }));
      results.push(item(q, vs, method, cal, cfg));
      window.browserEngine.progress?.(i + 1, b.questions.length);
    }
    const forward = (performance.now() - t0) / 1000;
    return finish(b, cfg, spec, results, prompts, forward, posSum, posCnt, cal, tpl,
      {...window.browserEngine.info(), label: spec.label});
  }

  window.demoApi = async (path, body) => {
    const cfg = await configWithBrowser();
    if (path === "/api/config") {
      const browserGone = cfg.models.find(m => m.key === loaded)?.browser && !window.browserEngine.info().loaded;
      return {...cfg, engine: loaded && !browserGone ? {loaded, device: "demo"} : {}, calibrations: {...calibrations}};
    }
    if (path === "/api/bench") return cached("data/bench.json");
    if (path === "/api/test-questions") return cached("data/test_questions.json");
    if (path === "/api/examples") return cached("data/examples.json");
    if (path === "/api/load" && cfg.models.find(m => m.key === body.model)?.browser) {
      const r = await window.browserEngine.load(body.model);
      loaded = body.model;
      return {...r, label: cfg.models.find(m => m.key === body.model).label};
    }
    if (path === "/api/load") {
      await answersFor(body.model);
      loaded = body.model;
      const spec = cfg.models.find(m => m.key === body.model);
      return {loaded, label: spec.label, repo: spec.repo, device: "demo", vram_gb: null, load_seconds: 0};
    }
    if (path === "/api/score") {
      loaded = body.model;
      return score(body, cfg);
    }
    if (path === "/api/calibrate") {
      if (body.source !== "questions") throw new Error("Nella demo statica si calibra solo sulle domande scelte: Global-MMLU richiede il server.");
      const qs = (body.questions || []).filter(q => q.answer);
      if (qs.length < 10) throw new Error("Servono almeno 10 domande con risposta nota per calibrare");
      const k = calKey(body);
      delete calibrations[k];
      const rep = await score({...body, questions: qs}, cfg);
      const scores = rep.results.map(r => 1 - r.probs[LETTERS.indexOf(r.answer)]).sort((a, b) => a - b);
      const rank = Math.ceil((scores.length + 1) * (1 - body.epsilon));
      const qhat = rank > scores.length ? 1 : scores[rank - 1];
      calibrations[k] = {epsilon: body.epsilon, qhat, threshold: 1 - qhat, n: scores.length, key: k};
      const calibrated = await score({...body, questions: qs}, cfg);
      return {calibration: calibrations[k], summary: calibrated.summary, forward_s: calibrated.summary.forward_s};
    }
    throw new Error(`Non disponibile nella demo: ${path}`);
  };
})();
