// In-browser engine: LFM2.5 1.2B Instruct (ONNX q8) through Transformers.js on CPU (WASM). Same prompt and letter
// reading as core.py: one forward pass per prompt, softmax over the letter tokens of the last position.
(() => {
  const TJS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";
  const REPO = "LiquidAI/LFM2.5-1.2B-Instruct-ONNX", RECORDED = "lfm2.5-1.2b", SIZE_GB = 1.7;
  const LETTERS = "ABCDEFGH";
  let lib = null, tok = null, model = null, letterIds = null, pending = null;
  const listeners = new Set();
  const emit = state => listeners.forEach(f => f(state));

  function load() {
    if (model) return Promise.resolve(info());
    pending ||= doLoad().finally(() => { pending = null; });
    return pending;
  }

  async function doLoad() {
    lib ||= await import(TJS);
    const files = {}, t0 = performance.now();
    const progress = p => {
      if (p.status === "progress" && p.total) files[p.file] = [p.loaded, p.total];
      const vals = Object.values(files);
      emit({phase: "download", loaded: vals.reduce((a, f) => a + f[0], 0), total: vals.reduce((a, f) => a + f[1], 0)});
    };
    tok = await lib.AutoTokenizer.from_pretrained(REPO, {progress_callback: progress});
    // dtype "q8" would select model_quantized.onnx, whose embedding op (GatherBlockQuantized) WASM lacks
    model = await lib.AutoModelForCausalLM.from_pretrained(REPO, {model_file_name: "model_q8", dtype: "fp32",
      use_external_data_format: true, device: "wasm", progress_callback: progress});
    letterIds = LETTERS.split("").map(L => [...new Set([L, " " + L]
      .map(s => tok.encode(s, {add_special_tokens: false})).filter(t => t.length === 1).map(t => t[0]))]);
    emit({phase: "ready"});
    return {...info(), load_seconds: (performance.now() - t0) / 1000};
  }

  // Letter probabilities of the next token after `prompt`, letter mass and unconstrained top token.
  async function letters(prompt, n) {
    const enc = tok(prompt, {add_special_tokens: false});
    const out = await model({input_ids: enc.input_ids, attention_mask: enc.attention_mask});
    const L = out.logits.data, V = out.logits.dims.at(-1), base = L.length - V;
    let max = -Infinity, top = 0;
    for (let i = 0; i < V; i++) if (L[base + i] > max) { max = L[base + i]; top = i; }
    let z = 0;
    for (let i = 0; i < V; i++) z += Math.exp(L[base + i] - max);
    const p = letterIds.slice(0, n).map(ids => ids.reduce((a, i) => a + Math.exp(L[base + i] - max), 0) / z);
    const mass = p.reduce((a, b) => a + b, 0);
    return {probs: p.map(x => x / mass), mass, top: tok.decode([top])};
  }

  async function storage() {
    const est = await navigator.storage?.estimate?.();
    return {usage_gb: est ? est.usage / 2 ** 30 : null, cached: (await caches.keys()).includes("transformers-cache")};
  }

  async function clear() {
    await model?.dispose?.();
    model = null;
    await caches.delete("transformers-cache");
    emit({phase: "cleared"});
  }

  function info() {
    return {loaded: model ? "browser" : null, label: model ? "LFM2.5 1.2B nel browser (q8)" : null, repo: REPO,
      device: "browser", vram_gb: null};
  }

  window.browserEngine = {load, letters, clear, storage, info, RECORDED, SIZE_GB, onChange: f => listeners.add(f)};
})();
