// In-browser engine: Qwen3.5 0.8B (ONNX) through Transformers.js. Same prompt and letter reading as core.py:
// one forward pass per prompt, softmax over the letter tokens of the last position.
(() => {
  const TJS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";
  const REPO = "onnx-community/Qwen3.5-0.8B-Text-ONNX";
  const LETTERS = "ABCDEFGH";
  // int8 export: the closest to bf16 (same argmax as the server on 97% of the test questions)
  const DTYPE = "quantized", SIZE_GB = 0.9;
  let lib = null, tok = null, model = null, current = null, device = null, letterIds = null;
  const listeners = new Set();
  const emit = state => listeners.forEach(f => f(state));

  async function pickDevice() {
    try {
      const adapter = navigator.gpu && await navigator.gpu.requestAdapter();
      // a software adapter (SwiftShader) is far slower than WASM
      if (adapter && !adapter.isFallbackAdapter && !adapter.info?.isFallbackAdapter) {
        return "webgpu";
      }
    } catch { /* no WebGPU */ }
    return "wasm";
  }

  let pending = null;
  function load(key) {
    if (current === key && model) return Promise.resolve(info());
    pending ||= doLoad(key).finally(() => { pending = null; });
    return pending;
  }

  async function doLoad(key) {
    await unload();
    lib ||= await import(TJS);
    device ||= await pickDevice();
    const files = {};
    const progress = p => {
      if (p.status === "progress" && p.total) files[p.file] = [p.loaded, p.total];
      const vals = Object.values(files), done = vals.reduce((a, f) => a + f[0], 0), total = vals.reduce((a, f) => a + f[1], 0);
      emit({phase: "download", loaded: done, total, device});
    };
    const t0 = performance.now();
    tok = await lib.AutoTokenizer.from_pretrained(REPO, {progress_callback: progress});
    letterIds = LETTERS.split("").map(L => [...new Set([L, " " + L]
      .map(s => tok.encode(s, {add_special_tokens: false})).filter(t => t.length === 1).map(t => t[0]))]);
    // a WebGPU session can build and still fail on its first run (missing kernels): try one pass, else use WASM
    try {
      model = await lib.AutoModelForCausalLM.from_pretrained(REPO, {dtype: DTYPE, device, progress_callback: progress});
      await letters("A", 2);
    } catch (e) {
      if (device === "wasm") throw e;
      await unload();
      device = "wasm";
      model = await lib.AutoModelForCausalLM.from_pretrained(REPO, {dtype: DTYPE, device, progress_callback: progress});
    }
    current = key;
    emit({phase: "ready", device});
    return {...info(), load_seconds: (performance.now() - t0) / 1000};
  }

  async function unload() {
    if (model) await model.dispose?.();
    model = null; current = null;
  }

  // Letter log-probabilities of the next token after `prompt`: [n] probabilities, letter mass, unconstrained top token.
  async function letters(prompt, n) {
    const enc = tok(prompt, {add_special_tokens: false});
    const out = await model({input_ids: enc.input_ids, attention_mask: enc.attention_mask,
      num_logits_to_keep: new lib.Tensor("int64", [1n], [])});
    const L = out.logits.data, V = out.logits.dims.at(-1), base = L.length - V;
    let max = -Infinity, top = 0;
    for (let i = 0; i < V; i++) if (L[base + i] > max) { max = L[base + i]; top = i; }
    let z = 0;
    for (let i = 0; i < V; i++) z += Math.exp(L[base + i] - max);
    const p = letterIds.slice(0, n).map(ids => ids.reduce((a, i) => a + Math.exp(L[base + i] - max), 0) / z);
    const mass = p.reduce((a, b) => a + b, 0);
    out.logits.dispose?.();
    return {probs: p.map(x => x / mass), mass, top: tok.decode([top])};
  }

  async function storage() {
    const est = await navigator.storage?.estimate?.();
    const cached = (await caches.keys()).includes("transformers-cache");
    return {usage_gb: est ? est.usage / 2 ** 30 : null, cached};
  }

  async function clear() {
    await unload();
    await caches.delete("transformers-cache");
    emit({phase: "cleared"});
  }

  function info() {
    return {loaded: current, label: current ? "Qwen3.5 0.8B nel browser (int8)" : null, repo: REPO,
      device: "browser", backend: device, threads: lib?.env.backends.onnx.wasm.numThreads, vram_gb: null};
  }

  async function plan() {
    device ||= await pickDevice();
    return {backend: device, size_gb: SIZE_GB, isolated: self.crossOriginIsolated};
  }

  window.browserEngine = {load, letters, clear, storage, info, plan, onChange: f => listeners.add(f)};
})();
