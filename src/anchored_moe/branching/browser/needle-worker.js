/* Classic worker hosting one official, process-global Needle WASM instance. */
let engine;
let modelPointer;
let maxSequenceLength;
const utf8 = new TextEncoder();

function allocate(bytes, nul = false) {
  const pointer = engine._malloc(bytes.length + Number(nul));
  if (!pointer) throw new Error("Needle allocation failed");
  engine.HEAPU8.set(bytes, pointer);
  if (nul) engine.HEAPU8[pointer + bytes.length] = 0;
  return pointer;
}

self.onmessage = async ({ data }) => {
  try {
    if (data.operation === "load") {
      if (engine) throw new Error("Worker already owns a model");
      importScripts(data.engineUrl);
      const response = await fetch(data.wasmUrl);
      if (!response.ok) throw new Error(`WASM download failed: ${response.status}`);
      engine = await createNeedle({ wasmBinary: new Uint8Array(await response.arrayBuffer()) });
      const bytes = new Uint8Array(data.model);
      if (bytes.length < 196 || new DataView(bytes.buffer).getUint32(0, true) !== 0x05e12a84) {
        throw new Error("Expected a pinned Needle 3 archive");
      }
      maxSequenceLength = new DataView(bytes.buffer).getUint32(52, true);
      modelPointer = allocate(bytes);
      // Retain archive memory for the lifetime of this worker; native may view it.
      if (engine._needle_load(modelPointer, BigInt(bytes.length)) < 0) throw new Error("needle_load failed");
      self.postMessage({ id: data.id, result: { loaded: true } });
    } else if (data.operation === "complete") {
      if (!engine) throw new Error("Worker has no model");
      const pointers = [];
      try {
        const string = value => {
          if (typeof value !== "string" || value.includes("\0") || !value.isWellFormed()) throw new Error("Invalid native string");
          const pointer = allocate(utf8.encode(value), true);
          pointers.push(pointer);
          return pointer;
        };
        const system = string(data.system);
        const tools = string(typeof data.tools === "string" ? data.tools : JSON.stringify(data.tools));
        const input = string(data.input);
        const capacity = 1024 * 1024;
        const output = engine._malloc(capacity);
        if (!output) throw new Error("Output allocation failed");
        pointers.push(output);
        engine._needle_reset();
        const prefixTokens = engine._needle_init(system, tools, 0);
        if (prefixTokens < 0) throw new Error("needle_init failed");
        // The public API has no tokenizer/count endpoint. Escape spaces as the
        // pinned BPE tokenizer does, then bound tokens by UTF-8 bytes; reserve
        // 32 more for native turn framing/dummy-prefix tokens.
        // Reject conservatively rather than letting native silently trim history.
        const inputTokenBound = utf8.encode(data.input.replaceAll(" ", "\u2581")).length;
        if (prefixTokens + inputTokenBound + 32 + data.maxNewTokens > maxSequenceLength) {
          throw new Error("Request plus generation reserve exceeds native context budget");
        }
        const code = engine._needle_complete(input, data.maxNewTokens, output, capacity);
        if (code < 0) throw new Error(`needle_complete failed (${code})`);
        const result = JSON.parse(engine.UTF8ToString(output));
        self.postMessage({ id: data.id, result });
      } finally { for (const pointer of pointers) engine._free(pointer); }
    } else { throw new Error("Unknown worker operation"); }
  } catch (error) {
    self.postMessage({ id: data.id, error: String(error.message || error) });
  }
};
