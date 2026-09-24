/** Run with Node >=22 and the temporary fixture directory from pytest. */
import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { runInNewContext } from "node:vm";
import { BrowserBranchBundle, applyCactPatch, renderRoutingRequest, parseGenerativeRoute, sha256 } from "../../src/anchored_moe/branching/browser/bundle.mjs";
import { BrowserBranchedModel, GenerativeToolRouter, NeedleWorkerBackend } from "../../src/anchored_moe/branching/browser/runtime.mjs";

const root = process.argv[2];
const read = async name => new Uint8Array(await readFile(join(root, name)));
const base = await read("base.bin"), patch = await read("patch.bin"), target = await read("target.bin");
const contract = JSON.parse(await readFile(join(root, "contract.json"), "utf8"));
const config = contract.config;
const choice = branch_id => ({ function_calls: [{ name: "select_branch", arguments: { branch_id } }] });

test("Python-generated patches reconstruct exact bytes and detect corruption", async () => {
  assert.deepEqual(await applyCactPatch(base, patch), target);
  const corrupt = patch.slice(); corrupt[corrupt.length - 1] ^= 1;
  await assert.rejects(() => applyCactPatch(base, corrupt));
  await assert.rejects(() => applyCactPatch(base.subarray(1), patch));
  const size = new DataView(patch.buffer).getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(patch.subarray(12, 12 + size)));
  header.operations[0][1] = -1;
  const encoded = new TextEncoder().encode(JSON.stringify(header));
  const malformed = new Uint8Array(12 + encoded.length + patch.length - 12 - size);
  malformed.set(patch.subarray(0, 8)); new DataView(malformed.buffer).setUint32(8, encoded.length, true);
  malformed.set(encoded, 12); malformed.set(patch.subarray(12 + size), 12 + encoded.length);
  await assert.rejects(() => applyCactPatch(base, malformed), /range/);
});

test("browser context equals training context including Unicode and truncation", () => {
  assert.equal(renderRoutingRequest(config, contract.request, contract.history), contract.expected);
  assert.throws(() => renderRoutingRequest(config, "x".repeat(500)), /exceeds/);
  assert.throws(() => renderRoutingRequest(config, "x", [{ role: "system", content: "bad" }]));
  for (const invalid of ["bad", {}, null, [], { suppressed_calls: choice("specialist").function_calls },
    choice("unknown"), { function_calls: [...choice("base").function_calls, ...choice("specialist").function_calls] }]) {
    assert.equal(parseGenerativeRoute(invalid, config).reason, "invalid_router_output");
  }
  assert.equal(parseGenerativeRoute(choice("specialist"), config).branch_id, "specialist");
  assert.equal(parseGenerativeRoute({ ...choice("specialist"), success: false }, config).reason, "invalid_router_output");
});

test("lazy assets download the base once and only the requested branch patch", async () => {
  const baseMeta = { file: "base.cact", bytes: base.length, sha256: await sha256(base) };
  const patchMeta = { file: "specialist.ndpatch", bytes: patch.length, sha256: await sha256(patch) };
  const manifest = { format: "needle-browser-branches-v1", base: baseMeta,
    routing: { ...config, strategy: "generative-tool-v1" }, models: {
      base: { kind: "base", target: baseMeta },
      specialist: { kind: "patch", patch: patchMeta, target: { bytes: target.length, sha256: await sha256(target) } },
      router: { kind: "base", target: baseMeta },
    } };
  const hits = [];
  const files = { "base.cact": base, "specialist.ndpatch": patch, "manifest.json": JSON.stringify(manifest) };
  const fetcher = async function(url) {
    assert.equal(this, undefined, "fetch must not be bound to a bundle object");
    const file = new URL(url).pathname.split("/").pop(); hits.push(file); return new Response(files[file]);
  };
  const bundle = await BrowserBranchBundle.load("https://example.test/models/manifest.json", { fetcher, cache: false });
  const [first, second] = await Promise.all([bundle.modelBytes("specialist"), bundle.modelBytes("base")]);
  assert.deepEqual(first, target); assert.deepEqual(second, base);
  await bundle.modelBytes("specialist");
  assert.deepEqual(hits.sort(), ["base.cact", "manifest.json", "specialist.ndpatch"]);
  first.fill(0); assert.deepEqual(await bundle.modelBytes("specialist"), target);
  await assert.rejects(() => bundle.modelBytes("missing"));
});

test("generated route locks dispatch and router failures choose the base", async () => {
  const calls = [];
  let mode = "valid";
  const backend = { async complete(id, options) {
    calls.push([id, options]);
    if (id !== "router") return { function_calls: [] };
    if (mode === "error") throw new Error("generation failed");
    if (mode === "timeout") { const error = new Error("timed out"); error.name = "TimeoutError"; throw error; }
    return mode === "valid" ? choice("specialist") : choice("unknown");
  }, dispose() {} };
  const bundle = { manifest: { routing: config } };
  const model = new BrowserBranchedModel(bundle, backend);
  for (const [value, selected, reason] of [["valid", "specialist", "generated_route"],
    ["invalid", "base", "invalid_router_output"], ["error", "base", "router_failure"], ["timeout", "base", "router_timeout"]]) {
    mode = value; calls.length = 0;
    const result = await model.complete(contract.request, { history: contract.history });
    assert.deepEqual(result.route, { branch_id: selected, reason });
    assert.deepEqual(calls.map(c => c[0]), ["router", selected]);
    assert.equal(calls[0][1].input, contract.expected);
    assert.equal(calls[1][1].input, contract.expected);
    assert.equal(calls[0][1].tools, config.tools_json);
  }
  const manual = new BrowserBranchedModel(bundle, backend, { router: { route: async () => ({ branch_id: "base", reason: "manual" }) } });
  assert.equal((await manual.complete("hello")).route.reason, "manual");
});

test("worker timeout terminates execution, frees ownership and permits fallback", async () => {
  const workers = [];
  const backend = new NeedleWorkerBackend({ modelBytes: async () => new Uint8Array([1, 2]) }, {
    engineUrl: "https://example.test/needle.js", loadTimeoutMs: 500,
    workerFactory() {
      const worker = { terminated: false, terminate() { this.terminated = true; },
        postMessage(message) {
          if (message.operation === "load") queueMicrotask(() => this.onmessage({ data: { id: message.id, result: {} } }));
          // First model intentionally hangs in synchronous inference.
          else if (workers.length > 1) queueMicrotask(() => this.onmessage({ data: { id: message.id, result: choice("base") } }));
        } };
      workers.push(worker); return worker;
    },
  });
  const router = new GenerativeToolRouter({ ...config, timeout_ms: 10 });
  assert.equal((await router.route("hello", [], backend)).reason, "router_timeout");
  assert.equal(workers[0].terminated, true);
  await backend.complete("base", { input: "hello" });
  backend.dispose(); assert.equal(workers[1].terminated, true);
  await assert.rejects(() => backend.complete("base", { input: "hello" }), /closed/);
});

test("concurrent requests fail without contaminating another request", async () => {
  let release;
  const first = new Promise(resolve => { release = resolve; });
  const backend = { complete: async () => ({}), dispose() {} };
  const model = new BrowserBranchedModel({ manifest: { routing: config } }, backend,
    { router: { route: () => first } });
  const running = model.complete("one");
  await assert.rejects(() => model.complete("two"), /Concurrent/);
  release({ branch_id: "base", reason: "manual" });
  assert.equal((await running).route.branch_id, "base");
});

test("official worker wrapper resets context and rejects overflow before decoding", async () => {
  const messages = [];
  let allocated = 256, decoded = 0, reset = 0;
  const engine = { HEAPU8: new Uint8Array(4 * 1024 * 1024),
    _malloc(size) { const pointer = allocated; allocated += size; return pointer; }, _free() {},
    _needle_load(pointer, size) { assert.equal(typeof size, "bigint"); return 0; },
    _needle_reset() { reset++; }, _needle_init() { return 2; },
    _needle_complete() { decoded++; return 0; }, UTF8ToString() { return JSON.stringify(choice("base")); },
  };
  const scope = { self: { postMessage(message) { messages.push(message); } },
    TextEncoder, Uint8Array, DataView, importScripts() {},
    fetch: async () => new Response(new Uint8Array([1])), createNeedle: async () => engine,
  };
  const source = await readFile(new URL("../../src/anchored_moe/branching/browser/needle-worker.js", import.meta.url), "utf8");
  runInNewContext(source, scope);
  const archive = new Uint8Array(196);
  new DataView(archive.buffer).setUint32(0, 0x05e12a84, true);
  new DataView(archive.buffer).setUint32(52, 64, true);
  await scope.self.onmessage({ data: { id: 1, operation: "load", model: archive.buffer } });
  assert.equal(messages.at(-1).result.loaded, true);
  const request = { operation: "complete", system: "", tools: [], input: "abc", maxNewTokens: 2 };
  await scope.self.onmessage({ data: { ...request, id: 2 } });
  assert.equal(decoded, 1);
  await scope.self.onmessage({ data: { ...request, input: "x".repeat(64), id: 3 } });
  assert.match(messages.at(-1).error, /context budget/);
  assert.equal(decoded, 1);
  assert.equal(reset, 2);
  await scope.self.onmessage({ data: { ...request, input: " ".repeat(12), id: 4 } });
  assert.match(messages.at(-1).error, /context budget/);
  assert.equal(decoded, 1);
});
