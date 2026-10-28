/** Exact byte reconstruction and integrity-checked, lazy browser asset loading. */
const MAX_MODEL_BYTES = 512 * 1024 * 1024;
const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: true });

export async function sha256(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), x => x.toString(16).padStart(2, "0")).join("");
}

function integer(value, low, high) {
  return Number.isSafeInteger(value) && value >= low && value <= high;
}

export async function verifyAsset(bytes, metadata) {
  if (!metadata || bytes.byteLength !== metadata.bytes ||
      await sha256(bytes) !== metadata.sha256) throw new Error("Asset integrity mismatch");
  return bytes;
}

export async function applyCactPatch(base, patch) {
  if (!(base instanceof Uint8Array) || !(patch instanceof Uint8Array) ||
      patch.length < 12 || decoder.decode(patch.subarray(0, 8)) !== "NDPATCH1") {
    throw new Error("Invalid Needle patch");
  }
  const headerSize = new DataView(patch.buffer, patch.byteOffset, patch.byteLength).getUint32(8, true);
  if (headerSize > 16 * 1024 * 1024 || headerSize + 12 > patch.length) {
    throw new Error("Invalid patch header size");
  }
  const header = JSON.parse(decoder.decode(patch.subarray(12, 12 + headerSize)));
  const payload = patch.subarray(12 + headerSize);
  if (header.format !== "needle-byte-patch-v1" ||
      !integer(header.target?.bytes, 1, MAX_MODEL_BYTES) ||
      !Array.isArray(header.operations) || !header.operations.length ||
      await sha256(payload) !== header.payload_sha256) throw new Error("Invalid patch metadata");
  await verifyAsset(base, header.base);
  let total = 0;
  for (const operation of header.operations) {
    if (!Array.isArray(operation) || operation.length !== 3) throw new Error("Invalid patch operation");
    const [source, offset, size] = operation;
    const input = source === "base" ? base : source === "patch" ? payload : null;
    if (!input || !integer(offset, 0, input.length) ||
        !integer(size, 1, input.length - offset)) throw new Error("Invalid patch range");
    total += size;
    if (total > header.target.bytes) throw new Error("Patch output overflow");
  }
  if (total !== header.target.bytes) throw new Error("Patch output size mismatch");
  const output = new Uint8Array(total);
  let position = 0;
  for (const [source, offset, size] of header.operations) {
    output.set((source === "base" ? base : payload).subarray(offset, offset + size), position);
    position += size;
  }
  return verifyAsset(output, header.target);
}

/** Storage is optional; failed/quota-limited CacheStorage falls back to fetching. */
export class BrowserBranchBundle {
  static async load(manifestUrl, { manifestSha256, fetcher = fetch, cache = true } = {}) {
    const url = new URL(manifestUrl, globalThis.location?.href);
    const response = await fetcher(url);
    if (!response.ok) throw new Error(`Manifest download failed: ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (manifestSha256 && await sha256(bytes) !== manifestSha256) throw new Error("Manifest integrity mismatch");
    const manifest = JSON.parse(decoder.decode(bytes));
    if (manifest.format !== "needle-browser-branches-v1" ||
        manifest.routing?.strategy !== "generative-tool-v1" ||
        !manifest.models || !manifest.base ||
        !Array.isArray(manifest.routing.branch_ids) ||
        !manifest.routing.branch_ids.includes(manifest.routing.fallback_branch) ||
        !manifest.models[manifest.routing.model_id] ||
        manifest.routing.branch_ids.some(id => !Object.hasOwn(manifest.models, id))) {
      throw new Error("Invalid browser branch manifest");
    }
    const storage = cache && globalThis.caches
      ? await globalThis.caches.open("needle-branch-assets-v1").catch(() => null) : null;
    return new BrowserBranchBundle(url, manifest, fetcher, storage);
  }

  constructor(url, manifest, fetcher, storage) {
    this.url = url;
    this.manifest = manifest;
    this.fetcher = (...args) => fetcher(...args);
    this.storage = storage;
    this.assets = new Map();
  }

  async asset(metadata) {
    if (!metadata || !/^[A-Za-z0-9_.-]+$/.test(metadata.file) ||
        !/^[a-f0-9]{64}$/.test(metadata.sha256) ||
        !integer(metadata.bytes, 1, MAX_MODEL_BYTES + 16 * 1024 * 1024)) {
      throw new Error("Invalid asset metadata");
    }
    const key = metadata.sha256;
    if (!this.assets.has(key)) {
      const promise = (async () => {
        const url = new URL(metadata.file, this.url);
        // Immutable digest keys allow reuse even if another bundle shares the base.
        const cacheUrl = new URL(`.needle-assets/${key}`, this.url.origin);
        const cached = await this.storage?.match(cacheUrl).catch(() => null);
        if (cached) {
          try { return await verifyAsset(new Uint8Array(await cached.arrayBuffer()), metadata); }
          catch { await this.storage.delete(cacheUrl).catch(() => {}); }
        }
        const response = await this.fetcher(url);
        if (!response.ok) throw new Error(`Asset download failed: ${response.status}`);
        const bytes = await verifyAsset(new Uint8Array(await response.arrayBuffer()), metadata);
        await this.storage?.put(cacheUrl, new Response(bytes)).catch(() => {});
        return bytes;
      })();
      this.assets.set(key, promise);
      promise.catch(() => this.assets.delete(key));
    }
    return this.assets.get(key);
  }

  async modelBytes(branchId) {
    if (!Object.hasOwn(this.manifest.models, branchId)) throw new Error("Unknown branch");
    const model = this.manifest.models[branchId];
    const base = await this.asset(this.manifest.base);
    if (model.kind === "base") return (await verifyAsset(base, model.target)).slice();
    if (model.kind !== "patch") throw new Error("Unknown model encoding");
    const patch = await this.asset(model.patch);
    return verifyAsset(await applyCactPatch(base, patch), model.target);
  }
}

function validText(value) {
  if (typeof value !== "string" || value.includes("\0") || !value.isWellFormed()) {
    throw new Error("Expected valid text without NUL characters");
  }
}

/** Same whole-message suffix and UTF-8 budget as GenerativeRouterConfig. */
export function renderRoutingRequest(config, request, history = []) {
  validText(request);
  if (!request.trim() || !Array.isArray(history)) throw new Error("Invalid request/history");
  for (const message of history) {
    if (!message || !["user", "assistant", "tool"].includes(message.role)) throw new Error("Invalid history role");
    validText(message.content);
  }
  if (!integer(config.max_history_messages, 0, 1000000) ||
      !integer(config.max_input_bytes, 1, MAX_MODEL_BYTES)) throw new Error("Invalid context limits");
  const recent = [];
  const render = () => JSON.stringify({ history: recent, request });
  if (encoder.encode(render()).length > config.max_input_bytes) throw new Error("Current request exceeds max_input_bytes");
  const candidates = config.max_history_messages ? history.slice(-config.max_history_messages) : [];
  for (const { role, content } of candidates.reverse()) {
    recent.unshift({ role, content });
    if (encoder.encode(render()).length > config.max_input_bytes) { recent.shift(); break; }
  }
  return render();
}

export function parseGenerativeRoute(response, config) {
  const fallback = { branch_id: config.fallback_branch, reason: "invalid_router_output" };
  try {
    if (typeof response === "string") response = JSON.parse(response);
    if (response?.success === false || response?.error != null) return fallback;
    const calls = response?.function_calls;
    if (!Array.isArray(calls) || calls.length !== 1 || calls[0]?.name !== "select_branch") return fallback;
    const args = calls[0].arguments;
    if (!args || Array.isArray(args) || Object.keys(args).length !== 1 ||
        !Object.hasOwn(args, "branch_id") || !config.branch_ids.includes(args.branch_id)) return fallback;
    return { branch_id: args.branch_id, reason: "generated_route" };
  } catch { return fallback; }
}
