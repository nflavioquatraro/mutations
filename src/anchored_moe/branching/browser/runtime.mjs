/** Composable context router and worker backend for official Needle execution. */
import { renderRoutingRequest, parseGenerativeRoute } from "./bundle.mjs";

class WorkerSession {
  constructor(worker) {
    this.worker = worker;
    this.pending = new Map();
    this.sequence = 0;
    this.closed = false;
    worker.onmessage = ({ data }) => {
      const pending = this.pending.get(data.id);
      if (!pending) return;
      this.pending.delete(data.id);
      clearTimeout(pending.timer);
      if (data.error) pending.reject(new Error(data.error));
      else pending.resolve(data.result);
    };
    worker.onerror = event => this.close(new Error(event.message || "Needle worker failed"));
    worker.onmessageerror = () => this.close(new Error("Needle worker protocol failed"));
  }

  call(message, timeoutMs, transfer = []) {
    if (this.closed) return Promise.reject(new Error("Worker is closed"));
    return new Promise((resolve, reject) => {
      const id = ++this.sequence;
      const timer = setTimeout(() => {
        const error = new Error("Needle worker timed out");
        error.name = "TimeoutError";
        this.close(error);
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      try { this.worker.postMessage({ id, ...message }, transfer); }
      catch (error) { this.close(error); }
    });
  }

  close(error = new Error("Worker closed")) {
    this.closed = true;
    this.worker.terminate();
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
  }
}

/** One worker per resident model, with bounded residency and hard timeouts. */
export class NeedleWorkerBackend {
  constructor(bundle, { engineUrl, wasmUrl, workerUrl = new URL("./needle-worker.js", import.meta.url),
    workerFactory = url => new Worker(url), maxResidentModels = 2, loadTimeoutMs = 60000 } = {}) {
    if (!engineUrl || !Number.isInteger(maxResidentModels) || maxResidentModels < 1 ||
        !Number.isFinite(loadTimeoutMs) || loadTimeoutMs <= 0) throw new Error("Invalid worker backend options");
    this.bundle = bundle;
    this.engineUrl = new URL(engineUrl, globalThis.location?.href).href;
    this.wasmUrl = new URL(wasmUrl || "needle.wasm", this.engineUrl).href;
    this.workerUrl = workerUrl;
    this.workerFactory = workerFactory;
    this.maxResidentModels = maxResidentModels;
    this.loadTimeoutMs = loadTimeoutMs;
    this.sessions = new Map();
    this.busy = false;
    this.closed = false;
  }

  async complete(branchId, { input, system = "", tools = [], maxNewTokens = 128, timeoutMs = 10000 }) {
    if (this.closed || this.busy) throw new Error("Backend is closed or busy");
    if (!Number.isInteger(maxNewTokens) || maxNewTokens <= 0 || !Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      throw new Error("Invalid generation budget");
    }
    this.busy = true;
    try {
      let session = this.sessions.get(branchId);
      if (session?.closed) { this.sessions.delete(branchId); session = null; }
      if (!session) {
        // Bound downloads and reconstruction as well as worker initialization.
        let timer;
        const bytes = await Promise.race([
          this.bundle.modelBytes(branchId),
          new Promise((_, reject) => { timer = setTimeout(() => {
            const error = new Error("Model loading timed out"); error.name = "TimeoutError"; reject(error);
          }, this.loadTimeoutMs); }),
        ]).finally(() => clearTimeout(timer));
        if (this.closed) throw new Error("Backend closed during load");
        while (this.sessions.size >= this.maxResidentModels) {
          const oldest = this.sessions.keys().next().value;
          this.sessions.get(oldest).close();
          this.sessions.delete(oldest);
        }
        session = new WorkerSession(this.workerFactory(this.workerUrl));
        this.sessions.set(branchId, session);
        await session.call({ operation: "load", engineUrl: this.engineUrl, wasmUrl: this.wasmUrl,
          model: bytes.buffer }, this.loadTimeoutMs, [bytes.buffer]);
      }
      this.sessions.delete(branchId);
      this.sessions.set(branchId, session);
      return await session.call({ operation: "complete", input, system, tools, maxNewTokens }, timeoutMs);
    } catch (error) {
      this.sessions.get(branchId)?.close();
      this.sessions.delete(branchId);
      throw error;
    } finally { this.busy = false; }
  }

  dispose() {
    this.closed = true;
    for (const session of this.sessions.values()) session.close();
    this.sessions.clear();
  }
}

/** Replace this strategy independently from model transport or execution. */
export class GenerativeToolRouter {
  constructor(config) { this.config = config; }

  async route(request, history, backend) {
    const config = this.config;
    try {
      const input = renderRoutingRequest(config, request, history);
      const result = await backend.complete(config.model_id, {
        input, system: config.system_prompt, tools: config.tools_json || config.tools,
        maxNewTokens: config.max_new_tokens, timeoutMs: config.timeout_ms,
      });
      return parseGenerativeRoute(result, config);
    } catch (error) {
      return { branch_id: config.fallback_branch,
        reason: error.name === "TimeoutError" ? "router_timeout" : "router_failure" };
    }
  }
}

/** Stateless requests with explicit context; tool calls are returned, never executed. */
export class BrowserBranchedModel {
  constructor(bundle, backend, { router = new GenerativeToolRouter(bundle.manifest.routing) } = {}) {
    this.bundle = bundle;
    this.backend = backend;
    this.router = router;
    this.busy = false;
  }

  async complete(request, { history = [], system = "", tools = [], maxNewTokens = 256, timeoutMs = 30000 } = {}) {
    if (this.busy) throw new Error("Concurrent requests require separate model owners");
    this.busy = true;
    try {
      const config = this.bundle.manifest.routing;
      // Validate once and send the identical bounded envelope to the response.
      const input = renderRoutingRequest(config, request, history);
      let route = await this.router.route(request, history, this.backend);
      if (!config.branch_ids.includes(route?.branch_id)) {
        route = { branch_id: config.fallback_branch, reason: "unknown_route" };
      }
      const response = await this.backend.complete(route.branch_id, { input, system, tools, maxNewTokens, timeoutMs });
      return { route, response };
    } finally { this.busy = false; }
  }

  dispose() { this.backend.dispose(); }
}
