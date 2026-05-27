import type { HttpTimeoutOptions } from "@llumiverse/common";
import { Agent } from "undici";

/**
 * Default HTTP timeouts used by {@link createDriverHttpAgent} when the
 * caller does not override them. These are deliberately tighter than
 * Node's undici default (5 minutes for headers and body) so a hung
 * upstream LLM-provider call surfaces in seconds rather than blocking
 * the whole request budget.
 *
 * Drivers with workloads that have legitimate long pauses (image
 * generation, tool-using agent streams) should pass higher overrides
 * via {@link HttpTimeoutOptions} rather than relying on the defaults.
 */
export const DEFAULT_DRIVER_HTTP_TIMEOUTS: Required<HttpTimeoutOptions> = {
    headersTimeout: 60_000,
    bodyTimeout: 60_000,
    connectTimeout: 10_000,
    keepAliveTimeout: 30_000,
};

/**
 * Build an undici `Agent` configured from {@link HttpTimeoutOptions},
 * falling back to {@link DEFAULT_DRIVER_HTTP_TIMEOUTS} for unset fields.
 *
 * The Agent pools sockets — reuse it for the lifetime of the driver
 * and close it on `destroy()`.
 *
 * Node-only. `@llumiverse/drivers` is itself Node-only in practice
 * because vertexai/bedrock pull `@google-cloud/*` / `@aws-sdk/*`, so
 * depending on undici from core is acceptable.
 */
export function createDriverHttpAgent(opts?: HttpTimeoutOptions): Agent {
    return new Agent({
        headersTimeout:   opts?.headersTimeout   ?? DEFAULT_DRIVER_HTTP_TIMEOUTS.headersTimeout,
        bodyTimeout:      opts?.bodyTimeout      ?? DEFAULT_DRIVER_HTTP_TIMEOUTS.bodyTimeout,
        connectTimeout:   opts?.connectTimeout   ?? DEFAULT_DRIVER_HTTP_TIMEOUTS.connectTimeout,
        keepAliveTimeout: opts?.keepAliveTimeout ?? DEFAULT_DRIVER_HTTP_TIMEOUTS.keepAliveTimeout,
    });
}

/**
 * Wrap the global `fetch` so every request routes through the given
 * undici Agent. The returned function is type-compatible with global
 * `fetch`, so it can be passed directly to SDKs that accept a `fetch`
 * option (OpenAI, Anthropic, `@google/genai`, Bedrock via Smithy, …)
 * or used as a drop-in replacement for the global `fetch` in drivers
 * that make raw HTTP calls.
 *
 * Goes through `globalThis.fetch` (which is undici under the hood in
 * Node 18+ / Bun) rather than calling `undici.fetch` directly, because
 * the undici package's exported `fetch` uses its own internal `Request`
 * class — if a caller constructs a `Request` via `globalThis.Request`
 * and passes it through, the `instanceof` check inside `undici.fetch`
 * fails and the input gets coerced to the string `"[object Request]"`,
 * producing an `Invalid URL` error. Routing through `globalThis.fetch`
 * keeps everything on the same Request class while still honoring the
 * undici-specific `dispatcher` init field (passed straight through).
 */
export function createAgentBackedFetch(agent: Agent): typeof fetch {
    return ((input: RequestInfo | URL, init?: RequestInit) =>
        globalThis.fetch(input, {
            ...(init ?? {}),
            // `dispatcher` is an undici-specific extension to RequestInit
            // that Node's wrapper passes straight through.
            dispatcher: agent,
        } as RequestInit & { dispatcher?: unknown })
    ) as typeof fetch;
}

/** Re-export the undici `Agent` type so driver code can type its agent
 *  field without adding an extra undici import. */
export type { Agent as DriverHttpAgent } from "undici";
