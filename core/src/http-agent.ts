import type { HttpTimeoutOptions } from "@llumiverse/common";
import { Agent, fetch as undiciFetch } from "undici";

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
 * Wrap `undici.fetch` so every request routes through the given Agent.
 * The returned function is type-compatible with the global `fetch`, so
 * it can be passed directly to SDKs that accept a `fetch` option
 * (OpenAI, Anthropic, `@google/genai`, Bedrock via Smithy, …) or used
 * as a drop-in replacement for the global `fetch` in drivers that
 * make raw HTTP calls.
 */
export function createAgentBackedFetch(agent: Agent): typeof fetch {
    return ((input: RequestInfo | URL, init?: RequestInit) =>
        undiciFetch(input as Parameters<typeof undiciFetch>[0], {
            ...init,
            dispatcher: agent,
        } as Parameters<typeof undiciFetch>[1])
    ) as unknown as typeof fetch;
}

/** Re-export the undici `Agent` type so driver code can type its agent
 *  field without adding an extra undici import. */
export type { Agent as DriverHttpAgent } from "undici";
