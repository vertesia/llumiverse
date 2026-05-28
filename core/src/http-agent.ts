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

export function resolveDriverHttpTimeouts(opts?: HttpTimeoutOptions): Required<HttpTimeoutOptions> {
    return {
        headersTimeout: opts?.headersTimeout ?? DEFAULT_DRIVER_HTTP_TIMEOUTS.headersTimeout,
        bodyTimeout: opts?.bodyTimeout ?? DEFAULT_DRIVER_HTTP_TIMEOUTS.bodyTimeout,
        connectTimeout: opts?.connectTimeout ?? DEFAULT_DRIVER_HTTP_TIMEOUTS.connectTimeout,
        keepAliveTimeout: opts?.keepAliveTimeout ?? DEFAULT_DRIVER_HTTP_TIMEOUTS.keepAliveTimeout,
    };
}

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
    const timeouts = resolveDriverHttpTimeouts(opts);
    return new Agent({
        headersTimeout: timeouts.headersTimeout,
        bodyTimeout: timeouts.bodyTimeout,
        connectTimeout: timeouts.connectTimeout,
        keepAliveTimeout: timeouts.keepAliveTimeout,
    });
}

/**
 * Wrap the global `fetch` so every request routes through the given Agent.
 * The returned function is type-compatible with global `fetch`, so it
 * can be passed directly to SDKs that accept a `fetch` option (OpenAI,
 * Anthropic, `@google/genai`, Bedrock via Smithy, …) or used as a
 * drop-in replacement for global `fetch` in drivers that make raw HTTP
 * calls.
 *
 * `@vertesia/api-fetch-client` builds requests with `globalThis.Request`,
 * while the undici package uses its own `Request` class. Calling
 * `undici.fetch` directly can parse a global Request as `"[object Request]"`.
 * Use `globalThis.fetch` (undici-backed in Node/Bun) and pass the undici
 * dispatcher extension through init so the Agent timeout behavior is still
 * applied without crossing Request/Response implementations.
 */
export function createAgentBackedFetch(agent: Agent): typeof fetch {
    return ((input: RequestInfo | URL, init?: RequestInit) => {
        const requestInput = typeof Request !== 'undefined' && input instanceof Request
            ? normalizeRequestInput(input, init)
            : { input, init };

        return globalThis.fetch(requestInput.input, {
            ...(requestInput.init ?? {}),
            dispatcher: agent,
        } as RequestInit & { dispatcher?: unknown });
    }) as unknown as typeof fetch;
}

type NormalizedRequestInput = {
    input: RequestInfo | URL;
    init?: RequestInit & { duplex?: 'half' };
};

function normalizeRequestInput(request: Request, init?: RequestInit): NormalizedRequestInput {
    const overrides = stripUndefinedRequestInit(init);
    const requestInit: RequestInit & { duplex?: 'half' } = {
        method: request.method,
        headers: request.headers,
        body: request.body,
        signal: request.signal,
        cache: request.cache,
        credentials: request.credentials,
        integrity: request.integrity,
        keepalive: request.keepalive,
        mode: request.mode,
        redirect: request.redirect,
        referrer: request.referrer,
        referrerPolicy: request.referrerPolicy,
    };

    if (request.body) {
        requestInit.duplex = 'half';
    }

    return {
        input: request.url,
        init: {
            ...requestInit,
            ...overrides,
        },
    };
}

function stripUndefinedRequestInit(init?: RequestInit): RequestInit | undefined {
    if (!init) {
        return undefined;
    }
    const entries = Object.entries(init).filter(([, value]) => value !== undefined);
    return Object.fromEntries(entries) as RequestInit;
}

/** Re-export the undici `Agent` type so driver code can type its agent
 *  field without adding an extra undici import. */
export type { Agent as DriverHttpAgent } from "undici";
