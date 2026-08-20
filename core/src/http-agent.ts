import { AsyncLocalStorage } from 'node:async_hooks';
import type { HttpTimeoutOptions } from '@llumiverse/common';
import { Agent } from 'undici';

/**
 * Default HTTP timeouts used by {@link createDriverHttpAgent} when the
 * caller does not override them. Provider response waits deliberately sit
 * beyond the hosting request boundary: application-level cancellation should
 * end user work first, while these limits remain a bounded-resource safety net.
 * Connect and keep-alive limits govern socket establishment and idle reuse,
 * not provider execution. They remain below the response safety horizon but
 * allow for transient network pressure and useful connection reuse.
 */
export const DEFAULT_DRIVER_REQUEST_TIMEOUT_MS = 15 * 60_000;

export const DEFAULT_DRIVER_HTTP_TIMEOUTS: Required<HttpTimeoutOptions> = {
    headersTimeout: DEFAULT_DRIVER_REQUEST_TIMEOUT_MS,
    bodyTimeout: DEFAULT_DRIVER_REQUEST_TIMEOUT_MS,
    connectTimeout: 60_000,
    keepAliveTimeout: 5 * 60_000,
};

export function resolveDriverRequestTimeoutMs(defaults?: HttpTimeoutOptions, override?: HttpTimeoutOptions): number {
    const timeouts = resolveDriverHttpTimeouts(mergeDriverHttpTimeoutOptions(defaults, override));
    return Math.max(timeouts.headersTimeout, timeouts.bodyTimeout);
}

const scopedHttpAgent = new AsyncLocalStorage<Agent>();

export function resolveDriverHttpTimeouts(opts?: HttpTimeoutOptions): Required<HttpTimeoutOptions> {
    return {
        headersTimeout: opts?.headersTimeout ?? DEFAULT_DRIVER_HTTP_TIMEOUTS.headersTimeout,
        bodyTimeout: opts?.bodyTimeout ?? DEFAULT_DRIVER_HTTP_TIMEOUTS.bodyTimeout,
        connectTimeout: opts?.connectTimeout ?? DEFAULT_DRIVER_HTTP_TIMEOUTS.connectTimeout,
        keepAliveTimeout: opts?.keepAliveTimeout ?? DEFAULT_DRIVER_HTTP_TIMEOUTS.keepAliveTimeout,
    };
}

export function mergeDriverHttpTimeoutOptions(
    defaults?: HttpTimeoutOptions,
    override?: HttpTimeoutOptions,
): HttpTimeoutOptions | undefined {
    if (!override) {
        return defaults;
    }
    return {
        ...(defaults ?? {}),
        ...stripUndefinedHttpTimeoutOptions(override),
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
 * `@vertesia/api-fetch-client` builds requests with `globalThis.Request`.
 * Keep that Request intact and pass the undici dispatcher extension through
 * init so the Agent timeout behavior is applied without rebuilding streamed
 * request bodies.
 */
export function createAgentBackedFetch(agent: Agent): typeof fetch {
    return ((input: RequestInfo | URL, init?: RequestInit) => {
        const dispatcher = scopedHttpAgent.getStore() ?? agent;
        return globalThis.fetch(input, {
            ...(stripUndefinedRequestInit(init) ?? {}),
            dispatcher,
        } as RequestInit & { dispatcher?: unknown });
    }) as unknown as typeof fetch;
}

export interface DriverHttpAgentScope {
    run<T>(callback: () => T): T;
    abort(): Promise<void>;
    close(): Promise<void>;
}

const NOOP_HTTP_AGENT_SCOPE: DriverHttpAgentScope = {
    run: <T>(callback: () => T): T => callback(),
    abort: () => Promise.resolve(),
    close: () => Promise.resolve(),
};

export function createDriverHttpAgentScope(
    defaults?: HttpTimeoutOptions,
    override?: HttpTimeoutOptions,
    force = false,
): DriverHttpAgentScope {
    if (!override && !force) {
        return NOOP_HTTP_AGENT_SCOPE;
    }

    const agent = createDriverHttpAgent(mergeDriverHttpTimeoutOptions(defaults, override));
    return {
        run: <T>(callback: () => T): T => scopedHttpAgent.run(agent, callback),
        abort: async () => {
            await agent.destroy().catch(() => {
                /* cancellation best-effort */
            });
        },
        close: async () => {
            await agent.close().catch(() => {
                /* shutdown best-effort */
            });
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

function stripUndefinedHttpTimeoutOptions(opts: HttpTimeoutOptions): HttpTimeoutOptions {
    const entries = Object.entries(opts).filter(([, value]) => value !== undefined);
    return Object.fromEntries(entries) as HttpTimeoutOptions;
}

/** Re-export the undici `Agent` type so driver code can type its agent
 *  field without adding an extra undici import. */
export type { Agent as DriverHttpAgent } from 'undici';
