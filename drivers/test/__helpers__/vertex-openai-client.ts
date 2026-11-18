import type { ServerSentEvent } from '@vertesia/api-fetch-client';
import type { VertexAIDriver } from '../../src/vertexai/index.js';

/** The call shape of the Vertex fetch client the Chat Completions tests were written against. */
type LegacyClient = {
    post(path: string, options: { payload: Record<string, unknown>; reader?: string }): Promise<unknown>;
};

export interface LegacyVertexDriverStub {
    getFetchClient?: () => LegacyClient;
    getFetchClientForRegion?: (region: string, apiVersion?: string, endpointRegion?: string) => LegacyClient;
    logger?: unknown;
    /** The driver's configured region, used when a model path names no location. */
    region?: string;
}

/** Replays an SSE fixture (events whose `data` is a JSON chunk) as the chunk stream the OpenAI SDK returns. */
function sseToChunkStream(events: unknown) {
    const reader = (events as ReadableStream<ServerSentEvent>).getReader();
    return {
        controller: { abort() {} },
        async *[Symbol.asyncIterator]() {
            for (;;) {
                const { done, value } = await reader.read();
                if (done) return;
                if (!value || !('data' in value) || !value.data || value.data === '[DONE]') continue;
                yield JSON.parse(value.data);
            }
        },
    };
}

/**
 * A Vertex driver stub whose OpenAI SDK client seam (`getOpenAIClientForRegion`) records each request through the
 * fetch-client stubs below, as `post(path, { payload })`, so request assertions keep reading the body sent.
 */
export function withVertexOpenAIClient(stub: LegacyVertexDriverStub): VertexAIDriver {
    const legacyClient = (region: string, apiVersion?: string, endpointRegion?: string): LegacyClient => {
        if (stub.getFetchClientForRegion) {
            return endpointRegion
                ? stub.getFetchClientForRegion(region, apiVersion, endpointRegion)
                : stub.getFetchClientForRegion(region, apiVersion);
        }
        if (stub.getFetchClient) return stub.getFetchClient();
        throw new Error('The driver stub has no fetch client');
    };
    const send = async (client: LegacyClient, path: string, body: Record<string, unknown>, stream: boolean) =>
        stream
            ? sseToChunkStream(await client.post(path, { payload: body, reader: 'sse' }))
            : client.post(path, { payload: body });

    return {
        logger: stub.logger,
        options: { region: stub.region ?? 'us-central1', project: 'test-project' },
        getRequestTimeoutMs: () => 60_000,
        getOpenAIClientForRegion: (region: string, apiVersion?: string, endpointRegion?: string) => {
            const client = legacyClient(region, apiVersion, endpointRegion);
            return {
                chat: {
                    completions: {
                        create: (body: Record<string, unknown>) =>
                            send(client, 'endpoints/openapi/chat/completions', body, body.stream === true),
                    },
                },
                post: (path: string, options: { body: Record<string, unknown>; stream?: boolean }) =>
                    send(client, path.replace(/^\//, ''), options.body, options.stream === true),
            };
        },
    } as unknown as VertexAIDriver;
}
