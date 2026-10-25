import { createServer, type IncomingMessage } from 'node:http';
import type { AddressInfo } from 'node:net';
import { GoogleGenAI } from '@google/genai';
import { type HttpTimeoutOptions, PromptRole } from '@llumiverse/core';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { VertexAIDriver } from './index.js';
import { applyGeminiLongDecodeStreamTimeout } from './models/gemini.js';

const MODEL = 'gemini-4-argon';

// Two long-decode hops: the first stops at the per-request cap with a continuation token, the second finishes.
// Each hop re-reads the prompt, so the second hop's prompt includes the first hop's output.
interface Hop {
    text: string;
    finishReason: string;
    continuationToken?: string;
    usageMetadata: Record<string, number>;
}

const HOPS: Hop[] = [
    {
        text: 'part one ',
        finishReason: 'CONTINUATION',
        continuationToken: 'tok-1',
        usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 100, totalTokenCount: 110 },
    },
    {
        text: 'part two',
        finishReason: 'STOP',
        usageMetadata: { promptTokenCount: 110, candidatesTokenCount: 50, totalTokenCount: 160 },
    },
];

async function readBody(req: IncomingMessage): Promise<Record<string, unknown>> {
    let body = '';
    for await (const chunk of req) body += chunk;
    return JSON.parse(body) as Record<string, unknown>;
}

async function startLongDecodeServer(hops: Hop[] = HOPS) {
    const bodies: Record<string, unknown>[] = [];
    const server = createServer((req, res) => {
        void readBody(req).then((body) => {
            bodies.push(body);
            const hop = body.continuationToken === 'tok-1' ? hops[1] : hops[0];
            const response = {
                candidates: [
                    {
                        content: { role: 'model', parts: [{ text: hop.text }] },
                        finishReason: hop.finishReason,
                        ...(hop.continuationToken && { continuationToken: hop.continuationToken }),
                    },
                ],
                usageMetadata: hop.usageMetadata,
            };
            if (req.url?.includes('streamGenerateContent')) {
                res.setHeader('content-type', 'text/event-stream');
                res.end(`data: ${JSON.stringify(response)}\n\n`);
            } else {
                res.setHeader('content-type', 'application/json');
                res.end(JSON.stringify(response));
            }
        });
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const close = async () => {
        server.closeAllConnections();
        await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
    };
    return { bodies, port: (server.address() as AddressInfo).port, close };
}

function connectDriver(port: number) {
    const warn = vi.fn();
    const logger = { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() };
    const driver = new VertexAIDriver({ project: 'test', region: 'global', geminiContextCache: false, logger });
    const httpOptions = (
        driver as unknown as { getGoogleGenAIHttpOptions(): { fetch: typeof fetch } }
    ).getGoogleGenAIHttpOptions();
    // Every continuation request must go through the driver fetch, which applies the driver's HTTP timeouts.
    const driverFetch = vi.fn(httpOptions.fetch);
    const client = new GoogleGenAI({
        vertexai: true,
        apiKey: 'test',
        httpOptions: { ...httpOptions, fetch: driverFetch, baseUrl: `http://127.0.0.1:${port}` },
    });
    vi.spyOn(driver, 'getGoogleGenAIClient').mockReturnValue(client);
    return { driver, driverFetch, warn };
}

describe('Gemini long decode', () => {
    afterEach(() => vi.restoreAllMocks());

    it.each(['execute', 'stream'] as const)('continues across requests on %s', async (mode) => {
        const { bodies, port, close } = await startLongDecodeServer();
        const { driver, driverFetch, warn } = connectDriver(port);
        const prompt = [{ role: PromptRole.user, content: 'Write a long proof.' }];
        const options = {
            model: MODEL,
            model_options: { _option_id: 'vertexai-gemini' as const, max_tokens: 200_000 },
        };
        try {
            let text = '';
            let finishReason: string | undefined;
            let tokenUsage: Record<string, number | undefined> | undefined;
            if (mode === 'execute') {
                const response = await driver.execute(prompt, options);
                text = response.result.map((part) => (part.type === 'text' ? part.value : '')).join('');
                finishReason = response.finish_reason;
                tokenUsage = response.token_usage;
            } else {
                const stream = await driver.stream(prompt, options);
                for await (const chunk of stream) text += chunk;
                finishReason = stream.completion?.finish_reason;
                tokenUsage = stream.completion?.token_usage;
            }

            expect(text).toBe('part one part two');
            expect(finishReason).toBe('stop');
            // The SDK sums each request's usage, which is what Vertex bills.
            expect(tokenUsage).toMatchObject({ prompt: 120, result: 150, total: 270 });
            expect(warn).not.toHaveBeenCalled();

            expect(bodies).toHaveLength(2);
            expect(driverFetch).toHaveBeenCalledTimes(2);
            expect(bodies[0]).not.toHaveProperty('continuationToken');
            expect(bodies[1]).toMatchObject({ continuationToken: 'tok-1' });
            // The continuation resends the original request instead of appending the partial output.
            expect(bodies[1].contents).toEqual(bodies[0].contents);
            expect(bodies[0]).toMatchObject({ generationConfig: { maxOutputTokens: 200_000 } });
        } finally {
            driver.destroy();
            await close();
        }
    });

    it.each(['execute', 'stream'] as const)(
        'returns the partial output as length when %s cannot be continued',
        async (mode) => {
            // A CONTINUATION without a token leaves the SDK nothing to resume from.
            const { port, close } = await startLongDecodeServer([
                {
                    text: 'partial',
                    finishReason: 'CONTINUATION',
                    usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 },
                },
            ]);
            const { driver } = connectDriver(port);
            const prompt = [{ role: PromptRole.user, content: 'Write a long proof.' }];
            const options = {
                model: MODEL,
                model_options: { _option_id: 'vertexai-gemini' as const, max_tokens: 200_000 },
            };
            try {
                if (mode === 'execute') {
                    const response = await driver.execute(prompt, options);
                    expect(response.result).toEqual([{ type: 'text', value: 'partial' }]);
                    expect(response.finish_reason).toBe('length');
                } else {
                    const stream = await driver.stream(prompt, options);
                    let text = '';
                    for await (const chunk of stream) text += chunk;
                    expect(text).toBe('partial');
                    expect(stream.completion?.finish_reason).toBe('length');
                }
            } finally {
                driver.destroy();
                await close();
            }
        },
    );
});

describe('applyGeminiLongDecodeStreamTimeout', () => {
    const driver = (httpTimeout?: HttpTimeoutOptions, defaultMs = 15 * 60_000) => ({
        options: { project: 'test', region: 'global', httpTimeout },
        getRequestTimeoutMs: () => defaultMs,
    });

    it('raises the built-in deadline to cover one 20-minute streamed long-decode request', () => {
        const payload = { model: MODEL, contents: [], config: { maxOutputTokens: 65_537 } };
        applyGeminiLongDecodeStreamTimeout(driver(), {}, payload);
        expect(payload.config).toEqual({ maxOutputTokens: 65_537, httpOptions: { timeout: 21 * 60_000 } });
    });

    it('still raises it when the caller only sets connection timeouts', () => {
        const payload = { model: MODEL, contents: [], config: { maxOutputTokens: 200_000 } };
        applyGeminiLongDecodeStreamTimeout(
            driver({ connectTimeout: 5_000 }),
            { httpTimeout: { keepAliveTimeout: 1 } },
            payload,
        );
        expect(payload.config).toEqual({ maxOutputTokens: 200_000, httpOptions: { timeout: 21 * 60_000 } });
    });

    it.each([
        ['no max_tokens', driver(), {}, {}],
        ['a single-request max_tokens', driver(), {}, { maxOutputTokens: 65_536 }],
        [
            'a per-call response timeout',
            driver(),
            { httpTimeout: { headersTimeout: 1_000 } },
            { maxOutputTokens: 200_000 },
        ],
        ['a driver response timeout', driver({ bodyTimeout: 600_000 }), {}, { maxOutputTokens: 200_000 }],
        ['a long enough default', driver(undefined, 30 * 60_000), {}, { maxOutputTokens: 200_000 }],
    ])('leaves the request unchanged with %s', (_case, target, options, config) => {
        const payload = { model: MODEL, contents: [], config: { ...config } };
        applyGeminiLongDecodeStreamTimeout(target, options, payload);
        expect(payload.config).toEqual(config);
    });
});
