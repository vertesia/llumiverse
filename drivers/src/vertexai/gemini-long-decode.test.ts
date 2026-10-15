import { createServer, type IncomingMessage } from 'node:http';
import type { AddressInfo } from 'node:net';
import { GoogleGenAI } from '@google/genai';
import { PromptRole } from '@llumiverse/core';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { VertexAIDriver } from './index.js';
import { applyGeminiLongDecodeTimeout } from './models/gemini.js';

const MODEL = 'gemini-4-argon';

// Two long-decode hops: the first stops at the per-request cap with a continuation token, the second finishes.
// Each hop re-reads the prompt, so the second hop's prompt includes the first hop's output.
const HOPS = [
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

async function startLongDecodeServer() {
    const bodies: Record<string, unknown>[] = [];
    const server = createServer((req, res) => {
        void readBody(req).then((body) => {
            bodies.push(body);
            const hop = body.continuationToken === 'tok-1' ? HOPS[1] : HOPS[0];
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

describe('Gemini long decode', () => {
    afterEach(() => vi.restoreAllMocks());

    it.each(['execute', 'stream'] as const)('continues across requests on %s', async (mode) => {
        const { bodies, port, close } = await startLongDecodeServer();
        const warn = vi.fn();
        const logger = { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() };
        const driver = new VertexAIDriver({ project: 'test', region: 'global', geminiContextCache: false, logger });
        const client = new GoogleGenAI({
            vertexai: true,
            apiKey: 'test',
            httpOptions: {
                ...(driver as unknown as { getGoogleGenAIHttpOptions(): object }).getGoogleGenAIHttpOptions(),
                baseUrl: `http://127.0.0.1:${port}`,
            },
        });
        vi.spyOn(driver, 'getGoogleGenAIClient').mockReturnValue(client);
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
});

describe('applyGeminiLongDecodeTimeout', () => {
    const driver = { getRequestTimeoutMs: () => 15 * 60_000 };

    it('raises the default deadline to cover one 20-minute long-decode request', () => {
        const payload = { model: MODEL, contents: [], config: { maxOutputTokens: 65_537 } };
        applyGeminiLongDecodeTimeout(driver, {}, payload);
        expect(payload.config).toEqual({ maxOutputTokens: 65_537, httpOptions: { timeout: 21 * 60_000 } });
    });

    it.each([
        ['no max_tokens', {}, {}],
        ['a single-request max_tokens', { maxOutputTokens: 65_536 }, {}],
        ['a caller-supplied timeout', { maxOutputTokens: 200_000 }, { httpTimeout: { headersTimeout: 1_000 } }],
    ])('leaves the request unchanged with %s', (_case, config, options) => {
        const payload = { model: MODEL, contents: [], config: { ...config } };
        applyGeminiLongDecodeTimeout(driver, options, payload);
        expect(payload.config).toEqual(config);
    });

    it('keeps a driver default that is already long enough', () => {
        const payload = { model: MODEL, contents: [], config: { maxOutputTokens: 200_000 } };
        applyGeminiLongDecodeTimeout({ getRequestTimeoutMs: () => 30 * 60_000 }, {}, payload);
        expect(payload.config).toEqual({ maxOutputTokens: 200_000 });
    });
});
