import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { GoogleGenAI } from '@google/genai';
import { PromptRole } from '@llumiverse/core';
import { expect, it, vi } from 'vitest';
import { VertexAIDriver } from './index.js';

it.each(['execute', 'stream'] as const)('applies per-call transport timeouts to Google SDK %s', async (mode) => {
    const server = createServer((req, res) => {
        const timer = setTimeout(() => {
            const data = JSON.stringify({
                candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }],
            });
            const streaming = req.url?.includes('streamGenerateContent');
            res.setHeader('content-type', streaming ? 'text/event-stream' : 'application/json');
            res.end(streaming ? `data: ${data}\n\n` : data);
        }, 1_500);
        res.on('close', () => clearTimeout(timer));
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const driver = new VertexAIDriver({ project: 'test', region: 'global', geminiContextCache: false });
    const client = new GoogleGenAI({
        apiKey: 'test',
        httpOptions: {
            baseUrl: `http://127.0.0.1:${(server.address() as AddressInfo).port}`,
            timeout: 5_000,
        },
    });
    const spy = vi.spyOn(driver, 'getGoogleGenAIClient').mockReturnValue(client);
    const prompt = [{ role: PromptRole.user, content: 'hello' }];
    const execute = async (headersTimeout?: number) => {
        const options = {
            model: 'gemini-3-flash-preview',
            ...(headersTimeout ? { httpTimeout: { headersTimeout } } : {}),
        };
        if (mode === 'execute') return driver.execute(prompt, options);
        const stream = await driver.stream(prompt, options);
        let text = '';
        for await (const chunk of stream) text += chunk;
        return text;
    };
    try {
        const [normal, short] = await Promise.allSettled([execute(), execute(100)]);
        expect(normal.status).toBe('fulfilled');
        expect(short).toMatchObject({ status: 'rejected' });
        if (short.status === 'rejected') {
            expect(String(short.reason)).toContain('fetch failed');
        }
    } finally {
        spy.mockRestore();
        driver.destroy();
        server.closeAllConnections();
        await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
    }
});
