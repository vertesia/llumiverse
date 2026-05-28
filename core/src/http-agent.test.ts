import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import {
    PromptRole,
    type AIModel,
    type Completion,
    type CompletionChunkObject,
    type DriverOptions,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    type ModelSearchPayload,
} from '@llumiverse/common';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Agent } from 'undici';
import { AbstractDriver } from './Driver.js';
import {
    DEFAULT_DRIVER_HTTP_TIMEOUTS,
    createAgentBackedFetch,
    createDriverHttpAgent,
    mergeDriverHttpTimeoutOptions,
    resolveDriverHttpTimeouts,
} from './http-agent.js';

class TestDriver extends AbstractDriver<DriverOptions, string> {
    provider = 'test';

    getAgent(): Agent {
        return this.getHttpAgent();
    }

    getFetch(): typeof fetch {
        return this.getDriverFetch();
    }

    async requestTextCompletion(prompt: string, _options: ExecutionOptions): Promise<Completion> {
        const response = await this.getDriverFetch()(prompt);
        return {
            result: [{ type: 'text', value: await response.text() }],
        };
    }

    async requestTextCompletionStream(
        _prompt: string,
        _options: ExecutionOptions,
    ): Promise<AsyncIterable<CompletionChunkObject>> {
        throw new Error('Not implemented');
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel[]> {
        return [];
    }

    async validateConnection(): Promise<boolean> {
        return true;
    }

    async generateEmbeddings(_options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        throw new Error('Not implemented');
    }
}

type TestServer = {
    url: string;
    close: () => Promise<void>;
};

async function startServer(handler: (req: IncomingMessage, res: ServerResponse) => void): Promise<TestServer> {
    const server = createServer(handler);

    await new Promise<void>((resolve, reject) => {
        server.once('error', reject);
        server.listen(0, '127.0.0.1', resolve);
    });

    const address = server.address();
    if (!address || typeof address === 'string') {
        throw new Error('Unable to resolve test server port');
    }

    return {
        url: `http://127.0.0.1:${(address as AddressInfo).port}`,
        close: () =>
            new Promise<void>((resolve, reject) => {
                server.close((error?: Error) => (error ? reject(error) : resolve()));
            }),
    };
}

describe('driver HTTP agent helpers', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('fills missing timeout options from defaults', () => {
        expect(resolveDriverHttpTimeouts()).toEqual(DEFAULT_DRIVER_HTTP_TIMEOUTS);
        expect(
            resolveDriverHttpTimeouts({
                headersTimeout: 123,
                keepAliveTimeout: 456,
            }),
        ).toEqual({
            headersTimeout: 123,
            bodyTimeout: DEFAULT_DRIVER_HTTP_TIMEOUTS.bodyTimeout,
            connectTimeout: DEFAULT_DRIVER_HTTP_TIMEOUTS.connectTimeout,
            keepAliveTimeout: 456,
        });
    });

    it('merges per-call timeout overrides without replacing defined defaults with undefined', () => {
        expect(
            mergeDriverHttpTimeoutOptions(
                {
                    headersTimeout: 1_000,
                    bodyTimeout: 2_000,
                    connectTimeout: 300,
                },
                {
                    headersTimeout: undefined,
                    bodyTimeout: 500,
                },
            ),
        ).toEqual({
            headersTimeout: 1_000,
            bodyTimeout: 500,
            connectTimeout: 300,
        });
    });

    it('accepts global Request instances without parsing them as strings', async () => {
        const server = await startServer((_req, res) => {
            res.writeHead(200, { 'content-type': 'text/plain' });
            res.end('ok');
        });
        const agent = createDriverHttpAgent();
        const wrappedFetch = createAgentBackedFetch(agent);

        try {
            const response = await wrappedFetch(new Request(server.url));
            await expect(response.text()).resolves.toBe('ok');
        } finally {
            await agent.close();
            await server.close();
        }
    });

    it('preserves a global Request body when callers pass undefined init fields', async () => {
        const server = await startServer((req, res) => {
            let body = '';
            req.setEncoding('utf8');
            req.on('data', (chunk) => {
                body += chunk;
            });
            req.on('end', () => {
                res.writeHead(200, { 'content-type': 'application/json' });
                res.end(
                    JSON.stringify({
                        method: req.method,
                        body,
                        contentType: req.headers['content-type'],
                    }),
                );
            });
        });
        const agent = createDriverHttpAgent();
        const wrappedFetch = createAgentBackedFetch(agent);
        const request = new Request(server.url, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ ok: true }),
        });

        try {
            const response = await wrappedFetch(request, {
                method: undefined,
                body: undefined,
            });
            await expect(response.json()).resolves.toEqual({
                method: 'POST',
                body: '{"ok":true}',
                contentType: 'application/json',
            });
        } finally {
            await agent.close();
            await server.close();
        }
    });

    it('enforces headersTimeout through the wrapped fetch', async () => {
        const server = await startServer((_req, res) => {
            const timer = setTimeout(() => {
                if (!res.destroyed) {
                    res.writeHead(200, { 'content-type': 'text/plain' });
                    res.end('late');
                }
            }, 3_000);
            res.on('close', () => clearTimeout(timer));
        });
        const agent = createDriverHttpAgent({ headersTimeout: 100 });
        const wrappedFetch = createAgentBackedFetch(agent);
        const startedAt = Date.now();

        try {
            await expect(wrappedFetch(server.url)).rejects.toThrow();
            expect(Date.now() - startedAt).toBeLessThan(2_500);
        } finally {
            await agent.close();
            await server.close();
        }
    });

    it('enforces bodyTimeout when a response body stalls', async () => {
        const server = await startServer((_req, res) => {
            res.writeHead(200, { 'content-type': 'text/plain' });
            res.write('first');
            const timer = setTimeout(() => {
                if (!res.destroyed) {
                    res.end('second');
                }
            }, 3_000);
            res.on('close', () => clearTimeout(timer));
        });
        const agent = createDriverHttpAgent({ bodyTimeout: 100 });
        const wrappedFetch = createAgentBackedFetch(agent);
        const startedAt = Date.now();

        try {
            const response = await wrappedFetch(server.url);
            await expect(response.text()).rejects.toThrow();
            expect(Date.now() - startedAt).toBeLessThan(2_500);
        } finally {
            await agent.close();
            await server.close();
        }
    });

    it('reuses and closes the lazy driver HTTP agent', () => {
        const driver = new TestDriver({});
        const agent = driver.getAgent();
        const closeSpy = vi.spyOn(agent, 'close').mockResolvedValue();

        expect(driver.getAgent()).toBe(agent);
        expect(driver.getFetch()).toBe(driver.getFetch());

        driver.destroy();

        expect(closeSpy).toHaveBeenCalledTimes(1);
        expect(driver.getAgent()).not.toBe(agent);

        driver.destroy();
    });

    it('applies per-execution timeout overrides without replacing the cached driver fetch', async () => {
        const server = await startServer((_req, res) => {
            const timer = setTimeout(() => {
                if (!res.destroyed) {
                    res.writeHead(200, { 'content-type': 'text/plain' });
                    res.end('late');
                }
            }, 3_000);
            res.on('close', () => clearTimeout(timer));
        });
        const driver = new TestDriver({ httpTimeout: { headersTimeout: 2_000 } });
        const cachedFetch = driver.getFetch();
        const startedAt = Date.now();

        try {
            await expect(
                driver.execute([{ role: PromptRole.user, content: server.url }], {
                    model: 'test-model',
                    httpTimeout: { headersTimeout: 100 },
                }),
            ).rejects.toThrow();
            expect(Date.now() - startedAt).toBeLessThan(2_500);
            expect(driver.getFetch()).toBe(cachedFetch);
        } finally {
            driver.destroy();
            await server.close();
        }
    });
});
