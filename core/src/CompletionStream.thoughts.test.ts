import type {
    AIModel,
    Completion,
    DriverCompletionStream,
    DriverOptions,
    EmbeddingsOptions,
    EmbeddingsResult,
    ExecutionOptions,
    ModelSearchPayload,
} from '@llumiverse/common';
import { describe, expect, it } from 'vitest';
import { DefaultCompletionStream, FallbackCompletionStream } from './CompletionStream.js';
import { AbstractDriver } from './Driver.js';

class ThoughtsStreamDriver extends AbstractDriver<DriverOptions, string> {
    provider = 'test';

    async requestTextCompletion(prompt: string): Promise<Completion> {
        if (prompt === 'video') {
            return {
                result: [
                    { type: 'video', value: 'gs://bucket/one.mp4' },
                    { type: 'video', value: 'gs://bucket/two.mp4' },
                ],
            };
        }
        return {
            result: [
                { type: 'thoughts', value: 'reason-fallback' },
                { type: 'text', value: 'answer-fallback' },
            ],
        };
    }

    async requestTextCompletionStream(prompt: string): Promise<DriverCompletionStream> {
        const nativeAssistant = { role: 'assistant', id: prompt };
        if (prompt === 'failed') {
            return {
                async *[Symbol.asyncIterator]() {
                    yield {
                        result: [],
                        token_usage: { prompt: 10, prompt_cached: 4, prompt_new: 6, result: 2, total: 12 },
                    };
                    throw new Error('provider failed after usage');
                },
            };
        }
        if (prompt === 'video') {
            return {
                async *[Symbol.asyncIterator]() {
                    yield { result: [{ type: 'video', value: 'gs://bucket/one.mp4' }] };
                    yield { result: [{ type: 'video', value: 'gs://bucket/two.mp4' }], finish_reason: 'stop' };
                },
                finalizeConversation: () => nativeAssistant,
            };
        }
        return {
            async *[Symbol.asyncIterator]() {
                yield { result: [{ type: 'thoughts', value: 'reason-' }] };
                yield { result: [{ type: 'thoughts', value: prompt }] };
                await Promise.resolve();
                yield {
                    result: [{ type: 'text', value: `answer-${prompt}` }],
                    finish_reason: 'stop',
                    service_tier: 'priority',
                };
            },
            finalizeConversation: () => nativeAssistant,
        };
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel[]> {
        return [];
    }

    async validateConnection(): Promise<boolean> {
        return true;
    }

    async generateEmbeddings(_options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        throw new Error('not used');
    }
}

describe('DefaultCompletionStream thoughts', () => {
    it('preserves terminal usage on a partial completion when the provider stream fails', async () => {
        const driver = new ThoughtsStreamDriver({});
        const stream = new DefaultCompletionStream(driver, 'failed', { model: 'test' });

        await expect(async () => {
            for await (const _chunk of stream) {
                // drain
            }
        }).rejects.toThrow('provider failed after usage');
        expect(stream.completion?.token_usage).toEqual({
            prompt: 10,
            prompt_cached: 4,
            prompt_new: 6,
            result: 2,
            total: 12,
        });
    });

    it('streams visible thoughts before the answer while preserving typed results', async () => {
        const driver = new ThoughtsStreamDriver({});
        const stream = new DefaultCompletionStream(driver, 'one', { model: 'test' });

        const visible: string[] = [];
        for await (const chunk of stream) visible.push(chunk);

        expect(visible).toEqual(['reason-', 'one', '\nanswer-one']);
        expect(stream.completion?.result).toEqual([
            { type: 'thoughts', value: 'reason-one' },
            { type: 'text', value: 'answer-one' },
        ]);
        expect(stream.completion?.conversation).toEqual({ role: 'assistant', id: 'one' });
        expect(stream.completion?.service_tier).toBe('priority');
    });

    it('separates thoughts from the answer in the fallback string stream', async () => {
        const driver = new ThoughtsStreamDriver({});
        const stream = new FallbackCompletionStream(driver, 'fallback', { model: 'test' });

        const visible: string[] = [];
        for await (const chunk of stream) visible.push(chunk);

        expect(visible).toEqual(['reason-fallback\nanswer-fallback']);
        expect(stream.completion?.result).toEqual([
            { type: 'thoughts', value: 'reason-fallback' },
            { type: 'text', value: 'answer-fallback' },
        ]);
    });

    it('isolates native finalizers across concurrent streams', async () => {
        const driver = new ThoughtsStreamDriver({});
        const streams = ['left', 'right'].map(
            (prompt) => new DefaultCompletionStream(driver, prompt, { model: 'test' } satisfies ExecutionOptions),
        );

        await Promise.all(
            streams.map(async (stream) => {
                for await (const _chunk of stream) {
                    // drain
                }
            }),
        );

        expect(streams[0].completion?.conversation).toEqual({ role: 'assistant', id: 'left' });
        expect(streams[1].completion?.conversation).toEqual({ role: 'assistant', id: 'right' });
    });

    it('keeps streamed video results discrete', async () => {
        const driver = new ThoughtsStreamDriver({});
        const stream = new DefaultCompletionStream(driver, 'video', { model: 'test' });

        const visible: string[] = [];
        for await (const chunk of stream) visible.push(chunk);

        expect(visible).toEqual(['\n[Video: gs://bucket/one.mp4]\n', '\n[Video: gs://bucket/two.mp4]\n']);
        expect(stream.completion?.result).toEqual([
            { type: 'video', value: 'gs://bucket/one.mp4' },
            { type: 'video', value: 'gs://bucket/two.mp4' },
        ]);
    });

    it('preserves video results through fallback streaming', async () => {
        const driver = new ThoughtsStreamDriver({});
        const stream = new FallbackCompletionStream(driver, 'video', { model: 'test' });

        const visible: string[] = [];
        for await (const chunk of stream) visible.push(chunk);

        expect(visible).toEqual(['[Video: gs://bucket/one.mp4][Video: gs://bucket/two.mp4]']);
        expect(stream.completion?.result).toHaveLength(2);
    });
});
