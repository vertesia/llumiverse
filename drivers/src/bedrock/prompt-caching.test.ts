import { type ConverseResponse, StopReason } from '@aws-sdk/client-bedrock-runtime';
import { type DataSource, type ExecutionOptions, PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { formatConversePrompt } from './converse.js';
import { BedrockDriver } from './index.js';

describe('Bedrock prompt caching', () => {
    const imageSource = (): DataSource => ({
        name: 'page.jpg',
        mime_type: 'image/jpeg',
        getStream: vi.fn().mockResolvedValue(
            new ReadableStream({
                start(controller) {
                    controller.enqueue(new Uint8Array([1, 2, 3]));
                    controller.close();
                },
            }),
        ),
        getURL: vi.fn(),
        getURI: vi.fn(),
    });

    it('places a cache point after stable source attachments and before the final task', async () => {
        const options: ExecutionOptions = {
            model: 'anthropic.claude-sonnet-4-6',
            prompt_cache_key: 'document-prefix',
            result_schema: { type: 'object', properties: { value: { type: 'string' } } },
        };
        const prompt = await formatConversePrompt(
            [
                { role: PromptRole.user, content: 'stable document source', files: [imageSource()] },
                { role: PromptRole.user, content: 'dynamic extraction task' },
            ],
            options,
        );
        const driver = new BedrockDriver({ region: 'us-east-1' });

        const payload = driver.preparePayload(prompt, options);

        expect(payload.messages).toEqual([
            {
                role: 'user',
                content: [
                    { text: 'stable document source' },
                    { image: { format: 'jpeg', source: { bytes: new Uint8Array([1, 2, 3]) } } },
                    { cachePoint: { type: 'default' } },
                    { text: expect.stringContaining('dynamic extraction task') },
                ],
            },
        ]);
        expect(payload.messages?.[0].content?.[3]).toMatchObject({ text: expect.stringContaining('"value"') });
    });

    it('keeps the source prefix stable when routed tasks use different schemas', async () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const createPayload = async (task: string, field: string) => {
            const options: ExecutionOptions = {
                model: 'anthropic.claude-sonnet-4-6',
                prompt_cache_key: 'document-prefix',
                result_schema: { type: 'object', properties: { [field]: { type: 'string' } } },
            };
            const prompt = await formatConversePrompt(
                [
                    { role: PromptRole.system, content: 'shared system' },
                    { role: PromptRole.user, content: 'stable document source' },
                    { role: PromptRole.user, content: task },
                ],
                options,
            );
            return driver.preparePayload(prompt, options);
        };

        const extraction = await createPayload('extract fields', 'invoice_number');
        const review = await createPayload('review fields', 'review_verdict');

        expect(extraction.system).toEqual(review.system);
        expect(extraction.messages?.[0].content?.slice(0, 2)).toEqual(review.messages?.[0].content?.slice(0, 2));
        expect(JSON.stringify(extraction.system)).not.toContain('invoice_number');
        expect(JSON.stringify(review.system)).not.toContain('review_verdict');
        expect(extraction.messages?.[0].content?.[2]).toMatchObject({
            text: expect.stringContaining('invoice_number'),
        });
        expect(review.messages?.[0].content?.[2]).toMatchObject({ text: expect.stringContaining('review_verdict') });
    });

    it('reports cache reads and writes in non-streaming usage', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const response = {
            output: { message: { role: 'assistant', content: [{ text: 'done' }] } },
            stopReason: StopReason.END_TURN,
            usage: {
                inputTokens: 25,
                outputTokens: 10,
                totalTokens: 185,
                cacheReadInputTokens: 100,
                cacheWriteInputTokens: 50,
            },
            metrics: { latencyMs: 1 },
        } satisfies ConverseResponse;

        expect(driver.getExtractedExecution(response).token_usage).toEqual({
            prompt: 175,
            prompt_new: 25,
            prompt_cached: 100,
            prompt_cache_write: 50,
            result: 10,
            total: 185,
        });
    });
});
