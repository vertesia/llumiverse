import type { ConverseRequest, ConverseResponse } from '@aws-sdk/client-bedrock-runtime';
import type { NovaMessagesPrompt } from '@llumiverse/core/formatters';
import { describe, expect, it, vi } from 'vitest';
import { BedrockDriver } from './index.js';
import type { TwelvelabsPegasusRequest } from './twelvelabs.js';

const MODEL = 'twelvelabs.pegasus-1-2-v1:0';
const PROMPT: TwelvelabsPegasusRequest = {
    inputPrompt: 'Summarize the video',
    mediaSource: { base64String: 'dmlkZW8=' },
};

describe('Bedrock service tiers', () => {
    it('returns the processing tier reported by Converse', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const completion = driver.getExtractedExecution({
            output: { message: { role: 'assistant', content: [{ text: 'answer' }] } },
            stopReason: 'end_turn',
            usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
            metrics: { latencyMs: 1 },
            serviceTier: { type: 'priority' },
        } as unknown as ConverseResponse);

        expect(completion.service_tier).toBe('priority');
    });

    it('uses the Converse service tier object for text models', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const prompt: ConverseRequest = {
            modelId: 'anthropic.claude-sonnet-4-6-v1:0',
            messages: [{ role: 'user', content: [{ text: 'hello' }] }],
        };

        expect(
            driver.preparePayload(prompt, {
                model: 'anthropic.claude-sonnet-4-6-v1:0',
                model_options: { _option_id: 'bedrock-claude', service_tier: 'future-tier' },
            }).serviceTier,
        ).toEqual({ type: 'future-tier' });
    });

    it('uses the InvokeModel service tier string for TwelveLabs', async () => {
        const invokeModel = vi.fn(async () => ({
            body: new TextEncoder().encode(JSON.stringify({ message: 'answer', finishReason: 'stop' })),
        }));
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ invokeModel, destroy: vi.fn() }),
        });

        await driver.requestTextCompletion(PROMPT, {
            model: MODEL,
            model_options: { _option_id: 'bedrock-twelvelabs-pegasus', service_tier: 'flex' },
        });

        expect(invokeModel).toHaveBeenCalledWith(expect.objectContaining({ serviceTier: 'flex' }));
    });

    it('uses the InvokeModelWithResponseStream service tier string for streaming TwelveLabs', async () => {
        const invokeModelWithResponseStream = vi.fn(async () => ({
            body: (async function* () {
                yield {
                    chunk: {
                        bytes: new TextEncoder().encode(JSON.stringify({ message: 'answer', finishReason: 'stop' })),
                    },
                };
            })(),
        }));
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ invokeModelWithResponseStream, destroy: vi.fn() }),
        });

        const stream = await driver.requestTextCompletionStream(PROMPT, {
            model: MODEL,
            model_options: { _option_id: 'bedrock-twelvelabs-pegasus', service_tier: 'reserved' },
        });
        for await (const _chunk of stream) {
            // Consume the provider stream.
        }

        expect(invokeModelWithResponseStream).toHaveBeenCalledWith(
            expect.objectContaining({ serviceTier: 'reserved' }),
        );
    });

    it('passes cancellation to Nova Canvas image generation', async () => {
        const invokeModel = vi.fn(async () => ({
            body: new TextEncoder().encode(JSON.stringify({ images: ['image'] })),
        }));
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ invokeModel, destroy: vi.fn() }),
        });
        const options = {
            model: 'amazon.nova-canvas-v1:0',
            model_options: {
                _option_id: 'bedrock-nova-canvas' as const,
                taskType: 'TEXT_IMAGE' as const,
                width: 512,
                height: 512,
            },
        };
        const prompt: NovaMessagesPrompt = {
            messages: [{ role: 'user', content: [{ text: 'Draw a tree' }] }],
        };
        const controller = new AbortController();

        const completion = await driver.requestImageGeneration(prompt, options, controller.signal);
        expect(completion.result).toEqual([{ type: 'image', value: 'data:image/png;base64,image' }]);

        expect(invokeModel).toHaveBeenCalledWith(expect.any(Object), {
            abortSignal: controller.signal,
            requestTimeout: 900_000,
        });
    });

    it.each([
        [{ error: 'Image generation was blocked' }, 'Image generation was blocked'],
        [{ images: [] }, 'No images returned by Nova Canvas'],
        [{}, 'No images returned by Nova Canvas'],
    ])('reports a Nova Canvas failure without image output %j', async (body, error) => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({
                invokeModel: vi.fn().mockResolvedValue({
                    body: new TextEncoder().encode(JSON.stringify(body)),
                }),
                destroy: vi.fn(),
            }),
        });
        const prompt: NovaMessagesPrompt = {
            messages: [{ role: 'user', content: [{ text: 'Draw a tree' }] }],
        };
        const result = await driver.requestImageGeneration(prompt, {
            model: 'amazon.nova-canvas-v1:0',
            model_options: { _option_id: 'bedrock-nova-canvas', taskType: 'TEXT_IMAGE' },
        });
        expect(result.error).toEqual({
            code: 'error' in body ? 'content_policy_violation' : 'validation_error',
            message: error,
        });
        expect(result.result).toEqual([]);
    });
});
