import type { ConverseRequest } from '@aws-sdk/client-bedrock-runtime';
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

        await driver.requestImageGeneration(prompt, options, controller.signal);

        expect(invokeModel).toHaveBeenCalledWith(expect.any(Object), {
            abortSignal: controller.signal,
            requestTimeout: 900_000,
        });
    });
});
