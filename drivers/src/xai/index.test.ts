import { type ExecutionOptions, PromptRole, Providers } from '@llumiverse/core';
import type { FetchClient } from '@vertesia/api-fetch-client';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { xAIDriver } from './index.js';

describe('xAI model listing', () => {
    it('uses verified catalog capabilities instead of runtime modality claims', async () => {
        const driver = new xAIDriver({ apiKey: 'test-key' });
        const get = vi.fn(async (path: string) => {
            if (path === '/image-generation-models') {
                return {
                    models: [
                        {
                            id: 'grok-imagine-image-2.0',
                            owned_by: 'xAI',
                            version: '2.0.0',
                            input_modalities: ['text', 'image'],
                            output_modalities: ['image'],
                        },
                    ],
                };
            }
            return {
                models: [
                    {
                        id: 'grok-4.3',
                        owned_by: 'xAI',
                        input_modalities: ['audio'],
                        output_modalities: ['video'],
                    },
                    {
                        id: 'text-embedding-future',
                        owned_by: 'xAI',
                        input_modalities: ['text'],
                        output_modalities: ['embedding'],
                    },
                ],
            };
        });
        driver.xai_service = { get } as unknown as FetchClient;

        expect(await driver.listModels()).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    id: 'grok-4.3',
                    provider: Providers.xai,
                    input_modalities: ['text', 'image'],
                    output_modalities: ['text'],
                    tool_support: true,
                }),
                expect.objectContaining({
                    id: 'grok-imagine-image-2.0',
                    provider: Providers.xai,
                    type: 'image',
                    input_modalities: ['text', 'image'],
                    output_modalities: ['image'],
                    tool_support: false,
                }),
            ]),
        );
        expect(get).toHaveBeenCalledWith('/language-models');
        expect(get).toHaveBeenCalledWith('/image-generation-models');
    });

    it('keeps language models visible when the API key cannot list image models', async () => {
        const driver = new xAIDriver({ apiKey: 'test-key' });
        const get = vi.fn(async (path: string) => {
            if (path === '/image-generation-models') throw new Error('Forbidden');
            return { models: [{ id: 'grok-4.5', owned_by: 'xAI' }] };
        });
        driver.xai_service = { get } as unknown as FetchClient;

        await expect(driver.listModels()).resolves.toEqual([
            expect.objectContaining({ id: 'grok-4.5', provider: Providers.xai }),
        ]);
    });
});

describe('xAI image generation', () => {
    it('routes all Grok image-family models through the image API', () => {
        const driver = new xAIDriver({ apiKey: 'test-key' });

        expect(driver.isImageModel('grok-2-image-1212')).toBe(true);
        expect(driver.isImageModel('grok-imagine-image-quality')).toBe(true);
        expect(driver.isImageModel('grok-imagine-image-2.0')).toBe(true);
        expect(driver.isImageModel('grok-4.3')).toBe(false);
    });

    it('sends xAI generation options and returns URL and base64 images', async () => {
        const driver = new xAIDriver({ apiKey: 'test-key' });
        const post = vi.fn(async () => ({
            data: [
                { url: 'https://example.com/image.jpeg', mime_type: 'image/jpeg' },
                { b64_json: 'aW1hZ2U=', mime_type: 'image/jpeg' },
            ],
        }));
        driver.xai_service = { post } as unknown as FetchClient;
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'A lighthouse in a storm' }], {
            model: 'grok-imagine-image-2.0',
        });
        const options: ExecutionOptions = {
            model: 'grok-imagine-image-2.0',
            model_options: {
                _option_id: 'xai-grok-image',
                aspect_ratio: '16:9',
                resolution: '2k',
                quality: 'low',
                response_format: 'b64_json',
                n: 2,
            },
        };

        await expect(driver.requestImageGeneration(prompt, options)).resolves.toEqual({
            result: [
                { type: 'image', value: 'https://example.com/image.jpeg' },
                { type: 'image', value: 'data:image/jpeg;base64,aW1hZ2U=' },
            ],
        });
        expect(post).toHaveBeenCalledWith('/images/generations', {
            payload: {
                model: 'grok-imagine-image-2.0',
                prompt: 'A lighthouse in a storm',
                aspect_ratio: '16:9',
                resolution: '2k',
                quality: 'low',
                response_format: 'b64_json',
                n: 2,
            },
        });
    });

    it('falls back from streaming to the image generation endpoint for Image 2.0', async () => {
        const driver = new xAIDriver({ apiKey: 'test-key' });
        const post = vi.fn(async () => ({ data: [{ url: 'https://example.com/image.jpeg' }] }));
        driver.xai_service = { post } as unknown as FetchClient;
        const options: ExecutionOptions = {
            model: 'grok-imagine-image-2.0',
            model_options: {
                _option_id: 'xai-grok-image',
                aspect_ratio: '1:1',
                resolution: '2k',
                quality: 'medium',
                response_format: 'url',
                n: 1,
            },
        };

        const stream = await driver.stream([{ role: PromptRole.user, content: 'A lighthouse in a storm' }], options);
        const chunks: string[] = [];
        for await (const chunk of stream) chunks.push(chunk);

        expect(chunks).toEqual(['[Image: https://ex...]']);
        expect(stream.completion?.result).toEqual([{ type: 'image', value: 'https://example.com/image.jpeg' }]);
        expect(post).toHaveBeenCalledWith('/images/generations', {
            payload: {
                model: 'grok-imagine-image-2.0',
                prompt: 'A lighthouse in a storm',
                aspect_ratio: '1:1',
                resolution: '2k',
                quality: 'medium',
                response_format: 'url',
                n: 1,
            },
        });
    });

    it('uses the edits endpoint when the prompt contains reference images', async () => {
        const driver = new xAIDriver({ apiKey: 'test-key' });
        const post = vi.fn(async () => ({ data: [{ url: 'https://example.com/edit.jpeg' }] }));
        driver.xai_service = { post } as unknown as FetchClient;
        const prompt: OpenAI.Responses.ResponseInputItem[] = [
            {
                type: 'message',
                role: 'user',
                content: [
                    { type: 'input_text', text: 'Turn this into a pencil sketch' },
                    { type: 'input_image', image_url: 'data:image/png;base64,aW1hZ2U=', detail: 'auto' },
                ],
            },
        ];

        await driver.requestImageGeneration(prompt, { model: 'grok-imagine-image-quality' });

        expect(post).toHaveBeenCalledWith('/images/edits', {
            payload: {
                model: 'grok-imagine-image-quality',
                prompt: 'Turn this into a pencil sketch',
                image: {
                    type: 'image_url',
                    url: 'data:image/png;base64,aW1hZ2U=',
                },
            },
        });
    });
});
