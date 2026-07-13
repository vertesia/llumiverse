import { PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { MistralAIDriver } from './index.js';

describe('MistralAIDriver official SDK transport', () => {
    it('maps canonical Chat requests and preserves typed content, tools, usage, and the native response', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key', endpoint_url: 'https://mistral.example.test' });
        const response = {
            id: 'mistral-1',
            object: 'chat.completion',
            created: 1,
            model: 'mistral-large',
            choices: [
                {
                    index: 0,
                    finishReason: 'tool_calls',
                    message: {
                        role: 'assistant',
                        content: [{ type: 'text', text: 'hello' }],
                        toolCalls: [
                            {
                                id: 'call_1',
                                index: 0,
                                type: 'function',
                                function: { name: 'lookup', arguments: { city: 'Paris' } },
                            },
                        ],
                    },
                },
            ],
            usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 },
        };
        const complete = vi.fn(async () => response);
        Object.defineProperty(driver.client.chat, 'complete', { value: complete });
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Hello' }], {
            model: 'mistral-large',
        });
        const completion = await driver.requestTextCompletion(prompt, {
            model: 'mistral-large',
            include_original_response: true,
            model_options: {
                _option_id: 'text-fallback',
                max_tokens: 12,
                top_p: 0.8,
                presence_penalty: 0.1,
                frequency_penalty: 0.2,
                stop_sequence: ['END'],
            },
            tools: [{ name: 'lookup', description: 'Lookup', input_schema: { type: 'object' } }],
        });

        expect(complete).toHaveBeenCalledWith(
            expect.objectContaining({
                maxTokens: 12,
                topP: 0.8,
                presencePenalty: 0.1,
                frequencyPenalty: 0.2,
                stop: ['END'],
                stream: false,
            }),
        );
        expect(completion.result).toEqual([{ type: 'text', value: 'hello' }]);
        expect(completion.tool_use?.[0]).toEqual({
            id: 'call_1',
            tool_name: 'lookup',
            tool_input: { city: 'Paris' },
        });
        expect(completion.original_response).toBe(response);
    });

    it('uses the SDK for models, validation, embeddings, endpoint override, and fetch wiring', async () => {
        const driver = new MistralAIDriver({ apiKey: 'test-key', endpoint_url: 'https://mistral.example.test' });
        const list = vi.fn(async () => ({
            object: 'list',
            data: [
                {
                    id: 'mistral-large',
                    name: 'Mistral Large',
                    description: 'Large model',
                    ownedBy: 'mistralai',
                    type: 'base',
                },
            ],
        }));
        Object.defineProperty(driver.client.models, 'list', { value: list });
        Object.defineProperty(driver.client.embeddings, 'create', {
            value: vi.fn(async () => ({
                id: 'emb-1',
                object: 'list',
                model: 'mistral-embed',
                data: [{ index: 0, embedding: [0.1, 0.2] }],
                usage: { promptTokens: 2, totalTokens: 2 },
            })),
        });

        await expect(driver.validateConnection()).resolves.toBe(true);
        await expect(driver.listModels()).resolves.toEqual([
            expect.objectContaining({ id: 'mistral-large', name: 'Mistral Large', owner: 'mistralai' }),
        ]);
        await expect(
            driver.generateEmbeddings({ model: 'mistral-embed', inputs: [{ type: 'text', text: 'hello' }] }),
        ).resolves.toEqual({
            model: 'mistral-embed',
            results: [{ outputs: [{ values: [0.1, 0.2], modality: 'text' }] }],
            usage: { input_tokens: 2, input_text_tokens: 2 },
        });
        expect(list).toHaveBeenCalledTimes(2);
    });
});
