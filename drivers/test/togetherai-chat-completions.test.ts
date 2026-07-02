import { Readable } from 'node:stream';
import { PromptRole, type ToolUse } from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, test, vi } from 'vitest';
import { TogetherAIDriver } from '../src/togetherai/index.js';

type ChatCreatePayload = OpenAI.Chat.ChatCompletionCreateParams;

function mockTogetherDriver(response: unknown) {
    const chatCreate = vi.fn(async () => response);
    const responsesCreate = vi.fn();
    const driver = new TogetherAIDriver({ apiKey: 'test-key' });
    driver.service = {
        chat: { completions: { create: chatCreate } },
        responses: { create: responsesCreate },
        get: vi.fn(),
        models: { list: vi.fn() },
    } as unknown as TogetherAIDriver['service'];
    return { driver, chatCreate, responsesCreate };
}

function textResponse(content: string | null, extra: Record<string, unknown> = {}) {
    return {
        id: 'chatcmpl-1',
        object: 'chat.completion',
        created: 1,
        model: 'together/model',
        choices: [
            {
                index: 0,
                message: { role: 'assistant', content, ...extra },
                finish_reason: 'stop',
                logprobs: null,
            },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
    };
}

async function* streamChunks(chunks: unknown[]): AsyncIterable<unknown> {
    for (const chunk of chunks) {
        yield chunk;
    }
}

async function collectText(stream: AsyncIterable<unknown>): Promise<string> {
    let text = '';
    for await (const chunk of stream) {
        const item = chunk as { result?: Array<{ type: string; value: string }> };
        text += item.result?.map((result) => (result.type === 'text' ? result.value : '')).join('') ?? '';
    }
    return text;
}

describe('TogetherAIDriver Chat Completions transport', () => {
    test('uses Chat Completions for normal text and never calls Responses', async () => {
        const { driver, chatCreate, responsesCreate } = mockTogetherDriver(textResponse('ok'));
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Hello' }], {
            model: 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
        });

        const completion = await driver.requestTextCompletion(prompt, {
            model: 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
            model_options: { _option_id: 'text-fallback', temperature: 0.2, max_tokens: 8 },
        });

        expect(responsesCreate).not.toHaveBeenCalled();
        expect(chatCreate).toHaveBeenCalledWith(
            expect.objectContaining({
                model: 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
                stream: false,
                temperature: 0.2,
                max_tokens: 8,
            }),
        );
        expect(completion.result).toEqual([{ type: 'text', value: 'ok' }]);
    });

    test('formats image prompts as Chat Completions image_url parts', async () => {
        const { driver, chatCreate } = mockTogetherDriver(textResponse('image ok'));
        const prompt = await driver.createPrompt(
            [
                {
                    role: PromptRole.user,
                    content: 'What is this?',
                    files: [
                        {
                            mime_type: 'image/png',
                            getStream: async () => Readable.from([Buffer.from('image-bytes')]),
                        },
                    ],
                },
            ],
            { model: 'meta-llama/Llama-4-Scout-17B-16E-Instruct' },
        );

        await driver.requestTextCompletion(prompt, {
            model: 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
            model_options: { _option_id: 'text-fallback' },
        });

        const payload = chatCreate.mock.calls[0][0] as ChatCreatePayload;
        expect(payload.messages).toEqual([
            {
                role: 'user',
                content: [
                    { type: 'text', text: 'What is this?' },
                    {
                        type: 'image_url',
                        image_url: {
                            url: `data:image/png;base64,${Buffer.from('image-bytes').toString('base64')}`,
                            detail: 'auto',
                        },
                    },
                ],
            },
        ]);
    });

    test('uses prompt schema instructions instead of response_format', async () => {
        const { driver, chatCreate } = mockTogetherDriver(textResponse('{"color":"green"}'));
        const result_schema = {
            type: 'object',
            properties: { color: { type: 'string' } },
            required: ['color'],
        };
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'grass' }], {
            model: 'Qwen/Qwen3',
            result_schema,
        });

        await driver.requestTextCompletion(prompt, {
            model: 'Qwen/Qwen3',
            model_options: { _option_id: 'text-fallback' },
            result_schema,
        });

        const payload = chatCreate.mock.calls[0][0] as ChatCreatePayload;
        expect(payload).not.toHaveProperty('response_format');
        expect(payload.messages).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    role: 'system',
                    content: expect.stringContaining('<response_schema>'),
                }),
            ]),
        );
    });

    test('passes tools and parses tool calls', async () => {
        const { driver, chatCreate } = mockTogetherDriver(
            textResponse(null, {
                tool_calls: [
                    {
                        id: 'call_1',
                        type: 'function',
                        function: { name: 'lookup', arguments: '{"city":"Paris"}' },
                    },
                ],
            }),
        );
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Weather?' }], {
            model: 'Qwen/Qwen3',
        });

        const completion = await driver.requestTextCompletion(prompt, {
            model: 'Qwen/Qwen3',
            model_options: { _option_id: 'text-fallback' },
            tools: [{ name: 'lookup', description: 'Lookup weather', input_schema: { type: 'object' } }],
        });

        const payload = chatCreate.mock.calls[0][0] as ChatCreatePayload;
        expect(payload.tools).toEqual([
            {
                type: 'function',
                function: { name: 'lookup', description: 'Lookup weather', parameters: { type: 'object' } },
            },
        ]);
        expect((completion.tool_use?.[0] as ToolUse).tool_input).toEqual({ city: 'Paris' });
    });

    test('streams through Chat Completions and preserves usage-only terminal chunks', async () => {
        const { driver, chatCreate, responsesCreate } = mockTogetherDriver(
            streamChunks([
                { choices: [{ index: 0, delta: { content: [{ type: 'text', text: 'hel' }] }, finish_reason: null }] },
                { choices: [{ index: 0, delta: { content: 'lo' }, finish_reason: 'stop' }] },
                { choices: [], usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 } },
            ]),
        );
        const prompt = await driver.createPrompt([{ role: PromptRole.user, content: 'Hello' }], {
            model: 'Qwen/Qwen3',
        });

        const stream = await driver.requestTextCompletionStream(prompt, {
            model: 'Qwen/Qwen3',
            model_options: { _option_id: 'text-fallback' },
        });

        expect(await collectText(stream)).toBe('hello');
        expect(responsesCreate).not.toHaveBeenCalled();
        expect(chatCreate).toHaveBeenCalledWith(expect.objectContaining({ stream: true }));
    });
});
