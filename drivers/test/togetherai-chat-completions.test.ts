import type { CompletionChunkObject, ToolUse } from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, test } from 'vitest';
import { mapChatCompletionStream } from '../src/openai/index.js';
import { convertResponseItemsToChatMessages } from '../src/openai/openai_format.js';

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;
type ChatCompletionChunk = OpenAI.Chat.Completions.ChatCompletionChunk;

async function* streamChunks(chunks: ChatCompletionChunk[]): AsyncIterable<ChatCompletionChunk> {
    for (const chunk of chunks) {
        yield chunk;
    }
}

async function collect(stream: AsyncIterable<CompletionChunkObject>): Promise<CompletionChunkObject[]> {
    const out: CompletionChunkObject[] = [];
    for await (const chunk of stream) {
        out.push(chunk);
    }
    return out;
}

describe('convertResponseItemsToChatMessages (Response API -> Chat Completions)', () => {
    test('maps a user input_image to a Chat Completions image_url part', () => {
        const items = [
            {
                type: 'message',
                role: 'user',
                content: [
                    { type: 'input_text', text: 'What is in this image?' },
                    { type: 'input_image', image_url: 'data:image/jpeg;base64,AAAA', detail: 'auto' },
                ],
            },
        ] as ResponseInputItem[];

        const messages = convertResponseItemsToChatMessages(items);

        expect(messages).toHaveLength(1);
        const msg = messages[0] as unknown as { role: string; content: Array<Record<string, unknown>> };
        expect(msg.role).toBe('user');
        expect(msg.content).toEqual([
            { type: 'text', text: 'What is in this image?' },
            { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,AAAA', detail: 'auto' } },
        ]);
    });

    test('maps system messages, tool outputs, and tool calls', () => {
        const items = [
            { type: 'message', role: 'system', content: 'You are helpful.' },
            { type: 'function_call', call_id: 'call_1', name: 'search', arguments: '{"q":"cats"}' },
            { type: 'function_call_output', call_id: 'call_1', output: 'result text' },
        ] as ResponseInputItem[];

        const messages = convertResponseItemsToChatMessages(items) as unknown as Array<Record<string, unknown>>;

        expect(messages[0]).toEqual({ role: 'system', content: 'You are helpful.' });
        expect(messages[1]).toEqual({
            role: 'assistant',
            content: null,
            tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'search', arguments: '{"q":"cats"}' } }],
        });
        expect(messages[2]).toEqual({ role: 'tool', tool_call_id: 'call_1', content: 'result text' });
    });
});

describe('mapChatCompletionStream (Chat Completions streaming)', () => {
    test('accumulates text deltas and terminal finish/usage', async () => {
        const chunks = [
            { choices: [{ index: 0, delta: { content: 'Hello ' }, finish_reason: null }] },
            { choices: [{ index: 0, delta: { content: 'world' }, finish_reason: null }] },
            { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
            {
                choices: [],
                usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
            },
        ] as unknown as ChatCompletionChunk[];

        const out = await collect(mapChatCompletionStream(streamChunks(chunks)));

        const text = out
            .flatMap((c) => c.result ?? [])
            .map((r) => (r.type === 'text' ? r.value : ''))
            .join('');
        expect(text).toBe('Hello world');

        // The 'stop' finish chunk carries finish_reason; the usage-only chunk must NOT overwrite it.
        expect(out.some((c) => c.finish_reason === 'stop')).toBe(true);
        const usageChunk = out.find((c) => c.token_usage);
        expect(usageChunk?.token_usage).toMatchObject({ prompt: 10, result: 5, total: 15 });
        expect(usageChunk?.finish_reason).toBeUndefined();
    });

    test('streams tool call deltas keyed by index and maps finish_reason to tool_use', async () => {
        const chunks = [
            {
                choices: [
                    {
                        index: 0,
                        delta: {
                            tool_calls: [
                                {
                                    index: 0,
                                    id: 'call_abc',
                                    type: 'function',
                                    function: { name: 'get_weather', arguments: '{"ci' },
                                },
                            ],
                        },
                        finish_reason: null,
                    },
                ],
            },
            {
                choices: [
                    {
                        index: 0,
                        delta: { tool_calls: [{ index: 0, function: { arguments: 'ty":"Paris"}' } }] },
                        finish_reason: null,
                    },
                ],
            },
            { choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
        ] as unknown as ChatCompletionChunk[];

        const out = await collect(mapChatCompletionStream(streamChunks(chunks)));

        const toolChunks = out.filter((c) => c.tool_use && c.tool_use.length > 0);
        const first = toolChunks[0].tool_use?.[0] as ToolUse<unknown> & { _actual_id?: string };
        expect(first.id).toBe('tool_0');
        expect(first._actual_id).toBe('call_abc');
        expect(first.tool_name).toBe('get_weather');

        // Argument fragments arrive across chunks and concatenate to valid JSON.
        const args = toolChunks.map((c) => (c.tool_use?.[0].tool_input as string) ?? '').join('');
        expect(args).toBe('{"city":"Paris"}');

        expect(out.some((c) => c.finish_reason === 'tool_use')).toBe(true);
    });
});
