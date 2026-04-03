import { ExecutionOptions } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { VertexAIDriver } from '../index.js';
import { ClaudeModelDefinition } from './claude.js';

function createAsyncStream(events: any[]): AsyncIterable<any> {
    return (async function* () {
        for (const event of events) {
            yield event;
        }
    })();
}

async function collectChunks(stream: AsyncIterable<any>) {
    const chunks: any[] = [];
    for await (const chunk of stream) {
        chunks.push(chunk);
    }
    return chunks;
}

describe('ClaudeModelDefinition streaming spacing', () => {
    it('does not leak deferred spacing when tool use follows thinking', async () => {
        const modelDef = new ClaudeModelDefinition('claude-sonnet-4-5');
        const driver = {
            logger: { warn: () => { }, info: () => { }, error: () => { } },
            getAnthropicClient: async () => ({
                messages: {
                    stream: async () => createAsyncStream([
                        {
                            type: 'content_block_delta',
                            delta: { type: 'thinking_delta', thinking: 'Thinking...' },
                        },
                        {
                            type: 'content_block_delta',
                            delta: { type: 'signature_delta' },
                        },
                        {
                            type: 'content_block_start',
                            content_block: { type: 'tool_use', id: 'tool-1', name: 'get_weather' },
                        },
                        {
                            type: 'content_block_delta',
                            delta: { type: 'input_json_delta', partial_json: '{"city":"Paris"}' },
                        },
                        {
                            type: 'content_block_stop',
                        },
                    ]),
                },
            }),
        } as unknown as VertexAIDriver;

        const prompt = {
            messages: [{ role: 'user', content: [{ type: 'text', text: 'Weather?' }] }],
        } as any;

        const options = {
            model: 'publishers/anthropic/models/claude-sonnet-4-5',
            model_options: {
                _option_id: 'vertexai-claude',
                include_thoughts: true,
            },
        } as ExecutionOptions;

        const stream = await modelDef.requestTextCompletionStream(driver, prompt, options);
        const chunks = await collectChunks(stream);

        const textOutput = chunks.flatMap(chunk => chunk.result ?? []).map(part => part.value).join('');
        const toolChunks = chunks.flatMap(chunk => chunk.tool_use ?? []);

        expect(textOutput).toBe('Thinking...');
        expect(toolChunks).toHaveLength(2);
        expect(toolChunks[0]).toMatchObject({ id: 'tool-1', tool_name: 'get_weather', tool_input: '' });
        expect(toolChunks[1]).toMatchObject({ id: 'tool-1', tool_name: '', tool_input: '{"city":"Paris"}' });
    });

    it('flushes deferred spacing into the first text delta after thinking', async () => {
        const modelDef = new ClaudeModelDefinition('claude-sonnet-4-5');
        const driver = {
            logger: { warn: () => { }, info: () => { }, error: () => { } },
            getAnthropicClient: async () => ({
                messages: {
                    stream: async () => createAsyncStream([
                        {
                            type: 'content_block_delta',
                            delta: { type: 'thinking_delta', thinking: 'Thinking...' },
                        },
                        {
                            type: 'content_block_delta',
                            delta: { type: 'signature_delta' },
                        },
                        {
                            type: 'content_block_delta',
                            delta: { type: 'text_delta', text: 'Answer' },
                        },
                    ]),
                },
            }),
        } as unknown as VertexAIDriver;

        const prompt = {
            messages: [{ role: 'user', content: [{ type: 'text', text: 'Question?' }] }],
        } as any;

        const options = {
            model: 'publishers/anthropic/models/claude-sonnet-4-5',
            model_options: {
                _option_id: 'vertexai-claude',
                include_thoughts: true,
            },
        } as ExecutionOptions;

        const stream = await modelDef.requestTextCompletionStream(driver, prompt, options);
        const chunks = await collectChunks(stream);

        const textParts = chunks.flatMap(chunk => chunk.result ?? []).map(part => part.value);
        expect(textParts).toEqual(['Thinking...', '\n\nAnswer']);
    });

    it('does not reintroduce deferred spacing when text arrives after a tool call', async () => {
        const modelDef = new ClaudeModelDefinition('claude-sonnet-4-5');
        const driver = {
            logger: { warn: () => { }, info: () => { }, error: () => { } },
            getAnthropicClient: async () => ({
                messages: {
                    stream: async () => createAsyncStream([
                        {
                            type: 'content_block_delta',
                            delta: { type: 'thinking_delta', thinking: 'Thinking...' },
                        },
                        {
                            type: 'content_block_delta',
                            delta: { type: 'signature_delta' },
                        },
                        {
                            type: 'content_block_start',
                            content_block: { type: 'tool_use', id: 'tool-1', name: 'get_weather' },
                        },
                        {
                            type: 'content_block_delta',
                            delta: { type: 'input_json_delta', partial_json: '{"city":"Paris"}' },
                        },
                        {
                            type: 'content_block_stop',
                        },
                        {
                            type: 'content_block_delta',
                            delta: { type: 'text_delta', text: 'Answer after tool' },
                        },
                    ]),
                },
            }),
        } as unknown as VertexAIDriver;

        const prompt = {
            messages: [{ role: 'user', content: [{ type: 'text', text: 'Weather?' }] }],
        } as any;

        const options = {
            model: 'publishers/anthropic/models/claude-sonnet-4-5',
            model_options: {
                _option_id: 'vertexai-claude',
                include_thoughts: true,
            },
        } as ExecutionOptions;

        const stream = await modelDef.requestTextCompletionStream(driver, prompt, options);
        const chunks = await collectChunks(stream);

        const textParts = chunks.flatMap(chunk => chunk.result ?? []).map(part => part.value);
        expect(textParts).toEqual(['Thinking...', 'Answer after tool']);
    });
});
