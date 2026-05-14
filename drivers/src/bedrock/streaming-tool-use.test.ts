import {
    AIModel,
    Completion,
    CompletionChunkObject,
    DriverOptions,
    EmbeddingsOptions,
    EmbeddingsResult,
    ExecutionOptions,
    ModelSearchPayload,
    PromptRole,
    PromptSegment,
} from '@llumiverse/common';
import { AbstractDriver } from '@llumiverse/core';
import { beforeEach, describe, expect, it } from 'vitest';
import { BedrockDriver } from './index.js';

// ---------------------------------------------------------------------------
// Unit tests: getExtractedStream tool use handling
// ---------------------------------------------------------------------------

describe('BedrockDriver getExtractedStream — tool use', () => {
    let driver: BedrockDriver;
    let toolBlocks: Map<number, { id: string; name: string }>;

    beforeEach(() => {
        driver = new BedrockDriver({ region: 'us-east-1' });
        toolBlocks = new Map();
    });

    it('emits an initial tool_use chunk on contentBlockStart', () => {
        const chunk = driver['getExtractedStream'](
            {
                contentBlockStart: {
                    contentBlockIndex: 1,
                    start: { toolUse: { toolUseId: 'tool-abc', name: 'my_tool' } },
                },
            },
            undefined,
            undefined,
            toolBlocks
        );

        expect(chunk.tool_use).toHaveLength(1);
        expect(chunk.tool_use![0]).toMatchObject({ id: 'tool-abc', tool_name: 'my_tool', tool_input: '' });
        expect(toolBlocks.get(1)).toEqual({ id: 'tool-abc', name: 'my_tool' });
    });

    it('emits a delta tool_use chunk on contentBlockDelta', () => {
        toolBlocks.set(1, { id: 'tool-abc', name: 'my_tool' });

        const chunk = driver['getExtractedStream'](
            {
                contentBlockDelta: {
                    contentBlockIndex: 1,
                    delta: { toolUse: { input: '{"key":' } },
                },
            },
            undefined,
            undefined,
            toolBlocks
        );

        expect(chunk.tool_use).toHaveLength(1);
        expect(chunk.tool_use![0]).toMatchObject({ id: 'tool-abc', tool_name: '', tool_input: '{"key":' });
    });

    it('removes the block from the map on contentBlockStop', () => {
        toolBlocks.set(1, { id: 'tool-abc', name: 'my_tool' });

        driver['getExtractedStream'](
            { contentBlockStop: { contentBlockIndex: 1 } },
            undefined,
            undefined,
            toolBlocks
        );

        expect(toolBlocks.has(1)).toBe(false);
    });

    it('tracks two interleaved tool calls by independent contentBlockIndex', () => {
        driver['getExtractedStream'](
            { contentBlockStart: { contentBlockIndex: 1, start: { toolUse: { toolUseId: 'id-1', name: 'tool_a' } } } },
            undefined, undefined, toolBlocks
        );
        driver['getExtractedStream'](
            { contentBlockStart: { contentBlockIndex: 3, start: { toolUse: { toolUseId: 'id-2', name: 'tool_b' } } } },
            undefined, undefined, toolBlocks
        );

        expect(toolBlocks.get(1)).toEqual({ id: 'id-1', name: 'tool_a' });
        expect(toolBlocks.get(3)).toEqual({ id: 'id-2', name: 'tool_b' });

        const chunk = driver['getExtractedStream'](
            { contentBlockDelta: { contentBlockIndex: 3, delta: { toolUse: { input: '"val"' } } } },
            undefined, undefined, toolBlocks
        );
        expect(chunk.tool_use![0].id).toBe('id-2');
    });

    it('still extracts text deltas when no tool use is present', () => {
        const chunk = driver['getExtractedStream'](
            { contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'hello' } } },
            undefined,
            undefined,
            toolBlocks
        );

        expect(chunk.result).toEqual([{ type: 'text', value: 'hello' }]);
        expect(chunk.tool_use).toBeUndefined();
    });

    it('emits finish_reason "tool_use" from messageStop', () => {
        const chunk = driver['getExtractedStream'](
            { messageStop: { stopReason: 'tool_use' } },
            undefined,
            undefined,
            toolBlocks
        );

        expect(chunk.finish_reason).toBe('tool_use');
    });
});

// ---------------------------------------------------------------------------
// Integration tests: full accumulation via driver.stream()
// ---------------------------------------------------------------------------

class FakeDriver extends AbstractDriver<DriverOptions, string> {
    provider = 'fake';
    chunks: CompletionChunkObject[] = [];

    async requestTextCompletion(_prompt: string, _options: ExecutionOptions): Promise<Completion> {
        throw new Error('not implemented');
    }

    async requestTextCompletionStream(_prompt: string, _options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const chunks = this.chunks;
        return (async function* () { for (const c of chunks) yield c; })();
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel[]> { return []; }
    async validateConnection(): Promise<boolean> { return true; }
    async generateEmbeddings(_options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        throw new Error('not implemented');
    }
}

const FAKE_SEGMENTS: PromptSegment[] = [{ role: PromptRole.user, content: 'test' }];

describe('driver.stream() — Bedrock tool use accumulation', () => {
    it('assembles and JSON-parses tool_input from streamed chunks', async () => {
        const driver = new FakeDriver({});
        const options: ExecutionOptions = { model: 'test-model' };

        // Simulate what the fixed getExtractedStream emits for one tool call
        driver.chunks = [
            { result: [], tool_use: [{ id: 'tool-1', tool_name: 'do_thing', tool_input: '' as any }] },
            { result: [], tool_use: [{ id: 'tool-1', tool_name: '', tool_input: '{"param"' as any }] },
            { result: [], tool_use: [{ id: 'tool-1', tool_name: '', tool_input: ':"hello"}' as any }] },
            { result: [], finish_reason: 'tool_use' },
        ];

        const stream = await driver.stream(FAKE_SEGMENTS, options);
        for await (const _ of stream) { /* drain */ }

        expect(stream.completion!.finish_reason).toBe('tool_use');
        expect(stream.completion!.tool_use).toHaveLength(1);
        expect(stream.completion!.tool_use![0]).toMatchObject({
            id: 'tool-1',
            tool_name: 'do_thing',
            tool_input: { param: 'hello' },
        });
    });

    it('handles two simultaneous tool calls', async () => {
        const driver = new FakeDriver({});
        const options: ExecutionOptions = { model: 'test-model' };

        driver.chunks = [
            { result: [], tool_use: [{ id: 'id-a', tool_name: 'tool_a', tool_input: '' as any }] },
            { result: [], tool_use: [{ id: 'id-b', tool_name: 'tool_b', tool_input: '' as any }] },
            { result: [], tool_use: [{ id: 'id-a', tool_name: '', tool_input: '{"x":1}' as any }] },
            { result: [], tool_use: [{ id: 'id-b', tool_name: '', tool_input: '{"y":2}' as any }] },
            { result: [], finish_reason: 'tool_use' },
        ];

        const stream = await driver.stream(FAKE_SEGMENTS, options);
        for await (const _ of stream) { /* drain */ }

        const toolUse = stream.completion!.tool_use!;
        expect(toolUse).toHaveLength(2);
        expect(toolUse.find(t => t.id === 'id-a')!.tool_input).toEqual({ x: 1 });
        expect(toolUse.find(t => t.id === 'id-b')!.tool_input).toEqual({ y: 2 });
    });

    it('drops truncated tool calls when finish_reason is length', async () => {
        const driver = new FakeDriver({});
        const options: ExecutionOptions = { model: 'test-model' };

        driver.chunks = [
            { result: [], tool_use: [{ id: 'trunc', tool_name: 'tool_c', tool_input: '' as any }] },
            { result: [], tool_use: [{ id: 'trunc', tool_name: '', tool_input: '{"incomplete' as any }] },
            { result: [], finish_reason: 'length' },
        ];

        const stream = await driver.stream(FAKE_SEGMENTS, options);
        for await (const _ of stream) { /* drain */ }

        expect(stream.completion!.tool_use).toBeUndefined();
    });
});

describe('BedrockDriver buildStreamingConversation', () => {
    it('writes streamed text and tool use blocks back into the assistant message', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const prompt = {
            modelId: 'anthropic.claude-sonnet',
            messages: [
                { role: 'user', content: [{ text: 'What is the weather in Paris?' }] },
            ],
        };

        const conversation = driver.buildStreamingConversation(
            prompt as any,
            [{ type: 'text', value: 'Let me check.' }] as any,
            [{
                id: 'tool-1',
                tool_name: 'get_weather',
                tool_input: { location: 'Paris' },
            }],
            { model: 'anthropic.claude-sonnet' } as ExecutionOptions
        ) as any;

        expect(conversation.messages).toHaveLength(2);
        expect(conversation.messages[0]).toEqual(prompt.messages[0]);
        expect(conversation.messages[1]).toEqual({
            role: 'assistant',
            content: [
                { text: 'Let me check.' },
                {
                    toolUse: {
                        toolUseId: 'tool-1',
                        name: 'get_weather',
                        input: { location: 'Paris' },
                    },
                },
            ],
        });
    });
});
