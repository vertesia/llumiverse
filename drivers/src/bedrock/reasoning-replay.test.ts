import type { ConverseRequest, ConverseResponse, ConverseStreamOutput } from '@aws-sdk/client-bedrock-runtime';
import { describe, expect, it, vi } from 'vitest';
import { BedrockDriver, excludesBedrockReasoningReplay } from './index.js';

const MODEL = 'anthropic.claude-sonnet-4-6-v1:0';

function prompt(): ConverseRequest {
    return { modelId: MODEL, messages: [{ role: 'user', content: [{ text: 'question' }] }] };
}

describe('Bedrock native reasoning replay', () => {
    it('preserves reasoning signatures and redacted bytes across a JSON roundtrip', async () => {
        const redacted = new Uint8Array([1, 2, 255]);
        const response: ConverseResponse = {
            output: {
                message: {
                    role: 'assistant',
                    content: [
                        { reasoningContent: { reasoningText: { text: 'plan', signature: 'signed-chain' } } },
                        { reasoningContent: { redactedContent: redacted } },
                        { text: 'answer' },
                    ],
                },
            },
            stopReason: 'end_turn',
            usage: { inputTokens: 1, outputTokens: 2, totalTokens: 3 },
            metrics: { latencyMs: 1 },
        };
        const converse = vi.fn(async (_request: ConverseRequest) => response);
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ converse, destroy: vi.fn() }),
        });

        const first = await driver.requestTextCompletion(prompt(), {
            model: MODEL,
            stripTextMaxTokens: 1,
            stripImagesAfterTurns: 0,
        });
        expect(first.result).toEqual([{ type: 'text', value: 'answer' }]);
        const persisted = JSON.parse(JSON.stringify(first.conversation));

        await driver.requestTextCompletion(
            { modelId: MODEL, messages: [{ role: 'user', content: [{ text: 'continue' }] }] },
            { model: MODEL, conversation: persisted },
        );
        const replay = converse.mock.calls[1][0] as ConverseRequest;
        expect(replay.messages).toContainEqual(response.output?.message);
        expect(replay.messages?.[1]?.content?.[1].reasoningContent?.redactedContent).toEqual(redacted);
    });

    it('reconstructs fragmented reasoning, text, redaction, and tool blocks by native index', async () => {
        const events: ConverseStreamOutput[] = [
            { contentBlockDelta: { contentBlockIndex: 0, delta: { reasoningContent: { text: 'plan ' } } } },
            { contentBlockDelta: { contentBlockIndex: 0, delta: { reasoningContent: { text: 'more' } } } },
            {
                contentBlockDelta: {
                    contentBlockIndex: 0,
                    delta: { reasoningContent: { signature: 'stream-signature' } },
                },
            },
            { contentBlockDelta: { contentBlockIndex: 1, delta: { text: 'answer' } } },
            {
                contentBlockStart: {
                    contentBlockIndex: 2,
                    start: { toolUse: { toolUseId: 'call-1', name: 'lookup' } },
                },
            },
            { contentBlockDelta: { contentBlockIndex: 2, delta: { toolUse: { input: '{"city":' } } } },
            { contentBlockDelta: { contentBlockIndex: 2, delta: { toolUse: { input: '"Paris"}' } } } },
            {
                contentBlockStart: {
                    contentBlockIndex: 3,
                    start: { toolUse: { toolUseId: 'call-truncated', name: 'lookup' } },
                },
            },
            { contentBlockDelta: { contentBlockIndex: 3, delta: { toolUse: { input: '{"city":' } } } },
            {
                contentBlockStart: {
                    contentBlockIndex: 4,
                    start: { reasoningContent: { redactedContent: new Uint8Array([9, 8]) } },
                },
            } as unknown as ConverseStreamOutput,
            { messageStop: { stopReason: 'max_tokens' } },
            { metadata: { usage: { inputTokens: 1, outputTokens: 2, totalTokens: 3 }, metrics: { latencyMs: 1 } } },
        ];
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({
                converseStream: vi.fn(async () => ({
                    stream: (async function* () {
                        for (const event of events) yield event;
                    })(),
                })),
                destroy: vi.fn(),
            }),
        });

        const stream = await driver.requestTextCompletionStream(prompt(), { model: MODEL });
        const results = [];
        for await (const chunk of stream) results.push(...chunk.result);
        const conversation = await stream.finalizeConversation?.();

        expect(results).toEqual([{ type: 'text', value: 'answer' }]);
        expect(conversation).toMatchObject({
            messages: expect.arrayContaining([
                {
                    role: 'assistant',
                    content: [
                        { reasoningContent: { reasoningText: { text: 'plan more', signature: 'stream-signature' } } },
                        { text: 'answer' },
                        { toolUse: { toolUseId: 'call-1', name: 'lookup', input: { city: 'Paris' } } },
                        { reasoningContent: { redactedContent: { _base64: 'CQg=' } } },
                    ],
                },
            ]),
        });
        const persistedContent = (conversation as ConverseRequest).messages?.at(-1)?.content;
        expect(persistedContent).not.toContainEqual(
            expect.objectContaining({
                toolUse: expect.objectContaining({ toolUseId: 'call-truncated' }),
            }),
        );
    });

    it('keeps the DeepSeek replay exclusion narrow', () => {
        expect(excludesBedrockReasoningReplay('deepseek.r1-v1:0')).toBe(true);
        expect(excludesBedrockReasoningReplay('us.deepseek.r1-v1:0')).toBe(true);
        expect(excludesBedrockReasoningReplay('deepseek.v3-v1:0')).toBe(false);
        expect(excludesBedrockReasoningReplay('vendor.deepseek-r1-compatible')).toBe(false);
    });
});
