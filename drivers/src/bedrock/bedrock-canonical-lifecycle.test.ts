import type {
    ContentBlock,
    ConverseRequest,
    ConverseResponse,
    ConverseStreamOutput,
} from '@aws-sdk/client-bedrock-runtime';
import { parseConversationDocument } from '@llumiverse/conversation';
import type { ExecutionOptions } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { BedrockDriver } from './index.js';

const MODEL = 'anthropic.claude-sonnet-4-6-v1:0';
const TOOLS = [{ name: 'lookup', description: 'Look up a value', input_schema: { type: 'object', properties: {} } }];

function runtimeOptions(input: {
    flow: string;
    operation: string;
    conversation?: unknown;
    tools?: typeof TOOLS;
}): ExecutionOptions {
    const recordedAt = `2026-09-30T00:00:0${input.operation === 'first' ? '0' : '1'}.000Z`;
    return {
        model: MODEL,
        ...(input.conversation === undefined ? {} : { conversation: input.conversation }),
        ...(input.tools === undefined ? {} : { tools: input.tools }),
        conversation_runtime: {
            conversation_id: `conversation:${input.flow}`,
            request_id: `request:${input.flow}:${input.operation}`,
            attempt_id: `attempt:${input.flow}:${input.operation}`,
            input_operation_id: `input:${input.flow}:${input.operation}`,
            response_operation_id: `response:${input.flow}:${input.operation}`,
            recorded_at: recordedAt,
            started_at: recordedAt,
            completed_at: recordedAt,
        },
    };
}

function prompt(content: NonNullable<ConverseRequest['messages']>[number]['content']): ConverseRequest {
    return { modelId: MODEL, messages: [{ role: 'user', content }] };
}

describe('Bedrock canonical driver lifecycle', () => {
    it('keeps native call IDs stable across sync request and application execution receipts', async () => {
        const responses: ConverseResponse[] = [
            {
                output: {
                    message: {
                        role: 'assistant',
                        content: [
                            { toolUse: { toolUseId: 'native-call-17', name: 'lookup', input: { city: 'Tokyo' } } },
                        ],
                    },
                },
                stopReason: 'tool_use',
                usage: { inputTokens: 3, outputTokens: 2, totalTokens: 5 },
                metrics: { latencyMs: 1 },
            },
            {
                output: { message: { role: 'assistant', content: [{ text: 'Tokyo is clear.' }] } },
                stopReason: 'end_turn',
                usage: {
                    inputTokens: 4,
                    outputTokens: 3,
                    totalTokens: 9,
                    cacheReadInputTokens: 1,
                    cacheWriteInputTokens: 1,
                },
                metrics: { latencyMs: 1 },
            },
        ];
        const converse = vi.fn(async (_request: ConverseRequest) => {
            const response = responses.shift();
            if (response === undefined) throw new Error('Unexpected Converse request');
            return response;
        });
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ converse, destroy: vi.fn() }),
        });

        const first = await driver.requestTextCompletion(
            prompt([{ text: 'Check Tokyo.' }]),
            runtimeOptions({
                flow: 'sync',
                operation: 'first',
                tools: TOOLS,
            }),
        );
        expect(first.tool_use).toEqual([{ id: 'native-call-17', tool_name: 'lookup', tool_input: { city: 'Tokyo' } }]);

        const second = await driver.requestTextCompletion(
            prompt([
                {
                    toolResult: {
                        toolUseId: 'native-call-17',
                        content: [{ json: { temperature: 24, conditions: 'clear' } }],
                        _llumiverse_tool_result_status: 'cancelled',
                    },
                } as unknown as ContentBlock,
            ]),
            runtimeOptions({ flow: 'sync', operation: 'second', conversation: first.conversation, tools: TOOLS }),
        );
        const document = parseConversationDocument(second.conversation);
        const callBlocks = document.turns.flatMap((turn) =>
            turn.blocks.flatMap((block) => (block.type === 'tool_call' ? [block] : [])),
        );
        const executionReceipts = Object.values(document.execution_receipts);
        const generations = Object.values(document.generations);

        expect(callBlocks).toHaveLength(1);
        expect(callBlocks[0]).toMatchObject({ call_id: 'native-call-17', tool_name: 'lookup' });
        expect(executionReceipts).toHaveLength(1);
        expect(executionReceipts[0]).toMatchObject({
            call_id: 'native-call-17',
            executor: 'application',
            status: 'cancelled',
        });
        expect(generations).toHaveLength(2);
        expect(generations.every((generation) => generation.record_source === 'executed')).toBe(true);
        expect(
            generations.every(
                (generation) =>
                    generation.request_receipt?.item_mappings.every((mapping) =>
                        mapping.kind === 'call' ? mapping.native_id === 'native-call-17' : true,
                    ) === true,
            ),
        ).toBe(true);
        expect((converse.mock.calls[1][0] as ConverseRequest).messages).toContainEqual({
            role: 'user',
            content: [
                {
                    toolResult: {
                        toolUseId: 'native-call-17',
                        content: [{ json: { temperature: 24, conditions: 'clear' } }],
                        status: 'error',
                    },
                },
            ],
        });
        expect(generations[1]?.usage).toMatchObject({
            input_tokens: 6,
            input_new_tokens: 4,
            cache_read_tokens: 1,
            cache_write_tokens: 1,
            output_tokens: 3,
            total_tokens: 9,
        });
    });

    it('records streamed native tool identity and terminal generation receipt', async () => {
        const events: ConverseStreamOutput[] = [
            {
                contentBlockStart: {
                    contentBlockIndex: 0,
                    start: { toolUse: { toolUseId: 'stream-call-9', name: 'lookup' } },
                },
            },
            { contentBlockDelta: { contentBlockIndex: 0, delta: { toolUse: { input: '{"city":"Osaka"}' } } } },
            { messageStop: { stopReason: 'tool_use' } },
            { metadata: { usage: { inputTokens: 2, outputTokens: 1, totalTokens: 3 }, metrics: { latencyMs: 1 } } },
        ];
        const converseStream = vi.fn(async () => ({
            stream: (async function* () {
                for (const event of events) yield event;
            })(),
            $metadata: { requestId: 'aws-stream-response' },
        }));
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ converseStream, destroy: vi.fn() }),
        });

        const stream = await driver.requestTextCompletionStream(
            prompt([{ text: 'Check Osaka.' }]),
            runtimeOptions({ flow: 'stream', operation: 'first', tools: TOOLS }),
        );
        for await (const _chunk of stream) {
            // Drain the native stream before finalizing its canonical response.
        }
        const document = parseConversationDocument(await stream.finalizeConversation?.());
        const generation = Object.values(document.generations)[0];
        const call = document.turns.flatMap((turn) =>
            turn.blocks.flatMap((block) => (block.type === 'tool_call' ? [block] : [])),
        )[0];

        expect(call).toMatchObject({
            type: 'tool_call',
            call_id: 'stream-call-9',
            tool_name: 'lookup',
            arguments: { type: 'json', value: { city: 'Osaka' } },
        });
        expect(generation).toMatchObject({
            provider_response_id: 'aws-stream-response',
            finish_reason: 'tool_use',
            request_id: 'request:stream:first',
        });
        expect(generation?.request_receipt?.item_mappings).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ kind: 'turn' }),
                expect.objectContaining({ kind: 'block' }),
            ]),
        );
    });
});
