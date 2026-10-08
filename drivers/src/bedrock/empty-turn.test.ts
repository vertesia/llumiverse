import type { ConverseRequest, ConverseStreamOutput, Message } from '@aws-sdk/client-bedrock-runtime';
import { describe, expect, it, vi } from 'vitest';
import { sanitizeConverseMessages } from './converse.js';
import { BedrockDriver } from './index.js';

const MODEL = 'anthropic.claude-sonnet-4-6-v1:0';

const toolTurn: Message[] = [
    { role: 'user', content: [{ text: 'question' }] },
    { role: 'assistant', content: [{ toolUse: { toolUseId: 'call-1', name: 'think', input: { thought: 'x' } } }] },
    { role: 'user', content: [{ toolResult: { toolUseId: 'call-1', content: [{ text: 'Thought recorded.' }] } }] },
];

function expectValidConverseMessages(messages: Message[] | undefined) {
    expect(messages?.length).toBeGreaterThan(0);
    messages?.forEach((message, i) => {
        expect(message.content?.length).toBeGreaterThan(0);
        for (const block of message.content ?? []) {
            if (block.text !== undefined) expect(block.text.trim()).not.toBe('');
        }
        if (i > 0) expect(message.role).not.toBe(messages[i - 1].role);
    });
}

function streamingDriver(...responses: ConverseStreamOutput[][]) {
    const converseStream = vi.fn(async (_request: ConverseRequest) => {
        const events = responses.shift() ?? [];
        return {
            stream: (async function* () {
                for (const event of events) yield event;
            })(),
        };
    });
    const driver = new BedrockDriver({ region: 'us-east-1' });
    Object.defineProperty(driver, 'getExecutor', { value: () => ({ converseStream, destroy: vi.fn() }) });
    return { driver, converseStream };
}

const usage: ConverseStreamOutput = {
    metadata: { usage: { inputTokens: 1, outputTokens: 9, totalTokens: 10 }, metrics: { latencyMs: 1 } },
};

describe('sanitizeConverseMessages', () => {
    it('drops empty messages and blank text blocks, then merges same-role neighbours', () => {
        const messages: Message[] = [
            ...toolTurn,
            { role: 'assistant', content: [] },
            { role: 'user', content: [{ text: 'continue' }] },
            { role: 'assistant', content: [{ text: '  \n' }, { text: 'answer' }] },
        ];

        expect(sanitizeConverseMessages(messages)).toEqual([
            toolTurn[0],
            toolTurn[1],
            { role: 'user', content: [...(toolTurn[2].content ?? []), { text: 'continue' }] },
            { role: 'assistant', content: [{ text: 'answer' }] },
        ]);
    });

    it('returns valid conversations unchanged', () => {
        const messages = [...toolTurn, { role: 'assistant', content: [{ text: 'done' }] } satisfies Message];
        const sanitized = sanitizeConverseMessages(messages);
        expect(sanitized).toBe(messages);
        sanitized.forEach((message, i) => {
            expect(message).toBe(messages[i]);
        });
    });
});

describe('Bedrock empty assistant turns', () => {
    it('does not persist a content-filtered response that produced no blocks', async () => {
        const { driver, converseStream } = streamingDriver(
            [{ messageStart: { role: 'assistant' } }, { messageStop: { stopReason: 'content_filtered' } }, usage],
            [
                { contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'resumed' } } },
                { messageStop: { stopReason: 'end_turn' } },
                usage,
            ],
        );

        const first = await driver.requestTextCompletionStream(
            { modelId: MODEL, messages: structuredClone(toolTurn) },
            { model: MODEL },
        );
        for await (const _chunk of first) {
            // drain
        }
        const conversation = (await first.finalizeConversation?.()) as ConverseRequest;
        expect(conversation.messages?.at(-1)?.role).toBe('user');
        expectValidConverseMessages(conversation.messages);

        const second = await driver.requestTextCompletionStream(
            { modelId: MODEL, messages: [{ role: 'user', content: [{ text: 'The plan is not completed.' }] }] },
            { model: MODEL, conversation: JSON.parse(JSON.stringify(conversation)) },
        );
        for await (const _chunk of second) {
            // drain
        }
        const request = converseStream.mock.calls[1][0];
        expectValidConverseMessages(request.messages);
        // The tool result and the follow-up instruction now share the final user turn.
        expect(request.messages?.map((message) => message.role)).toEqual(['user', 'assistant', 'user']);
        expect(request.messages?.at(-1)?.content?.at(-1)).toEqual({ text: 'The plan is not completed.' });
    });

    it('repairs a stored conversation that already contains an empty assistant turn', async () => {
        const { driver, converseStream } = streamingDriver([
            { contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'ok' } } },
            { messageStop: { stopReason: 'end_turn' } },
            usage,
        ]);
        const stored: ConverseRequest = { modelId: MODEL, messages: [...toolTurn, { role: 'assistant', content: [] }] };

        const stream = await driver.requestTextCompletionStream(
            { modelId: MODEL, messages: [{ role: 'user', content: [{ text: 'continue' }] }] },
            { model: MODEL, conversation: stored },
        );
        for await (const _chunk of stream) {
            // drain
        }

        expectValidConverseMessages(converseStream.mock.calls[0][0].messages);
        const persisted = (await stream.finalizeConversation?.()) as ConverseRequest | undefined;
        expectValidConverseMessages(persisted?.messages);
    });

    it('keeps a real tool result that an empty turn separated from its tool use', async () => {
        const { driver, converseStream } = streamingDriver([
            { contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'ok' } } },
            { messageStop: { stopReason: 'end_turn' } },
            usage,
        ]);
        const stored: ConverseRequest = {
            modelId: MODEL,
            messages: [toolTurn[0], toolTurn[1], { role: 'assistant', content: [] }, toolTurn[2]],
        };

        const stream = await driver.requestTextCompletionStream(
            { modelId: MODEL, messages: [{ role: 'user', content: [{ text: 'continue' }] }] },
            {
                model: MODEL,
                conversation: stored,
                tools: [{ name: 'think', description: 'Record a thought', input_schema: { type: 'object' } }],
            },
        );
        for await (const _chunk of stream) {
            // drain
        }

        const request = converseStream.mock.calls[0][0];
        expectValidConverseMessages(request.messages);
        expect(request.messages?.at(-1)?.content).toEqual([...(toolTurn[2].content ?? []), { text: 'continue' }]);
    });

    it('does not persist a blank assistant turn when a non-streaming response has no message', async () => {
        const converse = vi.fn(async (_request: ConverseRequest) => ({
            output: undefined,
            stopReason: 'content_filtered',
            usage: { inputTokens: 1, outputTokens: 0, totalTokens: 1 },
            metrics: { latencyMs: 1 },
        }));
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'getExecutor', { value: () => ({ converse, destroy: vi.fn() }) });

        const completion = await driver.requestTextCompletion(
            { modelId: MODEL, messages: structuredClone(toolTurn) },
            { model: MODEL },
        );

        expectValidConverseMessages((completion.conversation as ConverseRequest).messages);
        expect((completion.conversation as ConverseRequest).messages?.at(-1)?.role).toBe('user');
    });
});
