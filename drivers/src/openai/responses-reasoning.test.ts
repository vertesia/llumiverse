import { Providers } from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIResponsesDriverBase } from './index.js';

class TestResponsesDriver extends OpenAIResponsesDriverBase {
    provider: Providers.openai = Providers.openai;
    service: OpenAI;

    constructor(create: (request: unknown) => Promise<unknown>) {
        super({});
        this.service = { responses: { create } } as unknown as OpenAI;
    }
}

const reasoningItem = {
    id: 'reason-1',
    type: 'reasoning' as const,
    summary: [{ type: 'summary_text' as const, text: 'visible plan' }],
    encrypted_content: 'encrypted-replay-state',
    status: 'completed' as const,
};
const messageItem = {
    id: 'msg-1',
    type: 'message' as const,
    role: 'assistant' as const,
    status: 'completed' as const,
    content: [{ type: 'output_text' as const, text: 'answer', annotations: [], logprobs: [] }],
};

function response() {
    return {
        id: 'response-1',
        object: 'response',
        created_at: 1,
        model: 'gpt-5',
        status: 'completed',
        output: [reasoningItem, messageItem],
        output_text: 'answer',
        parallel_tool_calls: true,
        tool_choice: 'auto',
        tools: [],
        error: null,
        incomplete_details: null,
        instructions: null,
        metadata: null,
        temperature: null,
        top_p: null,
        usage: { input_tokens: 1, output_tokens: 2, total_tokens: 3, input_tokens_details: { cached_tokens: 0 } },
    } as unknown as OpenAI.Responses.Response;
}

describe('OpenAI Responses reasoning', () => {
    it.each([
        'gpt-5.4',
        'gpt-5.5',
        'gpt-5.6',
        'gpt-5.6-sol',
    ])('uses current-turn reasoning context for %s', async (model) => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create);

        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
            model,
            model_options: { _option_id: 'openai-thinking' },
        });

        expect(create).toHaveBeenCalledWith(
            expect.objectContaining({
                reasoning: expect.objectContaining({ context: 'current_turn' }),
            }),
        );
    });

    it('does not request cross-turn reasoning controls for models without documented support', async () => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create);

        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'question' }], {
            model: 'gpt-5',
            model_options: { _option_id: 'openai-thinking' },
        });

        expect(create.mock.calls[0][0]).toMatchObject({ reasoning: { summary: 'auto' } });
        expect((create.mock.calls[0][0] as { reasoning: Record<string, unknown> }).reasoning).not.toHaveProperty(
            'context',
        );
    });

    it('projects reasoning by default and replays the exact encrypted output item after JSON roundtrip', async () => {
        const create = vi.fn(async (_request: unknown) => response());
        const driver = new TestResponsesDriver(create);
        const prompt = [{ type: 'message', role: 'user', content: 'question' }] as OpenAI.Responses.ResponseInputItem[];

        const first = await driver.requestTextCompletion(prompt, {
            model: 'gpt-5',
            model_options: { _option_id: 'openai-thinking' },
        });
        expect(first.result).toEqual([
            { type: 'thoughts', value: 'visible plan' },
            { type: 'text', value: 'answer' },
        ]);
        expect(create).toHaveBeenCalledWith(expect.objectContaining({ include: ['reasoning.encrypted_content'] }));

        const persisted = JSON.parse(JSON.stringify(first.conversation));
        await driver.requestTextCompletion([{ type: 'message', role: 'user', content: 'continue' }], {
            model: 'gpt-5',
            model_options: { _option_id: 'openai-thinking' },
            conversation: persisted,
        });
        expect(create.mock.calls[1][0]).toMatchObject({ input: expect.arrayContaining([reasoningItem]) });

        const hidden = await driver.requestTextCompletion(prompt, {
            model: 'gpt-5',
            model_options: { _option_id: 'openai-thinking', include_thoughts: false },
        });
        expect(hidden.result).toEqual([{ type: 'text', value: 'answer' }]);
        expect(JSON.stringify(hidden.conversation)).toContain('encrypted-replay-state');
    });

    it('streams reasoning separately and finalizes from the authoritative response output', async () => {
        const final = response();
        const create = vi.fn(async () =>
            (async function* () {
                yield {
                    type: 'response.reasoning_summary_text.delta',
                    item_id: 'reason-1',
                    output_index: 0,
                    summary_index: 0,
                    sequence_number: 1,
                    delta: 'visible plan',
                };
                yield {
                    type: 'response.output_text.delta',
                    item_id: 'msg-1',
                    output_index: 1,
                    content_index: 0,
                    sequence_number: 2,
                    delta: 'answer',
                    logprobs: [],
                };
                yield { type: 'response.completed', sequence_number: 3, response: final };
            })(),
        );
        const driver = new TestResponsesDriver(create);
        const stream = await driver.requestTextCompletionStream(
            [{ type: 'message', role: 'user', content: 'question' }],
            { model: 'gpt-5', model_options: { _option_id: 'openai-thinking' } },
        );
        const results = [];
        for await (const chunk of stream) results.push(...chunk.result);
        const conversation = await stream.finalizeConversation?.({ result: results });

        expect(results).toEqual([
            { type: 'thoughts', value: 'visible plan' },
            { type: 'text', value: 'answer' },
        ]);
        expect(JSON.stringify(conversation)).toContain('encrypted-replay-state');
    });
});
