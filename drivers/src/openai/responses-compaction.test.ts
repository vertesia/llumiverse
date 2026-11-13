import { type ExecutionOptions, PromptRole, Providers, unwrapConversationArray } from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIResponsesDriverBase } from './index.js';

class TestResponsesDriver extends OpenAIResponsesDriverBase {
    provider: OpenAIResponsesDriverBase['provider'] = Providers.openai;
    service: OpenAI;

    constructor(
        create: (request: unknown) => Promise<unknown>,
        compact: (request: unknown) => Promise<unknown> = () => Promise.reject(new Error('not expected')),
    ) {
        super({});
        this.service = { responses: { create, compact } } as unknown as OpenAI;
    }
}

const compactionItem = { id: 'cmp-1', type: 'compaction' as const, encrypted_content: 'opaque-summary' };
const messageItem = {
    id: 'msg-1',
    type: 'message' as const,
    role: 'assistant' as const,
    status: 'completed' as const,
    content: [{ type: 'output_text' as const, text: 'answer', annotations: [], logprobs: [] }],
};

function response(output: unknown[]) {
    return {
        id: 'response-1',
        object: 'response',
        created_at: 1,
        model: 'gpt-5',
        status: 'completed',
        output,
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

async function execute(compaction: ExecutionOptions['compaction'], output: unknown[] = [messageItem]) {
    const create = vi.fn(async (_request: unknown) => response(output));
    const driver = new TestResponsesDriver(create);
    const warn = vi.fn();
    driver.logger = { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() };
    const completion = await driver.execute([{ role: PromptRole.user, content: 'Next question' }], {
        model: 'gpt-5',
        compaction,
        conversation: [
            { role: 'user', content: 'Earlier question' },
            { role: 'assistant', content: 'Earlier answer' },
        ],
    });
    const request = create.mock.calls[0][0] as OpenAI.Responses.ResponseCreateParams;
    return { request, completion, warn };
}

describe('OpenAI Responses compaction', () => {
    it('requests compaction at the caller trigger', async () => {
        const { request, warn } = await execute({ trigger_tokens: 200000 });
        expect(request.context_management).toEqual([{ type: 'compaction', compact_threshold: 200000 }]);
        expect(warn).not.toHaveBeenCalled();
    });

    it("leaves the threshold to OpenAI's default when it is omitted", async () => {
        const { request } = await execute({});
        expect(request.context_management).toEqual([{ type: 'compaction' }]);
    });

    it('sends no context management without compaction', async () => {
        const { request } = await execute(undefined);
        expect(request.context_management).toBeUndefined();
    });

    it('warns that the Responses API cannot carry custom instructions', async () => {
        const { request, warn } = await execute({ instructions: 'Keep file names.' });
        expect(request).not.toHaveProperty('context_management.0.instructions');
        expect(warn).toHaveBeenCalledWith(
            { model: 'gpt-5', option_names: ['instructions'], reason: 'openai_compaction_instructions' },
            expect.any(String),
        );
    });

    it('starts the returned conversation at the compaction item', async () => {
        const { completion } = await execute({}, [compactionItem, messageItem]);
        const conversation = unwrapConversationArray(completion.conversation) ?? completion.conversation;
        expect(conversation).toEqual([compactionItem, messageItem]);
    });

    it('keeps the full conversation when no compaction ran', async () => {
        const { completion } = await execute({}, [messageItem]);
        const conversation = (unwrapConversationArray(completion.conversation) ?? completion.conversation) as unknown[];
        expect(conversation).toHaveLength(4);
    });
});

describe('OpenAI Responses on-demand compaction', () => {
    const userItem = { type: 'message' as const, role: 'user' as const, content: 'Earlier question' };
    const compacted = {
        id: 'cmp-response',
        object: 'response.compaction',
        created_at: 1,
        output: [userItem, compactionItem],
        usage: {
            input_tokens: 5000,
            output_tokens: 300,
            total_tokens: 5300,
            input_tokens_details: { cached_tokens: 0 },
        },
    };

    function setup() {
        const create = vi.fn();
        const compact = vi.fn(async (_request: unknown) => compacted);
        const driver = new TestResponsesDriver(create, compact);
        const warn = vi.fn();
        driver.logger = { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() };
        const options: ExecutionOptions = {
            model: 'gpt-5',
            compact_conversation: { instructions: 'Keep file names.' },
            compaction: { trigger_tokens: 200000 },
            conversation: [
                { role: 'user', content: 'Earlier question' },
                { role: 'assistant', content: 'Earlier answer' },
            ],
        };
        return { create, compact, driver, warn, options };
    }

    it('compacts the conversation instead of answering, and continues from the compacted output', async () => {
        const { create, compact, driver, warn, options } = setup();
        const completion = await driver.execute([], options);

        expect(create).not.toHaveBeenCalled();
        expect(compact.mock.calls[0][0]).toEqual({
            model: 'gpt-5',
            input: [
                { role: 'user', content: 'Earlier question' },
                { role: 'assistant', content: 'Earlier answer' },
            ],
            prompt_cache_key: undefined,
        });
        expect(completion.finish_reason).toBe('compaction');
        expect(completion.token_usage).toMatchObject({ prompt: 5000, result: 300, total: 5300 });
        // The user messages come before the compaction item and are kept.
        const conversation = unwrapConversationArray(completion.conversation) ?? completion.conversation;
        expect(conversation).toEqual([userItem, compactionItem]);
        expect(warn).toHaveBeenCalledWith(
            { model: 'gpt-5', option_names: ['instructions'], reason: 'openai_compaction_instructions' },
            expect.any(String),
        );
        expect(warn).toHaveBeenCalledWith(
            { model: 'gpt-5', option_names: ['compaction'], reason: 'openai_on_demand_compaction' },
            expect.any(String),
        );
    });

    it('streams the compaction as one final chunk', async () => {
        const { driver, options } = setup();
        const stream = await driver.stream([], options);
        const chunks: unknown[] = [];
        for await (const chunk of stream) chunks.push(chunk);

        expect(stream.completion?.finish_reason).toBe('compaction');
        const conversation =
            unwrapConversationArray(stream.completion?.conversation) ?? stream.completion?.conversation;
        expect(conversation).toEqual([userItem, compactionItem]);
    });
});
