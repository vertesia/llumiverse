import type {
    ContentBlock,
    ConverseRequest,
    ConverseResponse,
    ConverseStreamOutput,
} from '@aws-sdk/client-bedrock-runtime';
import { parseConversationDocument } from '@llumiverse/conversation';
import { type ExecutionOptions, PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { BedrockDriver, exportLegacyBedrockConverseConversation } from './index.js';

const MODEL = 'anthropic.claude-sonnet-4-6-v1:0';
const TOOLS = [{ name: 'lookup', description: 'Look up a value', input_schema: { type: 'object', properties: {} } }];
const RESULT_SCHEMA = {
    type: 'object' as const,
    properties: { answer: { type: 'string' as const } },
    required: ['answer'],
    additionalProperties: false,
};

function runtimeOptions(input: {
    flow: string;
    operation: string;
    attempt?: string;
    model?: string;
    conversation?: unknown;
    tools?: typeof TOOLS;
}): ExecutionOptions {
    const recordedAt = `2026-09-30T00:00:0${input.operation === 'first' ? '0' : '1'}.000Z`;
    return {
        model: input.model ?? MODEL,
        ...(input.conversation === undefined ? {} : { conversation: input.conversation }),
        ...(input.tools === undefined ? {} : { tools: input.tools }),
        conversation_runtime: {
            conversation_id: `conversation:${input.flow}`,
            request_id: `request:${input.flow}:${input.operation}`,
            attempt_id: `attempt:${input.flow}:${input.attempt ?? input.operation}`,
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
    it('executes directly into a canonical response and recovers the accepted operation without another request', async () => {
        const converse = vi.fn(
            async (): Promise<ConverseResponse> => ({
                output: { message: { role: 'assistant', content: [{ text: 'Canonical answer.' }] } },
                stopReason: 'end_turn',
                usage: { inputTokens: 3, outputTokens: 2, totalTokens: 5 },
                metrics: { latencyMs: 1 },
                serviceTier: { type: 'priority' },
            }),
        );
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ converse, destroy: vi.fn() }),
        });
        const segments = [{ role: PromptRole.user, content: 'Answer canonically.' }];
        const firstOptions = runtimeOptions({ flow: 'canonical-public-sync', operation: 'generate', attempt: 'first' });
        const first = await driver.executeCanonical(segments, firstOptions);

        expect(first.accepted_output.turn.blocks).toEqual(
            expect.arrayContaining([expect.objectContaining({ type: 'text', text: 'Canonical answer.' })]),
        );
        expect(first.accepted_output.generation.usage).toMatchObject({
            input_tokens: 3,
            output_tokens: 2,
            total_tokens: 5,
        });
        expect(first.service_tier).toBe('priority');
        expect(first).not.toHaveProperty('prompt');

        const retried = await driver.executeCanonical(segments, {
            ...runtimeOptions({
                flow: 'canonical-public-sync',
                operation: 'generate',
                attempt: 'retry',
                conversation: JSON.parse(JSON.stringify(first.conversation)),
            }),
        });
        expect(retried.accepted_output).toEqual(first.accepted_output);
        await expect(
            driver.executeCanonical(segments, {
                ...runtimeOptions({
                    flow: 'canonical-public-sync',
                    operation: 'generate',
                    attempt: 'changed-options',
                    conversation: first.conversation,
                }),
                model_options: { _option_id: 'bedrock-claude', temperature: 0.2 },
            }),
        ).rejects.toThrow('incompatible request identity');
        expect(converse).toHaveBeenCalledOnce();
    });

    it('streams directly into a canonical response with trailing usage and service tier', async () => {
        const events: ConverseStreamOutput[] = [
            { contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'Canonical ' } } },
            { contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'stream.' } } },
            { messageStop: { stopReason: 'end_turn' } },
            {
                metadata: {
                    usage: { inputTokens: 4, outputTokens: 2, totalTokens: 6 },
                    metrics: { latencyMs: 1 },
                    serviceTier: { type: 'priority' },
                },
            },
        ];
        const converseStream = vi.fn(async () => ({
            stream: (async function* () {
                for (const event of events) yield event;
            })(),
            $metadata: { requestId: 'canonical-stream-response' },
        }));
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'canStream', { value: async () => true });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ converseStream, destroy: vi.fn() }),
        });

        const stream = await driver.streamCanonical([{ role: PromptRole.user, content: 'Stream canonically.' }], {
            ...runtimeOptions({ flow: 'canonical-public-stream', operation: 'generate' }),
        });
        let preview = '';
        for await (const chunk of stream) preview += chunk;

        expect(preview).toBe('Canonical stream.');
        expect(stream.completion?.accepted_output.turn.blocks).toEqual(
            expect.arrayContaining([expect.objectContaining({ type: 'text', text: 'Canonical stream.' })]),
        );
        expect(stream.completion?.accepted_output.generation.usage).toMatchObject({
            input_tokens: 4,
            output_tokens: 2,
            total_tokens: 6,
        });
        expect(stream.completion?.service_tier).toBe('priority');
        expect(stream.completion?.chunks).toBe(2);
        expect(stream.completion).not.toHaveProperty('prompt');
    });

    it('aborts a pending native read before unwinding canonical stream cancellation', async () => {
        let providerSignal: AbortSignal | undefined;
        let readStarted: (() => void) | undefined;
        const started = new Promise<void>((resolve) => {
            readStarted = resolve;
        });
        const converseStream = vi.fn(
            async (_request: ConverseRequest, requestOptions?: { abortSignal?: AbortSignal }) => {
                providerSignal = requestOptions?.abortSignal;
                return {
                    stream: (async function* () {
                        readStarted?.();
                        await new Promise<void>((_resolve, reject) => {
                            providerSignal?.addEventListener(
                                'abort',
                                () => reject(providerSignal?.reason ?? new Error('aborted')),
                                { once: true },
                            );
                        });
                        yield { messageStop: { stopReason: 'end_turn' } };
                    })(),
                    $metadata: { requestId: 'pending-stream' },
                };
            },
        );
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'canStream', { value: async () => true });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ converseStream, destroy: vi.fn() }),
        });
        const stream = await driver.streamCanonical([{ role: PromptRole.user, content: 'Wait.' }], {
            ...runtimeOptions({ flow: 'canonical-cancel', operation: 'generate' }),
        });
        const iterator = stream[Symbol.asyncIterator]();
        const pending = iterator.next();
        await started;

        await stream.cancel();

        expect(providerSignal?.aborted).toBe(true);
        await expect(pending).rejects.toThrow();
    });

    it('marks invalid required structured output as failed for sync and stream canonical execution', async () => {
        const converse = vi.fn(
            async (): Promise<ConverseResponse> => ({
                output: { message: { role: 'assistant', content: [{ text: '{"wrong":true}' }] } },
                stopReason: 'end_turn',
                usage: { inputTokens: 2, outputTokens: 2, totalTokens: 4 },
                metrics: { latencyMs: 1 },
            }),
        );
        const streamEvents: ConverseStreamOutput[] = [
            { contentBlockDelta: { contentBlockIndex: 0, delta: { text: '{"wrong":true}' } } },
            { messageStop: { stopReason: 'end_turn' } },
            { metadata: { usage: { inputTokens: 2, outputTokens: 2, totalTokens: 4 }, metrics: { latencyMs: 1 } } },
        ];
        const converseStream = vi.fn(async () => ({
            stream: (async function* () {
                for (const event of streamEvents) yield event;
            })(),
            $metadata: { requestId: 'invalid-structured-stream' },
        }));
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'canStream', { value: async () => true });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ converse, converseStream, destroy: vi.fn() }),
        });

        const sync = await driver.executeCanonical([{ role: PromptRole.user, content: 'Return JSON.' }], {
            ...runtimeOptions({ flow: 'invalid-structured-sync', operation: 'generate' }),
            result_schema: RESULT_SCHEMA,
        });
        expect(sync.accepted_output.generation.status).toBe('failed');
        expect(sync.accepted_output.turn.status).toBe('failed');
        expect(sync.accepted_output.turn.blocks).toEqual(
            expect.arrayContaining([expect.objectContaining({ type: 'text', text: '{"wrong":true}' })]),
        );
        expect(sync.conversation.generations[sync.accepted_output.generation.id]?.metadata).toMatchObject({
            structured_output: { status: 'invalid', code: 'validation_error' },
        });

        const streamed = await driver.streamCanonical([{ role: PromptRole.user, content: 'Return JSON.' }], {
            ...runtimeOptions({ flow: 'invalid-structured-stream', operation: 'generate' }),
            result_schema: RESULT_SCHEMA,
        });
        for await (const _chunk of streamed) {
            // Drain to the canonical terminal response.
        }
        expect(streamed.completion?.accepted_output.generation.status).toBe('failed');
        expect(streamed.completion?.accepted_output.turn.status).toBe('failed');
    });

    it('normalizes structured output when the only tool use is provider-executed in sync and stream', async () => {
        const converse = vi.fn(
            async (): Promise<ConverseResponse> => ({
                output: {
                    message: {
                        role: 'assistant',
                        content: [
                            {
                                toolUse: {
                                    toolUseId: 'sync-server-tool',
                                    name: 'tool_search_tool_regex',
                                    input: { query: 'answer' },
                                    type: 'server_tool_use',
                                },
                            },
                            { text: '{"answer":"Tokyo"}' },
                        ],
                    },
                },
                stopReason: 'end_turn',
                usage: { inputTokens: 2, outputTokens: 2, totalTokens: 4 },
                metrics: { latencyMs: 1 },
            }),
        );
        const streamEvents: ConverseStreamOutput[] = [
            {
                contentBlockStart: {
                    contentBlockIndex: 0,
                    start: {
                        toolUse: {
                            toolUseId: 'stream-server-tool',
                            name: 'tool_search_tool_regex',
                            type: 'server_tool_use',
                        },
                    },
                },
            },
            { contentBlockDelta: { contentBlockIndex: 0, delta: { toolUse: { input: '{"query":"answer"}' } } } },
            { contentBlockStop: { contentBlockIndex: 0 } },
            { contentBlockDelta: { contentBlockIndex: 1, delta: { text: '{"answer":"Osaka"}' } } },
            { contentBlockStop: { contentBlockIndex: 1 } },
            { messageStop: { stopReason: 'end_turn' } },
            { metadata: { usage: { inputTokens: 2, outputTokens: 2, totalTokens: 4 }, metrics: { latencyMs: 1 } } },
        ];
        const converseStream = vi.fn(async () => ({
            stream: (async function* () {
                for (const event of streamEvents) yield event;
            })(),
            $metadata: { requestId: 'provider-tool-structured-stream' },
        }));
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'canStream', { value: async () => true });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ converse, converseStream, destroy: vi.fn() }),
        });

        const sync = await driver.executeCanonical([{ role: PromptRole.user, content: 'Return JSON.' }], {
            ...runtimeOptions({ flow: 'provider-tool-structured-sync', operation: 'generate' }),
            result_schema: RESULT_SCHEMA,
        });
        expect(sync.accepted_output.turn.blocks).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ type: 'json', value: { answer: 'Tokyo' } }),
                expect.objectContaining({ type: 'tool_call', executor: 'provider' }),
            ]),
        );

        const streamed = await driver.streamCanonical([{ role: PromptRole.user, content: 'Return JSON.' }], {
            ...runtimeOptions({ flow: 'provider-tool-structured-stream', operation: 'generate' }),
            result_schema: RESULT_SCHEMA,
        });
        for await (const _chunk of streamed) {
            // Drain through terminal canonical normalization.
        }
        expect(streamed.completion?.accepted_output.turn.blocks).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ type: 'json', value: { answer: 'Osaka' } }),
                expect.objectContaining({ type: 'tool_call', executor: 'provider' }),
            ]),
        );
    });

    it('normalizes public sync output canonically and recovers an accepted retry without another request', async () => {
        const converse = vi.fn(
            async (): Promise<ConverseResponse> => ({
                output: {
                    message: {
                        role: 'assistant',
                        content: [
                            { reasoningContent: { reasoningText: { text: 'private plan', signature: 'signed-plan' } } },
                            { text: '{ "answer" :' },
                            { text: ' "Tokyo" }' },
                        ],
                    },
                },
                stopReason: 'end_turn',
                usage: { inputTokens: 4, outputTokens: 3, totalTokens: 7 },
                metrics: { latencyMs: 1 },
                serviceTier: { type: 'priority' },
            }),
        );
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ converse, destroy: vi.fn() }),
        });
        const segments = [{ role: PromptRole.user, content: 'Return JSON.' }];
        const firstOptions = {
            ...runtimeOptions({ flow: 'public-sync', operation: 'generate', attempt: 'first' }),
            result_schema: RESULT_SCHEMA,
        };
        const first = await driver.execute(segments, firstOptions);

        expect(first.result).toEqual([{ type: 'json', value: { answer: 'Tokyo' } }]);
        expect(first.service_tier).toBe('priority');
        const document = parseConversationDocument(JSON.parse(JSON.stringify(first.conversation)));
        const generated = document.turns.find((turn) => turn.kind === 'agent' && turn.provenance.type === 'generated');
        expect(generated?.blocks).toEqual(
            expect.arrayContaining([expect.objectContaining({ type: 'json', value: { answer: 'Tokyo' } })]),
        );
        expect(generated?.blocks.filter((block) => block.type === 'json')).toHaveLength(1);
        expect(generated?.blocks.some((block) => block.type === 'text')).toBe(false);
        const replay = generated?.blocks.find((block) => block.type === 'native_replay');
        if (replay?.type !== 'native_replay' || generated === undefined) {
            throw new Error('Expected signed structured-output replay');
        }
        const generatedIndex = document.turns.findIndex((turn) => turn.id === generated.id);
        const signedPrefixTurns = document.turns.slice(0, generatedIndex + 1);
        const expectedPrefixBlocks = signedPrefixTurns.flatMap((turn) =>
            turn.blocks.flatMap((block) => [
                ...(block.type === 'native_replay' ? [] : [block.id]),
                ...(block.type === 'tool_result' ? block.content.map((nested) => nested.id) : []),
            ]),
        );
        expect(new Set(replay.dependencies.turn_ids)).toEqual(new Set(signedPrefixTurns.map((turn) => turn.id)));
        expect(new Set(replay.dependencies.block_ids)).toEqual(new Set(expectedPrefixBlocks));
        expect(
            exportLegacyBedrockConverseConversation(document, { provider: 'bedrock', model: MODEL }).messages?.at(-1),
        ).toEqual({
            role: 'assistant',
            content: [
                { reasoningContent: { reasoningText: { text: 'private plan', signature: 'signed-plan' } } },
                { text: '{ "answer" :' },
                { text: ' "Tokyo" }' },
            ],
        });
        const mutated = structuredClone(document);
        const normalized = mutated.turns
            .find((turn) => turn.kind === 'agent' && turn.provenance.type === 'generated')
            ?.blocks.find((block) => block.type === 'json');
        if (normalized?.type !== 'json') throw new Error('Expected normalized JSON block');
        normalized.value = { answer: 'Changed' };
        expect(() => exportLegacyBedrockConverseConversation(mutated, { provider: 'bedrock', model: MODEL })).toThrow(
            /Structured output replay block .* no longer matches canonical data/,
        );

        const retried = await driver.execute(segments, {
            ...runtimeOptions({
                flow: 'public-sync',
                operation: 'generate',
                attempt: 'retry',
                conversation: document,
            }),
            result_schema: RESULT_SCHEMA,
        });
        expect(retried.result).toEqual(first.result);
        expect(retried.token_usage).toEqual(first.token_usage);
        expect(converse).toHaveBeenCalledTimes(1);
    });

    it('normalizes public streamed output and retains trailing usage and service tier metadata', async () => {
        const events: ConverseStreamOutput[] = [
            { contentBlockDelta: { contentBlockIndex: 0, delta: { text: '{"answer":' } } },
            { contentBlockDelta: { contentBlockIndex: 0, delta: { text: '"Osaka"}' } } },
            { messageStop: { stopReason: 'end_turn' } },
            {
                metadata: {
                    usage: { inputTokens: 5, outputTokens: 2, totalTokens: 7 },
                    metrics: { latencyMs: 1 },
                    serviceTier: { type: 'priority' },
                },
            },
        ];
        const converseStream = vi.fn(async () => ({
            stream: (async function* () {
                for (const event of events) yield event;
            })(),
            serviceTier: { type: 'priority' },
            $metadata: { requestId: 'aws-structured-stream' },
        }));
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'canStream', { value: async () => true });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ converseStream, destroy: vi.fn() }),
        });

        const stream = await driver.stream([{ role: PromptRole.user, content: 'Return JSON.' }], {
            ...runtimeOptions({ flow: 'public-stream', operation: 'generate' }),
            result_schema: RESULT_SCHEMA,
        });
        for await (const _chunk of stream) {
            // Drain the public stream so terminal canonical finalization runs.
        }

        expect(stream.completion?.result).toEqual([{ type: 'json', value: { answer: 'Osaka' } }]);
        expect(stream.completion?.token_usage).toEqual({
            prompt: 5,
            prompt_new: 5,
            prompt_cached: 0,
            prompt_cache_write: 0,
            result: 2,
            total: 7,
        });
        expect(stream.completion?.service_tier).toBe('priority');
        const document = parseConversationDocument(stream.completion?.conversation);
        expect(document.turns.some((turn) => turn.blocks.some((block) => block.type === 'json'))).toBe(true);
        expect(
            exportLegacyBedrockConverseConversation(document, { provider: 'bedrock', model: MODEL }).messages?.at(-1),
        ).toEqual({ role: 'assistant', content: [{ text: '{"answer":"Osaka"}' }] });
    });

    it('continues a JSON-reloaded DeepSeek conversation while deliberately excluding unsigned reasoning', async () => {
        const model = 'us.deepseek.r1-v1:0';
        const requests: ConverseRequest[] = [];
        const responses: ConverseResponse[] = [
            {
                output: {
                    message: {
                        role: 'assistant',
                        content: [
                            { reasoningContent: { reasoningText: { text: 'private unsigned plan' } } },
                            { text: 'First answer.' },
                        ],
                    },
                },
                stopReason: 'end_turn',
                usage: { inputTokens: 2, outputTokens: 2, totalTokens: 4 },
                metrics: { latencyMs: 1 },
            },
            {
                output: { message: { role: 'assistant', content: [{ text: 'Second answer.' }] } },
                stopReason: 'end_turn',
                usage: { inputTokens: 3, outputTokens: 1, totalTokens: 4 },
                metrics: { latencyMs: 1 },
            },
        ];
        const converse = vi.fn(async (request: ConverseRequest) => {
            requests.push(request);
            const response = responses.shift();
            if (response === undefined) throw new Error('Unexpected Converse request');
            return response;
        });
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'getExecutor', { value: () => ({ converse, destroy: vi.fn() }) });

        const first = await driver.execute([{ role: PromptRole.user, content: 'First.' }], {
            ...runtimeOptions({ flow: 'deepseek', operation: 'first', model }),
        });
        const persisted = JSON.parse(JSON.stringify(first.conversation)) as unknown;
        const firstDocument = parseConversationDocument(persisted);
        expect(
            firstDocument.turns.some((turn) =>
                turn.blocks.some((block) => block.type === 'reasoning' && block.text === 'private unsigned plan'),
            ),
        ).toBe(true);

        await driver.execute([{ role: PromptRole.user, content: 'Continue.' }], {
            ...runtimeOptions({ flow: 'deepseek', operation: 'second', model, conversation: persisted }),
        });
        expect(requests[1].messages).toContainEqual({ role: 'assistant', content: [{ text: 'First answer.' }] });
        expect(
            requests[1].messages?.flatMap((message) => message.content ?? []).some((block) => block.reasoningContent),
        ).toBe(false);
    });

    it('continues a JSON-reloaded streamed GPT OSS conversation without replaying unsigned reasoning', async () => {
        const model = 'openai.gpt-oss-120b-1:0';
        const events: ConverseStreamOutput[] = [
            { contentBlockDelta: { contentBlockIndex: 0, delta: { reasoningContent: { text: 'unsigned plan' } } } },
            { contentBlockDelta: { contentBlockIndex: 1, delta: { text: 'Stream answer.' } } },
            { messageStop: { stopReason: 'end_turn' } },
            { metadata: { usage: { inputTokens: 2, outputTokens: 2, totalTokens: 4 }, metrics: { latencyMs: 1 } } },
        ];
        const converseStream = vi.fn(async () => ({
            stream: (async function* () {
                for (const event of events) yield event;
            })(),
            $metadata: { requestId: 'gpt-oss-stream' },
        }));
        let continuationRequest: ConverseRequest | undefined;
        const converse = vi.fn(async (request: ConverseRequest): Promise<ConverseResponse> => {
            continuationRequest = request;
            return {
                output: { message: { role: 'assistant', content: [{ text: 'Continued.' }] } },
                stopReason: 'end_turn',
                usage: { inputTokens: 3, outputTokens: 1, totalTokens: 4 },
                metrics: { latencyMs: 1 },
            };
        });
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'canStream', { value: async () => true });
        Object.defineProperty(driver, 'getExecutor', {
            value: () => ({ converseStream, converse, destroy: vi.fn() }),
        });

        const stream = await driver.stream([{ role: PromptRole.user, content: 'First.' }], {
            ...runtimeOptions({ flow: 'gpt-oss', operation: 'first', model }),
        });
        for await (const _chunk of stream) {
            // Drain and finalize the canonical conversation.
        }
        const persisted = JSON.parse(JSON.stringify(stream.completion?.conversation)) as unknown;
        const firstDocument = parseConversationDocument(persisted);
        expect(
            firstDocument.turns.some((turn) =>
                turn.blocks.some((block) => block.type === 'reasoning' && block.text === 'unsigned plan'),
            ),
        ).toBe(true);

        await driver.execute([{ role: PromptRole.user, content: 'Continue.' }], {
            ...runtimeOptions({ flow: 'gpt-oss', operation: 'second', model, conversation: persisted }),
        });
        expect(continuationRequest?.messages).toContainEqual({
            role: 'assistant',
            content: [{ text: 'Stream answer.' }],
        });
        expect(
            continuationRequest?.messages
                ?.flatMap((message) => message.content ?? [])
                .some((block) => block.reasoningContent),
        ).toBe(false);
    });

    it.each([
        [
            'missing stop reason',
            {
                output: { message: { role: 'assistant', content: [{ text: 'answer' }] } },
                usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
            } as unknown as ConverseResponse,
            /stopReason/,
        ],
        [
            'wrong output role',
            {
                output: { message: { role: 'user', content: [{ text: 'answer' }] } },
                stopReason: 'end_turn',
                usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
            } as unknown as ConverseResponse,
            /output role must be assistant/,
        ],
        [
            'empty output content',
            {
                output: { message: { role: 'assistant', content: [] } },
                stopReason: 'end_turn',
                usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
            } as unknown as ConverseResponse,
            /must contain content/,
        ],
    ])('rejects a public sync response with %s', async (_label, response, expected) => {
        const converse = vi.fn(async () => response);
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'getExecutor', { value: () => ({ converse, destroy: vi.fn() }) });

        await expect(
            driver.execute([{ role: PromptRole.user, content: 'Question.' }], {
                ...runtimeOptions({ flow: `malformed-${_label}`, operation: 'first' }),
            }),
        ).rejects.toThrow(expected);
        expect(converse).toHaveBeenCalledOnce();
    });

    it.each(['guardrail_intervened', 'content_filtered'] as const)(
        'records %s as an explicit failed generated turn',
        async (stopReason) => {
            const converse = vi.fn(
                async (): Promise<ConverseResponse> => ({
                    output: { message: { role: 'assistant', content: [{ text: 'Filtered response.' }] } },
                    stopReason,
                    usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
                    metrics: { latencyMs: 1 },
                }),
            );
            const driver = new BedrockDriver({ region: 'us-east-1' });
            Object.defineProperty(driver, 'getExecutor', { value: () => ({ converse, destroy: vi.fn() }) });

            const completion = await driver.execute([{ role: PromptRole.user, content: 'Question.' }], {
                ...runtimeOptions({ flow: stopReason, operation: 'first' }),
            });
            const document = parseConversationDocument(completion.conversation);
            expect(
                document.turns.find((turn) => turn.kind === 'agent' && turn.provenance.type === 'generated')?.status,
            ).toBe('failed');
            expect(Object.values(document.generations)[0]?.finish_reason).toBe(stopReason);
        },
    );

    it('keeps native call IDs stable across sync request and application execution receipts', async () => {
        const responses: ConverseResponse[] = [
            {
                output: {
                    message: {
                        role: 'assistant',
                        content: [
                            {
                                toolUse: {
                                    toolUseId: 'native-call-17',
                                    name: 'lookup',
                                    input: { city: 'Tokyo' },
                                },
                            },
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
            role: 'assistant',
            content: [
                {
                    toolUse: {
                        toolUseId: 'native-call-17',
                        name: 'lookup',
                        input: { city: 'Tokyo' },
                    },
                },
            ],
        });
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

    it('keeps server tool use provider-owned while exposing a following client tool call', async () => {
        const converse = vi.fn(
            async (): Promise<ConverseResponse> => ({
                output: {
                    message: {
                        role: 'assistant',
                        content: [
                            {
                                toolUse: {
                                    toolUseId: 'server-search-1',
                                    name: 'tool_search_tool_regex',
                                    input: { query: 'lookup' },
                                    type: 'server_tool_use',
                                },
                            },
                            {
                                toolUse: {
                                    toolUseId: 'application-call-1',
                                    name: 'lookup',
                                    input: { city: 'Tokyo' },
                                    type: 'tool_use',
                                },
                            } as unknown as ContentBlock,
                        ],
                    },
                },
                stopReason: 'tool_use',
                usage: { inputTokens: 3, outputTokens: 2, totalTokens: 5 },
                metrics: { latencyMs: 1 },
            }),
        );
        const driver = new BedrockDriver({ region: 'us-east-1' });
        Object.defineProperty(driver, 'getExecutor', { value: () => ({ converse, destroy: vi.fn() }) });

        const completion = await driver.requestTextCompletion(
            prompt([{ text: 'Find and call the lookup tool.' }]),
            runtimeOptions({ flow: 'typed-tools', operation: 'first', tools: TOOLS }),
        );
        const document = parseConversationDocument(completion.conversation);
        const calls = document.turns.flatMap((turn) =>
            turn.blocks.flatMap((block) => (block.type === 'tool_call' ? [block] : [])),
        );

        expect(completion.tool_use).toEqual([
            { id: 'application-call-1', tool_name: 'lookup', tool_input: { city: 'Tokyo' } },
        ]);
        expect(calls.map(({ call_id, executor }) => [call_id, executor])).toEqual([
            ['server-search-1', 'provider'],
            ['application-call-1', 'application'],
        ]);
        expect(exportLegacyBedrockConverseConversation(document).messages).toContainEqual({
            role: 'assistant',
            content: [
                {
                    toolUse: {
                        toolUseId: 'server-search-1',
                        name: 'tool_search_tool_regex',
                        input: { query: 'lookup' },
                        type: 'server_tool_use',
                    },
                },
                {
                    toolUse: {
                        toolUseId: 'application-call-1',
                        name: 'lookup',
                        input: { city: 'Tokyo' },
                        type: 'tool_use',
                    },
                },
            ],
        });
    });

    it('records streamed native tool identity and terminal generation receipt', async () => {
        const events: ConverseStreamOutput[] = [
            {
                contentBlockStart: {
                    contentBlockIndex: 0,
                    start: {
                        toolUse: { toolUseId: 'stream-call-9', name: 'lookup' },
                    },
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
        expect(exportLegacyBedrockConverseConversation(document).messages).toContainEqual({
            role: 'assistant',
            content: [
                {
                    toolUse: {
                        toolUseId: 'stream-call-9',
                        name: 'lookup',
                        input: { city: 'Osaka' },
                    },
                },
            ],
        });
    });
});
