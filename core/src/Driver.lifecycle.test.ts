import {
    type AIModel,
    type Completion,
    type CompletionChunkObject,
    type CompletionStream,
    type DriverCompletionStream,
    type DriverOptions,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    type ModelSearchPayload,
    type PromptOptions,
    PromptRole,
    type PromptSegment,
} from '@llumiverse/common';
import {
    appendConversationRecords,
    createConversationDocument,
    parseConversationDocument,
} from '@llumiverse/conversation';
import { describe, expect, it, vi } from 'vitest';
import {
    CanonicalAcceptedOutputRecovered,
    type CanonicalExecutionContextInputOptions,
    type CanonicalExecutionContextOptions,
    type CanonicalExecutionInputOptions,
    type CanonicalExecutionOptions,
    type CanonicalExecutionStream,
    createCanonicalExecutionResponse,
} from './CanonicalExecution.js';
import {
    type CanonicalExecutionEventStream,
    type CanonicalStreamOpenOptions,
    type CanonicalStreamTerminalEvent,
    FallbackCanonicalExecutionEventStream,
} from './CanonicalStreaming.js';
import { DEFAULT_COMPLETION_STREAM_START_TIMEOUT_MS, leaseCompletionStream } from './CompletionStream.js';
import { AbstractDriver } from './Driver.js';

class LifecycleTestDriver extends AbstractDriver<DriverOptions, string> {
    provider = 'lifecycle-test';
    completion = Promise.resolve<Completion>({ result: [{ type: 'text', value: 'done' }] });
    imageCompletion = Promise.resolve<Completion>({ result: [{ type: 'image', value: 'image-data' }] });
    models = Promise.resolve<AIModel[]>([]);
    embeddings = Promise.resolve<EmbeddingsResult>({ results: [], model: 'test-model' });
    imageModel = false;
    streaming = true;
    completionSignal?: AbortSignal;
    waitForCompletionAbort = false;
    completionStreamSignal?: AbortSignal;
    completionStream: DriverCompletionStream = {
        async *[Symbol.asyncIterator]() {
            yield { result: [{ type: 'text', value: 'first' }] } satisfies CompletionChunkObject;
        },
    };
    requestTextCompletionCalls = 0;
    requestTextCompletionStreamCalls = 0;
    createPromptCalls = 0;

    constructor(
        private readonly cleanup: () => void,
        options: DriverOptions = {},
    ) {
        super(options);
    }

    async requestTextCompletion(
        _prompt: string,
        _options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        this.requestTextCompletionCalls += 1;
        this.completionSignal = signal;
        if (this.waitForCompletionAbort) {
            return new Promise((_resolve, reject) => {
                signal?.addEventListener('abort', () => reject(signal.reason), {
                    once: true,
                });
            });
        }
        return this.completion;
    }

    async requestTextCompletionStream(
        _prompt: string,
        _options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        this.requestTextCompletionStreamCalls += 1;
        this.completionStreamSignal = signal;
        return this.completionStream;
    }

    override async createPrompt(segments: PromptSegment[], opts: PromptOptions): Promise<string> {
        this.createPromptCalls += 1;
        return await super.createPrompt(segments, opts);
    }

    async requestImageGeneration(_prompt: string, _options: ExecutionOptions): Promise<Completion> {
        return this.imageCompletion;
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel[]> {
        return this.models;
    }

    async validateConnection(): Promise<boolean> {
        return true;
    }

    async generateEmbeddings(_options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        return this.embeddings;
    }

    protected override destroyProviderResources(): void {
        this.cleanup();
    }

    protected override isImageModel(_model: string): boolean {
        return this.imageModel;
    }

    protected override canStream(_options?: ExecutionOptions, _signal?: AbortSignal): Promise<boolean> {
        return Promise.resolve(this.streaming);
    }
}

class CanonicalLifecycleTestDriver extends LifecycleTestDriver {
    protected override supportsCanonicalConversation(_options: ExecutionOptions): boolean {
        return true;
    }
}

class FiniteCanonicalLifecycleTestDriver extends LifecycleTestDriver {
    canonicalCalls = 0;

    protected override supportsCanonicalConversation(_options: ExecutionOptions): boolean {
        return true;
    }

    protected override supportsCanonicalImageGeneration(_options: ExecutionOptions): boolean {
        return true;
    }

    override async requestCanonicalImageGeneration(
        _prompt: string,
        options: ExecutionOptions,
    ): Promise<ReturnType<typeof createCanonicalExecutionResponse>> {
        this.canonicalCalls += 1;
        if (options.model !== 'test-model') throw new Error('changed request target');
        return createCanonicalExecutionResponse(parseConversationDocument(options.conversation), 'response-operation');
    }
}

class RecoveredCanonicalLifecycleTestDriver extends CanonicalLifecycleTestDriver {
    recovery: CanonicalAcceptedOutputRecovered | undefined;

    override async _executeCanonical(): Promise<never> {
        if (!this.recovery) throw new Error('missing accepted recovery fixture');
        throw this.recovery;
    }
}

class CanonicalOptionSnapshotDriver extends CanonicalLifecycleTestDriver {
    capturedOptions: CanonicalExecutionOptions | undefined;

    override async requestCanonicalTextCompletion(
        _prompt: string,
        options: CanonicalExecutionOptions,
    ): Promise<ReturnType<typeof createCanonicalExecutionResponse>> {
        this.capturedOptions = options;
        return createCanonicalExecutionResponse(acceptedFiniteDocument(), 'response-operation');
    }
}

class CanonicalContextSnapshotDriver extends CanonicalLifecycleTestDriver {
    capturedOptions: CanonicalExecutionContextOptions | undefined;
    canonicalContextEventCalls = 0;

    protected override supportsCanonicalContextConversation(_options: CanonicalExecutionContextOptions): boolean {
        return true;
    }

    override async requestCanonicalContextCompletion(
        options: CanonicalExecutionContextOptions,
    ): Promise<ReturnType<typeof createCanonicalExecutionResponse>> {
        this.capturedOptions = options;
        return createCanonicalExecutionResponse(acceptedFiniteDocument(), 'response-operation');
    }

    override async requestCanonicalContextCompletionEventStream(
        options: CanonicalExecutionContextOptions,
        _signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        this.capturedOptions = options;
        this.canonicalContextEventCalls += 1;
        const response = createCanonicalExecutionResponse(acceptedFiniteDocument(), 'response-operation');
        return new FallbackCanonicalExecutionEventStream(
            {
                request_id: response.accepted_output.generation.request_id,
                attempt_id: response.accepted_output.generation.attempt_id,
                response_operation_id: options.conversation_runtime.response_operation_id,
                generation_id: response.accepted_output.generation.id,
                draft_turn_id: response.accepted_output.turn.id,
            },
            async () => response,
            open,
        );
    }
}

class TypedCanonicalLifecycleTestDriver extends CanonicalLifecycleTestDriver {
    canonicalEventCalls = 0;
    overriddenStringStreamCalls = 0;
    openedStreamId: string | undefined;

    override async streamCanonical(
        _segments: PromptSegment[],
        _options: ExecutionOptions,
        _signal?: AbortSignal,
    ): Promise<CanonicalExecutionStream> {
        this.overriddenStringStreamCalls += 1;
        throw new Error('legacy CompletionChunkObject canonical stream selected');
    }

    override async requestCanonicalTextCompletionEventStream(
        _prompt: string,
        options: CanonicalExecutionOptions,
        _signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        this.canonicalEventCalls += 1;
        this.openedStreamId = open.stream_id;
        const response = createCanonicalExecutionResponse(options.conversation, 'response-operation');
        return new FallbackCanonicalExecutionEventStream(
            {
                request_id: response.accepted_output.generation.request_id,
                attempt_id: response.accepted_output.generation.attempt_id,
                response_operation_id: options.conversation_runtime.response_operation_id,
                generation_id: response.accepted_output.generation.id,
                draft_turn_id: response.accepted_output.turn.id,
            },
            async () => response,
            open,
        );
    }
}

class OverriddenStreamDriver extends LifecycleTestDriver {
    readonly cancelStream = vi.fn().mockResolvedValue(undefined);

    override async stream(
        _segments: PromptSegment[],
        _options: ExecutionOptions,
        _signal?: AbortSignal,
    ): Promise<CompletionStream<string>> {
        return {
            completion: undefined,
            cancel: this.cancelStream,
            async *[Symbol.asyncIterator]() {
                yield 'custom';
            },
        };
    }
}

class PendingStreamCreationDriver extends LifecycleTestDriver {
    protected override canStream(_options?: ExecutionOptions, signal?: AbortSignal): Promise<boolean> {
        return new Promise((_resolve, reject) => {
            signal?.addEventListener('abort', () => reject(signal.reason), { once: true });
        });
    }
}

class ThrowingIteratorDriver extends LifecycleTestDriver {
    readonly cancelStream = vi.fn().mockResolvedValue(undefined);

    override async stream(_segments: PromptSegment[], _options: ExecutionOptions): Promise<CompletionStream<string>> {
        return {
            completion: undefined,
            cancel: this.cancelStream,
            [Symbol.asyncIterator]() {
                throw new Error('iterator creation failed');
            },
        };
    }
}

const canonicalTerminal = { type: 'stream_terminated' } as CanonicalStreamTerminalEvent;

class OverriddenCanonicalEventStreamDriver extends CanonicalLifecycleTestDriver {
    readonly cancelEventStream = vi
        .fn<() => Promise<CanonicalStreamTerminalEvent>>()
        .mockResolvedValue(canonicalTerminal);
    releaseIteratorReturn?: () => void;
    releaseClosed?: () => void;
    holdClosed = false;
    private releaseRead?: () => void;

    finishRead(): boolean {
        const release = this.releaseRead;
        if (release === undefined) return false;
        this.releaseRead = undefined;
        release();
        return true;
    }

    override async streamCanonicalEvents(
        _segments: PromptSegment[],
        _options: ExecutionOptions,
        _signal: AbortSignal | undefined,
        _open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        const driver = this;
        const closed = this.holdClosed
            ? new Promise<void>((resolve) => {
                  this.releaseClosed = resolve;
              })
            : Promise.resolve();
        return {
            completion: undefined,
            terminal_event: undefined,
            execution_started: false,
            closed,
            cancel: this.cancelEventStream,
            [Symbol.asyncIterator]() {
                return {
                    next: () =>
                        new Promise<IteratorResult<never>>((resolve) => {
                            driver.releaseRead = () => resolve({ done: true, value: undefined });
                        }),
                    return: async () => {
                        driver.releaseRead?.();
                        await new Promise<void>((resolve) => {
                            driver.releaseIteratorReturn = resolve;
                        });
                        return { done: true, value: undefined };
                    },
                };
            },
        };
    }
}

const segments = [{ role: PromptRole.user, content: 'hello' }];
const options = { model: 'test-model' };
const canonicalStreamOpen = { stream_id: 'stream:lifecycle' };
const RECORDED_AT = '2026-09-30T00:00:00.000Z';

function acceptedFiniteDocument() {
    const initial = createConversationDocument({ id: 'conversation:finite', created_at: RECORDED_AT });
    const requestReceipt = {
        id: 'request-receipt',
        request_id: 'request:original',
        attempt_id: 'attempt:original',
        source: { conversation_id: initial.id, revision: initial.revision },
        context_fingerprint: 'sha256:context',
        tool_set_fingerprint: 'sha256:tools',
        request_fingerprint: 'sha256:request',
        target: {
            provider: 'lifecycle-test',
            protocol: 'test.finite',
            model: 'test-model',
            adapter_version: 'test',
        },
        tool_definition_ids: [],
        asset_versions: [],
        item_mappings: [],
        recorded_at: RECORDED_AT,
    };
    const generation = {
        id: 'generation:original',
        record_source: 'executed' as const,
        request_id: 'request:original',
        attempt_id: 'attempt:original',
        purpose: 'interaction',
        requested_model: 'test-model',
        provider: 'lifecycle-test',
        protocol: 'test.finite',
        adapter_version: 'test',
        status: 'completed' as const,
        finish_reason: 'stop',
        timestamps: { recorded_at: RECORDED_AT, completed_at: RECORDED_AT },
        source: { conversation_id: initial.id, revision: initial.revision },
        request_receipt: requestReceipt,
    };
    const turn = {
        id: 'turn:original',
        kind: 'agent' as const,
        authority: 'ordinary' as const,
        blocks: [{ id: 'block:original', type: 'text' as const, text: 'accepted', format: 'plain' as const }],
        status: 'completed' as const,
        timestamps: { recorded_at: RECORDED_AT, completed_at: RECORDED_AT },
        provenance: { type: 'generated' as const },
        model_visibility: 'include' as const,
        generation_id: generation.id,
    };
    return appendConversationRecords(
        initial,
        { turns: [turn], generations: [generation] },
        {
            expected_revision: initial.revision,
            operation_id: 'response-operation',
            payload_fingerprint: 'sha256:response',
            recorded_at: RECORDED_AT,
        },
    ).document;
}

function canonicalOptions(
    conversation = createConversationDocument({ id: 'conversation:lifecycle', created_at: RECORDED_AT }),
): CanonicalExecutionInputOptions {
    return {
        model: 'test-model',
        conversation,
        conversation_runtime: {
            conversation_id: conversation.id,
            request_id: 'request:lifecycle',
            attempt_id: 'attempt:lifecycle',
            input_operation_id: 'input:lifecycle',
            response_operation_id: 'response-operation',
            recorded_at: RECORDED_AT,
        },
    };
}

function canonicalContextOptions(conversation = acceptedFiniteDocument()): CanonicalExecutionContextInputOptions {
    return {
        model: 'test-model',
        conversation,
        conversation_runtime: {
            conversation_id: conversation.id,
            request_id: 'request:context',
            attempt_id: 'attempt:context',
            input_operation_id: 'input:context',
            response_operation_id: 'response-operation',
            recorded_at: RECORDED_AT,
        },
    };
}

function holdStreamCancellation(driver: OverriddenStreamDriver): () => void {
    let release!: () => void;
    driver.cancelStream.mockImplementationOnce(
        () =>
            new Promise<void>((resolve) => {
                release = resolve;
            }),
    );
    return () => release();
}

describe('AbstractDriver lifecycle', () => {
    it('does not wrap a host accepted-output recovery as a provider failure', async () => {
        const driver = new RecoveredCanonicalLifecycleTestDriver(vi.fn());
        const response = createCanonicalExecutionResponse(acceptedFiniteDocument(), 'response-operation');
        const recovery = new CanonicalAcceptedOutputRecovered({ accepted_output: response.accepted_output });
        driver.recovery = recovery;

        await expect(
            driver.executeCanonical(
                [],
                canonicalOptions(createConversationDocument({ id: 'conversation:new', created_at: RECORDED_AT })),
            ),
        ).rejects.toBe(recovery);
    });

    it('preserves accepted recovery through the canonical stream lease and releases driver resources', async () => {
        const cleanup = vi.fn();
        const driver = new RecoveredCanonicalLifecycleTestDriver(cleanup);
        driver.streaming = false;
        const response = createCanonicalExecutionResponse(acceptedFiniteDocument(), 'response-operation');
        const recovery = new CanonicalAcceptedOutputRecovered({ accepted_output: response.accepted_output });
        driver.recovery = recovery;

        const stream = await driver.streamCanonical(
            [],
            canonicalOptions(createConversationDocument({ id: 'conversation:new', created_at: RECORDED_AT })),
        );
        const chunks: string[] = [];
        for await (const chunk of stream) chunks.push(chunk);

        expect(chunks).toEqual([]);
        expect(stream.completion).toBeUndefined();
        expect(stream.accepted_recovery).toBe(recovery);
        driver.destroy();
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('delivers a finite accepted retry under its retained identity and rejects a changed target', async () => {
        const driver = new FiniteCanonicalLifecycleTestDriver(vi.fn());
        driver.imageModel = true;
        driver.streaming = false;
        const conversation = acceptedFiniteDocument();
        const recoveredOptions: CanonicalExecutionInputOptions = {
            model: 'test-model',
            conversation,
            conversation_runtime: {
                conversation_id: conversation.id,
                request_id: 'request:retry',
                attempt_id: 'attempt:retry',
                input_operation_id: 'input-operation',
                response_operation_id: 'response-operation',
                recorded_at: RECORDED_AT,
            },
        };
        const recovered = await driver.streamCanonicalEvents(segments, recoveredOptions, undefined, {
            stream_id: 'stream:finite:retry',
        });
        const recoveredEvents = [];
        for await (const event of recovered) recoveredEvents.push(event);
        expect(recovered.execution_started).toBe(false);
        expect(recoveredEvents).toEqual([
            expect.objectContaining({
                type: 'response_accepted',
                origin: 'accepted_recovery',
                request_id: 'request:original',
                attempt_id: 'attempt:original',
            }),
        ]);

        const changed = await driver.streamCanonicalEvents(
            segments,
            { ...recoveredOptions, model: 'changed-model' },
            undefined,
            { stream_id: 'stream:finite:changed-target' },
        );
        const changedEvents = [];
        for await (const event of changed) changedEvents.push(event);
        expect(changedEvents).toEqual([expect.objectContaining({ type: 'stream_terminated', outcome: 'failed' })]);
        expect(changedEvents.some((event) => event.type === 'response_accepted')).toBe(false);
        expect(driver.canonicalCalls).toBe(2);
    });

    it('rejects canonical input on unadopted drivers before invoking provider methods', async () => {
        const driver = new LifecycleTestDriver(vi.fn());
        const conversation = createConversationDocument({
            id: 'conversation:unsupported',
            created_at: '2026-09-11T00:00:00.000Z',
        });
        const canonicalOptions = { ...options, conversation };

        await expect(driver.execute(segments, canonicalOptions)).rejects.toThrow(
            'Provider lifecycle-test model test-model does not support canonical conversation input',
        );
        await expect(driver.stream(segments, canonicalOptions)).rejects.toThrow(
            'Provider lifecycle-test model test-model does not support canonical conversation input',
        );
        expect(driver.requestTextCompletionCalls).toBe(0);
        expect(driver.requestTextCompletionStreamCalls).toBe(0);
        await expect(driver.supportsCanonicalExecution(canonicalOptions)).resolves.toBe(false);
    });

    it('rejects new segments with a materialized canonical input before preparing a provider request', async () => {
        const driver = new CanonicalLifecycleTestDriver(vi.fn());
        const conversation = createConversationDocument({
            id: 'conversation:materialized',
            created_at: '2026-09-30T00:00:00.000Z',
        });
        const canonicalOptions: CanonicalExecutionInputOptions = {
            ...options,
            conversation,
            conversation_runtime: {
                conversation_id: conversation.id,
                request_id: 'request:materialized',
                attempt_id: 'attempt:materialized',
                input_operation_id: 'operation:unused-input',
                response_operation_id: 'operation:response',
                recorded_at: '2026-09-30T00:00:00.000Z',
                materialized_input: { operation_id: 'operation:materialized-input', result_revision: 1 },
            },
        };

        await expect(driver.supportsCanonicalExecution(canonicalOptions)).resolves.toBe(true);

        await expect(driver.executeCanonical(segments, canonicalOptions)).rejects.toThrow(
            'A materialized canonical input requires an empty new prompt',
        );
        await expect(driver.streamCanonical(segments, canonicalOptions)).rejects.toThrow(
            'A materialized canonical input requires an empty new prompt',
        );
        expect(driver.createPromptCalls).toBe(0);
        expect(driver.requestTextCompletionCalls).toBe(0);
        expect(driver.requestTextCompletionStreamCalls).toBe(0);
    });

    it('rejects missing identity and opaque native history before prompt preparation or provider dispatch', async () => {
        const driver = new TypedCanonicalLifecycleTestDriver(vi.fn());
        const nativeHistory = { messages: [{ role: 'user', content: 'legacy' }] };
        const runtime = {
            conversation_id: 'conversation:boundary',
            request_id: 'request:boundary',
            attempt_id: 'attempt:boundary',
            input_operation_id: 'input:boundary',
            response_operation_id: 'response:boundary',
            recorded_at: RECORDED_AT,
        };

        await expect(
            driver.executeCanonical([], { model: 'test-model' } as unknown as CanonicalExecutionInputOptions),
        ).rejects.toThrow();
        await expect(
            driver.streamCanonical([], {
                model: 'test-model',
                conversation: nativeHistory,
                conversation_runtime: runtime,
            } as unknown as CanonicalExecutionInputOptions),
        ).rejects.toThrow('Conversation document validation failed');
        await expect(
            driver.streamCanonicalEvents(
                [],
                {
                    model: 'test-model',
                    conversation: nativeHistory,
                    conversation_runtime: runtime,
                } as unknown as CanonicalExecutionInputOptions,
                undefined,
                { stream_id: 'stream:boundary' },
            ),
        ).rejects.toThrow('Conversation document validation failed');

        expect(driver.createPromptCalls).toBe(0);
        expect(driver.requestTextCompletionCalls).toBe(0);
        expect(driver.requestTextCompletionStreamCalls).toBe(0);
        expect(driver.canonicalEventCalls).toBe(0);
        expect(driver.overriddenStringStreamCalls).toBe(0);
    });

    it('snapshots canonical identity, document, tools and JSON options before the first await', async () => {
        const driver = new CanonicalOptionSnapshotDriver(vi.fn());
        const conversation = createConversationDocument({ id: 'conversation:snapshot', created_at: RECORDED_AT });
        const runtime = {
            conversation_id: conversation.id,
            request_id: 'request:snapshot',
            attempt_id: 'attempt:snapshot',
            input_operation_id: 'input:snapshot',
            response_operation_id: 'response:snapshot',
            recorded_at: RECORDED_AT,
        };
        const tools = [
            {
                name: 'lookup',
                input_schema: {
                    type: 'object',
                    properties: { city: { type: 'string' } },
                    required: ['city'],
                    additionalProperties: false,
                },
            },
        ];
        const resultSchema = {
            type: 'object' as const,
            properties: { answer: { type: 'string' as const } },
            required: ['answer'],
            additionalProperties: false,
        };
        const execution = driver.executeCanonical([], {
            model: 'test-model',
            conversation,
            conversation_runtime: runtime,
            tools,
            result_schema: resultSchema,
            model_options: { temperature: 0.2 },
            labels: { purpose: 'snapshot' },
        });

        conversation.id = 'conversation:mutated';
        runtime.request_id = 'request:mutated';
        tools[0].name = 'mutated';
        tools[0].input_schema.properties.city.type = 'number';
        resultSchema.properties.answer.type = 'number' as 'string';

        await execution;
        expect(driver.capturedOptions).toMatchObject({
            conversation: { id: 'conversation:snapshot' },
            conversation_runtime: { request_id: 'request:snapshot', purpose: 'conversation' },
            tools: [{ name: 'lookup', input_schema: { properties: { city: { type: 'string' } } } }],
            result_schema: { properties: { answer: { type: 'string' } } },
            model_options: { temperature: 0.2 },
            labels: { purpose: 'snapshot' },
        });
        expect(driver.capturedOptions?.conversation).not.toBe(conversation);
        expect(driver.capturedOptions?.conversation_runtime).not.toBe(runtime);
        expect(driver.capturedOptions?.tools).not.toBe(tools);
    });

    it('creates a deterministic empty canonical document from the validated runtime identity', async () => {
        const driver = new CanonicalOptionSnapshotDriver(vi.fn());
        await driver.executeCanonical([], {
            model: 'test-model',
            conversation_runtime: {
                conversation_id: 'conversation:new',
                request_id: 'request:new',
                attempt_id: 'attempt:new',
                input_operation_id: 'input:new',
                response_operation_id: 'response:new',
                recorded_at: RECORDED_AT,
            },
        });

        expect(driver.capturedOptions?.conversation).toMatchObject({
            id: 'conversation:new',
            created_at: RECORDED_AT,
            revision: 0,
        });
        expect(driver.capturedOptions?.conversation_runtime).toMatchObject({
            conversation_id: 'conversation:new',
            purpose: 'conversation',
        });
    });

    it('executes a synchronously owned canonical context without prompt authoring', async () => {
        const driver = new CanonicalContextSnapshotDriver(vi.fn());
        const conversation = acceptedFiniteDocument();
        const contextOptions = canonicalContextOptions(conversation);
        const resultSchema = {
            type: 'object' as const,
            properties: { answer: { type: 'string' as const } },
            required: ['answer'],
            additionalProperties: false,
        };
        contextOptions.result_schema = resultSchema;
        contextOptions.model_options = { temperature: 0.2 };
        contextOptions.labels = { purpose: 'context-snapshot' };

        const execution = driver.executeCanonicalContext(contextOptions);
        conversation.id = 'conversation:mutated';
        contextOptions.conversation_runtime.request_id = 'request:mutated';
        resultSchema.properties.answer.type = 'number' as 'string';
        (contextOptions.model_options as { temperature: number }).temperature = 0.9;
        if (contextOptions.labels) contextOptions.labels.purpose = 'mutated';

        await expect(execution).resolves.toMatchObject({ accepted_output: { turn: { status: 'completed' } } });
        expect(driver.capturedOptions).toMatchObject({
            conversation: { id: 'conversation:finite' },
            conversation_runtime: { request_id: 'request:context', purpose: 'conversation' },
            result_schema: { properties: { answer: { type: 'string' } } },
            model_options: { temperature: 0.2 },
            labels: { purpose: 'context-snapshot' },
        });
        expect(driver.capturedOptions?.conversation).not.toBe(conversation);
        expect(driver.capturedOptions?.conversation_runtime).not.toBe(contextOptions.conversation_runtime);
        expect(driver.createPromptCalls).toBe(0);
        expect(driver.requestTextCompletionCalls).toBe(0);
        expect(driver.requestTextCompletionStreamCalls).toBe(0);
    });

    it('rejects native history and legacy authoring policy before canonical context provider dispatch', async () => {
        const driver = new CanonicalContextSnapshotDriver(vi.fn());
        const valid = canonicalContextOptions();

        await expect(
            driver.executeCanonicalContext({
                ...valid,
                conversation: { messages: [{ role: 'user', content: 'legacy' }] },
            } as unknown as CanonicalExecutionContextInputOptions),
        ).rejects.toThrow('Conversation document validation failed');
        await expect(
            driver.executeCanonicalContext({ ...valid, tools: [] } as unknown as CanonicalExecutionContextInputOptions),
        ).rejects.toThrow('uses the document active tool definitions');
        await expect(
            driver.executeCanonicalContext({
                ...valid,
                format: () => 'legacy',
            } as unknown as CanonicalExecutionContextInputOptions),
        ).rejects.toThrow('does not accept a prompt formatter');
        await expect(
            driver.executeCanonicalContext({
                ...valid,
                output_modality: 'audio',
            } as unknown as CanonicalExecutionContextInputOptions),
        ).rejects.toThrow('does not accept legacy output modality policy');

        expect(driver.capturedOptions).toBeUndefined();
        expect(driver.createPromptCalls).toBe(0);
        expect(driver.requestTextCompletionCalls).toBe(0);
        expect(driver.requestTextCompletionStreamCalls).toBe(0);
    });

    it('projects canonical context strings only from the typed context event stream', async () => {
        const driver = new CanonicalContextSnapshotDriver(vi.fn());
        const stream = await driver.streamCanonicalContext(canonicalContextOptions());
        const chunks: string[] = [];
        for await (const chunk of stream) chunks.push(chunk);

        expect(chunks).toEqual(['accepted']);
        expect(stream.completion?.accepted_output.turn.blocks).toEqual([
            { id: 'block:original', type: 'text', text: 'accepted', format: 'plain' },
        ]);
        expect(driver.canonicalContextEventCalls).toBe(1);
        expect(driver.createPromptCalls).toBe(0);
        expect(driver.requestTextCompletionStreamCalls).toBe(0);
    });

    it('derives the compatibility string preview only from the typed canonical event stream', async () => {
        const driver = new TypedCanonicalLifecycleTestDriver(vi.fn());
        const conversation = acceptedFiniteDocument();
        const stream = await driver.streamCanonical([], canonicalOptions(conversation));
        const chunks: string[] = [];
        for await (const chunk of stream) chunks.push(chunk);

        expect(chunks).toEqual(['accepted']);
        expect(stream.completion?.conversation).toEqual(conversation);
        expect(driver.openedStreamId).toBe('response-operation');
        expect(driver.canonicalEventCalls).toBe(1);
        expect(driver.overriddenStringStreamCalls).toBe(0);
        expect(driver.requestTextCompletionStreamCalls).toBe(0);
    });

    it('defers destruction until an in-flight execution finishes', async () => {
        let resolveCompletion!: (completion: Completion) => void;
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);
        driver.completion = new Promise((resolve) => {
            resolveCompletion = resolve;
        });

        const execution = driver.execute(segments, options);
        driver.destroy();

        expect(cleanup).not.toHaveBeenCalled();
        resolveCompletion({ result: [{ type: 'text', value: 'done' }] });
        await execution;
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('defers destruction until an in-flight image generation finishes', async () => {
        let resolveImage!: (completion: Completion) => void;
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);
        driver.imageModel = true;
        driver.imageCompletion = new Promise((resolve) => {
            resolveImage = resolve;
        });

        const execution = driver.execute(segments, options);
        driver.destroy();

        expect(cleanup).not.toHaveBeenCalled();
        resolveImage({ result: [{ type: 'image', value: 'image-data' }] });
        await execution;
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('defers destruction until a stream is cancelled', async () => {
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);
        driver.completionStream = {
            async *[Symbol.asyncIterator]() {
                yield { result: [{ type: 'text', value: 'first' }] } satisfies CompletionChunkObject;
                yield { result: [{ type: 'text', value: 'second' }] } satisfies CompletionChunkObject;
            },
        };
        const stream = await driver.stream(segments, options);
        const iterator = stream[Symbol.asyncIterator]();

        await iterator.next();
        driver.destroy();
        expect(cleanup).not.toHaveBeenCalled();

        await iterator.return?.();
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('holds the stream lease between creation and delayed consumption', async () => {
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);

        const stream = await driver.stream(segments, options);
        driver.destroy();

        expect(cleanup).not.toHaveBeenCalled();
        const chunks: string[] = [];
        for await (const chunk of stream) {
            chunks.push(chunk);
        }
        expect(chunks).toEqual(['first']);
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('transfers lifecycle ownership for overridden stream implementations', async () => {
        const cleanup = vi.fn();
        const driver = new OverriddenStreamDriver(cleanup);

        const stream = await driver.stream(segments, options);
        driver.destroy();

        expect(cleanup).not.toHaveBeenCalled();
        await stream.cancel();
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('coalesces concurrent cancellation of a leased stream', async () => {
        const driver = new OverriddenStreamDriver(vi.fn());
        const releaseCancellation = holdStreamCancellation(driver);
        const stream = await driver.stream(segments, options);

        const first = stream.cancel();
        const second = stream.cancel();

        expect(second).toBe(first);
        expect(driver.cancelStream).toHaveBeenCalledOnce();
        releaseCancellation();
        await Promise.all([first, second]);
    });

    it('coalesces abort-signal and explicit cancellation', async () => {
        const driver = new OverriddenStreamDriver(vi.fn());
        const releaseCancellation = holdStreamCancellation(driver);
        const controller = new AbortController();
        const stream = await driver.stream(segments, options, controller.signal);

        controller.abort();
        const explicitCancellation = stream.cancel();

        expect(driver.cancelStream).toHaveBeenCalledOnce();
        releaseCancellation();
        await explicitCancellation;
    });

    it('coalesces stream-start timeout and explicit cancellation', async () => {
        vi.useFakeTimers();
        try {
            const driver = new OverriddenStreamDriver(vi.fn(), { streamStartTimeoutMs: 100 });
            const releaseCancellation = holdStreamCancellation(driver);
            const stream = await driver.stream(segments, options);

            await vi.advanceTimersByTimeAsync(101);
            const explicitCancellation = stream.cancel();

            expect(driver.cancelStream).toHaveBeenCalledOnce();
            releaseCancellation();
            await explicitCancellation;
            const iterator = stream[Symbol.asyncIterator]();
            await expect(iterator.next()).rejects.toThrow('Completion stream was not consumed within 100ms');
        } finally {
            vi.useRealTimers();
        }
    });

    it('releases lifecycle ownership when an overridden stream iterator throws during creation', async () => {
        const cleanup = vi.fn();
        const driver = new ThrowingIteratorDriver(cleanup);
        const stream = await driver.stream(segments, options);

        expect(() => stream[Symbol.asyncIterator]()).toThrow('iterator creation failed');
        driver.destroy();

        await vi.waitFor(() => expect(driver.cancelStream).toHaveBeenCalledOnce());
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('releases an abandoned stream lease after its start timeout', async () => {
        vi.useFakeTimers();
        try {
            const cleanup = vi.fn();
            const driver = new LifecycleTestDriver(cleanup, { streamStartTimeoutMs: 100 });

            const stream = await driver.stream(segments, options);
            driver.destroy();
            expect(cleanup).not.toHaveBeenCalled();

            await vi.advanceTimersByTimeAsync(101);

            expect(cleanup).toHaveBeenCalledOnce();
            const iterator = stream[Symbol.asyncIterator]();
            await expect(iterator.next()).rejects.toThrow('Completion stream was not consumed within 100ms');
        } finally {
            vi.useRealTimers();
        }
    });

    it('cancels an overridden stream when its start lease expires', async () => {
        vi.useFakeTimers();
        try {
            const driver = new OverriddenStreamDriver(vi.fn(), { streamStartTimeoutMs: 100 });
            await driver.stream(segments, options);

            await vi.advanceTimersByTimeAsync(101);

            expect(driver.cancelStream).toHaveBeenCalledOnce();
        } finally {
            vi.useRealTimers();
        }
    });

    it('releases a direct stream lease exactly once when its start timeout expires', async () => {
        vi.useFakeTimers();
        try {
            const release = vi.fn();
            const cancel = vi.fn().mockResolvedValue(undefined);
            const source: CompletionStream<string> = {
                completion: undefined,
                cancel,
                async *[Symbol.asyncIterator]() {},
            };
            const stream = leaseCompletionStream(source, release, 100);

            await vi.advanceTimersByTimeAsync(101);
            await stream.cancel();

            expect(cancel).toHaveBeenCalledOnce();
            expect(release).toHaveBeenCalledOnce();
        } finally {
            vi.useRealTimers();
        }
    });

    it('claims direct stream cancellation before a source cancellation callback can reenter', async () => {
        const release = vi.fn();
        let reentrant: Promise<void> | undefined;
        let stream!: CompletionStream<string>;
        const cancel = vi.fn(async () => {
            reentrant = stream.cancel();
        });
        const source: CompletionStream<string> = {
            completion: undefined,
            cancel,
            async *[Symbol.asyncIterator]() {},
        };
        stream = leaseCompletionStream(source, release, 10_000);

        const cancellation = stream.cancel();
        await cancellation;

        expect(reentrant).toBe(cancellation);
        await expect(reentrant).resolves.toBeUndefined();
        expect(cancel).toHaveBeenCalledOnce();
        expect(release).toHaveBeenCalledOnce();
    });

    it('keeps the default abandoned-stream lease beyond the request boundary', async () => {
        expect(DEFAULT_COMPLETION_STREAM_START_TIMEOUT_MS).toBe(900_000);

        vi.useFakeTimers();
        try {
            const cleanup = vi.fn();
            const driver = new LifecycleTestDriver(cleanup);

            await driver.stream(segments, options);
            driver.destroy();

            await vi.advanceTimersByTimeAsync(5 * 60_000);
            expect(cleanup).not.toHaveBeenCalled();

            await vi.advanceTimersByTimeAsync(10 * 60_000);
            expect(cleanup).toHaveBeenCalledOnce();
        } finally {
            vi.useRealTimers();
        }
    });

    it('allows an unconsumed stream to be cancelled explicitly', async () => {
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);

        const stream = await driver.stream(segments, options);
        await stream.cancel?.();
        driver.destroy();

        expect(cleanup).toHaveBeenCalledOnce();
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).rejects.toThrow('Completion stream was cancelled before consumption');
    });

    it('cancels the provider while a public iterator next call is pending', async () => {
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);
        driver.completionStream = {
            [Symbol.asyncIterator]() {
                return {
                    next: async () => {
                        await new Promise<void>((resolve) => {
                            driver.completionStreamSignal?.addEventListener('abort', () => resolve(), { once: true });
                        });
                        return { done: true, value: undefined };
                    },
                };
            },
        };

        const stream = await driver.stream(segments, options);
        const iterator = stream[Symbol.asyncIterator]();
        const read = iterator.next();
        await stream.cancel();

        expect(driver.completionStreamSignal?.aborted).toBe(true);
        await expect(read).resolves.toEqual({ done: true, value: undefined });
    });

    it('cancels a pending fallback provider request', async () => {
        const driver = new LifecycleTestDriver(vi.fn());
        driver.streaming = false;
        driver.waitForCompletionAbort = true;

        const stream = await driver.stream(segments, options);
        const read = stream[Symbol.asyncIterator]().next();
        await stream.cancel();

        expect(driver.completionSignal?.aborted).toBe(true);
        await expect(read).resolves.toEqual({ done: true, value: undefined });
    });

    it('holds a typed canonical stream lease through pending iterator cancellation cleanup', async () => {
        const cleanup = vi.fn();
        const driver = new OverriddenCanonicalEventStreamDriver(cleanup);
        driver.holdClosed = true;
        const stream = await driver.streamCanonicalEvents(segments, canonicalOptions(), undefined, canonicalStreamOpen);
        const read = stream[Symbol.asyncIterator]().next();
        driver.destroy();

        const cancellation = stream.cancel();
        await vi.waitFor(() => expect(driver.cancelEventStream).toHaveBeenCalledOnce());
        await vi.waitFor(() => expect(driver.releaseIteratorReturn).toBeTypeOf('function'));
        await expect(cancellation).resolves.toBe(canonicalTerminal);
        await expect(read).resolves.toEqual({ done: true, value: undefined });
        expect(cleanup).not.toHaveBeenCalled();

        driver.releaseClosed?.();
        await vi.waitFor(() => expect(cleanup).toHaveBeenCalledOnce());
        driver.releaseIteratorReturn?.();
    });

    it('keeps projected canonical delivery independent from its owned closed lease', async () => {
        const cleanup = vi.fn();
        const driver = new OverriddenCanonicalEventStreamDriver(cleanup);
        driver.holdClosed = true;
        const stream = await driver.streamCanonical(segments, canonicalOptions());
        const read = stream[Symbol.asyncIterator]().next();
        await vi.waitFor(() => expect(driver.finishRead()).toBe(true));
        await expect(read).resolves.toEqual({ done: true, value: undefined });

        driver.destroy();
        expect(cleanup).not.toHaveBeenCalled();
        driver.releaseClosed?.();
        await vi.waitFor(() => expect(cleanup).toHaveBeenCalledOnce());
    });

    it('reads the typed canonical stream abort signal from the third argument', async () => {
        const cleanup = vi.fn();
        const driver = new OverriddenCanonicalEventStreamDriver(cleanup);
        const controller = new AbortController();
        await driver.streamCanonicalEvents(segments, canonicalOptions(), controller.signal, canonicalStreamOpen);
        driver.destroy();

        controller.abort();

        await vi.waitFor(() => expect(driver.cancelEventStream).toHaveBeenCalledOnce());
        await vi.waitFor(() => expect(cleanup).toHaveBeenCalledOnce());
    });

    it('cancels and releases an unused typed canonical stream after its start timeout', async () => {
        vi.useFakeTimers();
        try {
            const cleanup = vi.fn();
            const driver = new OverriddenCanonicalEventStreamDriver(cleanup, { streamStartTimeoutMs: 100 });
            await driver.streamCanonicalEvents(segments, canonicalOptions(), undefined, canonicalStreamOpen);
            driver.destroy();

            await vi.advanceTimersByTimeAsync(101);

            expect(driver.cancelEventStream).toHaveBeenCalledOnce();
            expect(cleanup).toHaveBeenCalledOnce();
        } finally {
            vi.useRealTimers();
        }
    });

    it('cancels stream creation before a provider stream exists', async () => {
        const cleanup = vi.fn();
        const driver = new PendingStreamCreationDriver(cleanup);
        const controller = new AbortController();

        const stream = driver.stream(segments, options, controller.signal);
        controller.abort();

        await expect(stream).rejects.toBe(controller.signal.reason);
        driver.destroy();
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('keeps the stream signal active after creation', async () => {
        const driver = new LifecycleTestDriver(vi.fn());
        const controller = new AbortController();
        const stream = await driver.stream(segments, options, controller.signal);
        const read = stream[Symbol.asyncIterator]().next();

        controller.abort();

        await expect(read).resolves.toEqual({ done: true, value: undefined });
        expect(driver.completionStreamSignal?.aborted).toBe(true);
    });

    it('rejects invalid stream start timeouts', () => {
        expect(() => new LifecycleTestDriver(vi.fn(), { streamStartTimeoutMs: 0 })).toThrow(
            'streamStartTimeoutMs must be a positive integer no greater than 2147483647',
        );
        expect(() => new LifecycleTestDriver(vi.fn(), { streamStartTimeoutMs: 1.5 })).toThrow(RangeError);
        expect(() => new LifecycleTestDriver(vi.fn(), { streamStartTimeoutMs: 2_147_483_648 })).toThrow(RangeError);
    });

    it('defers destruction until model listing finishes', async () => {
        let resolveModels!: (models: AIModel[]) => void;
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);
        driver.models = new Promise((resolve) => {
            resolveModels = resolve;
        });

        const listing = driver.listModels();
        driver.destroy();

        expect(cleanup).not.toHaveBeenCalled();
        resolveModels([]);
        await listing;
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('defers destruction until embedding generation finishes', async () => {
        let resolveEmbeddings!: (result: EmbeddingsResult) => void;
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);
        driver.embeddings = new Promise((resolve) => {
            resolveEmbeddings = resolve;
        });

        const generation = driver.generateEmbeddings({ inputs: [{ type: 'text', text: 'hello' }] });
        driver.destroy();

        expect(cleanup).not.toHaveBeenCalled();
        resolveEmbeddings({ results: [], model: 'test-model' });
        await generation;
        expect(cleanup).toHaveBeenCalledOnce();
    });

    it('destroys immediately when idle and only once', async () => {
        const cleanup = vi.fn();
        const driver = new LifecycleTestDriver(cleanup);

        driver.destroy();
        driver.destroy();

        expect(cleanup).toHaveBeenCalledOnce();
        await expect(driver.listModels()).rejects.toThrow('Cannot use destroyed lifecycle-test driver');
        await expect(driver.generateEmbeddings({ inputs: [{ type: 'text', text: 'hello' }] })).rejects.toThrow(
            'Cannot use destroyed lifecycle-test driver',
        );
    });
});
