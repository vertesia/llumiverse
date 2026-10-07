import {
    type Content,
    FinishReason,
    type GenerateContentParameters,
    type GenerateContentResponse,
    type GoogleGenAI,
    Language,
} from '@google/genai';
import {
    appendConversationRecords,
    type ConversationDocument,
    type ConversationStreamEvent,
    createConversationDocument,
    createProgramTurn,
    createTextBlock,
    createUserTurn,
    fingerprintJson,
    hashContentBytes,
    parseConversationDocument,
    resolveToolExecutionRequest,
} from '@llumiverse/conversation';
import {
    CANONICAL_REQUIRED_TOOL_CALL_MISSING,
    type CanonicalExecutionInputOptions,
    type ExecutionOptions,
    PromptRole,
} from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { VertexAIDriver } from '../index.js';
import { GeminiModelDefinition } from './gemini.js';
import {
    exportLegacyGeminiConversation,
    GEMINI_GENERATE_CONTENT_PROTOCOL,
    prepareGeminiCanonicalState,
} from './gemini-conversation-adapter.js';

const MODEL = 'publishers/google/models/gemini-2.5-pro';
const LEGACY_GEMINI_GENERATED_SCHEMA_PREFIXES = [
    'Fill all appropriate fields in the JSON output.',
    'When not calling tools, the output must be a JSON object using the following JSON Schema:\n',
    'The output must be a JSON object using the following JSON Schema:\n',
] as const;

type Generate = (request: GenerateContentParameters) => Promise<GenerateContentResponse>;
type GenerateStream = (request: GenerateContentParameters) => Promise<AsyncIterable<GenerateContentResponse>>;

class TestGeminiDriver extends VertexAIDriver {
    constructor(
        private readonly generate: Generate,
        private readonly generateStream: GenerateStream = async () => (async function* () {})(),
    ) {
        super({ project: 'test-project', region: 'global', geminiContextCache: false });
    }

    override getGoogleGenAIClient(): GoogleGenAI {
        return {
            models: {
                generateContent: this.generate,
                generateContentStream: this.generateStream,
            },
        } as unknown as GoogleGenAI;
    }
}

class TestFiniteGeminiDriver extends TestGeminiDriver {
    protected override canStream(_options: ExecutionOptions): Promise<boolean> {
        return Promise.resolve(false);
    }
}

function runtimeOptions(input: {
    flow: string;
    operation: string;
    attempt: string;
    recorded_at: string;
    conversation?: ConversationDocument;
}): CanonicalExecutionInputOptions {
    return {
        model: MODEL,
        ...(input.conversation === undefined ? {} : { conversation: input.conversation }),
        conversation_runtime: {
            conversation_id: `conversation:${input.flow}`,
            request_id: `request:${input.flow}:${input.operation}`,
            attempt_id: `attempt:${input.flow}:${input.attempt}`,
            input_operation_id: `input:${input.flow}:${input.operation}`,
            response_operation_id: `response:${input.flow}:${input.operation}`,
            recorded_at: input.recorded_at,
            started_at: input.recorded_at,
            completed_at: input.recorded_at,
        },
    };
}

function response(input: {
    id: string;
    content?: Content;
    finish_reason?: FinishReason;
    usage?: GenerateContentResponse['usageMetadata'];
}): GenerateContentResponse {
    return {
        responseId: input.id,
        modelVersion: 'gemini-2.5-pro-002',
        candidates:
            input.content === undefined
                ? []
                : [
                      {
                          finishReason: input.finish_reason ?? FinishReason.STOP,
                          content: input.content,
                          safetyRatings: [],
                      },
                  ],
        usageMetadata: input.usage ?? {
            promptTokenCount: 100,
            cachedContentTokenCount: 25,
            candidatesTokenCount: 13,
            thoughtsTokenCount: 7,
            totalTokenCount: 120,
            trafficType: 'ON_DEMAND_PRIORITY',
        },
    } as unknown as GenerateContentResponse;
}

function retainedContextDocument(flow: string, withTools = true, schemaLookingSystem = false): ConversationDocument {
    const recordedAt = '2026-10-01T00:30:00.000Z';
    const initial = createConversationDocument({ id: `conversation:${flow}`, created_at: recordedAt });
    const turn = createUserTurn({
        id: `turn:${flow}:input`,
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: recordedAt },
        model_visibility: 'include',
        provenance: { type: 'received' },
        blocks: [createTextBlock({ id: `block:${flow}:input`, text: 'Return the answer.', format: 'plain' })],
    });
    const program = schemaLookingSystem
        ? createProgramTurn({
              id: `turn:${flow}:program`,
              authority: 'system',
              status: 'completed',
              timestamps: { recorded_at: recordedAt },
              model_visibility: 'include',
              provenance: { type: 'received' },
              blocks: LEGACY_GEMINI_GENERATED_SCHEMA_PREFIXES.map((prefix, index) =>
                  createTextBlock({
                      id: `block:${flow}:program:${index}`,
                      text: `${prefix}Retained source text ${index + 1}.`,
                      format: 'plain',
                  }),
              ),
          })
        : undefined;
    const turns = program === undefined ? [turn] : [program, turn];
    return appendConversationRecords(
        initial,
        {
            turns,
            context_entries: turns.map((sourceTurn) => ({
                id: `context:${sourceTurn.id}`,
                type: 'source_turn' as const,
                turn_id: sourceTurn.id,
            })),
            ...(withTools
                ? {
                      tool_definitions: [
                          {
                              id: 'tool-definition:write:v2',
                              name: 'write',
                              version: 'v2',
                              description: 'Write a value',
                              input_schema: { type: 'object', properties: { value: { type: 'string' } } },
                              result_capabilities: ['text' as const],
                          },
                          {
                              id: 'tool-definition:lookup:v1',
                              name: 'lookup',
                              version: 'v1',
                              description: 'Look up a value',
                              input_schema: { type: 'object', properties: { key: { type: 'string' } } },
                              result_capabilities: ['json' as const],
                          },
                      ],
                      active_tool_definition_ids: ['tool-definition:write:v2', 'tool-definition:lookup:v1'],
                  }
                : { active_tool_definition_ids: [] }),
        },
        {
            expected_revision: initial.revision,
            operation_id: `input:${flow}:materialized`,
            payload_fingerprint: `sha256:${flow}:materialized`,
            recorded_at: recordedAt,
        },
    ).document;
}

async function retainedAudioContextDocument(flow: string): Promise<ConversationDocument> {
    const recordedAt = '2026-10-01T00:35:00.000Z';
    const initial = createConversationDocument({ id: `conversation:${flow}`, created_at: recordedAt });
    const audio = new TextEncoder().encode('retained-audio-bytes');
    const assetId = `asset:${flow}:audio`;
    const turn = createUserTurn({
        id: `turn:${flow}:input`,
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: recordedAt },
        model_visibility: 'include',
        provenance: { type: 'received' },
        blocks: [
            createTextBlock({ id: `block:${flow}:prompt`, text: 'Transcribe retained audio.', format: 'plain' }),
            { id: `block:${flow}:audio`, type: 'audio', asset_id: assetId },
        ],
    });
    return appendConversationRecords(
        initial,
        {
            turns: [turn],
            assets: [
                {
                    id: assetId,
                    kind: 'audio',
                    mime_type: 'audio/wav',
                    storage: { type: 'inline_base64', data: Buffer.from(audio).toString('base64') },
                    provenance: { type: 'received', source_turn_id: turn.id },
                    created_at: recordedAt,
                    ...(await hashContentBytes(audio)),
                },
            ],
            context_entries: [{ id: `context:${flow}:input`, type: 'source_turn', turn_id: turn.id }],
            active_tool_definition_ids: [],
        },
        {
            expected_revision: initial.revision,
            operation_id: `input:${flow}:materialized`,
            payload_fingerprint: `sha256:${flow}:materialized`,
            recorded_at: recordedAt,
        },
    ).document;
}

function requestContents(request: GenerateContentParameters): Content[] {
    if (!Array.isArray(request.contents)) throw new Error('expected Gemini content array');
    return request.contents as Content[];
}

async function drain(stream: AsyncIterable<unknown>): Promise<void> {
    for await (const _chunk of stream) {
        // Drain provider or recovered output before finalizing its conversation.
    }
}

async function* nativeSource<T>(...values: T[]): AsyncIterable<T> {
    yield* values;
}

async function collectCanonicalEvents(
    stream: AsyncIterable<ConversationStreamEvent>,
): Promise<ConversationStreamEvent[]> {
    const events: ConversationStreamEvent[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

function acceptedOutputWithoutProviderTimestamps(value: unknown): unknown {
    return JSON.parse(JSON.stringify(value), (key, item) =>
        key === 'recorded_at' || key === 'completed_at' ? '<provider-completed-at>' : item,
    );
}

function latestGeneratedJson(value: unknown): unknown {
    const document = parseConversationDocument(value);
    for (let index = document.turns.length - 1; index >= 0; index -= 1) {
        const turn = document.turns[index];
        if (turn.kind !== 'agent' || turn.provenance.type !== 'generated') continue;
        return turn.blocks.find((block) => block.type === 'json')?.value;
    }
    return undefined;
}

describe('Gemini canonical lifecycle', () => {
    it('enforces bound tool selection before sync and typed canonical acceptance', async () => {
        const textResponse = response({
            id: 'response-selection-text',
            content: { role: 'model', parts: [{ text: 'No tool call.' }] },
        });
        const toolResponse = response({
            id: 'response-selection-tool',
            content: { role: 'model', parts: [{ functionCall: { name: 'lookup', args: { city: 'Tokyo' } } }] },
        });
        const segments = [{ role: PromptRole.user, content: 'Look up Tokyo.' }];
        const requiredOptions = (
            flow: string,
            conversation?: ConversationDocument,
        ): CanonicalExecutionInputOptions => ({
            ...runtimeOptions({
                flow,
                operation: 'generate',
                attempt: conversation === undefined ? 'first' : 'retry',
                recorded_at: '2026-10-01T00:00:00.000Z',
                ...(conversation === undefined ? {} : { conversation }),
            }),
            tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            model_options: {
                _option_id: 'vertexai-gemini',
                tool_choice: 'required',
                required_tool_name: 'lookup',
            } as ExecutionOptions['model_options'] & { required_tool_name: string },
        });

        const invalidSyncTransport = vi.fn<Generate>(async () => textResponse);
        await expect(
            new TestGeminiDriver(invalidSyncTransport).executeCanonical(
                segments,
                requiredOptions('selection-invalid-sync'),
            ),
        ).rejects.toThrow('violated the requested tool-selection policy');
        expect(invalidSyncTransport).toHaveBeenCalledOnce();

        const finiteTransport = vi.fn<Generate>(async () => textResponse);
        const finiteStream = await new TestFiniteGeminiDriver(finiteTransport).streamCanonicalEvents(
            segments,
            requiredOptions('selection-invalid-finite-stream'),
            undefined,
            { stream_id: 'stream:gemini:selection-invalid-finite' },
        );
        const finiteEvents = await collectCanonicalEvents(finiteStream);
        expect(finiteEvents.at(-1)).toMatchObject({
            type: 'stream_terminated',
            outcome: 'failed',
            diagnostic: { code: CANONICAL_REQUIRED_TOOL_CALL_MISSING, retryable: false },
        });
        expect(finiteEvents.some((event) => event.type === 'response_accepted')).toBe(false);
        expect(finiteStream.completion).toBeUndefined();
        expect(finiteTransport).toHaveBeenCalledOnce();

        const invalidStreamTransport = vi.fn<GenerateStream>(async () => nativeSource(textResponse));
        const invalidStream = await new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, invalidStreamTransport).streamCanonicalEvents(
            segments,
            requiredOptions('selection-invalid-stream'),
            undefined,
            {
                stream_id: 'stream:gemini:selection-invalid',
            },
        );
        const invalidEvents = await collectCanonicalEvents(invalidStream);
        expect(invalidEvents.at(-1)).toMatchObject({
            type: 'stream_terminated',
            outcome: 'failed',
            diagnostic: { code: CANONICAL_REQUIRED_TOOL_CALL_MISSING, retryable: false },
        });
        expect(invalidEvents.some((event) => event.type === 'response_accepted')).toBe(false);
        expect(invalidStream.completion).toBeUndefined();

        const validTransport = vi.fn<Generate>(async () => toolResponse);
        const driver = new TestGeminiDriver(validTransport);
        const first = await driver.executeCanonical(segments, requiredOptions('selection-recovery'));
        const firstGeneration = Object.values(first.conversation.generations).find(
            (candidate) => candidate.id === first.accepted_output.generation.id,
        );
        if (firstGeneration?.request_receipt === undefined) throw new Error('Expected retained request receipt');
        expect(firstGeneration.request_receipt.target.options).toEqual({
            canonical_tool_selection: { mode: 'required', tool_name: 'lookup' },
        });
        const persisted = JSON.parse(JSON.stringify(first.conversation));
        const recovered = await driver.executeCanonical(segments, requiredOptions('selection-recovery', persisted));
        expect(recovered.accepted_output).toEqual(first.accepted_output);
        expect(validTransport).toHaveBeenCalledOnce();

        await expect(
            driver.executeCanonical(segments, {
                ...requiredOptions('selection-recovery', persisted),
                model_options: {
                    _option_id: 'vertexai-gemini',
                    tool_choice: 'none',
                } as ExecutionOptions['model_options'] & { tool_choice: 'none' },
            }),
        ).rejects.toThrow('incompatible canonical tool-selection policy');
        expect(validTransport).toHaveBeenCalledOnce();

        const malformed = JSON.parse(JSON.stringify(first.conversation));
        const generation = Object.values(parseConversationDocument(malformed).generations).find(
            (candidate) => candidate.record_source === 'executed',
        );
        if (generation === undefined) throw new Error('Expected executed generation');
        const mutableGeneration = (malformed.generations as Record<string, typeof generation>)[generation.id];
        if (mutableGeneration === undefined) throw new Error('Expected mutable executed generation');
        mutableGeneration.request_receipt.target.options = {
            canonical_tool_selection: { mode: 'required', extra: true },
        };
        await expect(
            driver.executeCanonical(segments, requiredOptions('selection-recovery', malformed)),
        ).rejects.toThrow('unknown field');
        expect(validTransport).toHaveBeenCalledOnce();

        const legacy = JSON.parse(JSON.stringify(first.conversation));
        const legacyGeneration = Object.values(parseConversationDocument(legacy).generations).find(
            (candidate) => candidate.record_source === 'executed',
        );
        if (legacyGeneration === undefined) throw new Error('Expected legacy executed generation');
        delete (legacy.generations as Record<string, typeof legacyGeneration>)[legacyGeneration.id]?.request_receipt
            .target.options;
        await expect(
            driver.executeCanonical(segments, requiredOptions('selection-recovery', legacy)),
        ).resolves.toMatchObject({ accepted_output: first.accepted_output });
        expect(validTransport).toHaveBeenCalledOnce();
    });

    it('executes retained context with exact ordered tools and request-local schema guidance', async () => {
        const flow = 'retained-context';
        const document = retainedContextDocument(flow, true, true);
        const nativeResponse = response({
            id: 'response-retained-context',
            content: { role: 'model', parts: [{ text: '{"answer":"Tokyo"}' }] },
        });
        const generate = vi.fn<Generate>(async () => nativeResponse);
        const driver = new TestGeminiDriver(generate);
        const resultSchema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const first = await driver.executeCanonicalContext({
            ...runtimeOptions({
                flow,
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-10-01T00:31:00.000Z',
                conversation: document,
            }),
            conversation: document,
            result_schema: resultSchema,
        });

        const request = generate.mock.calls[0]?.[0];
        expect(requestContents(request as GenerateContentParameters)).toEqual([
            { role: 'user', parts: [{ text: 'Return the answer.' }] },
        ]);
        expect(request?.config?.tools).toMatchObject([
            {
                functionDeclarations: [
                    { name: 'write', description: 'Write a value' },
                    { name: 'lookup', description: 'Look up a value' },
                ],
            },
        ]);
        const systemInstruction = request?.config?.systemInstruction;
        expect(typeof systemInstruction).toBe('object');
        const systemInstructionParts =
            systemInstruction &&
            typeof systemInstruction === 'object' &&
            !Array.isArray(systemInstruction) &&
            'parts' in systemInstruction
                ? systemInstruction.parts
                : undefined;
        expect(systemInstructionParts).toEqual([
            ...LEGACY_GEMINI_GENERATED_SCHEMA_PREFIXES.map((prefix, index) => ({
                text: `${prefix}Retained source text ${index + 1}.`,
            })),
            {
                text: expect.stringContaining(
                    'When not calling tools, the output must be a JSON object using the following JSON Schema:',
                ),
            },
        ]);
        expect(first.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'json', value: { answer: 'Tokyo' } }),
        );
        expect(document.revision).toBe(1);
        expect(document.generations).toEqual({});
        const generation = Object.values(first.conversation.generations).find(
            (candidate) =>
                candidate.record_source === 'executed' && candidate.id === first.accepted_output.generation.id,
        );
        expect(generation?.record_source).toBe('executed');
        expect(generation?.record_source === 'executed' ? generation.request_receipt.source : undefined).toEqual({
            conversation_id: document.id,
            revision: document.revision,
        });

        const recovered = await driver.executeCanonicalContext({
            ...runtimeOptions({
                flow,
                operation: 'generate',
                attempt: 'retry',
                recorded_at: '2026-10-01T00:32:00.000Z',
                conversation: first.conversation,
            }),
            conversation: first.conversation,
            result_schema: resultSchema,
        });
        expect(recovered.accepted_output).toEqual(first.accepted_output);
        expect(generate).toHaveBeenCalledOnce();

        await expect(
            driver.executeCanonicalContext({
                ...runtimeOptions({
                    flow,
                    operation: 'generate',
                    attempt: 'changed-schema',
                    recorded_at: '2026-10-01T00:33:00.000Z',
                    conversation: first.conversation,
                }),
                conversation: first.conversation,
                result_schema: { ...resultSchema, required: [] },
            }),
        ).rejects.toThrow('incompatible request identity');
        expect(generate).toHaveBeenCalledOnce();
    });

    it('streams retained canonical context as typed events without authoring an empty input', async () => {
        const flow = 'retained-context-stream';
        const document = retainedContextDocument(flow, false);
        const generateStream = vi.fn<GenerateStream>(async () =>
            nativeSource(
                response({
                    id: 'response-retained-context-stream',
                    content: { role: 'model', parts: [{ text: 'Canonical stream answer.' }] },
                }),
            ),
        );
        const driver = new TestGeminiDriver(async () => {
            throw new Error('Blocking transport must not run');
        }, generateStream);
        const stream = await driver.streamCanonicalContextEvents(
            {
                ...runtimeOptions({
                    flow,
                    operation: 'generate',
                    attempt: 'first',
                    recorded_at: '2026-10-01T00:34:00.000Z',
                    conversation: document,
                }),
                conversation: document,
            },
            undefined,
            { stream_id: 'stream:gemini:retained-context' },
        );
        const events = await collectCanonicalEvents(stream);

        expect(events).toContainEqual(
            expect.objectContaining({ type: 'draft_text_delta', text: 'Canonical stream answer.' }),
        );
        expect(events).toContainEqual(expect.objectContaining({ type: 'response_accepted' }));
        expect(stream.completion?.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'text', text: 'Canonical stream answer.' }),
        );
        expect(generateStream).toHaveBeenCalledOnce();
        expect(document.revision).toBe(1);
    });

    it('executes retained inline audio through the finite typed path and exact-recovers it', async () => {
        const flow = 'retained-audio-context';
        const document = await retainedAudioContextDocument(flow);
        const nativeResponse = response({
            id: 'response-retained-audio-context',
            content: {
                role: 'model',
                parts: [
                    {
                        audioTranscription: {
                            text: 'Retained audio transcript.',
                            languageCode: 'en',
                        },
                    },
                ],
            },
        });
        const generate = vi.fn<Generate>(async () => nativeResponse);
        const driver = new TestGeminiDriver(generate);
        const options = {
            ...runtimeOptions({
                flow,
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-10-01T00:36:00.000Z',
                conversation: document,
            }),
            model: 'gemini-3.5-transcribe-preview',
            conversation: document,
        };

        expect(await driver.supportsCanonicalContextExecution(options)).toBe(true);
        const stream = await driver.streamCanonicalContextEvents(options, undefined, {
            stream_id: 'stream:gemini:retained-audio-context',
        });
        const events = await collectCanonicalEvents(stream);
        const request = generate.mock.calls[0]?.[0];
        const parts = requestContents(request as GenerateContentParameters).flatMap((content) => content.parts ?? []);

        expect(events).toEqual([
            expect.objectContaining({ type: 'response_accepted', origin: 'live_transport', sequence: 0 }),
        ]);
        expect(parts).toContainEqual({ text: 'Transcribe retained audio.' });
        expect(parts).toContainEqual({
            inlineData: {
                mimeType: 'audio/wav',
                data: Buffer.from('retained-audio-bytes').toString('base64'),
            },
        });
        expect(stream.completion?.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'text', text: 'Retained audio transcript.' }),
        );
        expect(generate).toHaveBeenCalledOnce();

        const accepted = stream.completion;
        if (accepted === undefined) throw new Error('Expected accepted retained audio response');
        const recovered = await driver.executeCanonicalContext({ ...options, conversation: accepted.conversation });
        expect(recovered.accepted_output).toEqual(accepted.accepted_output);
        expect(generate).toHaveBeenCalledOnce();
    });

    it('rejects unsupported retained audio controls before provider transport', async () => {
        const flow = 'retained-audio-tools';
        const document = await retainedAudioContextDocument(flow);
        document.tool_definitions['tool:unsupported'] = {
            id: 'tool:unsupported',
            name: 'unsupported',
            version: 'v1',
            input_schema: { type: 'object' },
            result_capabilities: ['text'],
        };
        document.context.active_tool_definition_ids.push('tool:unsupported');
        const generate = vi.fn<Generate>(async () => {
            throw new Error('provider must not run');
        });

        await expect(
            new TestGeminiDriver(generate).executeCanonicalContext({
                ...runtimeOptions({
                    flow,
                    operation: 'generate',
                    attempt: 'first',
                    recorded_at: '2026-10-01T00:37:00.000Z',
                    conversation: document,
                }),
                model: 'gemini-3.5-transcribe-preview',
                conversation: document,
            }),
        ).rejects.toThrow('does not support tool definitions');
        expect(generate).not.toHaveBeenCalled();
    });

    it('validates structured sync output, preserves usage, and durably recovers an accepted response', async () => {
        const raw = '{ "answer" : "Tokyo", "note" : null }';
        const nativeResponse = response({
            id: 'response-structured',
            content: { role: 'model', parts: [{ text: raw }] },
        });
        const generate = vi.fn<Generate>(async () => nativeResponse);
        const driver = new TestGeminiDriver(generate);
        const segments = [{ role: PromptRole.user, content: 'Return JSON.' }];
        const result_schema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' }, note: { type: 'null' } },
            required: ['answer', 'note'],
            additionalProperties: false,
        };
        const first = await driver.execute(segments, {
            ...runtimeOptions({
                flow: 'structured',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:00:00.000Z',
            }),
            result_schema,
        });

        expect(first.result).toEqual([{ type: 'json', value: { answer: 'Tokyo', note: null } }]);
        expect(first.token_usage).toEqual({
            total: 120,
            prompt: 100,
            prompt_cached: 25,
            prompt_new: 75,
            result: 20,
        });
        expect(first.service_tier).toBe('priority');
        expect(latestGeneratedJson(first.conversation)).toEqual({ answer: 'Tokyo', note: null });
        const document = parseConversationDocument(JSON.parse(JSON.stringify(first.conversation)));
        const generation = Object.values(document.generations).find(
            (candidate) => candidate.record_source === 'executed',
        );
        expect(generation).toMatchObject({
            provider: 'vertexai',
            protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
            requested_model: MODEL,
            resolved_model: 'gemini-2.5-pro-002',
            provider_response_id: 'response-structured',
            usage: {
                input_tokens: 100,
                cache_read_tokens: 25,
                input_new_tokens: 75,
                output_tokens: 20,
                reasoning_tokens: 7,
                total_tokens: 120,
            },
        });
        expect(exportLegacyGeminiConversation(document)._arrayConversation.at(-1)).toEqual(
            nativeResponse.candidates?.[0]?.content,
        );

        const retried = await driver.execute(segments, {
            ...runtimeOptions({
                flow: 'structured',
                operation: 'generate',
                attempt: 'retry',
                recorded_at: '2026-09-30T01:05:00.000Z',
                conversation: document,
            }),
            result_schema,
        });
        expect(retried.result).toEqual(first.result);
        expect(retried.token_usage).toEqual(first.token_usage);
        expect(retried.service_tier).toBe(first.service_tier);
        expect(retried.conversation).toEqual(document);
        expect(generate).toHaveBeenCalledTimes(1);
    });

    it('returns direct canonical structured output and retries without another provider request', async () => {
        const nativeResponse = response({
            id: 'response-direct-structured',
            content: { role: 'model', parts: [{ text: '{"answer":"Tokyo"}' }] },
        });
        const generate = vi.fn<Generate>(async () => nativeResponse);
        const driver = new TestGeminiDriver(generate);
        const segments = [{ role: PromptRole.user, content: 'Return JSON.' }];
        const result_schema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const first = await driver.executeCanonical(segments, {
            ...runtimeOptions({
                flow: 'direct-structured',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:06:00.000Z',
            }),
            result_schema,
        });
        expect(first.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'json', value: { answer: 'Tokyo' } }),
        );
        expect(first.accepted_output.generation).toMatchObject({
            status: 'completed',
            usage: { input_tokens: 100, output_tokens: 20, total_tokens: 120 },
        });
        expect(first.service_tier).toBe('priority');
        expect(exportLegacyGeminiConversation(first.conversation)._arrayConversation.at(-1)).toEqual(
            nativeResponse.candidates?.[0]?.content,
        );

        const retried = await driver.executeCanonical(segments, {
            ...runtimeOptions({
                flow: 'direct-structured',
                operation: 'generate',
                attempt: 'retry',
                recorded_at: '2026-09-30T01:07:00.000Z',
                conversation: first.conversation,
            }),
            result_schema,
        });
        expect(retried.conversation).toEqual(first.conversation);
        expect(retried.service_tier).toBe(first.service_tier);
        await expect(
            driver.executeCanonical(segments, {
                ...runtimeOptions({
                    flow: 'direct-structured',
                    operation: 'generate',
                    attempt: 'changed-options',
                    recorded_at: '2026-09-30T01:08:00.000Z',
                    conversation: parseConversationDocument(first.conversation),
                }),
                result_schema,
                model_options: { _option_id: 'vertexai-gemini', temperature: 0.2 },
            }),
        ).rejects.toThrow('incompatible request identity');
        expect(generate).toHaveBeenCalledOnce();
    });

    it('marks invalid required structured output failed in direct sync and stream results', async () => {
        const invalid = response({
            id: 'response-direct-invalid',
            content: { role: 'model', parts: [{ text: '{"wrong":42}' }] },
        });
        const result_schema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const syncDriver = new TestGeminiDriver(async () => invalid);
        const sync = await syncDriver.executeCanonical([{ role: PromptRole.user, content: 'Return JSON.' }], {
            ...runtimeOptions({
                flow: 'direct-invalid-sync',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:08:00.000Z',
            }),
            result_schema,
        });
        expect(sync.accepted_output.generation.status).toBe('failed');
        expect(sync.accepted_output.turn.status).toBe('failed');
        expect(sync.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'text', text: '{"wrong":42}' }),
        );

        const generateStream = vi.fn<GenerateStream>(async () =>
            (async function* () {
                yield {
                    candidates: [{ content: { role: 'model', parts: [{ text: '{"wrong":42}' }] } }],
                } as GenerateContentResponse;
                yield invalid;
                yield {
                    usageMetadata: {
                        promptTokenCount: 9,
                        cachedContentTokenCount: 3,
                        candidatesTokenCount: 4,
                        totalTokenCount: 13,
                        trafficType: 'ON_DEMAND_FLEX',
                    },
                } as GenerateContentResponse;
            })(),
        );
        const streamDriver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const stream = await streamDriver.streamCanonical([{ role: PromptRole.user, content: 'Return JSON.' }], {
            ...runtimeOptions({
                flow: 'direct-invalid-stream',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:09:00.000Z',
            }),
            result_schema,
        });
        for await (const _chunk of stream) {
            // Drain the provider preview so canonical finalization runs.
        }
        expect(stream.completion?.accepted_output.generation).toMatchObject({
            status: 'failed',
            usage: { input_tokens: 9, output_tokens: 4, total_tokens: 13 },
        });
        expect(stream.completion?.accepted_output.turn.status).toBe('failed');
        expect(stream.completion?.service_tier).toBe('flex');
    });

    it('records a Gemini max-token terminal as interrupted and cancelled', async () => {
        const driver = new TestGeminiDriver(async () =>
            response({
                id: 'response-cutoff',
                content: { role: 'model', parts: [{ text: 'partial' }] },
                finish_reason: FinishReason.MAX_TOKENS,
            }),
        );
        const result = await driver.executeCanonical([{ role: PromptRole.user, content: 'Continue.' }], {
            ...runtimeOptions({
                flow: 'direct-cutoff',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:10:00.000Z',
            }),
        });

        expect(result.accepted_output.turn.status).toBe('interrupted');
        expect(result.accepted_output.generation).toMatchObject({ status: 'cancelled', finish_reason: 'length' });
    });

    it('aborts a pending direct canonical Gemini read before iterator cleanup', async () => {
        let providerSignal: AbortSignal | undefined;
        const generateStream = vi.fn<GenerateStream>(async (request) => {
            providerSignal = request.config?.abortSignal;
            return {
                [Symbol.asyncIterator]() {
                    return {
                        next: () =>
                            new Promise<IteratorResult<GenerateContentResponse>>((resolve) => {
                                providerSignal?.addEventListener(
                                    'abort',
                                    () => resolve({ done: true, value: undefined }),
                                    { once: true },
                                );
                            }),
                        return: async () => ({ done: true, value: undefined }),
                    };
                },
            };
        });
        const driver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const stream = await driver.streamCanonical(
            [{ role: PromptRole.user, content: 'Wait.' }],
            runtimeOptions({
                flow: 'direct-cancel',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:11:00.000Z',
            }),
        );
        const iterator = stream[Symbol.asyncIterator]();
        const pending = iterator.next();
        await vi.waitFor(() => expect(providerSignal).toBeDefined());
        await stream.cancel();
        await expect(pending).rejects.toThrow('cancelled');
        expect(providerSignal?.aborted).toBe(true);
        expect(stream.completion).toBeUndefined();
    });

    it('emits typed structured drafts at cumulative Gemini part positions and recovers without transport', async () => {
        const generateStream = vi.fn<GenerateStream>(async () =>
            (async function* () {
                yield {
                    candidates: [{ content: { role: 'model', parts: [{ text: '{"answer":' }] } }],
                } as GenerateContentResponse;
                yield response({
                    id: 'response-typed-structured',
                    content: { role: 'model', parts: [{ text: '"Tokyo"}' }] },
                });
            })(),
        );
        const driver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const segments = [{ role: PromptRole.user, content: 'Return JSON.' }];
        const result_schema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const firstOptions = {
            ...runtimeOptions({
                flow: 'typed-structured',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:11:30.000Z',
            }),
            result_schema,
        };
        const typed = await driver.streamCanonicalEvents(segments, firstOptions, undefined, {
            stream_id: 'stream:gemini:typed-structured',
        });
        const events = await collectCanonicalEvents(typed);

        expect(events.filter((event) => event.type === 'draft_text_delta')).toMatchObject([
            {
                text: '{"answer":',
                native_position: {
                    protocol: GEMINI_GENERATE_CONTENT_PROTOCOL,
                    path: ['candidates', 0, 'content', 'parts', 0],
                },
            },
            { text: '"Tokyo"}' },
        ]);
        expect(events.at(-1)).toMatchObject({
            type: 'response_accepted',
            origin: 'live_transport',
            reconciliations: [{ disposition: 'structured_output' }],
        });
        expect(typed.completion?.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'json', value: { answer: 'Tokyo' } }),
        );

        const legacyDriver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const legacy = await legacyDriver.streamCanonical(segments, firstOptions);
        await drain(legacy);
        expect(acceptedOutputWithoutProviderTimestamps(legacy.completion?.accepted_output)).toEqual(
            acceptedOutputWithoutProviderTimestamps(typed.completion?.accepted_output),
        );

        if (typed.completion === undefined) throw new Error('Expected typed Gemini completion');
        const recovered = await driver.streamCanonicalEvents(
            segments,
            {
                ...runtimeOptions({
                    flow: 'typed-structured',
                    operation: 'generate',
                    attempt: 'retry',
                    recorded_at: '2026-09-30T01:11:31.000Z',
                    conversation: JSON.parse(JSON.stringify(typed.completion.conversation)),
                }),
                result_schema,
            },
            undefined,
            { stream_id: 'stream:gemini:typed-structured:retry' },
        );
        expect(await collectCanonicalEvents(recovered)).toEqual([
            expect.objectContaining({
                type: 'response_accepted',
                origin: 'accepted_recovery',
                sequence: 0,
            }),
        ]);
        expect(recovered.completion?.accepted_output).toEqual(typed.completion.accepted_output);
        await expect(
            driver.streamCanonicalEvents(
                segments,
                {
                    ...runtimeOptions({
                        flow: 'typed-structured',
                        operation: 'generate',
                        attempt: 'changed-options',
                        recorded_at: '2026-09-30T01:11:32.000Z',
                        conversation: JSON.parse(JSON.stringify(typed.completion.conversation)),
                    }),
                    result_schema,
                    model_options: { _option_id: 'vertexai-gemini', temperature: 0.2 },
                },
                undefined,
                { stream_id: 'stream:gemini:typed-structured:changed-options' },
            ),
        ).rejects.toThrow('incompatible request identity');
        expect(generateStream).toHaveBeenCalledTimes(2);
    });

    it('accepts invalid required structured output as an explicit failed Gemini response', async () => {
        const generateStream = vi.fn<GenerateStream>(async () =>
            (async function* () {
                yield response({
                    id: 'response-typed-invalid-structured',
                    content: { role: 'model', parts: [{ text: '{"wrong":42}' }] },
                });
            })(),
        );
        const driver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Return JSON.' }],
            {
                ...runtimeOptions({
                    flow: 'typed-invalid-structured',
                    operation: 'generate',
                    attempt: 'first',
                    recorded_at: '2026-09-30T01:11:35.000Z',
                }),
                result_schema: {
                    type: 'object',
                    properties: { answer: { type: 'string' } },
                    required: ['answer'],
                    additionalProperties: false,
                },
            },
            undefined,
            { stream_id: 'stream:gemini:typed-invalid-structured' },
        );
        const events = await collectCanonicalEvents(stream);

        expect(events.at(-1)?.type).toBe('response_accepted');
        expect(stream.completion?.accepted_output.generation.status).toBe('failed');
        expect(stream.completion?.accepted_output.turn).toMatchObject({
            status: 'failed',
            blocks: [expect.objectContaining({ type: 'text', text: '{"wrong":42}' })],
        });
    });

    it.each([
        ['with a provider message', 'Prompt blocked.'],
        ['without a provider message', undefined],
    ] as const)(
        'accepts a candidate-less Gemini prompt block %s without fabricating a draft',
        async (_label, message) => {
            const terminal = {
                responseId: `response-typed-blocked-${message === undefined ? 'empty' : 'message'}`,
                modelVersion: 'gemini-2.5-pro-002',
                candidates: [],
                promptFeedback: {
                    blockReason: 'BLOCKLIST',
                    ...(message === undefined ? {} : { blockReasonMessage: message }),
                },
                usageMetadata: { promptTokenCount: 8, totalTokenCount: 8 },
            } as unknown as GenerateContentResponse;
            const generate = vi.fn<Generate>(async () => terminal);
            const generateStream = vi.fn<GenerateStream>(async () =>
                (async function* () {
                    yield terminal;
                })(),
            );
            const segments = [{ role: PromptRole.user, content: 'Blocked prompt.' }];
            const options = runtimeOptions({
                flow: `typed-blocked-${message === undefined ? 'empty' : 'message'}`,
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:11:36.000Z',
            });
            const sync = await new TestGeminiDriver(generate).executeCanonical(segments, options);
            const streamDriver = new TestGeminiDriver(async () => {
                throw new Error('blocking transport not expected');
            }, generateStream);
            const legacy = await streamDriver.streamCanonical(segments, options);
            await drain(legacy);
            const typed = await streamDriver.streamCanonicalEvents(segments, options, undefined, {
                stream_id: `stream:gemini:typed-blocked-${message === undefined ? 'empty' : 'message'}`,
            });
            const events = await collectCanonicalEvents(typed);

            expect(events.some((event) => event.type === 'draft_block_started')).toBe(false);
            expect(events.at(-1)).toMatchObject({ type: 'response_accepted', reconciliations: [] });
            expect(typed.completion?.accepted_output.turn.blocks).toEqual(
                message === undefined ? [] : [expect.objectContaining({ type: 'text', text: message })],
            );
            expect(acceptedOutputWithoutProviderTimestamps(typed.completion?.accepted_output)).toEqual(
                acceptedOutputWithoutProviderTimestamps(sync.accepted_output),
            );
            expect(acceptedOutputWithoutProviderTimestamps(legacy.completion?.accepted_output)).toEqual(
                acceptedOutputWithoutProviderTimestamps(sync.accepted_output),
            );
            expect(generate).toHaveBeenCalledOnce();
            expect(generateStream).toHaveBeenCalledTimes(2);
        },
    );

    it('keeps max-token Gemini calls identically interrupted and non-executable across canonical APIs', async () => {
        const terminal = response({
            id: 'response-typed-cutoff-tool',
            content: {
                role: 'model',
                parts: [{ functionCall: { name: 'lookup', args: { city: 'Tokyo' } } }],
            },
            finish_reason: FinishReason.MAX_TOKENS,
        });
        const streamTransport = vi.fn<GenerateStream>(async () =>
            (async function* () {
                yield terminal;
            })(),
        );
        const segments = [{ role: PromptRole.user, content: 'Look up Tokyo.' }];
        const options = {
            ...runtimeOptions({
                flow: 'typed-cutoff-tool',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:11:37.000Z',
            }),
            tools: [{ name: 'lookup', input_schema: { type: 'object' as const } }],
        } satisfies CanonicalExecutionInputOptions;

        const syncDriver = new TestGeminiDriver(async () => terminal);
        const sync = await syncDriver.executeCanonical(segments, options);
        const legacyDriver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, streamTransport);
        const legacy = await legacyDriver.streamCanonical(segments, options);
        await drain(legacy);
        const typedDriver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, streamTransport);
        const typed = await typedDriver.streamCanonicalEvents(segments, options, undefined, {
            stream_id: 'stream:gemini:typed-cutoff-tool',
        });
        const events = await collectCanonicalEvents(typed);
        const call = typed.completion?.accepted_output.turn.blocks.find((block) => block.type === 'tool_call');

        expect(call).toMatchObject({
            type: 'tool_call',
            executor: 'application',
            arguments: { type: 'invalid', error: expect.stringContaining(FinishReason.MAX_TOKENS) },
        });
        expect(typed.completion?.accepted_output.turn.status).toBe('interrupted');
        expect(typed.completion?.accepted_output.generation).toMatchObject({
            status: 'cancelled',
            finish_reason: 'length',
        });
        expect(events).toContainEqual(expect.objectContaining({ type: 'draft_block_finished', outcome: 'malformed' }));
        expect(events).toContainEqual(expect.objectContaining({ type: 'draft_finished', outcome: 'interrupted' }));
        expect(acceptedOutputWithoutProviderTimestamps(legacy.completion?.accepted_output)).toEqual(
            acceptedOutputWithoutProviderTimestamps(sync.accepted_output),
        );
        expect(acceptedOutputWithoutProviderTimestamps(typed.completion?.accepted_output)).toEqual(
            acceptedOutputWithoutProviderTimestamps(sync.accepted_output),
        );

        if (typed.completion === undefined || call?.type !== 'tool_call') {
            throw new Error('Expected interrupted Gemini tool call');
        }
        const sourceTurn = typed.completion.conversation.turns.find(
            (turn) => turn.id === typed.completion?.accepted_output.turn.id,
        );
        const sourceCall = sourceTurn?.blocks.find((block) => block.id === call.id);
        if (sourceCall?.type !== 'tool_call') throw new Error('Expected retained interrupted Gemini tool call');
        await expect(
            resolveToolExecutionRequest(
                typed.completion.conversation,
                {
                    conversation: {
                        conversation_id: typed.completion.conversation.id,
                        revision: typed.completion.conversation.revision,
                    },
                    turn_id: typed.completion.accepted_output.turn.id,
                    block_id: sourceCall.id,
                    call_id: sourceCall.call_id,
                    call_fingerprint: await fingerprintJson(sourceCall),
                },
                async function* () {
                    // Interrupted inline arguments must fail before asset resolution.
                },
            ),
        ).rejects.toThrow(/invalid arguments/);
    });

    it('keeps Gemini thought signatures private while reconciling parallel native tool identities', async () => {
        const generateStream = vi.fn<GenerateStream>(async () =>
            (async function* () {
                yield {
                    candidates: [
                        {
                            content: {
                                role: 'model',
                                parts: [
                                    { text: 'plan', thought: true, thoughtSignature: 'secret-reasoning-signature' },
                                ],
                            },
                        },
                    ],
                } as GenerateContentResponse;
                yield response({
                    id: 'response-typed-tools',
                    content: {
                        role: 'model',
                        parts: [
                            {
                                functionCall: { id: 'call-a', name: 'lookup', args: { city: 'Tokyo' } },
                                thoughtSignature: 'secret-call-signature-a',
                            },
                            {
                                functionCall: { id: 'call-b', name: 'lookup', args: { city: 'Paris' } },
                                thoughtSignature: 'secret-call-signature-b',
                            },
                        ],
                    },
                });
            })(),
        );
        const driver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Look up both cities.' }],
            {
                ...runtimeOptions({
                    flow: 'typed-tools',
                    operation: 'generate',
                    attempt: 'first',
                    recorded_at: '2026-09-30T01:11:40.000Z',
                }),
                tools: [{ name: 'lookup', input_schema: { type: 'object', additionalProperties: true } }],
            },
            undefined,
            { stream_id: 'stream:gemini:typed-tools' },
        );
        const events = await collectCanonicalEvents(stream);
        const calls = stream.completion?.accepted_output.turn.blocks.filter((block) => block.type === 'tool_call');

        expect(events.filter((event) => event.type === 'draft_tool_arguments_delta')).toMatchObject([
            { arguments: { encoding: 'json_value_snapshot', value: { city: 'Tokyo' } } },
            { arguments: { encoding: 'json_value_snapshot', value: { city: 'Paris' } } },
        ]);
        expect(calls).toMatchObject([
            { call_id: 'call-a', tool_name: 'lookup', executor: 'application' },
            { call_id: 'call-b', tool_name: 'lookup', executor: 'application' },
        ]);
        expect(JSON.stringify(events)).not.toContain('secret-reasoning-signature');
        expect(JSON.stringify(events)).not.toContain('secret-call-signature');
        expect(events.at(-1)?.type).toBe('response_accepted');
    });

    it('assigns distinct stable canonical identities to parallel ID-less same-name Gemini calls', async () => {
        const generateStream = vi.fn<GenerateStream>(async () =>
            (async function* () {
                yield response({
                    id: 'response-typed-idless-tools',
                    content: {
                        role: 'model',
                        parts: [
                            { functionCall: { name: 'lookup', args: { city: 'Tokyo' } } },
                            { functionCall: { name: 'lookup', args: { city: 'Paris' } } },
                        ],
                    },
                });
            })(),
        );
        const driver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Look up both cities.' }],
            {
                ...runtimeOptions({
                    flow: 'typed-idless-tools',
                    operation: 'generate',
                    attempt: 'first',
                    recorded_at: '2026-09-30T01:11:45.000Z',
                }),
                tools: [{ name: 'lookup', input_schema: { type: 'object', additionalProperties: true } }],
            },
            undefined,
            { stream_id: 'stream:gemini:typed-idless-tools' },
        );
        const events = await collectCanonicalEvents(stream);
        const starts = events.filter((event) => event.type === 'draft_block_started');
        const calls = stream.completion?.accepted_output.turn.blocks.filter((block) => block.type === 'tool_call');

        expect(calls).toHaveLength(2);
        expect(calls?.[0]?.call_id).not.toBe(calls?.[1]?.call_id);
        expect(starts.flatMap((event) => (event.block.type === 'tool_call' ? [event.block.call_id] : []))).toEqual(
            calls?.map((call) => call.call_id),
        );
        expect(calls?.map((call) => call.arguments)).toEqual([
            { type: 'json', value: { city: 'Tokyo' } },
            { type: 'json', value: { city: 'Paris' } },
        ]);
        expect(events.at(-1)?.type).toBe('response_accepted');
    });

    it('rejects typed Gemini bounds and publication failures before opening provider transport', async () => {
        const generateStream = vi.fn<GenerateStream>(async () => (async function* () {})());
        const publish = vi.fn(async () => undefined);
        const driver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const segments = [{ role: PromptRole.user, content: 'Answer.' }];

        await expect(
            driver.streamCanonicalEvents(
                segments,
                {
                    ...runtimeOptions({
                        flow: 'typed-invalid-open',
                        operation: 'generate',
                        attempt: 'first',
                        recorded_at: '2026-09-30T01:11:50.000Z',
                    }),
                    on_canonical_request_prepared: publish,
                },
                undefined,
                { stream_id: 'stream:gemini:typed-invalid-open', max_buffered_events: 0 },
            ),
        ).rejects.toThrow();
        expect(publish).not.toHaveBeenCalled();
        expect(generateStream).not.toHaveBeenCalled();

        await expect(
            driver.streamCanonicalEvents(
                segments,
                {
                    ...runtimeOptions({
                        flow: 'typed-barrier',
                        operation: 'generate',
                        attempt: 'first',
                        recorded_at: '2026-09-30T01:11:51.000Z',
                    }),
                    on_canonical_request_prepared: async () => {
                        throw new Error('durability barrier failed');
                    },
                },
                undefined,
                { stream_id: 'stream:gemini:typed-barrier' },
            ),
        ).rejects.toThrow('durability barrier failed');
        expect(generateStream).not.toHaveBeenCalled();
    });

    it.each([
        ['retryable URL fetch throttle', 'URL_REJECTED-REJECTED_CLIENT_THROTTLED', true],
        ['permanent URL rejection', 'URL_REJECTED-ROBOTS_DENIED', false],
    ] as const)('preserves %s classification on the public typed stream', async (_label, marker, retryable) => {
        const providerFailure = Object.assign(new Error(`private provider details ${marker}`), { status: 400 });
        const generateStream = vi.fn<GenerateStream>(async () => {
            throw providerFailure;
        });
        const driver = new TestGeminiDriver(vi.fn<Generate>(), generateStream);
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Analyze the media.' }],
            runtimeOptions({
                flow: `typed-provider-failure-${retryable}`,
                operation: 'one',
                attempt: 'one',
                recorded_at: '2026-09-30T06:00:00.000Z',
            }),
            undefined,
            { stream_id: `stream:gemini:typed-provider-failure:${retryable}` },
        );

        const events = await collectCanonicalEvents(stream);

        expect(generateStream).toHaveBeenCalledOnce();
        expect(events.at(-1)).toMatchObject({
            type: 'stream_terminated',
            outcome: 'failed',
            diagnostic: {
                code: 'PROVIDER_STREAM_FAILED',
                message: 'Provider stream ended before canonical response acceptance',
                retryable,
            },
        });
        expect(JSON.stringify(events)).not.toContain('private provider details');
        expect(JSON.stringify(events)).not.toContain(marker);
    });

    it('retains the authoritative Gemini response when final event delivery exceeds its budget', async () => {
        const generateStream = vi.fn<GenerateStream>(async () =>
            (async function* () {
                yield response({
                    id: 'response-typed-delivery-budget',
                    content: { role: 'model', parts: [{ text: 'authoritative text' }] },
                });
            })(),
        );
        const driver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Answer.' }],
            runtimeOptions({
                flow: 'typed-delivery-budget',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:11:51.500Z',
            }),
            undefined,
            { stream_id: 'stream:gemini:typed-delivery-budget', max_events: 6 },
        );
        const events = await collectCanonicalEvents(stream);

        expect(events.at(-1)).toMatchObject({
            type: 'stream_terminated',
            outcome: 'failed',
            diagnostic: { code: 'CANONICAL_EVENT_DELIVERY_FAILED' },
        });
        expect(events.some((event) => event.type === 'response_accepted')).toBe(false);
        expect(stream.completion?.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({ type: 'text', text: 'authoritative text' }),
        );
    });

    it('accepts an omitted Gemini extension without inventing a display draft', async () => {
        const generateStream = vi.fn<GenerateStream>(async () =>
            (async function* () {
                yield response({
                    id: 'response-typed-extension',
                    content: {
                        role: 'model',
                        parts: [{ executableCode: { language: Language.PYTHON, code: 'print(1)' } }],
                    },
                });
            })(),
        );
        const driver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Run code.' }],
            runtimeOptions({
                flow: 'typed-extension',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:11:51.750Z',
            }),
            undefined,
            { stream_id: 'stream:gemini:typed-extension' },
        );
        const events = await collectCanonicalEvents(stream);

        expect(events.some((event) => event.type === 'draft_block_started')).toBe(false);
        expect(events.at(-1)?.type).toBe('response_accepted');
        expect(stream.completion?.accepted_output.turn.blocks).toEqual([]);
        expect(stream.completion?.accepted_output.completeness).toMatchObject({
            semantic_content: 'partial',
            omitted_block_ids: expect.arrayContaining([expect.any(String)]),
        });
    });

    it('aborts a pending typed Gemini read and emits one cancellation terminal', async () => {
        let providerSignal: AbortSignal | undefined;
        const generateStream = vi.fn<GenerateStream>(async (request) => {
            providerSignal = request.config?.abortSignal;
            return {
                [Symbol.asyncIterator]() {
                    let sent = false;
                    return {
                        next: async (): Promise<IteratorResult<GenerateContentResponse>> => {
                            if (!sent) {
                                sent = true;
                                return {
                                    done: false,
                                    value: {
                                        candidates: [{ content: { role: 'model', parts: [{ text: 'started' }] } }],
                                    } as GenerateContentResponse,
                                };
                            }
                            return new Promise((resolve) => {
                                providerSignal?.addEventListener(
                                    'abort',
                                    () => resolve({ done: true, value: undefined }),
                                    { once: true },
                                );
                            });
                        },
                        return: async () => ({ done: true, value: undefined }),
                    };
                },
            };
        });
        const driver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: 'Wait.' }],
            runtimeOptions({
                flow: 'typed-cancel',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:11:52.000Z',
            }),
            undefined,
            { stream_id: 'stream:gemini:typed-cancel' },
        );
        const iterator = stream[Symbol.asyncIterator]();
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_started' }, done: false });
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_block_started' }, done: false });
        await expect(iterator.next()).resolves.toMatchObject({ value: { type: 'draft_text_delta' }, done: false });

        const terminal = await stream.cancel();

        expect(providerSignal?.aborted).toBe(true);
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        await expect(iterator.next()).resolves.toMatchObject({ value: terminal, done: false });
        await expect(iterator.next()).resolves.toEqual({ value: undefined, done: true });
        expect(stream.completion).toBeUndefined();
    });

    it.each([
        ['transcription only', [], ['Hello', { text: 'Hello', speaker_label: 'A' }]],
        ['ordinary text present', [{ text: 'Summary' }], ['Summary', { text: 'Hello', speaker_label: 'A' }]],
    ] as const)('delivers %s through the finite typed audio path', async (_label, prefix, expected) => {
        const nativeResponse = response({
            id: `response-typed-${_label}`,
            content: {
                role: 'model',
                parts: [
                    ...prefix,
                    {
                        audioTranscription: {
                            text: 'Hello',
                            speakerLabel: 'A',
                            words: [{ word: 'Hello', startOffset: '0s', endOffset: '1s' }],
                        },
                    },
                ],
            },
        });
        const generate = vi.fn<Generate>(async () => nativeResponse);
        const driver = new TestGeminiDriver(generate);
        const audio = {
            name: 'recording.wav',
            mime_type: 'audio/wav',
            getStream: vi.fn(async () => new Blob(['audio']).stream()),
            getURL: vi.fn(async () => 'https://example.test/recording.wav'),
            getURI: vi.fn(async () => 'gs://bucket/recording.wav'),
        };
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: '', files: [audio] }],
            {
                ...runtimeOptions({
                    flow: `typed-audio-${_label}`,
                    operation: 'generate',
                    attempt: 'first',
                    recorded_at: '2026-09-30T01:11:55.000Z',
                }),
                model: 'gemini-3.5-transcribe-preview',
            },
            undefined,
            { stream_id: `stream:gemini:typed-audio-${_label}` },
        );
        const events = await collectCanonicalEvents(stream);
        const blocks = stream.completion?.accepted_output.turn.blocks ?? [];

        expect(events).toEqual([
            expect.objectContaining({ type: 'response_accepted', origin: 'live_transport', sequence: 0 }),
        ]);
        expect(blocks.filter((block) => block.type === 'text').map((block) => block.text)).toEqual([expected[0]]);
        expect(blocks.find((block) => block.type === 'json')).toMatchObject({
            type: 'json',
            value: expect.objectContaining(expected[1]),
        });
        expect(generate).toHaveBeenCalledOnce();
        expect(audio.getStream).not.toHaveBeenCalled();
    });

    it('cancels finite typed Gemini audio before accepting a response', async () => {
        let providerSignal: AbortSignal | undefined;
        let transportOpened: (() => void) | undefined;
        const opened = new Promise<void>((resolve) => {
            transportOpened = resolve;
        });
        const generate = vi.fn<Generate>(async (request) => {
            providerSignal = request.config?.abortSignal;
            transportOpened?.();
            return new Promise<GenerateContentResponse>((_resolve, reject) => {
                providerSignal?.addEventListener(
                    'abort',
                    () => reject(providerSignal?.reason ?? new Error('aborted')),
                    { once: true },
                );
            });
        });
        const driver = new TestGeminiDriver(generate);
        const audio = {
            name: 'recording.wav',
            mime_type: 'audio/wav',
            getStream: vi.fn(async () => new Blob(['audio']).stream()),
            getURL: vi.fn(async () => 'https://example.test/recording.wav'),
            getURI: vi.fn(async () => 'gs://bucket/recording.wav'),
        };
        const stream = await driver.streamCanonicalEvents(
            [{ role: PromptRole.user, content: '', files: [audio] }],
            {
                ...runtimeOptions({
                    flow: 'typed-audio-cancel',
                    operation: 'generate',
                    attempt: 'first',
                    recorded_at: '2026-09-30T01:11:56.000Z',
                }),
                model: 'gemini-3.5-transcribe-preview',
            },
            undefined,
            { stream_id: 'stream:gemini:typed-audio-cancel' },
        );
        const iterator = stream[Symbol.asyncIterator]();
        const pending = iterator.next();
        await opened;

        const terminal = await stream.cancel();

        expect(providerSignal?.aborted).toBe(true);
        expect(terminal).toMatchObject({ type: 'stream_terminated', outcome: 'cancelled' });
        await expect(pending).resolves.toMatchObject({ value: terminal, done: false });
        await expect(iterator.next()).resolves.toEqual({ value: undefined, done: true });
        expect(stream.completion).toBeUndefined();
    });

    it.each([
        ['array', '[1,null]', { type: 'array' }, [1, null]],
        ['null', 'null', { type: 'null' }, null],
        ['string', '"Tokyo"', { type: 'string' }, 'Tokyo'],
        ['number', '42', { type: 'number' }, 42],
        ['boolean', 'true', { type: 'boolean' }, true],
    ] as const)('persists a top-level JSON %s as canonical structured output', async (label, raw, schema, expected) => {
        const nativeResponse = response({
            id: `response-${label}`,
            content: { role: 'model', parts: [{ text: raw }] },
        });
        const driver = new TestGeminiDriver(async () => nativeResponse);
        const completion = await driver.execute([{ role: PromptRole.user, content: `Return a JSON ${label}.` }], {
            ...runtimeOptions({
                flow: `structured-${label}`,
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:10:00.000Z',
            }),
            result_schema: schema,
        });

        expect(completion.result).toEqual([{ type: 'json', value: expected }]);
        expect(latestGeneratedJson(completion.conversation)).toEqual(expected);
        expect(
            exportLegacyGeminiConversation(parseConversationDocument(completion.conversation))._arrayConversation.at(
                -1,
            ),
        ).toEqual(nativeResponse.candidates?.[0]?.content);
    });

    it('normalizes split streamed JSON around signed reasoning and recovers exact native parts', async () => {
        const firstFragment = '```json\n{"answer":';
        const secondFragment = '"Tokyo"}\n```';
        const nativeParts = [
            { text: firstFragment },
            { text: 'Check the requested shape.', thought: true, thoughtSignature: 'signed-structured-reasoning' },
            { text: secondFragment },
            { text: '', thoughtSignature: 'signed-empty-answer-terminal' },
            { text: '', thought: true, thoughtSignature: 'signed-empty-reasoning-terminal' },
        ];
        const generateStream = vi.fn<GenerateStream>(async () =>
            (async function* () {
                yield {
                    candidates: [{ content: { role: 'model', parts: [nativeParts[0]] } }],
                } as GenerateContentResponse;
                yield {
                    candidates: [{ content: { role: 'model', parts: [nativeParts[1]] } }],
                } as GenerateContentResponse;
                yield {
                    candidates: [{ content: { role: 'model', parts: [nativeParts[2]] } }],
                } as GenerateContentResponse;
                yield response({
                    id: 'response-structured-stream',
                    content: { role: 'model', parts: nativeParts.slice(3) },
                });
            })(),
        );
        const driver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const segments = [{ role: PromptRole.user, content: 'Return the city as JSON.' }];
        const result_schema: NonNullable<ExecutionOptions['result_schema']> = {
            type: 'object',
            properties: { answer: { type: 'string' } },
            required: ['answer'],
            additionalProperties: false,
        };
        const first = await driver.stream(segments, {
            ...runtimeOptions({
                flow: 'structured-stream',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T01:20:00.000Z',
            }),
            result_schema,
        });
        await drain(first);

        expect(first.completion?.result).toEqual([
            { type: 'json', value: { answer: 'Tokyo' } },
            { type: 'thoughts', value: 'Check the requested shape.' },
        ]);
        expect(latestGeneratedJson(first.completion?.conversation)).toEqual({ answer: 'Tokyo' });
        const persisted = parseConversationDocument(JSON.parse(JSON.stringify(first.completion?.conversation)));
        expect(exportLegacyGeminiConversation(persisted)._arrayConversation.at(-1)?.parts).toEqual(nativeParts);

        const retried = await driver.stream(segments, {
            ...runtimeOptions({
                flow: 'structured-stream',
                operation: 'generate',
                attempt: 'retry',
                recorded_at: '2026-09-30T01:25:00.000Z',
                conversation: persisted,
            }),
            result_schema,
        });
        await drain(retried);
        expect(retried.completion?.result).toEqual(first.completion?.result);
        expect(retried.completion?.conversation).toEqual(persisted);
        expect(generateStream).toHaveBeenCalledTimes(1);
    });

    it('preserves native call identity and internal error status through a tool continuation', async () => {
        const requests: GenerateContentParameters[] = [];
        const replies = [
            response({
                id: 'response-tool-call',
                content: {
                    role: 'model',
                    parts: [
                        {
                            functionCall: { id: 'native-call-1', name: 'lookup', args: { city: 'Tokyo' } },
                            thoughtSignature: 'signed-call',
                        },
                    ],
                },
            }),
            response({
                id: 'response-tool-result',
                content: { role: 'model', parts: [{ text: 'The lookup failed.' }] },
            }),
        ];
        const generate = vi.fn<Generate>(async (request) => {
            requests.push(request);
            const next = replies.shift();
            if (next === undefined) throw new Error('unexpected provider call');
            return next;
        });
        const driver = new TestGeminiDriver(generate);
        const tools: NonNullable<ExecutionOptions['tools']> = [
            { name: 'lookup', input_schema: { type: 'object', additionalProperties: true } },
        ];
        const first = await driver.execute([{ role: PromptRole.user, content: 'Look it up.' }], {
            ...runtimeOptions({
                flow: 'tools',
                operation: 'ask',
                attempt: 'ask',
                recorded_at: '2026-09-30T02:00:00.000Z',
            }),
            tools,
        });
        expect(first.tool_use).toEqual([
            {
                id: 'native-call-1',
                tool_name: 'lookup',
                tool_input: { city: 'Tokyo' },
                thought_signature: 'signed-call',
            },
        ]);

        const second = await driver.execute(
            [
                {
                    role: PromptRole.tool,
                    tool_use_id: 'native-call-1',
                    tool_result_status: 'error',
                    content: '{"error":"not found"}',
                },
            ],
            {
                ...runtimeOptions({
                    flow: 'tools',
                    operation: 'continue',
                    attempt: 'continue',
                    recorded_at: '2026-09-30T02:01:00.000Z',
                    conversation: parseConversationDocument(first.conversation),
                }),
                tools,
            },
        );
        const projectedResponse = requestContents(requests[1])
            .flatMap((content) => content.parts ?? [])
            .find((part) => part.functionResponse !== undefined)?.functionResponse;
        expect(projectedResponse).toEqual({
            id: 'native-call-1',
            name: 'lookup',
            response: { error: 'not found' },
        });
        expect(JSON.stringify(requests[1])).not.toContain('_llumiverse_tool_result_status');
        expect(JSON.stringify(requests[1])).not.toContain('_llumiverse_tool_result_text');
        const document = parseConversationDocument(second.conversation);
        const toolTurn = document.turns.find((turn) => turn.kind === 'tool');
        expect(toolTurn?.blocks[0]).toMatchObject({ call_id: 'native-call-1', status: 'error' });
        expect(toolTurn?.blocks[0].content).toContainEqual(
            expect.objectContaining({ type: 'text', text: '{"error":"not found"}' }),
        );
        expect(Object.values(document.execution_receipts)).toContainEqual(
            expect.objectContaining({ call_id: 'native-call-1', status: 'error' }),
        );
    });

    it('reuses an accepted input operation exactly once with ordered audio content', async () => {
        const prompt = {
            contents: [
                {
                    role: 'user' as const,
                    parts: [
                        { text: 'Describe this clip.' },
                        { inlineData: { data: 'YXVkaW8=', mimeType: 'audio/mpeg' } },
                    ],
                },
            ],
        };
        const firstOptions = runtimeOptions({
            flow: 'accepted-input',
            operation: 'generate',
            attempt: 'first',
            recorded_at: '2026-09-30T03:00:00.000Z',
        });
        const inputOnly = await prepareGeminiCanonicalState({
            conversation: undefined,
            prompt,
            options: firstOptions,
            provider: 'vertexai',
        });
        const requests: GenerateContentParameters[] = [];
        const generate = vi.fn<Generate>(async (request) => {
            requests.push(request);
            return response({ id: 'response-audio-input', content: { role: 'model', parts: [{ text: 'Audio.' }] } });
        });
        const model = new GeminiModelDefinition('gemini-2.5-pro');
        const driver = new TestGeminiDriver(generate);
        await model.requestTextCompletion(driver, prompt, {
            ...runtimeOptions({
                flow: 'accepted-input',
                operation: 'generate',
                attempt: 'retry',
                recorded_at: '2026-09-30T03:05:00.000Z',
                conversation: JSON.parse(JSON.stringify(inputOnly.document)),
            }),
        });

        expect(requests).toHaveLength(1);
        expect(requestContents(requests[0])).toEqual(prompt.contents);
        const audioParts = requestContents(requests[0])
            .flatMap((content) => content.parts ?? [])
            .filter((part) => part.inlineData?.mimeType === 'audio/mpeg');
        expect(audioParts).toHaveLength(1);
    });

    it('streams signed parts, uses trailing usage, and recovers without a second provider stream', async () => {
        const generateStream = vi.fn<GenerateStream>(async () =>
            (async function* () {
                yield {
                    candidates: [
                        {
                            content: {
                                role: 'model',
                                parts: [{ text: 'plan', thought: true, thoughtSignature: 'reasoning-signature' }],
                            },
                        },
                    ],
                } as unknown as GenerateContentResponse;
                yield {
                    candidates: [
                        {
                            content: {
                                role: 'model',
                                parts: [
                                    {
                                        functionCall: { id: 'native-stream-call', name: 'lookup', args: { q: 'x' } },
                                        thoughtSignature: 'call-signature',
                                    },
                                ],
                            },
                        },
                    ],
                } as unknown as GenerateContentResponse;
                yield {
                    responseId: 'response-stream',
                    modelVersion: 'gemini-2.5-pro-002',
                    candidates: [
                        {
                            finishReason: FinishReason.STOP,
                            content: { role: 'model', parts: [{ text: 'done' }] },
                        },
                    ],
                } as GenerateContentResponse;
                yield {
                    usageMetadata: {
                        promptTokenCount: 10,
                        cachedContentTokenCount: 2,
                        candidatesTokenCount: 3,
                        thoughtsTokenCount: 1,
                        totalTokenCount: 14,
                        trafficType: 'ON_DEMAND_FLEX',
                    },
                } as GenerateContentResponse;
            })(),
        );
        const driver = new TestGeminiDriver(async () => {
            throw new Error('blocking transport not expected');
        }, generateStream);
        const model = new GeminiModelDefinition('gemini-2.5-pro');
        const prompt = { contents: [{ role: 'user' as const, parts: [{ text: 'Stream.' }] }] };
        const tools: NonNullable<ExecutionOptions['tools']> = [
            { name: 'lookup', input_schema: { type: 'object', additionalProperties: true } },
        ];
        const firstOptions = {
            ...runtimeOptions({
                flow: 'stream',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T04:00:00.000Z',
            }),
            tools,
        };
        const stream = await model.requestTextCompletionStream(driver, prompt, firstOptions);
        const chunks = [];
        for await (const chunk of stream) chunks.push(chunk);
        const document = parseConversationDocument(await stream.finalizeConversation?.());
        expect(chunks.flatMap((chunk) => chunk.tool_use ?? [])).toContainEqual(
            expect.objectContaining({ id: 'native-stream-call', tool_name: 'lookup' }),
        );
        expect(chunks.at(-1)?.token_usage).toEqual({
            total: 14,
            prompt: 10,
            prompt_cached: 2,
            prompt_new: 8,
            result: 4,
        });
        const generation = Object.values(document.generations).find(
            (candidate) => candidate.record_source === 'executed',
        );
        expect(generation).toMatchObject({
            finish_reason: 'tool_use',
            usage: { input_tokens: 10, output_tokens: 4, reasoning_tokens: 1, total_tokens: 14 },
        });
        expect(exportLegacyGeminiConversation(document)._arrayConversation.at(-1)?.parts).toEqual([
            { text: 'plan', thought: true, thoughtSignature: 'reasoning-signature' },
            {
                functionCall: { id: 'native-stream-call', name: 'lookup', args: { q: 'x' } },
                thoughtSignature: 'call-signature',
            },
            { text: 'done' },
        ]);

        const recovered = await model.requestTextCompletionStream(driver, prompt, {
            ...runtimeOptions({
                flow: 'stream',
                operation: 'generate',
                attempt: 'retry',
                recorded_at: '2026-09-30T04:05:00.000Z',
                conversation: JSON.parse(JSON.stringify(document)),
            }),
            tools,
        });
        await drain(recovered);
        expect(await recovered.finalizeConversation?.()).toEqual(document);
        expect(generateStream).toHaveBeenCalledTimes(1);
    });

    it('fails closed for truncated or ambiguous responses and accepts an explicit prompt block terminal', async () => {
        const model = new GeminiModelDefinition('gemini-2.5-pro');
        const prompt = { contents: [{ role: 'user' as const, parts: [{ text: 'Continue.' }] }] };
        const truncatedDriver = new TestGeminiDriver(
            async () => {
                throw new Error('blocking transport not expected');
            },
            async () =>
                (async function* () {
                    yield {
                        candidates: [{ content: { role: 'model', parts: [{ text: 'partial' }] } }],
                    } as GenerateContentResponse;
                })(),
        );
        const truncated = await model.requestTextCompletionStream(
            truncatedDriver,
            prompt,
            runtimeOptions({
                flow: 'truncated',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T05:00:00.000Z',
            }),
        );
        await drain(truncated);
        await expect(truncated.finalizeConversation?.()).rejects.toThrow(/without a terminal finish reason/);

        const ambiguousDriver = new TestGeminiDriver(
            async () =>
                ({
                    candidates: [
                        { finishReason: FinishReason.STOP, content: { role: 'model', parts: [{ text: 'one' }] } },
                        { finishReason: FinishReason.STOP, content: { role: 'model', parts: [{ text: 'two' }] } },
                    ],
                }) as GenerateContentResponse,
        );
        await expect(
            model.requestTextCompletion(
                ambiguousDriver,
                prompt,
                runtimeOptions({
                    flow: 'ambiguous',
                    operation: 'generate',
                    attempt: 'first',
                    recorded_at: '2026-09-30T05:10:00.000Z',
                }),
            ),
        ).rejects.toThrow(/requires one candidate/);

        const missingDriver = new TestGeminiDriver(
            async () => ({ candidates: [] }) as unknown as GenerateContentResponse,
        );
        await expect(
            model.requestTextCompletion(
                missingDriver,
                prompt,
                runtimeOptions({
                    flow: 'missing',
                    operation: 'generate',
                    attempt: 'first',
                    recorded_at: '2026-09-30T05:20:00.000Z',
                }),
            ),
        ).rejects.toThrow(/no candidate or prompt block reason/);

        const blockedDriver = new TestGeminiDriver(
            async () =>
                ({
                    candidates: [],
                    promptFeedback: { blockReason: 'BLOCKLIST', blockReasonMessage: 'Prompt blocked.' },
                }) as unknown as GenerateContentResponse,
        );
        const blocked = await model.requestTextCompletion(
            blockedDriver,
            prompt,
            runtimeOptions({
                flow: 'blocked',
                operation: 'generate',
                attempt: 'first',
                recorded_at: '2026-09-30T05:30:00.000Z',
            }),
        );
        expect(blocked.result).toEqual([{ type: 'text', value: 'Prompt blocked.' }]);
        const blockedTurn = parseConversationDocument(blocked.conversation).turns.at(-1);
        expect(blockedTurn?.kind).toBe('agent');
        expect(blockedTurn?.blocks).toContainEqual(expect.objectContaining({ type: 'text', text: 'Prompt blocked.' }));
    });

    it('keeps generated conversational audio explicitly unsupported until typed audio media exists', async () => {
        const model = new GeminiModelDefinition('gemini-2.5-pro');
        const driver = new TestGeminiDriver(async () =>
            response({
                id: 'response-audio-output',
                content: {
                    role: 'model',
                    parts: [{ inlineData: { data: 'YXVkaW8=', mimeType: 'audio/mpeg' } }],
                },
            }),
        );
        await expect(
            model.requestTextCompletion(
                driver,
                { contents: [{ role: 'user', parts: [{ text: 'Speak.' }] }] },
                runtimeOptions({
                    flow: 'audio-output',
                    operation: 'generate',
                    attempt: 'first',
                    recorded_at: '2026-09-30T06:00:00.000Z',
                }),
            ),
        ).rejects.toThrow(/Audio output requires a file speech model/);
    });
});
