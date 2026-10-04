import {
    appendConversationRecordsWithProcessing,
    appendToolExecutionResult,
    type ConversationDocument,
    createToolResultTextExternalizationProcessor,
    fingerprintJson,
    isToolResultTextProcessor,
    type ProcessingStore,
    parseConversationDocument,
    runProcessingJob,
    setProcessingPolicy,
    toolResultExternalizationArchiveInputs,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import { compileBedrockConverseConversation } from '../bedrock/bedrock-converse-conversation-adapter.js';
import { compileOpenAIChatCompletionsConversation } from '../openai/openai-chat-conversation-adapter.js';
import {
    compileOpenAIResponsesConversation,
    importOpenAIResponsesHistory,
} from '../openai/openai-responses-conversation-adapter.js';
import { compileClaudeMessagesConversation } from '../shared/claude-messages-conversation-adapter.js';
import { compileGeminiConversation } from '../vertexai/models/gemini-conversation-adapter.js';
import { prepareCanonicalContext } from './canonical-runtime.js';

const AT = '2026-09-11T00:01:00.000Z';
const ORIGINAL = `Exact result α\n${'long result '.repeat(1000)}`;

class MemoryStore implements ProcessingStore {
    constructor(public current: ConversationDocument) {}
    async load(): Promise<ConversationDocument> {
        return structuredClone(this.current);
    }
    async commit(revision: number, document: ConversationDocument): Promise<boolean> {
        if (this.current.revision !== revision) return false;
        this.current = parseConversationDocument(structuredClone(document));
        return true;
    }
}

async function externalizedToolResult(): Promise<ConversationDocument> {
    const { document: initial } = await importOpenAIResponsesHistory(
        [
            {
                type: 'function_call',
                id: 'item:call',
                call_id: 'call:one',
                name: 'write_artifact',
                arguments: '{"path":"report.txt","content":"executed arguments"}',
                status: 'completed',
            },
        ],
        {
            conversation_id: 'conversation:tool-result-reference',
            recorded_at: AT,
            provider: 'openai',
            model: 'gpt-4o-mini',
            source_request_id: 'request:tool-result',
            tool_definitions: [
                {
                    id: 'definition:write',
                    name: 'write_artifact',
                    version: '1',
                    input_schema: {
                        type: 'object',
                        properties: { path: { type: 'string' }, content: { type: 'string' } },
                        required: ['path', 'content'],
                        additionalProperties: false,
                    },
                },
            ],
        },
    );
    const callTurn = initial.turns.find(
        (turn) => turn.kind === 'agent' && turn.blocks.some((block) => block.type === 'tool_call'),
    );
    const callBlock =
        callTurn?.kind === 'agent' ? callTurn.blocks.find((block) => block.type === 'tool_call') : undefined;
    if (callTurn === undefined || callBlock?.type !== 'tool_call') throw new Error('Imported call is absent');
    expect(callTurn.blocks.some((block) => block.type === 'native_replay')).toBe(true);
    const configured = (
        await setProcessingPolicy(initial, {
            operation_id: 'policy:tool-result',
            expected_revision: initial.revision,
            recorded_at: AT,
            enabled: true,
            processors: [
                {
                    id: 'externalize-tool-result-text',
                    version: '1',
                    scope: 'on_append',
                    config: {},
                    required: true,
                    failure_behavior: 'block',
                },
            ],
        })
    ).document;
    const source = {
        conversation: { conversation_id: initial.id, revision: configured.revision },
        turn_id: callTurn.id,
        block_id: callBlock.id,
        call_id: callBlock.call_id,
        call_fingerprint: await fingerprintJson(callBlock),
    };
    const resultBlock = {
        id: 'block:result',
        type: 'tool_result' as const,
        call_id: source.call_id,
        status: 'success' as const,
        content: [{ id: 'block:text', type: 'text' as const, format: 'plain' as const, text: ORIGINAL }],
    };
    const accepted = await appendToolExecutionResult(
        configured,
        {
            source,
            turn: {
                id: 'turn:result',
                kind: 'tool',
                authority: 'ordinary',
                status: 'completed',
                timestamps: { recorded_at: AT },
                provenance: { type: 'received' },
                model_visibility: 'include',
                blocks: [resultBlock],
                execution_id: 'execution:one',
            },
            execution_receipt: {
                id: 'execution:one',
                call_id: source.call_id,
                executor: 'application',
                status: 'success',
                result_turn_id: 'turn:result',
                result_fingerprint: await fingerprintJson(resultBlock),
                recorded_at: AT,
                call_source: source,
            },
        },
        {
            expected_revision: configured.revision,
            operation_id: 'append:result',
            recorded_at: AT,
        },
    );
    const job = Object.values(accepted.document.processing.jobs ?? {}).find(isToolResultTextProcessor);
    if (job?.selection.kind !== 'entries') throw new Error('Expected one tool-result processing job');
    const archive = await toolResultExternalizationArchiveInputs(accepted.document, job);
    const integrity = archive.integrities[0];
    if (!integrity) throw new Error('Expected text integrity');
    const readInputSchema = {
        type: 'object',
        properties: { asset_id: { type: 'string' } },
        required: ['asset_id'],
        additionalProperties: false,
    };
    const definitionVersion = await fingerprintJson(readInputSchema);
    const archived = await appendConversationRecordsWithProcessing(
        accepted.document,
        {
            assets: [
                {
                    id: 'asset:result',
                    kind: 'text',
                    mime_type: 'text/plain',
                    storage: { type: 'external', resolver: 'test.blob', locator: { key: 'result' } },
                    provenance: { type: 'received' },
                    created_at: AT,
                    ...integrity,
                },
            ],
            tool_definitions: [
                {
                    id: 'definition:read',
                    name: 'read_artifact',
                    version: definitionVersion,
                    input_schema: readInputSchema,
                },
            ],
            active_tool_definition_ids: ['definition:read'],
        },
        {
            expected_revision: accepted.document.revision,
            operation_id: `processing:archive:${job.id}`,
            payload_fingerprint: archive.payload_fingerprint,
            recorded_at: AT,
        },
    );
    const store = new MemoryStore(archived.document);
    const processor = createToolResultTextExternalizationProcessor(({ asset }) => ({
        capability: 'read_artifact',
        version: 1,
        tool_definition_id: 'definition:read',
        arguments: { asset_id: asset.id },
    }));
    const completion = await runProcessingJob(store, { resolve: () => processor }, job.id, 'attempt:one', () => AT);
    if (completion.status !== 'completed') throw new Error('Tool-result externalization did not complete');
    return store.current;
}

const compilers = {
    'OpenAI Chat': (document: ConversationDocument) => compileOpenAIChatCompletionsConversation(document).conversation,
    'OpenAI Responses': (document: ConversationDocument) => compileOpenAIResponsesConversation(document).conversation,
    'Claude Messages': (document: ConversationDocument) => compileClaudeMessagesConversation(document).conversation,
    Gemini: (document: ConversationDocument) => compileGeminiConversation(document).conversation,
    'Bedrock Converse': (document: ConversationDocument) => compileBedrockConverseConversation(document).conversation,
};

describe('native projection of actual tool-result externalization', () => {
    it('prepares a retained materialized tool result through its accepted replacement receipt', async () => {
        const document = await externalizedToolResult();
        const accepted = document.operation_receipts['append:result'];
        if (!accepted) throw new Error('Accepted materialized tool result is absent');
        const options = {
            model: 'gpt-4o-mini',
            conversation: document,
            conversation_runtime: {
                conversation_id: document.id,
                request_id: 'request:after-tool-result',
                attempt_id: 'attempt:after-tool-result',
                input_operation_id: 'input:after-tool-result',
                response_operation_id: 'response:after-tool-result',
                recorded_at: AT,
                purpose: 'interaction' as const,
                materialized_input: {
                    operation_id: accepted.id,
                    result_revision: accepted.result_revision,
                },
            },
        };
        expect(
            (
                await prepareCanonicalContext({
                    options,
                    provider: 'openai',
                    protocol: 'openai.responses',
                    adapter_version: 'openai-responses@1',
                })
            ).document,
        ).toEqual(document);
        const changed = structuredClone(document);
        const compaction = Object.values(changed.compactions)[0];
        if (!compaction?.metadata) throw new Error('Accepted tool-result compaction metadata is absent');
        compaction.metadata.payload_fingerprint = `sha256:${'0'.repeat(64)}`;
        await expect(
            prepareCanonicalContext({
                options: { ...options, conversation: changed },
                provider: 'openai',
                protocol: 'openai.responses',
                adapter_version: 'openai-responses@1',
            }),
        ).rejects.toThrow('accepted compaction receipt');
    });
    it.each(Object.entries(compilers))(
        '%s keeps the exact call and result with a bounded retrieval cue',
        async (_name, compile) => {
            const document = await externalizedToolResult();
            const native = JSON.stringify(compile(document)) ?? '';
            expect(native).toContain('read_artifact');
            expect(native).toContain('asset:result');
            if (_name === 'Gemini') {
                // Gemini's native function pair has name/arguments but no call-id field for this target.
                expect(compile(document)).toMatchObject({
                    contents: [
                        {
                            role: 'model',
                            parts: [
                                {
                                    functionCall: {
                                        name: 'write_artifact',
                                        args: { path: 'report.txt', content: 'executed arguments' },
                                    },
                                },
                            ],
                        },
                        {
                            role: 'user',
                            parts: [{ functionResponse: { name: 'write_artifact' } }],
                        },
                    ],
                });
            } else {
                expect(native).toContain('call:one');
            }
            expect(native).not.toContain(ORIGINAL);
            expect(native.length).toBeLessThan(8192);
            const originalCall = document.turns.find((turn) =>
                turn.blocks.some((block) => block.type === 'tool_call' && block.call_id === 'call:one'),
            );
            const callReplay = originalCall?.blocks.find((block) => block.type === 'native_replay');
            expect(callReplay?.type === 'native_replay' ? callReplay.payload : undefined).toMatchObject({
                type: 'openai_responses_items',
                items: [
                    {
                        type: 'function_call',
                        id: 'item:call',
                        call_id: 'call:one',
                        name: 'write_artifact',
                        arguments: '{"path":"report.txt","content":"executed arguments"}',
                    },
                ],
            });
            const original = document.turns.find((turn) => turn.id === 'turn:result');
            if (original?.kind !== 'tool' || original.blocks[0]?.type !== 'tool_result') {
                throw new Error('Original executed tool result is absent');
            }
            expect(original.blocks[0].content[0]).toMatchObject({ type: 'text', text: ORIGINAL });
            expect(document.execution_receipts['execution:one']?.result_fingerprint).toBe(
                await fingerprintJson(original.blocks[0]),
            );
            const derived = Object.values(document.compactions)
                .flatMap((compaction) => compaction.replacement_turns)
                .find((turn) => turn.kind === 'tool');
            if (derived?.kind !== 'tool' || derived.blocks[0]?.type !== 'tool_result') {
                throw new Error('Projected tool result is absent');
            }
            const reference = derived.blocks[0].content.find((block) => block.type === 'external_reference');
            if (reference?.type !== 'external_reference') throw new Error('Projected retrieval reference is absent');
            expect(reference.asset_id).toBe('asset:result');
            expect(reference.content_hash).toBe(document.assets['asset:result']?.content_hash);
            expect(reference.preview).toBe(ORIGINAL.slice(0, 512));
            expect(reference.retrieval.arguments).toEqual({ asset_id: 'asset:result' });
            // Native JSON escapes the preview's line breaks; compare its exact encoded field content.
            expect(native).toContain(JSON.stringify(`Preview: ${reference.preview}`).slice(1, -1));
            expect(native.match(/Full original text is available/g)).toHaveLength(1);
        },
    );
});
