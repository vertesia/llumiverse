import type { ExecutionOptions } from '@llumiverse/common';
import { type ContentBlock, type ConversationDocument, parseConversationDocument } from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import {
    compileBedrockConverseConversation,
    prepareBedrockConverseCanonicalState,
} from '../bedrock/bedrock-converse-conversation-adapter.js';
import {
    compileOpenAIChatCompletionsConversation,
    prepareOpenAIChatCanonicalState,
} from '../openai/openai-chat-conversation-adapter.js';
import {
    compileOpenAIResponsesConversation,
    prepareOpenAIResponsesCanonicalState,
} from '../openai/openai-responses-conversation-adapter.js';
import {
    appendClaudeCanonicalResponse,
    compileClaudeMessagesConversation,
    decodeClaudeCanonicalResponse,
    finalizeClaudePreparedRequest,
    prepareClaudeCanonicalState,
} from '../shared/claude-messages-conversation-adapter.js';
import {
    compileGeminiConversation,
    prepareGeminiCanonicalState,
} from '../vertexai/models/gemini-conversation-adapter.js';
import { canonicalToolDefinitions } from './canonical-runtime.js';
import {
    type CanonicalNativeConversationProtocol,
    importBedrockConverseHistory,
    importClaudeMessagesHistory,
    importGeminiGenerateContentHistory,
    importNativeConversationHistory,
    importOpenAIChatCompletionsHistory,
    importOpenAIResponsesHistory,
    NativeConversationImportError,
} from './index.js';

const recordedAt = '2026-10-01T00:00:00.000Z';
function options(name: string): ExecutionOptions {
    return {
        model: name === 'bedrock' ? 'anthropic.claude-3-5-sonnet-20241022-v2:0' : 'fixture-model',
        tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
        conversation_runtime: {
            conversation_id: `conversation:${name}`,
            request_id: `request:${name}`,
            attempt_id: `attempt:${name}`,
            input_operation_id: `input:${name}`,
            response_operation_id: `response:${name}`,
            recorded_at: recordedAt,
        },
    };
}

const fixtures = [
    {
        name: 'chat',
        provider: 'openai',
        protocol: 'openai.chat.completions',
        history: {
            _is_openai_chat_completions: true,
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'look' },
                        { type: 'image_url', image_url: { url: 'data:image/png;base64,YWJj', detail: 'high' } },
                    ],
                },
                {
                    role: 'assistant',
                    content: 'answer',
                    reasoning_content: 'private',
                    tool_calls: [
                        {
                            id: 'call-exact',
                            type: 'function',
                            function: { name: 'lookup', arguments: '{ "city" : "Tokyo" }' },
                        },
                    ],
                },
                { role: 'tool', tool_call_id: 'call-exact', content: 'sunny' },
            ],
        },
        prepare: (conversation: unknown) =>
            prepareOpenAIChatCanonicalState({
                conversation,
                prompt: { _is_openai_chat_completions: true, messages: [{ role: 'user', content: 'continue' }] },
                options: options('chat'),
                provider: 'openai',
            }),
    },
    {
        name: 'responses',
        provider: 'openai',
        protocol: 'openai.responses',
        history: [
            {
                role: 'user',
                content: [{ type: 'input_image', image_url: 'data:image/png;base64,YWJj', detail: 'auto' }],
            },
            {
                type: 'reasoning',
                id: 'reasoning-exact',
                summary: [{ type: 'summary_text', text: 'private' }],
                encrypted_content: 'opaque',
            },
            { type: 'function_call', call_id: 'call-exact', name: 'lookup', arguments: '{ "city" : "Tokyo" }' },
            { type: 'function_call_output', call_id: 'call-exact', output: 'sunny' },
        ],
        prepare: (conversation: unknown) =>
            prepareOpenAIResponsesCanonicalState({
                conversation,
                prompt: [{ role: 'user', content: 'continue' }],
                options: options('responses'),
                provider: 'openai',
            }),
    },
    {
        name: 'claude',
        provider: 'anthropic',
        protocol: 'anthropic.messages',
        history: {
            messages: [
                {
                    role: 'user',
                    content: [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'YWJj' } }],
                },
                {
                    role: 'assistant',
                    content: [
                        { type: 'thinking', thinking: 'private', signature: 'signature-exact' },
                        { type: 'tool_use', id: 'call-exact', name: 'lookup', input: { city: 'Tokyo' } },
                    ],
                },
                { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'call-exact', content: 'sunny' }] },
            ],
        },
        prepare: (conversation: unknown) =>
            prepareClaudeCanonicalState({
                conversation,
                prompt: { messages: [{ role: 'user', content: 'continue' }] },
                options: options('claude'),
                provider: 'anthropic',
            }),
    },
    {
        name: 'gemini',
        provider: 'vertexai',
        protocol: 'google.generate_content',
        history: [
            { role: 'user', parts: [{ inlineData: { mimeType: 'image/png', data: 'YWJj' } }] },
            {
                role: 'model',
                parts: [
                    { text: 'answer', thoughtSignature: 'signature-exact' },
                    { functionCall: { id: 'call-exact', name: 'lookup', args: { city: 'Tokyo' } } },
                ],
            },
            {
                role: 'user',
                parts: [{ functionResponse: { id: 'call-exact', name: 'lookup', response: { output: 'sunny' } } }],
            },
        ],
        prepare: (conversation: unknown) =>
            prepareGeminiCanonicalState({
                conversation,
                prompt: { contents: [{ role: 'user', parts: [{ text: 'continue' }] }] },
                options: options('gemini'),
                provider: 'vertexai',
            }),
    },
    {
        name: 'bedrock',
        provider: 'bedrock',
        protocol: 'aws.bedrock.converse',
        history: {
            messages: [
                {
                    role: 'user',
                    content: [{ image: { format: 'png', source: { bytes: new Uint8Array([97, 98, 99]) } } }],
                },
                {
                    role: 'assistant',
                    content: [
                        { reasoningContent: { reasoningText: { text: 'private', signature: 'signature-exact' } } },
                        { toolUse: { toolUseId: 'call-exact', name: 'lookup', input: { city: 'Tokyo' } } },
                    ],
                },
                { role: 'user', content: [{ toolResult: { toolUseId: 'call-exact', content: [{ text: 'sunny' }] } }] },
            ],
        },
        prepare: (conversation: unknown) =>
            prepareBedrockConverseCanonicalState({
                conversation,
                prompt: {
                    modelId: 'anthropic.claude-3-5-sonnet-20241022-v2:0',
                    messages: [{ role: 'user', content: [{ text: 'continue' }] }],
                },
                options: options('bedrock'),
                provider: 'bedrock',
            }),
    },
];

function importOptions(fixture: (typeof fixtures)[number]) {
    return {
        conversation_id: `conversation:${fixture.name}`,
        recorded_at: recordedAt,
        source_request_id: `request:${fixture.name}`,
        provider: fixture.provider,
        protocol: fixture.protocol as CanonicalNativeConversationProtocol,
        model: options(fixture.name).model,
    };
}

describe('preparation from recorded native origin', () => {
    it.each(fixtures)('preserves $name semantic content with recorded model origin', async (fixture) => {
        const imported = await importNativeConversationHistory(fixture.history, {
            ...importOptions(fixture),
            tool_definitions: await canonicalToolDefinitions(options(fixture.name).tools),
        });
        const state = await fixture.prepare(imported.document);
        expect(parseConversationDocument(state.document)).toEqual(state.document);
        const call = state.document.turns
            .flatMap<ContentBlock>((turn) => turn.blocks)
            .find((block) => block.type === 'tool_call');
        expect(call).toMatchObject({ call_id: 'call-exact', tool_name: 'lookup' });
        expect(state.document.turns.filter((turn) => turn.provenance.type === 'imported')).not.toHaveLength(0);
        expect(Object.values(state.document.assets)).toEqual(
            expect.arrayContaining([expect.objectContaining({ storage: { type: 'inline_base64', data: 'YWJj' } })]),
        );
    });
});

describe('direct registered native imports', () => {
    it.each([
        { fixture: fixtures[0], importer: importOpenAIChatCompletionsHistory },
        { fixture: fixtures[1], importer: importOpenAIResponsesHistory },
        { fixture: fixtures[2], importer: importClaudeMessagesHistory },
        { fixture: fixtures[3], importer: importGeminiGenerateContentHistory },
        { fixture: fixtures[4], importer: importBedrockConverseHistory },
    ])('validates direct protocol entrypoint $fixture.name with a mandatory report', async ({ fixture, importer }) => {
        const { protocol: _protocol, ...origin } = importOptions(fixture);
        const result = await importer(fixture.history, origin);
        expect(result.report).toMatchObject({ protocol: fixture.protocol, readiness: 'not_validated' });
        await expect(importer({ not_native_history: true }, origin)).rejects.toMatchObject({
            code: 'IMPORT_INVALID_HISTORY',
        });
        await expect(importer(fixture.history, { ...origin, recorded_at: 'invented' })).rejects.toMatchObject({
            code: 'IMPORT_INVALID_OPTIONS',
        });
    });

    it.each(fixtures)('preserves $name source identities and media without receiving a prompt', async (fixture) => {
        const imported = await importNativeConversationHistory(fixture.history, {
            ...importOptions(fixture),
            tool_definitions: await canonicalToolDefinitions(options(fixture.name).tools),
        });
        const prepared = await fixture.prepare(imported.document);
        const result = await importNativeConversationHistory(fixture.history, {
            ...importOptions(fixture),
            tool_definitions: Object.values(prepared.document.tool_definitions),
            completeness: 'complete',
        });
        expect(result.document.turns).toEqual(
            prepared.document.turns.filter((turn) => turn.provenance.type === 'imported'),
        );
        expect(result.document.assets).toEqual(prepared.document.assets);
        expect(result.document.execution_receipts).toEqual(prepared.document.execution_receipts);
        expect(parseConversationDocument(JSON.parse(JSON.stringify(result.document)))).toEqual(result.document);
        const blocks = result.document.turns.flatMap<ContentBlock>((turn) => turn.blocks);
        expect(blocks).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    type: 'tool_call',
                    call_id: 'call-exact',
                    tool_name: 'lookup',
                }),
                expect.objectContaining({ type: 'tool_result', call_id: 'call-exact' }),
            ]),
        );
        expect(JSON.stringify(blocks)).toContain('Tokyo');
        if (fixture.name === 'chat' || fixture.name === 'responses') {
            const legacy = JSON.stringify(result.document);
            expect(legacy).toContain(JSON.stringify('{ "city" : "Tokyo" }').slice(1, -1));
        }
        expect(JSON.stringify(blocks)).toContain('sunny');
        expect(Object.values(result.document.assets)[0]).toMatchObject({
            mime_type: 'image/png',
            storage: { type: 'inline_base64', data: 'YWJj' },
        });
        expect(blocks.filter((block) => block.type === 'native_replay')).not.toHaveLength(0);
        expect(JSON.stringify(blocks)).toContain(
            fixture.name === 'responses' ? 'opaque' : fixture.name === 'chat' ? 'private' : 'signature-exact',
        );
        expect(result.report).toMatchObject({
            protocol: fixture.protocol,
            completeness: 'complete',
            readiness: 'not_validated',
        });
        expect(result.report.diagnostics.map((entry) => entry.code)).toContain('IMPORT_CONTINUATION_NOT_VALIDATED');
        expect(result.report.diagnostics.map((entry) => entry.code)).not.toContain('IMPORT_HISTORY_INCOMPLETE');
        expect(
            await importNativeConversationHistory(fixture.history, {
                ...importOptions(fixture),
                tool_definitions: Object.values(prepared.document.tool_definitions),
                completeness: 'complete',
            }),
        ).toEqual(result);
    });

    it.each(fixtures)('reports missing $name evidence without manufacturing a model', async (fixture) => {
        const { model: _model, ...origin } = importOptions(fixture);
        const result = await importNativeConversationHistory(fixture.history, origin);
        expect(result.report).toMatchObject({ completeness: 'unknown', readiness: 'not_validated' });
        expect(result.report.diagnostics.map((entry) => entry.code)).toEqual(
            expect.arrayContaining([
                'IMPORT_HISTORY_INCOMPLETE',
                'IMPORT_METADATA_MISSING',
                'IMPORT_TOOL_DEFINITION_MISSING',
            ]),
        );
        const protectedBlocks = result.document.turns
            .flatMap<ContentBlock>((turn) => turn.blocks)
            .filter(
                (block) => block.type === 'native_replay' && block.dependency_policy !== 'discard_on_dependency_change',
            );
        if (protectedBlocks.length) {
            expect(result.report.diagnostics.map((entry) => entry.code)).toContain(
                'IMPORT_PROTECTED_REPLAY_MODEL_UNKNOWN',
            );
        }
        for (const block of protectedBlocks)
            expect(block).toMatchObject({ compatibility_scope: { provider: fixture.provider } });
        expect(JSON.stringify(result.document)).not.toContain('fixture-model');
    });

    it('imports raw Claude arrays when protocol is explicit', async () => {
        const history = fixtures[2].history as { messages: unknown[] };
        const { protocol: _protocol, ...origin } = importOptions(fixtures[2]);
        const result = await importClaudeMessagesHistory(history.messages, origin);
        expect(result.document.turns).not.toHaveLength(0);
        expect(result.report.protocol).toBe('anthropic.messages');
    });

    it('requires explicit protocol for ambiguous text arrays with no selector', async () => {
        const { protocol: _protocol, ...origin } = importOptions(fixtures[0]);
        await expect(
            importNativeConversationHistory(
                [{ role: 'user', content: 'hello' }],
                origin as typeof origin & { protocol: CanonicalNativeConversationProtocol },
            ),
        ).rejects.toMatchObject({ code: 'IMPORT_PROTOCOL_REQUIRED' });
    });

    it('rejects an own undefined protocol as invalid JSON options', async () => {
        await expect(
            importNativeConversationHistory([{ role: 'user', content: 'hello' }], {
                ...importOptions(fixtures[0]),
                protocol: undefined as unknown as CanonicalNativeConversationProtocol,
            }),
        ).rejects.toMatchObject({ code: 'IMPORT_INVALID_OPTIONS' });
    });

    it.each(fixtures)('rejects missing $name hosting provenance with stable diagnostic', async (fixture) => {
        await expect(
            importNativeConversationHistory(fixture.history, { ...importOptions(fixture), provider: '' }),
        ).rejects.toMatchObject({ code: 'IMPORT_INVALID_OPTIONS' });
    });

    it('retains exact cause and code for an orphaned Responses result', async () => {
        const { protocol: _protocol, ...origin } = importOptions(fixtures[1]);
        await expect(
            importOpenAIResponsesHistory([{ type: 'function_call_output', output: 'orphaned' }], origin),
        ).rejects.toMatchObject({
            code: 'IMPORT_INVALID_HISTORY',
            message: 'OpenAI Responses function_call_output at items/0 has no call_id',
            cause: { message: 'OpenAI Responses function_call_output at items/0 has no call_id' },
        });
    });

    it('rejects unsupported native blocks instead of silently dropping them', async () => {
        await expect(
            importNativeConversationHistory(
                { messages: [{ role: 'assistant', content: [{ type: 'future', payload: 'protected' }] }] },
                importOptions(fixtures[2]),
            ),
        ).rejects.toBeInstanceOf(NativeConversationImportError);
    });

    it('distinguishes fragment completeness and unresolved remote media', async () => {
        const result = await importNativeConversationHistory(
            {
                _is_openai_chat_completions: true,
                messages: [
                    {
                        role: 'user',
                        content: [{ type: 'image_url', image_url: { url: 'https://example.com/image.png' } }],
                    },
                ],
            },
            { ...importOptions(fixtures[0]), completeness: 'fragment' },
        );
        expect(result.report).toMatchObject({ completeness: 'fragment', readiness: 'not_validated' });
        expect(result.report.diagnostics.map((entry) => entry.code)).toEqual(
            expect.arrayContaining(['IMPORT_HISTORY_INCOMPLETE', 'IMPORT_EXTERNAL_ASSET_UNRESOLVED']),
        );
    });

    it('bounds Bedrock binary traversal before conversion and cloning', async () => {
        const bytes = new Uint8Array(25 * 1024 * 1024);
        await expect(
            importNativeConversationHistory(
                {
                    messages: [
                        {
                            role: 'user',
                            content: [
                                {
                                    image: {
                                        format: 'png',
                                        source: { bytes },
                                    },
                                },
                            ],
                        },
                    ],
                },
                importOptions(fixtures[4]),
            ),
        ).rejects.toMatchObject({ code: 'IMPORT_INVALID_HISTORY' });
    });
});

const compilers: Record<
    string,
    (document: ConversationDocument, target?: { provider?: string; model?: string }) => unknown
> = {
    chat: compileOpenAIChatCompletionsConversation,
    responses: compileOpenAIResponsesConversation,
    claude: compileClaudeMessagesConversation,
    gemini: compileGeminiConversation,
    bedrock: compileBedrockConverseConversation,
};

describe('protected native origin readiness', () => {
    it.each(fixtures)('requires recorded origin and exact target for $name protected state', async (fixture) => {
        await expect(fixture.prepare(fixture.history)).rejects.toThrow(/unknown recorded model origin/);
        const { model: _model, ...unknownOrigin } = importOptions(fixture);
        const unknown = await importNativeConversationHistory(fixture.history, unknownOrigin);
        expect(unknown.report.diagnostics).toContainEqual(
            expect.objectContaining({
                code: 'IMPORT_PROTECTED_REPLAY_MODEL_UNKNOWN',
            }),
        );
        const compile = compilers[fixture.name];
        const target = { provider: fixture.provider, model: options(fixture.name).model };
        expect(() => compile(unknown.document, target)).toThrow(/unknown recorded model origin/);
        const known = await importNativeConversationHistory(fixture.history, importOptions(fixture));
        expect(known.report.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain(
            'IMPORT_PROTECTED_REPLAY_MODEL_UNKNOWN',
        );
        expect(() => compile(known.document, target)).not.toThrow();
        expect(() => compile(known.document)).toThrow(/compatibility scope/);
        expect(() => compile(known.document, { ...target, model: 'different-model' })).toThrow(/compatibility scope/);
        expect(() => compile(known.document, { ...target, provider: 'different-provider' })).toThrow(
            /compatibility scope/,
        );
        // An explicit host context selection may omit protected source state, without deleting it from the document.
        const selected = structuredClone(unknown.document);
        const firstUser = selected.turns.find((turn) => turn.kind === 'user');
        if (firstUser === undefined) throw new Error('Expected portable user context');
        selected.context.entries = [{ id: 'explicit-context-selection', type: 'source_turn', turn_id: firstUser.id }];
        expect(() => compile(selected, target)).not.toThrow();
        expect(selected.turns).toEqual(unknown.document.turns);
    });

    it.each([
        {
            fixture: fixtures[0],
            history: { _is_openai_chat_completions: true, messages: [{ role: 'user', content: 'portable' }] },
        },
        { fixture: fixtures[1], history: [{ role: 'user', content: 'portable' }] },
        { fixture: fixtures[2], history: { messages: [{ role: 'user', content: 'portable' }] } },
        { fixture: fixtures[3], history: [{ role: 'user', parts: [{ text: 'portable' }] }] },
        { fixture: fixtures[4], history: { messages: [{ role: 'user', content: [{ text: 'portable' }] }] } },
    ])(
        'continues portable $fixture.name native history without invented model origin',
        async ({ fixture, history }) => {
            const state = await fixture.prepare(history);
            expect(JSON.stringify(state.native_conversation)).toContain('portable');
            expect(
                state.document.turns
                    .flatMap<ContentBlock>((turn) => turn.blocks)
                    .filter((block) => block.type === 'native_replay')
                    .every((block) => block.compatibility_scope.model === undefined),
            ).toBe(true);
        },
    );
});

it('binds Claude signed generation to the requested invocation separately from provider resolved model', async () => {
    const invocationOptions = options('claude');
    if (invocationOptions.conversation_runtime === undefined) throw new Error('Expected fixture runtime');
    const state = await prepareClaudeCanonicalState({
        conversation: undefined,
        prompt: { messages: [{ role: 'user', content: 'question' }] },
        options: {
            ...invocationOptions,
            conversation_runtime: { ...invocationOptions.conversation_runtime, completed_at: recordedAt },
        },
        provider: 'anthropic',
    });
    const prepared = await finalizeClaudePreparedRequest(state, {
        model: 'fixture-model',
        max_tokens: 128,
        messages: state.native_conversation.messages,
    });
    const decoded = await decodeClaudeCanonicalResponse(
        {
            id: 'claude-response:origin',
            type: 'message',
            role: 'assistant',
            model: 'provider-resolved-version',
            container: null,
            diagnostics: null,
            stop_details: null,
            content: [
                { type: 'thinking', thinking: 'private plan', signature: 'signature-exact' },
                { type: 'text', text: 'answer', citations: null },
            ],
            stop_reason: 'end_turn',
            stop_sequence: null,
            usage: {
                input_tokens: 1,
                output_tokens: 1,
                cache_creation: null,
                cache_creation_input_tokens: null,
                cache_read_input_tokens: null,
                inference_geo: null,
                output_tokens_details: null,
                server_tool_use: null,
                service_tier: null,
            },
        },
        prepared,
    );
    expect(decoded.generation).toMatchObject({
        requested_model: 'fixture-model',
        resolved_model: 'provider-resolved-version',
    });
    const document = appendClaudeCanonicalResponse(prepared, decoded);
    expect(decoded.turns[0].blocks).toContainEqual(
        expect.objectContaining({
            type: 'native_replay',
            compatibility_scope: expect.objectContaining({ model: 'fixture-model' }),
        }),
    );
    const target = { provider: 'anthropic', model: 'fixture-model' };
    expect(compileClaudeMessagesConversation(document, target).conversation.messages.at(-1)?.content).toContainEqual({
        type: 'thinking',
        thinking: 'private plan',
        signature: 'signature-exact',
    });
    expect(() =>
        compileClaudeMessagesConversation(document, { ...target, model: 'provider-resolved-version' }),
    ).toThrow(/compatibility scope/);
});
