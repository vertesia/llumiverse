import type { MessageParam } from '@anthropic-ai/sdk/resources/messages.js';
import type { ExecutionOptions } from '@llumiverse/common';
import {
    type Asset,
    createConversationDocument,
    createToolTurn,
    externalizeToolCallArguments,
    parseConversationDocument,
    prepareToolArgumentExternalization,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import {
    compileBedrockConverseConversation,
    importBedrockConverseConversation,
} from '../bedrock/bedrock-converse-conversation-adapter.js';
import type { OpenAIChatCompletionsPrompt } from '../openai/openai_chat_completions.js';
import {
    compileOpenAIChatCompletionsConversation,
    exportLegacyOpenAIChatCompletionsConversation,
    prepareOpenAIChatCanonicalState,
} from '../openai/openai-chat-conversation-adapter.js';
import {
    compileOpenAIResponsesConversation,
    prepareOpenAIResponsesCanonicalState,
} from '../openai/openai-responses-conversation-adapter.js';
import type { ClaudePrompt } from '../shared/claude-messages.js';
import {
    compileClaudeMessagesConversation,
    exportLegacyClaudeMessagesConversation,
    prepareClaudeCanonicalState,
} from '../shared/claude-messages-conversation-adapter.js';
import {
    compileGeminiConversation,
    prepareGeminiCanonicalState,
} from '../vertexai/models/gemini-conversation-adapter.js';
import { exportLegacyConversation } from './index.js';

const recordedAt = '2026-09-11T00:00:00.000Z';

function executionOptions(model: string, conversationId: string): ExecutionOptions {
    return {
        model,
        conversation_runtime: {
            conversation_id: conversationId,
            request_id: `request:${conversationId}`,
            attempt_id: `attempt:${conversationId}`,
            input_operation_id: `input:${conversationId}`,
            response_operation_id: `response:${conversationId}`,
            recorded_at: recordedAt,
        },
    };
}

async function importOpenAI(history: OpenAIChatCompletionsPrompt, conversationId: string) {
    return prepareOpenAIChatCanonicalState({
        conversation: history,
        prompt: { _is_openai_chat_completions: true, messages: [] },
        options: executionOptions('gpt-test', conversationId),
        provider: 'openai',
    });
}

async function importClaude(history: ClaudePrompt, conversationId: string) {
    return prepareClaudeCanonicalState({
        conversation: history,
        prompt: { messages: [] },
        options: executionOptions('claude-test', conversationId),
        provider: 'anthropic',
    });
}

function durableAsset(input: {
    id: string;
    kind: 'text' | 'document';
    mime_type: 'text/plain' | 'application/json';
    content_hash: string;
    byte_length: number;
}): Asset {
    return {
        ...input,
        storage: {
            type: 'external',
            resolver: 'test.artifact',
            locator: { artifact_path: `tool-inputs/${input.id}` },
        },
        provenance: { type: 'imported', source: 'test' },
        created_at: recordedAt,
    };
}

describe('canonical native adapter conformance', () => {
    it('compiles a checkpoint summary followed by a preserved tool exchange across supported protocols', () => {
        const document = createConversationDocument({ id: 'checkpoint-sequence', created_at: recordedAt });
        document.turns.push(
            {
                id: 'summary',
                kind: 'agent',
                authority: 'ordinary',
                status: 'completed',
                timestamps: { recorded_at: recordedAt },
                provenance: { type: 'received' },
                model_visibility: 'include',
                blocks: [{ id: 'summary-text', type: 'text', text: 'Compacted context.', format: 'plain' }],
            },
            {
                id: 'pending-agent',
                kind: 'agent',
                authority: 'ordinary',
                status: 'completed',
                timestamps: { recorded_at: recordedAt },
                provenance: { type: 'received' },
                model_visibility: 'include',
                blocks: [
                    {
                        id: 'pending-call',
                        type: 'tool_call',
                        call_id: 'call-1',
                        tool_name: 'lookup',
                        executor: 'application',
                        arguments: { type: 'json', value: { city: 'Tokyo' } },
                    },
                ],
            },
            createToolTurn({
                id: 'tool-result',
                authority: 'ordinary',
                status: 'completed',
                timestamps: { recorded_at: recordedAt },
                provenance: { type: 'received' },
                model_visibility: 'include',
                blocks: [
                    {
                        id: 'result-block',
                        type: 'tool_result',
                        call_id: 'call-1',
                        status: 'success',
                        content: [{ id: 'result-text', type: 'text', text: 'sunny', format: 'plain' }],
                    },
                ],
            }),
        );
        document.context.entries = document.turns.map((turn) => ({
            id: `context-${turn.id}`,
            type: 'source_turn' as const,
            turn_id: turn.id,
        }));
        const parsed = parseConversationDocument(document);

        expect(() => compileClaudeMessagesConversation(parsed)).not.toThrow();
        expect(() => compileOpenAIChatCompletionsConversation(parsed)).not.toThrow();
        expect(() => compileGeminiConversation(parsed)).not.toThrow();
        expect(() => compileBedrockConverseConversation(parsed)).not.toThrow();
    });

    it('preserves OpenAI reasoning, raw tool arguments, image detail, and native call identities', async () => {
        const history: OpenAIChatCompletionsPrompt = {
            _is_openai_chat_completions: true,
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'look exactly' },
                        { type: 'image_url', image_url: { url: 'data:image/png;base64,YWJj', detail: 'high' } },
                    ],
                },
                {
                    role: 'assistant',
                    content: 'checking',
                    reasoning_content: 'private reasoning',
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
        };
        const state = await prepareOpenAIChatCanonicalState({
            conversation: history,
            prompt: { _is_openai_chat_completions: true, messages: [] },
            options: {
                ...executionOptions('gpt-test', 'openai-fidelity'),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            },
            provider: 'openai',
        });

        expect(parseConversationDocument(state.document)).toEqual(state.document);
        expect(exportLegacyOpenAIChatCompletionsConversation(state.document).messages).toEqual(history.messages);
        expect(exportLegacyConversation(state.document)).toEqual(
            exportLegacyOpenAIChatCompletionsConversation(state.document),
        );
        const toolResult = state.document.turns.find((turn) => turn.kind === 'tool');
        expect(toolResult?.blocks[0]).toMatchObject({ call_id: 'call-exact', status: 'unknown' });
    });

    it('preserves Claude signed and redacted blocks and marks imported status without evidence unknown', async () => {
        const messages: MessageParam[] = [
            { role: 'user', content: [{ type: 'text', text: 'question' }] },
            {
                role: 'assistant',
                content: [
                    { type: 'thinking', thinking: 'plan', signature: 'signed-plan' },
                    { type: 'redacted_thinking', data: 'opaque-redaction' },
                    { type: 'text', text: 'answer' },
                    { type: 'tool_use', id: 'call-exact', name: 'lookup', input: { city: 'Tokyo' } },
                ],
            },
            {
                role: 'user',
                content: [{ type: 'tool_result', tool_use_id: 'call-exact', content: 'sunny' }],
            },
        ];
        const history = { messages } as ClaudePrompt;
        const state = await prepareClaudeCanonicalState({
            conversation: history,
            prompt: { messages: [] },
            options: {
                ...executionOptions('claude-test', 'claude-fidelity'),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            },
            provider: 'anthropic',
        });

        const projection = exportLegacyClaudeMessagesConversation(state.document);
        expect(projection.messages[1]).toEqual(messages[1]);
        const toolResult = state.document.turns.find((turn) => turn.kind === 'tool');
        expect(toolResult?.blocks[0]).toMatchObject({ call_id: 'call-exact', status: 'unknown' });
    });

    it('transfers plain text and image media across protocols without losing payload bytes', async () => {
        const claude = await importClaude(
            {
                messages: [
                    {
                        role: 'user',
                        content: [
                            { type: 'text', text: 'caption' },
                            {
                                type: 'image',
                                source: { type: 'base64', media_type: 'image/png', data: 'YWJj' },
                            },
                        ],
                    },
                ],
            },
            'cross-media',
        );

        const openAIProjection = compileOpenAIChatCompletionsConversation(claude.document).conversation;
        expect(openAIProjection.messages).toEqual([
            {
                role: 'user',
                content: [
                    { type: 'text', text: 'caption' },
                    { type: 'image_url', image_url: { url: 'data:image/png;base64,YWJj' } },
                ],
            },
        ]);
        expect(compileClaudeMessagesConversation(claude.document).conversation.messages[0]).toEqual({
            role: 'user',
            content: [
                { type: 'text', text: 'caption' },
                { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'YWJj' } },
            ],
        });
    });

    it('uses the same byte digest for inline media across all native adapters', async () => {
        const responses = await prepareOpenAIResponsesCanonicalState({
            conversation: [
                {
                    role: 'user',
                    content: [{ type: 'input_image', image_url: 'data:image/png;base64,YWJj', detail: 'auto' }],
                },
            ],
            prompt: [],
            options: executionOptions('gpt-test', 'responses-inline-integrity'),
            provider: 'openai',
        });
        const responsesGeneratedImage = await prepareOpenAIResponsesCanonicalState({
            conversation: [
                {
                    type: 'image_generation_call',
                    id: 'generated-image',
                    status: 'completed',
                    result: 'YWJj',
                },
            ],
            prompt: [],
            options: executionOptions('gpt-test', 'responses-generated-image-integrity'),
            provider: 'openai',
        });
        const gemini = await prepareGeminiCanonicalState({
            conversation: [{ role: 'user', parts: [{ inlineData: { data: 'YWJj', mimeType: 'image/png' } }] }],
            prompt: { contents: [] },
            options: executionOptions('gemini-test', 'gemini-inline-integrity'),
            provider: 'google',
        });
        const bedrock = await importBedrockConverseConversation(
            {
                messages: [
                    {
                        role: 'user',
                        content: [{ image: { format: 'png', source: { bytes: new Uint8Array([97, 98, 99]) } } }],
                    },
                ],
            },
            { conversation_id: 'bedrock-inline-integrity', recorded_at: recordedAt, provider: 'bedrock' },
        );
        const documents = [
            (
                await importClaude(
                    {
                        messages: [
                            {
                                role: 'user',
                                content: [
                                    {
                                        type: 'image',
                                        source: { type: 'base64', media_type: 'image/png', data: 'YWJj' },
                                    },
                                ],
                            },
                        ],
                    },
                    'claude-inline-integrity',
                )
            ).document,
            (
                await importOpenAI(
                    {
                        _is_openai_chat_completions: true,
                        messages: [
                            {
                                role: 'user',
                                content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,YWJj' } }],
                            },
                        ],
                    },
                    'chat-inline-integrity',
                )
            ).document,
            responses.document,
            responsesGeneratedImage.document,
            gemini.document,
            bedrock,
        ];

        for (const document of documents) {
            expect(Object.values(document.assets)).toEqual([
                expect.objectContaining({
                    content_hash: 'sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
                    byte_length: 3,
                }),
            ]);
        }
    });

    it('does not claim byte integrity for unverified external media locators', async () => {
        const responses = await prepareOpenAIResponsesCanonicalState({
            conversation: [
                {
                    role: 'user',
                    content: [{ type: 'input_image', image_url: 'https://example.com/image.png', detail: 'auto' }],
                },
            ],
            prompt: [],
            options: executionOptions('gpt-test', 'responses-external-integrity'),
            provider: 'openai',
        });
        const gemini = await prepareGeminiCanonicalState({
            conversation: [
                {
                    role: 'user',
                    parts: [{ fileData: { fileUri: 'gs://bucket/image.png', mimeType: 'image/png' } }],
                },
            ],
            prompt: { contents: [] },
            options: executionOptions('gemini-test', 'gemini-external-integrity'),
            provider: 'google',
        });
        const bedrock = await importBedrockConverseConversation(
            {
                messages: [
                    {
                        role: 'user',
                        content: [
                            {
                                image: {
                                    format: 'png',
                                    source: { s3Location: { uri: 's3://bucket/image.png' } },
                                },
                            },
                        ],
                    },
                ],
            },
            { conversation_id: 'bedrock-external-integrity', recorded_at: recordedAt, provider: 'bedrock' },
        );
        const documents = [
            (
                await importClaude(
                    {
                        messages: [
                            {
                                role: 'user',
                                content: [
                                    { type: 'image', source: { type: 'url', url: 'https://example.com/image.png' } },
                                ],
                            },
                        ],
                    },
                    'claude-external-integrity',
                )
            ).document,
            (
                await importOpenAI(
                    {
                        _is_openai_chat_completions: true,
                        messages: [
                            {
                                role: 'user',
                                content: [{ type: 'image_url', image_url: { url: 'https://example.com/image.png' } }],
                            },
                        ],
                    },
                    'chat-external-integrity',
                )
            ).document,
            responses.document,
            gemini.document,
            bedrock,
        ];

        for (const document of documents) {
            const assets = Object.values(document.assets);
            expect(assets).toHaveLength(1);
            expect(assets[0]?.storage.type).toBe('external');
            expect(assets[0]).not.toHaveProperty('content_hash');
            expect(assets[0]).not.toHaveProperty('byte_length');
        }
    });

    it('preserves matching raw Chat arguments but discards stale lexical replay', async () => {
        const raw = '{ "city" : "Paris" }';
        const imported = await importOpenAI(
            {
                _is_openai_chat_completions: true,
                messages: [
                    {
                        role: 'assistant',
                        content: null,
                        tool_calls: [
                            {
                                id: 'call-weather',
                                type: 'function',
                                function: { name: 'weather', arguments: raw },
                            },
                        ],
                    },
                ],
            },
            'chat-raw-arguments',
        );
        const initial = compileOpenAIChatCompletionsConversation(imported.document).conversation.messages[0];
        expect(initial?.tool_calls?.[0]?.function.arguments).toBe(raw);

        const changed = structuredClone(imported.document);
        const call = changed.turns[0]?.blocks.find((block) => block.type === 'tool_call');
        if (call?.type !== 'tool_call' || call.arguments.type !== 'json') throw new Error('Expected JSON tool call');
        call.arguments.value = { city: 'Tokyo' };
        const compiled = compileOpenAIChatCompletionsConversation(parseConversationDocument(changed)).conversation
            .messages[0];
        expect(compiled?.tool_calls?.[0]?.function.arguments).toBe('{"city":"Tokyo"}');

        const protectedDocument = structuredClone(changed);
        const replay = protectedDocument.turns[0]?.blocks.find((block) => block.type === 'native_replay');
        if (replay?.type !== 'native_replay') throw new Error('Expected raw argument replay');
        delete replay.dependency_policy;
        expect(() => compileOpenAIChatCompletionsConversation(parseConversationDocument(protectedDocument))).toThrow(
            'no longer matches canonical call call-weather',
        );
    });

    it('omits foreign discardable lexical replay while projecting the portable tool call', async () => {
        const imported = await importOpenAI(
            {
                _is_openai_chat_completions: true,
                messages: [
                    {
                        role: 'assistant',
                        content: null,
                        tool_calls: [
                            {
                                id: 'call-weather',
                                type: 'function',
                                function: { name: 'weather', arguments: '{ "city" : "Paris" }' },
                            },
                        ],
                    },
                ],
            },
            'portable-chat-tool-call',
        );

        const projections = [
            compileClaudeMessagesConversation(imported.document).conversation.messages,
            compileGeminiConversation(imported.document).conversation.contents,
            compileBedrockConverseConversation(imported.document).conversation.messages,
            compileOpenAIResponsesConversation(imported.document).conversation,
        ];
        for (const projection of projections) {
            expect(JSON.stringify(projection)).toContain('weather');
            expect(JSON.stringify(projection)).not.toContain('openai_chat_assistant_fields');
        }
    });

    it('externalizes a real Responses function call while archiving raw lexical replay', async () => {
        const state = await prepareOpenAIResponsesCanonicalState({
            conversation: [
                {
                    type: 'function_call',
                    id: 'fc-write',
                    call_id: 'call-write',
                    name: 'write_artifact',
                    arguments: '{ "path" : "notes.txt", "content" : "exact retained content" }',
                    status: 'completed',
                },
            ],
            prompt: [],
            options: executionOptions('gpt-test', 'responses-externalization'),
            provider: 'openai',
        });
        const prepared = await prepareToolArgumentExternalization(state.document, 'call-write', ['content']);
        expect(prepared.replay_archives).toHaveLength(1);
        const replayArchives = prepared.replay_archives.map((archive, index) => ({
            replay_block_id: archive.replay_block_id,
            asset: durableAsset({
                id: `responses-replay-${index}`,
                kind: 'document',
                mime_type: 'application/json',
                content_hash: archive.content_hash,
                byte_length: archive.byte_length,
            }),
        }));
        const externalized = await externalizeToolCallArguments(state.document, {
            operation_id: 'externalize-responses-call',
            expected_revision: state.document.revision,
            recorded_at: recordedAt,
            call_id: 'call-write',
            input_path: ['content'],
            model_value: { path: 'notes.txt', content: '[stored externally]' },
            exact_arguments_hash: prepared.exact_arguments_hash,
            asset: durableAsset({
                id: 'responses-content',
                kind: 'text',
                mime_type: 'text/plain',
                content_hash: prepared.content_hash,
                byte_length: prepared.byte_length,
            }),
            replay_archives: replayArchives,
        });

        const responses = compileOpenAIResponsesConversation(externalized.document).conversation;
        expect(responses).toContainEqual(
            expect.objectContaining({
                type: 'function_call',
                call_id: 'call-write',
                name: 'write_artifact',
                arguments: '{"path":"notes.txt","content":"[stored externally]"}',
            }),
        );
        expect(JSON.stringify(responses)).not.toContain('exact retained content');
        expect(compileClaudeMessagesConversation(externalized.document).conversation.messages).toEqual([
            {
                role: 'assistant',
                content: [
                    {
                        type: 'tool_use',
                        id: 'call-write',
                        name: 'write_artifact',
                        input: { path: 'notes.txt', content: '[stored externally]' },
                    },
                ],
            },
        ]);
    });

    it('protects every Responses item in the active exchange when encrypted reasoning is retained', async () => {
        const state = await prepareOpenAIResponsesCanonicalState({
            conversation: [
                { role: 'user', content: 'Write the artifact.' },
                {
                    type: 'function_call',
                    id: 'fc-protected-write',
                    call_id: 'call-protected-write',
                    name: 'write_artifact',
                    arguments: '{"path":"notes.txt","content":"exact retained content"}',
                    status: 'completed',
                },
                {
                    type: 'function_call_output',
                    call_id: 'call-protected-write',
                    output: 'written',
                    _llumiverse_tool_result_status: 'success',
                },
                {
                    id: 'reasoning-write',
                    type: 'reasoning',
                    summary: [{ type: 'summary_text', text: 'Verify the write.' }],
                    encrypted_content: 'encrypted-replay-state',
                    status: 'completed',
                },
            ],
            prompt: [],
            options: executionOptions('gpt-test', 'responses-protected-adjacent'),
            provider: 'openai',
        });
        await expect(
            prepareToolArgumentExternalization(state.document, 'call-protected-write', ['content']),
        ).rejects.toThrow(/Tool call call-protected-write is protected by native replay .* and cannot be externalized/);
        expect(compileOpenAIResponsesConversation(state.document).conversation).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ type: 'reasoning', encrypted_content: 'encrypted-replay-state' }),
            ]),
        );
    });

    it('projects a real unsigned Gemini function call across protocols', async () => {
        const state = await prepareGeminiCanonicalState({
            conversation: [
                {
                    role: 'model',
                    parts: [
                        {
                            functionCall: {
                                id: 'gemini-call',
                                name: 'weather',
                                args: { city: 'Tokyo' },
                            },
                        },
                    ],
                },
            ],
            prompt: { contents: [] },
            options: executionOptions('gemini-test', 'gemini-portable-call'),
            provider: 'google',
        });
        const replay = state.document.turns[0]?.blocks.find((block) => block.type === 'native_replay');
        expect(replay).toMatchObject({ dependency_policy: 'discard_on_dependency_change' });
        expect(
            JSON.stringify(compileOpenAIChatCompletionsConversation(state.document).conversation.messages),
        ).toContain('gemini-call');
        expect(JSON.stringify(compileOpenAIResponsesConversation(state.document).conversation)).toContain(
            'gemini-call',
        );
    });

    it('fails closed for protected foreign replay and visible unsupported media', async () => {
        const openAI = await importOpenAI(
            {
                _is_openai_chat_completions: true,
                messages: [{ role: 'assistant', content: 'answer', reasoning_content: 'provider reasoning' }],
            },
            'foreign-openai-replay',
        );
        expect(() => compileClaudeMessagesConversation(openAI.document)).toThrow(
            'cannot discard protected openai.chat.completions replay',
        );

        const claudeThinking = await importClaude(
            {
                messages: [
                    {
                        role: 'assistant',
                        content: [{ type: 'thinking', thinking: 'plan', signature: 'signature' }],
                    },
                ],
            },
            'foreign-claude-replay',
        );
        expect(() => compileOpenAIChatCompletionsConversation(claudeThinking.document)).toThrow(
            'cannot discard protected anthropic.messages replay',
        );

        const claudeDocument = await importClaude(
            {
                messages: [
                    {
                        role: 'user',
                        content: [
                            {
                                type: 'document',
                                source: { type: 'base64', media_type: 'application/pdf', data: 'YWJj' },
                            },
                        ],
                    },
                ],
            },
            'unsupported-document',
        );
        expect(() => compileOpenAIChatCompletionsConversation(claudeDocument.document)).toThrow(
            'cannot project canonical document block',
        );
    });
});
