import type { ExecutionOptions } from '@llumiverse/common';
import { parseConversationDocument } from '@llumiverse/conversation';
import type OpenAI from 'openai';
import { describe, expect, it } from 'vitest';
import {
    compileOpenAIResponsesConversation,
    exportLegacyOpenAIResponsesConversation,
    prepareOpenAIResponsesCanonicalState,
} from '../openai/openai-responses-conversation-adapter.js';
import { compileClaudeMessagesConversation } from '../shared/claude-messages-conversation-adapter.js';
import { selectedCanonicalTurns } from './canonical-runtime.js';

const recordedAt = '2026-09-12T00:00:00.000Z';

function options(id: string): ExecutionOptions {
    return {
        model: 'gpt-test',
        tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
        conversation_runtime: {
            conversation_id: id,
            request_id: `${id}:request`,
            attempt_id: `${id}:attempt`,
            input_operation_id: `${id}:input`,
            response_operation_id: `${id}:response`,
            recorded_at: recordedAt,
        },
    };
}

function importHistory(history: OpenAI.Responses.ResponseInputItem[], id: string) {
    return prepareOpenAIResponsesCanonicalState({
        conversation: history,
        prompt: [],
        options: options(id),
        provider: 'openai',
    });
}

const protectedHistory: OpenAI.Responses.ResponseInputItem[] = [
    { role: 'user', content: 'Find the weather in Tokyo.' },
    {
        type: 'reasoning',
        id: 'reasoning-exact',
        summary: [{ type: 'summary_text', text: 'Check the forecast.' }],
        encrypted_content: 'opaque-encrypted-state',
        status: 'completed',
    },
    {
        type: 'function_call',
        id: 'item-exact',
        call_id: 'call-exact',
        name: 'lookup',
        arguments: '{ "city" : "Tokyo", "optional" : null }',
        status: 'completed',
    },
    {
        type: 'function_call_output',
        call_id: 'call-exact',
        output: [
            { type: 'input_text', text: 'Clear skies.' },
            { type: 'input_image', image_url: 'data:image/png;base64,YWJj', detail: 'high' },
            { type: 'input_file', file_data: 'data:application/pdf;base64,YWJj', filename: 'forecast.pdf' },
        ],
    },
    {
        type: 'message',
        id: 'message-exact',
        role: 'assistant',
        status: 'completed',
        content: [
            {
                type: 'output_text',
                text: 'Clear skies in Tokyo.',
                annotations: [
                    {
                        type: 'url_citation',
                        start_index: 0,
                        end_index: 11,
                        title: 'Forecast',
                        url: 'https://example.com/forecast',
                    },
                ],
                logprobs: [],
            },
        ],
    },
];

describe('OpenAI Responses independent canonical conformance', () => {
    it('round trips native identities, raw arguments, rich tool output, and protected replay through JSON', async () => {
        const original = structuredClone(protectedHistory);
        const state = await importHistory(protectedHistory, 'responses-fidelity');
        const persisted = parseConversationDocument(JSON.parse(JSON.stringify(state.document)));

        expect(exportLegacyOpenAIResponsesConversation(persisted)).toEqual(original);
        expect(protectedHistory).toEqual(original);
        const result = persisted.turns.find((turn) => turn.kind === 'tool');
        expect(result?.blocks[0]).toMatchObject({ call_id: 'call-exact', status: 'unknown' });
        expect(
            result?.blocks[0].content.filter((block) => block.type !== 'native_replay').map((block) => block.type),
        ).toEqual(['text', 'image', 'document']);
    });

    it('compiles selected context without deleting unselected source history or mutating the document', async () => {
        const state = await importHistory(
            [
                { role: 'user', content: 'Older question.' },
                { role: 'user', content: 'Current question.' },
            ],
            'responses-context',
        );
        const document = structuredClone(state.document);
        const lastTurn = document.turns.at(-1);
        if (!lastTurn) throw new Error('Expected imported user turn');
        document.context.entries = document.context.entries.filter((entry) => entry.turn_id === lastTurn.id);
        const before = structuredClone(document);

        const projected = compileOpenAIResponsesConversation(document, { provider: 'openai', model: 'gpt-test' });
        expect(JSON.stringify(projected.conversation)).toContain('Current question.');
        expect(JSON.stringify(projected.conversation)).not.toContain('Older question.');
        expect(document).toEqual(before);
        expect(document.turns).toHaveLength(2);
    });

    it('rejects protected replay outside its provider or model scope and on a different protocol', async () => {
        const { document } = await importHistory(protectedHistory, 'responses-protected');

        expect(() => compileOpenAIResponsesConversation(document, { provider: 'xai', model: 'gpt-test' })).toThrow();
        expect(() =>
            compileOpenAIResponsesConversation(document, { provider: 'openai', model: 'other-model' }),
        ).toThrow();
        expect(() =>
            compileClaudeMessagesConversation(document, { provider: 'anthropic', model: 'claude-test' }),
        ).toThrow(/replay/);
    });

    it('permits a model change for ordinary response text without dropping native message fields', async () => {
        const history: OpenAI.Responses.ResponseInputItem[] = [
            { role: 'user', content: 'Hello.' },
            {
                type: 'message',
                id: 'ordinary-message',
                role: 'assistant',
                status: 'completed',
                content: [{ type: 'output_text', text: 'Hello back.', annotations: [], logprobs: [] }],
            },
        ];
        const { document } = await importHistory(history, 'responses-model-change');
        const projected = compileOpenAIResponsesConversation(document, { provider: 'openai', model: 'other-model' });

        expect(projected.conversation).toEqual(history);
    });

    it('does not restore omitted semantic blocks from an opaque replay payload', async () => {
        const state = await importHistory(protectedHistory, 'responses-partial-replay');
        const document = structuredClone(state.document);
        const turn = document.turns.find(
            (candidate) => candidate.kind === 'agent' && candidate.blocks.some((block) => block.type === 'reasoning'),
        );
        const replay = turn?.blocks.find((block) => block.type === 'native_replay');
        const entry = document.context.entries.find((candidate) => candidate.turn_id === turn?.id);
        if (!turn || !replay || !entry) throw new Error('Expected selected protected reasoning turn');
        entry.block_ids = [replay.id];

        expect(() => compileOpenAIResponsesConversation(document, { provider: 'openai', model: 'gpt-test' })).toThrow();
    });

    it('rejects stale native replay after tool-result media changes', async () => {
        const state = await importHistory(protectedHistory, 'responses-changed-media');
        const document = structuredClone(state.document);
        const asset = Object.values(document.assets).find((candidate) => candidate.kind === 'image');
        if (!asset) throw new Error('Expected imported tool-result image');
        asset.storage = { type: 'inline_base64', data: 'ZGlmZmVyZW50' };

        expect(() => compileOpenAIResponsesConversation(document, { provider: 'openai', model: 'gpt-test' })).toThrow();
    });

    it('does not silently replace a changed canonical tool-result association with the native call ID', async () => {
        const state = await importHistory(protectedHistory, 'responses-changed-call');
        const document = structuredClone(state.document);
        const turn = document.turns.find((candidate) => candidate.kind === 'tool');
        if (!turn) throw new Error('Expected imported tool result');
        turn.blocks[0].call_id = 'another-call';

        expect(() => compileOpenAIResponsesConversation(document, { provider: 'openai', model: 'gpt-test' })).toThrow();
    });

    it('allows interrupted history only through an adapter that can replay its exact native state', async () => {
        const history: OpenAI.Responses.ResponseInputItem[] = [
            { role: 'user', content: 'Continue the explanation.' },
            {
                type: 'message',
                id: 'partial-message',
                role: 'assistant',
                status: 'incomplete',
                content: [{ type: 'output_text', text: 'First,', annotations: [], logprobs: [] }],
            },
        ];
        const { document } = await importHistory(history, 'responses-interrupted');

        expect(document.turns.at(-1)?.status).toBe('interrupted');
        expect(() => selectedCanonicalTurns(document)).toThrow(/interrupted/);
        expect(compileOpenAIResponsesConversation(document).conversation).toEqual(history);
    });
});
