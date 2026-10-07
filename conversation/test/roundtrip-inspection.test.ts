import { describe, expect, it } from 'vitest';
import {
    type ConversationDocument,
    conversationDocumentFromJson,
    conversationDocumentToJson,
    getPendingToolCallIds,
    inspectConversation,
    validateConversationDocument,
} from '../src/index.js';
import {
    emptyDocument,
    generatedAgentTurn,
    importedGeneration,
    RECORDED_AT,
    toolCallBlock,
    toolResultTurn,
    userTurn,
} from './fixtures.js';

function richDocument(): ConversationDocument {
    const document = emptyDocument();
    const user = userTurn('user', 'user-text');
    user.blocks.push({ id: 'user-image', type: 'image', asset_id: 'image', caption: 'exact caption' });
    const call = toolCallBlock('call-block', 'call');
    call.definition_id = 'read-tool';
    const agent = generatedAgentTurn('agent', 'generation', [
        {
            id: 'replay',
            type: 'native_replay',
            adapter: 'adapter',
            protocol: 'protocol',
            compatibility_scope: {
                provider: 'provider',
                protocol: 'protocol',
                adapter_version: 'adapter-v1',
            },
            payload: { constructor: 'opaque', toString: 'signature:\\n[]{}' },
            dependencies: {
                turn_ids: ['user'],
                block_ids: ['user-text'],
                call_ids: [],
                request_ids: [],
            },
        },
        call,
    ]);
    const result = toolResultTurn('tool-result', 'call');
    result.blocks[0].content.push({ id: 'result-json', type: 'json', value: { output: '  exact  ' } });
    document.turns.push(user, agent, result);

    const generation = importedGeneration('generation');
    generation.usage = {
        input_tokens: 2,
        output_tokens: 3,
        total_tokens: 5,
        accounting_provenance: {
            input_tokens: { method: 'reported', accounting_basis: 'provider-v1' },
            output_tokens: { method: 'reported', accounting_basis: 'provider-v1' },
            total_tokens: { method: 'derived', accounting_basis: 'provider-v1' },
        },
        reported_usage: [{ source: 'provider', payload: { input: 2, output: 3 } }],
    };
    document.generations.generation = generation;
    document.assets.image = {
        id: 'image',
        kind: 'image',
        mime_type: 'image/png',
        storage: { type: 'inline_base64', data: 'AAEC/w==' },
        provenance: { type: 'received', source_turn_id: 'user' },
        byte_length: 4,
        content_hash: 'sha256:image',
        media: { width: 1, height: 1 },
        created_at: RECORDED_AT,
    };
    document.tool_definitions['read-tool'] = {
        id: 'read-tool',
        name: 'read',
        version: '1',
        input_schema: {
            type: 'object',
            properties: { path: { type: 'string' } },
            required: ['path'],
            additionalProperties: false,
        },
        result_capabilities: ['text', 'json'],
    };
    document.execution_receipts.receipt = {
        id: 'receipt',
        call_id: 'call',
        executor: 'application',
        status: 'success',
        result_turn_id: 'tool-result',
        result_fingerprint: 'sha256:result',
        recorded_at: RECORDED_AT,
    };
    document.context.entries = [
        { id: 'context-user', type: 'source_turn', turn_id: 'user' },
        { id: 'context-agent', type: 'source_turn', turn_id: 'agent' },
        { id: 'context-tool', type: 'source_turn', turn_id: 'tool-result' },
    ];
    document.context.active_tool_definition_ids = ['read-tool'];
    document.metadata = { constructor: { preserved: true }, toString: 'preserved too' };
    return document;
}

describe('materialized JSON round trip', () => {
    it('preserves media, tool exchanges, opaque replay payloads, and built-in-looking keys', () => {
        const input = richDocument();
        const text = conversationDocumentToJson(input);
        const restored = conversationDocumentFromJson(text);

        expect(restored).toEqual(input);
        expect(Object.hasOwn(restored.metadata ?? {}, 'constructor')).toBe(true);
        expect(restored.assets.image.storage).toEqual({ type: 'inline_base64', data: 'AAEC/w==' });
        const replay = restored.turns[1]?.blocks[0];
        expect(replay?.type).toBe('native_replay');
        if (replay?.type === 'native_replay') {
            expect(replay.payload).toEqual({ constructor: 'opaque', toString: 'signature:\\n[]{}' });
        }
    });

    it('enforces the materialized JSON byte limit in both directions', () => {
        const input = richDocument();
        expect(() => conversationDocumentToJson(input, { json_input_limits: { max_bytes: 128 } })).toThrow();
        const text = JSON.stringify(input);
        expect(() => conversationDocumentFromJson(text, { json_input_limits: { max_bytes: 128 } })).toThrow();
    });
});

describe('inspection and execution receipts', () => {
    it('reports source/context counts and resolves completed calls', () => {
        const document = richDocument();
        expect(inspectConversation(document)).toEqual({
            source_turn_count: 3,
            context_turn_count: 3,
            source_turns_by_kind: { user: 1, agent: 1, tool: 1, program: 0 },
            context_turns_by_kind: { user: 1, agent: 1, tool: 1, program: 0 },
            generation_count: 1,
            pending_tool_call_ids: [],
        });
    });

    it('uses terminal receipts after call/result history deletion and never queues provider calls', () => {
        const document = emptyDocument();
        document.execution_receipts.provider = {
            id: 'provider',
            call_id: 'removed-provider-call',
            executor: 'provider',
            status: 'success',
            result_fingerprint: 'sha256:provider',
            recorded_at: RECORDED_AT,
        };
        document.turns.push(
            generatedAgentTurn('agent', 'generation', [toolCallBlock('provider-call', 'provider-pending', 'provider')]),
        );
        document.generations.generation = importedGeneration('generation');

        expect(validateConversationDocument(document)).toMatchObject({ success: true });
        expect(getPendingToolCallIds(document)).toEqual([]);
    });

    it('lets a provider receipt resolve its retained result after source-call deletion', () => {
        const document = emptyDocument();
        document.turns.push(toolResultTurn('provider-result', 'removed-provider-call'));
        document.execution_receipts.provider = {
            id: 'provider',
            call_id: 'removed-provider-call',
            executor: 'provider',
            status: 'success',
            result_turn_id: 'provider-result',
            result_fingerprint: 'sha256:provider',
            recorded_at: RECORDED_AT,
        };

        expect(validateConversationDocument(document)).toMatchObject({ success: true });
    });

    it('rejects a receipt status that contradicts its retained result', () => {
        const document = richDocument();
        document.execution_receipts.receipt.status = 'error';
        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'TOOL_RECEIPT_MISMATCH')).toBe(true);
    });

    it('rejects a call name that contradicts its pinned definition', () => {
        const document = richDocument();
        document.tool_definitions['read-tool'].name = 'write';
        const result = validateConversationDocument(document);
        expect(result.success).toBe(false);
        expect(result.diagnostics.some((diagnostic) => diagnostic.code === 'TOOL_DEFINITION_MISMATCH')).toBe(true);
    });
});
