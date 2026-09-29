import {
    appendConversationRecords,
    createConversationDocument,
    externalizeToolCallArguments,
    prepareToolArgumentExternalization,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import {
    createCanonicalExecutionResponse,
    FallbackCanonicalExecutionStream,
    legacyCompletionFromCanonicalExecution,
} from './CanonicalExecution.js';
import { MalformedStreamingToolArgumentsError } from './CompletionStream.js';

const RECORDED_AT = '2026-09-30T00:00:00Z';

function acceptedDocument() {
    const initial = createConversationDocument({ id: 'conversation', created_at: RECORDED_AT });
    const requestReceipt = {
        id: 'request-receipt',
        request_id: 'request',
        attempt_id: 'attempt',
        source: { conversation_id: initial.id, revision: initial.revision },
        context_fingerprint: 'sha256:context',
        tool_set_fingerprint: 'sha256:tools',
        request_fingerprint: 'sha256:request',
        target: {
            provider: 'provider',
            protocol: 'provider.protocol',
            model: 'model',
            adapter_version: 'adapter',
        },
        tool_definition_ids: [],
        asset_versions: [],
        item_mappings: [],
        recorded_at: RECORDED_AT,
    };
    const generation = {
        id: 'generation',
        record_source: 'executed' as const,
        request_id: 'request',
        attempt_id: 'attempt',
        purpose: 'interaction',
        requested_model: 'model',
        provider: 'provider',
        protocol: 'provider.protocol',
        adapter_version: 'adapter',
        status: 'completed' as const,
        finish_reason: 'tool_use',
        timestamps: { recorded_at: RECORDED_AT, completed_at: RECORDED_AT },
        source: { conversation_id: initial.id, revision: initial.revision },
        usage: {
            input_tokens: 5,
            output_tokens: 3,
            total_tokens: 8,
            accounting_provenance: {
                input_tokens: { method: 'reported' as const, accounting_basis: 'provider' as const },
                output_tokens: { method: 'reported' as const, accounting_basis: 'provider' as const },
                total_tokens: { method: 'derived' as const, accounting_basis: 'provider' as const },
            },
        },
        request_receipt: requestReceipt,
    };
    const turn = {
        id: 'agent-turn',
        kind: 'agent' as const,
        authority: 'ordinary' as const,
        blocks: [
            { id: 'text', type: 'text' as const, text: 'answer', format: 'plain' as const },
            { id: 'reasoning', type: 'reasoning' as const, text: 'why', representation: 'summary' as const },
            { id: 'json', type: 'json' as const, value: { ok: true } },
            {
                id: 'call-block',
                type: 'tool_call' as const,
                call_id: 'call-1',
                tool_name: 'lookup',
                executor: 'application' as const,
                arguments: { type: 'json' as const, value: { query: 'answer' } },
            },
        ],
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
            expected_revision: 0,
            operation_id: 'response-operation',
            payload_fingerprint: 'sha256:response',
            recorded_at: RECORDED_AT,
        },
    ).document;
}

describe('canonical execution response', () => {
    it('keeps the complete document authoritative and projects legacy completion only at its boundary', () => {
        const document = acceptedDocument();
        const response = createCanonicalExecutionResponse(document, 'response-operation', { execution_time: 12 });

        expect(response.conversation).toEqual(document);
        expect(response.accepted_output.source).toEqual({ conversation_id: document.id, revision: document.revision });
        const legacy = legacyCompletionFromCanonicalExecution(response, { include_reasoning: true });
        expect(legacy.result).toEqual([
            { type: 'text', value: 'answer' },
            { type: 'thoughts', value: 'why' },
            { type: 'json', value: { ok: true } },
        ]);
        expect(legacy.tool_use).toEqual([{ id: 'call-1', tool_name: 'lookup', tool_input: { query: 'answer' } }]);
        expect(legacy.token_usage).toMatchObject({ prompt: 5, result: 3 });
        expect(legacy.finish_reason).toBe('tool_use');
        expect(legacy.conversation).toEqual(document);
        expect(response).not.toHaveProperty('prompt');
    });

    it('recovers the exact accepted output from a verified fragment after later argument externalization', async () => {
        const accepted = acceptedDocument();
        const original = createCanonicalExecutionResponse(accepted, 'response-operation');
        const prepared = await prepareToolArgumentExternalization(accepted, 'call-1', ['query']);
        const externalized = await externalizeToolCallArguments(accepted, {
            operation_id: 'externalize-operation',
            expected_revision: accepted.revision,
            recorded_at: RECORDED_AT,
            call_id: 'call-1',
            input_path: ['query'],
            model_value: { query: '[stored externally]' },
            exact_arguments_hash: prepared.exact_arguments_hash,
            asset: {
                id: 'externalized-argument',
                kind: 'text',
                mime_type: 'text/plain',
                storage: {
                    type: 'external',
                    resolver: 'test.artifact',
                    locator: { artifact_path: 'tool-inputs/call-1.txt' },
                },
                provenance: { type: 'imported', source: 'test' },
                byte_length: prepared.byte_length,
                content_hash: prepared.content_hash,
                created_at: RECORDED_AT,
            },
        });

        expect(() => createCanonicalExecutionResponse(externalized.document, 'response-operation')).toThrow();
        const recovered = createCanonicalExecutionResponse(
            externalized.document,
            'response-operation',
            {},
            original.accepted_output,
        );

        expect(recovered.conversation.revision).toBe(2);
        expect(recovered.accepted_output).toEqual(original.accepted_output);
        expect(legacyCompletionFromCanonicalExecution(recovered).tool_use).toEqual([
            { id: 'call-1', tool_name: 'lookup', tool_input: { query: 'answer' } },
        ]);
    });

    it('fails closed instead of exposing invalid or compact model-only arguments for execution', () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const call = response.accepted_output.turn.blocks.find((block) => block.type === 'tool_call');
        if (call?.type !== 'tool_call') throw new Error('Expected tool call');

        call.arguments = { type: 'invalid', raw: '{' };
        expect(() => legacyCompletionFromCanonicalExecution(response)).toThrow(MalformedStreamingToolArgumentsError);

        call.arguments = {
            type: 'externalized_json',
            value: {},
            model_value: { content: '[stored externally]' },
            exact_arguments_hash: 'sha256:exact',
            hydration: [
                {
                    type: 'text_asset',
                    input_path: ['content'],
                    asset_id: 'asset:tool-input',
                    content_hash: 'sha256:content',
                },
            ],
        };
        expect(() => legacyCompletionFromCanonicalExecution(response)).toThrow(/lossless argument hydration/);
    });

    it('does not expose provider-executed calls as application tool use', () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const call = response.accepted_output.turn.blocks.find((block) => block.type === 'tool_call');
        if (call?.type !== 'tool_call') throw new Error('Expected tool call');
        call.executor = 'provider';

        expect(legacyCompletionFromCanonicalExecution(response).tool_use).toBeUndefined();
    });

    it('projects an accepted canonical structured-output failure onto the legacy error boundary', () => {
        const document = acceptedDocument();
        const generation = document.generations.generation;
        if (generation === undefined) throw new Error('Expected generation');
        generation.status = 'failed';
        generation.metadata = {
            structured_output: {
                status: 'invalid',
                code: 'json_error',
                message: 'The response is not valid JSON',
            },
        };
        const turn = document.turns.find((candidate) => candidate.id === 'agent-turn');
        if (turn === undefined) throw new Error('Expected generated turn');
        turn.status = 'failed';

        expect(
            legacyCompletionFromCanonicalExecution(createCanonicalExecutionResponse(document, 'response-operation'))
                .error,
        ).toEqual({ code: 'json_error', message: 'The response is not valid JSON' });
    });

    it('preserves terminal cutoff reasons when a response also contains a complete tool call', () => {
        const document = acceptedDocument();
        const generation = document.generations.generation;
        if (generation === undefined) throw new Error('Expected generation');
        generation.finish_reason = 'max_tokens';

        expect(
            legacyCompletionFromCanonicalExecution(createCanonicalExecutionResponse(document, 'response-operation'))
                .finish_reason,
        ).toBe('length');
    });

    it('drops only malformed calls at a length cutoff and preserves complete parallel calls', () => {
        const document = acceptedDocument();
        const generation = document.generations.generation;
        if (generation === undefined) throw new Error('Expected generation');
        generation.finish_reason = 'max_tokens';
        const response = createCanonicalExecutionResponse(document, 'response-operation');
        response.accepted_output.turn.blocks.push({
            id: 'partial-call-block',
            type: 'tool_call',
            call_id: 'call-partial',
            tool_name: 'lookup',
            executor: 'application',
            arguments: { type: 'invalid', raw: '{"query":' },
        });

        const legacy = legacyCompletionFromCanonicalExecution(response);
        expect(legacy.finish_reason).toBe('length');
        expect(legacy.tool_use).toEqual([{ id: 'call-1', tool_name: 'lookup', tool_input: { query: 'answer' } }]);
        expect(response.accepted_output.turn.blocks).toContainEqual(
            expect.objectContaining({
                type: 'tool_call',
                call_id: 'call-partial',
                arguments: { type: 'invalid', raw: '{"query":' },
            }),
        );
    });

    it('preserves malformed streamed tool error identity without a cutoff', () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const call = response.accepted_output.turn.blocks.find((block) => block.type === 'tool_call');
        if (call?.type !== 'tool_call') throw new Error('Expected tool call');
        call.arguments = { type: 'invalid', raw: '{"query":' };

        expect(() => legacyCompletionFromCanonicalExecution(response)).toThrow(MalformedStreamingToolArgumentsError);
    });

    it('preserves supported legacy usage dimensions from retained provider evidence', () => {
        const document = acceptedDocument();
        const generation = document.generations.generation;
        if (generation === undefined || generation.usage === undefined) throw new Error('Expected generation usage');
        generation.protocol = 'aws.bedrock.converse';
        generation.usage.reported_usage = [
            {
                source: 'provider',
                protocol: 'aws.bedrock.converse',
                payload: { cacheDetails: [{ ttl: '1h', inputTokens: 2 }] },
            },
        ];
        generation.usage.cost = { amount: '0.125', currency: 'USD', provenance: 'reported' };
        const response = createCanonicalExecutionResponse(document, 'response-operation');

        expect(legacyCompletionFromCanonicalExecution(response).token_usage).toMatchObject({
            prompt_cache_write_1h: 2,
            provider_cost_usd: 0.125,
        });
    });

    it('does not resolve inherited asset record keys', () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        response.accepted_output.turn.blocks.push({ id: 'image', type: 'image', asset_id: 'toString' });

        expect(() => legacyCompletionFromCanonicalExecution(response)).toThrow(/missing image asset toString/);
    });

    it('projects external URL audio with typed media metadata at the legacy boundary', () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        response.accepted_output.assets.audio = {
            id: 'audio',
            kind: 'audio',
            mime_type: 'audio/pcm',
            storage: { type: 'external', resolver: 'url', locator: { url: 'gs://bucket/speech.pcm' } },
            provenance: { type: 'generated', generation_id: 'generation', source_turn_id: 'agent-turn' },
            media: {
                container: 'raw',
                codec: 'pcm',
                sample_rate: 24000,
                channels: 1,
                sample_encoding: 'int16',
                byte_order: 'little',
            },
            created_at: RECORDED_AT,
        };
        response.accepted_output.turn.blocks.push({ id: 'audio-block', type: 'audio', asset_id: 'audio' });

        expect(legacyCompletionFromCanonicalExecution(response).result).toContainEqual({
            type: 'audio',
            value: 'gs://bucket/speech.pcm',
            mime_type: 'audio/pcm',
            container: 'raw',
            codec: 'pcm',
            sample_rate: 24000,
            channels: 1,
            sample_encoding: 'int16',
            byte_order: 'little',
        });
    });

    it('keeps reasoning authoritative while hiding it from fallback previews unless explicitly requested', async () => {
        const response = createCanonicalExecutionResponse(acceptedDocument(), 'response-operation');
        const hidden = new FallbackCanonicalExecutionStream(async () => response);
        let hiddenPreview = '';
        for await (const chunk of hidden) hiddenPreview += chunk;
        expect(hiddenPreview).toBe('answer{"ok":true}');
        expect(hidden.completion?.accepted_output.turn.blocks.some((block) => block.type === 'reasoning')).toBe(true);

        const visible = new FallbackCanonicalExecutionStream(async () => response, true);
        let visiblePreview = '';
        for await (const chunk of visible) visiblePreview += chunk;
        expect(visiblePreview).toBe('answerwhy{"ok":true}');
    });
});
