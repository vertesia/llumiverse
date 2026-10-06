import { describe, expect, it } from 'vitest';
import { createConversationDocument } from '../src/builders.js';
import { canonicalJsonContentString } from '../src/content-integrity.js';
import { appendConversationRecords } from '../src/runtime.js';
import {
    ConversationTranscriptFragmentSchema,
    ConversationTranscriptProjectionInputSchema,
} from '../src/schemas/transcript.js';
import {
    type ConversationTranscriptProjectionInput,
    createConversationTranscriptFragment,
    parseConversationTranscriptFragment,
} from '../src/transcript.js';
import type { Asset, ConversationTurn, ExecutedGeneration, ImportedGeneration } from '../src/types.js';
import { parseConversationDocument } from '../src/validation.js';

const recordedAt = '2026-10-01T00:00:00.000Z';
const source = { conversation_id: 'conversation:transcript', revision: 17 };

function asset(id: string, kind: 'image' | 'audio', provenance: Asset['provenance']): Asset {
    return {
        id,
        kind,
        mime_type: kind === 'image' ? 'image/png' : 'audio/wav',
        storage: { type: 'external', resolver: 'signed-media', locator: { object: `${id}.bin` } },
        provenance,
        byte_length: 8,
        content_hash: `hash:${id}`,
        media: kind === 'image' ? { width: 10, height: 20 } : { duration_seconds: 1 },
        created_at: recordedAt,
        metadata: { private_asset_metadata: 'must-not-leak' },
    };
}

function executedGeneration(id = 'generation:agent'): ExecutedGeneration {
    return {
        id,
        record_source: 'executed',
        request_id: 'private:request',
        attempt_id: 'private:attempt',
        provider_response_id: 'private:provider-response',
        purpose: 'interaction',
        requested_model: 'model:requested',
        resolved_model: 'model:resolved',
        provider: 'provider',
        protocol: 'provider.protocol',
        model_options: { private_model_option: true },
        adapter_version: 'private:adapter-version',
        status: 'completed',
        finish_reason: 'stop',
        timestamps: {
            recorded_at: recordedAt,
            started_at: recordedAt,
            completed_at: recordedAt,
            provider_duration_ms: 42,
        },
        source,
        context_fingerprint: 'private:context-fingerprint',
        tool_set_fingerprint: 'private:tool-set-fingerprint',
        usage: {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            reasoning_tokens: 2,
            accounting_provenance: {
                input_tokens: { method: 'reported', accounting_basis: 'provider' },
                output_tokens: { method: 'reported', accounting_basis: 'provider' },
                total_tokens: { method: 'derived', accounting_basis: 'provider' },
            },
            reported_usage: [{ source: 'provider', payload: { private_provider_usage: true } }],
            cost: { amount: '0.01', currency: 'USD', provenance: 'reported' },
        },
        metadata: { private_generation_metadata: true },
        request_receipt: {
            id: 'private:request-receipt',
            request_id: 'private:request',
            attempt_id: 'private:attempt',
            source,
            context_fingerprint: 'private:context-fingerprint',
            tool_set_fingerprint: 'private:tool-set-fingerprint',
            request_fingerprint: 'private:request-fingerprint',
            target: {
                provider: 'provider',
                protocol: 'provider.protocol',
                model: 'model:requested',
                adapter_version: 'private:adapter-version',
            },
            tool_definition_ids: [],
            asset_versions: [],
            item_mappings: [],
            measurement: {
                input_tokens: 999,
                method: 'exact',
                tokenizer: 'private:tokenizer',
                adapter: 'private:adapter',
                adapter_version: 'private:adapter-version',
                source_fingerprint: 'private:measurement-source',
                target_model: 'model:requested',
                measured_at: recordedAt,
            },
            recorded_at: recordedAt,
            metadata: { private_receipt_metadata: true },
        },
    };
}

function importedGeneration(id: string, withUsage: boolean): ImportedGeneration {
    return {
        id,
        record_source: 'imported',
        purpose: 'historical-import',
        requested_model: 'model:imported',
        provider: 'provider:imported',
        protocol: 'provider.imported',
        status: 'completed',
        finish_reason: 'imported',
        timestamps: { recorded_at: recordedAt },
        source,
        ...(withUsage
            ? { usage: { input_tokens: 3, output_tokens: 4, total_tokens: 7 } }
            : { missing_metadata: ['usage'] }),
    };
}

function commonTurn(id: string) {
    return {
        id,
        authority: 'ordinary' as const,
        status: 'completed' as const,
        timestamps: { recorded_at: recordedAt, completed_at: recordedAt },
        model_visibility: 'include' as const,
        provenance: { type: 'received' as const },
        actor_id: 'private:actor',
        execution_id: 'private:execution',
        exchange_id: 'private:exchange',
        metadata: { private_turn_metadata: 'must-not-leak' },
    };
}

function userTurn(input: ConversationTranscriptProjectionInput): Extract<ConversationTurn, { kind: 'user' }> {
    const turn = input.turns.find(
        (candidate): candidate is Extract<ConversationTurn, { kind: 'user' }> => candidate.kind === 'user',
    );
    if (turn === undefined) throw new Error('Missing user turn fixture');
    return turn;
}

function toolTurn(input: ConversationTranscriptProjectionInput): Extract<ConversationTurn, { kind: 'tool' }> {
    const turn = input.turns.find(
        (candidate): candidate is Extract<ConversationTurn, { kind: 'tool' }> => candidate.kind === 'tool',
    );
    if (turn === undefined) throw new Error('Missing tool turn fixture');
    return turn;
}

function externalReference(id: string, assetId: string) {
    return {
        id,
        type: 'external_reference' as const,
        asset_id: assetId,
        original_type: 'image' as const,
        description: 'unsupported external reference',
        retrieval: { capability: 'retrieve-safe-asset', version: 1, arguments: {} },
    };
}

function transcriptInput(): ConversationTranscriptProjectionInput {
    const turns: ConversationTurn[] = [
        {
            ...commonTurn('turn:user'),
            kind: 'user',
            blocks: [
                { id: 'block:user:text', type: 'text', text: 'question', format: 'plain' },
                { id: 'block:user:json', type: 'json', value: { question: true } },
                { id: 'block:user:image', type: 'image', asset_id: 'asset:user:image', caption: 'input' },
                {
                    id: 'block:user:extension',
                    type: 'extension',
                    namespace: 'private-extension',
                    version: '1',
                    payload: { secret: true },
                },
                {
                    id: 'block:user:external',
                    type: 'external_reference',
                    asset_id: 'asset:private-reference',
                    original_type: 'text',
                    description: 'private external description',
                    content_hash: 'private-reference-hash',
                    retrieval: {
                        capability: 'private-retrieval-capability',
                        version: 1,
                        arguments: { secret: true },
                    },
                },
            ],
        },
        {
            ...commonTurn('turn:agent'),
            kind: 'agent',
            provenance: { type: 'generated' },
            generation_id: 'generation:agent',
            blocks: [
                { id: 'block:agent:text', type: 'text', text: 'answer', format: 'markdown' },
                { id: 'block:agent:reasoning', type: 'reasoning', text: 'summary', representation: 'summary' },
                {
                    id: 'block:agent:tool',
                    type: 'tool_call',
                    call_id: 'call:lookup',
                    tool_name: 'lookup',
                    definition_id: 'private:definition',
                    executor: 'application',
                    native_id: { protocol: 'provider', scope: 'call', value: 'private-native-id' },
                    arguments: {
                        type: 'externalized_json',
                        value: { query: 'exact private hydrated value' },
                        model_value: { query: '[referenced input]' },
                        exact_arguments_hash: 'private-exact-hash',
                        hydration: [
                            {
                                type: 'text_asset',
                                input_path: ['query'],
                                asset_id: 'asset:private-hydration',
                                content_hash: 'private-hydration-hash',
                            },
                        ],
                        invalidated_replay_archives: [
                            {
                                replay_block_id: 'private:replay',
                                asset_id: 'asset:private-archive',
                                content_hash: 'private-archive-hash',
                            },
                        ],
                    },
                },
                {
                    id: 'block:agent:invalid-tool',
                    type: 'tool_call',
                    call_id: 'call:invalid',
                    tool_name: 'invalid_lookup',
                    executor: 'provider',
                    arguments: { type: 'invalid', raw: '{broken', error: 'private parser stack' },
                },
                {
                    id: 'block:agent:native',
                    type: 'native_replay',
                    adapter: 'private-adapter',
                    protocol: 'provider',
                    compatibility_scope: {
                        provider: 'provider',
                        protocol: 'provider',
                        adapter_version: '1',
                    },
                    payload: { signature: 'private-signature' },
                    dependencies: { turn_ids: [], block_ids: [], call_ids: [], request_ids: [] },
                },
            ],
        },
        {
            ...commonTurn('turn:tool'),
            kind: 'tool',
            blocks: [
                {
                    id: 'block:tool:result',
                    type: 'tool_result',
                    call_id: 'call:lookup',
                    status: 'success',
                    native_id: { protocol: 'provider', scope: 'result', value: 'private-result-id' },
                    content: [
                        { id: 'block:tool:text', type: 'text', text: 'result', format: 'plain' },
                        { id: 'block:tool:audio', type: 'audio', asset_id: 'asset:tool:audio' },
                        {
                            id: 'block:tool:extension',
                            type: 'extension',
                            namespace: 'private-tool-extension',
                            version: '1',
                            payload: { secret: true },
                        },
                    ],
                },
            ],
        },
        {
            ...commonTurn('turn:program:internal'),
            kind: 'program',
            blocks: [{ id: 'block:program:internal', type: 'text', text: 'controller secret', format: 'plain' }],
        },
        {
            ...commonTurn('turn:program:explicit-internal'),
            kind: 'program',
            presentation: 'internal',
            blocks: [
                {
                    id: 'block:program:explicit-internal',
                    type: 'text',
                    text: 'explicit controller secret',
                    format: 'plain',
                },
            ],
        },
        {
            ...commonTurn('turn:program:visible'),
            kind: 'program',
            presentation: 'transcript',
            blocks: [{ id: 'block:program:visible', type: 'text', text: 'visible notice', format: 'plain' }],
        },
    ];
    return {
        source,
        turns,
        generations: [executedGeneration()],
        assets: [
            asset('asset:user:image', 'image', { type: 'received', source_turn_id: 'turn:user' }),
            asset('asset:tool:audio', 'audio', {
                type: 'generated',
                generation_id: 'generation:agent',
                source_turn_id: 'turn:tool',
            }),
        ],
        window: {
            gap_before: true,
            gap_after: false,
            omitted_turns: [{ turn_id: 'turn:compacted', reason: 'compacted' }],
            omitted_generations: [],
        },
    };
}

describe('safe canonical transcript projection', () => {
    it('projects every visible turn class and safe content without private authority or replay fields', () => {
        const input = transcriptInput();
        const original = structuredClone(input);
        const fragment = createConversationTranscriptFragment(input);

        expect(input).toEqual(original);
        expect(fragment.source).toEqual(source);
        expect(fragment.turns.map((turn) => turn.kind)).toEqual(['user', 'agent', 'tool', 'program']);
        expect(fragment.turns.map((turn) => turn.id)).not.toContain('turn:program:internal');
        expect(fragment.turns.find((turn) => turn.kind === 'agent')).toMatchObject({
            generation_id: 'generation:agent',
        });
        expect(fragment.generations['generation:agent']).toEqual({
            id: 'generation:agent',
            record_source: 'executed',
            purpose: 'interaction',
            requested_model: 'model:requested',
            resolved_model: 'model:resolved',
            provider: 'provider',
            protocol: 'provider.protocol',
            status: 'completed',
            finish_reason: 'stop',
            timestamps: {
                recorded_at: recordedAt,
                started_at: recordedAt,
                completed_at: recordedAt,
                provider_duration_ms: 42,
            },
            usage: {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
                reasoning_tokens: 2,
                accounting_provenance: {
                    input_tokens: { method: 'reported', accounting_basis: 'provider' },
                    output_tokens: { method: 'reported', accounting_basis: 'provider' },
                    total_tokens: { method: 'derived', accounting_basis: 'provider' },
                },
                cost: { amount: '0.01', currency: 'USD', provenance: 'reported' },
            },
        });
        expect(fragment.completeness.omitted_turns).toEqual([
            { turn_id: 'turn:compacted', reason: 'compacted' },
            { turn_id: 'turn:program:internal', reason: 'internal_program' },
            { turn_id: 'turn:program:explicit-internal', reason: 'internal_program' },
        ]);
        expect(fragment.completeness.omitted_blocks).toEqual(
            expect.arrayContaining([
                { turn_id: 'turn:user', block_id: 'block:user:extension', reason: 'unsupported_extension' },
                {
                    turn_id: 'turn:user',
                    block_id: 'block:user:external',
                    reason: 'unsupported_external_reference',
                },
                { turn_id: 'turn:agent', block_id: 'block:agent:native', reason: 'native_replay' },
                { turn_id: 'turn:tool', block_id: 'block:tool:extension', reason: 'unsupported_extension' },
            ]),
        );

        const serialized = JSON.stringify(fragment);
        for (const forbidden of [
            'private_turn_metadata',
            'private_generation_metadata',
            'private_receipt_metadata',
            'private_provider_usage',
            'private:request',
            'private:attempt',
            'private:provider-response',
            'private:adapter-version',
            'private:measurement-source',
            'private_asset_metadata',
            'private:actor',
            'private:execution',
            'private:exchange',
            'private-retrieval-capability',
            'private external description',
            'private parser stack',
            'private-native-id',
            'private-result-id',
            'private-signature',
            'private-exact-hash',
            'private-hydration-hash',
            'private-archive-hash',
            'exact private hydrated value',
        ]) {
            expect(serialized).not.toContain(forbidden);
        }
        expect(serialized).not.toContain('authority');
        expect(fragment.turns.every((turn) => !Object.hasOwn(turn, 'provenance'))).toBe(true);
        expect(Object.values(fragment.assets).every((item) => !Object.hasOwn(item, 'provenance'))).toBe(true);
        expect(serialized).toContain('[referenced input]');

        const agentTurn = fragment.turns.find((turn) => turn.kind === 'agent');
        const toolCall = agentTurn?.blocks.find((block) => block.type === 'tool_call');
        expect(toolCall).toEqual({
            id: 'block:agent:tool',
            type: 'tool_call',
            call_id: 'call:lookup',
            tool_name: 'lookup',
            executor: 'application',
            arguments: { type: 'json', value: { query: '[referenced input]' } },
        });
        expect(agentTurn?.blocks.find((block) => block.id === 'block:agent:invalid-tool')).toEqual({
            id: 'block:agent:invalid-tool',
            type: 'tool_call',
            call_id: 'call:invalid',
            tool_name: 'invalid_lookup',
            executor: 'provider',
            arguments: { type: 'invalid', raw: '{broken' },
        });
        expect(fragment.assets['asset:user:image']).toMatchObject({ kind: 'image', storage: { type: 'external' } });
        expect(fragment.assets['asset:tool:audio']).toMatchObject({ kind: 'audio', storage: { type: 'external' } });
    });

    it('round-trips strict JSON and rejects unknown transcript fields', () => {
        const input = transcriptInput();
        const persistedInput = JSON.parse(JSON.stringify(input));
        expect(ConversationTranscriptProjectionInputSchema.safeParse(persistedInput).success).toBe(true);
        expect(
            ConversationTranscriptProjectionInputSchema.safeParse({ ...persistedInput, surprise: true }).success,
        ).toBe(false);

        const fragment = createConversationTranscriptFragment(input);
        const json = JSON.stringify(fragment);
        const parsed = parseConversationTranscriptFragment(JSON.parse(json));
        expect(parsed).toEqual(fragment);
        expect(JSON.stringify(parsed)).toBe(json);
        expect(ConversationTranscriptFragmentSchema.safeParse({ ...fragment, provider_history: {} }).success).toBe(
            false,
        );

        const mismatchedAsset = structuredClone(fragment);
        const image = mismatchedAsset.assets['asset:user:image'];
        if (image === undefined) throw new Error('Missing projected image fixture');
        image.kind = 'audio';
        expect(() => parseConversationTranscriptFragment(mismatchedAsset)).toThrow(
            'Transcript fragment asset references are inconsistent',
        );

        const mismatchedRecordId = structuredClone(fragment);
        const record = mismatchedRecordId.assets['asset:user:image'];
        if (record === undefined) throw new Error('Missing projected image fixture');
        record.id = 'asset:different';
        expect(() => parseConversationTranscriptFragment(mismatchedRecordId)).toThrow(
            'Transcript fragment asset record identities are inconsistent',
        );
    });

    it('preserves recorded imported metadata without synthesizing missing imported usage', () => {
        const input = transcriptInput();
        const firstAgentIndex = input.turns.findIndex((turn) => turn.kind === 'agent');
        if (firstAgentIndex < 0) throw new Error('Missing agent fixture');
        input.turns[firstAgentIndex] = {
            ...commonTurn('turn:agent'),
            kind: 'agent',
            provenance: { type: 'imported', source: 'archive' },
            generation_id: 'generation:imported:usage',
            blocks: [{ id: 'block:agent:imported:usage', type: 'text', text: 'historical', format: 'plain' }],
        };
        input.generations = [importedGeneration('generation:imported:usage', true)];
        input.turns.push({
            ...commonTurn('turn:agent:imported-without-usage'),
            kind: 'agent',
            provenance: { type: 'imported', source: 'archive' },
            generation_id: 'generation:imported:no-usage',
            blocks: [{ id: 'block:agent:imported', type: 'text', text: 'historical', format: 'plain' }],
        });
        input.generations.push(importedGeneration('generation:imported:no-usage', false));

        const fragment = createConversationTranscriptFragment(input);
        expect(fragment.generations['generation:imported:usage']).toMatchObject({
            record_source: 'imported',
            requested_model: 'model:imported',
            usage: { input_tokens: 3, output_tokens: 4, total_tokens: 7 },
        });
        expect(fragment.generations['generation:imported:no-usage']).not.toHaveProperty('usage');
        expect(fragment.turns.filter((turn) => turn.kind !== 'agent').every((turn) => !('generation_id' in turn))).toBe(
            true,
        );
    });

    it('shares one bounded generation record across linked agent turns', () => {
        const input = transcriptInput();
        input.turns.push({
            ...commonTurn('turn:agent:continuation'),
            kind: 'agent',
            provenance: { type: 'generated' },
            generation_id: 'generation:agent',
            blocks: [{ id: 'block:agent:continuation', type: 'text', text: 'continued', format: 'plain' }],
        });

        const fragment = createConversationTranscriptFragment(input);
        expect(fragment.turns.filter((turn) => turn.kind === 'agent')).toHaveLength(2);
        expect(Object.keys(fragment.generations)).toEqual(['generation:agent']);
    });

    it('records a referenced generation missing from the bounded source without inventing usage', () => {
        const input = transcriptInput();
        input.generations = [];
        const fragment = createConversationTranscriptFragment(input);

        expect(fragment.generations).toEqual({});
        expect(fragment.turns.find((turn) => turn.kind === 'agent')).toMatchObject({
            generation_id: 'generation:agent',
        });
        expect(fragment.completeness.omitted_generations).toEqual([
            { generation_id: 'generation:agent', reason: 'not_supplied' },
        ]);
        expect(fragment.completeness.semantic_content).toBe('partial');
    });

    it('preserves an explicit retained-generation gap reason without synthesizing a record', () => {
        const input = transcriptInput();
        input.generations = [];
        input.window.omitted_generations = [{ generation_id: 'generation:agent', reason: 'not_retained' }];

        expect(createConversationTranscriptFragment(input).completeness.omitted_generations).toEqual([
            { generation_id: 'generation:agent', reason: 'not_retained' },
        ]);
    });

    it('rejects inconsistent generation records, links, and omissions', () => {
        const fragment = createConversationTranscriptFragment(transcriptInput());
        const mismatchedRecordId = structuredClone(fragment);
        const generation = mismatchedRecordId.generations['generation:agent'];
        if (generation === undefined) throw new Error('Missing projected generation fixture');
        generation.id = 'generation:different';
        expect(() => parseConversationTranscriptFragment(mismatchedRecordId)).toThrow(
            'Transcript fragment generation record identities are inconsistent',
        );

        const missingLink = structuredClone(fragment);
        delete missingLink.generations['generation:agent'];
        expect(() => parseConversationTranscriptFragment(missingLink)).toThrow(
            'Transcript fragment includes inconsistent included and omitted identities',
        );

        const unreferenced = transcriptInput();
        unreferenced.generations.push(importedGeneration('generation:unreferenced', false));
        expect(() => createConversationTranscriptFragment(unreferenced)).toThrow(
            'Transcript source contains an unreferenced generation',
        );

        const invalidOmission = transcriptInput();
        invalidOmission.window.omitted_generations.push({
            generation_id: 'generation:unreferenced',
            reason: 'not_retained',
        });
        expect(() => createConversationTranscriptFragment(invalidOmission)).toThrow(
            'Transcript source contains an unreferenced generation omission',
        );
    });

    it('omits a tool call when model-visible externalized arguments are absent without exposing exact values', () => {
        const input = transcriptInput();
        const unsafe: unknown = structuredClone(input);
        if (typeof unsafe !== 'object' || unsafe === null || !('turns' in unsafe) || !Array.isArray(unsafe.turns)) {
            throw new Error('Missing cloned transcript turns');
        }
        const unsafeAgent = unsafe.turns.find(
            (turn: unknown): turn is Record<string, unknown> =>
                typeof turn === 'object' && turn !== null && 'kind' in turn && turn.kind === 'agent',
        );
        if (!unsafeAgent || !Array.isArray(unsafeAgent.blocks)) throw new Error('Missing cloned agent blocks');
        const unsafeCall = unsafeAgent.blocks.find(
            (block: unknown): block is Record<string, unknown> =>
                typeof block === 'object' && block !== null && 'type' in block && block.type === 'tool_call',
        );
        if (!unsafeCall || typeof unsafeCall.arguments !== 'object' || unsafeCall.arguments === null) {
            throw new Error('Missing cloned externalized arguments');
        }
        delete (unsafeCall.arguments as Record<string, unknown>).model_value;

        const fragment = createConversationTranscriptFragment(unsafe);
        expect(
            fragment.turns
                .find((turn) => turn.kind === 'agent')
                ?.blocks.some((block) => block.id === 'block:agent:tool'),
        ).toBe(false);
        expect(fragment.completeness.omitted_blocks).toContainEqual({
            turn_id: 'turn:agent',
            block_id: 'block:agent:tool',
            reason: 'model_visible_arguments_unavailable',
        });
        expect(JSON.stringify(fragment)).not.toContain('exact private hydrated value');
    });

    it.each(['before', 'after'] as const)(
        'keeps a supported user media asset when an omitted external reference appears %s it',
        (order) => {
            const input = transcriptInput();
            const user = userTurn(input);
            const image = user.blocks.find((block) => block.id === 'block:user:image');
            if (image?.type !== 'image') throw new Error('Missing user image fixture');
            const reference = externalReference('block:user:shared-external', image.asset_id);
            const remaining = user.blocks.filter((block) => block.id !== image.id);
            user.blocks = order === 'before' ? [reference, image, ...remaining] : [image, reference, ...remaining];

            const fragment = createConversationTranscriptFragment(input);
            expect(fragment.assets[image.asset_id]?.kind).toBe('image');
            expect(fragment.completeness.omitted_blocks).toContainEqual({
                turn_id: user.id,
                block_id: reference.id,
                reason: 'unsupported_external_reference',
            });
            expect(fragment.completeness.omitted_assets.some((item) => item.asset_id === image.asset_id)).toBe(false);
        },
    );

    it.each(['before', 'after'] as const)(
        'keeps shared nested tool media when an omitted external reference appears %s it',
        (order) => {
            const input = transcriptInput();
            const tool = toolTurn(input);
            const result = tool.blocks[0];
            const audio = result.content.find((block) => block.id === 'block:tool:audio');
            if (audio?.type !== 'audio') throw new Error('Missing tool audio fixture');
            const reference = externalReference('block:tool:shared-external', audio.asset_id);
            const remaining = result.content.filter((block) => block.id !== audio.id);
            result.content = order === 'before' ? [reference, audio, ...remaining] : [audio, reference, ...remaining];

            const fragment = createConversationTranscriptFragment(input);
            expect(fragment.assets[audio.asset_id]?.kind).toBe('audio');
            expect(fragment.completeness.omitted_blocks).toContainEqual({
                turn_id: tool.id,
                block_id: reference.id,
                reason: 'unsupported_external_reference',
            });
            expect(fragment.completeness.omitted_assets.some((item) => item.asset_id === audio.asset_id)).toBe(false);
        },
    );

    it.each(['before', 'after'] as const)(
        'keeps a correctly typed asset when a mismatched media block appears %s it',
        (order) => {
            const input = transcriptInput();
            const user = userTurn(input);
            const image = user.blocks.find((block) => block.id === 'block:user:image');
            if (image?.type !== 'image') throw new Error('Missing user image fixture');
            const mismatched = { id: 'block:user:mismatched-audio', type: 'audio' as const, asset_id: image.asset_id };
            const remaining = user.blocks.filter((block) => block.id !== image.id);
            user.blocks = order === 'before' ? [mismatched, image, ...remaining] : [image, mismatched, ...remaining];

            const fragment = createConversationTranscriptFragment(input);
            expect(fragment.assets[image.asset_id]?.kind).toBe('image');
            expect(fragment.completeness.omitted_blocks).toContainEqual({
                turn_id: user.id,
                block_id: mismatched.id,
                reason: 'referenced_asset_kind_mismatch',
            });
            expect(fragment.completeness.omitted_assets.some((item) => item.asset_id === image.asset_id)).toBe(false);
        },
    );

    it('does not report a supplied asset as missing when it is referenced only by an unsupported block', () => {
        const input = transcriptInput();
        input.assets.push(asset('asset:private-reference', 'image', { type: 'received' }));
        const fragment = createConversationTranscriptFragment(input);

        expect(fragment.assets).not.toHaveProperty('asset:private-reference');
        expect(fragment.completeness.omitted_blocks).toContainEqual({
            turn_id: 'turn:user',
            block_id: 'block:user:external',
            reason: 'unsupported_external_reference',
        });
        expect(fragment.completeness.omitted_assets.some((item) => item.asset_id === 'asset:private-reference')).toBe(
            false,
        );
    });

    it('records missing media and compaction gaps instead of inventing content', () => {
        const input = transcriptInput();
        input.assets = input.assets.filter((candidate) => candidate.id !== 'asset:tool:audio');
        const fragment = createConversationTranscriptFragment(input);
        const toolTurn = fragment.turns.find((turn) => turn.kind === 'tool');
        expect(toolTurn?.blocks[0]?.content).toEqual([
            { id: 'block:tool:text', type: 'text', text: 'result', format: 'plain' },
        ]);
        expect(fragment.completeness).toMatchObject({
            gap_before: true,
            gap_after: false,
            semantic_content: 'partial',
        });
        expect(fragment.completeness.omitted_assets).toContainEqual({
            asset_id: 'asset:tool:audio',
            reason: 'not_supplied',
        });
    });

    it('bounds the selected generation input and projected generation map', () => {
        const input = transcriptInput();
        input.generations = Array.from({ length: 101 }, (_value, index) =>
            importedGeneration(`generation:bounded:${index}`, false),
        );
        expect(ConversationTranscriptProjectionInputSchema.safeParse(input).success).toBe(false);

        const fragment = createConversationTranscriptFragment(transcriptInput());
        const generation = fragment.generations['generation:agent'];
        if (generation === undefined) throw new Error('Missing projected generation fixture');
        fragment.generations = Object.fromEntries(
            Array.from({ length: 101 }, (_value, index) => {
                const id = `generation:bounded:${index}`;
                return [id, { ...generation, id }];
            }),
        );
        expect(ConversationTranscriptFragmentSchema.safeParse(fragment).success).toBe(false);
    });

    it('enforces the canonical serialized working-set byte limit', () => {
        const input = transcriptInput();
        const user = input.turns.find(
            (turn): turn is Extract<ConversationTurn, { kind: 'user' }> =>
                typeof turn === 'object' && turn !== null && 'kind' in turn && turn.kind === 'user',
        );
        if (user?.blocks[0]?.type !== 'text') throw new Error('Missing user text fixture');
        user.blocks[0].text = 'x'.repeat(512);
        expect(() => createConversationTranscriptFragment(input, { json_input_limits: { max_bytes: 256 } })).toThrow(
            'Transcript projection input failed JSON preflight',
        );
    });

    it('keeps absent program presentation absent across document parsing and canonical fingerprints', () => {
        const initial = createConversationDocument({ id: source.conversation_id, created_at: recordedAt });
        const appended = appendConversationRecords(
            initial,
            {
                turns: [
                    {
                        id: 'turn:program:legacy',
                        kind: 'program',
                        authority: 'ordinary',
                        status: 'completed',
                        timestamps: { recorded_at: recordedAt, completed_at: recordedAt },
                        model_visibility: 'include',
                        provenance: { type: 'received' },
                        blocks: [{ id: 'block:program:legacy', type: 'text', text: 'internal', format: 'plain' }],
                    },
                ],
            },
            {
                expected_revision: initial.revision,
                operation_id: 'operation:program',
                payload_fingerprint: 'program-fingerprint',
                recorded_at: recordedAt,
            },
        ).document;
        const before = canonicalJsonContentString(appended);
        const parsed = parseConversationDocument(JSON.parse(JSON.stringify(appended)));
        const legacy = parsed.turns[0];
        expect(legacy?.kind).toBe('program');
        expect(legacy).not.toHaveProperty('presentation');
        expect(canonicalJsonContentString(parsed)).toBe(before);
    });
});

describe('explicit host-selected transcript metadata', () => {
    it('defaults to omission and copies only selected existing keys without mutating the source', () => {
        const input = transcriptInput();
        userTurn(input).metadata = {
            editing_action: { operation_id: 'edit:one', text: 'actual accepted text' },
            private_proof: { secret: 'hidden' },
        };
        const omitted = createConversationTranscriptFragment(input);
        expect(omitted.turns.every((turn) => turn.metadata === undefined)).toBe(true);
        expect(omitted.completeness.metadata).toBe('omitted');
        const fragment = createConversationTranscriptFragment(input, {
            select_turn_metadata_keys: (turn) => (turn.kind === 'user' ? ['editing_action', 'not-present'] : []),
        });
        const projected = fragment.turns.find((turn) => turn.id === userTurn(input).id);
        if (!projected?.metadata) throw new Error('Expected explicitly selected actual metadata');
        expect(projected.metadata).toEqual({
            editing_action: { operation_id: 'edit:one', text: 'actual accepted text' },
        });
        expect(fragment.completeness.metadata).toBe('partial');
        expect(JSON.stringify(fragment)).not.toContain('private_proof');
        projected.metadata.editing_action = { text: 'changed local projection' };
        expect(userTurn(input).metadata?.editing_action).toEqual({
            operation_id: 'edit:one',
            text: 'actual accepted text',
        });
        expect(parseConversationTranscriptFragment(fragment).completeness.metadata).toBe('partial');
        expect(() =>
            parseConversationTranscriptFragment({
                ...fragment,
                completeness: { ...fragment.completeness, metadata: 'omitted' },
            }),
        ).toThrow('metadata');
    });
    it.each(
        [['__proto__'], ['constructor'], ['prototype'], [''], ['editing_action', 'editing_action']].map((keys) => ({
            keys,
        })),
    )('rejects unsafe or ambiguous selector keys $keys', ({ keys }) => {
        expect(() =>
            createConversationTranscriptFragment(transcriptInput(), { select_turn_metadata_keys: () => keys }),
        ).toThrow('metadata');
    });
    it('gives the selector only frozen canonical turn identity and cannot invent missing metadata', () => {
        const fragment = createConversationTranscriptFragment(transcriptInput(), {
            select_turn_metadata_keys: (turn) => {
                expect(Object.keys(turn).sort()).toEqual(['id', 'kind']);
                expect(Object.isFrozen(turn)).toBe(true);
                return ['not-present'];
            },
        });
        expect(fragment.completeness.metadata).toBe('omitted');
        expect(fragment.turns.every((turn) => turn.metadata === undefined)).toBe(true);
    });
});

describe('bounded external original transcript cues', () => {
    it.each(['text', 'json'] as const)(
        'projects an exact %s tool original without its private read capability',
        (kind) => {
            const input = transcriptInput();
            const original: Asset = {
                id: 'asset:original',
                kind,
                mime_type: kind === 'json' ? 'application/json' : 'text/plain',
                storage: {
                    type: 'external',
                    resolver: 'private-resolver',
                    locator: { private_path: 'not-a-public-grant' },
                },
                provenance: { type: 'received' },
                content_hash: 'sha256:exact-original',
                byte_length: 1_000_000,
                created_at: recordedAt,
            };
            input.assets.push(original);
            const reference = {
                id: 'block:original',
                type: 'external_reference' as const,
                asset_id: original.id,
                original_type: kind,
                content_hash: original.content_hash,
                description: 'Original description '.repeat(100),
                preview: `${'x'.repeat(511)}😀full original must not be emitted`,
                retrieval: {
                    capability: 'read_artifact',
                    version: 1,
                    tool_definition_id: 'private:definition',
                    arguments: { private_path: 'not-a-public-grant' },
                },
            };
            toolTurn(input).blocks[0].content.push(reference);
            const fragment = createConversationTranscriptFragment(input);
            const turn = fragment.turns.find((turn) => turn.kind === 'tool');
            const cue = turn?.blocks[0]?.content.find((block) => block.id === reference.id);
            expect(cue).toEqual({
                id: reference.id,
                type: reference.type,
                asset_id: original.id,
                original_type: kind,
                content_hash: original.content_hash,
                description: reference.description.slice(0, 512),
                preview: 'x'.repeat(511),
            });
            expect(fragment.assets).not.toHaveProperty(original.id);
            expect(fragment.completeness.omitted_blocks.some((block) => block.block_id === reference.id)).toBe(false);
            expect(parseConversationTranscriptFragment(JSON.parse(JSON.stringify(fragment)))).toEqual(fragment);
            for (const secret of [
                'private-resolver',
                'not-a-public-grant',
                'private:definition',
                'full original must not be emitted',
            ])
                expect(JSON.stringify(fragment)).not.toContain(secret);
            for (const mutation of ['missing asset', 'changed hash', 'changed kind'] as const) {
                const forged = structuredClone(input);
                if (mutation === 'missing asset')
                    forged.assets = forged.assets.filter((asset) => asset.id !== original.id);
                else {
                    const asset = forged.assets.find((asset) => asset.id === original.id);
                    if (!asset) throw new Error('Missing exact original fixture');
                    if (mutation === 'changed hash') asset.content_hash = 'different-original';
                    else asset.kind = 'image';
                }
                const rejected = createConversationTranscriptFragment(forged);
                expect(rejected.completeness.omitted_blocks, mutation).toContainEqual({
                    turn_id: toolTurn(input).id,
                    block_id: reference.id,
                    reason: 'unsupported_external_reference',
                });
            }
            if (!cue) throw new Error('Missing projected original cue');
            const unsafeWire = {
                ...fragment,
                turns: fragment.turns.map((turn) =>
                    turn.kind === 'tool'
                        ? {
                              ...turn,
                              blocks: [
                                  {
                                      ...turn.blocks[0],
                                      content: [
                                          ...turn.blocks[0].content,
                                          { ...cue, id: 'block:forged', retrieval: reference.retrieval },
                                      ],
                                  },
                              ],
                          }
                        : turn,
                ),
            };
            expect(ConversationTranscriptFragmentSchema.safeParse(unsafeWire).success).toBe(false);
        },
    );
});
