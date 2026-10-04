import { describe, expect, it } from 'vitest';
import { createConversationDocument } from '../src/builders.js';
import {
    ConversationOutputProjectionError,
    type ConversationOutputReceipt,
    createAcceptedOutputFragment,
    matchesRetainedAcceptedOutputFragment,
    parseAcceptedOutputFragment,
    validateAcceptedOutputFragment,
} from '../src/output.js';
import {
    cloneSemanticallyValidAcceptedOutputFragment,
    conversationOutputReceiptsEqual,
} from '../src/output-runtime.js';
import { appendConversationRecords } from '../src/runtime.js';
import { externalizeToolCallArguments, prepareToolArgumentExternalization } from '../src/tool-arguments.js';
import type { AgentContentBlock, Asset } from '../src/types.js';

const recordedAt = '2026-09-30T00:00:00.000Z';

function acceptedDocument(options: { assetId?: string; receivedMedia?: boolean } = {}) {
    const initial = createConversationDocument({ id: 'conversation:output', created_at: recordedAt });
    const assetId = options.assetId ?? 'asset:image';
    const asset: Asset = {
        id: assetId,
        kind: 'image' as const,
        mime_type: 'image/png',
        storage: { type: 'inline_base64' as const, data: 'AAAA' },
        provenance: options.receivedMedia
            ? ({ type: 'received' } as const)
            : ({ type: 'generated', generation_id: 'generation:1', source_turn_id: 'turn:agent:1' } as const),
        created_at: recordedAt,
        metadata: { provider_private: 'must-not-leak' },
    };
    const generation = {
        id: 'generation:1',
        record_source: 'executed' as const,
        request_id: 'request:1',
        attempt_id: 'attempt:1',
        provider_response_id: 'provider-response:1',
        purpose: 'interaction',
        requested_model: 'model:requested',
        resolved_model: 'model:resolved',
        provider: 'provider',
        protocol: 'provider.protocol',
        model_options: { private_input_option: true },
        adapter_version: 'adapter:1',
        status: 'completed' as const,
        finish_reason: 'stop',
        timestamps: { recorded_at: recordedAt, completed_at: recordedAt },
        source: { conversation_id: initial.id, revision: initial.revision },
        usage: {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            accounting_provenance: {
                input_tokens: { method: 'reported' as const, accounting_basis: 'provider' },
                output_tokens: { method: 'reported' as const, accounting_basis: 'provider' },
                total_tokens: { method: 'derived' as const, accounting_basis: 'provider' },
            },
            reported_usage: [{ source: 'provider' as const, payload: { opaque: 'provider-payload' } }],
        },
        metadata: { provider_private: 'must-not-leak' },
        request_receipt: {
            id: 'request-receipt:1',
            request_id: 'request:1',
            attempt_id: 'attempt:1',
            source: { conversation_id: initial.id, revision: initial.revision },
            context_fingerprint: 'context-fingerprint',
            tool_set_fingerprint: 'tool-set-fingerprint',
            request_fingerprint: 'request-fingerprint',
            target: {
                provider: 'provider',
                protocol: 'provider.protocol',
                model: 'model:requested',
                adapter_version: 'adapter:1',
            },
            tool_definition_ids: [],
            asset_versions: [],
            item_mappings: [],
            recorded_at: recordedAt,
            metadata: { provider_input: 'must-not-leak' },
        },
    };
    const blocks: AgentContentBlock[] = [
        { id: 'block:text', type: 'text', text: 'answer', format: 'plain' },
        { id: 'block:reasoning', type: 'reasoning', text: 'reasoning', representation: 'summary' },
        { id: 'block:image', type: 'image', asset_id: asset.id },
        {
            id: 'block:tool',
            type: 'tool_call',
            call_id: 'call:1',
            tool_name: 'lookup',
            executor: 'application',
            arguments: { type: 'json', value: { query: 'answer' } },
            native_id: { protocol: 'provider.protocol', scope: 'call', value: 'private-native-id' },
        },
        {
            id: 'block:replay',
            type: 'native_replay',
            adapter: 'adapter',
            protocol: 'provider.protocol',
            compatibility_scope: {
                provider: 'provider',
                protocol: 'provider.protocol',
                model: 'model:requested',
                adapter_version: 'adapter:1',
            },
            payload: { entire_input_prefix: 'must-not-leak' },
            dependencies: { turn_ids: [], block_ids: [], call_ids: [], request_ids: [] },
        },
        {
            id: 'block:extension',
            type: 'extension',
            namespace: 'provider-private',
            version: '1',
            payload: { private_input: 'must-not-leak' },
        },
    ];
    const turn = {
        id: 'turn:agent:1',
        kind: 'agent' as const,
        authority: 'ordinary' as const,
        status: 'completed' as const,
        timestamps: { recorded_at: recordedAt, completed_at: recordedAt },
        model_visibility: 'include' as const,
        provenance: { type: 'generated' as const },
        generation_id: generation.id,
        actor_id: 'agent:private-host-identity',
        exchange_id: 'exchange:private-host-group',
        metadata: { provider_input: 'must-not-leak' },
        blocks,
    };

    return appendConversationRecords(
        initial,
        { turns: [turn], generations: [generation], assets: [asset] },
        {
            expected_revision: initial.revision,
            operation_id: 'response:1',
            payload_fingerprint: 'response-fingerprint',
            recorded_at: recordedAt,
        },
    ).document;
}

describe('accepted conversation output projection', () => {
    it('compares every accepted-output receipt field independently of object key order', () => {
        const receipt = createAcceptedOutputFragment(acceptedDocument(), 'response:1').receipt;
        const reordered = {
            accepted_asset_ids: receipt.accepted_asset_ids,
            accepted_generation_ids: receipt.accepted_generation_ids,
            accepted_turn_ids: receipt.accepted_turn_ids,
            recorded_at: receipt.recorded_at,
            result_revision: receipt.result_revision,
            base_revision: receipt.base_revision,
            conversation_id: receipt.conversation_id,
            id: receipt.id,
        } satisfies ConversationOutputReceipt;
        expect(conversationOutputReceiptsEqual(receipt, reordered)).toBe(true);

        const changed: ConversationOutputReceipt[] = [
            { ...receipt, id: 'response:other' },
            { ...receipt, conversation_id: 'conversation:other' },
            { ...receipt, base_revision: receipt.base_revision + 1 },
            { ...receipt, result_revision: receipt.result_revision + 1 },
            { ...receipt, recorded_at: '2026-09-30T00:00:01.000Z' },
            { ...receipt, accepted_turn_ids: ['turn:other'] },
            { ...receipt, accepted_generation_ids: ['generation:other'] },
            { ...receipt, accepted_asset_ids: ['asset:other'] },
        ];
        expect(changed.every((candidate) => !conversationOutputReceiptsEqual(receipt, candidate))).toBe(true);
    });

    it('omits provider-only usage while preserving its full-document evidence', () => {
        const document = acceptedDocument();
        const generation = document.generations['generation:1'];
        if (generation === undefined) throw new Error('Missing fixture generation');
        generation.usage = {
            reported_usage: [
                {
                    source: 'provider',
                    protocol: 'provider.protocol',
                    accounting_basis: 'provider_seconds',
                    payload: { predict_time: 0.7, total_time: 0.9 },
                },
            ],
        };
        const originalEvidence = structuredClone(generation.usage);

        const fragment = parseAcceptedOutputFragment(
            JSON.parse(JSON.stringify(createAcceptedOutputFragment(document, 'response:1'))),
        );

        expect(fragment.generation.usage).toBeUndefined();
        expect(document.generations['generation:1']?.usage).toEqual(originalEvidence);
    });

    it('retains normalized zero usage while omitting provider payloads', () => {
        const document = acceptedDocument();
        const generation = document.generations['generation:1'];
        if (generation === undefined) throw new Error('Missing fixture generation');
        generation.usage = {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            accounting_provenance: {
                input_tokens: { method: 'reported', accounting_basis: 'provider_tokens' },
                output_tokens: { method: 'reported', accounting_basis: 'provider_tokens' },
                total_tokens: { method: 'derived', accounting_basis: 'provider_tokens' },
            },
            reported_usage: [{ source: 'provider', payload: { opaque: true } }],
        };

        const fragment = createAcceptedOutputFragment(document, 'response:1');

        expect(fragment.generation.usage).toEqual({
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            accounting_provenance: {
                input_tokens: { method: 'reported', accounting_basis: 'provider_tokens' },
                output_tokens: { method: 'reported', accounting_basis: 'provider_tokens' },
                total_tokens: { method: 'derived', accounting_basis: 'provider_tokens' },
            },
        });
        expect(document.generations['generation:1']?.usage?.reported_usage).toEqual([
            { source: 'provider', payload: { opaque: true } },
        ]);
    });

    it('retains typed PCM format metadata through JSON persistence and the accepted output projection', () => {
        const document = acceptedDocument();
        const asset = document.assets['asset:image'];
        if (asset === undefined) throw new Error('Missing fixture asset');
        asset.kind = 'audio';
        asset.mime_type = 'audio/L16;codec=pcm;rate=24000';
        asset.media = {
            container: 'raw',
            codec: 'pcm',
            sample_rate: 24000,
            channels: 1,
            sample_encoding: 'int16',
            byte_order: 'little',
        };
        const turn = document.turns[0];
        if (turn?.kind !== 'agent') throw new Error('Missing fixture agent turn');
        turn.blocks = turn.blocks.map((block) =>
            block.type === 'image' ? { id: block.id, type: 'audio', asset_id: block.asset_id } : block,
        );
        const fragment = parseAcceptedOutputFragment(
            JSON.parse(JSON.stringify(createAcceptedOutputFragment(document, 'response:1'))),
        );
        expect(fragment.assets[asset.id]?.media).toEqual(asset.media);
        expect(fragment.assets[asset.id]).not.toHaveProperty('metadata');
    });

    it('retains canonical semantic identities while excluding replay and opaque metadata', () => {
        const document = acceptedDocument();
        const original = structuredClone(document);
        const fragment = createAcceptedOutputFragment(document, 'response:1');

        expect(fragment.source).toEqual({ conversation_id: document.id, revision: 1 });
        expect(fragment.receipt.accepted_turn_ids).toEqual(['turn:agent:1']);
        expect(fragment.turn.blocks.map((block) => block.type)).toEqual(['text', 'reasoning', 'image', 'tool_call']);
        expect(fragment.turn.blocks.find((block) => block.type === 'tool_call')).not.toHaveProperty('native_id');
        expect(fragment.turn).not.toHaveProperty('metadata');
        expect(fragment.turn).not.toHaveProperty('actor_id');
        expect(fragment.turn).not.toHaveProperty('exchange_id');
        expect(fragment.generation).not.toHaveProperty('request_receipt');
        expect(fragment.generation).not.toHaveProperty('model_options');
        expect(fragment.generation).not.toHaveProperty('metadata');
        expect(fragment.generation.usage).not.toHaveProperty('reported_usage');
        expect(fragment.assets['asset:image']).not.toHaveProperty('metadata');
        expect(fragment.completeness).toEqual({
            history: 'omitted',
            native_replay: 'omitted',
            metadata: 'omitted',
            semantic_content: 'partial',
            omitted_block_ids: ['block:replay', 'block:extension'],
            omitted_asset_ids: [],
        });
        expect(JSON.stringify(fragment)).not.toContain('must-not-leak');
        expect(document).toEqual(original);
    });

    it('omits a media block instead of retaining a dangling or input-backed asset reference', () => {
        const fragment = createAcceptedOutputFragment(acceptedDocument({ receivedMedia: true }), 'response:1');

        expect(fragment.turn.blocks.some((block) => block.id === 'block:image')).toBe(false);
        expect(fragment.assets).toEqual({});
        expect(fragment.completeness.semantic_content).toBe('partial');
        expect(fragment.completeness.omitted_block_ids).toContain('block:image');
        expect(fragment.completeness.omitted_asset_ids).toEqual(['asset:image']);
    });

    it('preserves a generated asset whose valid ID looks like an Object prototype property', () => {
        const fragment = createAcceptedOutputFragment(acceptedDocument({ assetId: 'constructor' }), 'response:1');

        expect(Object.hasOwn(fragment.assets, 'constructor')).toBe(true);
        const assetKey = 'constructor';
        expect(fragment.assets[assetKey]?.id).toBe(assetKey);
        expect(fragment.turn.blocks.find((block) => block.type === 'image')).toMatchObject({
            asset_id: 'constructor',
        });
    });

    it('rejects an operation that does not resolve one accepted response', () => {
        expect(() => createAcceptedOutputFragment(acceptedDocument(), 'response:missing')).toThrow(
            ConversationOutputProjectionError,
        );
    });

    it('rejects projecting an accepted response after a later operation changed the materialized document', async () => {
        const accepted = acceptedDocument();
        const fragment = createAcceptedOutputFragment(accepted, 'response:1');
        await expect(matchesRetainedAcceptedOutputFragment(accepted, fragment)).resolves.toBe(true);
        const prepared = await prepareToolArgumentExternalization(accepted, 'call:1', ['query']);
        const externalized = await externalizeToolCallArguments(accepted, {
            operation_id: 'externalize:1',
            expected_revision: accepted.revision,
            recorded_at: recordedAt,
            call_id: 'call:1',
            input_path: ['query'],
            model_value: { query: '[stored]' },
            exact_arguments_hash: prepared.exact_arguments_hash,
            asset: {
                id: 'asset:tool-input',
                kind: 'text',
                mime_type: 'text/plain',
                storage: {
                    type: 'external',
                    resolver: 'test.artifact',
                    locator: { artifact_path: 'tool-inputs/asset.txt' },
                },
                provenance: { type: 'imported', source: 'host.externalization' },
                byte_length: prepared.byte_length,
                content_hash: prepared.content_hash,
                created_at: recordedAt,
            },
        });

        expect(externalized.document.revision).toBe(2);
        expect(() => createAcceptedOutputFragment(externalized.document, 'response:1')).toThrow(
            ConversationOutputProjectionError,
        );
        await expect(matchesRetainedAcceptedOutputFragment(externalized.document, fragment)).resolves.toBe(true);

        const reordered = structuredClone(externalized.document);
        const visible = reordered.turns[0]?.blocks;
        if (!visible) throw new Error('Fixture requires accepted output blocks');
        const textIndex = visible.findIndex((block) => block.id === 'block:text');
        const reasoningIndex = visible.findIndex((block) => block.id === 'block:reasoning');
        if (textIndex < 0 || reasoningIndex < 0) throw new Error('Fixture requires two visible output blocks');
        const textBlock = visible[textIndex];
        const reasoningBlock = visible[reasoningIndex];
        if (!textBlock || !reasoningBlock) throw new Error('Fixture requires two visible output blocks');
        visible[textIndex] = reasoningBlock;
        visible[reasoningIndex] = textBlock;
        await expect(matchesRetainedAcceptedOutputFragment(reordered, fragment)).resolves.toBe(false);

        const omittedAcceptedText = structuredClone(fragment);
        omittedAcceptedText.turn.blocks = omittedAcceptedText.turn.blocks.filter((block) => block.id !== 'block:text');
        omittedAcceptedText.completeness.omitted_block_ids.push('block:text');
        omittedAcceptedText.completeness.semantic_content = 'partial';
        await expect(matchesRetainedAcceptedOutputFragment(externalized.document, omittedAcceptedText)).resolves.toBe(
            false,
        );

        const changedArguments = structuredClone(fragment);
        const originalCall = changedArguments.turn.blocks.find((block) => block.type === 'tool_call');
        if (
            originalCall?.type !== 'tool_call' ||
            originalCall.arguments.type !== 'json' ||
            originalCall.arguments.value === null ||
            typeof originalCall.arguments.value !== 'object' ||
            Array.isArray(originalCall.arguments.value)
        ) {
            throw new Error('Fixture requires accepted JSON tool arguments');
        }
        originalCall.arguments.value.query = 'changed output';
        await expect(matchesRetainedAcceptedOutputFragment(externalized.document, changedArguments)).resolves.toBe(
            false,
        );

        const changedHash = structuredClone(externalized.document);
        const changedCall = changedHash.turns[0]?.blocks.find((block) => block.type === 'tool_call');
        if (changedCall?.type !== 'tool_call' || changedCall.arguments.type !== 'externalized_json') {
            throw new Error('Fixture requires retained externalized arguments');
        }
        changedCall.arguments.exact_arguments_hash = `sha256:${'0'.repeat(64)}`;
        await expect(matchesRetainedAcceptedOutputFragment(changedHash, fragment)).resolves.toBe(false);

        const changedAsset = structuredClone(externalized.document);
        changedAsset.assets['asset:tool-input'].byte_length = prepared.byte_length + 1;
        await expect(matchesRetainedAcceptedOutputFragment(changedAsset, fragment)).resolves.toBe(false);

        const changedCallIdentity = structuredClone(externalized.document);
        const identityCall = changedCallIdentity.turns[0]?.blocks.find((block) => block.type === 'tool_call');
        if (identityCall?.type !== 'tool_call') throw new Error('Fixture requires retained executed call');
        identityCall.tool_name = 'other_tool';
        await expect(matchesRetainedAcceptedOutputFragment(changedCallIdentity, fragment)).resolves.toBe(false);

        const changedExecutor = structuredClone(externalized.document);
        const executorCall = changedExecutor.turns[0]?.blocks.find((block) => block.type === 'tool_call');
        if (executorCall?.type !== 'tool_call') throw new Error('Fixture requires retained executed call');
        executorCall.executor = 'provider';
        await expect(matchesRetainedAcceptedOutputFragment(changedExecutor, fragment)).resolves.toBe(false);

        const changedReceipt = structuredClone(externalized.document);
        changedReceipt.operation_receipts['externalize:1'].payload_fingerprint = `sha256:${'0'.repeat(64)}`;
        await expect(matchesRetainedAcceptedOutputFragment(changedReceipt, fragment)).resolves.toBe(false);

        const changedText = structuredClone(externalized.document);
        const text = changedText.turns[0]?.blocks.find((block) => block.type === 'text');
        if (text?.type !== 'text') throw new Error('Fixture requires retained accepted text');
        text.text = 'changed output text';
        await expect(matchesRetainedAcceptedOutputFragment(changedText, fragment)).resolves.toBe(false);

        const changedGeneration = structuredClone(externalized.document);
        changedGeneration.generations['generation:1'].requested_model = 'other-model';
        await expect(matchesRetainedAcceptedOutputFragment(changedGeneration, fragment)).resolves.toBe(false);

        const changedAcceptedReceipt = structuredClone(externalized.document);
        changedAcceptedReceipt.operation_receipts['response:1'].recorded_at = '2026-09-30T00:01:00.000Z';
        await expect(matchesRetainedAcceptedOutputFragment(changedAcceptedReceipt, fragment)).resolves.toBe(false);
    });

    it('accepts unchanged, already externalized generated tool arguments on a later head', async () => {
        const accepted = acceptedDocument();
        const prepared = await prepareToolArgumentExternalization(accepted, 'call:1', ['query']);
        const asset: Asset = {
            id: 'asset:accepted-tool-argument',
            kind: 'text',
            mime_type: 'text/plain',
            storage: { type: 'external', resolver: 'test.artifact', locator: { key: 'accepted-argument' } },
            provenance: { type: 'generated', generation_id: 'generation:1', source_turn_id: 'turn:agent:1' },
            content_hash: prepared.content_hash,
            byte_length: prepared.byte_length,
            created_at: recordedAt,
        };
        accepted.assets[asset.id] = asset;
        accepted.operation_receipts['response:1'].accepted_asset_ids = [
            ...(accepted.operation_receipts['response:1'].accepted_asset_ids ?? []),
            asset.id,
        ];
        const call = accepted.turns[0]?.blocks.find((block) => block.type === 'tool_call');
        if (call?.type !== 'tool_call') throw new Error('Fixture requires accepted tool call');
        call.arguments = {
            type: 'externalized_json',
            value: {},
            model_value: { query: '[stored]' },
            exact_arguments_hash: prepared.exact_arguments_hash,
            hydration: [
                { type: 'text_asset', input_path: ['query'], asset_id: asset.id, content_hash: prepared.content_hash },
            ],
        };
        const fragment = createAcceptedOutputFragment(accepted, 'response:1');
        const later = appendConversationRecords(
            accepted,
            { turns: [], generations: [], context_entries: [] },
            {
                expected_revision: accepted.revision,
                operation_id: 'append:later',
                payload_fingerprint: 'later',
                recorded_at: recordedAt,
            },
        ).document;
        await expect(matchesRetainedAcceptedOutputFragment(later, fragment)).resolves.toBe(true);
    });

    it('validates a standalone fragment without rewriting prototype-like asset keys', () => {
        const fragment = createAcceptedOutputFragment(acceptedDocument({ assetId: 'constructor' }), 'response:1');
        const parsed = parseAcceptedOutputFragment(fragment);

        expect(parsed).not.toBe(fragment);
        expect(Object.hasOwn(parsed.assets, 'constructor')).toBe(true);
        expect(validateAcceptedOutputFragment(parsed)).toMatchObject({ success: true });
    });

    it('clones and semantically validates a structurally checked fragment without schema parsing', () => {
        const source = createAcceptedOutputFragment(acceptedDocument({ assetId: 'constructor' }), 'response:1');
        const clone = cloneSemanticallyValidAcceptedOutputFragment(source);

        expect(clone).toEqual(source);
        expect(clone).not.toBe(source);
        expect(Object.hasOwn(clone.assets, 'constructor')).toBe(true);
        source.turn.blocks[0] = { id: 'changed', type: 'text', text: 'changed', format: 'plain' };
        expect(clone.turn.blocks[0]).toMatchObject({ id: 'block:text', text: 'answer' });

        const inconsistent = createAcceptedOutputFragment(acceptedDocument(), 'response:1');
        inconsistent.generation.status = 'failed';
        expect(() => cloneSemanticallyValidAcceptedOutputFragment(inconsistent)).toThrow(
            ConversationOutputProjectionError,
        );
    });

    it.each([
        [
            'source revision',
            (value: ReturnType<typeof createAcceptedOutputFragment>) => {
                value.source.revision = 2;
            },
        ],
        [
            'receipt turn identity',
            (value: ReturnType<typeof createAcceptedOutputFragment>) => {
                value.receipt.accepted_turn_ids[0] = 'turn:other';
            },
        ],
        [
            'generation source',
            (value: ReturnType<typeof createAcceptedOutputFragment>) => {
                value.generation.source.revision = 99;
            },
        ],
        [
            'omitted block overlap',
            (value: ReturnType<typeof createAcceptedOutputFragment>) =>
                void value.completeness.omitted_block_ids.push('block:text'),
        ],
        [
            'asset map identity',
            (value: ReturnType<typeof createAcceptedOutputFragment>) => {
                value.assets['asset:image'].id = 'asset:other';
            },
        ],
        [
            'missing referenced asset',
            (value: ReturnType<typeof createAcceptedOutputFragment>) => void delete value.assets['asset:image'],
        ],
    ])('rejects persisted fragment tampering of %s', (_name, mutate) => {
        const fragment = createAcceptedOutputFragment(acceptedDocument(), 'response:1');
        mutate(fragment);

        expect(() => parseAcceptedOutputFragment(fragment)).toThrow(ConversationOutputProjectionError);
        expect(validateAcceptedOutputFragment(fragment)).toMatchObject({
            success: false,
            error: expect.objectContaining({ code: 'invalid_fragment' }),
        });
    });

    it('rejects private metadata or native replay injected into a persisted fragment', () => {
        const fragment = createAcceptedOutputFragment(acceptedDocument(), 'response:1') as Record<string, unknown>;
        (fragment.turn as Record<string, unknown>).metadata = { private: true };

        expect(() => parseAcceptedOutputFragment(fragment)).toThrow('schema validation failed');
    });

    it('rejects a tool argument hydration reference whose asset does not match', () => {
        const fragment = createAcceptedOutputFragment(acceptedDocument(), 'response:1');
        const toolCall = fragment.turn.blocks.find((block) => block.type === 'tool_call');
        if (toolCall?.type !== 'tool_call') throw new Error('Expected tool call output');
        toolCall.arguments = {
            type: 'externalized_json',
            value: {},
            model_value: { query: '[stored]' },
            exact_arguments_hash: 'arguments-hash',
            hydration: [
                {
                    type: 'text_asset',
                    input_path: ['query'],
                    asset_id: 'asset:image',
                    content_hash: 'different-hash',
                },
            ],
        };

        expect(() => parseAcceptedOutputFragment(fragment)).toThrow(ConversationOutputProjectionError);
    });

    it('reuses generation accounting and turn-status semantic validation', () => {
        const invalidUsage = createAcceptedOutputFragment(acceptedDocument(), 'response:1');
        if (invalidUsage.generation.usage === undefined) throw new Error('Expected usage');
        invalidUsage.generation.usage.total_tokens = 14;
        expect(() => parseAcceptedOutputFragment(invalidUsage)).toThrow(ConversationOutputProjectionError);

        const invalidStatus = createAcceptedOutputFragment(acceptedDocument(), 'response:1');
        invalidStatus.generation.status = 'failed';
        expect(() => parseAcceptedOutputFragment(invalidStatus)).toThrow(ConversationOutputProjectionError);
    });

    it('reuses tool call identity and hydration path semantic validation', () => {
        const duplicateCall = createAcceptedOutputFragment(acceptedDocument(), 'response:1');
        const duplicateSource = duplicateCall.turn.blocks.find((block) => block.type === 'tool_call');
        if (duplicateSource?.type !== 'tool_call') throw new Error('Expected tool call output');
        duplicateCall.turn.blocks.push({ ...duplicateSource, id: 'block:tool:duplicate' });
        expect(() => parseAcceptedOutputFragment(duplicateCall)).toThrow(ConversationOutputProjectionError);

        const overlappingHydration = createAcceptedOutputFragment(acceptedDocument(), 'response:1');
        overlappingHydration.turn.blocks = overlappingHydration.turn.blocks.filter(
            (block) => block.id !== 'block:image',
        );
        overlappingHydration.completeness.omitted_block_ids.push('block:image');
        const priorAsset = overlappingHydration.assets['asset:image'];
        overlappingHydration.assets['asset:image'] = {
            id: priorAsset.id,
            kind: 'text',
            mime_type: 'text/plain',
            storage: {
                type: 'external',
                resolver: 'test.artifact',
                locator: { artifact_path: 'outputs/tool-input.txt' },
            },
            provenance: priorAsset.provenance,
            byte_length: 6,
            content_hash: 'tool-content-hash',
            created_at: priorAsset.created_at,
        };
        const toolCall = overlappingHydration.turn.blocks.find((block) => block.type === 'tool_call');
        if (toolCall?.type !== 'tool_call') throw new Error('Expected tool call output');
        toolCall.arguments = {
            type: 'externalized_json',
            value: {},
            model_value: { query: '[stored]' },
            exact_arguments_hash: 'arguments-hash',
            hydration: [
                {
                    type: 'text_asset',
                    input_path: ['query'],
                    asset_id: 'asset:image',
                    content_hash: 'tool-content-hash',
                },
            ],
        };
        expect(parseAcceptedOutputFragment(overlappingHydration)).toEqual(overlappingHydration);

        toolCall.arguments.hydration.push({
            type: 'text_asset',
            input_path: ['query', 'nested'],
            asset_id: 'asset:image',
            content_hash: 'tool-content-hash',
        });
        expect(() => parseAcceptedOutputFragment(overlappingHydration)).toThrow(ConversationOutputProjectionError);
    });
});
