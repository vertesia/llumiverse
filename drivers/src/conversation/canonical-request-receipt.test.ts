import {
    type AssetStorage,
    type ConversationDocument,
    createConversationDocument,
    deriveConversationId,
    externalizeToolCallArguments,
    inlineAssetContentIntegrity,
    type JsonValue,
    type NativeItemMapping,
    prepareToolArgumentExternalization,
} from '@llumiverse/conversation';
import { describe, expect, it, vi } from 'vitest';
import {
    acceptedCanonicalRequestDocument,
    assertAcceptedCanonicalRequest,
    type CanonicalPreparedState,
    canonicalToolSelectionTargetOptions,
    createExecutedGeneration,
    createRequestReceipt,
    publishCanonicalPreparedRequest,
    type ResolvedConversationRuntimeContext,
} from './canonical-runtime.js';

const now = '2026-09-30T00:00:00.000Z';
const runtime: ResolvedConversationRuntimeContext = {
    conversation_id: 'document',
    request_id: 'request',
    attempt_id: 'attempt',
    input_operation_id: 'input',
    response_operation_id: 'response',
    recorded_at: now,
    purpose: 'conversation',
};
const target = { provider: 'provider', protocol: 'test.protocol', model: 'model', adapter_version: '1' };

function document(): ConversationDocument {
    const doc = createConversationDocument({ id: 'document', created_at: now });
    doc.turns = [
        {
            id: 'user',
            kind: 'user',
            authority: 'ordinary',
            model_visibility: 'include',
            status: 'completed',
            timestamps: { recorded_at: now },
            provenance: { type: 'received' },
            blocks: [
                { id: 'text', type: 'text', format: 'plain', text: 'original question' },
                { id: 'image', type: 'image', asset_id: 'asset' },
            ],
        },
        {
            id: 'hidden',
            kind: 'user',
            authority: 'ordinary',
            model_visibility: 'include',
            status: 'completed',
            timestamps: { recorded_at: now },
            provenance: { type: 'received' },
            blocks: [{ id: 'hidden-text', type: 'text', format: 'plain', text: 'unselected history' }],
        },
    ];
    doc.context.entries = [{ id: 'entry', type: 'source_turn', turn_id: 'user' }];
    doc.assets = {
        asset: {
            id: 'asset',
            kind: 'image',
            mime_type: 'image/png',
            provenance: { type: 'received' },
            created_at: now,
            storage: { type: 'external', resolver: 'url', locator: { url: 'https://example.com/first.png' } },
        },
    };
    return doc;
}

async function receipt(doc: ConversationDocument, mappings: NativeItemMapping[] = []) {
    return createRequestReceipt(doc, runtime, target, { messages: [] }, mappings, []);
}

describe('canonical request receipt binding', () => {
    async function acceptedState(options?: {
        marker?: JsonValue;
        response_selection_policy?: CanonicalPreparedState<unknown>['response_selection_policy'];
    }): Promise<Pick<CanonicalPreparedState<unknown>, 'accepted_response' | 'response_selection_policy' | 'runtime'>> {
        const doc = document();
        const nativePayload = { messages: [] };
        const requestReceipt = await createRequestReceipt(
            doc,
            runtime,
            {
                ...target,
                ...(options?.marker !== undefined ? { options: { canonical_tool_selection: options.marker } } : {}),
            },
            nativePayload,
            [],
            [],
        );
        const generation = await createExecutedGeneration({
            id: 'accepted-generation',
            runtime,
            receipt: requestReceipt,
            ...target,
            requested_model: target.model,
        });
        return {
            runtime,
            ...(options?.response_selection_policy === undefined
                ? {}
                : { response_selection_policy: options.response_selection_policy }),
            accepted_response: {
                generation,
                turn: {
                    id: 'accepted-turn',
                    kind: 'agent',
                    authority: 'ordinary',
                    model_visibility: 'include',
                    status: 'completed',
                    provenance: { type: 'generated' },
                    timestamps: { recorded_at: now },
                    generation_id: generation.id,
                    blocks: [{ id: 'accepted-text', type: 'text', format: 'plain', text: 'answer' }],
                },
            },
        };
    }

    it('binds explicit tool selection while retaining legacy marker-free accepted recovery', async () => {
        const nativePayload = { messages: [] };
        const exact = await acceptedState({
            marker: { mode: 'required', tool_name: 'lookup' },
            response_selection_policy: { mode: 'required', tool_name: 'lookup' },
        });
        await expect(assertAcceptedCanonicalRequest(exact, target, nativePayload)).resolves.toBeUndefined();
        await expect(
            assertAcceptedCanonicalRequest(
                { ...exact, response_selection_policy: { mode: 'none' } },
                target,
                nativePayload,
            ),
        ).rejects.toThrow('incompatible canonical tool-selection policy');
        await expect(
            assertAcceptedCanonicalRequest({ ...exact, response_selection_policy: undefined }, target, nativePayload),
        ).rejects.toThrow('incompatible canonical tool-selection policy');
        const malformed = await acceptedState({
            marker: { mode: 'required', unexpected: true },
            response_selection_policy: { mode: 'required' },
        });
        await expect(assertAcceptedCanonicalRequest(malformed, target, nativePayload)).rejects.toThrow('unknown field');

        const legacy = await acceptedState({ response_selection_policy: { mode: 'none' } });
        await expect(assertAcceptedCanonicalRequest(legacy, target, nativePayload)).resolves.toBeUndefined();
        expect(canonicalToolSelectionTargetOptions({ provider_option: true }, { mode: 'none' })).toEqual({
            provider_option: true,
            canonical_tool_selection: { mode: 'none' },
        });
        expect(() =>
            canonicalToolSelectionTargetOptions({ canonical_tool_selection: { mode: 'auto' } }, { mode: 'required' }),
        ).toThrow('reserve');
    });

    it('checks accepted recovery only after the exact prepared request is durably published', async () => {
        const doc = document();
        const requestReceipt = await receipt(doc);
        const state: CanonicalPreparedState<{ messages: never[] }> = {
            document: doc,
            native_conversation: { messages: [] },
            receipt: requestReceipt,
            runtime,
            generation_id: await deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
            response_turn_id: await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
            tool_definitions: [],
        };
        const calls: string[] = [];
        const prepared = await publishCanonicalPreparedRequest(state, {
            model: 'model',
            on_canonical_request_prepared: async (candidate) => {
                calls.push('persist');
                expect(candidate.record.runtime).toEqual(runtime);
            },
            load_recovered_canonical_output: async (identity) => {
                calls.push('recover');
                expect(identity.prepared_request?.runtime).toEqual(runtime);
                expect(identity.prepared_request?.request_receipt).toEqual(requestReceipt);
                return undefined;
            },
        });

        expect(calls).toEqual(['persist', 'recover']);
        expect(prepared?.record.request_receipt).toEqual(requestReceipt);
    });

    it('fingerprints edited selected content even when its IDs and context revision are unchanged', async () => {
        const before = document();
        const after = structuredClone(before);
        const text = after.turns[0]?.blocks[0];
        if (text?.type !== 'text') throw new Error('Expected text fixture');
        text.text = 'different question';
        expect(after.context).toEqual(before.context);
        expect((await receipt(after)).context_fingerprint).not.toBe((await receipt(before)).context_fingerprint);
    });

    it('binds selected verified asset versions, omits unverified locators, and ignores unselected history', async () => {
        const before = document();
        const after = structuredClone(before);
        expect((await receipt(before)).asset_versions).toEqual([]);
        const verified = structuredClone(before);
        verified.assets.asset.content_hash = 'sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';
        verified.assets.asset.byte_length = 3;
        expect((await receipt(verified)).asset_versions).toEqual([
            {
                asset_id: 'asset',
                content_hash: 'sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
            },
        ]);
        const hidden = after.turns[1]?.blocks[0];
        if (hidden?.type !== 'text') throw new Error('Expected hidden fixture');
        hidden.text = 'history was edited';
        expect((await receipt(after)).context_fingerprint).toBe((await receipt(before)).context_fingerprint);
        after.assets.asset.storage = {
            type: 'external',
            resolver: 'url',
            locator: { url: 'https://example.com/changed.png' },
        };
        expect((await receipt(after)).context_fingerprint).not.toBe((await receipt(before)).context_fingerprint);
    });

    it.each([
        { type: 'inline_base64', data: 'YWJj' },
        { type: 'inline_text', text: 'abc 😀' },
        { type: 'inline_json', value: { z: '😀', a: [1, true, null] } },
    ] satisfies AssetStorage[])(
        'verifies selected $type bytes before preparing or recovering a request',
        async (storage) => {
            const doc = document();
            const actual = await inlineAssetContentIntegrity(storage);
            if (actual === undefined) throw new Error('Expected inline integrity');
            doc.assets.asset = { ...doc.assets.asset, storage, ...actual };
            const bound = await receipt(doc, [{ canonical_id: 'user', kind: 'turn', native_id: 'messages/0' }]);
            expect(bound.asset_versions).toEqual([{ asset_id: 'asset', content_hash: actual.content_hash }]);
            const accepted: NonNullable<CanonicalPreparedState<unknown>['accepted_response']> = {
                generation: await createExecutedGeneration({
                    id: 'accepted-generation',
                    runtime,
                    receipt: bound,
                    ...target,
                    requested_model: target.model,
                }),
                turn: {
                    id: 'accepted-turn',
                    kind: 'agent',
                    authority: 'ordinary',
                    model_visibility: 'include',
                    status: 'completed',
                    provenance: { type: 'generated' },
                    timestamps: { recorded_at: now },
                    generation_id: 'accepted-generation',
                    blocks: [{ id: 'accepted-text', type: 'text', format: 'plain', text: 'answer' }],
                },
            };
            await expect(acceptedCanonicalRequestDocument(doc, accepted)).resolves.toMatchObject({ id: doc.id });
            for (const tamper of [
                { content_hash: `sha256:${'0'.repeat(64)}` },
                { byte_length: actual.byte_length + 1 },
            ]) {
                const changed = structuredClone(doc);
                Object.assign(changed.assets.asset, tamper);
                await expect(receipt(changed)).rejects.toThrow(
                    /Selected inline asset asset .* does not match its bytes/,
                );
                await expect(acceptedCanonicalRequestDocument(changed, accepted)).rejects.toThrow(
                    /Selected inline asset asset .* does not match its bytes/,
                );
            }
        },
    );

    it('does not hash unselected or model-excluded inline content and never fetches external locators', async () => {
        const doc = document();
        doc.assets.asset.storage = { type: 'inline_text', text: 'actual' };
        doc.assets.asset.content_hash = `sha256:${'0'.repeat(64)}`;
        const fetch = vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('Unexpected asset fetch'));
        try {
            doc.context.entries[0].block_ids = ['text'];
            await expect(receipt(doc)).resolves.toMatchObject({ asset_versions: [] });
            delete doc.context.entries[0].block_ids;
            doc.turns[0].model_visibility = 'exclude';
            await expect(receipt(doc)).resolves.toMatchObject({ asset_versions: [] });
            doc.turns[0].model_visibility = 'include';
            doc.assets.asset.storage = {
                type: 'external',
                resolver: 'url',
                locator: { url: 'https://example.test/asset' },
            };
            await expect(receipt(doc)).resolves.toMatchObject({
                asset_versions: [{ asset_id: 'asset', content_hash: doc.assets.asset.content_hash }],
            });
            expect(fetch).not.toHaveBeenCalled();
        } finally {
            fetch.mockRestore();
        }
    });

    it('rejects lossy inline UTF-8 and mismatched byte lengths without requiring a declared hash', async () => {
        const doc = document();
        doc.assets.asset.storage = { type: 'inline_text', text: '\ud800' };
        await expect(receipt(doc)).rejects.toThrow(/unpaired surrogate/);
        doc.assets.asset.storage = { type: 'inline_text', text: '😀' };
        doc.assets.asset.byte_length = 2;
        await expect(receipt(doc)).rejects.toThrow(/byte length does not match/);
    });

    it('rejects absent, misclassified, and unselected mapping IDs before transport', async () => {
        const doc = document();
        for (const mapping of [
            { canonical_id: 'missing', kind: 'block' },
            { canonical_id: 'text', kind: 'turn' },
            { canonical_id: 'hidden-text', kind: 'block' },
        ] as const) {
            await expect(receipt(doc, [{ ...mapping, native_id: 'messages/0' }])).rejects.toThrow(/Request mapping/);
        }
        await expect(
            receipt(doc, [{ canonical_id: 'image', kind: 'block', native_id: 'messages/0/image' }]),
        ).resolves.toMatchObject({ item_mappings: [{ canonical_id: 'image', kind: 'block' }] });
    });

    it('resolves call identities and nested result blocks in a tool exchange', async () => {
        const doc = document();
        doc.turns.push(
            {
                id: 'agent',
                kind: 'agent',
                authority: 'ordinary',
                model_visibility: 'include',
                status: 'completed',
                timestamps: { recorded_at: now },
                provenance: { type: 'inserted' },
                blocks: [
                    {
                        id: 'call-block',
                        type: 'tool_call',
                        call_id: 'call',
                        tool_name: 'lookup',
                        executor: 'application',
                        arguments: { type: 'json', value: {} },
                    },
                ],
            },
            {
                id: 'tool',
                kind: 'tool',
                authority: 'ordinary',
                model_visibility: 'include',
                status: 'completed',
                timestamps: { recorded_at: now },
                provenance: { type: 'received' },
                blocks: [
                    {
                        id: 'result',
                        type: 'tool_result',
                        call_id: 'call',
                        status: 'success',
                        content: [{ id: 'nested', type: 'json', value: { answer: 42 } }],
                    },
                ],
            },
        );
        doc.context.entries.push(
            { id: 'agent-entry', type: 'source_turn', turn_id: 'agent' },
            { id: 'tool-entry', type: 'source_turn', turn_id: 'tool' },
        );
        await expect(
            receipt(doc, [
                { canonical_id: 'call', kind: 'call', native_id: 'native-call' },
                { canonical_id: 'nested', kind: 'block', native_id: 'messages/2/content/0' },
            ]),
        ).resolves.toMatchObject({ item_mappings: [{ canonical_id: 'call' }, { canonical_id: 'nested' }] });
    });

    it('binds selected tool argument hydration assets and ignores excluded calls', async () => {
        const doc = document();
        doc.turns.push({
            id: 'agent-write',
            kind: 'agent',
            authority: 'ordinary',
            model_visibility: 'include',
            status: 'completed',
            timestamps: { recorded_at: now },
            provenance: { type: 'inserted' },
            blocks: [
                {
                    id: 'write-block',
                    type: 'tool_call',
                    call_id: 'write-call',
                    tool_name: 'write_artifact',
                    executor: 'application',
                    arguments: { type: 'json', value: { name: 'large.txt', content: 'exact content' } },
                },
            ],
        });
        doc.context.entries.push({ id: 'agent-write-entry', type: 'source_turn', turn_id: 'agent-write' });
        const prepared = await prepareToolArgumentExternalization(doc, 'write-call', ['content']);
        const externalized = await externalizeToolCallArguments(doc, {
            operation_id: 'externalize-write',
            expected_revision: doc.revision,
            recorded_at: now,
            call_id: 'write-call',
            input_path: ['content'],
            model_value: { name: 'large.txt', content: '[stored]' },
            exact_arguments_hash: prepared.exact_arguments_hash,
            asset: {
                id: 'write-asset',
                kind: 'text',
                mime_type: 'text/plain',
                storage: { type: 'external', resolver: 'test.artifact', locator: { path: 'write.txt' } },
                provenance: { type: 'imported', source: 'test' },
                byte_length: prepared.byte_length,
                content_hash: prepared.content_hash,
                created_at: now,
            },
        });

        await expect(receipt(externalized.document)).resolves.toMatchObject({
            asset_versions: [{ asset_id: 'write-asset', content_hash: prepared.content_hash }],
        });

        const excluded = structuredClone(externalized.document);
        excluded.context.entries = excluded.context.entries.filter((entry) => entry.turn_id !== 'agent-write');
        expect((await receipt(excluded)).asset_versions).toEqual([]);

        const missing = structuredClone(externalized.document);
        delete missing.assets['write-asset'];
        await expect(receipt(missing)).rejects.toThrow('Selected context references missing asset write-asset');
    });
});
