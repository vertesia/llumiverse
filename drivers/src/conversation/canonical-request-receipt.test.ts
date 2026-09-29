import {
    type ConversationDocument,
    createConversationDocument,
    externalizeToolCallArguments,
    type NativeItemMapping,
    prepareToolArgumentExternalization,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import { createRequestReceipt, type ResolvedConversationRuntimeContext } from './canonical-runtime.js';

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
    it('fingerprints edited selected content even when its IDs and context revision are unchanged', async () => {
        const before = document();
        const after = structuredClone(before);
        const text = after.turns[0]?.blocks[0];
        if (text?.type !== 'text') throw new Error('Expected text fixture');
        text.text = 'different question';
        expect(after.context).toEqual(before.context);
        expect((await receipt(after)).context_fingerprint).not.toBe((await receipt(before)).context_fingerprint);
    });

    it('binds selected asset versions but ignores changes to unselected history', async () => {
        const before = document();
        const after = structuredClone(before);
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
