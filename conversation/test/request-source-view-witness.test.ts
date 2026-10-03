import { describe, expect, it } from 'vitest';
import {
    AssetSchema,
    appendConversationRecords,
    type ConversationPreparedRequest,
    canonicalJsonContentBytes,
    createConversationDocument,
    createPreparedRequestSourceViewArtifacts,
    deriveConversationId,
    hashContentBytes,
    RequestSourceProjectedTurnSchema,
    RequestSourceViewIndexPageSchema,
    RequestSourceViewManifestSchema,
    resolvePreparedRequestSourceWorkingSet,
} from '../src/index.js';

const AT = '2026-09-30T00:00:00.000Z';
// Captured private v1 layout before accepted-record and original-block-position witnesses existed.
const LEGACY_V1_MANIFEST_TEXT =
    '{"active_tool_definition_ids_hash":"sha256:0000000000000000000000000000000000000000000000000000000000000000","completeness":"selected_content_unverified","context_fingerprint":"sha256:context","context_revision":1,"index_pages":[{"content_hash":"sha256:0000000000000000000000000000000000000000000000000000000000000000","size_bytes":1,"storage_key":"archive/selected-requests/witness/indexes/0000000000000000000000000000000000000000000000000000000000000000.json"}],"record_count":1,"request_fingerprint":"sha256:native-payload","request_receipt_id":"request:witness","segments":[{"content_hash":"sha256:0000000000000000000000000000000000000000000000000000000000000000","size_bytes":1,"storage_key":"archive/selected-requests/witness/segments/0000000000000000000000000000000000000000000000000000000000000000.jsonpart"}],"selected_entry_ids_hash":"sha256:0000000000000000000000000000000000000000000000000000000000000000","source":{"conversation_id":"conversation:witness","revision":1},"version":1}';

async function prepared(withAsset = false): Promise<ConversationPreparedRequest> {
    const assetHash = (await hashContentBytes(Uint8Array.from([0, 1, 2, 3]))).content_hash;
    const document = appendConversationRecords(
        createConversationDocument({ id: 'conversation:witness', created_at: AT }),
        {
            turns: [
                {
                    id: 'turn:witness',
                    kind: 'user',
                    authority: 'ordinary',
                    blocks: [
                        { id: 'left', type: 'text', text: 'unselected left', format: 'plain' },
                        { id: 'center', type: 'text', text: 'selected center', format: 'plain' },
                        { id: 'right', type: 'text', text: 'unselected right', format: 'plain' },
                        ...(withAsset ? [{ id: 'picture', type: 'image' as const, asset_id: 'asset:picture' }] : []),
                    ],
                    status: 'completed',
                    timestamps: { recorded_at: AT },
                    provenance: { type: 'received' },
                    model_visibility: 'include',
                },
            ],
            context_entries: [
                {
                    id: 'entry:witness',
                    type: 'source_turn',
                    turn_id: 'turn:witness',
                    block_ids: withAsset ? ['center', 'picture'] : ['center'],
                },
            ],
            ...(withAsset
                ? {
                      assets: [
                          {
                              id: 'asset:picture',
                              kind: 'image' as const,
                              mime_type: 'image/png',
                              storage: { type: 'inline_base64' as const, data: 'AAECAw==' },
                              provenance: { type: 'received' as const, source_turn_id: 'turn:witness' },
                              byte_length: 4,
                              content_hash: assetHash,
                              created_at: AT,
                          },
                      ],
                  }
                : {}),
        },
        {
            expected_revision: 0,
            operation_id: 'operation:witness',
            payload_fingerprint: 'sha256:input',
            recorded_at: AT,
        },
    ).document;
    const runtime = {
        conversation_id: document.id,
        request_id: 'request:witness',
        attempt_id: 'attempt:witness',
        input_operation_id: 'operation:witness',
        response_operation_id: 'operation:response',
        recorded_at: AT,
        purpose: 'interaction',
    };
    return {
        document,
        record: {
            source: { conversation_id: document.id, revision: document.revision },
            runtime,
            request_receipt: {
                id: await deriveConversationId('request_receipt', runtime.request_id, runtime.attempt_id),
                request_id: runtime.request_id,
                attempt_id: runtime.attempt_id,
                source: { conversation_id: document.id, revision: document.revision },
                source_tail_turn_id: 'turn:witness',
                context_fingerprint: 'sha256:context',
                tool_set_fingerprint: 'sha256:tools',
                request_fingerprint: 'sha256:native-payload',
                target: {
                    provider: 'test',
                    protocol: 'test.generate',
                    model: 'model-a',
                    adapter_version: '1',
                    options: { temperature: 0.5 },
                },
                tool_definition_ids: [],
                asset_versions: withAsset ? [{ asset_id: 'asset:picture', content_hash: assetHash }] : [],
                item_mappings: [{ canonical_id: 'center', native_id: 'parts/0', kind: 'block' }],
                recorded_at: AT,
            },
            generation_id: await deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
            response_turn_id: await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
        },
    };
}

async function archived(withAsset = false) {
    const input = await prepared(withAsset);
    const artifacts = await createPreparedRequestSourceViewArtifacts(input, 'archive/selected-requests/witness');
    const bytes: Map<string, Uint8Array> = new Map(
        [artifacts.manifest, ...artifacts.index_pages, ...artifacts.segments].map((artifact) => [
            artifact.storage_key,
            Uint8Array.from(artifact.bytes),
        ]),
    );
    input.record.request_receipt.source_view = {
        version: 1,
        completeness: 'selected_content_unverified',
        source: input.record.source,
        context_revision: input.document.context.revision,
        manifest_storage_key: artifacts.manifest.storage_key,
        manifest_content_hash: artifacts.manifest.content_hash,
        manifest_size_bytes: artifacts.manifest.bytes.byteLength,
        context_fingerprint: input.record.request_receipt.context_fingerprint,
        request_fingerprint: input.record.request_receipt.request_fingerprint,
    };
    const reader = {
        read: async (key: string) => {
            const value = bytes.get(key);
            if (value === undefined) throw new Error(`Missing ${key}`);
            return Uint8Array.from(value);
        },
    };
    return { input, artifacts, bytes, reader };
}

async function rewriteAndRehashRecord(
    archive: Awaited<ReturnType<typeof archived>>,
    kind: 'source_turn' | 'asset',
    rewrite: (value: unknown) => unknown,
): Promise<void> {
    const { input, artifacts, bytes } = archive;
    const decoder = new TextDecoder();
    const manifest = RequestSourceViewManifestSchema.parse(JSON.parse(decoder.decode(artifacts.manifest.bytes)));
    const index = RequestSourceViewIndexPageSchema.parse(JSON.parse(decoder.decode(artifacts.index_pages[0]?.bytes)));
    const record = index.records.find((item) => item.kind === kind);
    if (record?.parts.length !== 1 || record.parts[0] === undefined) throw new Error('Expected one record part');
    const part = record.parts[0];
    const segment = artifacts.segments[part.segment_index];
    if (segment === undefined) throw new Error('Expected selected source segment');
    const oldBytes = segment.bytes.subarray(part.offset, part.offset + part.length);
    const changedBytes = canonicalJsonContentBytes(rewrite(JSON.parse(decoder.decode(oldBytes))));
    const delta = changedBytes.byteLength - oldBytes.byteLength;
    const changedSegmentBytes = new Uint8Array(segment.bytes.byteLength + delta);
    changedSegmentBytes.set(segment.bytes.subarray(0, part.offset));
    changedSegmentBytes.set(changedBytes, part.offset);
    changedSegmentBytes.set(segment.bytes.subarray(part.offset + part.length), part.offset + changedBytes.byteLength);
    record.size_bytes = changedBytes.byteLength;
    record.content_hash = (await hashContentBytes(changedBytes)).content_hash;
    part.length = changedBytes.byteLength;
    for (const other of index.records) {
        if (other === record) continue;
        for (const later of other.parts) {
            if (later.segment_index === part.segment_index && later.offset > part.offset) later.offset += delta;
        }
    }
    const segmentHash = (await hashContentBytes(changedSegmentBytes)).content_hash;
    const segmentKey = `archive/selected-requests/witness/segments/${segmentHash.slice(7)}.jsonpart`;
    manifest.segments[part.segment_index] = {
        storage_key: segmentKey,
        content_hash: segmentHash,
        size_bytes: changedSegmentBytes.byteLength,
    };
    bytes.delete(segment.storage_key);
    bytes.set(segmentKey, changedSegmentBytes);
    const indexBytes = canonicalJsonContentBytes(index);
    const indexHash = (await hashContentBytes(indexBytes)).content_hash;
    const indexKey = `archive/selected-requests/witness/indexes/${indexHash.slice(7)}.json`;
    manifest.index_pages[0] = { storage_key: indexKey, content_hash: indexHash, size_bytes: indexBytes.byteLength };
    bytes.delete(artifacts.index_pages[0]?.storage_key);
    bytes.set(indexKey, indexBytes);
    const manifestBytes = canonicalJsonContentBytes(manifest);
    const manifestHash = (await hashContentBytes(manifestBytes)).content_hash;
    const manifestKey = `archive/selected-requests/witness/manifests/${manifestHash.slice(7)}.json`;
    bytes.delete(artifacts.manifest.storage_key);
    bytes.set(manifestKey, manifestBytes);
    const reference = input.record.request_receipt.source_view;
    if (reference === undefined) throw new Error('Expected retained manifest reference');
    reference.manifest_storage_key = manifestKey;
    reference.manifest_content_hash = manifestHash;
    reference.manifest_size_bytes = manifestBytes.byteLength;
}

describe('selected source-view request witness', () => {
    it('parses the private v1 manifest for inspection but refuses to infer missing execution witnesses', async () => {
        const bytes = new TextEncoder().encode(LEGACY_V1_MANIFEST_TEXT);
        const parsed = RequestSourceViewManifestSchema.parse(JSON.parse(LEGACY_V1_MANIFEST_TEXT));
        expect(parsed.version).toBe(1);
        const input = await prepared();
        const contentHash = (await hashContentBytes(bytes)).content_hash;
        const key = `archive/selected-requests/witness/manifests/${contentHash.slice(7)}.json`;
        input.record.request_receipt.source_view = {
            version: 1,
            completeness: 'selected_content_unverified',
            source: input.record.source,
            context_revision: input.document.context.revision,
            manifest_storage_key: key,
            manifest_content_hash: contentHash,
            manifest_size_bytes: bytes.byteLength,
            context_fingerprint: input.record.request_receipt.context_fingerprint,
            request_fingerprint: input.record.request_receipt.request_fingerprint,
        };
        const reads: string[] = [];
        await expect(
            resolvePreparedRequestSourceWorkingSet(input.record, {
                read: async (storageKey) => {
                    reads.push(storageKey);
                    if (storageKey !== key) throw new Error('Legacy inspection fetched unrelated content');
                    return Uint8Array.from(bytes);
                },
            }),
        ).rejects.toThrow('lacks prepared-record and original-position witnesses');
        expect(reads).toEqual([key]);
    });
    it('retains original block positions rather than renumbering selected content', async () => {
        const { input, artifacts, reader } = await archived();
        expect(artifacts.working_set.turns[0]?.selected_block_positions).toEqual([1]);
        const restored = await resolvePreparedRequestSourceWorkingSet(input.record, reader);
        expect(restored.turns[0]?.selected_blocks.map((block) => block.id)).toEqual(['center']);
        expect(restored.turns[0]?.selected_block_positions).toEqual([1]);
        expect(restored.turns[0]?.source_block_count).toBe(3);
    });

    it('rejects a changed accepted target or options even when the native payload hash is unchanged', async () => {
        const { input, reader } = await archived();
        const changedTarget = structuredClone(input.record);
        changedTarget.request_receipt.target.model = 'model-b';
        await expect(resolvePreparedRequestSourceWorkingSet(changedTarget, reader)).rejects.toThrow(
            'manifest differs from the accepted prepared request',
        );
        const changedOptions = structuredClone(input.record);
        changedOptions.request_receipt.target.options = { temperature: 0.9 };
        await expect(resolvePreparedRequestSourceWorkingSet(changedOptions, reader)).rejects.toThrow(
            'manifest differs from the accepted prepared request',
        );
    });

    it('binds runtime and generation identities even when the accepted receipt and payload are unchanged', async () => {
        const { input, reader } = await archived();
        for (const field of ['request_id', 'attempt_id'] as const) {
            const changedRequest = structuredClone(input.record);
            changedRequest.runtime[field] = `other:${field}`;
            await expect(resolvePreparedRequestSourceWorkingSet(changedRequest, reader)).rejects.toThrow(
                'manifest differs from the accepted prepared request',
            );
        }
        const changedRuntime = structuredClone(input.record);
        changedRuntime.runtime.input_operation_id = 'operation:other';
        await expect(resolvePreparedRequestSourceWorkingSet(changedRuntime, reader)).rejects.toThrow(
            'manifest differs from the accepted prepared request',
        );
        const changedResponse = structuredClone(input.record);
        changedResponse.runtime.response_operation_id = 'operation:other-response';
        await expect(resolvePreparedRequestSourceWorkingSet(changedResponse, reader)).rejects.toThrow(
            'manifest differs from the accepted prepared request',
        );
        const changedGeneration = structuredClone(input.record);
        changedGeneration.generation_id = 'generation:other';
        await expect(resolvePreparedRequestSourceWorkingSet(changedGeneration, reader)).rejects.toThrow(
            'manifest differs from the accepted prepared request',
        );
        const changedTurn = structuredClone(input.record);
        changedTurn.response_turn_id = 'turn:other';
        await expect(resolvePreparedRequestSourceWorkingSet(changedTurn, reader)).rejects.toThrow(
            'manifest differs from the accepted prepared request',
        );
    });

    it('rejects a rehashed archive whose projected block no longer matches the accepted selection', async () => {
        const { input, artifacts, bytes, reader } = await archived();
        const decoder = new TextDecoder();
        const manifest = RequestSourceViewManifestSchema.parse(JSON.parse(decoder.decode(artifacts.manifest.bytes)));
        const index = RequestSourceViewIndexPageSchema.parse(
            JSON.parse(decoder.decode(artifacts.index_pages[0]?.bytes)),
        );
        const source = index.records.find((record) => record.kind === 'source_turn');
        expect(source?.parts).toHaveLength(1);
        if (source === undefined || source.parts[0] === undefined) throw new Error('Expected one selected turn part');
        const part = source.parts[0];
        const oldSegment = artifacts.segments[part.segment_index];
        if (oldSegment === undefined) throw new Error('Expected selected source segment');
        const oldRecordBytes = oldSegment.bytes.subarray(part.offset, part.offset + part.length);
        const projection = RequestSourceProjectedTurnSchema.parse(JSON.parse(decoder.decode(oldRecordBytes)));
        const selected = projection.selected_blocks[0];
        if (selected === undefined) throw new Error('Expected selected block');
        projection.selected_blocks[0] = { ...selected, id: 'wrongx' };
        const changedRecordBytes = canonicalJsonContentBytes(projection);
        expect(changedRecordBytes.byteLength).toBe(oldRecordBytes.byteLength);
        const changedSegmentBytes = Uint8Array.from(oldSegment.bytes);
        changedSegmentBytes.set(changedRecordBytes, part.offset);
        source.content_hash = (await hashContentBytes(changedRecordBytes)).content_hash;
        const segmentHash = (await hashContentBytes(changedSegmentBytes)).content_hash;
        const segmentKey = `archive/selected-requests/witness/segments/${segmentHash.slice(7)}.jsonpart`;
        manifest.segments[part.segment_index] = {
            storage_key: segmentKey,
            content_hash: segmentHash,
            size_bytes: changedSegmentBytes.byteLength,
        };
        bytes.delete(oldSegment.storage_key);
        bytes.set(segmentKey, changedSegmentBytes);
        const changedIndexBytes = canonicalJsonContentBytes(index);
        const indexHash = (await hashContentBytes(changedIndexBytes)).content_hash;
        const indexKey = `archive/selected-requests/witness/indexes/${indexHash.slice(7)}.json`;
        manifest.index_pages[0] = {
            storage_key: indexKey,
            content_hash: indexHash,
            size_bytes: changedIndexBytes.byteLength,
        };
        bytes.delete(artifacts.index_pages[0]?.storage_key);
        bytes.set(indexKey, changedIndexBytes);
        const changedManifestBytes = canonicalJsonContentBytes(manifest);
        const manifestHash = (await hashContentBytes(changedManifestBytes)).content_hash;
        const manifestKey = `archive/selected-requests/witness/manifests/${manifestHash.slice(7)}.json`;
        bytes.delete(artifacts.manifest.storage_key);
        bytes.set(manifestKey, changedManifestBytes);
        const reference = input.record.request_receipt.source_view;
        if (reference === undefined) throw new Error('Expected retained manifest reference');
        reference.manifest_storage_key = manifestKey;
        reference.manifest_content_hash = manifestHash;
        reference.manifest_size_bytes = changedManifestBytes.byteLength;
        await expect(resolvePreparedRequestSourceWorkingSet(input.record, reader)).rejects.toThrow(
            'block projection differs from its context selection',
        );
    });

    it('rejects duplicate selected block identities at distinct original positions after every archive hash is rewritten', async () => {
        const archive = await archived();
        await rewriteAndRehashRecord(archive, 'source_turn', (value) => {
            const projection = RequestSourceProjectedTurnSchema.parse(value);
            const selected = projection.selected_blocks[0];
            if (selected === undefined) throw new Error('Expected selected block');
            projection.selected_blocks.push(structuredClone(selected));
            projection.selected_block_positions.push(2);
            return projection;
        });
        await expect(resolvePreparedRequestSourceWorkingSet(archive.input.record, archive.reader)).rejects.toThrow(
            'block projection differs from its context selection',
        );
    });

    it('recomputes the complete ordered block-ID hash before trusting a full-turn projection', async () => {
        const archive = await archived();
        await rewriteAndRehashRecord(archive, 'source_turn', (value) => {
            const projection = RequestSourceProjectedTurnSchema.parse(value);
            projection.completeness = 'full_turn';
            projection.source_block_count = projection.selected_blocks.length;
            projection.selected_block_positions = projection.selected_blocks.map((_, index) => index);
            projection.source_block_ids_hash = `sha256:${'0'.repeat(64)}`;
            return projection;
        });
        await expect(resolvePreparedRequestSourceWorkingSet(archive.input.record, archive.reader)).rejects.toThrow(
            'full-turn block order differs',
        );
    });

    it('rejects a rehashed selected asset record whose version differs from the accepted request', async () => {
        const archive = await archived(true);
        expect(
            (await resolvePreparedRequestSourceWorkingSet(archive.input.record, archive.reader)).assets[
                'asset:picture'
            ],
        ).toBeDefined();
        const differentHash = (await hashContentBytes(Uint8Array.from([4, 5, 6, 7]))).content_hash;
        await rewriteAndRehashRecord(archive, 'asset', (value) => {
            const asset = AssetSchema.parse(value);
            return { ...asset, content_hash: differentHash };
        });
        await expect(resolvePreparedRequestSourceWorkingSet(archive.input.record, archive.reader)).rejects.toThrow(
            'asset differs from its accepted version',
        );
    });
});
