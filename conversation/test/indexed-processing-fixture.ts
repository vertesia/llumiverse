import { hashContentBytes, hashUtf8Content } from '../src/content-integrity.js';
import { fingerprintJson } from '../src/identity.js';
import {
    type IndexedConversationRecordStore,
    loadIndexedProcessingSelectedContext,
    type StagedIndexedConversationRoot,
    stageIndexedConversationSnapshot,
    stageIndexedProcessingPhase,
    stageIndexedRecordBatch,
} from '../src/indexed-conversation.js';
import { resolveIndexedProcessingTextInput } from '../src/indexed-processing-working-set.js';
import { setProcessingPolicy } from '../src/processing.js';
import { appendConversationRecordsWithProcessing } from '../src/runtime.js';
import {
    type IndexedProcessingClaimWorkspace,
    IndexedProcessingClaimWorkspaceSchema,
} from '../src/schemas/indexed-processing.js';
import type { ConversationDocument, ConversationRecordBatch } from '../src/types.js';
import { emptyDocument, RECORDED_AT, userTurn } from './fixtures.js';

/** Real canonical policy/append/archive and indexed phase records; no constructed full-document
 * facade or caller authority. A host test must still authenticate its run/namespace/custody.
 */
export interface IndexedTextClaimFixture {
    store: IndexedConversationRecordStore;
    staged: StagedIndexedConversationRoot;
    workspace: IndexedProcessingClaimWorkspace;
    document: ConversationDocument;
    reads: string[];
    records: Map<string, Uint8Array>;
    pages: Map<string, Uint8Array>;
}
export async function indexedTextClaimFixture(
    coldCount = 0,
    content: 'text' | 'mixed' = 'text',
    textBlockCount = 1,
): Promise<IndexedTextClaimFixture> {
    const pages = new Map<string, Uint8Array>();
    const records = new Map<string, Uint8Array>();
    const reads: string[] = [];
    const store: IndexedConversationRecordStore = {
        async read(ref) {
            const bytes = pages.get(ref.content_hash);
            if (!bytes) throw new Error('Fixture page absent');
            return Uint8Array.from(bytes);
        },
        async write(bytes, ref) {
            pages.set(ref.content_hash, Uint8Array.from(bytes));
        },
        async readRecord(ref) {
            reads.push(`${ref.kind}:${ref.id}`);
            const bytes = records.get(ref.content_hash);
            if (!bytes) throw new Error('Fixture record absent');
            return Uint8Array.from(bytes);
        },
        async writeRecord(ref, bytes) {
            if ((await hashContentBytes(bytes)).content_hash !== ref.content_hash)
                throw new Error('Fixture write hash mismatch');
            records.set(ref.content_hash, Uint8Array.from(bytes));
        },
        async assertExternalAssetIntegrity(asset) {
            const integrity = await hashUtf8Content('original');
            if (asset.content_hash !== integrity.content_hash || asset.byte_length !== integrity.byte_length)
                throw new Error('Fixture archive custody mismatch');
        },
    };
    const configuration = {
        id: 'externalize-text',
        version: '1',
        scope: 'on_append' as const,
        config: {},
        required: true,
        failure_behavior: 'block' as const,
    };
    const enabled = (
        await setProcessingPolicy(emptyDocument('conversation:indexed-workspace'), {
            operation_id: 'operation:policy',
            expected_revision: 0,
            recorded_at: RECORDED_AT,
            enabled: true,
            processors: [configuration],
        })
    ).document;
    const turn = userTurn('turn:input', 'block:input');
    turn.blocks = [
        ...Array.from({ length: textBlockCount }, (_, index) => ({
            id: textBlockCount === 1 ? 'block:input' : `block:input:${index}`,
            type: 'text' as const,
            text: 'original',
            format: 'plain' as const,
        })),
        ...(content === 'mixed'
            ? [{ id: 'block:json', type: 'json' as const, value: { preserve: [null, 3, false] } }]
            : []),
    ];
    const inputBatch: ConversationRecordBatch = {
        turns: [turn],
        context_entries: [{ id: 'entry:input', type: 'source_turn', turn_id: turn.id }],
        tool_definitions: [
            {
                id: 'definition:read',
                name: 'read_artifact',
                version: 'content-version',
                input_schema: {
                    type: 'object',
                    properties: { path: { type: 'string' }, asset_id: { type: 'string' } },
                    required: ['path'],
                    additionalProperties: false,
                },
            },
        ],
        active_tool_definition_ids: ['definition:read'],
    };
    const inputOptions = {
        expected_revision: enabled.revision,
        operation_id: 'operation:input',
        payload_fingerprint: await fingerprintJson({ original: true }),
        recorded_at: RECORDED_AT,
    };
    const accepted = (await appendConversationRecordsWithProcessing(enabled, inputBatch, inputOptions)).document;
    const job = Object.values(accepted.processing.jobs ?? {})[0];
    if (!job) throw new Error('Fixture input has no real accepted job');
    const integrity = await hashUtf8Content('original');
    const asset = {
        id: 'asset:archive',
        kind: 'text' as const,
        mime_type: 'text/plain',
        storage: { type: 'external' as const, resolver: 'fixture.archive', locator: { key: 'original' } },
        provenance: { type: 'received' as const },
        content_hash: integrity.content_hash,
        byte_length: integrity.byte_length,
        created_at: RECORDED_AT,
    };
    const assets = Array.from({ length: textBlockCount }, (_, index) => ({
        ...asset,
        id: textBlockCount === 1 ? asset.id : `${asset.id}:${index}`,
    }));
    const archiveBatch = { assets };
    const archiveOptions = {
        operation_id: `processing:archive:${job.id}`,
        expected_revision: accepted.revision,
        payload_fingerprint: integrity.content_hash,
        recorded_at: RECORDED_AT,
    };
    const archived = (await appendConversationRecordsWithProcessing(accepted, archiveBatch, archiveOptions)).document;
    const document = archived; // The materialized parity witness stays small; it is never a cold-history facade.
    let staged: StagedIndexedConversationRoot;
    if (coldCount === 0) {
        staged = await stageIndexedConversationSnapshot(archived, undefined, store);
    } else {
        // Reuse the genuine one-time 100k cold migration profile without exceeding the source
        // array bound by adding the active turn. Accept that turn/archive through bounded indexed append.
        const coldTemplate = userTurn('turn:cold-template', 'block:cold-template');
        const cold = {
            ...enabled,
            turns: Array.from({ length: coldCount }, (_, i) => ({
                ...coldTemplate,
                id: `turn:cold:${i}`,
                blocks: [{ ...coldTemplate.blocks[0], id: `block:cold:${i}` }],
            })),
        };
        staged = await stageIndexedConversationSnapshot(cold, undefined, store);
        const appended = await stageIndexedRecordBatch(
            staged.root,
            { conversation_id: enabled.id, batch: inputBatch, options: inputOptions },
            store,
        );
        if (!appended.locator) throw new Error('Fixture indexed input acceptance has no root');
        const archive = await stageIndexedRecordBatch(
            appended.root,
            { conversation_id: enabled.id, batch: archiveBatch, options: archiveOptions },
            store,
        );
        if (!archive.locator) throw new Error('Fixture indexed archive acceptance has no root');
        staged = { root: archive.root, locator: archive.locator };
    }
    const selected = await loadIndexedProcessingSelectedContext(store, staged.root, staged.locator);
    const resolution = await resolveIndexedProcessingTextInput(selected, job, RECORDED_AT);
    staged = await stageIndexedProcessingPhase(store, staged.root, staged.locator, {
        phase: 'resolve',
        value: resolution,
    });
    const attempt = {
        job_id: job.id,
        resolved_input_fingerprint: await fingerprintJson(resolution),
        attempt_token: 'attempt:original',
        started_at: RECORDED_AT,
    };
    staged = await stageIndexedProcessingPhase(store, staged.root, staged.locator, {
        phase: 'attempt',
        value: attempt,
    });
    const workspace = IndexedProcessingClaimWorkspaceSchema.parse({
        version: 1,
        selected: await loadIndexedProcessingSelectedContext(store, staged.root, staged.locator),
        job,
        resolution,
        attempt,
        configuration,
        snapshot_at: RECORDED_AT,
        archives: {
            assets,
            acceptance: archived.operation_receipts[`processing:archive:${job.id}`],
            retrievals: assets.map((item) => ({
                capability: 'read_artifact',
                version: 1,
                tool_definition_id: 'definition:read',
                arguments: { path: 'original', asset_id: item.id },
            })),
        },
    });
    return { store, staged, workspace, document, reads, records, pages };
}
