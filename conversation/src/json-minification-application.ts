import { canonicalJsonContentString } from './content-integrity.js';
import { createContextTurnIndex, resolveContextEntry } from './context-entry-resolution.js';
import { deriveConversationId, fingerprintJson } from './identity.js';
import { assertRetainedJsonMinificationOutput } from './json-minification-evidence.js';
import { jsonMinificationProcessor } from './json-minification-processor.js';
import { ConversationTurnSchema } from './schemas/content.js';
import { JsonMinificationApplicationSchema } from './schemas/json-minification.js';
import type {
    ContentBlock,
    ContextEntry,
    ConversationDocument,
    ConversationTurn,
    DerivedBlockLineageGroup,
    OperationReceipt,
    ProcessingJob,
    ProcessingOutputReceipt,
    ProcessingResolvedInput,
    SourceBlockSlice,
} from './types.js';
import { parseConversationDocument } from './validation.js';

/** Only the durable processing runner calls this after retaining the exact host-measured output. */
export async function applyJsonMinificationOutput(
    source: ConversationDocument,
    job: ProcessingJob,
    resolution: ProcessingResolvedInput,
    output: Extract<ProcessingOutputReceipt, { kind: 'json_minification' }>,
    recordedAt: string,
    signal?: AbortSignal,
): Promise<ConversationDocument> {
    signal?.throwIfAborted();
    await assertRetainedJsonMinificationOutput(source, job, resolution, output);
    const proposal = output.proposal;
    const { measurement: _measurement, kind: _kind, ...plan } = proposal;
    const expected = await jsonMinificationProcessor.run({
        document: structuredClone(source),
        job: structuredClone(job),
        resolved_input: structuredClone(resolution),
        signal,
        configuration: {
            id: job.processor_id,
            version: job.processor_version,
            config: job.configuration,
            scope: job.scope,
            required: job.required,
            failure_behavior: job.failure_behavior,
        },
    });
    if (
        canonicalJsonContentString(expected) !==
        canonicalJsonContentString({ ...plan, kind: 'json_minification_candidate' })
    )
        throw new Error('Durable JSON minification output does not match its retained source/configuration/target');
    const operationId = `processing:complete:${job.id}`;
    const compactionId = await deriveConversationId('compaction', operationId);
    const turns = createContextTurnIndex(source);
    const replacementTurns: ConversationTurn[] = [];
    const replacements = new Map<string, ContextEntry>();
    const allSlices: SourceBlockSlice[] = [];
    const selectedEntries: { id: string; fingerprint: string }[] = [];
    const positions: number[] = [];
    for (const [position, entry] of source.context.entries.entries()) {
        signal?.throwIfAborted();
        const transforms = proposal.transforms.filter((item) => item.entry_id === entry.id);
        if (!transforms.length) continue;
        const { turn, blocks } = resolveContextEntry(turns, entry);
        if (
            turn.kind === 'tool' ||
            turn.execution_id !== undefined ||
            turn.exchange_id !== undefined ||
            blocks.some((block) => block.type !== 'text' && block.type !== 'json')
        )
            throw new Error('JSON minification cannot rewrite executable or replay-bearing turns');
        const turnId = await deriveConversationId('turn', operationId, entry.id);
        const nextBlocks: ContentBlock[] = [];
        const groups: DerivedBlockLineageGroup[] = [];
        for (const block of blocks) {
            signal?.throwIfAborted();
            const transform = transforms.find((item) => item.source_slice.block_id === block.id);
            const id = await deriveConversationId('block', operationId, entry.id, block.id);
            const slice: SourceBlockSlice = {
                source: { conversation_id: source.id, revision: source.revision },
                turn_id: turn.id,
                block_id: block.id,
                block_fingerprint: await fingerprintJson(block),
                selection: { kind: 'whole' },
            };
            nextBlocks.push(
                transform && block.type === 'text'
                    ? { ...block, id, text: transform.replacement_text }
                    : { ...block, id },
            );
            groups.push(
                transform
                    ? {
                          transform: 'json_minification',
                          parser: 'rfc8259-lexical-v1',
                          fidelity: 'value_preserving',
                          target_block_ids: [id],
                          source_slices: [slice],
                      }
                    : {
                          transform: 'block_copy',
                          fidelity: 'value_preserving',
                          target_block_ids: [id],
                          source_slices: [slice],
                      },
            );
            allSlices.push(slice);
        }
        const replacement = {
            ...turn,
            id: turnId,
            blocks: nextBlocks,
            provenance: {
                type: 'derived',
                derivation_id: compactionId,
                source_turn_ids: [turn.id],
                source_block_ids: blocks.map((block) => block.id),
                source_hash: resolution.source_fingerprint,
                block_lineage: { version: 1, groups },
            },
        };
        // Full schema parsing below checks the preserved role's allowed blocks and metadata.
        const parsed = ConversationTurnSchema.parse(replacement);
        replacementTurns.push(parsed);
        replacements.set(entry.id, {
            id: await deriveConversationId('context_entry', operationId, entry.id),
            type: 'replacement_turn',
            compaction_id: compactionId,
            turn_id: turnId,
        });
        selectedEntries.push({ id: entry.id, fingerprint: await fingerprintJson(entry) });
        positions.push(position);
    }
    const createdEntries = [...replacements.values()];
    const application = JsonMinificationApplicationSchema.parse({
        kind: 'json_minification',
        compaction_id: compactionId,
        source: { conversation_id: source.id, revision: source.revision },
        source_context_revision: source.context.revision,
        source_fingerprint: resolution.source_fingerprint,
        output_fingerprint: output.output_fingerprint,
        selected_entries: selectedEntries,
        source_slices: allSlices,
        source_entry_positions: positions,
        created_entries: await Promise.all(
            createdEntries.map(async (item) => ({ id: item.id, fingerprint: await fingerprintJson(item) })),
        ),
        created_turns: await Promise.all(
            replacementTurns.map(async (item) => ({ id: item.id, fingerprint: await fingerprintJson(item) })),
        ),
    });
    const completion = {
        job_id: job.id,
        output_fingerprint: output.output_fingerprint,
        status: 'applied' as const,
        result_revision: source.revision + 1,
        inserted_entry_ids: createdEntries.map((entry) => entry.id),
        recorded_at: recordedAt,
    };
    const receipt: OperationReceipt = {
        id: operationId,
        conversation_id: source.id,
        base_revision: source.revision,
        result_revision: source.revision + 1,
        payload_fingerprint: await fingerprintJson({ completion, application }),
        recorded_at: recordedAt,
        operation_kind: 'processing',
        accepted_turn_ids: [],
        accepted_generation_ids: [],
        accepted_asset_ids: [],
        accepted_tool_definition_ids: [],
        accepted_execution_receipt_ids: [],
        accepted_context_entry_ids: createdEntries.map((entry) => entry.id),
        processing_operation: {
            phase: 'complete',
            job_id: job.id,
            policy_revision: source.processing.policy_revision,
            result_fingerprint: await fingerprintJson(completion),
            transform_application: application,
        },
    };
    const { coverage: _coverage, ...processing } = source.processing;
    signal?.throwIfAborted();
    return parseConversationDocument({
        ...source,
        revision: receipt.result_revision,
        updated_at: recordedAt,
        context: {
            ...source.context,
            revision: source.context.revision + 1,
            entries: source.context.entries.map((entry) => replacements.get(entry.id) ?? entry),
        },
        operation_receipts: { ...source.operation_receipts, [operationId]: receipt },
        compactions: {
            ...source.compactions,
            [compactionId]: {
                id: compactionId,
                operation_id: operationId,
                strategy: proposal.strategy,
                source: {
                    turn_ids: [...new Set(allSlices.map((slice) => slice.turn_id))],
                    block_ids: [...new Set(allSlices.map((slice) => slice.block_id))],
                    source_fingerprint: resolution.source_fingerprint,
                },
                replacement_turns: replacementTurns,
                fidelity: 'value_preserving',
                retained_asset_ids: [],
                generation_ids: [],
                created_at: recordedAt,
            },
        },
        processing: { ...processing, completions: { ...processing.completions, [job.id]: completion } },
    });
}
