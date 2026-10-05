import { describe, expect, it } from 'vitest';
import { fingerprintJson } from '../src/identity.js';
import { indexedPredecessorEntrySelection } from '../src/indexed-processing-working-set.js';
import { ContextChangeRequestSchema } from '../src/schemas/context-change.js';
import { IndexedProcessingPredecessorEvidenceSchema } from '../src/schemas/indexed-processing.js';
import { ProcessingJobSchema } from '../src/schemas/processing.js';
import { RECORDED_AT, userTurn } from './fixtures.js';

async function predecessorFixture(status: 'applied' | 'no_op' | 'skipped' = 'no_op') {
    const configuration = {};
    const selection = {
        kind: 'entries' as const,
        entry_ids: ['entry:original'],
        selected_entries: [{ id: 'entry:original', type: 'source_turn' as const, turn_id: 'turn:original' }],
        selected_block_ids: { 'entry:original': ['block:selected'] },
    };
    const prior = ProcessingJobSchema.parse({
        id: 'job:first',
        source_operation_id: 'operation:append',
        enqueue_revision: 1,
        policy_revision: 1,
        stage_index: 0,
        processor_index: 0,
        processor_id: 'externalize-text',
        processor_version: '1',
        configuration,
        configuration_fingerprint: await fingerprintJson(configuration),
        scope: 'on_append',
        required: false,
        failure_behavior: 'skip_with_diagnostic',
        selection,
        selection_fingerprint: await fingerprintJson(selection),
    });
    const resolution = {
        job_id: prior.id,
        source_revision: 1,
        context_revision: 1,
        entry_ids: selection.entry_ids,
        selected_entries: selection.selected_entries,
        selected_block_ids: selection.selected_block_ids,
        source_fingerprint: await fingerprintJson(selection),
        context_fingerprint: await fingerprintJson({ context: 1 }),
        source_turn_ids: ['turn:original'],
        recorded_at: RECORDED_AT,
    };
    const resolvedIdentity = await fingerprintJson(resolution);
    const attempted = status !== 'no_op';
    const payload = {
        job_id: prior.id,
        resolved_input_fingerprint: resolvedIdentity,
        recorded_at: RECORDED_AT,
        ...(attempted ? { attempt_token: 'attempt:first' } : {}),
        ...(status === 'applied'
            ? {
                  kind: 'proposal',
                  proposal: {
                      kind: 'replace_with_compaction',
                      compaction_id: 'compaction:first',
                      strategy: {
                          id: 'externalize-text',
                          version: '1',
                          configuration_fingerprint: prior.configuration_fingerprint,
                      },
                      replacement_turns: [
                          {
                              ...userTurn('turn:replacement'),
                              provenance: {
                                  type: 'derived',
                                  derivation_id: 'compaction:first',
                                  source_turn_ids: ['turn:original'],
                                  source_hash: resolution.source_fingerprint,
                              },
                          },
                      ],
                      fidelity: 'heuristic',
                      retained_asset_ids: [],
                      generation_ids: [],
                      placement: { mode: 'first_selected', causal_order: 'contiguous' },
                  },
              }
            : status === 'skipped'
              ? { kind: 'failed', diagnostic: 'Processor failed before mutation' }
              : { kind: 'no_op', reason: 'no_eligible_blocks' }),
    };
    const output = { ...payload, output_fingerprint: await fingerprintJson(payload) };
    const completion = {
        job_id: prior.id,
        output_fingerprint: output.output_fingerprint,
        status,
        result_revision: 4,
        inserted_entry_ids: status === 'applied' ? ['entry:replacement', 'entry:remainder'] : [],
        ...(status === 'applied' ? { context_change_operation_id: `processing:apply:${prior.id}` } : {}),
        recorded_at: RECORDED_AT,
    };
    const completionIdentity = await fingerprintJson(completion);
    const evidence = IndexedProcessingPredecessorEvidenceSchema.parse({
        job: prior,
        resolution,
        output,
        completion,
        ...(attempted
            ? {
                  attempt: {
                      job_id: prior.id,
                      resolved_input_fingerprint: resolvedIdentity,
                      attempt_token: 'attempt:first',
                      started_at: RECORDED_AT,
                  },
              }
            : {}),
        resolution_receipt: {
            id: `processing:resolve:${prior.id}`,
            conversation_id: 'conversation:stages',
            payload_fingerprint: resolvedIdentity,
            base_revision: 1,
            result_revision: 2,
            recorded_at: RECORDED_AT,
            operation_kind: 'processing',
            processing_operation: {
                phase: 'resolve',
                job_id: prior.id,
                policy_revision: 1,
                result_fingerprint: resolvedIdentity,
            },
        },
        receipt: {
            id: status === 'applied' ? `processing:apply:${prior.id}` : `processing:complete:${prior.id}`,
            conversation_id: 'conversation:stages',
            payload_fingerprint: completionIdentity,
            base_revision: 3,
            result_revision: 4,
            recorded_at: RECORDED_AT,
            ...(status === 'applied'
                ? {
                      operation_kind: 'context_change',
                      accepted_context_entry_ids: completion.inserted_entry_ids,
                      context_change: {
                          kind: 'replace_with_compaction',
                          removed_entry_ids: selection.entry_ids,
                          inserted_entry_ids: completion.inserted_entry_ids,
                          source_fingerprint: resolution.source_fingerprint,
                          selected_block_ids: selection.selected_block_ids,
                          remainder_entry_ids: ['entry:remainder'],
                      },
                  }
                : {
                      operation_kind: 'processing',
                      processing_operation: {
                          phase: 'complete',
                          job_id: prior.id,
                          policy_revision: 1,
                          result_fingerprint: completionIdentity,
                      },
                  }),
        },
    });
    if (status === 'applied' && evidence.output.kind === 'proposal') {
        evidence.receipt.payload_fingerprint = await fingerprintJson(
            ContextChangeRequestSchema.parse({
                operation_id: evidence.receipt.id,
                recorded_at: evidence.receipt.recorded_at,
                expected_revision: evidence.receipt.base_revision,
                expected_context_revision: resolution.context_revision,
                expected_source_fingerprint: evidence.receipt.context_change!.source_fingerprint,
                entry_ids: resolution.entry_ids,
                selected_block_ids: resolution.selected_block_ids,
                selected_entries: resolution.selected_entries,
                proposal: evidence.output.proposal,
            }),
        );
    }
    const nextSelection = { kind: 'predecessor_output' as const, job_id: prior.id };
    const job = ProcessingJobSchema.parse({
        ...prior,
        id: 'job:second',
        stage_index: 1,
        processor_index: 1,
        selection: nextSelection,
        selection_fingerprint: await fingerprintJson(nextSelection),
    });
    return { job, evidence };
}

describe('indexed ordered predecessor evidence', () => {
    it('selects exactly accepted inserted entries including a partial remainder and replays identically', async () => {
        const { job, evidence } = await predecessorFixture('applied');
        const expected = { entry_ids: ['entry:replacement', 'entry:remainder'] };
        expect(await indexedPredecessorEntrySelection(job, evidence)).toEqual(expected);
        expect(await indexedPredecessorEntrySelection(job, structuredClone(evidence))).toEqual(expected);
    });

    it.each(['no_op', 'skipped'] as const)('retains the immutable partial resolution after %s', async (status) => {
        const { job, evidence } = await predecessorFixture(status);
        expect(await indexedPredecessorEntrySelection(job, evidence)).toEqual({
            entry_ids: evidence.resolution.entry_ids,
            selected_entries: evidence.resolution.selected_entries,
            selected_block_ids: evidence.resolution.selected_block_ids,
        });
    });

    it('blocks a required failed predecessor even when its policy asks to skip failures', async () => {
        const { job, evidence } = await predecessorFixture('skipped');
        evidence.job.required = true;
        await expect(indexedPredecessorEntrySelection(job, evidence)).rejects.toThrow();
    });

    it('rejects a different cohort, policy order, resolution receipt, attempt, or partial insertion receipt', async () => {
        const { job, evidence } = await predecessorFixture('applied');
        const corruptions = [
            { ...job, source_operation_id: 'operation:unrelated' },
            { ...job, policy_revision: 2 },
            { ...job, processor_index: 0 },
            { ...job, stage_index: 3 },
        ];
        for (const corrupted of corruptions)
            await expect(indexedPredecessorEntrySelection(corrupted, evidence)).rejects.toThrow();
        const corruptResolution = structuredClone(evidence);
        corruptResolution.resolution_receipt.processing_operation!.result_fingerprint = await fingerprintJson({
            wrong: 1,
        });
        await expect(indexedPredecessorEntrySelection(job, corruptResolution)).rejects.toThrow();
        const corruptAttempt = structuredClone(evidence);
        corruptAttempt.attempt!.attempt_token = 'attempt:unrelated';
        await expect(indexedPredecessorEntrySelection(job, corruptAttempt)).rejects.toThrow();
        const corruptSource = structuredClone(evidence);
        corruptSource.receipt.context_change!.source_fingerprint = await fingerprintJson({ changedSource: 1 });
        await expect(indexedPredecessorEntrySelection(job, corruptSource)).rejects.toThrow();
        const corruptPartial = structuredClone(evidence);
        corruptPartial.receipt.context_change!.selected_block_ids!['entry:original'] = ['block:other'];
        await expect(indexedPredecessorEntrySelection(job, corruptPartial)).rejects.toThrow();
        const corruptProposal = structuredClone(evidence);
        if (
            corruptProposal.output.kind !== 'proposal' ||
            corruptProposal.output.proposal.kind !== 'replace_with_compaction'
        )
            throw new Error('Applied fixture lost its proposal');
        const replacementBlock = corruptProposal.output.proposal.replacement_turns[0].blocks[0];
        if (replacementBlock.type !== 'text') throw new Error('Applied fixture lost its text block');
        replacementBlock.text = 'different replacement';
        const { output_fingerprint: _priorFingerprint, ...changedOutput } = corruptProposal.output;
        corruptProposal.output.output_fingerprint = await fingerprintJson(changedOutput);
        corruptProposal.completion.output_fingerprint = corruptProposal.output.output_fingerprint;
        await expect(indexedPredecessorEntrySelection(job, corruptProposal)).rejects.toThrow('exact output proposal');
        const corruptReceipt = structuredClone(evidence);
        corruptReceipt.receipt.context_change!.inserted_entry_ids.pop();
        await expect(indexedPredecessorEntrySelection(job, corruptReceipt)).rejects.toThrow();
    });
});
