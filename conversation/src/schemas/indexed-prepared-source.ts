import { z } from 'zod';
import { PagedRecordRefSchema } from '../paged-record-index.js';
import { OperationReceiptSchema } from './execution.js';
import { IndexedProcessingCoverageCommandSchema } from './indexed-head.js';
import {
    ContentHashSchema,
    ConversationRefSchema,
    IdentifierSchema,
    NonnegativeSafeIntegerSchema,
} from './primitives.js';

/** Validation of one selected text request against an immutable indexed root. */
export const INDEXED_TEXT_PREPARED_VALIDATOR_PROFILE =
    'llumiverse.conversation/indexed-selected-text/2026-10-03.v1' as const;

export const INDEXED_DEPENDENCY_PREPARED_VALIDATOR_PROFILE =
    'llumiverse.conversation/indexed-selected-dependencies/2026-10-04.v1' as const;

export const INDEXED_MEDIA_COMPACTION_PREPARED_VALIDATOR_PROFILE =
    'llumiverse.conversation/indexed-selected-media-compaction/2026-10-04.v1' as const;

/** Processed input is a separate private validator path. These locators are equality constraints;
 * only the host authenticates the retained epoch/closure and reconstructs their finite phases. */
export const INDEXED_PROCESSED_INPUT_PREPARED_VALIDATOR_PROFILE =
    'llumiverse.conversation/indexed-processed-input/2026-10-05.v1' as const;
export const IndexedProcessedInputPreparedWitnessSchema = z
    .strictObject({
        version: z.literal(1),
        activation_operation_id: IdentifierSchema,
        original_source: ConversationRefSchema,
        original_root: PagedRecordRefSchema,
        accepted_input_revision: NonnegativeSafeIntegerSchema,
        accepted_input_receipt_fingerprint: ContentHashSchema,
        settled_source: ConversationRefSchema,
        settled_root: PagedRecordRefSchema,
        coverage: IndexedProcessingCoverageCommandSchema,
    })
    .superRefine((witness, ctx) => {
        if (
            witness.original_source.conversation_id !== witness.settled_source.conversation_id ||
            witness.accepted_input_revision !== witness.original_source.revision + 1 ||
            witness.settled_source.revision <= witness.accepted_input_revision ||
            witness.coverage.expected_revision !== witness.settled_source.revision
        )
            ctx.addIssue({
                code: 'custom',
                message:
                    'Processed indexed evidence must preserve its exact original input and settled coverage source',
            });
    });
export type IndexedProcessedInputPreparedWitness = z.infer<typeof IndexedProcessedInputPreparedWitnessSchema>;

/** A generic accepted-input/native-count witness. The host independently authenticates the
 * accepted receipt and finite processing lineage; these immutable references grant no authority. */
export const INDEXED_MEASURED_NATIVE_PREPARED_VALIDATOR_PROFILE =
    'llumiverse.conversation/indexed-measured-native/2026-10-05.v1' as const;
/** An executed accepted output is not materialized input. Its original receipt can precede
 * the authenticated retained physical root (for example after policy edits or migration). */
export const INDEXED_MEASURED_OUTPUT_PREPARED_VALIDATOR_PROFILE =
    'llumiverse.conversation/indexed-measured-output/2026-10-06.v1' as const;
/** The complete original append receipt, including payload fingerprint. Public output receipts
 * intentionally omit private append evidence and cannot stand in for this hash-bound witness. */
export const IndexedMeasuredOutputReceiptSchema = OperationReceiptSchema.extend({
    accepted_turn_ids: z.array(IdentifierSchema).length(1),
    accepted_generation_ids: z.array(IdentifierSchema).length(1),
    operation_kind: z.never().optional(),
    context_change: z.never().optional(),
    conversation_edit: z.never().optional(),
    conversation_delete: z.never().optional(),
    processing_operation: z.never().optional(),
    indexed_upgrade: z.never().optional(),
}).superRefine((receipt, ctx) => {
    if (receipt.result_revision !== receipt.base_revision + 1)
        ctx.addIssue({ code: 'custom', message: 'Measured output witness requires one authentic append revision' });
});
export const IndexedMeasuredNativePreparedWitnessSchema = z
    .strictObject({
        version: z.literal(1),
        accepted_source: ConversationRefSchema,
        accepted_root: PagedRecordRefSchema,
        accepted_operation_id: IdentifierSchema,
        runtime_input_operation_id: IdentifierSchema,
        accepted_receipt_fingerprint: ContentHashSchema,
        retained_output: IndexedMeasuredOutputReceiptSchema.optional(),
        settled_source: ConversationRefSchema,
        settled_root: PagedRecordRefSchema,
        coverage: IndexedProcessingCoverageCommandSchema,
    })
    .superRefine((witness, ctx) => {
        if (
            (witness.retained_output !== undefined &&
                (witness.retained_output.id !== witness.accepted_operation_id ||
                    witness.retained_output.conversation_id !== witness.accepted_source.conversation_id ||
                    witness.retained_output.result_revision > witness.accepted_source.revision)) ||
            witness.accepted_source.conversation_id !== witness.settled_source.conversation_id ||
            witness.settled_source.revision < witness.accepted_source.revision ||
            witness.coverage.expected_revision !== witness.settled_source.revision ||
            (witness.settled_source.revision === witness.accepted_source.revision &&
                (witness.accepted_root.content_hash !== witness.settled_root.content_hash ||
                    witness.accepted_root.size_bytes !== witness.settled_root.size_bytes))
        )
            ctx.addIssue({ code: 'custom', message: 'Measured native source lost its exact accepted input lineage' });
    });
export type IndexedMeasuredNativePreparedWitness = z.infer<typeof IndexedMeasuredNativePreparedWitnessSchema>;

/** Private prepared evidence. The root is loaded by hash under an authenticated run prefix. */
export const IndexedPreparedSourceSchema = z
    .strictObject({
        version: z.literal(1),
        validator_profile: z.enum([
            INDEXED_TEXT_PREPARED_VALIDATOR_PROFILE,
            INDEXED_DEPENDENCY_PREPARED_VALIDATOR_PROFILE,
            INDEXED_MEDIA_COMPACTION_PREPARED_VALIDATOR_PROFILE,
            INDEXED_PROCESSED_INPUT_PREPARED_VALIDATOR_PROFILE,
            INDEXED_MEASURED_NATIVE_PREPARED_VALIDATOR_PROFILE,
            INDEXED_MEASURED_OUTPUT_PREPARED_VALIDATOR_PROFILE,
        ]),
        root: PagedRecordRefSchema,
        context_revision: NonnegativeSafeIntegerSchema,
        processing_input: IndexedProcessedInputPreparedWitnessSchema.optional(),
        native_measurement: IndexedMeasuredNativePreparedWitnessSchema.optional(),
    })
    .superRefine((source, ctx) => {
        if (
            (source.validator_profile === INDEXED_MEASURED_OUTPUT_PREPARED_VALIDATOR_PROFILE) !==
            (source.native_measurement?.retained_output !== undefined)
        )
            ctx.addIssue({
                code: 'custom',
                path: ['native_measurement', 'retained_output'],
                message: 'Measured accepted-output preparation requires its distinct original output receipt',
            });
        if (
            (source.validator_profile === INDEXED_MEASURED_NATIVE_PREPARED_VALIDATOR_PROFILE ||
                source.validator_profile === INDEXED_MEASURED_OUTPUT_PREPARED_VALIDATOR_PROFILE) !==
            (source.native_measurement !== undefined)
        )
            ctx.addIssue({
                code: 'custom',
                path: ['native_measurement'],
                message: 'Measured native preparation requires its distinct exact accepted-input/coverage witness',
            });
        if (
            (source.validator_profile === INDEXED_PROCESSED_INPUT_PREPARED_VALIDATOR_PROFILE) !==
            (source.processing_input !== undefined)
        )
            ctx.addIssue({
                code: 'custom',
                path: ['processing_input'],
                message: 'Processed indexed preparation requires its distinct exact input/phase/coverage witness',
            });
    });

export type IndexedPreparedSource = z.infer<typeof IndexedPreparedSourceSchema>;
