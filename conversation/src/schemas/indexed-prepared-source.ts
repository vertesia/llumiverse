import { z } from 'zod';
import { PagedRecordRefSchema } from '../paged-record-index.js';
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
export const IndexedMeasuredNativePreparedWitnessSchema = z
    .strictObject({
        version: z.literal(1),
        accepted_source: ConversationRefSchema,
        accepted_root: PagedRecordRefSchema,
        accepted_operation_id: IdentifierSchema,
        runtime_input_operation_id: IdentifierSchema,
        accepted_receipt_fingerprint: ContentHashSchema,
        settled_source: ConversationRefSchema,
        settled_root: PagedRecordRefSchema,
        coverage: IndexedProcessingCoverageCommandSchema,
    })
    .superRefine((witness, ctx) => {
        if (
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
        ]),
        root: PagedRecordRefSchema,
        context_revision: NonnegativeSafeIntegerSchema,
        processing_input: IndexedProcessedInputPreparedWitnessSchema.optional(),
        native_measurement: IndexedMeasuredNativePreparedWitnessSchema.optional(),
    })
    .superRefine((source, ctx) => {
        if (
            (source.validator_profile === INDEXED_MEASURED_NATIVE_PREPARED_VALIDATOR_PROFILE) !==
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
