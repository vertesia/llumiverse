import { z } from 'zod';
import { canonicalJsonContentBytes } from '../content-integrity.js';
import { AssetSchema, RetrievalCapabilitySchema } from './content.js';
import { ProcessorConfigurationSchema } from './document.js';
import { OperationReceiptSchema } from './execution.js';
import { IndexedProcessingSelectedContextSchema } from './indexed-head.js';
import { TimestampSchema } from './primitives.js';
import {
    ProcessingAttemptReceiptSchema,
    ProcessingCompletionReceiptSchema,
    ProcessingJobSchema,
    ProcessingOutputReceiptSchema,
    ProcessingResolvedInputSchema,
} from './processing.js';

/** The indexed text profile admits only bounded transfer/binding metadata. These are enforced
 * in the shared contract, before archive acceptance or worker output construction. */
export const INDEXED_PROCESSING_ARCHIVE_ASSET_MAX_BYTES = 2048;
export const INDEXED_PROCESSING_RETRIEVAL_MAX_BYTES = 2048;
export const INDEXED_PROCESSING_ARCHIVE_RECEIPT_BASE_MAX_BYTES = 32 * 1024;
export const INDEXED_PROCESSING_ARCHIVE_RECEIPT_PER_ASSET_MAX_BYTES = 2048;
export const IndexedProcessingArchiveAssetSchema = AssetSchema.superRefine((asset, ctx) => {
    if (canonicalJsonContentBytes(asset).byteLength > INDEXED_PROCESSING_ARCHIVE_ASSET_MAX_BYTES) {
        ctx.addIssue({ code: 'custom', message: 'Indexed archive asset exceeds metadata profile' });
    }
});
export const IndexedProcessingRetrievalSchema = RetrievalCapabilitySchema.superRefine((retrieval, ctx) => {
    if (canonicalJsonContentBytes(retrieval).byteLength > INDEXED_PROCESSING_RETRIEVAL_MAX_BYTES) {
        ctx.addIssue({ code: 'custom', message: 'Indexed archive retrieval exceeds metadata profile' });
    }
});
export const IndexedProcessingArchivesSchema = z
    .strictObject({
        assets: z.array(IndexedProcessingArchiveAssetSchema).max(4096),
        acceptance: OperationReceiptSchema,
        retrievals: z.array(IndexedProcessingRetrievalSchema).max(4096),
    })
    .superRefine((archives, ctx) => {
        const maxBytes =
            INDEXED_PROCESSING_ARCHIVE_RECEIPT_BASE_MAX_BYTES +
            archives.assets.length * INDEXED_PROCESSING_ARCHIVE_RECEIPT_PER_ASSET_MAX_BYTES;
        if (canonicalJsonContentBytes(archives.acceptance).byteLength > maxBytes) {
            ctx.addIssue({ code: 'custom', message: 'Indexed archive acceptance exceeds metadata profile' });
        }
    });

/** A point-addressed prior stage, not a current-head or caller-supplied selection grant.
 * The indexed host reads every member from the same immutable accepted job cohort. */
export const IndexedProcessingPredecessorEvidenceSchema = z.strictObject({
    job: ProcessingJobSchema,
    resolution: ProcessingResolvedInputSchema,
    output: ProcessingOutputReceiptSchema,
    resolution_receipt: OperationReceiptSchema,
    attempt: ProcessingAttemptReceiptSchema.optional(),
    completion: ProcessingCompletionReceiptSchema,
    receipt: OperationReceiptSchema,
});
export type IndexedProcessingPredecessorEvidence = z.infer<typeof IndexedProcessingPredecessorEvidenceSchema>;

/** Integrity-bound private processing input. Every turn is explicitly projected, never a full history. */
export const IndexedProcessingClaimWorkspaceSchema = z.strictObject({
    version: z.literal(1),
    selected: IndexedProcessingSelectedContextSchema,
    job: ProcessingJobSchema,
    configuration: ProcessorConfigurationSchema,
    resolution: ProcessingResolvedInputSchema,
    attempt: ProcessingAttemptReceiptSchema,
    predecessor: IndexedProcessingPredecessorEvidenceSchema.optional(),
    snapshot_at: TimestampSchema,
    archives: IndexedProcessingArchivesSchema,
});
export type IndexedProcessingClaimWorkspace = z.infer<typeof IndexedProcessingClaimWorkspaceSchema>;
