import { z } from 'zod';
import { ConversationDocumentSchema } from './document.js';
import { ConversationRuntimeContextSchema, RequestReceiptSchema } from './execution.js';
import { IndexedPreparedSourceSchema } from './indexed-prepared-source.js';
import { ConversationRefSchema, IdentifierSchema } from './primitives.js';

/** Runtime identity after adapter defaults have been resolved. */
export const ResolvedConversationRuntimeContextSchema = ConversationRuntimeContextSchema.extend({
    conversation_id: IdentifierSchema,
    purpose: IdentifierSchema,
}).meta({ id: 'ConversationResolvedRuntimeContext' });

/** Serializable evidence produced after native request preparation and before provider transport. */
export const ConversationPreparedRequestRecordSchema = z
    .strictObject({
        source: ConversationRefSchema,
        runtime: ResolvedConversationRuntimeContextSchema,
        request_receipt: RequestReceiptSchema,
        generation_id: IdentifierSchema,
        response_turn_id: IdentifierSchema,
        indexed_source: IndexedPreparedSourceSchema.optional(),
    })
    .meta({ id: 'ConversationPreparedRequestRecord' });

/**
 * Host persistence payload for the pre-provider durability barrier.
 *
 * The document is the complete current working document, not a bounded output fragment. Hosts use it
 * transiently to validate the prepared record. Retention policies that exclude input history persist
 * only the record; permitted working-history storage is managed independently from this evidence.
 */
export const ConversationPreparedRequestSchema = z
    .strictObject({
        document: ConversationDocumentSchema,
        record: ConversationPreparedRequestRecordSchema,
    })
    .meta({ id: 'ConversationPreparedRequest' });
