import type { z } from 'zod';
import {
    AppendConversationRecordsOptionsSchema,
    AppendConversationRecordsResultSchema,
    ContentBlockSchema,
    ConversationDiagnosticSchema,
    ConversationDocumentSchema,
    ConversationInspectionSchema,
    ConversationRecordBatchSchema,
    ConversationTurnSchema,
    DecodedConversationResponseSchema,
    GenerationSchema,
} from './schemas/index.js';
import { CONVERSATION_EXPERIMENTAL_REVISION } from './schemas/primitives.js';

export type GeneratedConversationJsonSchema = Readonly<Record<string, unknown>>;

const JSON_SCHEMA_OPTIONS = {
    target: 'draft-2020-12',
    io: 'input',
    cycles: 'ref',
    reused: 'ref',
    unrepresentable: 'throw',
} as const;

function sortJsonValue<T>(value: T): T {
    if (Array.isArray(value)) {
        return value.map((item) => sortJsonValue(item)) as T;
    }
    if (value !== null && typeof value === 'object') {
        const sorted = Object.fromEntries(
            Object.entries(value)
                .sort(([first], [second]) => (first < second ? -1 : first > second ? 1 : 0))
                .map(([key, item]) => [key, sortJsonValue(item)]),
        );
        return sorted as T;
    }
    return value;
}

function deepFreeze<T>(value: T): Readonly<T> {
    if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
        for (const child of Object.values(value)) {
            deepFreeze(child);
        }
        Object.freeze(value);
    }
    return value;
}

function emitJsonSchema(schema: z.ZodType, name: string): GeneratedConversationJsonSchema {
    const generated = schema.toJSONSchema(JSON_SCHEMA_OPTIONS);
    return deepFreeze(
        sortJsonValue({
            ...generated,
            $id: `urn:llumiverse:conversation:${CONVERSATION_EXPERIMENTAL_REVISION}:${name}`,
        }),
    );
}

export const ConversationDocumentJsonSchema = emitJsonSchema(ConversationDocumentSchema, 'document');
export const ConversationTurnJsonSchema = emitJsonSchema(ConversationTurnSchema, 'turn');
export const ConversationContentBlockJsonSchema = emitJsonSchema(ContentBlockSchema, 'content-block');
export const ConversationGenerationJsonSchema = emitJsonSchema(GenerationSchema, 'generation');
export const ConversationDiagnosticJsonSchema = emitJsonSchema(ConversationDiagnosticSchema, 'diagnostic');
export const ConversationInspectionJsonSchema = emitJsonSchema(ConversationInspectionSchema, 'inspection');
export const ConversationRecordBatchJsonSchema = emitJsonSchema(ConversationRecordBatchSchema, 'record-batch');
export const AppendConversationRecordsOptionsJsonSchema = emitJsonSchema(
    AppendConversationRecordsOptionsSchema,
    'append-options',
);
export const AppendConversationRecordsResultJsonSchema = emitJsonSchema(
    AppendConversationRecordsResultSchema,
    'append-result',
);
export const DecodedConversationResponseJsonSchema = emitJsonSchema(
    DecodedConversationResponseSchema,
    'decoded-response',
);

export const CONVERSATION_JSON_SCHEMAS = Object.freeze({
    append_options: AppendConversationRecordsOptionsJsonSchema,
    append_result: AppendConversationRecordsResultJsonSchema,
    content_block: ConversationContentBlockJsonSchema,
    diagnostic: ConversationDiagnosticJsonSchema,
    decoded_response: DecodedConversationResponseJsonSchema,
    document: ConversationDocumentJsonSchema,
    generation: ConversationGenerationJsonSchema,
    inspection: ConversationInspectionJsonSchema,
    record_batch: ConversationRecordBatchJsonSchema,
    turn: ConversationTurnJsonSchema,
});
