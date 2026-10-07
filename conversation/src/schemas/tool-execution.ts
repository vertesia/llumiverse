import { z } from 'zod';
import { ApplicationToolCallBlockSchema, AssetSchema, ToolTurnSchema } from './content.js';

export { ApplicationToolCallBlockSchema } from './content.js';

import { ExecutionReceiptSchema, ToolCallSourceRefSchema } from './execution.js';
import { IdentifierSchema, JsonObjectSchema } from './primitives.js';

export const PendingApplicationToolCallSchema = z
    .strictObject({
        source: ToolCallSourceRefSchema,
        call: ApplicationToolCallBlockSchema.pick({
            call_id: true,
            tool_name: true,
            definition_id: true,
            executor: true,
        }),
    })
    .superRefine((value, ctx) => {
        if (value.call.call_id !== value.source.call_id) {
            ctx.addIssue({
                code: 'custom',
                path: ['call', 'call_id'],
                message: 'Pending call ID must match its source call ID',
            });
        }
    })
    .meta({ id: 'ConversationPendingApplicationToolCall' });

export const ApplicationToolExecutionReceiptSchema = ExecutionReceiptSchema.extend({
    executor: z.literal('application'),
    call_source: ToolCallSourceRefSchema,
    result_turn_id: IdentifierSchema,
}).meta({ id: 'ConversationApplicationToolExecutionReceipt' });

export const ExecutedToolTurnSchema = ToolTurnSchema.extend({
    execution_id: IdentifierSchema,
}).meta({ id: 'ConversationExecutedToolTurn' });

/** Transient exact input presented to a tool runner; it is not persisted as another argument copy. */
export const ConversationToolExecutionRequestSchema = z
    .strictObject({
        source: ToolCallSourceRefSchema,
        call: ApplicationToolCallBlockSchema,
        arguments: JsonObjectSchema,
    })
    .meta({ id: 'ConversationToolExecutionRequest' });

/** Canonical records produced by one terminal application tool execution. */
export const ConversationToolExecutionResultSchema = z
    .strictObject({
        source: ToolCallSourceRefSchema,
        turn: ExecutedToolTurnSchema,
        assets: z.array(AssetSchema).readonly().optional(),
        execution_receipt: ApplicationToolExecutionReceiptSchema,
    })
    .meta({ id: 'ConversationToolExecutionResult' });
