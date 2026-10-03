import { createConversationChangeSchema } from './change.js';
import { ProcessingChangeOperationSchema } from './processing-operation.js';

export const ProcessingChangeSchema = createConversationChangeSchema(
    ProcessingChangeOperationSchema,
    'ConversationProcessingChange',
);
