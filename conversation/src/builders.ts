import type { z } from 'zod';
import { ConversationValidationError } from './diagnostics.js';
import { preflightJsonInput } from './json-preflight.js';
import {
    ContentBlockSchema,
    ConversationTurnSchema,
    GeneratedAgentTurnSchema,
    ProgramTurnSchema,
    TextBlockSchema,
    ToolTurnSchema,
    UserTurnSchema,
} from './schemas/index.js';
import {
    CONVERSATION_EXPERIMENTAL_REVISION,
    CONVERSATION_FORMAT,
    CONVERSATION_SCHEMA_VERSION,
} from './schemas/primitives.js';
import type {
    ContentBlock,
    ConversationDocument,
    ConversationMetadata,
    ConversationTurn,
    GeneratedAgentTurn,
    ProgramTurn,
    TextBlock,
    ToolTurn,
    UserTurn,
} from './types.js';
import { diagnosticsFromZodError, parseConversationDocument } from './validation.js';

export type ConversationDocumentBuilderInput = Pick<ConversationDocument, 'id' | 'created_at'> &
    Partial<Pick<ConversationDocument, 'revision' | 'updated_at'>> & {
        metadata?: ConversationMetadata;
    };

export type TextBlockBuilderInput = Omit<TextBlock, 'type'>;
export type UserTurnBuilderInput = Omit<UserTurn, 'kind'>;
export type GeneratedAgentTurnBuilderInput = Omit<GeneratedAgentTurn, 'kind'>;
export type ToolTurnBuilderInput = Omit<ToolTurn, 'kind'>;
export type ProgramTurnBuilderInput = Omit<ProgramTurn, 'kind'>;

function buildSchemaValue<T>(schema: z.ZodType<T>, input: unknown): T {
    const preflight = preflightJsonInput(input);
    if (!preflight.success) {
        throw new ConversationValidationError('Conversation value failed JSON preflight', preflight.diagnostics);
    }
    const result = schema.safeParse(input);
    if (!result.success) {
        throw new ConversationValidationError(
            'Conversation value failed schema validation',
            diagnosticsFromZodError(result.error),
        );
    }
    // The preflight and schema prove that the original value is JSON-safe and shape-valid. Clone the
    // original to avoid Zod record reconstruction changing an accepted own-key representation.
    return structuredClone(input) as T;
}

function assertJsonBuilderInput(input: unknown): void {
    const preflight = preflightJsonInput(input);
    if (!preflight.success) {
        throw new ConversationValidationError(
            'Conversation builder input failed JSON preflight',
            preflight.diagnostics,
        );
    }
}

export function createConversationDocument(input: ConversationDocumentBuilderInput): ConversationDocument {
    assertJsonBuilderInput(input);
    const revision = input.revision ?? 0;
    return parseConversationDocument({
        format: CONVERSATION_FORMAT,
        schema_version: CONVERSATION_SCHEMA_VERSION,
        experimental_revision: CONVERSATION_EXPERIMENTAL_REVISION,
        id: input.id,
        revision,
        created_at: input.created_at,
        updated_at: input.updated_at ?? input.created_at,
        turns: [],
        generations: {},
        operation_receipts: {},
        execution_receipts: {},
        assets: {},
        tool_definitions: {},
        context: {
            revision,
            entries: [],
            active_tool_definition_ids: [],
            protected_entry_ids: [],
            retrieval_requirements: [],
        },
        compactions: {},
        processing: {
            enabled: false,
            policy_revision: 0,
            processors: [],
        },
        ...(input.metadata === undefined ? {} : { metadata: input.metadata }),
    });
}

export function createTextBlock(input: TextBlockBuilderInput): TextBlock {
    assertJsonBuilderInput(input);
    return buildSchemaValue(TextBlockSchema, { ...input, type: 'text' });
}

export function createUserTurn(input: UserTurnBuilderInput): UserTurn {
    assertJsonBuilderInput(input);
    return buildSchemaValue(UserTurnSchema, { ...input, kind: 'user' });
}

export function createGeneratedAgentTurn(input: GeneratedAgentTurnBuilderInput): GeneratedAgentTurn {
    assertJsonBuilderInput(input);
    return buildSchemaValue(GeneratedAgentTurnSchema, { ...input, kind: 'agent' });
}

export function createToolTurn(input: ToolTurnBuilderInput): ToolTurn {
    assertJsonBuilderInput(input);
    return buildSchemaValue(ToolTurnSchema, { ...input, kind: 'tool' });
}

export function createProgramTurn(input: ProgramTurnBuilderInput): ProgramTurn {
    assertJsonBuilderInput(input);
    return buildSchemaValue(ProgramTurnSchema, { ...input, kind: 'program' });
}

export function buildConversationTurn(input: unknown): ConversationTurn {
    return buildSchemaValue(ConversationTurnSchema, input);
}

export function buildContentBlock(input: unknown): ContentBlock {
    return buildSchemaValue(ContentBlockSchema, input);
}
