import {
    type ConversationDocument,
    ConversationValidationError,
    parseConversationDocument,
    preflightJsonInput,
} from '@llumiverse/conversation';
import {
    BEDROCK_CONVERSE_PROTOCOL,
    exportLegacyBedrockConverseConversation,
    importBedrockConverseHistory,
} from '../bedrock/bedrock-converse-conversation-adapter.js';
import {
    exportLegacyOpenAIChatCompletionsConversation,
    importOpenAIChatCompletionsHistory,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
} from '../openai/openai-chat-conversation-adapter.js';
import {
    exportLegacyOpenAIResponsesConversation,
    importOpenAIResponsesHistory,
    OPENAI_RESPONSES_PROTOCOL,
} from '../openai/openai-responses-conversation-adapter.js';
import {
    CLAUDE_MESSAGES_PROTOCOL,
    exportLegacyClaudeMessagesConversation,
    importClaudeMessagesHistory,
} from '../shared/claude-messages-conversation-adapter.js';
import {
    exportLegacyGeminiConversation,
    GEMINI_GENERATE_CONTENT_PROTOCOL,
    importGeminiGenerateContentHistory,
} from '../vertexai/models/gemini-conversation-adapter.js';
import { NativeConversationImportError } from './native-import.js';

export type {
    BedrockConverseConversation,
    PreparedBedrockConverseConversation,
} from '../bedrock/bedrock-converse-conversation-adapter.js';
export {
    appendBedrockConverseCanonicalResponse,
    BEDROCK_CONVERSE_ADAPTER_VERSION,
    BEDROCK_CONVERSE_PROTOCOL,
    compileBedrockConverseConversation,
    decodeBedrockConverseCanonicalResponse,
    exportLegacyBedrockConverseConversation,
    finalizeBedrockConversePreparedRequest,
    importBedrockConverseHistory,
    prepareBedrockConverseCanonicalState,
} from '../bedrock/bedrock-converse-conversation-adapter.js';

export {
    compileOpenAIChatProspectiveJsonMinification,
    exportLegacyOpenAIChatCompletionsConversation,
    importOpenAIChatCompletionsHistory,
    OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    type OpenAIChatProspectiveJsonMinificationProjection,
} from '../openai/openai-chat-conversation-adapter.js';
export {
    exportLegacyOpenAIResponsesConversation,
    importOpenAIResponsesHistory,
    OPENAI_RESPONSES_ADAPTER_VERSION,
    OPENAI_RESPONSES_PROTOCOL,
} from '../openai/openai-responses-conversation-adapter.js';
export {
    CLAUDE_MESSAGES_ADAPTER_VERSION,
    CLAUDE_MESSAGES_PROTOCOL,
    exportLegacyClaudeMessagesConversation,
    importClaudeMessagesHistory,
} from '../shared/claude-messages-conversation-adapter.js';
export type {
    LegacyGeminiConversation,
    PreparedGeminiConversation,
} from '../vertexai/models/gemini-conversation-adapter.js';
export {
    appendGeminiCanonicalResponse,
    compileGeminiConversation,
    decodeGeminiCanonicalResponse,
    exportLegacyGeminiConversation,
    finalizeGeminiPreparedRequest,
    GEMINI_GENERATE_CONTENT_ADAPTER_VERSION,
    GEMINI_GENERATE_CONTENT_PROTOCOL,
    geminiGenerationUsage,
    geminiToolUsesFromContent,
    importGeminiGenerateContentHistory,
    isGeminiGenerateContentHistory,
    prepareGeminiCanonicalState,
} from '../vertexai/models/gemini-conversation-adapter.js';
export type {
    CanonicalStructuredOutputBinding,
    CanonicalStructuredOutputEvidence,
    InvalidStructuredOutputEvidence,
    StructuredOutputReplayRewriter,
} from './structured-output.js';
export {
    assertStructuredOutputEvidence,
    normalizeDecodedStructuredOutput,
    parseStructuredOutputEvidence,
    rejectDecodedStructuredOutput,
    remapStructuredOutputReplayDependencies,
    structuredOutputEvidence,
} from './structured-output.js';

export type CanonicalNativeConversationProtocol =
    | typeof OPENAI_CHAT_COMPLETIONS_PROTOCOL
    | typeof OPENAI_RESPONSES_PROTOCOL
    | typeof CLAUDE_MESSAGES_PROTOCOL
    | typeof GEMINI_GENERATE_CONTENT_PROTOCOL
    | typeof BEDROCK_CONVERSE_PROTOCOL;

export type LegacyConversationProjection =
    | ReturnType<typeof exportLegacyOpenAIChatCompletionsConversation>
    | ReturnType<typeof exportLegacyOpenAIResponsesConversation>
    | ReturnType<typeof exportLegacyClaudeMessagesConversation>
    | ReturnType<typeof exportLegacyGeminiConversation>
    | ReturnType<typeof exportLegacyBedrockConverseConversation>;

function latestSupportedProtocol(document: ConversationDocument): CanonicalNativeConversationProtocol | undefined {
    for (let index = document.turns.length - 1; index >= 0; index -= 1) {
        const turn = document.turns[index];
        if (turn.kind === 'agent' && 'generation_id' in turn && typeof turn.generation_id === 'string') {
            const generation = Object.hasOwn(document.generations, turn.generation_id)
                ? document.generations[turn.generation_id]
                : undefined;
            if (generation?.protocol === OPENAI_CHAT_COMPLETIONS_PROTOCOL) return OPENAI_CHAT_COMPLETIONS_PROTOCOL;
            if (generation?.protocol === OPENAI_RESPONSES_PROTOCOL) return OPENAI_RESPONSES_PROTOCOL;
            if (generation?.protocol === CLAUDE_MESSAGES_PROTOCOL) return CLAUDE_MESSAGES_PROTOCOL;
            if (generation?.protocol === GEMINI_GENERATE_CONTENT_PROTOCOL) return GEMINI_GENERATE_CONTENT_PROTOCOL;
            if (generation?.protocol === BEDROCK_CONVERSE_PROTOCOL) return BEDROCK_CONVERSE_PROTOCOL;
        }
        if (turn.provenance.type === 'imported') {
            if (turn.provenance.source === OPENAI_CHAT_COMPLETIONS_PROTOCOL) return OPENAI_CHAT_COMPLETIONS_PROTOCOL;
            if (turn.provenance.source === OPENAI_RESPONSES_PROTOCOL) return OPENAI_RESPONSES_PROTOCOL;
            if (turn.provenance.source === CLAUDE_MESSAGES_PROTOCOL) return CLAUDE_MESSAGES_PROTOCOL;
            if (turn.provenance.source === GEMINI_GENERATE_CONTENT_PROTOCOL) return GEMINI_GENERATE_CONTENT_PROTOCOL;
            if (turn.provenance.source === BEDROCK_CONVERSE_PROTOCOL) return BEDROCK_CONVERSE_PROTOCOL;
        }
    }
    return undefined;
}

/**
 * Read-only projection for legacy API clients while internal execution persists canonical history.
 * An explicit protocol is required when the document has no supported generation/import provenance.
 */
export function exportLegacyConversation(
    input: ConversationDocument,
    protocol?: CanonicalNativeConversationProtocol,
): LegacyConversationProjection {
    const document = parseConversationDocument(input);
    const resolvedProtocol = protocol ?? latestSupportedProtocol(document);
    if (resolvedProtocol === OPENAI_CHAT_COMPLETIONS_PROTOCOL) {
        return exportLegacyOpenAIChatCompletionsConversation(document);
    }
    if (resolvedProtocol === OPENAI_RESPONSES_PROTOCOL) {
        return exportLegacyOpenAIResponsesConversation(document);
    }
    if (resolvedProtocol === CLAUDE_MESSAGES_PROTOCOL) {
        return exportLegacyClaudeMessagesConversation(document);
    }
    if (resolvedProtocol === GEMINI_GENERATE_CONTENT_PROTOCOL) {
        return exportLegacyGeminiConversation(document);
    }
    if (resolvedProtocol === BEDROCK_CONVERSE_PROTOCOL) {
        return exportLegacyBedrockConverseConversation(document);
    }
    throw new TypeError('Canonical conversation has no supported native protocol provenance; provide a protocol');
}

export type {
    NativeConversationImportDiagnostic,
    NativeConversationImportDiagnosticCode,
    NativeConversationImportFailureCode,
    NativeConversationImportOptions,
    NativeConversationImportReport,
    NativeConversationImportResult,
} from './native-import.js';
export { NativeConversationImportError } from './native-import.js';

/** Protocol selection is explicit: generic text arrays cannot establish protocol or hosting provenance. */
export async function importNativeConversationHistory(
    history: unknown,
    options: import('./native-import.js').NativeConversationImportOptions & {
        protocol: CanonicalNativeConversationProtocol;
    },
): Promise<import('./native-import.js').NativeConversationImportResult> {
    const preflight = preflightJsonInput(options);
    if (!preflight.success)
        throw new NativeConversationImportError(
            'IMPORT_INVALID_OPTIONS',
            'Native import options failed bounded preflight',
            new ConversationValidationError('Invalid options', preflight.diagnostics),
        );
    const { protocol, ...origin } = options;
    switch (protocol) {
        case OPENAI_CHAT_COMPLETIONS_PROTOCOL:
            return importOpenAIChatCompletionsHistory(history, origin);
        case OPENAI_RESPONSES_PROTOCOL:
            return importOpenAIResponsesHistory(history, origin);
        case CLAUDE_MESSAGES_PROTOCOL:
            return importClaudeMessagesHistory(history, origin);
        case GEMINI_GENERATE_CONTENT_PROTOCOL:
            return importGeminiGenerateContentHistory(history, origin);
        case BEDROCK_CONVERSE_PROTOCOL:
            return importBedrockConverseHistory(history, origin);
        default:
            throw new NativeConversationImportError(
                'IMPORT_PROTOCOL_REQUIRED',
                'Native conversation import requires a supported explicit protocol',
            );
    }
}
