import { type ConversationDocument, parseConversationDocument } from '@llumiverse/conversation';
import {
    exportLegacyOpenAIChatCompletionsConversation,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
} from '../openai/openai-chat-conversation-adapter.js';
import {
    exportLegacyOpenAIResponsesConversation,
    OPENAI_RESPONSES_PROTOCOL,
} from '../openai/openai-responses-conversation-adapter.js';
import {
    CLAUDE_MESSAGES_PROTOCOL,
    exportLegacyClaudeMessagesConversation,
} from '../shared/claude-messages-conversation-adapter.js';

export {
    exportLegacyOpenAIChatCompletionsConversation,
    OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
} from '../openai/openai-chat-conversation-adapter.js';
export {
    exportLegacyOpenAIResponsesConversation,
    OPENAI_RESPONSES_ADAPTER_VERSION,
    OPENAI_RESPONSES_PROTOCOL,
} from '../openai/openai-responses-conversation-adapter.js';
export {
    CLAUDE_MESSAGES_ADAPTER_VERSION,
    CLAUDE_MESSAGES_PROTOCOL,
    exportLegacyClaudeMessagesConversation,
} from '../shared/claude-messages-conversation-adapter.js';

export type CanonicalNativeConversationProtocol =
    | typeof OPENAI_CHAT_COMPLETIONS_PROTOCOL
    | typeof OPENAI_RESPONSES_PROTOCOL
    | typeof CLAUDE_MESSAGES_PROTOCOL;

export type LegacyConversationProjection =
    | ReturnType<typeof exportLegacyOpenAIChatCompletionsConversation>
    | ReturnType<typeof exportLegacyOpenAIResponsesConversation>
    | ReturnType<typeof exportLegacyClaudeMessagesConversation>;

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
        }
        if (turn.provenance.type === 'imported') {
            if (turn.provenance.source === OPENAI_CHAT_COMPLETIONS_PROTOCOL) return OPENAI_CHAT_COMPLETIONS_PROTOCOL;
            if (turn.provenance.source === OPENAI_RESPONSES_PROTOCOL) return OPENAI_RESPONSES_PROTOCOL;
            if (turn.provenance.source === CLAUDE_MESSAGES_PROTOCOL) return CLAUDE_MESSAGES_PROTOCOL;
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
    throw new TypeError('Canonical conversation has no supported native protocol provenance; provide a protocol');
}
