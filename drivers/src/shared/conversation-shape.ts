import { LlumiverseError, type LlumiverseErrorContext, unwrapConversationArray } from '@llumiverse/core';

/**
 * Refuses a conversation that a message-array driver cannot continue: anything but no conversation, an array of
 * messages, or an array wrapped with conversation metadata. Spreading such a value threw a TypeError that the driver
 * reported as a retryable provider failure, so a malformed request was retried and surfaced as an upstream error.
 */
export function assertConversationArray(conversation: unknown, context: LlumiverseErrorContext): void {
    if (conversation === undefined || conversation === null) return;
    if (Array.isArray(conversation) || unwrapConversationArray(conversation)) return;
    const message = `[${context.provider}] The conversation must be an array of messages`;
    const cause = new Error(message);
    cause.name = 'InvalidConversationError';
    throw new LlumiverseError(message, false, context, cause, 400, cause.name);
}

/**
 * Refuses a conversation that a Chat Completions driver cannot continue: anything but no conversation, a stored
 * `{ messages }` container, or a bare message array. A conversation another API wrote — such as a Responses
 * conversation, an array wrapped with conversation metadata — has no `messages`, so reading it would silently
 * continue with no history.
 */
export function assertChatCompletionsConversation(conversation: unknown, context: LlumiverseErrorContext): void {
    if (conversation === undefined || conversation === null || Array.isArray(conversation)) return;
    if (typeof conversation === 'object' && Array.isArray((conversation as { messages?: unknown }).messages)) return;
    const message = `[${context.provider}] The conversation was not written by a Chat Completions driver`;
    const cause = new Error(message);
    cause.name = 'InvalidConversationError';
    throw new LlumiverseError(message, false, context, cause, 400, cause.name);
}
