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
