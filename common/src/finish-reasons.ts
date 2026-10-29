/**
 * Provider finish reasons that can end a response in the middle of a tool call: a token limit, a refusal
 * (Claude `refusal`), or a content filter (OpenAI `content_filter`, Bedrock `content_filtered` and
 * `guardrail_intervened`). Drivers pass these through unnormalized apart from `length`.
 */
const CUT_OFF_FINISH_REASONS = new Set([
    'length',
    'refusal',
    'content_filter',
    'content_filtered',
    'guardrail_intervened',
]);

/** Whether a finish reason means the response may have stopped in the middle of its last tool call. */
export function isCutOffFinishReason(finishReason: string | null | undefined): boolean {
    return !!finishReason && CUT_OFF_FINISH_REASONS.has(finishReason);
}
