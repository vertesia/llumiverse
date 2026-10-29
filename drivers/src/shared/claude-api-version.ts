/**
 * Claude API versions pinned for the Anthropic SDK clients.
 *
 * Anthropic SDK v1 defaults to API version 2026-10-21, which rejects `temperature`, `top_p` and `top_k`
 * and changes several request rules. The drivers stay on the versions the v0 SDKs sent until they move
 * to the new API deliberately. The Bedrock and Vertex clients move the `anthropic-version` header into
 * the request body's `anthropic_version`, so they take their platform's body value instead.
 */

/** First-party Messages API (Anthropic, Foundry, Bedrock Mantle). */
export const CLAUDE_API_VERSION = '2023-06-01';
/** Bedrock InvokeModel `anthropic_version`. */
export const CLAUDE_BEDROCK_API_VERSION = 'bedrock-2023-05-31';
/** Vertex AI `anthropic_version`. */
export const CLAUDE_VERTEX_API_VERSION = 'vertex-2023-10-16';

export function claudeApiVersionHeaders(version: string = CLAUDE_API_VERSION) {
    return { 'anthropic-version': version };
}
