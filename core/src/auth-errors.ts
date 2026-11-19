/**
 * OAuth 2.0 token errors that a retry cannot fix (RFC 6749 section 5.2). `invalid_request` is left out because it is
 * also part of OpenAI's `invalid_request_error`, which is not an auth failure.
 */
const PERMANENT_OAUTH_ERROR_CODES = [
    'invalid_grant',
    'invalid_client',
    'unauthorized_client',
    'unsupported_grant_type',
    'invalid_scope',
];

/**
 * Whether an error message reports an auth failure that a retry cannot fix, even without a numeric status code: an
 * OAuth token error, such as google-auth-library's `invalid_grant`, or a workload identity issuer that cannot be reached.
 */
export function isPermanentAuthErrorMessage(message: string): boolean {
    const lowerMessage = message.toLowerCase();
    return (
        PERMANENT_OAUTH_ERROR_CODES.some((code) => lowerMessage.includes(code)) ||
        lowerMessage.includes("credential's issuer")
    );
}
