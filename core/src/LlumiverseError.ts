import { LlumiverseErrorContext } from "@llumiverse/common";

/**
 * Standardized error class for Llumiverse driver errors.
 * 
 * Normalizes errors from different LLM providers (OpenAI, Anthropic, Bedrock, VertexAI, etc.)
 * into a consistent format. The primary value is the `retryable` flag, which enables upstream
 * consumers to implement smart retry logic.
 * 
 * @example
 * ```typescript
 * try {
 *   const result = await driver.execute(segments, options);
 * } catch (error) {
 *   if (LlumiverseError.isLlumiverseError(error)) {
 *     console.log(`Provider: ${error.context.provider}`);
 *     console.log(`Model: ${error.context.model}`);
 *     console.log(`Retryable: ${error.retryable}`);
 *     
 *     if (error.retryable) {
 *       // Implement retry logic with exponential backoff
 *       await retryWithBackoff(() => driver.execute(segments, options));
 *     } else {
 *       // Handle non-retryable error (e.g., invalid API key, malformed request)
 *       logError(error);
 *     }
 *   }
 *   throw error;
 * }
 * ```
 */
export class LlumiverseError extends Error {
    /** 
     * HTTP status code (e.g., 429, 500) if available.
     * Undefined if the error doesn't have a numeric status code.
     */
    readonly code?: number;

    /**
     * Provider-specific error name/type (e.g., "ThrottlingException", "ValidationException").
     * Optional - used to preserve the semantic error type from the provider SDK.
     */
    readonly name: string;

    /** 
     * Whether this error is retryable.
     * True for transient errors (rate limits, timeouts, server errors).
     * False for permanent errors (auth failures, invalid requests, malformed schemas).
     */
    readonly retryable: boolean;

    /**
     * Context about where and how the error occurred.
     * Includes provider, model, operation type, and optionally the prompt.
     */
    readonly context: LlumiverseErrorContext;

    /**
     * The original error from the provider SDK.
     * Preserved for debugging and detailed error inspection.
     */
    readonly originalError: unknown;

    constructor(
        message: string,
        retryable: boolean,
        context: LlumiverseErrorContext,
        originalError: unknown,
        code?: number,
        name?: string
    ) {
        super(message);
        this.name = name || 'LlumiverseError';
        this.code = code;
        this.retryable = retryable;
        this.context = context;
        this.originalError = originalError;

        // Preserve stack trace from original error if available
        if (originalError instanceof Error && originalError.stack) {
            this.stack = originalError.stack;
        }
    }

    /**
     * Serialize the error to JSON for logging or transmission.
     * Includes all error properties except the original error object itself.
     */
    toJSON(): Record<string, unknown> {
        return {
            name: this.name,
            message: this.message,
            code: this.code,
            retryable: this.retryable,
            context: this.context,
            stack: this.stack,
            // Include original error message if available
            originalErrorMessage: this.originalError instanceof Error
                ? this.originalError.message
                : String(this.originalError),
        };
    }

    /**
     * Type guard to check if an error is a LlumiverseError.
     * Useful for conditional error handling.
     * 
     * @param error - The error to check
     * @returns True if the error is a LlumiverseError
     */
    static isLlumiverseError(error: unknown): error is LlumiverseError {
        return error instanceof LlumiverseError;
    }
}
