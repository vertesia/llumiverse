import type { Logger } from '@llumiverse/core';

export const MODEL_REFUSAL_LOG_MESSAGE = '[Llumiverse] Model response refused or content-filtered';

export interface RefusalLogContext {
    provider: string;
    model: string;
    streaming: boolean;
    /** Finish reason llumiverse returns to the caller. */
    finish_reason?: string;
    /** Finish reason as reported by the provider. */
    provider_finish_reason?: string;
    response_id?: string;
    /**
     * The provider's own refusal or safety object, copied verbatim so new provider fields are kept.
     * This includes the provider's refusal text or explanation, which is the point of the log. Do not
     * pass the prompt, generated answer content, or other request-derived data.
     */
    details?: unknown;
}

/**
 * Record a refusal or content-filter stop for root-cause analysis. A refusal is a normal provider
 * outcome rather than a platform fault, so it is logged at info level.
 */
export function logModelRefusal(logger: Logger | undefined, context: RefusalLogContext): void {
    logger?.info(context, MODEL_REFUSAL_LOG_MESSAGE);
}
