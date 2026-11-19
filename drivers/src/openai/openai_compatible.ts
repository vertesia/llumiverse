import { type DriverOptions, LlumiverseError, type LlumiverseErrorContext } from '@llumiverse/core';
import { AbstractDriver } from '@llumiverse/core/driver';
import {
    APIConnectionError,
    APIConnectionTimeoutError,
    APIError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ContentFilterFinishReasonError,
    InternalServerError,
    LengthFinishReasonError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
} from 'openai/error';

/**
 * OpenAI organization and project are settings of an OpenAI account. Without them, the OpenAI SDK reads
 * OPENAI_ORG_ID and OPENAI_PROJECT_ID from the environment and sends them as headers on every request, so a client
 * for another service (xAI, Azure, any OpenAI-compatible endpoint) would send a server's OpenAI account IDs to it.
 */
export const NON_OPENAI_ACCOUNT_OPTIONS = { organization: null, project: null } as const;

export type CompatibleAPIError = Error & {
    status?: number;
    statusCode?: number;
    code?: string | null;
    param?: string | null;
    type?: string;
    requestID?: string;
    requestId?: string;
};

/** Shared behavior for drivers backed by OpenAI-compatible wire protocols. */
export abstract class OpenAICompatibleDriverBase<
    OptionsT extends DriverOptions = DriverOptions,
    PromptT = unknown,
> extends AbstractDriver<OptionsT, PromptT> {
    public formatLlumiverseError(error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        if (!this.isCompatibleAPIError(error)) {
            return super.formatLlumiverseError(error, context);
        }
        return compatibleAPIErrorToLlumiverseError(error, context, (status, code, type) =>
            this.isOpenAIErrorRetryable(error, status, code, type),
        );
    }

    protected isOpenAIErrorRetryable(
        error: unknown,
        httpStatusCode: number | undefined,
        errorCode: string | null | undefined,
        errorType: string | undefined,
    ): boolean | undefined {
        return isCompatibleErrorRetryable(error, httpStatusCode, errorCode, errorType);
    }

    /** Provider SDKs can extend compatible error recognition without weakening the shared structural checks. */
    protected isCompatibleAPIError(error: unknown): error is CompatibleAPIError {
        return isCompatibleAPIError(error);
    }
}

/**
 * An OpenAI SDK or OpenAI-compatible API error as a LlumiverseError, for code that is not an OpenAI-compatible driver
 * but calls such an endpoint. Undefined for other errors, which the caller formats its own way.
 */
export function formatOpenAICompatibleLlumiverseError(
    error: unknown,
    context: LlumiverseErrorContext,
): LlumiverseError | undefined {
    if (!isCompatibleAPIError(error)) return undefined;
    return compatibleAPIErrorToLlumiverseError(error, context, (status, code, type) =>
        isCompatibleErrorRetryable(error, status, code, type),
    );
}

function compatibleAPIErrorToLlumiverseError(
    error: CompatibleAPIError,
    context: LlumiverseErrorContext,
    retryable: (
        httpStatusCode: number | undefined,
        errorCode: string | null | undefined,
        errorType: string | undefined,
    ) => boolean | undefined,
): LlumiverseError {
    const httpStatusCode = error.status ?? error.statusCode;
    const errorCode = error.code;
    const errorParam = error.param;
    const errorType = error.type;
    let userMessage = error.message || String(error);

    if (httpStatusCode) {
        userMessage = `[${httpStatusCode}] ${userMessage}`;
    }
    if (errorCode && !userMessage.includes(errorCode)) {
        userMessage += ` (code: ${errorCode})`;
    }
    if (errorParam && !userMessage.toLowerCase().includes(errorParam.toLowerCase())) {
        userMessage += ` [param: ${errorParam}]`;
    }
    const requestId = error.requestID ?? error.requestId;
    if (requestId) {
        userMessage += ` (Request ID: ${requestId})`;
    }

    return new LlumiverseError(
        `[${context.provider}] ${userMessage}`,
        retryable(httpStatusCode, errorCode, errorType),
        context,
        error,
        httpStatusCode,
        error.constructor?.name || 'OpenAICompatibleError',
    );
}

function isCompatibleAPIError(error: unknown): error is CompatibleAPIError {
    if (!(error instanceof Error)) {
        return false;
    }
    if (error instanceof APIError || error instanceof OpenAIError) {
        return true;
    }
    const candidate = error as CompatibleAPIError;
    return (
        typeof candidate.status === 'number' ||
        typeof candidate.statusCode === 'number' ||
        typeof candidate.code === 'string' ||
        ['RequestTimeoutError', 'ConnectionError'].includes(error.constructor.name)
    );
}

function isCompatibleErrorRetryable(
    error: unknown,
    httpStatusCode: number | undefined,
    errorCode: string | null | undefined,
    errorType: string | undefined,
): boolean | undefined {
    if (error instanceof RateLimitError) return errorCode !== 'insufficient_quota';

    if (
        error instanceof InternalServerError ||
        error instanceof APIConnectionTimeoutError ||
        (error instanceof Error && ['RequestTimeoutError', 'ConnectionError'].includes(error.constructor.name))
    ) {
        return true;
    }
    if (
        error instanceof BadRequestError ||
        error instanceof AuthenticationError ||
        error instanceof PermissionDeniedError ||
        error instanceof NotFoundError ||
        error instanceof ConflictError ||
        error instanceof UnprocessableEntityError ||
        error instanceof LengthFinishReasonError ||
        error instanceof ContentFilterFinishReasonError
    ) {
        return false;
    }

    if (errorCode) {
        if (['timeout', 'server_error', 'service_unavailable', 'rate_limit_exceeded'].includes(errorCode)) return true;
        if (
            [
                'invalid_api_key',
                'invalid_request_error',
                'model_not_found',
                'insufficient_quota',
                'invalid_model',
            ].includes(errorCode) ||
            errorCode.includes('invalid_')
        ) {
            return false;
        }
    }
    if (errorType === 'invalid_request_error' || errorType === 'authentication_error') return false;

    if (httpStatusCode !== undefined) {
        if ([408, 429, 502, 503, 504, 529].includes(httpStatusCode)) return true;
        if (httpStatusCode >= 500 && httpStatusCode < 600) return true;
        if (httpStatusCode >= 400 && httpStatusCode < 500) return false;
    }
    if (error instanceof APIConnectionError) return true;
    return undefined;
}
