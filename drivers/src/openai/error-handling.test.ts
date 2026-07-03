import { LlumiverseError, Providers } from '@llumiverse/core';
import OpenAI from 'openai';
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
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
} from 'openai/error';
import { beforeEach, describe, expect, it } from 'vitest';
import { exposePrivate, getProp } from '../../test/__helpers__/test-utils.js';
import { OpenAIResponsesDriverBase } from './index.js';

type OpenAIResponsesDriverBaseInternals = {
    isOpenAIErrorRetryable: (
        error: unknown,
        httpStatusCode: number | undefined,
        errorCode: string | null | undefined,
        errorType: string | undefined,
    ) => boolean | undefined;
};

// Test implementation of OpenAIResponsesDriverBase
class TestOpenAIResponsesDriver extends OpenAIResponsesDriverBase {
    provider: Providers.openai = Providers.openai;
    service: OpenAI;

    constructor() {
        super({});
        this.service = new OpenAI({ apiKey: 'test-key' });
    }
}

describe('OpenAIResponsesDriverBase usage mapping', () => {
    it('maps Together flat cached_tokens usage into prompt_cached', () => {
        const driver = new TestOpenAIResponsesDriver();
        const response = {
            status: 'completed',
            output: [
                {
                    type: 'message',
                    role: 'assistant',
                    content: [{ type: 'output_text', text: 'ok', annotations: [] }],
                },
            ],
            usage: {
                input_tokens: 100,
                output_tokens: 20,
                total_tokens: 120,
                cached_tokens: 45,
            },
        } as unknown as OpenAI.Responses.Response;

        const completion = driver.extractDataFromResponse({ model: 'test-model' }, response);

        expect(completion.token_usage).toEqual({
            prompt: 100,
            result: 20,
            total: 120,
            prompt_cached: 45,
            prompt_new: 55,
        });
    });
});

describe('OpenAIResponsesDriverBase Error Handling', () => {
    let driver: TestOpenAIResponsesDriver;

    beforeEach(() => {
        driver = new TestOpenAIResponsesDriver();
    });

    describe('formatLlumiverseError', () => {
        it('should handle BadRequestError with status code and parameter', () => {
            const headers = new Headers();
            headers.set('x-request-id', 'req_test_123');

            const openaiError = new BadRequestError(
                400,
                {
                    message:
                        "Invalid 'temperature': decimal above maximum value. Expected a value <= 2, but got 31313 instead.",
                    type: 'invalid_request_error',
                    param: 'temperature',
                    code: 'decimal_above_max_value',
                },
                "Invalid 'temperature': decimal above maximum value. Expected a value <= 2, but got 31313 instead.",
                headers,
            );

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error).toBeInstanceOf(LlumiverseError);
            expect(error.code).toBe(400);
            expect(error.message).toContain('[400]');
            expect(error.message).toContain('temperature');
            expect(error.message).toContain('decimal_above_max_value');
            expect(error.message).toContain('req_test_123');
            expect(error.name).toBe('BadRequestError');
            expect(error.retryable).toBe(false);
        });

        it('should handle RateLimitError as retryable', () => {
            const headers = new Headers();
            const openaiError = new RateLimitError(
                429,
                {
                    message: 'Rate limit exceeded',
                    type: 'rate_limit_error',
                    code: 'rate_limit_exceeded',
                },
                'Rate limit exceeded',
                headers,
            );

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error.code).toBe(429);
            expect(error.retryable).toBe(true);
            expect(error.name).toBe('RateLimitError');
        });

        it('should handle InternalServerError as retryable', () => {
            const openaiError = new InternalServerError(
                500,
                {
                    message: 'Internal server error',
                    type: 'server_error',
                    code: 'server_error',
                },
                'Internal server error',
                new Headers(),
            );

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error.code).toBe(500);
            expect(error.retryable).toBe(true);
            expect(error.name).toBe('InternalServerError');
        });

        it('should handle AuthenticationError as not retryable', () => {
            const openaiError = new AuthenticationError(
                401,
                {
                    message: 'Invalid API key',
                    type: 'authentication_error',
                    code: 'invalid_api_key',
                },
                'Invalid API key',
                new Headers(),
            );

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error.code).toBe(401);
            expect(error.retryable).toBe(false);
            expect(error.name).toBe('AuthenticationError');
        });

        it('should handle PermissionDeniedError as not retryable', () => {
            const openaiError = new PermissionDeniedError(
                403,
                {
                    message: 'Insufficient permissions',
                    type: 'permission_error',
                },
                'Insufficient permissions',
                new Headers(),
            );

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error.code).toBe(403);
            expect(error.retryable).toBe(false);
            expect(error.name).toBe('PermissionDeniedError');
        });

        it('should handle NotFoundError as not retryable', () => {
            const openaiError = new NotFoundError(
                404,
                {
                    message: 'Model not found',
                    type: 'invalid_request_error',
                    code: 'model_not_found',
                },
                'Model not found',
                new Headers(),
            );

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-invalid',
                operation: 'execute',
            });

            expect(error.code).toBe(404);
            expect(error.retryable).toBe(false);
            expect(error.name).toBe('NotFoundError');
        });

        it('should handle ConflictError as not retryable', () => {
            const openaiError = new ConflictError(
                409,
                {
                    message: 'Resource conflict',
                    type: 'conflict_error',
                },
                'Resource conflict',
                new Headers(),
            );

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error.code).toBe(409);
            expect(error.retryable).toBe(false);
            expect(error.name).toBe('ConflictError');
        });

        it('should handle UnprocessableEntityError as not retryable', () => {
            const openaiError = new UnprocessableEntityError(
                422,
                {
                    message: 'Validation failed',
                    type: 'validation_error',
                },
                'Validation failed',
                new Headers(),
            );

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error.code).toBe(422);
            expect(error.retryable).toBe(false);
            expect(error.name).toBe('UnprocessableEntityError');
        });

        it('should handle APIConnectionTimeoutError as retryable', () => {
            const openaiError = new APIConnectionTimeoutError({ message: 'Request timed out' });

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error.code).toBeUndefined();
            expect(error.retryable).toBe(true);
            expect(error.name).toBe('APIConnectionTimeoutError');
        });

        it('should handle APIConnectionError as retryable', () => {
            const openaiError = new APIConnectionError({ message: 'Connection failed' });

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error.code).toBeUndefined();
            expect(error.retryable).toBe(true);
            expect(error.name).toBe('APIConnectionError');
        });

        it('should handle LengthFinishReasonError as not retryable', () => {
            const openaiError = new LengthFinishReasonError();

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error.retryable).toBe(false);
            expect(error.name).toBe('LengthFinishReasonError');
        });

        it('should handle ContentFilterFinishReasonError as not retryable', () => {
            const openaiError = new ContentFilterFinishReasonError();

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error.retryable).toBe(false);
            expect(error.name).toBe('ContentFilterFinishReasonError');
        });

        it('should include error code in message', () => {
            const headers = new Headers();
            const openaiError = new BadRequestError(
                400,
                {
                    message: 'Invalid parameter',
                    code: 'invalid_parameter',
                },
                'Invalid parameter',
                headers,
            );

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error.message).toContain('invalid_parameter');
        });

        it('should include parameter in message when available', () => {
            const headers = new Headers();
            const openaiError = new BadRequestError(
                400,
                {
                    message: 'Invalid value',
                    param: 'max_tokens',
                    code: 'invalid_value',
                },
                'Invalid value',
                headers,
            );

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error.message).toContain('max_tokens');
        });

        it('should include request ID when available', () => {
            const headers = new Headers();
            headers.set('x-request-id', 'req_xyz789');

            const openaiError = new BadRequestError(400, { message: 'Bad request' }, 'Bad request', headers);

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error.message).toContain('req_xyz789');
        });

        it('should throw for non-OpenAI errors', () => {
            const regularError = new Error('Regular error');

            expect(() => {
                driver.formatLlumiverseError(regularError, {
                    provider: 'openai',
                    model: 'gpt-4',
                    operation: 'execute',
                });
            }).toThrow('Regular error');
        });

        it('should preserve original error for debugging', () => {
            const openaiError = new BadRequestError(400, { message: 'Test error' }, 'Test error', new Headers());

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error.originalError).toBe(openaiError);
        });
    });

    describe('isOpenAIErrorRetryable', () => {
        it('should classify retryable error types correctly', () => {
            const retryableErrors = [
                new RateLimitError(429, {}, 'Rate limit', new Headers()),
                new InternalServerError(500, {}, 'Server error', new Headers()),
                new APIConnectionTimeoutError({ message: 'Timeout' }),
            ];

            for (const error of retryableErrors) {
                const result = exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    error,
                    error.status,
                    null,
                    undefined,
                );
                expect(result, `${error.constructor.name} should be retryable`).toBe(true);
            }
        });

        it('should classify non-retryable error types correctly', () => {
            const nonRetryableErrors = [
                new BadRequestError(400, {}, 'Bad request', new Headers()),
                new AuthenticationError(401, {}, 'Auth error', new Headers()),
                new PermissionDeniedError(403, {}, 'Permission denied', new Headers()),
                new NotFoundError(404, {}, 'Not found', new Headers()),
                new ConflictError(409, {}, 'Conflict', new Headers()),
                new UnprocessableEntityError(422, {}, 'Validation error', new Headers()),
                new LengthFinishReasonError(),
                new ContentFilterFinishReasonError(),
            ];

            for (const error of nonRetryableErrors) {
                const status = getProp<number>(error, 'status');
                const result = exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    error,
                    status,
                    null,
                    undefined,
                );
                expect(result, `${error.constructor.name} should not be retryable`).toBe(false);
            }
        });

        it('should classify retryable error codes correctly', () => {
            const apiError = new APIError(undefined, {}, 'Error', new Headers());

            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    undefined,
                    'timeout',
                    undefined,
                ),
            ).toBe(true);
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    undefined,
                    'server_error',
                    undefined,
                ),
            ).toBe(true);
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    undefined,
                    'service_unavailable',
                    undefined,
                ),
            ).toBe(true);
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    undefined,
                    'rate_limit_exceeded',
                    undefined,
                ),
            ).toBe(true);
        });

        it('should classify non-retryable error codes correctly', () => {
            const apiError = new APIError(undefined, {}, 'Error', new Headers());

            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    undefined,
                    'invalid_api_key',
                    undefined,
                ),
            ).toBe(false);
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    undefined,
                    'invalid_request_error',
                    undefined,
                ),
            ).toBe(false);
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    undefined,
                    'model_not_found',
                    undefined,
                ),
            ).toBe(false);
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    undefined,
                    'insufficient_quota',
                    undefined,
                ),
            ).toBe(false);
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    undefined,
                    'invalid_model',
                    undefined,
                ),
            ).toBe(false);
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    undefined,
                    'invalid_parameter',
                    undefined,
                ),
            ).toBe(false);
        });

        it('should classify error types correctly', () => {
            const apiError = new APIError(undefined, {}, 'Error', new Headers());

            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    undefined,
                    null,
                    'invalid_request_error',
                ),
            ).toBe(false);
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    undefined,
                    null,
                    'authentication_error',
                ),
            ).toBe(false);
        });

        it('should use HTTP status codes when available', () => {
            const apiError = new APIError(429, {}, 'Too many requests', new Headers());
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    429,
                    null,
                    undefined,
                ),
            ).toBe(true);

            const apiError2 = new APIError(408, {}, 'Request timeout', new Headers());
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError2,
                    408,
                    null,
                    undefined,
                ),
            ).toBe(true);

            const apiError3 = new APIError(502, {}, 'Bad gateway', new Headers());
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError3,
                    502,
                    null,
                    undefined,
                ),
            ).toBe(true);

            const apiError4 = new APIError(503, {}, 'Service unavailable', new Headers());
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError4,
                    503,
                    null,
                    undefined,
                ),
            ).toBe(true);

            const apiError5 = new APIError(504, {}, 'Gateway timeout', new Headers());
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError5,
                    504,
                    null,
                    undefined,
                ),
            ).toBe(true);

            const apiError6 = new APIError(529, {}, 'Overloaded', new Headers());
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError6,
                    529,
                    null,
                    undefined,
                ),
            ).toBe(true);
        });

        it('should classify 4xx as non-retryable', () => {
            const apiError = new APIError(400, {}, 'Bad request', new Headers());
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    400,
                    null,
                    undefined,
                ),
            ).toBe(false);

            const apiError2 = new APIError(403, {}, 'Forbidden', new Headers());
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError2,
                    403,
                    null,
                    undefined,
                ),
            ).toBe(false);
        });

        it('should classify 5xx as retryable', () => {
            const apiError = new APIError(500, {}, 'Internal error', new Headers());
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    500,
                    null,
                    undefined,
                ),
            ).toBe(true);

            const apiError2 = new APIError(502, {}, 'Bad gateway', new Headers());
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError2,
                    502,
                    null,
                    undefined,
                ),
            ).toBe(true);
        });

        it('should classify APIConnectionError (non-timeout) as retryable', () => {
            const connectionError = new APIConnectionError({ message: 'Network failure' });
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    connectionError,
                    undefined,
                    null,
                    undefined,
                ),
            ).toBe(true);
        });

        it('should return undefined for unknown errors', () => {
            const apiError = new APIError(undefined, {}, 'Unknown error', undefined);
            expect(
                exposePrivate<OpenAIResponsesDriverBaseInternals>(driver).isOpenAIErrorRetryable(
                    apiError,
                    undefined,
                    null,
                    undefined,
                ),
            ).toBeUndefined();
        });
    });

    describe('Cross-provider compatibility', () => {
        it('should work with Azure OpenAI provider context', () => {
            const openaiError = new RateLimitError(
                429,
                { message: 'Rate limit exceeded' },
                'Rate limit exceeded',
                new Headers(),
            );

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'azure_openai',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error).toBeInstanceOf(LlumiverseError);
            expect(error.code).toBe(429);
            expect(error.retryable).toBe(true);
            expect(error.message).toContain('[azure_openai]');
        });

        it('should work with xAI provider context', () => {
            const openaiError = new BadRequestError(
                400,
                { message: 'Invalid parameter' },
                'Invalid parameter',
                new Headers(),
            );

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'xai',
                model: 'grok-2',
                operation: 'execute',
            });

            expect(error).toBeInstanceOf(LlumiverseError);
            expect(error.code).toBe(400);
            expect(error.retryable).toBe(false);
            expect(error.message).toContain('[xai]');
        });

        it('should work with Azure Foundry provider context', () => {
            const openaiError = new InternalServerError(
                500,
                { message: 'Server error' },
                'Server error',
                new Headers(),
            );

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'azure_foundry',
                model: 'gpt-4',
                operation: 'execute',
            });

            expect(error).toBeInstanceOf(LlumiverseError);
            expect(error.code).toBe(500);
            expect(error.retryable).toBe(true);
            expect(error.message).toContain('[azure_foundry]');
        });

        it('should work with OpenAI-compatible provider context', () => {
            const openaiError = new AuthenticationError(
                401,
                { message: 'Invalid API key' },
                'Invalid API key',
                new Headers(),
            );

            const error = driver.formatLlumiverseError(openaiError, {
                provider: 'openai_compatible',
                model: 'custom-model',
                operation: 'execute',
            });

            expect(error).toBeInstanceOf(LlumiverseError);
            expect(error.code).toBe(401);
            expect(error.retryable).toBe(false);
            expect(error.message).toContain('[openai_compatible]');
        });
    });
});
