import {
    APIConnectionError,
    APIConnectionTimeoutError,
    APIError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
} from '@anthropic-ai/sdk/error';
import { LlumiverseError } from '@llumiverse/core';
import { beforeEach, describe, expect, it } from 'vitest';
import { VertexAIDriver } from '../index.js';
import { ClaudeModelDefinition } from './claude.js';

describe('ClaudeModelDefinition Error Handling', () => {
    let modelDef: ClaudeModelDefinition;
    let driver: VertexAIDriver;

    beforeEach(() => {
        modelDef = new ClaudeModelDefinition('claude-haiku-4-5');
        driver = {
            provider: 'vertexai',
            logger: { warn: () => { }, info: () => { }, error: () => { } },
        } as any;
    });

    describe('formatLlumiverseError', () => {
        it('should handle BadRequestError with status code in message', () => {
            const headers = new Headers();
            headers.set('request-id', 'req_test_123');

            const anthropicError = new BadRequestError(
                400,
                {
                    type: 'error',
                    error: {
                        type: 'invalid_request_error',
                        message: 'temperature: range: 0..1'
                    }
                },
                '400 {"type":"error","error":{"type":"invalid_request_error","message":"temperature: range: 0..1"},"request_id":"req_test_123"}',
                headers
            );

            const error = modelDef.formatLlumiverseError(driver, anthropicError, {
                provider: 'vertexai',
                model: 'claude-haiku-4-5',
                operation: 'execute',
            });

            expect(error).toBeInstanceOf(LlumiverseError);
            expect(error.code).toBe(400);
            expect(error.message).toContain('[400]');
            expect(error.message).toContain('temperature: range: 0..1');
            expect(error.message).toContain('invalid_request_error');
            expect(error.message).toContain('req_test_123');
            expect(error.name).toBe('BadRequestError');
            expect(error.retryable).toBe(false);
        });

        it('should handle RateLimitError as retryable', () => {
            const anthropicError = new RateLimitError(
                429,
                { type: 'error', error: { type: 'rate_limit_error', message: 'Rate limit exceeded' } },
                'Rate limit exceeded',
                new Headers()
            );

            const error = modelDef.formatLlumiverseError(driver, anthropicError, {
                provider: 'vertexai',
                model: 'claude-haiku-4-5',
                operation: 'execute',
            });

            expect(error.code).toBe(429);
            expect(error.retryable).toBe(true);
            expect(error.name).toBe('RateLimitError');
        });

        it('should handle InternalServerError as retryable', () => {
            const anthropicError = new InternalServerError(
                500,
                { type: 'error', error: { type: 'internal_error', message: 'Internal server error' } },
                'Internal server error',
                new Headers()
            );

            const error = modelDef.formatLlumiverseError(driver, anthropicError, {
                provider: 'vertexai',
                model: 'claude-haiku-4-5',
                operation: 'execute',
            });

            expect(error.code).toBe(500);
            expect(error.retryable).toBe(true);
            expect(error.name).toBe('InternalServerError');
        });

        it('should handle AuthenticationError as not retryable', () => {
            const anthropicError = new AuthenticationError(
                401,
                { type: 'error', error: { type: 'authentication_error', message: 'Invalid API key' } },
                'Invalid API key',
                new Headers()
            );

            const error = modelDef.formatLlumiverseError(driver, anthropicError, {
                provider: 'vertexai',
                model: 'claude-haiku-4-5',
                operation: 'execute',
            });

            expect(error.code).toBe(401);
            expect(error.retryable).toBe(false);
            expect(error.name).toBe('AuthenticationError');
        });

        it('should handle PermissionDeniedError as not retryable', () => {
            const anthropicError = new PermissionDeniedError(
                403,
                { type: 'error', error: { type: 'permission_error', message: 'Insufficient permissions' } },
                'Insufficient permissions',
                new Headers()
            );

            const error = modelDef.formatLlumiverseError(driver, anthropicError, {
                provider: 'vertexai',
                model: 'claude-haiku-4-5',
                operation: 'execute',
            });

            expect(error.code).toBe(403);
            expect(error.retryable).toBe(false);
            expect(error.name).toBe('PermissionDeniedError');
        });

        it('should handle NotFoundError as not retryable', () => {
            const anthropicError = new NotFoundError(
                404,
                { type: 'error', error: { type: 'not_found_error', message: 'Model not found' } },
                'Model not found',
                new Headers()
            );

            const error = modelDef.formatLlumiverseError(driver, anthropicError, {
                provider: 'vertexai',
                model: 'claude-haiku-4-5',
                operation: 'execute',
            });

            expect(error.code).toBe(404);
            expect(error.retryable).toBe(false);
            expect(error.name).toBe('NotFoundError');
        });

        it('should handle ConflictError as not retryable', () => {
            const anthropicError = new ConflictError(
                409,
                { type: 'error', error: { type: 'conflict_error', message: 'Resource conflict' } },
                'Resource conflict',
                new Headers()
            );

            const error = modelDef.formatLlumiverseError(driver, anthropicError, {
                provider: 'vertexai',
                model: 'claude-haiku-4-5',
                operation: 'execute',
            });

            expect(error.code).toBe(409);
            expect(error.retryable).toBe(false);
            expect(error.name).toBe('ConflictError');
        });

        it('should handle UnprocessableEntityError as not retryable', () => {
            const anthropicError = new UnprocessableEntityError(
                422,
                { type: 'error', error: { type: 'validation_error', message: 'Validation failed' } },
                'Validation failed',
                new Headers()
            );

            const error = modelDef.formatLlumiverseError(driver, anthropicError, {
                provider: 'vertexai',
                model: 'claude-haiku-4-5',
                operation: 'execute',
            });

            expect(error.code).toBe(422);
            expect(error.retryable).toBe(false);
            expect(error.name).toBe('UnprocessableEntityError');
        });

        it('should handle APIConnectionTimeoutError as retryable', () => {
            const anthropicError = new APIConnectionTimeoutError({ message: 'Request timed out' });

            const error = modelDef.formatLlumiverseError(driver, anthropicError, {
                provider: 'vertexai',
                model: 'claude-haiku-4-5',
                operation: 'execute',
            });

            expect(error.code).toBeUndefined();
            expect(error.retryable).toBe(true);
            expect(error.name).toBe('APIConnectionTimeoutError');
        });

        it('should handle APIConnectionError as retryable', () => {
            const anthropicError = new APIConnectionError({ message: 'Connection failed' });

            const error = modelDef.formatLlumiverseError(driver, anthropicError, {
                provider: 'vertexai',
                model: 'claude-haiku-4-5',
                operation: 'execute',
            });

            expect(error.code).toBeUndefined();
            expect(error.retryable).toBe(true);
            expect(error.name).toBe('APIConnectionError');
        });

        it('should extract error type from nested error object', () => {
            const anthropicError = new BadRequestError(
                400,
                {
                    type: 'error',
                    error: {
                        type: 'invalid_request_error',
                        message: 'Missing required field'
                    }
                },
                'Missing required field',
                new Headers()
            );

            const error = modelDef.formatLlumiverseError(driver, anthropicError, {
                provider: 'vertexai',
                model: 'claude-haiku-4-5',
                operation: 'execute',
            });

            expect(error.message).toContain('invalid_request_error');
            expect(error.message).toContain('Missing required field');
        });

        it('should include request ID in message when available', () => {
            const headers = new Headers();
            headers.set('request-id', 'req_vrtx_test123');

            const anthropicError = new BadRequestError(
                400,
                { type: 'error', error: { type: 'invalid_request_error', message: 'Bad request' } },
                'Bad request',
                headers
            );

            const error = modelDef.formatLlumiverseError(driver, anthropicError, {
                provider: 'vertexai',
                model: 'claude-haiku-4-5',
                operation: 'execute',
            });

            expect(error.message).toContain('req_vrtx_test123');
        });

        it('should throw for non-Anthropic errors', () => {
            const regularError = new Error('Regular error');

            expect(() => {
                modelDef.formatLlumiverseError(driver, regularError, {
                    provider: 'vertexai',
                    model: 'claude-haiku-4-5',
                    operation: 'execute',
                });
            }).toThrow('Regular error');
        });

        it('should preserve original error for debugging', () => {
            const anthropicError = new BadRequestError(
                400,
                { type: 'error', error: { type: 'invalid_request_error', message: 'Test error' } },
                'Test error',
                new Headers()
            );

            const error = modelDef.formatLlumiverseError(driver, anthropicError, {
                provider: 'vertexai',
                model: 'claude-haiku-4-5',
                operation: 'execute',
            });

            expect(error.originalError).toBe(anthropicError);
        });
    });

    describe('isClaudeErrorRetryable', () => {
        it('should classify retryable error types correctly', () => {
            const retryableErrors = [
                new RateLimitError(429, {}, 'Rate limit', new Headers()),
                new InternalServerError(500, {}, 'Server error', new Headers()),
                new APIConnectionTimeoutError({ message: 'Timeout' }),
            ];

            for (const error of retryableErrors) {
                const result = (modelDef as any).isClaudeErrorRetryable(error, error.status, undefined);
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
            ];

            for (const error of nonRetryableErrors) {
                const result = (modelDef as any).isClaudeErrorRetryable(error, error.status, undefined);
                expect(result, `${error.constructor.name} should not be retryable`).toBe(false);
            }
        });

        it('should use HTTP status codes when available', () => {
            const apiError = new APIError(429, {}, 'Too many requests', new Headers());
            expect((modelDef as any).isClaudeErrorRetryable(apiError, 429, undefined)).toBe(true);

            const apiError2 = new APIError(408, {}, 'Request timeout', new Headers());
            expect((modelDef as any).isClaudeErrorRetryable(apiError2, 408, undefined)).toBe(true);

            const apiError3 = new APIError(529, {}, 'Overloaded', new Headers());
            expect((modelDef as any).isClaudeErrorRetryable(apiError3, 529, undefined)).toBe(true);

            const apiError4 = new APIError(503, {}, 'Service unavailable', new Headers());
            expect((modelDef as any).isClaudeErrorRetryable(apiError4, 503, undefined)).toBe(true);
        });

        it('should classify 4xx as non-retryable', () => {
            const apiError = new APIError(400, {}, 'Bad request', new Headers());
            expect((modelDef as any).isClaudeErrorRetryable(apiError, 400, undefined)).toBe(false);

            const apiError2 = new APIError(403, {}, 'Forbidden', new Headers());
            expect((modelDef as any).isClaudeErrorRetryable(apiError2, 403, undefined)).toBe(false);
        });

        it('should classify 5xx as retryable', () => {
            const apiError = new APIError(500, {}, 'Internal error', new Headers());
            expect((modelDef as any).isClaudeErrorRetryable(apiError, 500, undefined)).toBe(true);

            const apiError2 = new APIError(502, {}, 'Bad gateway', new Headers());
            expect((modelDef as any).isClaudeErrorRetryable(apiError2, 502, undefined)).toBe(true);
        });

        it('should classify invalid_request_error as non-retryable', () => {
            const apiError = new APIError(400, {}, 'Invalid request', new Headers());
            expect((modelDef as any).isClaudeErrorRetryable(apiError, 400, 'invalid_request_error')).toBe(false);
        });

        it('should classify APIConnectionError (non-timeout) as retryable', () => {
            const connectionError = new APIConnectionError({ message: 'Network failure' });
            expect((modelDef as any).isClaudeErrorRetryable(connectionError, undefined, undefined)).toBe(true);
        });

        it('should return undefined for unknown errors', () => {
            const apiError = new APIError(undefined, {}, 'Unknown error', undefined as any);
            expect((modelDef as any).isClaudeErrorRetryable(apiError, undefined, undefined)).toBeUndefined();
        });
    });

    describe('VertexAIDriver error routing', () => {
        it('should route to Claude-specific error handler', () => {
            const headers = new Headers();
            headers.set('request-id', 'req_test_routing');

            const anthropicError = new RateLimitError(
                429,
                {
                    type: 'error',
                    error: {
                        type: 'rate_limit_error',
                        message: 'Rate limit exceeded'
                    }
                },
                'Rate limit exceeded',
                headers
            );

            const error = modelDef.formatLlumiverseError(driver, anthropicError, {
                provider: 'vertexai',
                model: 'claude-haiku-4-5',
                operation: 'execute',
            });

            expect(error).toBeInstanceOf(LlumiverseError);
            expect(error.code).toBe(429);
            expect(error.retryable).toBe(true);
            expect(error.message).toContain('rate_limit_error');
            expect(error.message).toContain('req_test_routing');
            expect(error.name).toBe('RateLimitError');
        });

        it('should work with different Claude model versions', () => {
            const models = ['claude-haiku-4-5', 'claude-3-7-sonnet-20250219', 'claude-opus-4-5'];

            models.forEach((model) => {
                const modelDef = new ClaudeModelDefinition(model);
                const anthropicError = new BadRequestError(
                    400,
                    { type: 'error', error: { type: 'invalid_request_error', message: 'Invalid parameter' } },
                    'Invalid parameter',
                    new Headers()
                );

                const error = modelDef.formatLlumiverseError(driver, anthropicError, {
                    provider: 'vertexai',
                    model,
                    operation: 'execute',
                });

                expect(error.code).toBe(400);
                expect(error.retryable).toBe(false);
                expect(error.name).toBe('BadRequestError');
            });
        });
    });
});
