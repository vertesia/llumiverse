import { LlumiverseError } from '@llumiverse/core';
import { beforeEach, describe, expect, it } from 'vitest';
import { BedrockDriver } from './index.js';

describe('BedrockDriver Error Handling', () => {
    let driver: BedrockDriver;

    beforeEach(() => {
        driver = new BedrockDriver({ region: 'us-east-1' });
    });

    describe('formatLlumiverseError', () => {
        it('should handle ValidationException with status code in message', () => {
            const awsError = {
                name: 'ValidationException',
                message: "1 validation error detected: Value '32424.0' at 'inferenceConfig.temperature' failed to satisfy constraint: Member must have value less than or equal to 1",
                $fault: 'client',
                $metadata: {
                    httpStatusCode: 400,
                    requestId: 'e3ed39d8-bdf5-40e6-9d2c-9a1cc8323f61',
                    attempts: 1,
                    totalRetryDelay: 0,
                },
            };

            const error = driver.formatLlumiverseError(awsError, {
                provider: 'bedrock',
                model: 'test-model',
                operation: 'execute',
            });

            expect(error).toBeInstanceOf(LlumiverseError);
            expect(error.code).toBe(400);
            expect(error.retryable).toBe(false);
            expect(error.message).toContain('[400]');
            expect(error.message).toContain('validation error detected');
            expect(error.message).toContain('Request ID: e3ed39d8-bdf5-40e6-9d2c-9a1cc8323f61');
            expect(error.context.provider).toBe('bedrock');
        });

        it('should handle ThrottlingException as retryable', () => {
            const awsError = {
                name: 'ThrottlingException',
                message: 'The number of requests exceeds the limit',
                $fault: 'client',
                $metadata: {
                    httpStatusCode: 429,
                    requestId: 'abc123',
                },
            };

            const error = driver.formatLlumiverseError(awsError, {
                provider: 'bedrock',
                model: 'test-model',
                operation: 'execute',
            });

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(429);
            expect(error.message).toContain('[429]');
            expect(error.message).toContain('Request ID: abc123');
        });

        it('should handle InternalServerException as retryable', () => {
            const awsError = {
                name: 'InternalServerException',
                message: 'An internal server error occurred',
                $fault: 'server',
                $metadata: {
                    httpStatusCode: 500,
                    requestId: 'server-error-123',
                },
            };

            const error = driver.formatLlumiverseError(awsError, {
                provider: 'bedrock',
                model: 'test-model',
                operation: 'execute',
            });

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(500);
            expect(error.message).toContain('[500]');
        });

        it('should handle ServiceUnavailableException as retryable', () => {
            const awsError = {
                name: 'ServiceUnavailableException',
                message: 'Service is temporarily unavailable',
                $fault: 'server',
                $metadata: {
                    httpStatusCode: 503,
                    requestId: 'unavail-123',
                },
            };

            const error = driver.formatLlumiverseError(awsError, {
                provider: 'bedrock',
                model: 'test-model',
                operation: 'execute',
            });

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(503);
        });

        it('should handle AccessDeniedException as not retryable', () => {
            const awsError = {
                name: 'AccessDeniedException',
                message: 'Access denied',
                $fault: 'client',
                $metadata: {
                    httpStatusCode: 403,
                    requestId: 'access-denied-123',
                },
            };

            const error = driver.formatLlumiverseError(awsError, {
                provider: 'bedrock',
                model: 'test-model',
                operation: 'execute',
            });

            expect(error.retryable).toBe(false);
            expect(error.code).toBe(403);
        });

        it('should handle ResourceNotFoundException as not retryable', () => {
            const awsError = {
                name: 'ResourceNotFoundException',
                message: 'Resource not found',
                $fault: 'client',
                $metadata: {
                    httpStatusCode: 404,
                    requestId: 'not-found-123',
                },
            };

            const error = driver.formatLlumiverseError(awsError, {
                provider: 'bedrock',
                model: 'test-model',
                operation: 'execute',
            });

            expect(error.retryable).toBe(false);
            expect(error.code).toBe(404);
        });

        it('should handle ServiceQuotaExceededException as retryable', () => {
            const awsError = {
                name: 'ServiceQuotaExceededException',
                message: 'Service quota exceeded',
                $fault: 'client',
                $metadata: {
                    httpStatusCode: 402,
                    requestId: 'quota-123',
                },
            };

            const error = driver.formatLlumiverseError(awsError, {
                provider: 'bedrock',
                model: 'test-model',
                operation: 'execute',
            });

            expect(error.retryable).toBe(true);
        });

        it('should handle ConflictException as not retryable', () => {
            const awsError = {
                name: 'ConflictException',
                message: 'Conflict',
                $fault: 'client',
                $metadata: {
                    httpStatusCode: 409,
                    requestId: 'conflict-123',
                },
            };

            const error = driver.formatLlumiverseError(awsError, {
                provider: 'bedrock',
                model: 'test-model',
                operation: 'execute',
            });

            expect(error.retryable).toBe(false);
            expect(error.code).toBe(409);
        });

        it('should handle ResourceInUseException as not retryable', () => {
            const awsError = {
                name: 'ResourceInUseException',
                message: 'Resource in use',
                $fault: 'client',
                $metadata: {
                    httpStatusCode: 400,
                    requestId: 'in-use-123',
                },
            };

            const error = driver.formatLlumiverseError(awsError, {
                provider: 'bedrock',
                model: 'test-model',
                operation: 'execute',
            });

            expect(error.retryable).toBe(false);
        });

        it('should handle error without status code using error name', () => {
            const awsError = {
                name: 'ThrottlingException',
                message: 'Rate limited',
                $fault: 'client',
                $metadata: {
                    requestId: 'no-status-123',
                },
            };

            const error = driver.formatLlumiverseError(awsError, {
                provider: 'bedrock',
                model: 'test-model',
                operation: 'execute',
            });

            expect(error.code).toBeUndefined(); // No status code available
            expect(error.name).toBe('ThrottlingException'); // Error name preserved
            expect(error.retryable).toBe(true);
            expect(error.message).not.toContain('[ThrottlingException]'); // status code format only for numbers
            expect(error.message).toContain('Rate limited');
        });

        it('should fall back to fault type for unknown errors', () => {
            const serverError = {
                name: 'UnknownServerError',
                message: 'Unknown error',
                $fault: 'server',
                $metadata: {
                    httpStatusCode: 599,
                },
            };

            const error = driver.formatLlumiverseError(serverError, {
                provider: 'bedrock',
                model: 'test-model',
                operation: 'execute',
            });

            expect(error.retryable).toBe(true); // 5xx is retryable

            const clientError = {
                name: 'UnknownClientError',
                message: 'Unknown error',
                $fault: 'client',
                $metadata: {
                    httpStatusCode: 499,
                },
            };

            const error2 = driver.formatLlumiverseError(clientError, {
                provider: 'bedrock',
                model: 'test-model',
                operation: 'execute',
            });

            expect(error2.retryable).toBe(false); // 4xx is not retryable
        });

        it('should handle non-AWS errors by delegating to parent', () => {
            const regularError = new Error('Regular error');

            const error = driver.formatLlumiverseError(regularError, {
                provider: 'bedrock',
                model: 'test-model',
                operation: 'execute',
            });

            expect(error).toBeInstanceOf(LlumiverseError);
            expect(error.code).toBeUndefined(); // No numeric status code available
            expect(error.retryable).toBeUndefined(); // Unknown errors - let consumer decide
        });

        it('should preserve original error for debugging', () => {
            const awsError = {
                name: 'ValidationException',
                message: 'Validation failed',
                $fault: 'client',
                $metadata: {
                    httpStatusCode: 400,
                    requestId: 'test-123',
                },
            };

            const error = driver.formatLlumiverseError(awsError, {
                provider: 'bedrock',
                model: 'test-model',
                operation: 'execute',
            });

            expect(error.originalError).toBe(awsError);
            expect((error.originalError as any).$metadata.requestId).toBe('test-123');
        });
    });

    describe('isBedrockErrorRetryable', () => {
        it('should classify retryable errors correctly', () => {
            const retryableErrors = [
                'ThrottlingException',
                'ServiceUnavailableException',
                'InternalServerException',
                'ServiceQuotaExceededException',
            ];

            retryableErrors.forEach((errorName) => {
                const result = (driver as any).isBedrockErrorRetryable(errorName, undefined, undefined);
                expect(result, `${errorName} should be retryable`).toBe(true);
            });
        });

        it('should classify non-retryable errors correctly', () => {
            const nonRetryableErrors = [
                'ValidationException',
                'AccessDeniedException',
                'ResourceNotFoundException',
                'ConflictException',
                'ResourceInUseException',
                'TooManyTagsException',
            ];

            nonRetryableErrors.forEach((errorName) => {
                const result = (driver as any).isBedrockErrorRetryable(errorName, undefined, undefined);
                expect(result, `${errorName} should not be retryable`).toBe(false);
            });
        });

        it('should use HTTP status codes when available', () => {
            expect((driver as any).isBedrockErrorRetryable('UnknownError', 429, undefined)).toBe(true);
            expect((driver as any).isBedrockErrorRetryable('UnknownError', 408, undefined)).toBe(true);
            expect((driver as any).isBedrockErrorRetryable('UnknownError', 529, undefined)).toBe(true);
            expect((driver as any).isBedrockErrorRetryable('UnknownError', 500, undefined)).toBe(true);
            expect((driver as any).isBedrockErrorRetryable('UnknownError', 503, undefined)).toBe(true);
            expect((driver as any).isBedrockErrorRetryable('UnknownError', 400, undefined)).toBe(false);
            expect((driver as any).isBedrockErrorRetryable('UnknownError', 403, undefined)).toBe(false);
            expect((driver as any).isBedrockErrorRetryable('UnknownError', 404, undefined)).toBe(false);
        });

        it('should use fault type as fallback', () => {
            expect((driver as any).isBedrockErrorRetryable('UnknownError', undefined, 'server')).toBe(true);
            expect((driver as any).isBedrockErrorRetryable('UnknownError', undefined, 'client')).toBe(false);
        });
    });
});
