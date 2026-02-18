import {
    AIModel,
    Completion,
    CompletionChunkObject,
    DriverOptions,
    EmbeddingsOptions,
    EmbeddingsResult,
    ExecutionOptions,
    LlumiverseErrorContext,
    ModelSearchPayload,
} from '@llumiverse/common';
import { beforeEach, describe, expect, it } from 'vitest';
import { AbstractDriver } from './Driver.js';
import { LlumiverseError } from './LlumiverseError.js';

// Simple test driver implementation
class TestDriver extends AbstractDriver<DriverOptions, string> {
    provider = 'test-provider';

    async requestTextCompletion(_prompt: string, _options: ExecutionOptions): Promise<Completion> {
        throw new Error('Not implemented');
    }

    async requestTextCompletionStream(_prompt: string, _options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        throw new Error('Not implemented');
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel[]> {
        return [];
    }

    async validateConnection(): Promise<boolean> {
        return true;
    }

    async generateEmbeddings(_options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        throw new Error('Not implemented');
    }
}

describe('AbstractDriver Error Formatting', () => {
    let driver: TestDriver;
    const mockContext: LlumiverseErrorContext = {
        provider: 'test-provider',
        model: 'test-model',
        operation: 'execute',
    };

    beforeEach(() => {
        driver = new TestDriver({});
    });

    describe('isRetryableError', () => {
        describe('HTTP status codes', () => {
            it('should mark 429 as retryable (rate limit)', () => {
                expect(driver['isRetryableError'](429, 'Rate limit exceeded')).toBe(true);
            });

            it('should mark 408 as retryable (timeout)', () => {
                expect(driver['isRetryableError'](408, 'Request timeout')).toBe(true);
            });

            it('should mark 529 as retryable (overloaded)', () => {
                expect(driver['isRetryableError'](529, 'Service overloaded')).toBe(true);
            });

            it('should mark 5xx as retryable (server errors)', () => {
                expect(driver['isRetryableError'](500, 'Internal server error')).toBe(true);
                expect(driver['isRetryableError'](502, 'Bad gateway')).toBe(true);
                expect(driver['isRetryableError'](503, 'Service unavailable')).toBe(true);
                expect(driver['isRetryableError'](504, 'Gateway timeout')).toBe(true);
            });

            it('should mark 4xx as not retryable (except 429, 408)', () => {
                expect(driver['isRetryableError'](400, 'Bad request')).toBe(false);
                expect(driver['isRetryableError'](401, 'Unauthorized')).toBe(false);
                expect(driver['isRetryableError'](403, 'Forbidden')).toBe(false);
                expect(driver['isRetryableError'](404, 'Not found')).toBe(false);
            });

            it('should mark 2xx and 3xx as not retryable', () => {
                expect(driver['isRetryableError'](200, 'OK')).toBe(false);
                expect(driver['isRetryableError'](301, 'Moved permanently')).toBe(false);
            });
        });

        describe('message-based detection', () => {
            it('should detect rate limit in message', () => {
                expect(driver['isRetryableError'](undefined, 'Rate limit exceeded')).toBe(true);
                expect(driver['isRetryableError'](undefined, 'You have hit the rate limit')).toBe(true);
                expect(driver['isRetryableError'](undefined, 'RATE_LIMIT_EXCEEDED')).toBe(true);
            });

            it('should detect timeout in message', () => {
                expect(driver['isRetryableError'](undefined, 'Request timeout')).toBe(true);
                expect(driver['isRetryableError'](undefined, 'Connection timed out')).toBe(true);
                expect(driver['isRetryableError'](undefined, 'TIMEOUT_ERROR')).toBe(true);
            });

            it('should detect retry in message', () => {
                expect(driver['isRetryableError'](undefined, 'Please retry later')).toBe(true);
                expect(driver['isRetryableError'](undefined, 'Retry the request')).toBe(true);
            });

            it('should detect overload in message', () => {
                expect(driver['isRetryableError'](undefined, 'Service overloaded')).toBe(true);
                expect(driver['isRetryableError'](undefined, 'Server is overload')).toBe(true);
                expect(driver['isRetryableError'](undefined, 'System overloaded')).toBe(true);
            });

            it('should detect resource exhausted in message', () => {
                expect(driver['isRetryableError'](undefined, 'Resource exhausted')).toBe(true);
                expect(driver['isRetryableError'](undefined, 'Resources exhausted')).toBe(true);
            });

            it('should detect throttle in message', () => {
                expect(driver['isRetryableError'](undefined, 'Request throttled')).toBe(true);
                expect(driver['isRetryableError'](undefined, 'Throttling exception')).toBe(true);
            });

            it('should detect status codes in message', () => {
                expect(driver['isRetryableError'](undefined, 'Error 429: Too many requests')).toBe(true);
                expect(driver['isRetryableError'](undefined, 'HTTP 529 error')).toBe(true);
            });

            it('should mark unknown messages as undefined (let consumer decide)', () => {
                expect(driver['isRetryableError'](undefined, 'Invalid API key')).toBeUndefined();
                expect(driver['isRetryableError'](undefined, 'Bad request')).toBeUndefined();
                expect(driver['isRetryableError'](undefined, 'Model not found')).toBeUndefined();
            });
        });
    });

    describe('formatLlumiverseError', () => {
        it('should format error with status code', () => {
            const originalError = new Error('Rate limit exceeded');
            (originalError as any).status = 429;

            const formatted = driver['formatLlumiverseError'](originalError, mockContext);

            expect(formatted).toBeInstanceOf(LlumiverseError);
            expect(formatted.code).toBe(429);
            expect(formatted.retryable).toBe(true);
            expect(formatted.message).toContain('[test-provider]');
            expect(formatted.message).toContain('Rate limit exceeded');
            expect(formatted.context).toEqual(mockContext);
            expect(formatted.originalError).toBe(originalError);
        });

        it('should extract status from statusCode property', () => {
            const originalError = new Error('Server error');
            (originalError as any).statusCode = 500;

            const formatted = driver['formatLlumiverseError'](originalError, mockContext);

            expect(formatted.code).toBe(500);
            expect(formatted.retryable).toBe(true);
        });

        it('should extract status from code property', () => {
            const originalError = new Error('Timeout');
            (originalError as any).code = 408;

            const formatted = driver['formatLlumiverseError'](originalError, mockContext);

            expect(formatted.code).toBe(408);
            expect(formatted.retryable).toBe(true);
        });

        it('should use undefined when no status code found', () => {
            const originalError = new Error('Generic error');

            const formatted = driver['formatLlumiverseError'](originalError, mockContext);

            expect(formatted.code).toBeUndefined();
            expect(formatted.retryable).toBeUndefined(); // Unknown retryability
        });

        it('should handle non-Error objects', () => {
            const originalError = 'String error message';

            const formatted = driver['formatLlumiverseError'](originalError, mockContext);

            expect(formatted.message).toContain('String error message');
            expect(formatted.originalError).toBe(originalError);
        });

        it('should preserve provider in message', () => {
            const error = new Error('Test error');

            const formatted = driver['formatLlumiverseError'](error, mockContext);

            expect(formatted.message).toMatch(/^\[test-provider\]/);
        });

        it('should determine retryability based on status and message', () => {
            // Retryable by status
            const retryableError = new Error('Error');
            (retryableError as any).status = 429;
            const formatted1 = driver['formatLlumiverseError'](retryableError, mockContext);
            expect(formatted1.retryable).toBe(true);

            // Not retryable by status
            const nonRetryableError = new Error('Error');
            (nonRetryableError as any).status = 400;
            const formatted2 = driver['formatLlumiverseError'](nonRetryableError, mockContext);
            expect(formatted2.retryable).toBe(false);

            // Retryable by message
            const messageRetryable = new Error('Rate limit exceeded');
            const formatted3 = driver['formatLlumiverseError'](messageRetryable, mockContext);
            expect(formatted3.retryable).toBe(true);
        });
    });

    describe('driver override capability', () => {
        class CustomDriver extends TestDriver {
            public formatLlumiverseError(
                error: unknown,
                context: LlumiverseErrorContext
            ): LlumiverseError {
                // Custom logic: check for specific error type
                if ((error as any).type === 'custom_retryable') {
                    return new LlumiverseError(
                        `[${this.provider}] Custom retryable error`,
                        true,
                        context,
                        error,
                        undefined,
                        'CUSTOM_ERROR'
                    );
                }
                // Fall back to default
                return super.formatLlumiverseError(error, context);
            }
        }

        it('should allow drivers to override error formatting', () => {
            const customDriver = new CustomDriver({});
            const customError = { type: 'custom_retryable', message: 'Custom error' };

            const formatted = customDriver['formatLlumiverseError'](customError, mockContext);

            expect(formatted.name).toBe('CUSTOM_ERROR');
            expect(formatted.code).toBeUndefined();
            expect(formatted.retryable).toBe(true);
            expect(formatted.message).toContain('Custom retryable error');
        });

        it('should fall back to default for non-custom errors', () => {
            const customDriver = new CustomDriver({});
            const regularError = new Error('Regular error');
            (regularError as any).status = 500;

            const formatted = customDriver['formatLlumiverseError'](regularError, mockContext);

            expect(formatted.code).toBe(500);
            expect(formatted.retryable).toBe(true);
        });
    });
});
