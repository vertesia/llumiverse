import { LlumiverseErrorContext, LlumiverseError } from '@llumiverse/common';
import { describe, expect, it } from 'vitest';

describe('LlumiverseError', () => {
    const mockContext: LlumiverseErrorContext = {
        provider: 'test-provider',
        model: 'test-model',
        operation: 'execute' as const,
    };

    describe('constructor', () => {
        it('should create an error with all properties', () => {
            const originalError = new Error('Original error');
            const error = new LlumiverseError(
                'Test error message',
                true,
                mockContext,
                originalError,
                429,
                'RateLimitError'
            );

            expect(error).toBeInstanceOf(Error);
            expect(error).toBeInstanceOf(LlumiverseError);
            expect(error.name).toBe('RateLimitError');
            expect(error.message).toBe('Test error message');
            expect(error.code).toBe(429);
            expect(error.retryable).toBe(true);
            expect(error.context).toEqual(mockContext);
            expect(error.originalError).toBe(originalError);
        });

        it('should preserve stack trace from original error', () => {
            const originalError = new Error('Original error');
            const originalStack = originalError.stack;

            const error = new LlumiverseError(
                'Wrapped error',
                true,
                mockContext,
                originalError,
                500
            );

            expect(error.stack).toBe(originalStack);
        });

        it('should handle undefined code', () => {
            const error = new LlumiverseError(
                'Test error',
                true,
                mockContext,
                new Error('Unknown error')
            );

            expect(error.code).toBeUndefined();
            expect(error.name).toBe('LlumiverseError'); // Default name
        });

        it('should handle non-retryable errors', () => {
            const error = new LlumiverseError(
                'Auth error',
                false,
                mockContext,
                new Error('Unauthorized'),
                401,
                'AuthenticationError'
            );

            expect(error.retryable).toBe(false);
            expect(error.name).toBe('AuthenticationError');
        });
    });

    describe('toJSON', () => {
        it('should serialize to JSON', () => {
            const originalError = new Error('Original error');
            const error = new LlumiverseError(
                'Test error',
                true,
                mockContext,
                originalError,
                429,
                'RateLimitError'
            );

            const json = error.toJSON();

            expect(json).toHaveProperty('name', 'RateLimitError');
            expect(json).toHaveProperty('message', 'Test error');
            expect(json).toHaveProperty('code', 429);
            expect(json).toHaveProperty('retryable', true);
            expect(json).toHaveProperty('context', mockContext);
            expect(json).toHaveProperty('stack');
            expect(json).toHaveProperty('originalErrorMessage', 'Original error');
        });

        it('should handle non-Error original error', () => {
            const error = new LlumiverseError(
                'Test error',
                true,
                mockContext,
                'string error',
                500
            );

            const json = error.toJSON();
            expect(json.originalErrorMessage).toBe('string error');
        });
    });

    describe('isLlumiverseError', () => {
        it('should return true for LlumiverseError instances', () => {
            const error = new LlumiverseError(
                'Test error',
                true,
                mockContext,
                new Error('Original'),
                500
            );

            expect(LlumiverseError.isLlumiverseError(error)).toBe(true);
        });

        it('should return false for regular Error', () => {
            const error = new Error('Regular error');
            expect(LlumiverseError.isLlumiverseError(error)).toBe(false);
        });

        it('should return false for non-error values', () => {
            expect(LlumiverseError.isLlumiverseError(null)).toBe(false);
            expect(LlumiverseError.isLlumiverseError(undefined)).toBe(false);
            expect(LlumiverseError.isLlumiverseError('string')).toBe(false);
            expect(LlumiverseError.isLlumiverseError(123)).toBe(false);
            expect(LlumiverseError.isLlumiverseError({})).toBe(false);
        });
    });

    describe('error classification examples', () => {
        it('should handle rate limit errors', () => {
            const error = new LlumiverseError(
                'Rate limit exceeded',
                true,
                { ...mockContext, operation: 'execute' as const },
                new Error('Too many requests'),
                429
            );

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(429);
        });

        it('should handle server errors', () => {
            const error = new LlumiverseError(
                'Internal server error',
                true,
                { ...mockContext, operation: 'stream' as const },
                new Error('Server error'),
                500
            );

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(500);
        });

        it('should handle authentication errors', () => {
            const error = new LlumiverseError(
                'Invalid API key',
                false,
                mockContext,
                new Error('Unauthorized'),
                401
            );

            expect(error.retryable).toBe(false);
            expect(error.code).toBe(401);
        });

        it('should handle validation errors', () => {
            const error = new LlumiverseError(
                'Invalid request',
                false,
                mockContext,
                new Error('Bad request'),
                400
            );

            expect(error.retryable).toBe(false);
            expect(error.code).toBe(400);
        });

        it('should handle timeout errors', () => {
            const error = new LlumiverseError(
                'Request timeout',
                true,
                mockContext,
                new Error('Timeout'),
                408
            );

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(408);
        });

        it('should handle service overloaded errors', () => {
            const error = new LlumiverseError(
                'Service overloaded',
                true,
                mockContext,
                new Error('Overloaded'),
                529
            );

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(529);
        });
    });

    describe('context information', () => {
        it('should include provider information', () => {
            const error = new LlumiverseError(
                'Error',
                true,
                { ...mockContext, provider: 'openai' },
                new Error('Test'),
                500
            );

            expect(error.context.provider).toBe('openai');
        });

        it('should include model information', () => {
            const error = new LlumiverseError(
                'Error',
                true,
                { ...mockContext, model: 'gpt-4' },
                new Error('Test'),
                500
            );

            expect(error.context.model).toBe('gpt-4');
        });

        it('should include operation type', () => {
            const executeError = new LlumiverseError(
                'Error',
                true,
                { ...mockContext, operation: 'execute' as const },
                new Error('Test'),
                500
            );

            const streamError = new LlumiverseError(
                'Error',
                true,
                { ...mockContext, operation: 'stream' as const },
                new Error('Test'),
                500
            );

            expect(executeError.context.operation).toBe('execute');
            expect(streamError.context.operation).toBe('stream');
        });
    });
});
