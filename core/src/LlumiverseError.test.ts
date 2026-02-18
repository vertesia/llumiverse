import { LlumiverseErrorContext } from '@llumiverse/common';
import { describe, expect, it } from 'vitest';
import { LlumiverseError } from './LlumiverseError.js';

describe('LlumiverseError', () => {
    const mockContext: LlumiverseErrorContext = {
        provider: 'test-provider',
        model: 'test-model',
        operation: 'execute' as const,
        prompt: { test: 'prompt' },
    };

    describe('constructor', () => {
        it('should create an error with all properties', () => {
            const originalError = new Error('Original error');
            const error = new LlumiverseError(
                'Test error message',
                429,
                true,
                mockContext,
                originalError
            );

            expect(error).toBeInstanceOf(Error);
            expect(error).toBeInstanceOf(LlumiverseError);
            expect(error.name).toBe('LlumiverseError');
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
                500,
                true,
                mockContext,
                originalError
            );

            expect(error.stack).toBe(originalStack);
        });

        it('should handle string codes', () => {
            const error = new LlumiverseError(
                'Test error',
                'ThrottlingException',
                true,
                mockContext,
                new Error('AWS error')
            );

            expect(error.code).toBe('ThrottlingException');
        });

        it('should handle non-retryable errors', () => {
            const error = new LlumiverseError(
                'Auth error',
                401,
                false,
                mockContext,
                new Error('Unauthorized')
            );

            expect(error.retryable).toBe(false);
        });
    });

    describe('toJSON', () => {
        it('should serialize to JSON', () => {
            const originalError = new Error('Original error');
            const error = new LlumiverseError(
                'Test error',
                429,
                true,
                mockContext,
                originalError
            );

            const json = error.toJSON();

            expect(json).toHaveProperty('name', 'LlumiverseError');
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
                500,
                true,
                mockContext,
                'string error'
            );

            const json = error.toJSON();
            expect(json.originalErrorMessage).toBe('string error');
        });
    });

    describe('isLlumiverseError', () => {
        it('should return true for LlumiverseError instances', () => {
            const error = new LlumiverseError(
                'Test error',
                500,
                true,
                mockContext,
                new Error('Original')
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
                429,
                true,
                { ...mockContext, operation: 'execute' as const },
                new Error('Too many requests')
            );

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(429);
        });

        it('should handle server errors', () => {
            const error = new LlumiverseError(
                'Internal server error',
                500,
                true,
                { ...mockContext, operation: 'stream' as const },
                new Error('Server error')
            );

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(500);
        });

        it('should handle authentication errors', () => {
            const error = new LlumiverseError(
                'Invalid API key',
                401,
                false,
                mockContext,
                new Error('Unauthorized')
            );

            expect(error.retryable).toBe(false);
            expect(error.code).toBe(401);
        });

        it('should handle validation errors', () => {
            const error = new LlumiverseError(
                'Invalid request',
                400,
                false,
                mockContext,
                new Error('Bad request')
            );

            expect(error.retryable).toBe(false);
            expect(error.code).toBe(400);
        });

        it('should handle timeout errors', () => {
            const error = new LlumiverseError(
                'Request timeout',
                408,
                true,
                mockContext,
                new Error('Timeout')
            );

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(408);
        });

        it('should handle service overloaded errors', () => {
            const error = new LlumiverseError(
                'Service overloaded',
                529,
                true,
                mockContext,
                new Error('Overloaded')
            );

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(529);
        });
    });

    describe('context information', () => {
        it('should include provider information', () => {
            const error = new LlumiverseError(
                'Error',
                500,
                true,
                { ...mockContext, provider: 'openai' },
                new Error('Test')
            );

            expect(error.context.provider).toBe('openai');
        });

        it('should include model information', () => {
            const error = new LlumiverseError(
                'Error',
                500,
                true,
                { ...mockContext, model: 'gpt-4' },
                new Error('Test')
            );

            expect(error.context.model).toBe('gpt-4');
        });

        it('should include operation type', () => {
            const executeError = new LlumiverseError(
                'Error',
                500,
                true,
                { ...mockContext, operation: 'execute' as const },
                new Error('Test')
            );

            const streamError = new LlumiverseError(
                'Error',
                500,
                true,
                { ...mockContext, operation: 'stream' as const },
                new Error('Test')
            );

            expect(executeError.context.operation).toBe('execute');
            expect(streamError.context.operation).toBe('stream');
        });

        it('should optionally include prompt', () => {
            const errorWithPrompt = new LlumiverseError(
                'Error',
                500,
                true,
                { ...mockContext, prompt: 'test prompt' },
                new Error('Test')
            );

            const errorWithoutPrompt = new LlumiverseError(
                'Error',
                500,
                true,
                { provider: 'test', model: 'test', operation: 'execute' as const },
                new Error('Test')
            );

            expect(errorWithPrompt.context.prompt).toBe('test prompt');
            expect(errorWithoutPrompt.context.prompt).toBeUndefined();
        });
    });
});
