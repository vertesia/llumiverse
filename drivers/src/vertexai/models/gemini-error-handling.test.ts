import { LlumiverseError } from '@llumiverse/core';
import { beforeEach, describe, expect, it } from 'vitest';
import { VertexAIDriver } from '../index.js';
import { GeminiModelDefinition } from './gemini.js';
import { exposePrivate, getProp } from '../../../test/__helpers__/test-utils.js';

type GeminiModelInternals = {
    isGeminiErrorRetryable: (httpStatusCode: number, message?: string) => boolean | undefined;
};

describe('GeminiModelDefinition Error Handling', () => {
    let driver: VertexAIDriver;
    let modelDef: GeminiModelDefinition;

    beforeEach(() => {
        driver = new VertexAIDriver({
            project: 'test-project',
            region: 'us-central1',
        });
        modelDef = new GeminiModelDefinition('gemini-2.0-flash');
    });

    describe('formatLlumiverseError', () => {
        it('should handle INVALID_ARGUMENT error (400)', () => {
            const googleError = {
                status: 400,
                message: 'INVALID_ARGUMENT: Invalid value for temperature. Must be between 0 and 2.',
            };

            const error = modelDef.formatLlumiverseError(driver, googleError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            expect(error).toBeInstanceOf(LlumiverseError);
            expect(error.code).toBe(400);
            expect(error.retryable).toBe(false);
            expect(error.message).toContain('[400]');
            expect(error.message).toContain('Invalid value for temperature');
            expect(error.name).toBe('INVALID_ARGUMENT');
            expect(error.context.provider).toBe('vertexai');
        });

        it('should handle UNAUTHENTICATED error (401)', () => {
            const googleError = {
                status: 401,
                message: 'UNAUTHENTICATED: Request had invalid authentication credentials.',
            };

            const error = modelDef.formatLlumiverseError(driver, googleError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            expect(error.retryable).toBe(false);
            expect(error.code).toBe(401);
            expect(error.name).toBe('UNAUTHENTICATED');
        });

        it('should handle PERMISSION_DENIED error (403)', () => {
            const googleError = {
                status: 403,
                message: 'PERMISSION_DENIED: The caller does not have permission to execute the specified operation.',
            };

            const error = modelDef.formatLlumiverseError(driver, googleError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            expect(error.retryable).toBe(false);
            expect(error.code).toBe(403);
            expect(error.name).toBe('PERMISSION_DENIED');
        });

        it('should handle NOT_FOUND error (404)', () => {
            const googleError = {
                status: 404,
                message: 'NOT_FOUND: Requested entity was not found.',
            };

            const error = modelDef.formatLlumiverseError(driver, googleError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            expect(error.retryable).toBe(false);
            expect(error.code).toBe(404);
            expect(error.name).toBe('NOT_FOUND');
        });

        it('should handle RESOURCE_EXHAUSTED error (429) as retryable', () => {
            const googleError = {
                status: 429,
                message: 'RESOURCE_EXHAUSTED: Quota exceeded for quota metric',
            };

            const error = modelDef.formatLlumiverseError(driver, googleError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(429);
            expect(error.message).toContain('[429]');
            expect(error.name).toBe('RESOURCE_EXHAUSTED');
        });

        it('should handle INTERNAL error (500) as retryable', () => {
            const googleError = {
                status: 500,
                message: 'INTERNAL: Internal server error',
            };

            const error = modelDef.formatLlumiverseError(driver, googleError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(500);
            expect(error.message).toContain('[500]');
        });

        it('should handle BAD_GATEWAY error (502) as retryable', () => {
            const googleError = {
                status: 502,
                message: 'Bad gateway',
            };

            const error = modelDef.formatLlumiverseError(driver, googleError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(502);
        });

        it('should handle UNAVAILABLE error (503) as retryable', () => {
            const googleError = {
                status: 503,
                message: 'UNAVAILABLE: The service is currently unavailable.',
            };

            const error = modelDef.formatLlumiverseError(driver, googleError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(503);
            expect(error.name).toBe('UNAVAILABLE');
        });

        it('should handle DEADLINE_EXCEEDED error (504) as retryable', () => {
            const googleError = {
                status: 504,
                message: 'DEADLINE_EXCEEDED: Request timeout',
            };

            const error = modelDef.formatLlumiverseError(driver, googleError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(504);
            expect(error.name).toBe('DEADLINE_EXCEEDED');
        });

        it('should handle REQUEST_TIMEOUT error (408) as retryable', () => {
            const googleError = {
                status: 408,
                message: 'Request timeout',
            };

            const error = modelDef.formatLlumiverseError(driver, googleError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            expect(error.retryable).toBe(true);
            expect(error.code).toBe(408);
        });

        it('should preserve original error for debugging', () => {
            const googleError = {
                status: 429,
                message: 'RESOURCE_EXHAUSTED: Quota exceeded',
            };

            const error = modelDef.formatLlumiverseError(driver, googleError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            expect(error.originalError).toBe(googleError);
            expect(getProp<number>(error.originalError, 'status')).toBe(429);
        });

        it('should throw for non-Google API errors', () => {
            const regularError = new Error('Regular error');

            expect(() => {
                modelDef.formatLlumiverseError(driver, regularError, {
                    provider: 'vertexai',
                    model: 'gemini-2.0-flash',
                    operation: 'execute',
                });
            }).toThrow();
        });

        it('should extract error name from bracket format', () => {
            const googleError = {
                status: 400,
                message: '[INVALID_ARGUMENT] Invalid parameter',
            };

            const error = modelDef.formatLlumiverseError(driver, googleError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            expect(error.name).toBe('INVALID_ARGUMENT');
        });

        it('should extract error name from Error suffix format', () => {
            const googleError = {
                status: 400,
                message: 'ValidationError: Invalid input',
            };

            const error = modelDef.formatLlumiverseError(driver, googleError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            expect(error.name).toBe('ValidationError');
        });

        it('should mark URL_REJECTED-REJECTED_CLIENT_THROTTLED 400s as retryable', () => {
            // Reproduces a real Vertex AI failure: passing an audio file by URL
            // (sys:AnalyzeAudio) sometimes 400s with INVALID_ARGUMENT whose
            // message contains URL_REJECTED-REJECTED_CLIENT_THROTTLED — that's
            // really a transient throttle on Google's URL fetcher.
            const googleError = {
                status: 400,
                message:
                    'Cannot fetch content from the provided URL. Please ensure the URL is valid ' +
                    'and accessible by Vertex AI. Vertex AI respects robots.txt rules, so confirm ' +
                    'the URL is allowed to be crawled. Status: URL_REJECTED-REJECTED_CLIENT_THROTTLED',
            };

            const error = modelDef.formatLlumiverseError(driver, googleError, {
                provider: 'vertexai',
                model: 'gemini-2.5-flash',
                operation: 'execute',
            });

            expect(error.code).toBe(400);
            expect(error.retryable).toBe(true);
        });

        it('should handle errors without extractable name', () => {
            const googleError = {
                status: 500,
                message: 'Something went wrong',
            };

            const error = modelDef.formatLlumiverseError(driver, googleError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            // When no name is extracted, defaults to 'LlumiverseError'
            expect(error.name).toBe('LlumiverseError');
            expect(error.code).toBe(500);
        });
    });

    describe('isGeminiErrorRetryable', () => {
        it('should classify retryable status codes correctly', () => {
            const retryableStatusCodes = [408, 429, 500, 502, 503, 504];

            retryableStatusCodes.forEach((statusCode) => {
                const result = exposePrivate<GeminiModelInternals>(modelDef).isGeminiErrorRetryable(statusCode);
                expect(result, `Status code ${statusCode} should be retryable`).toBe(true);
            });
        });

        it('should classify non-retryable status codes correctly', () => {
            const nonRetryableStatusCodes = [400, 401, 403, 404, 409];

            nonRetryableStatusCodes.forEach((statusCode) => {
                const result = exposePrivate<GeminiModelInternals>(modelDef).isGeminiErrorRetryable(statusCode);
                expect(result, `Status code ${statusCode} should not be retryable`).toBe(false);
            });
        });

        it('should classify other 5xx errors as retryable', () => {
            expect(exposePrivate<GeminiModelInternals>(modelDef).isGeminiErrorRetryable(501)).toBe(true);
            expect(exposePrivate<GeminiModelInternals>(modelDef).isGeminiErrorRetryable(505)).toBe(true);
            expect(exposePrivate<GeminiModelInternals>(modelDef).isGeminiErrorRetryable(599)).toBe(true);
        });

        it('should classify other 4xx errors as non-retryable', () => {
            expect(exposePrivate<GeminiModelInternals>(modelDef).isGeminiErrorRetryable(402)).toBe(false);
            expect(exposePrivate<GeminiModelInternals>(modelDef).isGeminiErrorRetryable(405)).toBe(false);
            expect(exposePrivate<GeminiModelInternals>(modelDef).isGeminiErrorRetryable(499)).toBe(false);
        });

        it('should treat 400 URL_REJECTED-REJECTED_CLIENT_THROTTLED as retryable', () => {
            // Vertex AI surfaces inline-URL-fetcher throttling as 400 INVALID_ARGUMENT.
            // The status is misleading — it's a transient Google-side throttle.
            const message =
                'Cannot fetch content from the provided URL. ' + 'Status: URL_REJECTED-REJECTED_CLIENT_THROTTLED';
            expect(exposePrivate<GeminiModelInternals>(modelDef).isGeminiErrorRetryable(400, message)).toBe(true);
        });

        it('should treat 400 URL_REJECTED-REJECTED_RATE_LIMITED as retryable', () => {
            const message =
                'Cannot fetch content from the provided URL. ' + 'Status: URL_REJECTED-REJECTED_RATE_LIMITED';
            expect(exposePrivate<GeminiModelInternals>(modelDef).isGeminiErrorRetryable(400, message)).toBe(true);
        });

        it('should still treat plain 400 INVALID_ARGUMENT as non-retryable', () => {
            expect(
                exposePrivate<GeminiModelInternals>(modelDef).isGeminiErrorRetryable(
                    400,
                    'INVALID_ARGUMENT: bad input',
                ),
            ).toBe(false);
        });
    });

    describe('VertexAIDriver error routing', () => {
        it('should route to Gemini-specific error handler', () => {
            const googleError = {
                status: 429,
                message: 'RESOURCE_EXHAUSTED: Quota exceeded',
            };

            const error = driver.formatLlumiverseError(googleError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            expect(error).toBeInstanceOf(LlumiverseError);
            expect(error.code).toBe(429);
            expect(error.retryable).toBe(true);
            expect(error.name).toBe('RESOURCE_EXHAUSTED');
        });

        it('should fall back to default handler for non-Google errors', () => {
            const regularError = new Error('Regular error');

            const error = driver.formatLlumiverseError(regularError, {
                provider: 'vertexai',
                model: 'gemini-2.0-flash',
                operation: 'execute',
            });

            expect(error).toBeInstanceOf(LlumiverseError);
            expect(error.code).toBeUndefined();
            expect(error.retryable).toBeUndefined(); // Unknown errors - let consumer decide
        });

        it('should work with different Gemini model versions', () => {
            const googleError = {
                status: 400,
                message: 'INVALID_ARGUMENT: Invalid parameter',
            };

            const models = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-1.5-pro'];

            models.forEach((model) => {
                const error = driver.formatLlumiverseError(googleError, {
                    provider: 'vertexai',
                    model,
                    operation: 'execute',
                });

                expect(error.code).toBe(400);
                expect(error.retryable).toBe(false);
            });
        });
    });
});
