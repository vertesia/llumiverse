import { describe, expect, it, vi } from 'vitest';
import { getOpenAIExtraBody, mergeOpenAIExtraBody } from './extra_body.js';

describe('OpenAI-compatible extra body', () => {
    it('warns only for supplied extension fields changed by the request contract', () => {
        const warn = vi.fn();
        const logger = { warn, info: vi.fn(), debug: vi.fn(), error: vi.fn() };
        mergeOpenAIExtraBody(
            { model: 'actual-model', temperature: undefined, stream: false },
            { model: 'override', temperature: 0, stream: false, provider: { sort: 'price' } },
            logger,
            'actual-model',
        );
        expect(warn).toHaveBeenCalledExactlyOnceWith(
            { model: 'actual-model', option_names: ['model', 'temperature'], reason: 'openai_extra_body_precedence' },
            'Model option compatibility exception changed caller input',
        );
    });

    it('does not warn when nested extension values equal the generated request', () => {
        const logger = { warn: vi.fn(), info: vi.fn(), debug: vi.fn(), error: vi.fn() };
        mergeOpenAIExtraBody(
            { stream_options: { include_usage: true }, stop: ['one', 'two'] },
            { stop: ['one', 'two'], stream_options: { include_usage: true } },
            logger,
            'model',
        );
        expect(logger.warn).not.toHaveBeenCalled();
    });

    it('extracts only object-shaped extension fields', () => {
        expect(getOpenAIExtraBody({ extra_body: { provider: { sort: 'price' } } })).toEqual({
            provider: { sort: 'price' },
        });
        expect(getOpenAIExtraBody({ extra_body: ['invalid'] })).toBeUndefined();
        expect(getOpenAIExtraBody(undefined)).toBeUndefined();
    });

    it('merges extensions at the top level while preserving core request fields', () => {
        expect(
            mergeOpenAIExtraBody(
                { model: 'actual-model', stream: false },
                { provider: { sort: 'throughput' }, model: 'override', stream: true },
            ),
        ).toEqual({
            provider: { sort: 'throughput' },
            model: 'actual-model',
            stream: false,
        });
    });
});
