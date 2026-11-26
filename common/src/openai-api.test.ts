import { describe, expect, it } from 'vitest';
import { isOpenAIApi, OPENAI_API_DEFAULTS, resolveOpenAIApi } from './openai-api.js';

describe('OpenAI API selection', () => {
    it('keeps the API each provider used before the option existed', () => {
        expect(OPENAI_API_DEFAULTS).toEqual({
            openai_compatible: 'responses',
            openrouter: 'chat_completions',
            groq: 'chat_completions',
            togetherai: 'chat_completions',
            xai: 'responses',
        });
    });

    it.each(Object.keys(OPENAI_API_DEFAULTS))('resolves the option of %s, or its default when omitted', (provider) => {
        expect(resolveOpenAIApi(provider)).toBe(OPENAI_API_DEFAULTS[provider]);
        expect(resolveOpenAIApi(provider, 'responses')).toBe('responses');
        expect(resolveOpenAIApi(provider, 'chat_completions')).toBe('chat_completions');
    });

    it('has no API for a provider without the option', () => {
        expect(resolveOpenAIApi('openai', 'chat_completions')).toBeUndefined();
        expect(resolveOpenAIApi('anthropic')).toBeUndefined();
        expect(resolveOpenAIApi('toString')).toBeUndefined();
        expect(resolveOpenAIApi('__proto__', 'responses')).toBeUndefined();
    });

    it.each([
        ['responses', true],
        ['chat_completions', true],
        ['completions', false],
        ['', false],
        [undefined, false],
        [null, false],
    ])('isOpenAIApi(%j) is %s', (value, expected) => {
        expect(isOpenAIApi(value)).toBe(expected);
    });
});
