import { type ExecutionOptions, Providers } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { getAdvertisedTestOptions } from './utils.js';

describe('successful live smoke options', () => {
    it('uses provider-specific advertised controls without modifying the source fixture', () => {
        const fixture: ExecutionOptions = {
            model: 'gpt-4o-mini',
            model_options: {
                _option_id: 'text-fallback',
                max_tokens: 512,
                temperature: 0.3,
                stop_sequence: ['END'],
                presence_penalty: 0.1,
                frequency_penalty: -0.1,
            },
        };
        expect(getAdvertisedTestOptions(fixture, Providers.openai).model_options).toEqual({
            _option_id: 'text-fallback',
            max_tokens: 512,
            temperature: 0.3,
        });
        expect(getAdvertisedTestOptions(fixture, Providers.openai_compatible).model_options).toEqual(
            fixture.model_options,
        );
        expect(fixture.model_options).toHaveProperty('stop_sequence', ['END']);
    });

    it('does not inject reasoning-model sampling controls', () => {
        expect(
            getAdvertisedTestOptions(
                {
                    model: 'gpt-5.4-mini',
                    model_options: {
                        _option_id: 'text-fallback',
                        max_tokens: 512,
                        temperature: 0.3,
                        top_p: 0.7,
                    },
                },
                Providers.openai,
            ).model_options,
        ).toEqual({ _option_id: 'text-fallback', max_tokens: 512 });
    });
});
