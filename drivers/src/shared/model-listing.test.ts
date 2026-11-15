import { Providers } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { resolveModelListingMetadata } from './model-listing.js';

describe('model listing metadata precedence', () => {
    it('keeps complete catalog metadata when runtime metadata disagrees', () => {
        expect(
            resolveModelListingMetadata('nvidia.nemotron-nano-12b-v2-vl-bf16', Providers.bedrock, {
                input_modalities: ['TEXT'],
                output_modalities: ['TEXT', 'IMAGE'],
            }),
        ).toMatchObject({
            input_modalities: ['text', 'image'],
            output_modalities: ['text'],
        });
    });

    it.each([Providers.openai, Providers.azure_foundry, Providers.vertexai])(
        'keeps known family metadata for %s',
        (provider) => {
            expect(
                resolveModelListingMetadata('gpt-5.6-sol', provider, {
                    input_modalities: ['TEXT'],
                    output_modalities: ['AUDIO'],
                }),
            ).toMatchObject({
                input_modalities: ['text', 'image'],
                output_modalities: ['text'],
                tool_support: true,
            });
        },
    );

    it('uses explicitly supplied runtime modalities only when no catalog rule matched', () => {
        expect(
            resolveModelListingMetadata('nvidia.future-foundation-model', Providers.bedrock, {
                input_modalities: ['TEXT', 'IMAGE', 'FUTURE_MODALITY'],
                output_modalities: ['SPEECH'],
            }),
        ).toEqual({
            input_modalities: ['text', 'image', 'future_modality'],
            output_modalities: ['audio'],
        });
    });

    it('uses conservative text inference metadata for a wholly unknown model', () => {
        expect(resolveModelListingMetadata('future-provider/new-model', Providers.openai_compatible)).toEqual({
            input_modalities: ['text'],
            output_modalities: ['text'],
        });
    });
});

describe('model listing lifecycle dates', () => {
    it('prefers dates reported by the provider over the curated table', () => {
        expect(
            resolveModelListingMetadata('anthropic.claude-opus-4-1-20250805-v1:0', Providers.bedrock, {
                deprecation_date: new Date('2026-07-09T00:00:00Z'),
                retirement_date: new Date('2027-01-09T00:00:00Z'),
            }),
        ).toMatchObject({ deprecation_date: '2026-07-09', retirement_date: '2027-01-09' });
    });

    it('fills dates from the curated table when the provider reports none', () => {
        expect(
            resolveModelListingMetadata('us.anthropic.claude-opus-4-1-20250805-v1:0', Providers.bedrock),
        ).toMatchObject({ deprecation_date: '2026-07-08', retirement_date: '2027-01-08' });
        expect(resolveModelListingMetadata('gpt-4o', Providers.azure_foundry, { version: '2024-05-13' })).toMatchObject(
            { retirement_date: '2027-01-14' },
        );
    });

    it('omits lifecycle keys when no date is known', () => {
        const metadata = resolveModelListingMetadata('gemini-3.8-flash', Providers.vertexai, {
            retirement_date: null,
        });
        expect(metadata).not.toHaveProperty('retirement_date');
        expect(metadata).not.toHaveProperty('deprecation_date');
    });
});
