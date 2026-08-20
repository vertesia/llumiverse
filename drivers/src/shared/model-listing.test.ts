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
