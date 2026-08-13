import { describe, expect, it } from 'vitest';
import { resolveModelProfile } from '../model-directory.js';
import { ModelOptionsSchema } from '../schemas/model-options.js';
import { Providers } from '../types.js';
import { getXAIOptions, isXAIGrokImageModel } from './xai.js';

describe('xAI options', () => {
    it.each(['grok-2-image-1212', 'grok-imagine-image', 'grok-imagine-image-quality', 'grok-imagine-image-2.0'])(
        'recognizes the Grok image family for %s',
        (model) => {
            expect(isXAIGrokImageModel(model)).toBe(true);
            expect(resolveModelProfile(model, Providers.xai)).toMatchObject({
                family: 'image',
                source_provider: 'xai',
                capabilities: {
                    input: { text: true, image: true },
                    output: { image: true },
                    tool_support: false,
                },
            });
        },
    );

    it('exposes the xAI image API controls for current and future Imagine models', () => {
        expect(getXAIOptions('grok-imagine-image-2.0')).toMatchObject({
            _option_id: 'xai-grok-image',
            options: expect.arrayContaining([
                expect.objectContaining({ name: 'aspect_ratio' }),
                expect.objectContaining({ name: 'resolution' }),
                expect.objectContaining({ name: 'quality', default: 'medium' }),
                expect.objectContaining({ name: 'response_format' }),
                expect.objectContaining({ name: 'n', min: 1, max: 10 }),
            ]),
        });
    });

    it('keeps OpenAI-compatible text options for Grok language models', () => {
        expect(getXAIOptions('grok-4.5')._option_id).toBe('openai-text');
    });

    it('validates the published image options and batch limit', () => {
        expect(
            ModelOptionsSchema.safeParse({
                _option_id: 'xai-grok-image',
                aspect_ratio: '19.5:9',
                resolution: '2k',
                quality: 'low',
                response_format: 'b64_json',
                n: 10,
            }).success,
        ).toBe(true);
        expect(ModelOptionsSchema.safeParse({ _option_id: 'xai-grok-image', n: 11 }).success).toBe(false);
        expect(ModelOptionsSchema.safeParse({ _option_id: 'xai-grok-image', quality: 'high' }).success).toBe(false);
    });

    it('only exposes the quality control for Image 2.0', () => {
        expect(getXAIOptions('grok-imagine-image').options).not.toEqual(
            expect.arrayContaining([expect.objectContaining({ name: 'quality' })]),
        );
    });
});
