import { describe, expect, it } from 'vitest';
import { OpenAiGptImageOptionsSchema, OpenAiTextOptionsSchema } from '../schemas/model-options.js';
import { OptionType } from '../types.js';
import { getOpenAiOptions } from './openai.js';
import { isOpenAIImageVersionGTE } from './version-parsing.js';

describe('GPT Image versions and options', () => {
    it.each([
        'gpt-image-2',
        'gpt-image-2-2026-04-21',
        'deployment::gpt-image-2.5-flare',
        'openai/gpt-image-3',
        'OpenAI.GPT-IMAGE-2.5-SUNBURST',
    ])('recognizes %s', (model) => {
        expect(isOpenAIImageVersionGTE(model, 2)).toBe(true);
        for (const name of ['width', 'height']) {
            expect(getOpenAiOptions(model).options.find((option) => option.name === name)?.type).toBe(
                OptionType.numeric,
            );
        }
    });
    it('exposes expanded quality only at 2.5 onward', () => {
        for (const model of ['gpt-image-1.5', 'gpt-image-2', 'gpt-image-2.5-sunburst', 'gpt-image-3']) {
            const quality = getOpenAiOptions(model).options.find((option) => option.name === 'image_quality');
            expect(quality?.type).toBe(OptionType.enum);
            if (quality?.type === OptionType.enum) {
                expect(Object.values(quality.enum).includes('max')).toBe(isOpenAIImageVersionGTE(model, 2, 5));
            }
        }
    });
    it('publishes custom sizes and requires the opt-in tool model', () => {
        expect(
            OpenAiGptImageOptionsSchema.safeParse({
                width: 2048,
                height: 1024,
                image_quality: 'max',
                n: 2,
                output_compression: 0,
                partial_images: 0,
            }).success,
        ).toBe(true);
        expect(OpenAiTextOptionsSchema.safeParse({ image_generation: {} }).success).toBe(false);
        expect(
            OpenAiTextOptionsSchema.safeParse({ image_generation: { model: 'gpt-image-2.5-flare', force: true } })
                .success,
        ).toBe(true);
    });
});
