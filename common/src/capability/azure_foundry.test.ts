import { describe, expect, it } from 'vitest';
import { resolveModelProfile } from '../model-directory.js';
import { Providers } from '../types.js';

function foundry(model: string) {
    return resolveModelProfile(`deployment::${model}`, Providers.azure_foundry);
}

describe('Microsoft Foundry partner model capabilities', () => {
    // Families whose Foundry model cards say "Tool calling: Yes", including later generations of each family.
    it.each([
        'DeepSeek-V3.2',
        'DeepSeek-V3.2-Speciale',
        'DeepSeek-V4-Pro',
        'DeepSeek-V4.1-Flash',
        'DeepSeek-V5',
        'Kimi-K2.5',
        'Kimi-K3.1-Pro',
        'Kimi-K4',
        'grok-4',
        'grok-4.7',
        'grok-4-20-reasoning',
        'grok-code-fast-1',
        'grok-5',
        'Mistral-Large-3',
        'mistral-large-4-latest',
        'mistral-small-2503',
        'Ministral-3B',
        'Cohere-command-a',
        'Cohere-command-a-plus-05-2026',
        'MAI-Thinking-1',
        'MAI-Thinking-2',
    ])('turns tools on for %s', (model) => {
        const { capabilities } = foundry(model);
        expect(capabilities.tool_support).toBe(true);
        expect(capabilities.tool_support_streaming).toBe(true);
    });

    // Documented without tool calling, or not documented at all.
    it.each([
        'DeepSeek-R1-0528',
        'DeepSeek-V3-0324',
        'Kimi-K2-Thinking',
        'Llama-4-Maverick-17B-128E-Instruct-FP8',
        'Llama-4-Scout-17B-16E-Instruct',
        'Llama-3.3-70B-Instruct',
        'Llama-5-Scout',
        'mistral-medium-3-5',
        'mistral-medium-2505',
        'Codestral-2501',
        'Phi-4',
        'Phi-4-multimodal-instruct',
        'tsuzumi-7b',
        'AI21-Jamba-1.5-Large',
        'some-future-partner-model',
    ])('keeps tools off for %s', (model) => {
        const { capabilities } = foundry(model);
        expect(capabilities.tool_support).toBe(false);
        expect(capabilities.tool_support_streaming).toBe(false);
    });

    it.each([
        ['Kimi-K2.6', true],
        ['Kimi-K3.1-Pro', true],
        ['grok-4.1-fast-non-reasoning', true],
        ['grok-4.6', true],
        ['grok-4.7', true],
        ['grok-4', false],
        ['grok-4.3', false],
        ['grok-4-20-non-reasoning', false],
        ['Mistral-Large-3', true],
        ['Mistral-large', false],
        ['mistral-medium-3-5', true],
        ['DeepSeek-V4.1-Flash', false],
        ['Phi-4-multimodal-instruct', true],
        ['Phi-4', false],
        ['MAI-Thinking-1', undefined],
    ])('documents image input for %s as %s', (model, image) => {
        expect(foundry(model).capabilities.input.image).toBe(image);
    });

    it('documents audio input for Phi-4 multimodal only', () => {
        expect(foundry('Phi-4-multimodal-instruct').capabilities.input.audio).toBe(true);
        expect(foundry('Phi-4-mini-instruct').capabilities.input.audio).not.toBe(true);
    });

    it('keeps the OpenAI families on their own capability records', () => {
        for (const model of ['gpt-6.1-sol', 'gpt-6-luna', 'gpt-5.6-sol', 'gpt-7']) {
            const { capabilities } = foundry(model);
            expect(capabilities.tool_support).toBe(true);
            expect(capabilities.input.image).toBe(true);
        }
        expect(foundry('o1-mini').capabilities.tool_support).toBe(false);
    });

    it.each([
        ['Kimi-K2.6', 262_144, 262_144],
        ['Kimi-K3.1-Pro', 1_000_000, 262_144],
        ['MAI-Thinking-1', 256_000, 64_000],
    ])('uses the Foundry limits of %s', (model, context, output) => {
        const profile = foundry(model);
        expect(profile.context_window).toBe(context);
        expect(profile.max_output_tokens).toBe(output);
    });

    it('uses the shared 500K window of Grok 4.7 and later', () => {
        expect(foundry('grok-4.7').context_window).toBe(500_000);
        expect(foundry('grok-4.8').context_window).toBe(500_000);
    });

    it('keeps Claude, embeddings and images on their source capabilities', () => {
        expect(foundry('claude-opus-5-5').capabilities.tool_support).toBe(true);
        expect(foundry('text-embedding-3-large').capabilities.output.embed).toBe(true);
        expect(foundry('gpt-image-2.5').capabilities.output.image).toBe(true);
    });
});
