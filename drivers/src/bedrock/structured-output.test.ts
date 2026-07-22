import { PromptRole } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { formatConversePrompt } from './converse.js';
import { BedrockDriver } from './index.js';

const result_schema = {
    type: 'object' as const,
    properties: {
        object: { type: 'string' as const },
        color: { type: 'string' as const },
    },
    required: ['object', 'color'],
};

describe('Bedrock Converse structured output', () => {
    it('keeps native schemas out of non-Claude prompts', async () => {
        const prompt = await formatConversePrompt([{ role: PromptRole.user, content: 'hello' }], {
            model: 'google.gemma-3-12b-it',
            result_schema,
        });

        expect(prompt.system).toBeUndefined();
        expect(prompt.messages).toEqual([{ role: 'user', content: [{ text: 'hello' }] }]);
    });

    it('preserves Claude prompt-aligned schema behavior', async () => {
        const prompt = await formatConversePrompt([{ role: PromptRole.user, content: 'hello' }], {
            model: 'anthropic.claude-sonnet-4-6',
            result_schema,
        });

        expect(prompt.system).toEqual([
            { text: expect.stringContaining('The answer must be a JSON object using the following JSON Schema') },
        ]);
    });

    it('uses outputConfig for non-Claude models', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const payload = driver.preparePayload(
            { modelId: undefined, messages: [{ role: 'user', content: [{ text: 'hello' }] }] },
            { model: 'google.gemma-3-12b-it', result_schema },
        );

        expect(payload.outputConfig).toEqual({
            textFormat: {
                type: 'json_schema',
                structure: {
                    jsonSchema: {
                        name: 'output',
                        schema: JSON.stringify(result_schema),
                    },
                },
            },
        });
    });

    it('falls back to prompt alignment when Gemma 3 4B rejects outputConfig', async () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const prompt = await formatConversePrompt([{ role: PromptRole.user, content: 'hello' }], {
            model: 'google.gemma-3-4b-it',
            result_schema,
        });
        const payload = driver.preparePayload(prompt, { model: 'google.gemma-3-4b-it', result_schema });

        expect(prompt.system).toEqual([{ text: expect.stringContaining('JSON Schema') }]);
        expect(payload.outputConfig).toBeUndefined();
    });

    it.each([
        'us.amazon.nova-micro-v1:0',
        'global.amazon.nova-2-lite-v1:0',
        'ai21.jamba-1-5-large-v1:0',
        'cohere.command-r-v1:0',
        'us.meta.llama3-3-70b-instruct-v1:0',
        'us.deepseek.r1-v1:0',
        'mistral.mistral-large-2402-v1:0',
        'mistral.pixtral-large-2502-v1:0',
        'us.writer.palmyra-x5-v1:0',
    ])('falls back to prompt alignment when %s rejects outputConfig', async (model) => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const prompt = await formatConversePrompt([{ role: PromptRole.user, content: 'hello' }], {
            model,
            result_schema,
        });
        const payload = driver.preparePayload(prompt, { model, result_schema });

        expect(prompt.system).toEqual([{ text: expect.stringContaining('JSON Schema') }]);
        expect(payload.outputConfig).toBeUndefined();
    });

    it('supplements MiniMax M2.5 outputConfig with prompt alignment', async () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const prompt = await formatConversePrompt([{ role: PromptRole.user, content: 'hello' }], {
            model: 'minimax.minimax-m2.5',
            result_schema,
        });
        const payload = driver.preparePayload(prompt, { model: 'minimax.minimax-m2.5', result_schema });

        expect(prompt.system).toEqual([{ text: expect.stringContaining('JSON Schema') }]);
        expect(payload.outputConfig?.textFormat?.type).toBe('json_schema');
    });

    it('does not change Claude to outputConfig', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const payload = driver.preparePayload(
            { modelId: undefined, messages: [{ role: 'user', content: [{ text: 'hello' }] }] },
            { model: 'anthropic.claude-sonnet-4-6', result_schema },
        );

        expect(payload.outputConfig).toBeUndefined();
    });
});
