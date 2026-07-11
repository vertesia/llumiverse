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

    it('falls back to prompt alignment when Nova Micro rejects outputConfig', async () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const prompt = await formatConversePrompt([{ role: PromptRole.user, content: 'hello' }], {
            model: 'us.amazon.nova-micro-v1:0',
            result_schema,
        });
        const payload = driver.preparePayload(prompt, { model: 'us.amazon.nova-micro-v1:0', result_schema });

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
