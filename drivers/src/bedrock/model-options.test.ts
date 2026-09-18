import type { ModelOptions } from '@llumiverse/common';
import { describe, expect, it } from 'vitest';
import { BedrockDriver } from './index.js';

describe('Bedrock provider-specific model options', () => {
    it.each([
        ['amazon.nova-pro-v1:0', { _option_id: 'bedrock-nova', top_k: 12 }, { inferenceConfig: { topK: 12 } }],
        ['mistral.mixtral-8x7b-instruct-v0:1', { _option_id: 'bedrock-mistral', top_k: 12 }, { top_k: 12 }],
        ['cohere.command-text-v14', { _option_id: 'bedrock-cohere-command', top_k: 12 }, { k: 12 }],
        [
            'cohere.command-r-v1:0',
            { _option_id: 'bedrock-cohere-command', top_k: 12, presence_penalty: 0.4, frequency_penalty: 0.2 },
            { k: 12, presence_penalty: 0.4, frequency_penalty: 0.2 },
        ],
        [
            'ai21.jamba-1-5-large-v1:0',
            { _option_id: 'bedrock-ai21', presence_penalty: 0.4, frequency_penalty: 0.2 },
            { presence_penalty: 0.4, frequency_penalty: 0.2 },
        ],
        [
            'ai21.j2-ultra-v1',
            { _option_id: 'bedrock-ai21', presence_penalty: 0.4, frequency_penalty: 0.2 },
            { presencePenalty: { scale: 0.4 }, frequencyPenalty: { scale: 0.2 } },
        ],
    ] satisfies [string, ModelOptions, object][])(
        'serializes supported fields for %s',
        (model, model_options, expected) => {
            const driver = new BedrockDriver({ region: 'us-east-1' });
            const request = driver.preparePayload(
                { modelId: model, messages: [{ role: 'user', content: [{ text: 'hello' }] }] },
                { model, model_options },
            );
            expect(request.additionalModelRequestFields).toEqual(expected);
        },
    );
});
