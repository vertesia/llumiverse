// biome-ignore lint/suspicious/noDeprecatedImports: Modalities.image is the supported way to request image output; the deprecation note refers to reading the result modality via CompletionResult.type
import { type ExecutionOptions, Modalities } from '@llumiverse/core';
import { describe, expect, test } from 'vitest';
import { formatNovaImageGenerationPayload, NovaImageGenerationTaskType } from '../src/bedrock/nova-image-payload.js';
import { testPrompt_textToImage } from './samples.js';

// Live Nova Canvas generation tests were removed: amazon.nova-canvas-v1:0 has reached end of life on Bedrock.
describe('Nova Canvas payload', () => {
    test('generate a prompt Nova canvas', async () => {
        const options: ExecutionOptions = {
            model: 'amazon.nova-canvas-v1:0',
            output_modality: Modalities.image,
            model_options: {
                _option_id: 'bedrock-nova-canvas',
                taskType: 'TEXT_IMAGE',
            },
        };
        const res = await formatNovaImageGenerationPayload(
            NovaImageGenerationTaskType.TEXT_IMAGE,
            testPrompt_textToImage,
            options,
        );
        expect(res).toBeDefined();
        expect(res).toHaveProperty('textToImageParams');
        expect(res).toHaveProperty('taskType');
    });
});
