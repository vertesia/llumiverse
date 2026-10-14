import type { ExecutionOptions } from '@llumiverse/core';
import { describe, expect, test } from 'vitest';
import { formatNovaImageGenerationPayload, NovaImageGenerationTaskType } from '../src/bedrock/nova-image-payload.js';
import { testPrompt_textToImage } from './samples.js';

const imageOutputModality = 'image' as NonNullable<ExecutionOptions['output_modality']>;

// Live Nova Canvas generation tests were removed: amazon.nova-canvas-v1:0 has reached end of life on Bedrock.
describe('Nova Canvas payload', () => {
    test('generate a prompt Nova canvas', async () => {
        const options: ExecutionOptions = {
            model: 'amazon.nova-canvas-v1:0',
            output_modality: imageOutputModality,
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
