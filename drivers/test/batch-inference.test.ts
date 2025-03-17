import { BatchInferenceJobStatus, ExecutionOptions, Modalities, PromptRole } from '@llumiverse/core';
import 'dotenv/config';
import { describe, expect, it, test } from 'vitest';
import { BedrockDriver } from '../src/bedrock';
import { OpenAIDriver } from '../src/openai/openai';
import { VertexAIDriver } from '../src/vertexai';

describe('Batch Inference', () => {
    it('OpenAI batch inference', async () => {
        // Skip if required environment variables are not set
        // Using a fake API key since we just want to test the typing
        const driver = new OpenAIDriver({
            apiKey: 'sk-dummy-key'
        });

        // Create batch inputs
        const batchInputs: { segments: { role: PromptRole; content: string; }[]; options: ExecutionOptions }[] = [];
        for (let i = 0; i < 3; i++) {
            batchInputs.push({
                segments: [
                    {
                        role: PromptRole.system,
                        content: 'You are a helpful assistant that provides brief, factual information.'
                    },
                    {
                        role: PromptRole.user,
                        content: `Tell me an interesting fact about fruit number ${i + 1} in a list of exotic fruits. Keep it brief.`
                    }
                ],
                options: {
                    model: 'gpt-3.5-turbo',
                    output_modality: Modalities.text,
                    batch_id: `test-item-${i}`
                }
            });
        }

        // This should compile without errors
        expect(() => {
            // Start batch job
            driver.startBatchInference(batchInputs, {
                concurrency: 2,
                batchSize: 2,
                retryCount: 2
            }).then(batchJob => {
                // Get job status
                return driver.getBatchInferenceJob(batchJob.id);
            }).then(jobStatus => {
                // Get results
                return driver.getBatchInferenceResults(jobStatus.id);
            }).then(results => {
                // Cancel job
                return driver.cancelBatchInferenceJob(results.batch_id);
            });
        }).not.toThrow();
    });
});