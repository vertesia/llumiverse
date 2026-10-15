import type { AIModel, ExecutionOptions } from '@llumiverse/core';
import { describe, expect, test } from 'vitest';
import { assertCompletionOk, assertStreamingCompletionOk } from './assertions.js';
import { selectedDrivers } from './live-smoke-drivers.js';
import {
    testPrompt_color,
    testPrompt_describeImage,
    testSchema_animalDescription,
    testSchema_color,
} from './samples.js';
import { getLowEffortTestOptions, parseCompletionResultsToJson } from './utils.js';

const TIMEOUT = 90 * 1000;

function getTestOptions(model: string, provider: string): ExecutionOptions {
    const isGemini35FlashLite = model.toLowerCase().includes('gemini-3.5-flash-lite');

    // Exercises option mapping on models that advertise these controls; others drop them.
    // max_tokens leaves room for reasoning models (Qwen, MiniMax, DeepSeek) to think before they answer.
    return getLowEffortTestOptions(model, provider, {
        max_tokens: 2048,
        top_k: 40,
        top_p: 0.7, //Some models do not support top_p = 1.0, set to 0.99 or lower.
        ...(isGemini35FlashLite
            ? {}
            : {
                  presence_penalty: 0.1, //Cohere Command R does not support using presence & frequency penalty at the same time
                  frequency_penalty: -0.1,
              }),
        stop_sequence: ['haemoglobin'],
    });
}

describe.each(selectedDrivers)('Driver $name', ({ name, driver, models }) => {
    let fetchedModels: AIModel[];

    test(`${name}: list models`, { timeout: TIMEOUT, retry: 1 }, async () => {
        const r = await driver.listModels();
        fetchedModels = r;
        console.log(r);
        expect(r.length).toBeGreaterThan(0);
    });

    // Offline: checks image prompt formatting even where the live image test below cannot tell the model is multimodal.
    test.each(models)(`${name}: multimodal prompt generation for %s`, async (model) => {
        const p = await driver.createPrompt(testPrompt_describeImage, { model });
        expect(p).toBeDefined();
    });

    test.each(models)(`${name}: execute prompt on %s`, { timeout: TIMEOUT, retry: 2 }, async (model) => {
        const r = await driver.execute(testPrompt_color, getTestOptions(model, driver.provider));
        console.log(`Result for execute ${model}`, JSON.stringify(r));
        assertCompletionOk(r, model, driver);
    });

    test.each(models)(
        `${name}: execute prompt with streaming and schema on %s`,
        {
            timeout: TIMEOUT,
            retry: 2,
        },
        async (model) => {
            const r = await driver.stream(testPrompt_color, {
                ...getTestOptions(model, driver.provider),
                result_schema: testSchema_color,
            });
            const out = await assertStreamingCompletionOk(r, true);
            console.log(`Result for streaming with schema ${model}`, JSON.stringify(out));
            const json = parseCompletionResultsToJson(r.completion?.result ?? []) as { color?: unknown };
            expect(json.color).toEqual(expect.stringMatching(/\S/));
        },
    );

    test.each(models)(
        `${name}: multimodal test - describe image with %s`,
        {
            timeout: TIMEOUT,
            retry: 2,
        },
        async (model) => {
            if (!fetchedModels) {
                fetchedModels = await driver.listModels();
            }

            const isMultiModal = fetchedModels?.find((r) => r.id === model)?.is_multimodal;

            console.log(`${model} is multimodal: ${isMultiModal}`);
            // Only some drivers' listModels report is_multimodal (not Vertex, Bedrock, Mantle, Azure, Watsonx or
            // OpenRouter), so this passes without a call there. Their image paths are covered by the conversation test.
            if (!isMultiModal) return;

            const r = await driver.execute(testPrompt_describeImage, {
                ...getLowEffortTestOptions(model, driver.provider, { max_tokens: 1024 }),
                result_schema: testSchema_animalDescription,
            });
            console.log('Result', r);
            assertCompletionOk(r);
        },
    );
});
