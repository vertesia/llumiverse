import { type ExecutionOptions, type ExecutionResponse, type Providers, resolveModelProfile } from '@llumiverse/core';
import { describe, expect, test } from 'vitest';
import { assertCompletionOk, assertProviderCostReported, assertStreamingCompletionOk } from './assertions.js';
import { selectedDrivers } from './live-smoke-drivers.js';
import {
    testPrompt_color,
    testPrompt_describeImage,
    testSchema_animalDescription,
    testSchema_color,
} from './samples.js';
import {
    getLowEffortTestOptions,
    parseCompletionResultsToJson,
    retryUnlessRateLimited,
    withRateLimitBackoff,
} from './utils.js';

// Room for the 8192-token answers below: a slow reasoning model can take over a minute to produce them.
const TIMEOUT = 90 * 1000;
/**
 * Room for reasoning models (Qwen, MiniMax, DeepSeek) to think before they answer. 8192 is the lowest documented
 * output limit among the smoke models.
 */
const SMOKE_MAX_TOKENS = 8192;

function getTestOptions(model: string, provider: string): ExecutionOptions {
    const isGemini35FlashLite = model.toLowerCase().includes('gemini-3.5-flash-lite');

    // Exercises option mapping on models that advertise these controls; others drop them.
    return getLowEffortTestOptions(model, provider, {
        max_tokens: SMOKE_MAX_TOKENS,
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

// Drivers and their models run in parallel. The tests for one model stay in order (`concurrent: false`) so a
// throttled model is never sent several requests at once.
describe.concurrent.each(selectedDrivers)('Driver $name', ({ name, driver, models }) => {
    test(`${name}: list models`, { timeout: TIMEOUT, retry: 1, concurrent: false }, async () => {
        const r = await driver.listModels();
        console.log(r);
        expect(r.length).toBeGreaterThan(0);
    });

    describe.each(models)('%s', (model) => {
        // Offline: checks image prompt formatting, including for models the live image test below skips.
        test(`${name}: multimodal prompt generation for ${model}`, { concurrent: false }, async () => {
            const p = await driver.createPrompt(testPrompt_describeImage, { model });
            expect(p).toBeDefined();
        });

        test(`${name}: execute prompt on ${model}`, {
            timeout: TIMEOUT,
            retry: retryUnlessRateLimited(2),
            concurrent: false,
        }, async () => {
            const r = await withRateLimitBackoff(() =>
                driver.execute(testPrompt_color, getTestOptions(model, driver.provider)),
            );
            console.log(`Result for execute ${model}`, JSON.stringify(r));
            assertCompletionOk(r, model, driver);
            assertProviderCostReported(r, driver);
        });

        test(`${name}: execute prompt with streaming and schema on ${model}`, {
            timeout: TIMEOUT,
            retry: retryUnlessRateLimited(2),
            concurrent: false,
        }, async () => {
            const { r, out } = await withRateLimitBackoff(async () => {
                const stream = await driver.stream(testPrompt_color, {
                    ...getTestOptions(model, driver.provider),
                    result_schema: testSchema_color,
                });
                return { r: stream, out: await assertStreamingCompletionOk(stream, true) };
            });
            console.log(`Result for streaming with schema ${model}`, JSON.stringify(out));
            const json = parseCompletionResultsToJson(r.completion?.result ?? []) as { color?: unknown };
            expect(json.color).toEqual(expect.stringMatching(/\S/));
            assertProviderCostReported(r.completion as ExecutionResponse, driver);
        });

        test(`${name}: multimodal test - describe image with ${model}`, {
            timeout: TIMEOUT,
            retry: retryUnlessRateLimited(2),
            concurrent: false,
        }, async (context) => {
            // Image input comes from the model capability catalog, not listModels: several drivers' listings
            // do not say whether a model takes images.
            if (resolveModelProfile(model, driver.provider as Providers).capabilities.input.image !== true) {
                context.skip();
                return;
            }

            const r = await withRateLimitBackoff(() =>
                driver.execute(testPrompt_describeImage, {
                    ...getLowEffortTestOptions(model, driver.provider, { max_tokens: SMOKE_MAX_TOKENS }),
                    result_schema: testSchema_animalDescription,
                }),
            );
            console.log('Result', r);
            assertCompletionOk(r);
        });
    });
});
