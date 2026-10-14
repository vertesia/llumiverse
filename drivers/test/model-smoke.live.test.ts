import {
    type AbstractDriver,
    type ExecutionOptions,
    type ExecutionResponse,
    getMaxOutputTokens,
    getMaxTokensLimitBedrock,
    getMaxTokensLimitVertexAi,
    PromptRole,
    type PromptSegment,
    type Providers,
    resolveModelProfile,
} from '@llumiverse/core';
import 'dotenv/config';
import { GoogleAuth } from 'google-auth-library';
import { describe, expect, test } from 'vitest';
import {
    AzureOpenAIDriver,
    BedrockDriver,
    BedrockMantleDriver,
    GroqDriver,
    MistralAIDriver,
    OpenAIDriver,
    OpenRouterDriver,
    TogetherAIDriver,
    VertexAIDriver,
    WatsonxDriver,
    xAIDriver,
} from '../src/index.js';
import { assertCompletionOk, assertProviderCostReported, assertStreamingCompletionOk } from './assertions.js';
import { selectLiveTestDrivers } from './live-model-selection.js';
import {
    testPrompt_color,
    testPrompt_describeImage,
    testSchema_animalDescription,
    testSchema_color,
} from './samples.js';
import { getLowEffortTestOptions, retryUnlessRateLimited, withRateLimitBackoff } from './utils.js';

const TIMEOUT = 60 * 1000;

interface TestDriver {
    driver: AbstractDriver;
    models: string[];
    name: string;
}

const drivers: TestDriver[] = [];

if (process.env.GOOGLE_PROJECT_ID && process.env.GOOGLE_REGION) {
    new GoogleAuth();
    drivers.push({
        name: 'google-vertex',
        driver: new VertexAIDriver({
            project: process.env.GOOGLE_PROJECT_ID as string,
            region: process.env.GOOGLE_REGION as string,
        }),
        models: [
            'locations/global/publishers/google/models/gemini-3.5-flash-lite',
            'locations/global/publishers/anthropic/models/claude-haiku-5-5',
        ],
    });
} else {
    console.warn('Google Vertex tests are skipped: GOOGLE_PROJECT_ID environment variable is not set');
}

if (process.env.MISTRAL_API_KEY) {
    drivers.push({
        name: 'mistralai',
        driver: new MistralAIDriver({
            apiKey: process.env.MISTRAL_API_KEY as string,
            endpoint_url: (process.env.MISTRAL_ENDPOINT_URL as string) ?? undefined,
        }),
        models: ['mistral-small-latest'],
    });
} else {
    console.warn('MistralAI tests are skipped: MISTRAL_API_KEY environment variable is not set');
}

if (process.env.TOGETHER_API_KEY) {
    drivers.push({
        name: 'togetherai',
        driver: new TogetherAIDriver({
            apiKey: process.env.TOGETHER_API_KEY as string,
        }),
        // A cheap current serverless model; Together serves gpt-oss-20b only on dedicated endpoints.
        models: ['Qwen/Qwen3.5-9B'],
    });
} else {
    console.warn('TogetherAI tests are skipped: TOGETHER_API_KEY environment variable is not set');
}

if (process.env.OPENAI_API_KEY) {
    drivers.push({
        name: 'openai',
        driver: new OpenAIDriver({
            apiKey: process.env.OPENAI_API_KEY as string,
        }),
        models: ['gpt-6-luna'],
    });
} else {
    console.warn('OpenAI tests are skipped: OPENAI_API_KEY environment variable is not set');
}

// Azure OpenAI addresses models by deployment name; the deployment must exist on the configured resource.
const AZURE_OPENAI_MODELS = ['gpt-6-luna'];

if (process.env.AZURE_OPENAI_KEY && process.env.AZURE_OPENAI_ENDPOINT) {
    drivers.push({
        name: 'azure-openai-key',
        driver: new AzureOpenAIDriver({
            apiKey: process.env.AZURE_OPENAI_KEY as string,
            endpoint: process.env.AZURE_OPENAI_ENDPOINT as string,
            deployment: process.env.AZURE_OPENAI_DEPLOYMENT as string,
        }),
        models: AZURE_OPENAI_MODELS,
    });
} else {
    console.warn('Azure OpenAI tests are skipped: AZURE_OPENAI_KEY environment variable is not set');
}

if (process.env.BEDROCK_REGION) {
    drivers.push({
        name: 'bedrock',
        driver: new BedrockDriver({
            region: process.env.BEDROCK_REGION as string,
        }),
        // One global and one regional (us.) inference profile.
        models: ['global.anthropic.claude-haiku-5-5', 'us.amazon.nova-2-lite-v1:0'],
    });
} else {
    console.warn('Bedrock tests are skipped: BEDROCK_REGION environment variable is not set');
}

if (process.env.BEDROCK_MANTLE_REGION) {
    drivers.push({
        name: 'bedrock-mantle',
        driver: new BedrockMantleDriver({ region: process.env.BEDROCK_MANTLE_REGION }),
        // One model per Mantle protocol: Responses (gpt-6-luna), Chat Completions (gpt-oss-20b) and Messages (Claude).
        // Mantle does not list Claude Haiku 5.5 yet; Haiku 4.5 is its smallest Claude model.
        models: ['openai.gpt-6-luna', 'openai.gpt-oss-20b', 'anthropic.claude-haiku-4-5'],
    });
} else {
    console.warn('Bedrock Mantle tests are skipped: BEDROCK_MANTLE_REGION environment variable is not set');
}

if (process.env.GROQ_API_KEY) {
    drivers.push({
        name: 'groq',
        driver: new GroqDriver({
            apiKey: process.env.GROQ_API_KEY as string,
        }),
        // gpt-oss-20b is Groq's cheapest self-serve model; qwen3.8-27b covers Groq's Qwen output cap and effort levels.
        // Both take Groq's native structured outputs. Tool calling is covered in tools.live.test.ts.
        models: ['openai/gpt-oss-20b', 'qwen/qwen3.8-27b'],
    });
} else {
    console.warn('Groq tests are skipped: GROQ_API_KEY environment variable is not set');
}

if (process.env.WATSONX_API_KEY) {
    drivers.push({
        name: 'watsonx',
        driver: new WatsonxDriver({
            apiKey: process.env.WATSONX_API_KEY as string,
            projectId: process.env.WATSONX_PROJECT_ID as string,
            endpointUrl: process.env.WATSONX_ENDPOINT_URL as string,
        }),
        models: ['ibm/granite-4-h-small'],
    });
} else {
    console.warn('Watsonx tests are skipped: WATSONX_API_KEY environment variable is not set');
}

if (process.env.OPENROUTER_API_KEY) {
    drivers.push({
        name: 'openrouter',
        driver: new OpenRouterDriver({
            apiKey: process.env.OPENROUTER_API_KEY,
        }),
        // qwen/qwen3.8-flash was dropped: its upstream on OpenRouter answered 429 on most runs, outlasting the backoff.
        models: ['minimax/minimax-m3', 'deepseek/deepseek-v4-flash', 'google/gemini-3.5-flash-lite'],
    });
} else {
    console.warn('OpenRouter tests are skipped: OPENROUTER_API_KEY environment variable is not set');
}

if (process.env.XAI_API_KEY) {
    drivers.push({
        name: 'xai',
        driver: new xAIDriver({
            apiKey: process.env.XAI_API_KEY as string,
        }),
        models: ['grok-4.7'],
    });
} else {
    console.warn('xAI tests are skipped: XAI_API_KEY environment variable is not set');
}

const selectedDrivers = selectLiveTestDrivers(drivers, {
    providers: process.env.LLUMIVERSE_LIVE_PROVIDERS,
    models: process.env.LLUMIVERSE_LIVE_MODELS,
});

function getTestOptions(model: string, provider: string): ExecutionOptions {
    const isGemini35FlashLite = model.toLowerCase().includes('gemini-3.5-flash-lite');

    // Exercises option mapping on models that advertise these controls; others drop them.
    return getLowEffortTestOptions(model, provider, {
        max_tokens: 512,
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
            assertProviderCostReported(r.completion as ExecutionResponse, driver);
        });

        test(`${name}: max_tokens at documented limit on ${model}`, {
            timeout: TIMEOUT,
            retry: retryUnlessRateLimited(1),
            concurrent: false,
        }, async (context) => {
            // Use provider-specific limits where available; unknown Vertex limits must remain unknown.
            let limit: number | undefined;
            if (driver.provider === 'bedrock') {
                limit = getMaxTokensLimitBedrock(model);
            } else if (driver.provider === 'vertexai') {
                limit = getMaxTokensLimitVertexAi(model);
                if (limit === undefined) {
                    context.skip();
                    return;
                }
            }
            // Other providers: the provider-aware profile (e.g. Groq's own completion caps), else the canonical limit
            if (!limit) {
                limit =
                    resolveModelProfile(model, driver.provider as Providers).max_output_tokens ??
                    getMaxOutputTokens(model);
            }

            const shortPrompt: PromptSegment[] = [{ role: PromptRole.user, content: 'Say "ok".' }];
            const options = getLowEffortTestOptions(model, driver.provider, { max_tokens: limit });
            // Guard against the advertised-options filter dropping the option this test is about.
            expect((options.model_options as { max_tokens?: number } | undefined)?.max_tokens).toBe(limit);
            const r = await withRateLimitBackoff(() => driver.execute(shortPrompt, options));
            // If the provider rejects our limit value, r.error will be set
            expect(r.error).toBeFalsy();
            expect(r.finish_reason).toBeTruthy();
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
                    ...getLowEffortTestOptions(model, driver.provider, { max_tokens: 1024 }),
                    result_schema: testSchema_animalDescription,
                }),
            );
            console.log('Result', r);
            assertCompletionOk(r);
        });
    });
});
