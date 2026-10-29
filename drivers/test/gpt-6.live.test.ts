/** On-demand GPT compatibility checks; intentionally outside the CI live-test allowlist. */
import { type AbstractDriver, type ExecutionOptions, PromptRole, type PromptSegment } from '@llumiverse/core';
import 'dotenv/config';
import { assert, describe, expect, test } from 'vitest';
import { BedrockDriver, BedrockMantleDriver, OpenAIDriver } from '../src/index.js';
import { answerText } from './assertions.js';
import { selectLiveTestDrivers } from './live-model-selection.js';
import { parseCompletionResultsToJson } from './utils.js';

const models = ['gpt-6-astra', 'gpt-6-sol', 'gpt-6-luna', 'gpt-6.1-sol', 'gpt-6.1-terra'];
const drivers: { name: string; driver: AbstractDriver; models: string[] }[] = [];
if (process.env.OPENAI_API_KEY) {
    drivers.push({ name: 'openai', driver: new OpenAIDriver({ apiKey: process.env.OPENAI_API_KEY }), models });
}
if (process.env.BEDROCK_REGION) {
    drivers.push({
        name: 'bedrock',
        driver: new BedrockDriver({ region: process.env.BEDROCK_REGION }),
        models: models.map((model) => `${model === 'gpt-6.1-terra' ? 'global' : 'us'}.openai.${model}`),
    });
}
if (process.env.BEDROCK_MANTLE_REGION) {
    drivers.push({
        name: 'bedrock-mantle',
        driver: new BedrockMantleDriver({ region: process.env.BEDROCK_MANTLE_REGION }),
        models: models.map((model) => `openai.${model}`),
    });
}
const selected = selectLiveTestDrivers(drivers, {
    providers: process.env.LLUMIVERSE_LIVE_PROVIDERS,
    models: process.env.LLUMIVERSE_LIVE_MODELS,
});

async function run(driver: AbstractDriver, prompt: PromptSegment[], options: ExecutionOptions, streaming: boolean) {
    if (!streaming) return driver.execute(prompt, options);
    const stream = await driver.stream(prompt, options);
    for await (const _chunk of stream) {
        /* Consume stream to finalize the native conversation. */
    }
    assert(stream.completion, 'Stream must produce a completion');
    return stream.completion;
}

describe.each(selected)('GPT compatibility on $name', ({ driver, models }) => {
    test.each(models.flatMap((model) => [false, true].map((streaming) => ({ model, streaming }))))(
        'structured answer on $model when streaming=$streaming',
        { timeout: 90_000, retry: 0 },
        async ({ model, streaming }) => {
            const completion = await run(
                driver,
                [{ role: PromptRole.user, content: 'What color is grass?' }],
                {
                    model,
                    model_options: { max_tokens: 4096, effort: 'low', verbosity: 'low', include_thoughts: true },
                    result_schema: {
                        type: 'object',
                        properties: { color: { type: 'string' } },
                        required: ['color'],
                        additionalProperties: false,
                    },
                },
                streaming,
            );
            expect(completion.error).toBeFalsy();
            expect(completion.finish_reason).toBe('stop');
            expect(parseCompletionResultsToJson(completion.result)).toEqual({ color: expect.stringMatching(/green/i) });
        },
    );

    test.each(models)('streamed tool call and persisted replay on %s', { timeout: 90_000, retry: 0 }, async (model) => {
        const options: ExecutionOptions = {
            model,
            model_options: { max_tokens: 4096, effort: 'low', verbosity: 'low', include_thoughts: true },
            tools: [
                {
                    name: 'lookup_weather',
                    description: 'Look up current weather in a city.',
                    input_schema: {
                        type: 'object',
                        properties: { city: { type: 'string' } },
                        required: ['city'],
                        additionalProperties: false,
                    },
                },
            ],
        };
        const first = await run(
            driver,
            [{ role: PromptRole.user, content: 'Call lookup_weather for Paris now.' }],
            options,
            true,
        );
        expect(first.error).toBeFalsy();
        expect(first.finish_reason).toBe('tool_use');
        const call = first.tool_use?.find((tool) => tool.tool_name === 'lookup_weather');
        assert(call, 'Expected a lookup_weather tool call');
        expect(first.conversation).toBeDefined();
        const second = await run(
            driver,
            [
                {
                    role: PromptRole.tool,
                    tool_use_id: call.id,
                    content: 'The current temperature is 17 degrees Celsius.',
                },
            ],
            { ...options, conversation: JSON.parse(JSON.stringify(first.conversation)) },
            false,
        );
        expect(second.error).toBeFalsy();
        expect(second.finish_reason).toBe('stop');
        expect(answerText(second)).toContain('17');
        expect(answerText(second)).not.toContain('[Redacted thinking:');
    });
});
