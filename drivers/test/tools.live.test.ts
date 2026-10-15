/**
 * Live tool-use round trip, ending with a checkpoint turn.
 *
 * 1. The model calls a tool.
 * 2. The tool result is sent back and the model uses it in its answer.
 * 3. A follow-up is sent with tools=[] while the conversation still holds tool call/result blocks, as a
 *    checkpoint summary does. Providers reject tool blocks without a tool config (e.g. Bedrock's "The toolConfig
 *    field must be defined when using toolUse and toolResult content blocks"), so each driver must convert them to
 *    text when no tools are provided.
 */

import { type AbstractDriver, type ExecutionOptions, PromptRole, type PromptSegment } from '@llumiverse/core';
import 'dotenv/config';
import { GoogleAuth } from 'google-auth-library';
import { describe, expect, test } from 'vitest';
import { BedrockDriver, GroqDriver, OpenAIDriver, VertexAIDriver } from '../src/index.js';
import {
    completionResultToString,
    getLowEffortTestOptions,
    retryUnlessRateLimited,
    withRateLimitBackoff,
} from './utils.js';

const TIMEOUT = 90 * 1000;

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
            project: process.env.GOOGLE_PROJECT_ID,
            region: process.env.GOOGLE_REGION,
        }),
        models: [
            // US multi-region endpoint; the other live tests use the global one.
            'locations/us/publishers/google/models/gemini-3.5-flash-lite',
            'locations/global/publishers/anthropic/models/claude-haiku-5-5',
        ],
    });
} else {
    console.warn('Google Vertex tests are skipped: GOOGLE_PROJECT_ID environment variable is not set');
}

if (process.env.OPENAI_API_KEY) {
    drivers.push({
        name: 'openai',
        driver: new OpenAIDriver({
            apiKey: process.env.OPENAI_API_KEY,
        }),
        models: ['gpt-6-luna'],
    });
} else {
    console.warn('OpenAI tests are skipped: OPENAI_API_KEY environment variable is not set');
}

if (process.env.BEDROCK_REGION) {
    drivers.push({
        name: 'bedrock',
        driver: new BedrockDriver({
            region: process.env.BEDROCK_REGION,
        }),
        models: ['global.anthropic.claude-haiku-5-5'],
    });
} else {
    console.warn('Bedrock tests are skipped: BEDROCK_REGION environment variable is not set');
}

if (process.env.GROQ_API_KEY) {
    drivers.push({
        name: 'groq',
        driver: new GroqDriver({
            apiKey: process.env.GROQ_API_KEY,
        }),
        models: ['openai/gpt-oss-20b'],
    });
} else {
    console.warn('Groq tests are skipped: GROQ_API_KEY environment variable is not set');
}

const TOOL_PROMPT: PromptSegment[] = [{ role: PromptRole.user, content: 'What is the weather in Paris?' }];

const TOOL_DEF = {
    name: 'get_weather',
    description: 'Get the current weather in a given location',
    input_schema: {
        type: 'object' as const,
        properties: {
            location: {
                type: 'string',
                description: 'The city to get the weather for, e.g. Paris, San Francisco etc.',
            },
        },
        required: ['location'],
    },
};

describe.concurrent.each(drivers)('Driver $name', ({ name, driver, models }) => {
    test.each(models)(
        `${name}: tool call, tool result, then tools=[] checkpoint for %s`,
        { timeout: TIMEOUT, retry: retryUnlessRateLimited(1) },
        async (model) => {
            const options = getLowEffortTestOptions(model, driver.provider, { max_tokens: 1000 });
            const toolOptions: ExecutionOptions = { ...options, tools: [TOOL_DEF] };

            // Step 1: the model calls the tool
            const toolCall = await withRateLimitBackoff(() => driver.execute(TOOL_PROMPT, toolOptions));
            const toolUse = toolCall.tool_use;
            if (!toolUse) {
                throw new Error('Expected tool use in first driver response');
            }
            expect(toolUse.length).toBe(1);
            expect(toolUse[0].id).toBeDefined();
            expect(toolUse[0].tool_input).toBeDefined();
            expect(toolUse[0].tool_name).toBe('get_weather');
            expect(toolCall.conversation).toBeDefined();

            // Step 2: the model uses the tool result
            const toolAnswer = await withRateLimitBackoff(() =>
                driver.execute(
                    [{ role: PromptRole.tool, tool_use_id: toolUse[0].id, content: '15 degrees celsius, sunny' }],
                    { ...toolOptions, conversation: toolCall.conversation },
                ),
            );
            const answer = toolAnswer.result.map(completionResultToString).join('');
            expect(answer, `[${model}] Tool result not found in response`).toContain('15');
            // Without the conversation the checkpoint would start fresh and never meet the tool blocks.
            expect(toolAnswer.conversation).toBeDefined();

            // Step 3: checkpoint summary with tools=[] over a conversation holding tool blocks
            const checkpoint = await withRateLimitBackoff(() =>
                driver.execute(
                    [
                        {
                            role: PromptRole.user,
                            content:
                                'Summarize what happened in this conversation. Do NOT call any tools — just output text.',
                        },
                    ],
                    { ...options, tools: [], conversation: toolAnswer.conversation },
                ),
            );
            expect(checkpoint.finish_reason).not.toBe('tool_use');
            expect(checkpoint.result.length).toBeGreaterThan(0);
            // The tool call and result must reach the model as text, not be dropped.
            const summary = checkpoint.result.map(completionResultToString).join('');
            expect(summary, `[${model}] Checkpoint summary lost the tool data`).toMatch(/15|paris/i);
        },
    );
});
