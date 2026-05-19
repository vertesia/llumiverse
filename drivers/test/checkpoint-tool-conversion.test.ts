/**
 * Live integration tests for the checkpoint tool-block-to-text conversion fix.
 *
 * Bug: When tools=[] is passed but conversation contains tool call/result blocks
 * from prior turns, APIs reject the request (e.g. Bedrock's "The toolConfig field
 * must be defined when using toolUse and toolResult content blocks").
 *
 * Fix: Each driver converts tool blocks to text representations when no tools are
 * provided, preserving tool call data while satisfying API requirements.
 *
 * This test verifies the fix across all enabled drivers by:
 * 1. Running a tool call turn (to populate conversation with tool blocks)
 * 2. Then running a follow-up with tools=[] (checkpoint summary scenario)
 */

import { AbstractDriver, ExecutionOptions, PromptRole, PromptSegment } from '@llumiverse/core';
import 'dotenv/config';
import { GoogleAuth } from 'google-auth-library';
import { describe, expect, test } from 'vitest';
import { BedrockDriver, OpenAIDriver, VertexAIDriver } from '../src';

const TIMEOUT = 90_000;

interface TestDriver {
    driver: AbstractDriver;
    models: string[];
    name: string;
}

const drivers: TestDriver[] = [];

if (process.env.GOOGLE_PROJECT_ID && process.env.GOOGLE_REGION) {
    const _auth = new GoogleAuth();
    drivers.push({
        name: 'google-vertex',
        driver: new VertexAIDriver({
            project: process.env.GOOGLE_PROJECT_ID,
            region: process.env.GOOGLE_REGION,
        }),
        models: [
            'publishers/google/models/gemini-2.5-flash',
            'publishers/anthropic/models/claude-sonnet-4-5',
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
        models: [
            'gpt-4o-mini',
        ],
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
        models: [
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        ],
    });
} else {
    console.warn('Bedrock tests are skipped: BEDROCK_REGION environment variable is not set');
}

const TOOL_PROMPT: PromptSegment[] = [
    { role: PromptRole.user, content: 'What is the weather in Paris?' },
];

const TOOL_DEF = {
    name: 'get_weather',
    description: 'Get the current weather in a given location',
    input_schema: {
        type: 'object' as const,
        properties: {
            location: { type: 'string', description: 'The city name' },
        },
        required: ['location'],
    },
};

function getToolOptions(model: string): ExecutionOptions {
    return {
        model,
        model_options: {
            _option_id: 'text-fallback',
            max_tokens: 256,
            temperature: 0.3,
        },
        tools: [TOOL_DEF],
    };
}

describe.concurrent.each(drivers)('Driver $name - checkpoint tool conversion', ({ name, driver, models }) => {

    test.each(models)(`${name}: tools=[] with tool blocks in conversation for %s`, { timeout: TIMEOUT, retry: 1 }, async (model) => {
        // Step 1: Execute a tool call to get a conversation with tool blocks
        const toolOptions = getToolOptions(model);
        const toolResult = await driver.execute(TOOL_PROMPT, toolOptions);

        expect(toolResult.tool_use).toBeDefined();
        expect(toolResult.tool_use!.length).toBeGreaterThan(0);
        expect(toolResult.conversation).toBeDefined();

        // Step 2: Provide tool result to continue the conversation
        const toolResponse: PromptSegment = {
            role: PromptRole.tool,
            tool_use_id: toolResult.tool_use![0].id,
            content: '15 degrees celsius, sunny',
        };
        const continueResult = await driver.execute([toolResponse], {
            ...toolOptions,
            conversation: toolResult.conversation,
        });
        expect(continueResult.conversation).toBeDefined();

        // Step 3: Checkpoint scenario — send tools=[] with the conversation that has tool blocks
        // This is what createCheckpoint does when asking for a summary
        const checkpointPrompt: PromptSegment[] = [
            {
                role: PromptRole.user,
                content: 'Summarize what happened in this conversation. Do NOT call any tools — just output text.',
            },
        ];

        const checkpointResult = await driver.execute(checkpointPrompt, {
            model,
            model_options: {
                _option_id: 'text-fallback',
                max_tokens: 256,
                temperature: 0.3,
            },
            tools: [],  // No tools — checkpoint summary
            conversation: continueResult.conversation,
        });

        // Key assertions:
        // 1. The call succeeded (no API error about missing tool config)
        expect(checkpointResult).toBeDefined();
        // 2. finish_reason should NOT be tool_use (no tools were provided)
        expect(checkpointResult.finish_reason).not.toBe('tool_use');
        // 3. Should have text content back
        const hasContent = checkpointResult.result_text ||
            (checkpointResult.result && checkpointResult.result.length > 0);
        expect(hasContent).toBeTruthy();
    });
});
