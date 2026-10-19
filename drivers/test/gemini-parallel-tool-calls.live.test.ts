/**
 * Live round trip for a Gemini turn that calls one tool more than once.
 *
 * Every call must come back with its own tool-use id, and answering each id must produce a request
 * Vertex accepts: one function response per call, paired by Vertex's call id when it issued one.
 * Covers both a model that returns call ids and, where Vertex omits them, the generated-id path.
 */

import { type ExecutionOptions, PromptRole, type PromptSegment, type ToolUse } from '@llumiverse/core';
import 'dotenv/config';
import { describe, expect, test } from 'vitest';
import { VertexAIDriver } from '../src/index.js';
import { completionResultToString } from './utils.js';

const TIMEOUT = 120 * 1000;
const MODELS = [
    'locations/global/publishers/google/models/gemini-2.5-flash',
    'locations/global/publishers/google/models/gemini-3.1-flash-lite',
];
const CITIES = ['Paris', 'London', 'Tokyo'];

const tools: ExecutionOptions['tools'] = [
    {
        name: 'get_weather',
        description: 'Get the current weather for one city.',
        input_schema: {
            type: 'object',
            properties: { city: { type: 'string', description: 'The city name' } },
            required: ['city'],
        },
    },
];

const project = process.env.GOOGLE_PROJECT_ID;
const region = process.env.GOOGLE_REGION;

describe.skipIf(!project || !region)('Gemini parallel calls to one tool (live)', () => {
    const driver = new VertexAIDriver({ project: project ?? '', region: region ?? '' });

    // The model decides how many calls to make, so a run without a parallel turn is retried.
    test.each(MODELS)(
        '%s: every call gets its own id and each follow-up turn is accepted',
        { retry: 2, timeout: TIMEOUT },
        async (model) => {
            let segments: PromptSegment[] = [
                {
                    role: PromptRole.user,
                    content:
                        `Call get_weather once for each of these cities, all in this one turn: ` +
                        `${CITIES.join(', ')}. Do not answer until you have every result.`,
                },
            ];
            let conversation: unknown;
            const seenIds: string[] = [];
            let largestTurn = 0;
            let answer = '';
            for (let step = 0; step < 4; step++) {
                const stream = await driver.stream(segments, { model, tools, conversation });
                for await (const _chunk of stream) {
                    // Drain; the tool calls are on the completion.
                }
                // A rejected follow-up (e.g. a response/call count mismatch) fails here.
                expect(stream.completion?.error).toBeUndefined();
                conversation = stream.completion?.conversation;
                const calls = (stream.completion?.tool_use ?? []) as ToolUse<{ city?: string }>[];
                if (calls.length === 0) {
                    answer = (stream.completion?.result ?? []).map(completionResultToString).join('');
                    break;
                }
                console.info(`${model} call ids:`, calls.map((call) => call.id).join(', '));
                seenIds.push(...calls.map((call) => call.id));
                largestTurn = Math.max(largestTurn, calls.length);
                segments = calls.map((call) => ({
                    role: PromptRole.tool,
                    tool_use_id: call.id,
                    content: JSON.stringify({ city: call.tool_input?.city, forecast: 'sunny, 21C' }),
                    thought_signature: call.thought_signature,
                }));
            }

            expect(largestTurn).toBeGreaterThanOrEqual(2);
            expect(new Set(seenIds).size).toBe(seenIds.length);
            expect(answer.toLowerCase()).toContain('sunny');
        },
    );
});
