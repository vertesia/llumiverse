/**
 * Tests for multi-turn conversation handling across drivers.
 *
 * These tests verify that:
 * 1. Multi-turn conversations keep context when execute() and stream() turns are mixed
 * 2. Conversations with images work correctly
 * 3. The conversation object survives JSON.stringify/parse (no Uint8Array corruption)
 * 4. Claude prompt caching reports cached tokens on a later turn
 */

import {
    type AbstractDriver,
    type DataSource,
    type ExecutionOptions,
    PromptRole,
    type PromptSegment,
} from '@llumiverse/core';
import 'dotenv/config';
import { randomUUID } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, test } from 'vitest';
import { AnthropicDriver, BedrockDriver, OpenAIDriver, VertexAIDriver, xAIDriver } from '../src/index.js';
import { answerText } from './assertions.js';
import { getLowEffortTestOptions } from './utils.js';

const TIMEOUT = 120 * 1000;

interface TestDriver {
    driver: AbstractDriver;
    textModel: string;
    visionModel?: string;
    name: string;
}

const drivers: TestDriver[] = [];

if (process.env.GOOGLE_PROJECT_ID && process.env.GOOGLE_REGION) {
    drivers.push({
        name: 'vertexai-gemini',
        driver: new VertexAIDriver({
            project: process.env.GOOGLE_PROJECT_ID as string,
            region: process.env.GOOGLE_REGION as string,
        }),
        textModel: 'locations/global/publishers/google/models/gemini-3.5-flash-lite',
        visionModel: 'locations/global/publishers/google/models/gemini-3.5-flash-lite',
    });
    drivers.push({
        name: 'vertexai-claude',
        driver: new VertexAIDriver({
            project: process.env.GOOGLE_PROJECT_ID as string,
            region: process.env.GOOGLE_REGION as string,
        }),
        textModel: 'locations/global/publishers/anthropic/models/claude-haiku-5-5',
        visionModel: 'locations/global/publishers/anthropic/models/claude-haiku-5-5',
    });
} else {
    console.warn('VertexAI tests are skipped: GOOGLE_PROJECT_ID environment variable is not set');
}

if (process.env.OPENAI_API_KEY) {
    drivers.push({
        name: 'openai',
        driver: new OpenAIDriver({
            apiKey: process.env.OPENAI_API_KEY as string,
        }),
        textModel: 'gpt-6-luna',
        visionModel: 'gpt-6-luna',
    });
} else {
    console.warn('OpenAI tests are skipped: OPENAI_API_KEY environment variable is not set');
}

if (process.env.BEDROCK_REGION) {
    drivers.push({
        name: 'bedrock-claude',
        driver: new BedrockDriver({
            region: process.env.BEDROCK_REGION as string,
        }),
        textModel: 'global.anthropic.claude-haiku-5-5',
        visionModel: 'global.anthropic.claude-haiku-5-5',
    });
} else {
    console.warn('Bedrock tests are skipped: BEDROCK_REGION environment variable is not set');
}

if (process.env.XAI_API_KEY) {
    drivers.push({
        name: 'xai',
        driver: new xAIDriver({
            apiKey: process.env.XAI_API_KEY as string,
        }),
        textModel: 'grok-4.7',
        visionModel: 'grok-4.7',
    });
} else {
    console.warn('xAI tests are skipped: XAI_API_KEY environment variable is not set');
}

if (process.env.ANTHROPIC_API_KEY) {
    drivers.push({
        name: 'anthropic-claude',
        driver: new AnthropicDriver({
            apiKey: process.env.ANTHROPIC_API_KEY as string,
        }),
        textModel: 'claude-haiku-5-5',
        visionModel: 'claude-haiku-5-5',
    });
} else {
    console.warn('Anthropic tests are skipped: ANTHROPIC_API_KEY environment variable is not set');
}

const hasClaudeDrivers = drivers.some((driver) => driver.name.includes('claude'));

/**
 * DataSource implementation that provides pre-fetched image data as base64.
 * This more closely simulates how Studio sends images through the conversation.
 */
class Base64ImageSource implements DataSource {
    private base64Data: string;

    constructor(
        base64Data: string,
        public mime_type: string = 'image/png',
        private imageName: string = 'image',
    ) {
        this.base64Data = base64Data;
    }

    get name() {
        return this.imageName;
    }

    async getURI(): Promise<string> {
        // Return as data URL
        return `data:${this.mime_type};base64,${this.base64Data}`;
    }
    async getURL(): Promise<string> {
        return this.getURI();
    }

    async getStream(): Promise<ReadableStream<string | Uint8Array>> {
        // Convert base64 to Uint8Array and return as stream
        const binaryString = atob(this.base64Data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return new ReadableStream({
            start(controller) {
                controller.enqueue(bytes);
                controller.close();
            },
        });
    }

    /** Loads an image committed next to this test, so the test never depends on a remote host. */
    static fromTestFile(file: string, mimeType: string = 'image/png'): Base64ImageSource {
        const base64 = readFileSync(new URL(`./${file}`, import.meta.url)).toString('base64');
        return new Base64ImageSource(base64, mimeType, file);
    }
}

/** Returns execution options for a Claude model with prompt caching enabled. */
function getClaudeOptionsWithCache(model: string, driverName: string): ExecutionOptions {
    const optionId = driverName.includes('vertexai') ? 'vertexai-claude' : 'bedrock-claude';
    return {
        model,
        model_options: {
            _option_id: optionId,
            max_tokens: 1000,
            effort: 'low',
            cache_enabled: true,
        },
    };
}

/**
 * About 6.7k tokens with Claude Haiku 5.5's tokenizer. Vertex's global and multi-region endpoints route a request
 * to the region holding its cache only when they recognise the prompt; with prompts under roughly 4.7k tokens a
 * later turn that extends the conversation often lands elsewhere and misses. Larger prompts read reliably, as
 * real conversations do. The run ID keeps the prefix unique, so the test cannot pass on an earlier run's cache.
 */
function buildCacheableContext(): string {
    const repeatedSection =
        'Prompt caching verification context. Preserve this exact context across turns and do not summarize it unless explicitly asked. ';
    return `You are participating in prompt caching verification run ${randomUUID()}.\n\n${repeatedSection.repeat(180)}`;
}

/**
 * Verify that a conversation object can survive JSON serialization.
 * This catches Uint8Array corruption issues where bytes become { "0": 137, "1": 80, ... }
 */
function verifyConversationSerializable(conversation: unknown, driverName: string): void {
    const serialized = JSON.stringify(conversation);
    const deserialized = JSON.parse(serialized);

    // Check that no Uint8Array was corrupted into an object like { "0": 137, "1": 80 }
    // Such corrupted objects would have numeric string keys
    const checkForCorruptedBytes = (obj: unknown, path: string = ''): void => {
        if (obj === null || obj === undefined) return;

        if (typeof obj === 'object' && !Array.isArray(obj)) {
            const keys = Object.keys(obj as Record<string, unknown>);
            // If all keys are numeric strings, this might be a corrupted Uint8Array
            if (keys.length > 10 && keys.every((k) => /^\d+$/.test(k))) {
                throw new Error(
                    `[${driverName}] Possible corrupted Uint8Array at ${path}: ` +
                        `Found object with ${keys.length} numeric keys. ` +
                        `This suggests binary data was not properly stripped before serialization.`,
                );
            }
            for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
                checkForCorruptedBytes(value, `${path}.${key}`);
            }
        }

        if (Array.isArray(obj)) {
            obj.forEach((item, i) => {
                checkForCorruptedBytes(item, `${path}[${i}]`);
            });
        }
    };

    checkForCorruptedBytes(deserialized, 'conversation');
}

// Skip driver-specific tests if no drivers are configured
const hasDrivers = drivers.length > 0;

// Fallback test suite when no drivers are configured (prevents "no test suite found" error)
describe.skipIf(hasDrivers)('Multi-turn Conversations (no drivers configured)', () => {
    test('skipped - no API keys configured', () => {
        console.warn('All conversation tests skipped: No API keys configured for any driver');
        expect(true).toBe(true);
    });
});

describe.skipIf(hasClaudeDrivers)('Claude prompt caching (no Claude drivers configured)', () => {
    test('skipped - no Claude drivers configured', () => {
        console.warn('Claude prompt caching tests skipped: No Claude drivers configured');
        expect(true).toBe(true);
    });
});

async function consumeStream(stream: AsyncIterable<string>): Promise<void> {
    for await (const _chunk of stream) {
        // Consume the stream so the completion is populated
    }
}

describe.skipIf(!hasDrivers).concurrent.each(drivers)(
    'Driver $name - Multi-turn Conversations',
    ({ name, driver, textModel, visionModel }) => {
        /**
         * One conversation that alternates execute() → stream() → execute(), round-tripping the conversation
         * through JSON between turns as Studio does. This covers non-streaming and streaming turns that continue
         * a stored conversation, and conversations produced by either mode being accepted by the other.
         */
        test(`${name}: multi-turn conversation mixing execute and stream`, { timeout: TIMEOUT }, async () => {
            const options = getLowEffortTestOptions(textModel, driver.provider, { max_tokens: 1000 });

            // A plain personal fact: credential-like tokens ("Remember this code: ...") trip Gemini's safety filter.
            const result1 = await driver.execute(
                [{ role: PromptRole.user, content: 'I went to Paris last summer.' }],
                options,
            );
            expect(result1.result.length).toBeGreaterThan(0);
            verifyConversationSerializable(result1.conversation, name);
            const storedConversation1 = JSON.parse(JSON.stringify(result1.conversation));

            const stream2 = await driver.stream(
                [{ role: PromptRole.user, content: 'Which city did I go to last summer?' }],
                {
                    ...options,
                    conversation: storedConversation1,
                },
            );
            await consumeStream(stream2);
            const text2 = stream2.completion ? answerText(stream2.completion) : '';
            expect(text2).toContain('Paris');
            verifyConversationSerializable(stream2.completion?.conversation, name);
            const storedConversation2 = JSON.parse(JSON.stringify(stream2.completion?.conversation));

            const result3 = await driver.execute(
                [
                    {
                        role: PromptRole.user,
                        content: 'Which country is that city in? Answer with the country name only.',
                    },
                ],
                { ...options, conversation: storedConversation2 },
            );
            const text3 = answerText(result3);
            expect(text3).toContain('France');
            verifyConversationSerializable(result3.conversation, name);
        });

        test.skipIf(!name.includes('claude'))(
            `${name}: multi-turn prompt caching reports token usage`,
            { timeout: TIMEOUT, retry: 1 },
            async () => {
                const options = getClaudeOptionsWithCache(textModel, name);

                // The conversation cache point sits on the message before the last one once the history holds a
                // prior exchange, so turn 3 writes the cache and turn 4 is the first turn that can read it.
                let conversation: unknown;
                const cacheWriteTokensByTurn: number[] = [];
                let lastCachedTokens = 0;
                for (let turn = 1; turn <= 4; turn++) {
                    const prompt: PromptSegment[] =
                        turn === 1
                            ? [
                                  {
                                      role: PromptRole.system,
                                      content: 'Respond briefly. Preserve conversation context across turns.',
                                  },
                                  {
                                      role: PromptRole.user,
                                      content: `${buildCacheableContext()}\n\nAcknowledge with exactly: TURN-1-OK`,
                                  },
                              ]
                            : [{ role: PromptRole.user, content: `Reply with exactly: TURN-${turn}-OK` }];
                    const result = await driver.execute(prompt, { ...options, conversation });
                    expect(result.token_usage).toBeDefined();
                    verifyConversationSerializable(result.conversation, name);
                    conversation = JSON.parse(JSON.stringify(result.conversation));
                    cacheWriteTokensByTurn.push(result.token_usage?.prompt_cache_write ?? 0);
                    lastCachedTokens = result.token_usage?.prompt_cached ?? 0;
                }

                expect(
                    lastCachedTokens,
                    `[${name}] Expected prompt_cached tokens on turn 4 when cache_enabled=true. ` +
                        `Observed prompt_cache_write tokens by turn: ${cacheWriteTokensByTurn.join(', ')}`,
                ).toBeGreaterThan(0);
            },
        );

        /** Two images in one turn, then a follow-up after the images were stripped from the stored conversation. */
        test.skipIf(!visionModel)(`${name}: conversation with multiple images`, { timeout: TIMEOUT }, async () => {
            if (!visionModel) {
                throw new Error(`Expected vision model for ${name}`);
            }
            const options = getLowEffortTestOptions(visionModel, driver.provider, { max_tokens: 1000 });

            // Send the images as base64, as Studio does. Both were drawn with ImageMagick for these tests
            // (512x512, one solid shape on white), so they carry no third-party content.
            const circleImage = Base64ImageSource.fromTestFile('red_circle.png');
            const triangleImage = Base64ImageSource.fromTestFile('blue_triangle.png');

            const result1 = await driver.execute(
                [
                    {
                        role: PromptRole.user,
                        content: "I'm showing you two images. Name the shape and its colour in each one.",
                        files: [circleImage, triangleImage],
                    },
                ],
                options,
            );
            expect(result1.result.length).toBeGreaterThan(0);
            const text1 = answerText(result1).toLowerCase();
            expect(text1).toContain('circle');
            expect(text1).toContain('triangle');

            // Critical: binary image data must not corrupt the stored conversation
            verifyConversationSerializable(result1.conversation, name);
            const storedConversation1 = JSON.parse(JSON.stringify(result1.conversation));

            const result2 = await driver.execute(
                [{ role: PromptRole.user, content: 'What colour was the triangle? Answer with the colour only.' }],
                { ...options, conversation: storedConversation1 },
            );
            expect(result2.result.length).toBeGreaterThan(0);
            const text2 = answerText(result2).toLowerCase();
            // The images were stripped from the stored conversation, so this relies on the earlier turn's context
            expect(text2).toContain('blue');
            verifyConversationSerializable(result2.conversation, name);
        });
    },
);
