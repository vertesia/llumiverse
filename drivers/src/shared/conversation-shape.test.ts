import type { GoogleGenAI } from '@google/genai';
import { LlumiverseError, type Logger, PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIDriver } from '../openai/openai.js';
import { VertexAIDriver } from '../vertexai/index.js';
import { assertConversationArray } from './conversation-shape.js';

const context = { provider: 'openai', model: 'gpt-test', operation: 'execute' } as const;
const segments = [{ role: PromptRole.user, content: 'Hi' }];

function testLogger() {
    return { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() } satisfies Logger;
}

function expectInvalidConversation(error: unknown, provider: string) {
    expect(error).toBeInstanceOf(LlumiverseError);
    expect(error).toMatchObject({
        name: 'InvalidConversationError',
        code: 400,
        retryable: false,
        context: { provider },
        message: `[${provider}] The conversation must be an array of messages`,
    });
}

describe('assertConversationArray', () => {
    it.each([
        ['no conversation', undefined],
        ['null', null],
        ['an empty array', []],
        ['an array of messages', [{ role: 'user', content: 'Hi' }]],
        ['a wrapped array', { _arrayConversation: [{ role: 'user', content: 'Hi' }], _llumiverse_meta: {} }],
    ])('accepts %s', (_label, conversation) => {
        expect(() => assertConversationArray(conversation, context)).not.toThrow();
    });

    it.each([
        ['an object', { '': [null] }],
        ['a string', 'Hi'],
        ['a number', 0],
    ])('refuses %s as a non-retryable 400', (_label, conversation) => {
        let thrown: unknown;
        try {
            assertConversationArray(conversation, context);
        } catch (error) {
            thrown = error;
        }
        expectInvalidConversation(thrown, 'openai');
    });
});

describe('a conversation that is not an array', () => {
    const conversation = { '': [null] };

    it('fails an OpenAI execution before calling the provider', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test-key', logger: testLogger() });
        const create = vi.spyOn(driver.service.responses, 'create');

        const error = await driver.execute(segments, { model: 'gpt-test', conversation }).catch((e: unknown) => e);

        expectInvalidConversation(error, 'openai');
        expect(create).not.toHaveBeenCalled();
    });

    it('fails an OpenAI stream before calling the provider', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test-key', logger: testLogger() });
        const create = vi.spyOn(driver.service.responses, 'create');

        const error = await (async () => {
            for await (const _chunk of await driver.stream(segments, { model: 'gpt-test', conversation })) {
                // drain
            }
        })().catch((e: unknown) => e);

        expectInvalidConversation(error, 'openai');
        expect(create).not.toHaveBeenCalled();
    });

    it.each([
        ['execution', false],
        ['stream', true],
    ])('fails a Gemini %s before calling the provider', async (_label, streaming) => {
        const generateContent = vi.fn();
        const generateContentStream = vi.fn();
        class TestVertexAIDriver extends VertexAIDriver {
            override getGoogleGenAIClient(): GoogleGenAI {
                return { models: { generateContent, generateContentStream } } as unknown as GoogleGenAI;
            }
        }
        const driver = new TestVertexAIDriver({ project: 'test-project', region: 'us-central1', logger: testLogger() });
        const options = { model: 'publishers/google/models/gemini-2.5-flash', conversation };

        const error = await (async () => {
            if (!streaming) return driver.execute(segments, options);
            for await (const _chunk of await driver.stream(segments, options)) {
                // drain
            }
        })().catch((e: unknown) => e);

        expectInvalidConversation(error, 'vertexai');
        expect(generateContent).not.toHaveBeenCalled();
        expect(generateContentStream).not.toHaveBeenCalled();
    });
});
