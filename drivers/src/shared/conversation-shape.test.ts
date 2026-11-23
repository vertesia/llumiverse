import type { GoogleGenAI } from '@google/genai';
import { LlumiverseError, type Logger, PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIDriver } from '../openai/openai.js';
import { OpenAIChatCompletionsDriver } from '../openai/openai_chat_completions.js';
import { VertexAIDriver } from '../vertexai/index.js';
import { assertChatCompletionsConversation, assertConversationArray } from './conversation-shape.js';

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

/** A conversation as the Responses drivers store it: the item array, wrapped with the turn metadata. */
const RESPONSES_CONVERSATION = {
    _arrayConversation: [
        { role: 'user', content: 'Hi' },
        { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Hello' }] },
    ],
    _llumiverse_meta: { turnNumber: 1 },
};

/** A conversation as the Chat Completions drivers store it. */
const CHAT_COMPLETIONS_CONVERSATION = {
    _is_openai_chat_completions: true,
    messages: [
        { role: 'user', content: 'Hi' },
        { role: 'assistant', content: 'Hello' },
    ],
    _llumiverse_meta: { turnNumber: 1 },
};

function expectForeignChatConversation(error: unknown, provider: string) {
    expect(error).toBeInstanceOf(LlumiverseError);
    expect(error).toMatchObject({
        name: 'InvalidConversationError',
        code: 400,
        retryable: false,
        context: { provider },
        message: `[${provider}] The conversation was not written by a Chat Completions driver`,
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
        ['a Chat Completions conversation', CHAT_COMPLETIONS_CONVERSATION],
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

describe('assertChatCompletionsConversation', () => {
    it.each([
        ['no conversation', undefined],
        ['null', null],
        ['a Chat Completions conversation', CHAT_COMPLETIONS_CONVERSATION],
        ['a legacy array of messages', [{ role: 'user', content: 'Hi' }]],
    ])('accepts %s', (_label, conversation) => {
        expect(() => assertChatCompletionsConversation(conversation, context)).not.toThrow();
    });

    it.each([
        ['a Responses conversation', RESPONSES_CONVERSATION],
        ['an object without messages', { '': [null] }],
        ['an object whose messages is not an array', { messages: 'Hi' }],
        ['a string', 'Hi'],
        ['a number', 0],
    ])('refuses %s as a non-retryable 400', (_label, conversation) => {
        let thrown: unknown;
        try {
            assertChatCompletionsConversation(conversation, context);
        } catch (error) {
            thrown = error;
        }
        expectForeignChatConversation(thrown, 'openai');
    });
});

describe('a Responses conversation sent to a Chat Completions driver', () => {
    function chatDriver() {
        const driver = new OpenAIChatCompletionsDriver({
            apiKey: 'test-key',
            endpoint: 'https://compatible.example/v1',
            logger: testLogger(),
        });
        return { driver, create: vi.spyOn(driver.service.chat.completions, 'create') };
    }

    it('fails an execution before calling the provider', async () => {
        const { driver, create } = chatDriver();

        const error = await driver
            .execute(segments, { model: 'some-model', conversation: RESPONSES_CONVERSATION })
            .catch((e: unknown) => e);

        expectForeignChatConversation(error, 'openai_compatible');
        expect(create).not.toHaveBeenCalled();
    });

    it('fails a stream before calling the provider', async () => {
        const { driver, create } = chatDriver();

        const error = await (async () => {
            const stream = await driver.stream(segments, { model: 'some-model', conversation: RESPONSES_CONVERSATION });
            for await (const _chunk of stream) {
                // drain
            }
        })().catch((e: unknown) => e);

        expectForeignChatConversation(error, 'openai_compatible');
        expect(create).not.toHaveBeenCalled();
    });
});
