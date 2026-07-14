/**
 * Unit tests for the Gemini conversation mutation bug fix.
 *
 * Bug: When tools=[] is passed but the conversation contains functionCall/functionResponse
 * parts from prior turns, getGeminiPayload() was doing:
 *
 *   prompt.contents = convertGeminiFunctionPartsToText(prompt.contents);
 *
 * Since prompt.contents is the same object reference as the caller's conversation array,
 * this permanently corrupted the stored conversation with "[Tool call: ...]" text markers.
 * On the next turn the model would see those markers in context and echo them as literal output.
 *
 * Fix: use a local `payloadContents` variable so the caller's conversation is never mutated.
 */

import { FinishReason } from '@google/genai';
import { type DataSource, type ExecutionOptions, PromptRole, type PromptSegment } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import type { GenerateContentPrompt, VertexAIDriver } from '../index.js';
import { convertGeminiFunctionPartsToText, GeminiModelDefinition, getGeminiPayload } from './gemini.js';

// ---------------------------------------------------------------------------
// Pure function tests — no driver needed
// ---------------------------------------------------------------------------

describe('convertGeminiFunctionPartsToText', () => {
    it('does not mutate the input array', () => {
        const original = [
            {
                role: 'model',
                parts: [{ functionCall: { name: 'plan', args: { task: 'write tests' } } }],
            },
            {
                role: 'user',
                parts: [{ functionResponse: { name: 'plan', response: { output: 'done' } } }],
            },
        ];
        const originalItemRefs = original.map((c) => c);
        const originalPartRefs = original.map((c) => c.parts[0]);

        const result = convertGeminiFunctionPartsToText(original);

        // Result must be a different array
        expect(result).not.toBe(original);
        // Original items must be unchanged (same references, not mutated)
        original.forEach((item, i) => {
            expect(item).toBe(originalItemRefs[i]);
            expect(item.parts[0]).toBe(originalPartRefs[i]);
        });
        // Original functionCall part must still be a functionCall, not text
        expect(original[0].parts[0]).toHaveProperty('functionCall');
        expect(original[0].parts[0]).not.toHaveProperty('text');
        expect(original[1].parts[0]).toHaveProperty('functionResponse');
        expect(original[1].parts[0]).not.toHaveProperty('text');
    });

    it('converts functionCall parts to the expected text format', () => {
        const contents = [
            {
                role: 'model',
                parts: [{ functionCall: { name: 'get_weather', args: { location: 'Paris' } } }],
            },
        ];

        const result = convertGeminiFunctionPartsToText(contents);

        expect(result[0].parts?.[0]).toEqual({
            text: '[Tool call: get_weather({"location":"Paris"})]',
        });
    });

    it('converts functionResponse parts to the expected text format', () => {
        const contents = [
            {
                role: 'user',
                parts: [{ functionResponse: { name: 'get_weather', response: { temperature: '15°C' } } }],
            },
        ];

        const result = convertGeminiFunctionPartsToText(contents);

        expect(result[0].parts?.[0]).toEqual({
            text: '[Tool result for get_weather: {"temperature":"15°C"}]',
        });
    });

    it('leaves non-function parts intact', () => {
        const textPart = { text: 'Hello world' };
        const contents = [{ role: 'user', parts: [textPart] }];

        const result = convertGeminiFunctionPartsToText(contents);

        expect(result[0].parts?.[0]).toBe(textPart);
    });
});

describe('Gemini thinking configuration', () => {
    const prompt = { contents: [{ role: 'user' as const, parts: [{ text: 'Hello' }] }] };

    it('omits thinkingConfig when Gemini 2.5 Flash thinking is disabled by default', () => {
        const payload = getGeminiPayload({ model: 'publishers/google/models/gemini-2.5-flash' }, prompt);

        expect(payload.config?.thinkingConfig).toBeUndefined();
    });

    it('omits thinkingConfig when an explicit zero budget disables thinking', () => {
        const payload = getGeminiPayload(
            {
                model: 'publishers/google/models/gemini-2.5-flash',
                model_options: {
                    _option_id: 'vertexai-gemini',
                    thinking_budget_tokens: 0,
                },
            },
            prompt,
        );

        expect(payload.config?.thinkingConfig).toBeUndefined();
    });

    it('includes thought summaries when Gemini thinking is enabled', () => {
        const payload = getGeminiPayload(
            {
                model: 'publishers/google/models/gemini-2.5-flash',
                model_options: {
                    _option_id: 'vertexai-gemini',
                    thinking_budget_tokens: 128,
                },
            },
            prompt,
        );

        expect(payload.config?.thinkingConfig).toEqual({ includeThoughts: true, thinkingBudget: 128 });
    });
});

// ---------------------------------------------------------------------------
// Integration-level tests — verify the driver does not mutate the conversation
// ---------------------------------------------------------------------------

function makeContentsWithFunctionParts() {
    return [
        { role: 'model', parts: [{ functionCall: { name: 'plan', args: { task: 'test' } } }] },
        { role: 'user', parts: [{ functionResponse: { name: 'plan', response: { result: 'ok' } } }] },
    ];
}

function makeDriver(overrides: {
    generateContent?: (request?: unknown) => Promise<unknown>;
    generateContentStream?: (request?: unknown) => Promise<AsyncIterable<unknown>>;
}) {
    return {
        logger: { warn: () => {}, info: () => {}, error: () => {} },
        getGoogleGenAIClient: () => ({
            models: {
                generateContent: overrides.generateContent ?? (async () => ({})),
                generateContentStream: overrides.generateContentStream ?? (async () => (async function* () {})()),
            },
        }),
    } as unknown as VertexAIDriver;
}

const mockNonStreamingResponse = {
    usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 },
    candidates: [
        {
            finishReason: FinishReason.STOP,
            content: { role: 'model', parts: [{ text: 'Summary.' }] },
            safetyRatings: [],
        },
    ],
};

const mockStreamingChunk = {
    usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 },
    candidates: [
        {
            finishReason: FinishReason.STOP,
            content: { role: 'model', parts: [{ text: 'Summary.' }] },
            safetyRatings: [],
        },
    ],
};

describe('GeminiModelDefinition - no conversation mutation', () => {
    it('preserves signatures on arbitrary Parts and empty terminal Parts across JSON roundtrip', async () => {
        const modelDef = new GeminiModelDefinition('gemini-3-flash');
        const nativeContent = {
            role: 'model',
            parts: [
                { text: 'plan', thought: true, thoughtSignature: 'thought-signature' },
                {
                    functionCall: { name: 'first', args: { value: 1 } },
                    thoughtSignature: 'call-signature',
                },
                { text: 'answer', thoughtSignature: 'text-signature' },
                { text: '', thoughtSignature: 'terminal-signature' },
            ],
        };
        const requests: unknown[] = [];
        const driver = makeDriver({
            generateContent: async (request) => {
                requests.push(request);
                return {
                    usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1, totalTokenCount: 2 },
                    candidates: [{ finishReason: FinishReason.STOP, content: nativeContent, safetyRatings: [] }],
                };
            },
        });
        const first = await modelDef.requestTextCompletion(
            driver,
            { contents: [{ role: 'user', parts: [{ text: 'question' }] }] },
            {
                model: 'publishers/google/models/gemini-3-flash',
                tools: [
                    { name: 'first', input_schema: { type: 'object' } },
                    { name: 'second', input_schema: { type: 'object' } },
                ],
                stripTextMaxTokens: 1,
                stripImagesAfterTurns: 0,
            },
        );
        expect(first.result).toEqual([
            { type: 'thoughts', value: 'plan' },
            { type: 'text', value: 'answer' },
        ]);

        const persisted = JSON.parse(JSON.stringify(first.conversation));
        await modelDef.requestTextCompletion(
            driver,
            {
                contents: [
                    {
                        role: 'user',
                        parts: [
                            { functionResponse: { name: 'first', response: { output: 'one' } } },
                            { functionResponse: { name: 'second', response: { output: 'two' } } },
                        ],
                    },
                ],
            },
            {
                model: 'publishers/google/models/gemini-3-flash',
                conversation: persisted,
                tools: [
                    { name: 'first', input_schema: { type: 'object' } },
                    { name: 'second', input_schema: { type: 'object' } },
                ],
            },
        );
        expect(requests[1]).toMatchObject({ contents: expect.arrayContaining([nativeContent]) });

        const hidden = await modelDef.requestTextCompletion(
            driver,
            { contents: [{ role: 'user', parts: [{ text: 'hidden' }] }] },
            {
                model: 'publishers/google/models/gemini-3-flash',
                model_options: { _option_id: 'vertexai-gemini', include_thoughts: false },
            },
        );
        expect(hidden.result).toEqual([{ type: 'text', value: 'answer' }]);
        expect(JSON.stringify(hidden.conversation)).toContain('terminal-signature');
    });

    it('reconstructs streamed thought, parallel function-call, and terminal signatures in order', async () => {
        const modelDef = new GeminiModelDefinition('gemini-3-flash');
        const driver = makeDriver({
            generateContentStream: async () =>
                (async function* () {
                    yield {
                        candidates: [
                            {
                                content: {
                                    role: 'model',
                                    parts: [
                                        { text: 'plan ', thought: true },
                                        {
                                            functionCall: { name: 'first', args: { value: 1 } },
                                            thoughtSignature: 'first-signature',
                                        },
                                    ],
                                },
                            },
                        ],
                    };
                    yield {
                        usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 2, totalTokenCount: 3 },
                        candidates: [
                            {
                                finishReason: FinishReason.STOP,
                                content: {
                                    role: 'model',
                                    parts: [
                                        { text: 'continued', thought: true, thoughtSignature: 'thought-signature' },
                                        {
                                            functionCall: { name: 'second', args: { value: 2 } },
                                            thoughtSignature: 'second-signature',
                                        },
                                        { text: 'answer', thoughtSignature: 'answer-signature' },
                                        { text: '', thoughtSignature: 'terminal-signature' },
                                    ],
                                },
                                safetyRatings: [],
                            },
                        ],
                    };
                })(),
        });
        const stream = await modelDef.requestTextCompletionStream(
            driver,
            { contents: [{ role: 'user', parts: [{ text: 'question' }] }] },
            { model: 'publishers/google/models/gemini-3-flash' },
        );
        const results = [];
        for await (const chunk of stream) results.push(...chunk.result);
        const conversation = await stream.finalizeConversation?.({ result: results });

        expect(results).toContainEqual({ type: 'thoughts', value: 'plan ' });
        expect(results).toContainEqual({ type: 'thoughts', value: 'continued' });
        expect(conversation).toMatchObject({
            _arrayConversation: expect.arrayContaining([
                {
                    role: 'model',
                    parts: [
                        { text: 'plan ', thought: true },
                        {
                            functionCall: { name: 'first', args: { value: 1 } },
                            thoughtSignature: 'first-signature',
                        },
                        { text: 'continued', thought: true, thoughtSignature: 'thought-signature' },
                        {
                            functionCall: { name: 'second', args: { value: 2 } },
                            thoughtSignature: 'second-signature',
                        },
                        { text: 'answer', thoughtSignature: 'answer-signature' },
                        { text: '', thoughtSignature: 'terminal-signature' },
                    ],
                },
            ]),
        });
    });

    it('does not merge a signed streamed Part into an unsigned Part', async () => {
        const modelDef = new GeminiModelDefinition('gemini-3-flash');
        const driver = makeDriver({
            generateContentStream: async () =>
                (async function* () {
                    yield {
                        candidates: [{ content: { role: 'model', parts: [{ text: 'first ', thought: true }] } }],
                    };
                    yield {
                        candidates: [
                            {
                                finishReason: FinishReason.STOP,
                                content: {
                                    role: 'model',
                                    parts: [{ text: 'second', thought: true, thoughtSignature: 'signed-second' }],
                                },
                            },
                        ],
                    };
                })(),
        });

        const stream = await modelDef.requestTextCompletionStream(
            driver,
            { contents: [{ role: 'user', parts: [{ text: 'question' }] }] },
            { model: 'publishers/google/models/gemini-3-flash' },
        );
        for await (const _chunk of stream) {
            // Consume the stream so native finalization has all Parts.
        }
        const conversation = await stream.finalizeConversation?.({ result: [] });

        expect(conversation).toMatchObject({
            _arrayConversation: expect.arrayContaining([
                {
                    role: 'model',
                    parts: [
                        { text: 'first ', thought: true },
                        { text: 'second', thought: true, thoughtSignature: 'signed-second' },
                    ],
                },
            ]),
        });
    });

    it('createPrompt uses DataSource URIs directly for Gemini file data', async () => {
        const modelDef = new GeminiModelDefinition('gemini-2.0-flash');
        const file: DataSource = {
            name: 'doc.pdf',
            mime_type: 'application/pdf',
            getURI: vi.fn().mockResolvedValue('gs://test-bucket/doc.pdf'),
            getURL: vi.fn().mockResolvedValue('https://signed.example/doc.pdf'),
            getStream: vi.fn().mockResolvedValue(new ReadableStream()),
        };
        const segments: PromptSegment[] = [
            {
                role: PromptRole.user,
                files: [file],
            } as PromptSegment,
        ];
        const options: ExecutionOptions = { model: 'publishers/google/models/gemini-2.0-flash' };

        const prompt = await modelDef.createPrompt({} as VertexAIDriver, segments, options);

        expect(file.getURI).toHaveBeenCalledTimes(1);
        expect(file.getURL).not.toHaveBeenCalled();
        expect(file.getStream).not.toHaveBeenCalled();
        expect(prompt.contents[0].parts?.[0]).toEqual({
            fileData: {
                fileUri: 'gs://test-bucket/doc.pdf',
                mimeType: 'application/pdf',
            },
        });
    });

    it('requestTextCompletion: does not mutate prompt.contents when tools=[] and conversation has function parts', async () => {
        const modelDef = new GeminiModelDefinition('gemini-2.0-flash');
        const originalContents = makeContentsWithFunctionParts();
        const contentsSnapshot = JSON.stringify(originalContents);

        const driver = makeDriver({ generateContent: async () => mockNonStreamingResponse });
        const prompt = { contents: originalContents, system: undefined } as unknown as GenerateContentPrompt;
        const options: ExecutionOptions = { model: 'publishers/google/models/gemini-2.0-flash', tools: [] };

        await modelDef.requestTextCompletion(driver, prompt, options);

        expect(JSON.stringify(originalContents)).toBe(contentsSnapshot);
        expect(originalContents[0].parts[0]).toHaveProperty('functionCall');
        expect(originalContents[1].parts[0]).toHaveProperty('functionResponse');
    });

    it('requestTextCompletionStream: does not mutate prompt.contents when tools=[] and conversation has function parts', async () => {
        const modelDef = new GeminiModelDefinition('gemini-2.0-flash');
        const originalContents = makeContentsWithFunctionParts();
        const contentsSnapshot = JSON.stringify(originalContents);

        const driver = makeDriver({
            generateContentStream: async () =>
                (async function* () {
                    yield mockStreamingChunk;
                })(),
        });
        const prompt = { contents: originalContents, system: undefined } as unknown as GenerateContentPrompt;
        const options: ExecutionOptions = { model: 'publishers/google/models/gemini-2.0-flash', tools: [] };

        const stream = await modelDef.requestTextCompletionStream(driver, prompt, options);
        // Drain the stream to trigger all processing
        for await (const _chunk of stream) {
            /* noop */
        }

        expect(JSON.stringify(originalContents)).toBe(contentsSnapshot);
        expect(originalContents[0].parts[0]).toHaveProperty('functionCall');
        expect(originalContents[1].parts[0]).toHaveProperty('functionResponse');
    });
});
