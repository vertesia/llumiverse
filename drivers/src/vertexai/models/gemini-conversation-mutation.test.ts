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

import { ExecutionOptions } from '@llumiverse/core';
import { FinishReason } from '@google/genai';
import { describe, expect, it } from 'vitest';
import { VertexAIDriver } from '../index.js';
import { convertGeminiFunctionPartsToText, GeminiModelDefinition } from './gemini.js';

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
        const originalItemRefs = original.map(c => c);
        const originalPartRefs = original.map(c => c.parts[0]);

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

        expect(result[0].parts![0]).toEqual({
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

        expect(result[0].parts![0]).toEqual({
            text: '[Tool result for get_weather: {"temperature":"15°C"}]',
        });
    });

    it('leaves non-function parts intact', () => {
        const textPart = { text: 'Hello world' };
        const contents = [{ role: 'user', parts: [textPart] }];

        const result = convertGeminiFunctionPartsToText(contents);

        expect(result[0].parts![0]).toBe(textPart);
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

function makeDriver(overrides: { generateContent?: () => Promise<any>; generateContentStream?: () => Promise<AsyncIterable<any>> }) {
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
    candidates: [{
        finishReason: FinishReason.STOP,
        content: { role: 'model', parts: [{ text: 'Summary.' }] },
        safetyRatings: [],
    }],
};

const mockStreamingChunk = {
    usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 },
    candidates: [{
        finishReason: FinishReason.STOP,
        content: { role: 'model', parts: [{ text: 'Summary.' }] },
        safetyRatings: [],
    }],
};

describe('GeminiModelDefinition - no conversation mutation', () => {
    it('requestTextCompletion: does not mutate prompt.contents when tools=[] and conversation has function parts', async () => {
        const modelDef = new GeminiModelDefinition('gemini-2.0-flash');
        const originalContents = makeContentsWithFunctionParts();
        const contentsSnapshot = JSON.stringify(originalContents);

        const driver = makeDriver({ generateContent: async () => mockNonStreamingResponse });
        const prompt = { contents: originalContents, system: undefined } as any;
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
            generateContentStream: async () => (async function* () { yield mockStreamingChunk; })(),
        });
        const prompt = { contents: originalContents, system: undefined } as any;
        const options: ExecutionOptions = { model: 'publishers/google/models/gemini-2.0-flash', tools: [] };

        const stream = await modelDef.requestTextCompletionStream(driver, prompt, options);
        // Drain the stream to trigger all processing
        for await (const _chunk of stream) { /* noop */ }

        expect(JSON.stringify(originalContents)).toBe(contentsSnapshot);
        expect(originalContents[0].parts[0]).toHaveProperty('functionCall');
        expect(originalContents[1].parts[0]).toHaveProperty('functionResponse');
    });
});
