/**
 * Unit tests for fixOrphanedToolResults (Gemini).
 *
 * Gemini pairs a `functionResponse` part to its `functionCall` part by name. A
 * functionResponse left dangling after its functionCall was dropped — e.g. by
 * conversation compaction trimming the model tool-call turn, or an unmergeable
 * parallel batch — causes the Gemini/Vertex API to reject the request. This is
 * the same class of bug fixed in the Claude, Bedrock, and OpenAI drivers.
 */

import type { Content } from '@google/genai';
import type { ExecutionOptions } from '@llumiverse/core';
import { describe, expect, test } from 'vitest';
import { fixOrphanedToolResults, getGeminiPayload } from './gemini.js';

const OPTIONS_WITH_TOOLS = {
    model: 'gemini-2.5-flash',
    tools: [
        { name: 'a', description: '', input_schema: { type: 'object', properties: {} } },
        { name: 'b', description: '', input_schema: { type: 'object', properties: {} } },
    ],
} as unknown as ExecutionOptions;

function functionResponseNames(contents: Content[]): string[] {
    return contents.flatMap((content) =>
        (content.parts ?? []).flatMap((part) => (part.functionResponse?.name ? [part.functionResponse.name] : [])),
    );
}

describe('fixOrphanedToolResults - Gemini', () => {
    test('returns empty array for empty input', () => {
        expect(fixOrphanedToolResults([])).toEqual([]);
    });

    test('keeps a functionResponse that has a matching functionCall in the previous model content', () => {
        const contents: Content[] = [
            { role: 'model', parts: [{ functionCall: { name: 'search', args: {} } }] },
            { role: 'user', parts: [{ functionResponse: { name: 'search', response: { ok: true } } }] },
        ];
        expect(fixOrphanedToolResults(contents)).toEqual(contents);
    });

    test('keeps all responses of a parallel batch when both calls are present', () => {
        const contents: Content[] = [
            {
                role: 'model',
                parts: [{ functionCall: { name: 'a', args: {} } }, { functionCall: { name: 'b', args: {} } }],
            },
            {
                role: 'user',
                parts: [
                    { functionResponse: { name: 'a', response: {} } },
                    { functionResponse: { name: 'b', response: {} } },
                ],
            },
        ];
        expect(fixOrphanedToolResults(contents)).toEqual(contents);
    });

    test('drops a functionResponse whose functionCall is absent from the previous content', () => {
        const contents: Content[] = [
            { role: 'model', parts: [{ functionCall: { name: 'a', args: {} } }] },
            {
                role: 'user',
                parts: [
                    { functionResponse: { name: 'a', response: {} } },
                    { functionResponse: { name: 'gone', response: {} } },
                ],
            },
        ];

        const result = fixOrphanedToolResults(contents);
        expect(result[1].parts).toHaveLength(1);
        expect(result[1].parts?.[0].functionResponse?.name).toBe('a');
    });

    test('drops a content that becomes empty after removing orphaned responses', () => {
        const contents: Content[] = [
            { role: 'user', parts: [{ text: '[summary of prior work]' }] },
            { role: 'user', parts: [{ functionResponse: { name: 'gone', response: {} } }] },
        ];

        const result = fixOrphanedToolResults(contents);
        expect(result).toHaveLength(1);
        expect(result[0].parts?.[0].text).toBe('[summary of prior work]');
    });

    test('preserves non-functionResponse parts while dropping the orphan', () => {
        const contents: Content[] = [
            { role: 'model', parts: [{ text: 'thinking' }] },
            {
                role: 'user',
                parts: [{ functionResponse: { name: 'gone', response: {} } }, { text: 'continue please' }],
            },
        ];

        const result = fixOrphanedToolResults(contents);
        expect(result[1].parts).toHaveLength(1);
        expect(result[1].parts?.[0].text).toBe('continue please');
    });
});

describe('getGeminiPayload - orphaned tool results', () => {
    test('retains split parallel functionResponses by merging consecutive user contents before cleanup', () => {
        const contents: Content[] = [
            {
                role: 'model',
                parts: [{ functionCall: { name: 'a', args: {} } }, { functionCall: { name: 'b', args: {} } }],
            },
            { role: 'user', parts: [{ functionResponse: { name: 'a', response: { ok: true } } }] },
            { role: 'user', parts: [{ functionResponse: { name: 'b', response: { ok: true } } }] },
        ];

        const payload = getGeminiPayload(OPTIONS_WITH_TOOLS, { contents });

        expect(functionResponseNames(payload.contents as Content[])).toEqual(['a', 'b']);
    });
});
