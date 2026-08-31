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
import { fixOrphanedToolResults, getGeminiPayload, mergeFunctionResponseContents } from './gemini.js';

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
    test('maps the private required-tool hint to Gemini ANY mode and an allowed function', () => {
        const payload = getGeminiPayload(
            {
                ...OPTIONS_WITH_TOOLS,
                model_options: {
                    _option_id: 'vertexai-gemini',
                    tool_choice: 'required',
                    required_tool_name: 'a',
                },
            } as unknown as ExecutionOptions,
            { contents: [{ role: 'user', parts: [{ text: 'Act now.' }] }] },
        );

        expect(payload.config?.toolConfig?.functionCallingConfig).toMatchObject({
            mode: 'ANY',
            allowedFunctionNames: ['a'],
        });
    });

    test('maps the private no-tool hint to Gemini NONE mode', () => {
        const payload = getGeminiPayload(
            {
                ...OPTIONS_WITH_TOOLS,
                model_options: { _option_id: 'vertexai-gemini', tool_choice: 'none' },
            } as unknown as ExecutionOptions,
            { contents: [{ role: 'user', parts: [{ text: 'Return the receipt.' }] }] },
        );

        expect(payload.config?.toolConfig?.functionCallingConfig?.mode).toBe('NONE');
    });

    test('rejects a required tool choice when no tools are available', () => {
        let thrown: unknown;
        try {
            getGeminiPayload(
                {
                    model: 'gemini-2.5-flash',
                    tools: [],
                    model_options: { _option_id: 'vertexai-gemini', tool_choice: 'required' },
                } as unknown as ExecutionOptions,
                { contents: [{ role: 'user', parts: [{ text: 'Act now.' }] }] },
                'stream',
            );
        } catch (error: unknown) {
            thrown = error;
        }

        expect(thrown).toMatchObject({
            name: 'ToolChoiceConfigurationError',
            retryable: false,
            code: 400,
            context: {
                provider: 'vertexai',
                model: 'gemini-2.5-flash',
                operation: 'stream',
            },
            message: expect.stringContaining('required tool choice was requested, but no tools are available'),
        });
    });

    test('recombines split parallel functionResponses into one user turn matching the call turn', () => {
        // Gemini rejects a function-call turn whose responses are split across consecutive user
        // contents: "Please ensure that the number of function response parts is equal to the
        // number of function call parts of the function call turn." (400 INVALID_ARGUMENT)
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
        expect(payload.contents).toHaveLength(2);
        expect((payload.contents as Content[])[1].parts).toHaveLength(2);
    });
});

describe('mergeFunctionResponseContents', () => {
    test('merges a run of function-response-only user contents into one turn', () => {
        const contents: Content[] = [
            { role: 'user', parts: [{ text: 'checkpoint summary' }] },
            {
                role: 'model',
                parts: [
                    { functionCall: { name: 'think', args: {} }, thoughtSignature: 'sig-1' },
                    { functionCall: { name: 'wait_for', args: {} } },
                ],
            },
            {
                role: 'user',
                parts: [{ functionResponse: { name: 'think', response: { ok: true } }, thoughtSignature: 'sig-1' }],
            },
            { role: 'user', parts: [{ functionResponse: { name: 'wait_for', response: { ok: true } } }] },
        ];

        const result = mergeFunctionResponseContents(contents);

        expect(result).toHaveLength(3);
        expect(result[2].role).toBe('user');
        expect(result[2].parts).toHaveLength(2);
        // Part-level metadata such as thoughtSignature survives the merge.
        expect(result[2].parts?.[0].thoughtSignature).toBe('sig-1');
        // The input contents are not mutated.
        expect(contents).toHaveLength(4);
        expect(contents[2].parts).toHaveLength(1);
    });

    test('leaves consecutive user text segments separate (cache breakpoints)', () => {
        const contents: Content[] = [
            { role: 'user', parts: [{ text: 'catalog' }] },
            { role: 'user', parts: [{ text: 'task' }] },
        ];
        expect(mergeFunctionResponseContents(contents)).toEqual(contents);
    });

    test('does not merge a mixed text + functionResponse content into a response run', () => {
        const contents: Content[] = [
            { role: 'user', parts: [{ functionResponse: { name: 'a', response: {} } }] },
            { role: 'user', parts: [{ functionResponse: { name: 'b', response: {} } }, { text: 'and also' }] },
        ];
        expect(mergeFunctionResponseContents(contents)).toEqual(contents);
    });

    test('merges independent response runs separately', () => {
        const contents: Content[] = [
            {
                role: 'model',
                parts: [{ functionCall: { name: 'a', args: {} } }, { functionCall: { name: 'b', args: {} } }],
            },
            { role: 'user', parts: [{ functionResponse: { name: 'a', response: {} } }] },
            { role: 'user', parts: [{ functionResponse: { name: 'b', response: {} } }] },
            {
                role: 'model',
                parts: [{ functionCall: { name: 'c', args: {} } }, { functionCall: { name: 'd', args: {} } }],
            },
            { role: 'user', parts: [{ functionResponse: { name: 'c', response: {} } }] },
            { role: 'user', parts: [{ functionResponse: { name: 'd', response: {} } }] },
        ];

        const result = mergeFunctionResponseContents(contents);

        expect(result).toHaveLength(4);
        expect(result[1].parts).toHaveLength(2);
        expect(result[3].parts).toHaveLength(2);
    });
});
