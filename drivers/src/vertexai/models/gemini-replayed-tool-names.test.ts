import type { Content } from '@google/genai';
import type { ExecutionOptions, ToolDefinition } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { getGeminiPayload } from './gemini.js';

const tools: ToolDefinition[] = [
    { name: 'web_search', input_schema: { type: 'object' } },
    { name: 'mcp.github.search', input_schema: { type: 'object' } },
];
const EMITTED = ['', 'functions.x', 'mcp.github.search', 'web_search'];
const SENT = ['_', 'functions_x', 'mcp.github.search', 'web_search'];

const history = (): Content[] => [
    { role: 'user', parts: [{ text: 'go' }] },
    {
        role: 'model',
        parts: EMITTED.map((name, i) => ({ functionCall: { id: `c${i}`, name, args: {} } })),
    },
    {
        role: 'user',
        parts: EMITTED.map((name, i) => ({
            functionResponse: { id: `c${i}`, name, response: { error: 'Tool not found' } },
        })),
    },
];

describe('Gemini replayed tool names', () => {
    it('rewrites a call and its response to the same safe name, without changing the conversation', () => {
        const contents = history();
        const options: ExecutionOptions = { model: 'gemini-2.5-flash', tools };

        const payload = getGeminiPayload(options, { contents });

        const sent = payload.contents as Content[];
        expect(sent[1].parts?.map((part) => part.functionCall?.name)).toEqual(SENT);
        expect(sent[2].parts?.map((part) => part.functionResponse?.name)).toEqual(SENT);
        expect(contents).toEqual(history());
    });

    it('leaves a response that pairs by id without a name', () => {
        const payload = getGeminiPayload(
            { model: 'gemini-2.5-flash', tools },
            {
                contents: [
                    { role: 'model', parts: [{ functionCall: { id: 'c0', name: 'web_search', args: {} } }] },
                    { role: 'user', parts: [{ functionResponse: { id: 'c0', response: {} } }] },
                ],
            },
        );

        const response = (payload.contents as Content[])[1].parts?.[0].functionResponse;
        expect(response).toBeDefined();
        expect(response?.name).toBeUndefined();
    });
});
