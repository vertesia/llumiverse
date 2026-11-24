import type { Content } from '@google/genai';
import {
    type ExecutionOptions,
    incrementConversationTurn,
    PromptRole,
    type PromptSegment,
    type ToolDefinition,
} from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import type { VertexAIDriver } from '../index.js';
import { GEMINI_GENERATED_CALL_ID_PREFIX, GeminiModelDefinition, getGeminiPayload } from './gemini.js';

const MODEL = 'gemini-2.5-flash';
const tools: ToolDefinition[] = [
    { name: 'web_search', input_schema: { type: 'object' } },
    { name: 'mcp.github.search', input_schema: { type: 'object' } },
];
// Gemini accepts dots and colons, so only the empty name, the `/` and the leading digit need a rewrite.
const EMITTED = ['', 'functions.x', 'a/b', '1tool', 'mcp.github.search', 'web_search'];
const SENT = ['_', 'functions.x', 'a_b', '_1tool', 'mcp.github.search', 'web_search'];

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
        const options: ExecutionOptions = { model: MODEL, tools };

        const payload = getGeminiPayload(options, { contents });

        const sent = payload.contents as Content[];
        expect(sent[1].parts?.map((part) => part.functionCall?.name)).toEqual(SENT);
        expect(sent[2].parts?.map((part) => part.functionResponse?.name)).toEqual(SENT);
        expect(contents).toEqual(history());
    });

    it.each([
        { kind: 'empty', name: '', id: 'vertex-call-1', wrapped: true },
        { kind: 'empty', name: '', id: `${GEMINI_GENERATED_CALL_ID_PREFIX}1`, wrapped: true },
        { kind: 'empty', name: '', id: 'vertex-call-1', wrapped: false },
        { kind: 'omitted', name: undefined, id: 'vertex-call-1', wrapped: true },
        { kind: 'omitted', name: undefined, id: `${GEMINI_GENERATED_CALL_ID_PREFIX}1`, wrapped: true },
    ])(
        'keeps the tool result of a call with an $kind name and id $id paired with it (wrapped=$wrapped)',
        async ({ name, id, wrapped }) => {
            const call = name === undefined ? { id, args: {} } : { id, name, args: {} };
            const stored: Content[] = [
                { role: 'user', parts: [{ text: 'go' }] },
                { role: 'model', parts: [{ functionCall: call }] },
            ];
            // Stored conversations are wrapped with their turn metadata; a plain array is accepted too.
            const conversation = wrapped ? incrementConversationTurn(stored) : stored;
            const options: ExecutionOptions = { model: MODEL, tools, conversation };
            const result: PromptSegment = {
                role: PromptRole.tool,
                tool_use_id: id,
                content: '{"error":"Tool not found"}',
            };

            // The driver sends the stored conversation followed by the new turn.
            const prompt = await new GeminiModelDefinition(MODEL).createPrompt(
                null as unknown as VertexAIDriver,
                [result],
                options,
            );
            const payload = getGeminiPayload(options, { contents: stored.concat(prompt.contents) });

            const sent = payload.contents as Content[];
            const sentCall = sent[1].parts?.[0].functionCall;
            const response = sent[2]?.parts?.[0].functionResponse;
            expect(sentCall?.name).toBe('_');
            expect(response?.name).toBe('_');
            // Vertex only receives the call ids it issued.
            expect(sentCall?.id).toBe(id.startsWith(GEMINI_GENERATED_CALL_ID_PREFIX) ? undefined : id);
            expect(response?.id).toBe(sentCall?.id);
        },
    );

    it('gives an id-only response to a call with no name the same safe name as the call', () => {
        const payload = getGeminiPayload(
            { model: MODEL, tools },
            {
                contents: [
                    { role: 'model', parts: [{ functionCall: { id: 'c0', args: {} } }] },
                    { role: 'user', parts: [{ functionResponse: { id: 'c0', response: {} } }] },
                ],
            },
        );

        const sent = payload.contents as Content[];
        expect(sent[0].parts?.[0].functionCall?.name).toBe('_');
        expect(sent[1]?.parts?.[0].functionResponse?.name).toBe('_');
    });

    it('gives a response with no id and no name the same safe name as its unnamed call', () => {
        const payload = getGeminiPayload(
            { model: MODEL, tools },
            {
                contents: [
                    { role: 'model', parts: [{ functionCall: { args: {} } }] },
                    { role: 'user', parts: [{ functionResponse: { response: {} } }] },
                ],
            },
        );

        const sent = payload.contents as Content[];
        expect(sent[0].parts?.[0].functionCall?.name).toBe('_');
        expect(sent[1]?.parts?.[0].functionResponse?.name).toBe('_');
    });

    it("gives a response that carries only a call id its call's safe name", () => {
        const payload = getGeminiPayload(
            { model: MODEL, tools },
            {
                contents: [
                    {
                        role: 'model',
                        parts: [
                            { functionCall: { id: 'c0', name: 'a/b', args: {} } },
                            { functionCall: { id: 'c1', name: 'web_search', args: {} } },
                        ],
                    },
                    {
                        role: 'user',
                        parts: [
                            { functionResponse: { id: 'c0', response: {} } },
                            { functionResponse: { id: 'c1', response: {} } },
                        ],
                    },
                ],
            },
        );

        const responses = (payload.contents as Content[])[1].parts?.map((part) => part.functionResponse?.name);
        expect(responses).toEqual(['a_b', 'web_search']);
    });

    it('sends the same rewritten history on the next turn, so the cached prefix still matches', () => {
        const options: ExecutionOptions = { model: MODEL, tools };
        const turn2: Content[] = [
            ...history(),
            { role: 'model', parts: [{ text: 'Those tools do not exist.' }] },
            { role: 'user', parts: [{ text: 'Use web_search instead.' }] },
        ];

        const first = getGeminiPayload(options, { contents: history() }).contents as Content[];
        const second = getGeminiPayload(options, { contents: turn2 }).contents as Content[];

        expect(second.slice(0, first.length)).toEqual(first);
    });
});
