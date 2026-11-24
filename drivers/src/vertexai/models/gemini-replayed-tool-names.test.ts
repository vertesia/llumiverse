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
        const options: ExecutionOptions = { model: MODEL, tools };

        const payload = getGeminiPayload(options, { contents });

        const sent = payload.contents as Content[];
        expect(sent[1].parts?.map((part) => part.functionCall?.name)).toEqual(SENT);
        expect(sent[2].parts?.map((part) => part.functionResponse?.name)).toEqual(SENT);
        expect(contents).toEqual(history());
    });

    it.each([
        ['a provider-issued', 'vertex-call-1', true],
        ['a generated', `${GEMINI_GENERATED_CALL_ID_PREFIX}1`, true],
        ['a provider-issued', 'vertex-call-1', false],
    ])(
        'keeps the tool result of an empty-name call with %s id paired with it (wrapped=%s)',
        async (_kind, id, wrapped) => {
            const stored: Content[] = [
                { role: 'user', parts: [{ text: 'go' }] },
                { role: 'model', parts: [{ functionCall: { id, name: '', args: {} } }] },
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
            const call = sent[1].parts?.[0].functionCall;
            const response = sent[2]?.parts?.[0].functionResponse;
            expect(call?.name).toBe('_');
            expect(response?.name).toBe('_');
            // Vertex only receives the call ids it issued.
            expect(call?.id).toBe(id.startsWith(GEMINI_GENERATED_CALL_ID_PREFIX) ? undefined : id);
            expect(response?.id).toBe(call?.id);
        },
    );

    it("gives a response that carries only a call id its call's safe name", () => {
        const payload = getGeminiPayload(
            { model: MODEL, tools },
            {
                contents: [
                    {
                        role: 'model',
                        parts: [
                            { functionCall: { id: 'c0', name: 'functions.x', args: {} } },
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
        expect(responses).toEqual(['functions_x', 'web_search']);
    });
});
