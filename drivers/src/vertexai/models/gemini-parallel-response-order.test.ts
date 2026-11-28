/**
 * The request sent after a turn of parallel function calls must follow Vertex's documented shape:
 * every functionResponse in ONE user content, in the order of the turn's functionCall parts, and the
 * thought signature only on the functionCall part Gemini emitted it on (the first call of the turn).
 *
 * Tool results reach the driver in whatever order the caller finished them, one segment each, and
 * conversations stored by earlier releases copied the first call's signature onto its response.
 * https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling
 * https://docs.cloud.google.com/vertex-ai/generative-ai/docs/thought-signatures
 */

import type { Content } from '@google/genai';
import { PromptRole, type PromptSegment } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import type { VertexAIDriver } from '../index.js';
import { alignFunctionResponsesWithCalls, GeminiModelDefinition } from './gemini.js';

const MODEL = 'publishers/google/models/gemini-3.1-flash-lite';
const TOOLS = ['search', 'read', 'list', 'count'].map((name) => ({
    name,
    description: `${name} tool`,
    input_schema: { type: 'object' as const, properties: { q: { type: 'string' } } },
}));

type SentPart = {
    functionCall?: { id?: string; name?: string };
    functionResponse?: { id?: string; name?: string; response?: Record<string, unknown> };
    thoughtSignature?: string;
};
type SentRequest = { contents: Array<{ role: string; parts: SentPart[] }> };

function makeDriver(requests: unknown[]) {
    return {
        logger: { warn: vi.fn(), info: vi.fn(), error: vi.fn(), debug: vi.fn() },
        getGoogleGenAIClient: () => ({
            models: {
                generateContent: async () => ({}),
                generateContentStream: async (request: unknown) => {
                    requests.push(request);
                    return (async function* () {})();
                },
            },
        }),
    } as unknown as VertexAIDriver;
}

/** A model turn with four parallel calls, the signature on the first only, as Gemini returns it. */
function callTurn(ids: (string | undefined)[]): Content {
    return {
        role: 'model',
        parts: TOOLS.map((tool, index) => ({
            functionCall: { ...(ids[index] ? { id: ids[index] } : {}), name: tool.name, args: { q: tool.name } },
            ...(index === 0 ? { thoughtSignature: 'sig-1' } : {}),
        })),
    };
}

async function send(segments: PromptSegment[], conversation: unknown) {
    const requests: unknown[] = [];
    const driver = makeDriver(requests);
    const modelDef = new GeminiModelDefinition('gemini-3.1-flash-lite');
    const options = { model: MODEL, conversation, tools: TOOLS };
    const prompt = await modelDef.createPrompt(driver, segments, options);
    const stream = await modelDef.requestTextCompletionStream(driver, prompt, options);
    for await (const _chunk of stream) {
        // Drain so the request is issued.
    }
    return { request: requests.at(-1) as SentRequest, driver };
}

/** Results for the calls at `order`, one tool segment each, as an agent loop returns them. */
function results(ids: string[], order: number[], signatureOn?: number): PromptSegment[] {
    return order.map((index) => ({
        role: PromptRole.tool,
        tool_use_id: ids[index],
        content: `{"tool":"${TOOLS[index].name}"}`,
        thought_signature: index === signatureOn ? 'sig-1' : undefined,
    }));
}

function expectDocumentedShape(request: SentRequest) {
    const callTurnSent = request.contents.at(-2);
    const responseTurn = request.contents.at(-1);
    expect(callTurnSent?.role).toBe('model');
    expect(responseTurn?.role).toBe('user');
    // One user content holding every response, in call order.
    expect(responseTurn?.parts.map((part) => part.functionResponse?.name)).toEqual(TOOLS.map((tool) => tool.name));
    expect(responseTurn?.parts.map((part) => part.functionResponse?.response)).toEqual(
        TOOLS.map((tool) => ({ tool: tool.name })),
    );
    // The signature stays on the first call and nowhere else.
    expect(callTurnSent?.parts.map((part) => part.thoughtSignature)).toEqual([
        'sig-1',
        undefined,
        undefined,
        undefined,
    ]);
    for (const part of responseTurn?.parts ?? []) expect(part).not.toHaveProperty('thoughtSignature');
}

describe('Gemini: responses to parallel calls', () => {
    const vertexIds = ['call_a', 'call_b', 'call_c', 'call_d'];

    it('sends results that finished out of call order in call order (Vertex ids)', async () => {
        const conversation = [{ role: 'user', parts: [{ text: 'go' }] }, callTurn(vertexIds)];
        const { request } = await send(results(vertexIds, [2, 0, 3, 1], 0), conversation);

        expectDocumentedShape(request);
        expect(request.contents.at(-1)?.parts.map((part) => part.functionResponse?.id)).toEqual(vertexIds);
    });

    it('sends results in call order when Vertex issued no ids (generated ids are stripped)', async () => {
        const generated = vertexIds.map((id) => `llumiverse_call_${id}`);
        const conversation = [{ role: 'user', parts: [{ text: 'go' }] }, callTurn(generated)];
        const { request } = await send(results(generated, [3, 1, 0, 2], 0), conversation);

        expectDocumentedShape(request);
        for (const part of request.contents.at(-1)?.parts ?? []) expect(part.functionResponse).not.toHaveProperty('id');
    });

    it('sends a stored conversation of the old shape in the documented shape', async () => {
        // Stored by an earlier release: one user content per result, out of call order, and the
        // first call's signature copied onto its response.
        const stored = {
            _arrayConversation: [
                { role: 'user', parts: [{ text: 'go' }] },
                callTurn(vertexIds),
                ...[1, 0, 3, 2].map((index) => ({
                    role: 'user',
                    parts: [
                        {
                            functionResponse: {
                                id: vertexIds[index],
                                name: TOOLS[index].name,
                                response: { tool: TOOLS[index].name },
                            },
                            ...(index === 0 ? { thoughtSignature: 'sig-1' } : {}),
                        },
                    ],
                })),
            ],
        };
        const { request } = await send([{ role: PromptRole.user, content: 'continue' }], stored);

        const responseTurn = request.contents.at(-2);
        const callTurnSent = request.contents.at(-3);
        expect(responseTurn?.parts.map((part) => part.functionResponse?.id)).toEqual(vertexIds);
        for (const part of responseTurn?.parts ?? []) expect(part).not.toHaveProperty('thoughtSignature');
        expect(callTurnSent?.parts[0].thoughtSignature).toBe('sig-1');
        expect(request.contents.at(-1)?.parts).toEqual([{ text: 'continue' }]);
        // The stored conversation is left as it was.
        expect(stored._arrayConversation[3].parts?.[0]).toHaveProperty('thoughtSignature', 'sig-1');
    });

    it('does not store the signature on the result it sends', async () => {
        const conversation = [{ role: 'user', parts: [{ text: 'go' }] }, callTurn(vertexIds)];
        const requests: unknown[] = [];
        const modelDef = new GeminiModelDefinition('gemini-3.1-flash-lite');
        const prompt = await modelDef.createPrompt(makeDriver(requests), results(vertexIds, [0, 1, 2, 3], 0), {
            model: MODEL,
            conversation,
            tools: TOOLS,
        });

        for (const content of prompt.contents) {
            for (const part of content.parts ?? []) expect(part).not.toHaveProperty('thoughtSignature');
        }
    });
});

describe('alignFunctionResponsesWithCalls', () => {
    it('pairs responses without ids by name in sequence, including repeated calls to one tool', () => {
        const contents: Content[] = [
            {
                role: 'model',
                parts: [
                    { functionCall: { name: 'read', args: { q: 1 } }, thoughtSignature: 'sig' },
                    { functionCall: { name: 'search', args: {} } },
                    { functionCall: { name: 'read', args: { q: 2 } } },
                ],
            },
            {
                role: 'user',
                parts: [
                    { functionResponse: { name: 'search', response: { r: 's' } } },
                    { functionResponse: { name: 'read', response: { r: 1 } } },
                    { functionResponse: { name: 'read', response: { r: 2 } } },
                ],
            },
        ];
        const aligned = alignFunctionResponsesWithCalls(contents);

        expect(aligned[1].parts?.map((part) => part.functionResponse?.response)).toEqual([
            { r: 1 },
            { r: 's' },
            { r: 2 },
        ]);
        expect(aligned[0]).toBe(contents[0]);
    });

    it('puts a response signature back on its call only when the turn carries none', () => {
        const contents: Content[] = [
            {
                role: 'model',
                parts: [{ functionCall: { id: 'a', name: 'read' } }, { functionCall: { id: 'b', name: 'read' } }],
            },
            {
                role: 'user',
                parts: [
                    { functionResponse: { id: 'b', name: 'read', response: {} } },
                    { functionResponse: { id: 'a', name: 'read', response: {} }, thoughtSignature: 'sig-a' },
                ],
            },
        ];
        const logger = { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() };
        const aligned = alignFunctionResponsesWithCalls(contents, logger as never, 'gemini-x');

        expect(aligned[0].parts?.map((part) => part.thoughtSignature)).toEqual(['sig-a', undefined]);
        expect(aligned[1].parts?.map((part) => part.functionResponse?.id)).toEqual(['a', 'b']);
        for (const part of aligned[1].parts ?? []) expect(part).not.toHaveProperty('thoughtSignature');
        expect(logger.info).toHaveBeenCalledWith(
            expect.objectContaining({ reordered_response_turns: 1, signatures_moved_to_calls: 1 }),
            expect.any(String),
        );
        // The caller's contents are not modified.
        expect(contents[0].parts?.[0]).not.toHaveProperty('thoughtSignature');
        expect(contents[1].parts?.[1]).toHaveProperty('thoughtSignature', 'sig-a');
    });

    it('leaves responses already in call order, and unmatched responses, where they are', () => {
        const contents: Content[] = [
            { role: 'model', parts: [{ functionCall: { id: 'a', name: 'read' } }] },
            {
                role: 'user',
                parts: [
                    { functionResponse: { id: 'a', name: 'read', response: {} } },
                    { functionResponse: { name: 'other', response: {} } },
                ],
            },
        ];
        const aligned = alignFunctionResponsesWithCalls(contents);

        expect(aligned[1]).toBe(contents[1]);
    });
});
