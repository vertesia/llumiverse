/**
 * Gemini may omit function-call ids. When the model calls the same tool several times in one turn
 * (four `create_document` calls), canonical ingestion assigns a distinct deterministic id to each
 * call. This prevents the core completion stream from folding the batch into a single call while
 * the stored model turn keeps all four
 * functionCall parts, only one tool ran, and the next request carried one functionResponse for
 * four calls — rejected by Vertex with 400 "Please ensure that the number of function response
 * parts is equal to the number of function call parts of the function call turn."
 *
 * The stream and follow-up tool results use those distinct ids while the provider projection keeps
 * the shared function name.
 */

import { FinishReason } from '@google/genai';
import { PromptRole, type PromptSegment } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import type { VertexAIDriver } from '../index.js';
import { GeminiModelDefinition } from './gemini.js';

const MODEL = 'publishers/google/models/gemini-2.5-flash';

function makeDriver(generateContentStream: (request?: unknown) => Promise<AsyncIterable<unknown>>) {
    return {
        logger: { warn: () => {}, info: () => {}, error: () => {}, debug: () => {} },
        getGoogleGenAIClient: () => ({
            models: {
                generateContent: async () => ({}),
                generateContentStream,
            },
        }),
    } as unknown as VertexAIDriver;
}

const parallelCreateDocumentChunk = {
    usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 },
    candidates: [
        {
            finishReason: FinishReason.STOP,
            content: {
                role: 'model',
                parts: [
                    {
                        functionCall: { name: 'create_document', args: { name: 'Acme' } },
                        thoughtSignature: 'first-signature',
                    },
                    { functionCall: { name: 'create_document', args: { name: 'Globex' } } },
                    { functionCall: { name: 'create_document', args: { name: 'Soylent' } } },
                    { functionCall: { name: 'create_document', args: { name: 'Omni' } } },
                ],
            },
            safetyRatings: [],
        },
    ],
};

async function streamParallelCalls(requests: unknown[], conversation?: unknown) {
    const modelDef = new GeminiModelDefinition('gemini-2.5-flash');
    const driver = makeDriver(async (request) => {
        requests.push(request);
        return (async function* () {
            yield parallelCreateDocumentChunk;
        })();
    });
    const stream = await modelDef.requestTextCompletionStream(
        driver,
        { contents: [{ role: 'user', parts: [{ text: 'create four documents' }] }] },
        { model: MODEL, conversation },
    );
    const toolUse = [];
    for await (const chunk of stream) toolUse.push(...(chunk.tool_use ?? []));
    return { modelDef, driver, toolUse, conversation: await stream.finalizeConversation?.() };
}

describe('Gemini streaming: parallel calls to the same tool', () => {
    it('gives every streamed call its own stable tool-use id', async () => {
        const { toolUse } = await streamParallelCalls([]);

        expect(toolUse).toHaveLength(4);
        // Distinct keys: the core accumulator (keyed by `id`) must not merge them.
        expect(new Set(toolUse.map((tool) => tool.id)).size).toBe(4);
        expect(toolUse.map((tool) => tool.tool_input)).toEqual([
            { name: 'Acme' },
            { name: 'Globex' },
            { name: 'Soylent' },
            { name: 'Omni' },
        ]);
        expect(toolUse[0].thought_signature).toBe('first-signature');
        expect(toolUse[1].thought_signature).toBeUndefined();
    });

    it('sends one function response part per function call part on the next request', async () => {
        const requests: unknown[] = [];
        const { modelDef, driver, toolUse, conversation } = await streamParallelCalls(requests);

        // One tool result per call, carrying the distinct id returned by the prior completion.
        const results: PromptSegment[] = ['Acme', 'Globex', 'Soylent', 'Omni'].map((name, index) => ({
            role: PromptRole.tool,
            tool_use_id: toolUse[index].id,
            content: `{"error":"The 'source' parameter must be a reference (${name})"}`,
            thought_signature: index === 0 ? 'first-signature' : undefined,
        }));
        // Tools stay declared on the follow-up turn, as in an agent run; without them the driver
        // converts function parts to text instead of pairing responses with calls.
        const options = {
            model: MODEL,
            conversation,
            tools: [
                {
                    name: 'create_document',
                    description: 'Create a document',
                    input_schema: { type: 'object', properties: { name: { type: 'string' } } },
                },
            ],
        };
        const prompt = await modelDef.createPrompt(driver, results, options);
        const stream = await modelDef.requestTextCompletionStream(driver, prompt, options);
        for await (const _chunk of stream) {
            // Drain so the request is issued.
        }

        const request = requests.at(-1) as { contents: Array<{ role: string; parts: unknown[] }> };
        const modelTurn = request.contents.at(-2);
        const responseTurn = request.contents.at(-1);
        expect(modelTurn?.role).toBe('model');
        expect(modelTurn?.parts.filter((part) => (part as { functionCall?: unknown }).functionCall)).toHaveLength(4);
        expect(responseTurn?.role).toBe('user');
        const responseParts = (responseTurn?.parts ?? []).filter(
            (part) => (part as { functionResponse?: unknown }).functionResponse,
        ) as Array<{ thoughtSignature?: string }>;
        // Gemini requires the responses in ONE user turn, with as many parts as the call turn.
        expect(responseParts).toHaveLength(4);
        expect(responseParts).toMatchObject(
            ['Acme', 'Globex', 'Soylent', 'Omni'].map((name) => ({
                functionResponse: {
                    name: 'create_document',
                    response: { error: `The 'source' parameter must be a reference (${name})` },
                },
            })),
        );
        // The thought signature travels with the first call's response only, as it was emitted.
        expect(responseParts[0].thoughtSignature).toBe('first-signature');
        expect(responseParts[1].thoughtSignature).toBeUndefined();
    });
});
