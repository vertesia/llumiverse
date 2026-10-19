/**
 * When the model calls the same tool several times in one turn (four `create_document` calls),
 * every call must keep its own tool-use id. The driver used to return the function name as the id,
 * so a parallel batch shared one id: the core completion stream folded it into a single call, and
 * downstream every result, artifact and retry check keyed by that id collided.
 *
 * Each functionCall part now carries an id that is unique within the conversation: Vertex's own call
 * id when it sends one, a generated one otherwise. The id is stored on the part, returned as
 * `ToolUse.id`, and resolves a tool result back to its call. Generated ids never reach Vertex.
 */

import { FinishReason } from '@google/genai';
import { PromptRole, type PromptSegment, type ToolUse } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import type { VertexAIDriver } from '../index.js';
import { GEMINI_GENERATED_CALL_ID_PREFIX, GeminiModelDefinition } from './gemini.js';

const MODEL = 'publishers/google/models/gemini-2.5-flash';
const NAMES = ['Acme', 'Globex', 'Soylent', 'Omni'];
const TOOLS = [
    {
        name: 'create_document',
        description: 'Create a document',
        input_schema: { type: 'object' as const, properties: { name: { type: 'string' } } },
    },
];

type RequestPart = {
    functionCall?: { id?: string; name?: string };
    functionResponse?: Record<string, unknown>;
    thoughtSignature?: string;
};
type RequestContents = { contents: Array<{ role: string; parts: RequestPart[] }> };

function makeDriver(
    generateContentStream: (request?: unknown) => Promise<AsyncIterable<unknown>>,
    generateContent: (request?: unknown) => Promise<unknown> = async () => ({}),
) {
    return {
        logger: { warn: () => {}, info: () => {}, error: () => {}, debug: () => {} },
        getGoogleGenAIClient: () => ({ models: { generateContent, generateContentStream } }),
    } as unknown as VertexAIDriver;
}

/** One model chunk calling `create_document` four times; `ids` are Vertex's call ids, if any. */
function parallelCreateDocumentChunk(ids?: string[]) {
    return {
        usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 },
        candidates: [
            {
                finishReason: FinishReason.STOP,
                content: {
                    role: 'model',
                    parts: NAMES.map((name, index) => ({
                        functionCall: {
                            name: 'create_document',
                            args: { name },
                            ...(ids?.[index] ? { id: ids[index] } : {}),
                        },
                        ...(index === 0 ? { thoughtSignature: 'first-signature' } : {}),
                    })),
                },
                safetyRatings: [],
            },
        ],
    };
}

async function streamParallelCalls(requests: unknown[], options: { conversation?: unknown; ids?: string[] } = {}) {
    const modelDef = new GeminiModelDefinition('gemini-2.5-flash');
    const driver = makeDriver(async (request) => {
        requests.push(request);
        return (async function* () {
            yield parallelCreateDocumentChunk(options.ids);
        })();
    });
    const stream = await modelDef.requestTextCompletionStream(
        driver,
        { contents: [{ role: 'user', parts: [{ text: 'create four documents' }] }] },
        { model: MODEL, conversation: options.conversation, tools: TOOLS },
    );
    const toolUse: ToolUse<unknown>[] = [];
    for await (const chunk of stream) toolUse.push(...(chunk.tool_use ?? []));
    return { modelDef, driver, toolUse, conversation: await stream.finalizeConversation?.() };
}

/** The functionCall parts of the stored model turns. */
function storedCalls(conversation: unknown): Array<{ id?: string; name?: string }> {
    const turns = (conversation as { _arrayConversation: RequestContents['contents'] })._arrayConversation;
    return turns
        .filter((turn) => turn.role === 'model')
        .flatMap((turn) => turn.parts.flatMap((part) => (part.functionCall ? [part.functionCall] : [])));
}

/** Answer one call per id, then issue the next request and return it. */
async function sendResults(
    modelDef: GeminiModelDefinition,
    driver: VertexAIDriver,
    requests: unknown[],
    conversation: unknown,
    toolUseIds: string[],
): Promise<RequestContents> {
    const results: PromptSegment[] = toolUseIds.map((tool_use_id, index) => ({
        role: PromptRole.tool,
        tool_use_id,
        content: `{"created":"${NAMES[index]}"}`,
        thought_signature: index === 0 ? 'first-signature' : undefined,
    }));
    // Tools stay declared on the follow-up turn, as in an agent run; without them the driver
    // converts function parts to text instead of pairing responses with calls.
    const options = { model: MODEL, conversation, tools: TOOLS };
    const prompt = await modelDef.createPrompt(driver, results, options);
    const stream = await modelDef.requestTextCompletionStream(driver, prompt, options);
    for await (const _chunk of stream) {
        // Drain so the request is issued.
    }
    return requests.at(-1) as RequestContents;
}

function functionCalls(request: RequestContents, turn: number): RequestPart[] {
    return request.contents.at(turn)?.parts.filter((part) => part.functionCall) ?? [];
}

function functionResponses(request: RequestContents, turn: number): RequestPart[] {
    return request.contents.at(turn)?.parts.filter((part) => part.functionResponse) ?? [];
}

describe('Gemini: parallel calls to the same tool', () => {
    it('gives every call its own id and stores that id on the call', async () => {
        const { toolUse, conversation } = await streamParallelCalls([]);

        expect(toolUse).toHaveLength(4);
        expect(new Set(toolUse.map((tool) => tool.id)).size).toBe(4);
        for (const tool of toolUse) expect(tool.id.startsWith(GEMINI_GENERATED_CALL_ID_PREFIX)).toBe(true);
        expect(toolUse.map((tool) => tool.tool_name)).toEqual(NAMES.map(() => 'create_document'));
        expect(toolUse.map((tool) => tool.tool_input)).toEqual(NAMES.map((name) => ({ name })));
        expect(toolUse[0].thought_signature).toBe('first-signature');
        expect(toolUse[1].thought_signature).toBeUndefined();
        // The stored call carries the same id, so the result can be paired with it later.
        expect(storedCalls(conversation).map((call) => call.id)).toEqual(toolUse.map((tool) => tool.id));
    });

    it("keeps Vertex's own call ids", async () => {
        const ids = ['call_1', 'call_2', 'call_3', 'call_4'];
        const { toolUse, conversation } = await streamParallelCalls([], { ids });

        expect(toolUse.map((tool) => tool.id)).toEqual(ids);
        expect(storedCalls(conversation).map((call) => call.id)).toEqual(ids);
    });

    it('replaces a Vertex call id that repeats one in the same turn', async () => {
        const { toolUse } = await streamParallelCalls([], { ids: ['call_1', 'call_1', 'call_2', 'call_2'] });

        expect(new Set(toolUse.map((tool) => tool.id)).size).toBe(4);
        expect(toolUse[0].id).toBe('call_1');
        expect(toolUse[2].id).toBe('call_2');
    });

    it('replaces a Vertex call id already used earlier in the conversation', async () => {
        const conversation = [
            { role: 'user', parts: [{ text: 'earlier' }] },
            { role: 'model', parts: [{ functionCall: { id: 'call_1', name: 'create_document', args: {} } }] },
            { role: 'user', parts: [{ functionResponse: { id: 'call_1', name: 'create_document', response: {} } }] },
        ];
        const { toolUse } = await streamParallelCalls([], {
            conversation,
            ids: ['call_1', 'call_2', 'call_3', 'call_4'],
        });

        expect(toolUse[0].id).not.toBe('call_1');
        expect(toolUse.slice(1).map((tool) => tool.id)).toEqual(['call_2', 'call_3', 'call_4']);
    });

    it('answers generated-id calls with one response per call, sent without the generated ids', async () => {
        const requests: unknown[] = [];
        const { modelDef, driver, toolUse, conversation } = await streamParallelCalls(requests);

        const request = await sendResults(
            modelDef,
            driver,
            requests,
            conversation,
            toolUse.map((tool) => tool.id),
        );

        expect(request.contents.at(-2)?.role).toBe('model');
        const calls = functionCalls(request, -2);
        expect(calls).toHaveLength(4);
        // Vertex never sees an id it did not issue.
        for (const call of calls) expect(call.functionCall).not.toHaveProperty('id');
        expect(request.contents.at(-1)?.role).toBe('user');
        const responses = functionResponses(request, -1);
        // Gemini requires the responses in ONE user turn, with as many parts as the call turn.
        expect(responses.map((part) => part.functionResponse)).toEqual(
            NAMES.map((name) => ({ name: 'create_document', response: { created: name } })),
        );
        // The thought signature travels with the first call's response only, as it was emitted.
        expect(responses[0].thoughtSignature).toBe('first-signature');
        expect(responses[1].thoughtSignature).toBeUndefined();
    });

    it("echoes Vertex's call ids on the responses", async () => {
        const requests: unknown[] = [];
        const ids = ['call_1', 'call_2', 'call_3', 'call_4'];
        const { modelDef, driver, conversation } = await streamParallelCalls(requests, { ids });

        const request = await sendResults(modelDef, driver, requests, conversation, ids);

        expect(functionCalls(request, -2).map((part) => part.functionCall?.id)).toEqual(ids);
        expect(functionResponses(request, -1).map((part) => part.functionResponse)).toEqual(
            ids.map((id, index) => ({ id, name: 'create_document', response: { created: NAMES[index] } })),
        );
    });

    it('still answers a conversation stored when the tool-use id was the function name', async () => {
        const requests: unknown[] = [];
        const modelDef = new GeminiModelDefinition('gemini-2.5-flash');
        const driver = makeDriver(async (request) => {
            requests.push(request);
            return (async function* () {})();
        });
        // Stored by an earlier driver version: calls without ids, results keyed by the name.
        const conversation = {
            _arrayConversation: [
                { role: 'user', parts: [{ text: 'create four documents' }] },
                {
                    role: 'model',
                    parts: NAMES.map((name) => ({ functionCall: { name: 'create_document', args: { name } } })),
                },
            ],
        };

        const request = await sendResults(
            modelDef,
            driver,
            requests,
            conversation,
            NAMES.map(() => 'create_document'),
        );

        expect(functionResponses(request, -1).map((part) => part.functionResponse)).toEqual(
            NAMES.map((name) => ({ name: 'create_document', response: { created: name } })),
        );
    });

    it('gives non-streamed calls their own ids too', async () => {
        const modelDef = new GeminiModelDefinition('gemini-2.5-flash');
        const driver = makeDriver(
            async () => (async function* () {})(),
            async () => parallelCreateDocumentChunk(),
        );

        const completion = await modelDef.requestTextCompletion(
            driver,
            { contents: [{ role: 'user', parts: [{ text: 'create four documents' }] }] },
            { model: MODEL, tools: TOOLS },
        );

        const ids = (completion.tool_use ?? []).map((tool) => tool.id);
        expect(new Set(ids).size).toBe(4);
        expect(storedCalls(completion.conversation).map((call) => call.id)).toEqual(ids);
    });
});
