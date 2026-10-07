import type { Content, Part } from '@google/genai';
import { applyContextChange, parseConversationDocument, planContextChange } from '@llumiverse/conversation';
import type { ExecutionOptions } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { canonicalToolDefinitions, parseCanonicalConversation } from '../../conversation/canonical-runtime.js';
import {
    compileGeminiConversation,
    GEMINI_GENERATE_CONTENT_PROTOCOL,
    importGeminiGenerateContentHistory,
    prepareGeminiCanonicalState as prepareGeminiWithTarget,
} from './gemini-conversation-adapter.js';

function options(input: {
    flow: string;
    conversation?: unknown;
    model?: string;
    operation?: string;
}): ExecutionOptions {
    const operation = input.operation ?? 'request';
    return {
        model: input.model ?? 'gemini-2.5-pro',
        ...(input.conversation === undefined ? {} : { conversation: input.conversation }),
        conversation_runtime: {
            conversation_id: `conversation:${input.flow}`,
            request_id: `request:${input.flow}:${operation}`,
            attempt_id: `attempt:${input.flow}:${operation}`,
            input_operation_id: `input:${input.flow}:${operation}`,
            response_operation_id: `response:${input.flow}:${operation}`,
            recorded_at: '2026-09-30T00:00:00.000Z',
            started_at: '2026-09-30T00:00:00.000Z',
        },
        tools: [
            {
                name: 'lookup',
                description: 'Look up a value',
                input_schema: { type: 'object', additionalProperties: true },
            },
        ],
    };
}

function legacyConversation(contents: Content[], system?: Content): unknown {
    return {
        _arrayConversation: contents,
        _llumiverse_meta: { turnNumber: 8 },
        ...(system === undefined ? {} : { _llumiverse_system: system }),
    };
}

describe('Gemini canonical adapter', () => {
    it('assigns distinct deterministic call identities and pairs repeated name-only results in order', async () => {
        const conversation = legacyConversation([
            {
                role: 'model',
                parts: [
                    { functionCall: { name: 'lookup', args: { value: 'first' } } },
                    { functionCall: { name: 'lookup', args: { value: 'second' } } },
                ],
            },
            {
                role: 'user',
                parts: [
                    { functionResponse: { name: 'lookup', response: { output: 'one' } } },
                    { functionResponse: { name: 'lookup', response: { output: 'two' } } },
                ],
            },
        ]);
        const prepared = await prepareGeminiCanonicalState({
            conversation,
            prompt: { contents: [{ role: 'user', parts: [{ text: 'continue' }] }] },
            options: options({ flow: 'parallel', conversation }),
            provider: 'vertexai',
        });

        const document = parseConversationDocument(prepared.document);
        const calls = document.turns.flatMap((turn) =>
            turn.kind === 'agent' ? turn.blocks.filter((block) => block.type === 'tool_call') : [],
        );
        const results = document.turns.flatMap((turn) => (turn.kind === 'tool' ? [turn.blocks[0]] : []));
        expect(calls).toHaveLength(2);
        expect(new Set(calls.map((call) => call.call_id)).size).toBe(2);
        expect(results.map((result) => result.call_id)).toEqual(calls.map((call) => call.call_id));

        const compiled = compileGeminiConversation(document, {
            provider: 'vertexai',
            model: 'gemini-2.5-pro',
        }).conversation;
        const projectedResults = compiled.contents.flatMap((content) =>
            (content.parts ?? []).flatMap((part) => (part.functionResponse ? [part.functionResponse] : [])),
        );
        expect(projectedResults).toEqual([
            { name: 'lookup', response: { output: 'one' } },
            { name: 'lookup', response: { output: 'two' } },
        ]);
    });

    it('preserves native ids and signed thought replay while rejecting changed semantics or model scope', async () => {
        const signed: Content = {
            role: 'model',
            parts: [
                { text: 'private reasoning', thought: true, thoughtSignature: 'reasoning-signature' },
                {
                    functionCall: { id: 'native-call-1', name: 'lookup', args: { city: 'Tokyo', nullable: null } },
                    thoughtSignature: 'call-signature',
                },
            ],
        };
        const conversation = legacyConversation([signed]);
        const prepared = await prepareGeminiCanonicalState({
            conversation,
            prompt: { contents: [] },
            options: options({ flow: 'signed', conversation }),
            provider: 'vertexai',
        });
        const document = parseConversationDocument(prepared.document);
        const agent = document.turns.find((turn) => turn.kind === 'agent');
        expect(agent?.kind).toBe('agent');
        if (agent?.kind !== 'agent') throw new Error('missing agent turn');
        const call = agent.blocks.find((block) => block.type === 'tool_call');
        expect(call).toMatchObject({ call_id: 'native-call-1', tool_name: 'lookup' });
        expect(
            compileGeminiConversation(document, { provider: 'vertexai', model: 'gemini-2.5-pro' }).conversation
                .contents[0],
        ).toEqual(signed);
        expect(() => compileGeminiConversation(document, { provider: 'vertexai', model: 'gemini-2.5-flash' })).toThrow(
            /outside its compatibility scope/,
        );

        const changed = structuredClone(document);
        const changedAgent = changed.turns.find((turn) => turn.id === agent.id);
        if (changedAgent?.kind !== 'agent') throw new Error('missing cloned agent turn');
        const reasoning = changedAgent.blocks.find((block) => block.type === 'reasoning');
        if (reasoning?.type !== 'reasoning') throw new Error('missing reasoning block');
        reasoning.text = 'changed';
        expect(() => compileGeminiConversation(changed, { provider: 'vertexai', model: 'gemini-2.5-pro' })).toThrow(
            /no longer matches canonical data/,
        );
    });

    it('round trips empty signed model parts without inventing semantic text or reasoning', async () => {
        const native: Content = {
            role: 'model',
            parts: [
                { text: 'answer' },
                { text: '', thoughtSignature: 'signed-empty-answer' },
                { text: '', thought: true, thoughtSignature: 'signed-empty-reasoning' },
            ],
        };
        const conversation = legacyConversation([native]);
        const prepared = await prepareGeminiCanonicalState({
            conversation,
            prompt: { contents: [] },
            options: options({ flow: 'empty-signed-parts', conversation }),
            provider: 'vertexai',
        });
        const document = parseConversationDocument(prepared.document);
        const agent = document.turns.find((turn) => turn.kind === 'agent');
        expect(agent?.kind).toBe('agent');
        if (agent?.kind !== 'agent') throw new Error('missing agent turn');
        expect(agent.blocks.filter((block) => block.type !== 'native_replay')).toEqual([
            expect.objectContaining({ text: 'answer' }),
        ]);
        expect(
            compileGeminiConversation(document, { provider: 'vertexai', model: 'gemini-2.5-pro' }).conversation
                .contents[0],
        ).toEqual(native);
    });

    it('round trips ordered user media and keeps protected media fields out of portable projection', async () => {
        const parts: Part[] = [
            { inlineData: { data: 'aW1hZ2U=', mimeType: 'image/png', displayName: 'chart.png' } },
            { text: 'between' },
            {
                fileData: {
                    fileUri: 'gs://bucket/document.pdf',
                    mimeType: 'application/pdf',
                    displayName: 'document.pdf',
                },
            },
        ];
        const prepared = await prepareGeminiCanonicalState({
            conversation: undefined,
            prompt: { contents: [{ role: 'user', parts }] },
            options: options({ flow: 'media' }),
            provider: 'vertexai',
        });
        const compiled = compileGeminiConversation(prepared.document, {
            provider: 'vertexai',
            model: 'gemini-2.5-pro',
        });
        const userTurn = prepared.document.turns.find((turn) => turn.kind === 'user');
        const mediaBlock = userTurn?.blocks.find((block) => block.type === 'image');
        expect(mediaBlock).toBeDefined();
        expect(compiled.mappings).toContainEqual(
            expect.objectContaining({ canonical_id: mediaBlock?.id, kind: 'block' }),
        );
        expect(compiled.conversation.contents).toEqual([{ role: 'user', parts }]);

        await expect(
            prepareGeminiCanonicalState({
                conversation: undefined,
                prompt: {
                    contents: [
                        {
                            role: 'user',
                            parts: [
                                {
                                    inlineData: { data: 'aW1hZ2U=', mimeType: 'image/png' },
                                    thoughtSignature: 'must-not-leak',
                                },
                            ],
                        },
                    ],
                },
                options: options({ flow: 'protected-media' }),
                provider: 'vertexai',
            }),
        ).rejects.toThrow(/unsupported protected field thoughtSignature/);
    });

    it('preserves tool-result status, JSON, attachments, and explicit call association', async () => {
        const call: Content = {
            role: 'model',
            parts: [{ functionCall: { id: 'call-explicit', name: 'lookup', args: { city: 'Paris' } } }],
        };
        const resultPart = {
            functionResponse: {
                id: 'call-explicit',
                name: 'lookup',
                response: { error: 'not found', details: null },
                parts: [{ inlineData: { data: 'ZXJyb3I=', mimeType: 'image/png' } }],
            },
            _llumiverse_tool_result_status: 'error' as const,
            thoughtSignature: 'result-signature',
        } satisfies Part & { _llumiverse_tool_result_status: 'error' };
        const conversation = legacyConversation([call]);
        const prepared = await prepareGeminiCanonicalState({
            conversation,
            prompt: { contents: [{ role: 'user', parts: [resultPart] }] },
            options: options({ flow: 'tool-result', conversation }),
            provider: 'vertexai',
        });
        const document = parseConversationDocument(prepared.document);
        const toolTurn = document.turns.find((turn) => turn.kind === 'tool');
        expect(toolTurn?.kind).toBe('tool');
        if (toolTurn?.kind !== 'tool') throw new Error('missing tool turn');
        expect(toolTurn.blocks[0]).toMatchObject({ call_id: 'call-explicit', status: 'error' });
        expect(Object.values(document.execution_receipts)).toContainEqual(
            expect.objectContaining({ call_id: 'call-explicit', status: 'error' }),
        );
        const compiled = compileGeminiConversation(document, {
            provider: 'vertexai',
            model: 'gemini-2.5-pro',
        }).conversation;
        const projected = compiled.contents.at(-1)?.parts?.[0];
        expect(projected).toEqual({
            functionResponse: resultPart.functionResponse,
            thoughtSignature: 'result-signature',
        });
        expect(JSON.stringify(compiled)).not.toContain('_llumiverse_tool_result_status');
        expect(GEMINI_GENERATE_CONTENT_PROTOCOL).toBe('google.generate_content');
    });

    it.each([
        [
            'plain text',
            'IBM trades on the New York Stock Exchange (NYSE).',
            { output: 'IBM trades on the New York Stock Exchange (NYSE).' },
        ],
        ['JSON-looking text', '{"exchange":"NYSE"}', { exchange: 'NYSE' }],
    ])(
        'preserves exact legacy %s tool-result text while projecting its native response',
        async (_label, text, response) => {
            const call: Content = {
                role: 'model',
                parts: [{ functionCall: { id: 'call-text', name: 'lookup', args: {} } }],
            };
            const result = {
                functionResponse: { id: 'call-text', name: 'lookup', response },
                _llumiverse_tool_result_text: text,
            } satisfies Part & { _llumiverse_tool_result_text: string };
            const conversation = legacyConversation([call]);
            const prepared = await prepareGeminiCanonicalState({
                conversation,
                prompt: { contents: [{ role: 'user', parts: [result] }] },
                options: options({ flow: `legacy-text-${_label}`, conversation }),
                provider: 'vertexai',
            });
            const document = parseConversationDocument(prepared.document);
            const toolTurn = document.turns.find((turn) => turn.kind === 'tool');
            expect(toolTurn?.kind).toBe('tool');
            if (toolTurn?.kind !== 'tool') throw new Error('missing tool turn');
            expect(toolTurn.blocks[0].content).toContainEqual(expect.objectContaining({ type: 'text', text }));

            const compiled = compileGeminiConversation(document, {
                provider: 'vertexai',
                model: 'gemini-2.5-pro',
            }).conversation;
            expect(compiled.contents.at(-1)?.parts?.[0].functionResponse?.response).toEqual(response);
            expect(JSON.stringify(compiled)).not.toContain('_llumiverse_tool_result_text');
        },
    );

    it('rejects a legacy tool-result text carrier that does not reproduce the native response', async () => {
        const conversation = legacyConversation([
            { role: 'model', parts: [{ functionCall: { id: 'call-text', name: 'lookup', args: {} } }] },
        ]);
        await expect(
            prepareGeminiCanonicalState({
                conversation,
                prompt: {
                    contents: [
                        {
                            role: 'user',
                            parts: [
                                {
                                    functionResponse: {
                                        id: 'call-text',
                                        name: 'lookup',
                                        response: { output: 'different' },
                                    },
                                    _llumiverse_tool_result_text: 'original',
                                } as Part,
                            ],
                        },
                    ],
                },
                options: options({ flow: 'legacy-text-tampered', conversation }),
                provider: 'vertexai',
            }),
        ).rejects.toThrow(/legacy tool-result text does not match/);
    });

    it('rejects explicit tool-result ids that are unknown or name a different tool', async () => {
        const conversation = legacyConversation([
            {
                role: 'model',
                parts: [
                    { functionCall: { id: 'call-a', name: 'lookup', args: { value: 'first' } } },
                    { functionCall: { id: 'call-b', name: 'lookup', args: { value: 'second' } } },
                ],
            },
        ]);

        await expect(
            prepareGeminiCanonicalState({
                conversation,
                prompt: {
                    contents: [
                        {
                            role: 'user',
                            parts: [
                                { functionResponse: { id: 'call-missing', name: 'lookup', response: { output: 1 } } },
                            ],
                        },
                    ],
                },
                options: options({ flow: 'wrong-id', conversation }),
                provider: 'vertexai',
            }),
        ).rejects.toThrow(/call-missing has no matching function call/);

        await expect(
            prepareGeminiCanonicalState({
                conversation,
                prompt: {
                    contents: [
                        {
                            role: 'user',
                            parts: [{ functionResponse: { id: 'call-a', name: 'other', response: { output: 1 } } }],
                        },
                    ],
                },
                options: options({ flow: 'wrong-name', conversation }),
                provider: 'vertexai',
            }),
        ).rejects.toThrow(/names other, expected lookup/);
    });

    it('projects ordinary program contributions as user content without promoting authority', async () => {
        const prepared = await prepareGeminiCanonicalState({
            conversation: undefined,
            prompt: { contents: [{ role: 'user', parts: [{ text: 'ordinary instruction' }] }] },
            options: options({ flow: 'ordinary-program' }),
            provider: 'vertexai',
        });
        const source = prepared.document.turns[0];
        const ordinary = parseConversationDocument({
            ...prepared.document,
            turns: [{ ...source, kind: 'program', authority: 'ordinary' }],
        });
        expect(compileGeminiConversation(ordinary).conversation).toMatchObject({
            contents: [{ role: 'user', parts: [{ text: 'ordinary instruction' }] }],
        });

        const developer = parseConversationDocument({
            ...ordinary,
            turns: [{ ...ordinary.turns[0], authority: 'developer' }],
        });
        expect(() => compileGeminiConversation(developer)).toThrow(/developer program authority/);
    });

    it('honors selected blocks and fails closed when selection breaks protected replay', async () => {
        const prepared = await prepareGeminiCanonicalState({
            conversation: undefined,
            prompt: { contents: [{ role: 'user', parts: [{ text: 'keep' }, { text: 'omit' }] }] },
            options: options({ flow: 'selection' }),
            provider: 'vertexai',
        });
        const user = prepared.document.turns.find((turn) => turn.kind === 'user');
        if (user?.kind !== 'user') throw new Error('missing user turn');
        const userEntry = prepared.document.context.entries.find(
            (entry) => entry.type === 'source_turn' && entry.turn_id === user.id,
        );
        if (userEntry === undefined) throw new Error('missing user context entry');
        const selection = {
            expected_revision: prepared.document.revision,
            expected_context_revision: prepared.document.context.revision,
            entry_ids: [userEntry.id],
            selected_block_ids: { [userEntry.id]: [user.blocks[1].id] },
            selected_entries: [userEntry],
        };
        const plan = await planContextChange(prepared.document, selection);
        const selected = (
            await applyContextChange(prepared.document, {
                ...selection,
                operation_id: 'exclude:omitted-user-block',
                expected_source_fingerprint: plan.source_fingerprint,
                recorded_at: '2026-09-30T00:00:00.000Z',
                proposal: { kind: 'exclude' },
            })
        ).document;
        expect(compileGeminiConversation(selected).conversation.contents).toEqual([
            { role: 'user', parts: [{ text: 'keep' }] },
        ]);
        expect(prepared.document.turns.find((turn) => turn.id === user.id)?.blocks).toHaveLength(2);

        const signed = legacyConversation([
            {
                role: 'model',
                parts: [
                    { text: 'first', thought: true, thoughtSignature: 'first-signature' },
                    { text: 'second', thought: true, thoughtSignature: 'second-signature' },
                ],
            },
        ]);
        const signedPrepared = await prepareGeminiCanonicalState({
            conversation: signed,
            prompt: { contents: [] },
            options: options({ flow: 'signed-selection', conversation: signed }),
            provider: 'vertexai',
        });
        const agent = signedPrepared.document.turns.find((turn) => turn.kind === 'agent');
        if (agent?.kind !== 'agent') throw new Error('missing signed agent turn');
        const selectedSigned = {
            ...signedPrepared.document,
            context: {
                ...signedPrepared.document.context,
                entries: signedPrepared.document.context.entries.map((entry) =>
                    entry.type === 'source_turn' && entry.turn_id === agent.id
                        ? { ...entry, block_ids: [agent.blocks[0].id] }
                        : entry,
                ),
            },
        };
        expect(() =>
            compileGeminiConversation(selectedSigned, { provider: 'vertexai', model: 'gemini-2.5-pro' }),
        ).toThrow(/protected replay|replay dependency|no longer matches canonical data/);
        const signedEntry = signedPrepared.document.context.entries.find(
            (entry) => entry.type === 'source_turn' && entry.turn_id === agent.id,
        );
        if (signedEntry === undefined) throw new Error('missing signed agent context entry');
        await expect(
            planContextChange(signedPrepared.document, {
                expected_revision: signedPrepared.document.revision,
                expected_context_revision: signedPrepared.document.context.revision,
                entry_ids: [signedEntry.id],
                selected_block_ids: { [signedEntry.id]: [agent.blocks[1].id] },
                selected_entries: [signedEntry],
            }),
        ).rejects.toThrow(/protected replay|replay dependency/);
    });
});

/** This fixture archive declares its original invocation route; target options are not origin evidence. */
async function prepareGeminiCanonicalState(input: Parameters<typeof prepareGeminiWithTarget>[0]) {
    const conversation =
        input.conversation == null || parseCanonicalConversation(input.conversation) !== undefined
            ? input.conversation
            : (
                  await importGeminiGenerateContentHistory(input.conversation, {
                      conversation_id: input.options.conversation_runtime?.conversation_id ?? 'fixture-history',
                      recorded_at: input.options.conversation_runtime?.recorded_at ?? '2026-09-30T00:00:00.000Z',
                      provider: 'vertexai',
                      model: 'gemini-2.5-pro',
                      source_request_id: input.options.conversation_runtime?.request_id,
                      tool_definitions: await canonicalToolDefinitions(input.options.tools),
                  })
              ).document;
    return prepareGeminiWithTarget({ ...input, conversation });
}
