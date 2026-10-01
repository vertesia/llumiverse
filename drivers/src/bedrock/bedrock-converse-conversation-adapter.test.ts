import type { ContentBlock, ConverseRequest, TokenUsage } from '@aws-sdk/client-bedrock-runtime';
import {
    externalizeToolCallArguments,
    parseConversationDocument,
    prepareToolArgumentExternalization,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import {
    BEDROCK_CONVERSE_ADAPTER_VERSION,
    BEDROCK_CONVERSE_PROTOCOL,
    bedrockConverseGenerationUsage,
    bedrockConverseJsonValue,
    compileBedrockConverseConversation,
    exportLegacyBedrockConverseConversation,
    importBedrockConverseConversation,
    isBedrockConverseHistory,
} from './bedrock-converse-conversation-adapter.js';

const recordedAt = '2026-09-30T00:00:00.000Z';

function history(messages: NonNullable<ConverseRequest['messages']>): Pick<ConverseRequest, 'messages' | 'system'> {
    return {
        system: [{ text: 'Keep the first instruction.' }, { text: 'Keep the second instruction separate.' }],
        messages,
    };
}

function importHistory(value: unknown, id: string, model?: string) {
    return importBedrockConverseConversation(value, {
        conversation_id: id,
        recorded_at: recordedAt,
        provider: 'bedrock',
        ...(model === undefined ? {} : { model }),
    });
}

describe('Bedrock Converse canonical adapter', () => {
    it('round trips text, parallel calls, and typed JSON tool results through canonical JSON', async () => {
        const native = history([
            { role: 'user', content: [{ text: 'Compare two places.' }] },
            {
                role: 'assistant',
                content: [
                    {
                        toolUse: {
                            toolUseId: 'call-tokyo',
                            name: 'weather',
                            input: {
                                city: 'Tokyo',
                                protocol: 'customer.protocol',
                                adapter: { version: 7, enabled: true },
                            },
                        },
                    },
                    {
                        toolUse: {
                            toolUseId: 'call-osaka',
                            name: 'weather',
                            input: { city: 'Osaka', optional: null, values: [0, false, ''] },
                        },
                    },
                ],
            },
            {
                role: 'user',
                content: [
                    {
                        toolResult: {
                            toolUseId: 'call-tokyo',
                            status: 'success',
                            content: [
                                { json: { temperature: 28, protocol: 'result.protocol', adapter: ['unchanged'] } },
                            ],
                        },
                    },
                    {
                        toolResult: {
                            toolUseId: 'call-osaka',
                            status: 'error',
                            content: [{ text: 'sensor unavailable' }, { json: ['retry', { after: 3 }] }],
                        },
                    },
                ],
            },
        ]);
        const original = structuredClone(native);
        const document = await importHistory(native, 'bedrock-roundtrip');
        const persisted = parseConversationDocument(JSON.parse(JSON.stringify(document)));

        expect(exportLegacyBedrockConverseConversation(persisted)).toEqual(original);
        expect(native).toEqual(original);
        const calls = persisted.turns.flatMap((turn) =>
            turn.blocks.flatMap((block) => (block.type === 'tool_call' ? [block] : [])),
        );
        expect(calls.map((call) => [call.tool_name, call.call_id])).toEqual([
            ['weather', 'call-tokyo'],
            ['weather', 'call-osaka'],
        ]);
        const results = persisted.turns.filter((turn) => turn.kind === 'tool');
        expect(results[0]?.blocks[0].content[0]?.type).toBe('json');
        expect(results[1]?.blocks[0].content.map((block) => block.type)).toEqual(['text', 'json']);
    });

    it('preserves explicit client and provider tool-use types without making provider calls executable', async () => {
        const native = history([
            { role: 'user', content: [{ text: 'Find and call the weather tool.' }] },
            {
                role: 'assistant',
                content: [
                    {
                        toolUse: {
                            toolUseId: 'server-search',
                            name: 'tool_search_tool_regex',
                            input: { query: 'weather' },
                            type: 'server_tool_use',
                        },
                    },
                    {
                        toolUse: {
                            toolUseId: 'client-weather',
                            name: 'weather',
                            input: { city: 'Tokyo' },
                            // Accept the native client discriminator for compatibility even though the generated
                            // SDK enum currently only declares server_tool_use.
                            type: 'tool_use',
                        },
                    } as unknown as ContentBlock,
                ],
            },
        ]);
        const document = parseConversationDocument(
            JSON.parse(
                JSON.stringify(
                    await importHistory(native, 'typed-bedrock-tool-use', 'anthropic.claude-3-5-sonnet-20241022-v2:0'),
                ),
            ),
        );
        const calls = document.turns.flatMap((turn) =>
            turn.blocks.flatMap((block) => (block.type === 'tool_call' ? [block] : [])),
        );

        expect(calls.map(({ call_id, executor }) => [call_id, executor])).toEqual([
            ['server-search', 'provider'],
            ['client-weather', 'application'],
        ]);
        const replay = document.turns
            .flatMap((turn) => (turn.kind === 'agent' ? turn.blocks : []))
            .find((block) => block.type === 'native_replay');
        expect(replay).toMatchObject({
            dependencies: {
                call_ids: expect.arrayContaining(['server-search', 'client-weather']),
            },
        });
        expect(replay).not.toHaveProperty('dependency_policy');
        expect(exportLegacyBedrockConverseConversation(document)).toEqual(native);

        const withoutReplay = structuredClone(document);
        const agent = withoutReplay.turns.find((turn) => turn.kind === 'agent');
        if (agent?.kind !== 'agent') throw new Error('Expected agent turn');
        agent.blocks = agent.blocks.filter((block) => block.type !== 'native_replay');
        expect(() => compileBedrockConverseConversation(withoutReplay)).toThrow(
            /provider call server-search requires protected native replay evidence/,
        );

        const changedProviderCall = structuredClone(document);
        const providerCall = changedProviderCall.turns
            .flatMap((turn) => (turn.kind === 'agent' ? turn.blocks : []))
            .find((block) => block.type === 'tool_call' && block.executor === 'provider');
        if (providerCall?.type !== 'tool_call' || providerCall.arguments.type !== 'json') {
            throw new Error('Expected provider tool call');
        }
        providerCall.arguments.value = { query: 'different' };
        expect(() =>
            compileBedrockConverseConversation(changedProviderCall, {
                provider: 'bedrock',
                model: 'anthropic.claude-3-5-sonnet-20241022-v2:0',
            }),
        ).toThrow(/no longer matches protected content block/);

        for (const [callId, executor] of [
            ['server-search', 'application'],
            ['client-weather', 'provider'],
        ] as const) {
            const changedOwnership = structuredClone(document);
            const call = changedOwnership.turns
                .flatMap((turn) => (turn.kind === 'agent' ? turn.blocks : []))
                .find((block) => block.type === 'tool_call' && block.call_id === callId);
            if (call?.type !== 'tool_call') throw new Error(`Expected tool call ${callId}`);
            call.executor = executor;
            expect(() =>
                compileBedrockConverseConversation(changedOwnership, {
                    provider: 'bedrock',
                    model: 'anthropic.claude-3-5-sonnet-20241022-v2:0',
                }),
            ).toThrow(/no longer matches protected tool execution ownership/);
        }

        const discardableServerReplay = structuredClone(document);
        const protectedReplay = discardableServerReplay.turns
            .flatMap((turn) => (turn.kind === 'agent' ? turn.blocks : []))
            .find((block) => block.type === 'native_replay');
        if (protectedReplay?.type !== 'native_replay') throw new Error('Expected protected replay');
        protectedReplay.dependency_policy = 'discard_on_dependency_change';
        expect(() => compileBedrockConverseConversation(discardableServerReplay)).toThrow(
            /cannot discard protected native evidence/,
        );
    });

    it('accepts an own undefined tool-use type without inventing replay or changing execution ownership', async () => {
        const toolUse = { toolUseId: 'sdk-call', name: 'weather', input: { city: 'Tokyo' }, type: undefined };
        expect(Object.hasOwn(toolUse, 'type')).toBe(true);
        const document = await importHistory(
            history([{ role: 'assistant', content: [{ toolUse } as unknown as ContentBlock] }]),
            'undefined-bedrock-tool-use-type',
        );
        const blocks = document.turns.flatMap((turn) => (turn.kind === 'agent' ? turn.blocks : []));

        expect(blocks.find((block) => block.type === 'tool_call')).toMatchObject({
            call_id: 'sdk-call',
            executor: 'application',
        });
        expect(blocks.some((block) => block.type === 'native_replay')).toBe(false);
    });

    it('projects the compact model view of canonical externalized tool arguments', async () => {
        const document = await importHistory(
            history([
                {
                    role: 'assistant',
                    content: [
                        {
                            toolUse: {
                                toolUseId: 'write-call',
                                name: 'write_artifact',
                                input: { name: 'large.txt', content: 'exact executable content' },
                            },
                        },
                    ],
                },
            ]),
            'bedrock-externalized-tool-input',
        );
        const prepared = await prepareToolArgumentExternalization(document, 'write-call', ['content']);
        const externalized = await externalizeToolCallArguments(document, {
            operation_id: 'externalize-write-call',
            expected_revision: document.revision,
            recorded_at: recordedAt,
            call_id: 'write-call',
            input_path: ['content'],
            model_value: { name: 'large.txt', content: '[stored compact view]' },
            exact_arguments_hash: prepared.exact_arguments_hash,
            asset: {
                id: 'asset-write-call',
                kind: 'text',
                mime_type: 'text/plain',
                storage: {
                    type: 'external',
                    resolver: 'test.artifact',
                    locator: { path: 'tool-inputs/write-call.txt' },
                },
                provenance: { type: 'imported', source: 'test' },
                byte_length: prepared.byte_length,
                content_hash: prepared.content_hash,
                created_at: recordedAt,
            },
        });

        expect(compileBedrockConverseConversation(externalized.document).conversation.messages?.at(-1)).toEqual({
            role: 'assistant',
            content: [
                {
                    toolUse: {
                        toolUseId: 'write-call',
                        name: 'write_artifact',
                        input: { name: 'large.txt', content: '[stored compact view]' },
                    },
                },
            ],
        });
    });

    it('preserves interleaved tool results and user text in their original native message', async () => {
        const native = history([
            {
                role: 'assistant',
                content: [
                    { toolUse: { toolUseId: 'first', name: 'lookup', input: { value: 1 } } },
                    { toolUse: { toolUseId: 'second', name: 'lookup', input: { value: 2 } } },
                ],
            },
            {
                role: 'user',
                content: [
                    { toolResult: { toolUseId: 'first', content: [{ json: { value: 1 } }] } },
                    { text: 'Keep this exact position.' },
                    { toolResult: { toolUseId: 'second', content: [{ json: { value: 2 } }] } },
                ],
            },
        ]);
        const document = await importHistory(native, 'bedrock-interleaved-results');

        expect(exportLegacyBedrockConverseConversation(document)).toEqual(native);
    });

    it('round trips media plus signed and redacted reasoning through canonical JSON', async () => {
        const native = history([
            {
                role: 'user',
                content: [
                    { image: { format: 'png', source: { bytes: new Uint8Array([0, 1, 255]) } } },
                    { text: 'Describe this.' },
                ],
            },
            {
                role: 'assistant',
                content: [
                    { reasoningContent: { reasoningText: { text: 'private analysis', signature: 'signed-value' } } },
                    { reasoningContent: { redactedContent: new Uint8Array([9, 8, 7]) } },
                    { text: 'Visible answer.' },
                    { toolUse: { toolUseId: 'media-call', name: 'inspect', input: { exact: true } } },
                ],
            },
            {
                role: 'user',
                content: [
                    {
                        toolResult: {
                            toolUseId: 'media-call',
                            content: [
                                { image: { format: 'jpeg', source: { bytes: new Uint8Array([4, 5, 6]) } } },
                                { json: { protocol: 'customer', adapter: { _llumiverse_bedrock_bytes: 'literal' } } },
                            ],
                        },
                    },
                ],
            },
        ]);
        const document = await importHistory(native, 'bedrock-media-reasoning', 'anthropic.claude-sonnet-4-6');
        const persisted = parseConversationDocument(JSON.parse(JSON.stringify(document)));

        const signedTurnIndex = persisted.turns.findIndex((turn) =>
            turn.blocks.some((block) => block.type === 'native_replay'),
        );
        const signedTurn = persisted.turns[signedTurnIndex];
        const signedReplay = signedTurn?.blocks.find((block) => block.type === 'native_replay');
        if (signedReplay?.type !== 'native_replay') throw new Error('Expected signed replay block');
        const signedPrefixTurns = persisted.turns.slice(0, signedTurnIndex + 1);
        const expectedBlockIds = signedPrefixTurns.flatMap((turn) =>
            turn.blocks.flatMap((block) => [
                ...(block.type === 'native_replay' ? [] : [block.id]),
                ...(block.type === 'tool_result' ? block.content.map((nested) => nested.id) : []),
            ]),
        );
        const expectedCallIds = signedPrefixTurns.flatMap((turn) =>
            turn.blocks.flatMap((block) => (block.type === 'tool_call' ? [block.call_id] : [])),
        );
        expect(new Set(signedReplay.dependencies.turn_ids)).toEqual(new Set(signedPrefixTurns.map((turn) => turn.id)));
        expect(new Set(signedReplay.dependencies.block_ids)).toEqual(new Set(expectedBlockIds));
        expect(new Set(signedReplay.dependencies.call_ids)).toEqual(new Set(expectedCallIds));

        expect(
            exportLegacyBedrockConverseConversation(persisted, {
                provider: 'bedrock',
                model: 'anthropic.claude-sonnet-4-6',
            }),
        ).toEqual(native);
        expect(
            persisted.turns
                .filter((turn) => turn.kind === 'tool')
                .flatMap((turn) => turn.blocks[0].content)
                .find((block) => block.type === 'json'),
        ).toMatchObject({ value: { protocol: 'customer', adapter: { _llumiverse_bedrock_bytes: 'literal' } } });
    });

    it('rejects edits to signed reasoning evidence and its preceding native chain', async () => {
        const native = history([
            {
                role: 'user',
                content: [
                    { image: { format: 'png', source: { bytes: new Uint8Array([1, 2, 3]) } } },
                    { text: 'Bound prefix.' },
                ],
            },
            {
                role: 'assistant',
                content: [
                    { reasoningContent: { reasoningText: { text: 'bound reasoning', signature: 'signature' } } },
                    { toolUse: { toolUseId: 'bound-call', name: 'lookup', input: { exact: true } } },
                ],
            },
        ]);
        const reasoningEdited = await importHistory(native, 'bedrock-reasoning-edit', 'anthropic.claude-sonnet-4-6');
        const reasoning = reasoningEdited.turns
            .find((turn) => turn.kind === 'agent')
            ?.blocks.find((block) => block.type === 'reasoning');
        if (reasoning?.type !== 'reasoning') throw new Error('Expected reasoning block');
        reasoning.text = 'changed reasoning';
        expect(() =>
            compileBedrockConverseConversation(reasoningEdited, {
                provider: 'bedrock',
                model: 'anthropic.claude-sonnet-4-6',
            }),
        ).toThrow(/no longer matches protected reasoning block/);

        const prefixEdited = await importHistory(native, 'bedrock-prefix-edit', 'anthropic.claude-sonnet-4-6');
        const prefixTurn = prefixEdited.turns.find(
            (turn) =>
                turn.kind === 'user' &&
                turn.blocks.some((block) => block.type === 'text' && block.text === 'Bound prefix.'),
        );
        const prefixText =
            prefixTurn?.kind === 'user'
                ? prefixTurn.blocks.find((block) => block.type === 'text' && block.text === 'Bound prefix.')
                : undefined;
        if (prefixText?.type !== 'text') throw new Error('Expected prefix text block');
        prefixText.text = 'Changed prefix.';
        expect(() =>
            compileBedrockConverseConversation(prefixEdited, {
                provider: 'bedrock',
                model: 'anthropic.claude-sonnet-4-6',
            }),
        ).toThrow(/no longer matches protected preceding conversation/);

        const toolEdited = await importHistory(native, 'bedrock-tool-edit', 'anthropic.claude-sonnet-4-6');
        const call = toolEdited.turns
            .find((turn) => turn.kind === 'agent')
            ?.blocks.find((block) => block.type === 'tool_call');
        if (call?.type !== 'tool_call' || call.arguments.type !== 'json') throw new Error('Expected tool call block');
        call.arguments.value = { exact: false };
        expect(() =>
            compileBedrockConverseConversation(toolEdited, {
                provider: 'bedrock',
                model: 'anthropic.claude-sonnet-4-6',
            }),
        ).toThrow(/no longer matches protected content block/);

        const mediaEdited = await importHistory(native, 'bedrock-media-edit', 'anthropic.claude-sonnet-4-6');
        const image = Object.values(mediaEdited.assets).find((asset) => asset.kind === 'image');
        if (image?.storage.type !== 'inline_base64') throw new Error('Expected inline image asset');
        image.storage.data = Buffer.from([9, 9, 9]).toString('base64');
        expect(() =>
            compileBedrockConverseConversation(mediaEdited, {
                provider: 'bedrock',
                model: 'anthropic.claude-sonnet-4-6',
            }),
        ).toThrow(/no longer matches protected preceding conversation/);
    });

    it('keeps unknown input usage unknown and marks omitted cache counters as derived zero', () => {
        const missingInput = { outputTokens: 3, totalTokens: 3 } as unknown as TokenUsage;
        expect(bedrockConverseGenerationUsage(missingInput)).toMatchObject({
            output_tokens: 3,
        });
        expect(bedrockConverseGenerationUsage(missingInput)).not.toHaveProperty('input_tokens');
        expect(bedrockConverseGenerationUsage({ inputTokens: 4, outputTokens: 3, totalTokens: 7 })).toMatchObject({
            input_tokens: 4,
            input_new_tokens: 4,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            total_tokens: 7,
            accounting_provenance: {
                cache_read_tokens: { method: 'derived' },
                cache_write_tokens: { method: 'derived' },
            },
        });
        expect(bedrockConverseGenerationUsage({ inputTokens: -1, outputTokens: 3, totalTokens: 2 })).not.toHaveProperty(
            'input_tokens',
        );
    });

    it('reports model-family media and reasoning replay capability boundaries', async () => {
        const video = history([
            {
                role: 'user',
                content: [{ video: { format: 'mp4', source: { bytes: new Uint8Array([1, 2, 3]) } } }],
            },
        ]);
        const document = await importHistory(video, 'bedrock-video');

        expect(() =>
            compileBedrockConverseConversation(document, {
                provider: 'bedrock',
                model: 'anthropic.claude-sonnet-4-6',
            }),
        ).toThrow(/does not support canonical video input/);
        expect(
            compileBedrockConverseConversation(document, { provider: 'bedrock', model: 'amazon.nova-pro-v1:0' })
                .conversation,
        ).toEqual(video);
    });

    it('rejects non-JSON tool values while preserving a literal customer byte-sentinel object', async () => {
        await expect(
            importHistory(
                {
                    messages: [
                        {
                            role: 'assistant',
                            content: [
                                {
                                    toolUse: {
                                        toolUseId: 'bad-json',
                                        name: 'lookup',
                                        input: { bytes: new Uint8Array([1]) },
                                    },
                                },
                            ],
                        },
                    ],
                },
                'bedrock-bad-tool-json',
            ),
        ).rejects.toThrow(/must be exact JSON data/);

        const sentinel = { _llumiverse_bedrock_bytes: 'customer-owned', protocol: 'literal', adapter: false };
        const native = {
            messages: [
                {
                    role: 'assistant' as const,
                    content: [{ toolUse: { toolUseId: 'literal', name: 'lookup', input: sentinel } }],
                },
            ],
        };
        expect(
            exportLegacyBedrockConverseConversation(await importHistory(native, 'bedrock-literal-sentinel')),
        ).toEqual(native);
    });

    it('rejects reserved __proto__ keys without dropping them during transport cloning', () => {
        const primitive = JSON.parse('{"__proto__":"customer-value"}') as unknown;
        const object = JSON.parse('{"__proto__":{"customer":true}}') as unknown;

        expect(() => bedrockConverseJsonValue(primitive)).toThrow(/exact JSON data/);
        expect(() => bedrockConverseJsonValue(object)).toThrow(/exact JSON data/);
        expect(
            bedrockConverseJsonValue(JSON.parse('{"constructor":"customer-constructor","toString":{"customer":true}}')),
        ).toEqual({ constructor: 'customer-constructor', toString: { customer: true } });
    });

    it('compiles selected context without deleting excluded source history', async () => {
        const document = await importHistory(
            history([
                { role: 'user', content: [{ text: 'Old question.' }] },
                { role: 'assistant', content: [{ text: 'Old answer.' }] },
                { role: 'user', content: [{ text: 'Current question.' }] },
            ]),
            'bedrock-context',
        );
        const current = document.turns.find((turn) =>
            turn.blocks.some((block) => block.type === 'text' && block.text === 'Current question.'),
        );
        if (current === undefined) throw new Error('Expected current question turn');
        const selected = structuredClone(document);
        selected.context.entries = selected.context.entries.filter((entry) => {
            const turn = selected.turns.find((candidate) => candidate.id === entry.turn_id);
            return entry.turn_id === current.id || turn?.kind === 'program';
        });
        const before = structuredClone(selected);

        expect(compileBedrockConverseConversation(selected).conversation).toEqual({
            system: [{ text: 'Keep the first instruction.' }, { text: 'Keep the second instruction separate.' }],
            messages: [{ role: 'user', content: [{ text: 'Current question.' }] }],
        });
        expect(selected).toEqual(before);
        expect(selected.turns).toHaveLength(document.turns.length);
    });

    it('rejects protected replay outside the Bedrock provider and model scope', async () => {
        const document = await importHistory(
            history([{ role: 'assistant', content: [{ text: 'Answer.' }] }]),
            'bedrock-replay',
        );
        const agent = document.turns.find((turn) => turn.kind === 'agent');
        if (agent === undefined) throw new Error('Expected imported agent turn');
        agent.blocks.push({
            id: 'wrong-scope-replay',
            type: 'native_replay',
            adapter: BEDROCK_CONVERSE_ADAPTER_VERSION,
            protocol: BEDROCK_CONVERSE_PROTOCOL,
            compatibility_scope: {
                provider: 'other-provider',
                protocol: BEDROCK_CONVERSE_PROTOCOL,
                model: 'model-a',
                adapter_version: BEDROCK_CONVERSE_ADAPTER_VERSION,
            },
            payload: { protocol: 'user-owned', adapter: 'user-owned' },
            dependencies: {
                turn_ids: [agent.id],
                block_ids: agent.blocks.map((block) => block.id),
                call_ids: [],
                request_ids: [],
            },
        });

        expect(() => compileBedrockConverseConversation(document, { provider: 'bedrock', model: 'model-a' })).toThrow(
            /outside its compatibility scope/,
        );
    });

    it('requires a real native shape unless the Bedrock protocol is explicit', () => {
        expect(isBedrockConverseHistory({})).toBe(false);
        expect(isBedrockConverseHistory({ messages: undefined })).toBe(false);
        expect(isBedrockConverseHistory({}, BEDROCK_CONVERSE_PROTOCOL)).toBe(true);
        expect(isBedrockConverseHistory({ messages: [] }, BEDROCK_CONVERSE_PROTOCOL)).toBe(true);
    });

    it('projects ordinary program authorship as user and rejects distinct developer authority', async () => {
        const document = await importHistory(
            history([{ role: 'user', content: [{ text: 'Question.' }] }]),
            'authority',
        );
        const firstProgram = document.turns.find((turn) => turn.kind === 'program');
        if (firstProgram?.kind !== 'program') throw new Error('Expected program turn');
        firstProgram.authority = 'ordinary';
        expect(compileBedrockConverseConversation(document).conversation.messages?.[0]).toEqual({
            role: 'user',
            content: [{ text: 'Keep the first instruction.' }],
        });
        firstProgram.authority = 'developer';
        expect(() => compileBedrockConverseConversation(document)).toThrow(
            /no distinct developer-authority projection/,
        );
    });

    it('maps every nested tool-result content block in the compiled request', async () => {
        const document = await importHistory(
            history([
                {
                    role: 'assistant',
                    content: [{ toolUse: { toolUseId: 'mapped', name: 'lookup', input: {} } }],
                },
                {
                    role: 'user',
                    content: [
                        {
                            toolResult: {
                                toolUseId: 'mapped',
                                content: [{ text: 'one' }, { json: { two: 2 } }],
                            },
                        },
                    ],
                },
            ]),
            'nested-mappings',
        );
        const nestedIds = document.turns
            .filter((turn) => turn.kind === 'tool')
            .flatMap((turn) => turn.blocks[0].content.map((block) => block.id));
        const mappedIds = new Set(
            compileBedrockConverseConversation(document).mappings.map((mapping) => mapping.canonical_id),
        );

        expect(nestedIds).toHaveLength(2);
        expect(nestedIds.every((id) => mappedIds.has(id))).toBe(true);
    });

    it('rejects malformed roles and tool-call associations', async () => {
        await expect(
            importHistory({ messages: [{ role: 'system', content: [{ text: 'invalid' }] }] }, 'bad-role'),
        ).rejects.toThrow(/unsupported role/);
        await expect(
            importHistory(
                {
                    messages: [
                        {
                            role: 'user',
                            content: [{ toolResult: { toolUseId: 'missing-call', content: [{ json: { ok: true } }] } }],
                        },
                    ],
                },
                'missing-call',
            ),
        ).rejects.toThrow(/has no prior tool call/);
        const duplicateCall: ContentBlock = {
            toolUse: { toolUseId: 'duplicate', name: 'lookup', input: {} },
        };
        await expect(
            importHistory(
                {
                    messages: [
                        { role: 'assistant', content: [duplicateCall] },
                        { role: 'assistant', content: [structuredClone(duplicateCall)] },
                    ],
                },
                'duplicate-call',
            ),
        ).rejects.toThrow(/duplicated/);
        await expect(
            importHistory(
                {
                    messages: [
                        {
                            role: 'assistant',
                            content: [
                                {
                                    toolUse: {
                                        toolUseId: 'unsupported-type',
                                        name: 'lookup',
                                        input: {},
                                        type: 'unsupported_tool_use',
                                    },
                                },
                            ],
                        },
                    ],
                },
                'unsupported-tool-use-type',
            ),
        ).rejects.toThrow(/tool use type .* unsupported/);
        const oversizedType = 'x'.repeat(1_000);
        const oversizedFailure = await importHistory(
            {
                messages: [
                    {
                        role: 'assistant',
                        content: [
                            {
                                toolUse: {
                                    toolUseId: 'oversized-type',
                                    name: 'lookup',
                                    input: {},
                                    type: oversizedType,
                                },
                            },
                        ],
                    },
                ],
            },
            'oversized-tool-use-type',
        ).catch((error: unknown) => error);
        expect(oversizedFailure).toBeInstanceOf(Error);
        expect((oversizedFailure as Error).message).not.toContain(oversizedType);
    });
});
