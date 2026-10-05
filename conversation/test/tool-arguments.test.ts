import { describe, expect, it, vi } from 'vitest';
import {
    type Asset,
    type ConversationDocument,
    ConversationValidationError,
    externalizeToolCallArguments,
    fingerprintJson,
    hashUtf8Text,
    hydrateSelectedToolCallArguments,
    hydrateToolCallArguments,
    type JsonObject,
    type NativeReplayBlock,
    parseConversationDocument,
    prepareToolArgumentExternalization,
    toolArgumentsForModel,
    validateConversationDocument,
} from '../src/index.js';
import { emptyDocument, RECORDED_AT } from './fixtures.js';

const CALL_ID = 'call-write';
const BLOCK_ID = 'block-write';

function toolDocument(value: JsonObject = { name: 'file.txt', content: 'exact content' }) {
    const document = emptyDocument();
    document.turns.push({
        id: 'agent-turn',
        kind: 'agent',
        authority: 'ordinary',
        blocks: [
            {
                id: BLOCK_ID,
                type: 'tool_call',
                call_id: CALL_ID,
                tool_name: 'write_artifact',
                executor: 'application',
                arguments: { type: 'json', value },
            },
        ],
        status: 'completed',
        timestamps: { recorded_at: RECORDED_AT },
        provenance: { type: 'imported', source: 'test' },
        model_visibility: 'include',
    });
    document.context.entries.push({ id: 'agent-context', type: 'source_turn', turn_id: 'agent-turn' });
    return parseConversationDocument(document);
}

function externalAsset(contentHash: string, byteLength: number, id = 'asset-write'): Asset {
    return {
        id,
        kind: 'text',
        mime_type: 'text/plain',
        storage: {
            type: 'external',
            resolver: 'test.artifact',
            locator: { storage_id: 'run-1', artifact_path: `tool-inputs/${id}.txt` },
        },
        provenance: { type: 'imported', source: 'test' },
        byte_length: byteLength,
        content_hash: contentHash,
        created_at: RECORDED_AT,
    };
}

async function externalizedDocument(
    document: ConversationDocument,
    inputPath: (string | number)[],
    modelValue: JsonObject,
) {
    const prepared = await prepareToolArgumentExternalization(document, CALL_ID, inputPath);
    const asset = externalAsset(prepared.content_hash, prepared.byte_length);
    const result = await externalizeToolCallArguments(document, {
        operation_id: 'externalize-write',
        expected_revision: document.revision,
        recorded_at: RECORDED_AT,
        call_id: CALL_ID,
        input_path: inputPath,
        model_value: modelValue,
        exact_arguments_hash: prepared.exact_arguments_hash,
        asset,
    });
    return { ...result, prepared, asset };
}

async function* byteChunks(content: string): AsyncIterable<Uint8Array> {
    const bytes = new TextEncoder().encode(content);
    const split = Math.floor(bytes.byteLength / 2);
    yield bytes.slice(0, split);
    yield bytes.slice(split);
}

describe('canonical tool argument hydration', () => {
    it('externalizes only after a durable asset exists and hydrates exact executable arguments', async () => {
        const source = toolDocument();
        const { document, prepared, asset } = await externalizedDocument(source, ['content'], {
            name: 'file.txt',
            content: '[stored content accepted for execution]',
        });
        const call = document.turns[0].blocks[0];
        if (call.type !== 'tool_call') throw new Error('Expected tool call');

        expect(document.revision).toBe(1);
        expect(document.operation_receipts['externalize-write']).toMatchObject({
            base_revision: 0,
            result_revision: 1,
            accepted_asset_ids: [asset.id],
        });
        expect(call.arguments).toMatchObject({
            type: 'externalized_json',
            value: { name: 'file.txt' },
            model_value: { name: 'file.txt', content: '[stored content accepted for execution]' },
            exact_arguments_hash: prepared.exact_arguments_hash,
            hydration: [{ input_path: ['content'], asset_id: asset.id, content_hash: asset.content_hash }],
        });
        expect(toolArgumentsForModel(call.arguments)).toEqual({
            name: 'file.txt',
            content: '[stored content accepted for execution]',
        });
        await expect(
            hydrateToolCallArguments(document, CALL_ID, async () => byteChunks(prepared.content)),
        ).resolves.toEqual({ name: 'file.txt', content: 'exact content' });
    });

    it('enforces max_bytes for inline JSON arguments before returning them', async () => {
        const document = toolDocument({ content: 'x'.repeat(256) });
        const resolver = vi.fn();

        await expect(hydrateToolCallArguments(document, CALL_ID, resolver, { max_bytes: 32 })).rejects.toThrow(
            'inline arguments exceed max_bytes',
        );
        expect(resolver).not.toHaveBeenCalled();
    });

    it('restores a numeric array path without interpreting it as an object key', async () => {
        const source = toolDocument({ name: 'file.txt', parts: ['header', 'exact array content'] });
        const { document, prepared } = await externalizedDocument(source, ['parts', 1], {
            name: 'file.txt',
            parts: ['header', '[stored]'],
        });
        await expect(
            hydrateToolCallArguments(document, CALL_ID, async () => byteChunks(prepared.content)),
        ).resolves.toEqual({ name: 'file.txt', parts: ['header', 'exact array content'] });
    });

    it('recovers an exact retry without another revision', async () => {
        const source = toolDocument();
        const first = await externalizedDocument(source, ['content'], { name: 'file.txt', content: '[stored]' });
        const retry = await externalizeToolCallArguments(first.document, {
            operation_id: 'externalize-write',
            expected_revision: source.revision,
            recorded_at: RECORDED_AT,
            call_id: CALL_ID,
            input_path: ['content'],
            model_value: { name: 'file.txt', content: '[stored]' },
            exact_arguments_hash: first.prepared.exact_arguments_hash,
            asset: first.asset,
        });
        expect(retry.applied).toBe(false);
        expect(retry.document.revision).toBe(1);
    });

    it('treats inherited record names as absent and preserves supported ordinary property names', async () => {
        const source = toolDocument({ constructor: 'keep', toString: 'also keep', content: 'exact' });
        const prepared = await prepareToolArgumentExternalization(source, CALL_ID, ['content']);
        const asset = externalAsset(prepared.content_hash, prepared.byte_length, 'constructor');
        const result = await externalizeToolCallArguments(source, {
            operation_id: 'toString',
            expected_revision: source.revision,
            recorded_at: RECORDED_AT,
            call_id: CALL_ID,
            input_path: ['content'],
            model_value: { constructor: 'keep', toString: 'also keep', content: '[stored]' },
            exact_arguments_hash: prepared.exact_arguments_hash,
            asset,
        });
        expect(Object.hasOwn(result.document.assets, 'constructor')).toBe(true);
        expect(Object.hasOwn(result.document.operation_receipts, 'toString')).toBe(true);
        await expect(
            hydrateToolCallArguments(result.document, CALL_ID, async () => byteChunks('exact')),
        ).resolves.toEqual({ constructor: 'keep', toString: 'also keep', content: 'exact' });

        await expect(
            externalizeToolCallArguments(source, {
                operation_id: '__proto__',
                expected_revision: source.revision,
                recorded_at: RECORDED_AT,
                call_id: CALL_ID,
                input_path: ['content'],
                model_value: { content: '[stored]' },
                exact_arguments_hash: prepared.exact_arguments_hash,
                asset,
            }),
        ).rejects.toBeInstanceOf(ConversationValidationError);
    });

    it('rejects missing, unsafe, and wrong-kind paths', async () => {
        const source = toolDocument({ nested: { content: 'exact' } });
        await expect(prepareToolArgumentExternalization(source, CALL_ID, ['missing'])).rejects.toThrow(
            'is not a string',
        );
        await expect(prepareToolArgumentExternalization(source, CALL_ID, [0, 'content'])).rejects.toThrow(
            'is not a string',
        );
        await expect(prepareToolArgumentExternalization(source, CALL_ID, ['__proto__'])).rejects.toBeInstanceOf(
            ConversationValidationError,
        );
    });

    it('rejects missing assets and overlapping hydration paths during semantic validation', async () => {
        const source = toolDocument({ nested: { content: 'exact' } });
        const { document } = await externalizedDocument(source, ['nested', 'content'], {
            nested: { content: '[stored]' },
        });
        const missing = structuredClone(document);
        delete missing.assets['asset-write'];
        expect(validateConversationDocument(missing)).toMatchObject({
            success: false,
            diagnostics: expect.arrayContaining([expect.objectContaining({ code: 'REFERENCE_NOT_FOUND' })]),
        });

        const overlapping = structuredClone(document);
        const call = overlapping.turns[0].blocks[0];
        if (call.type !== 'tool_call' || call.arguments.type !== 'externalized_json') {
            throw new Error('Expected externalized tool call');
        }
        call.arguments.hydration.push({
            ...call.arguments.hydration[0],
            input_path: ['nested'],
        });
        expect(validateConversationDocument(overlapping)).toMatchObject({
            success: false,
            diagnostics: expect.arrayContaining([expect.objectContaining({ code: 'TOOL_ARGUMENT_HYDRATION_INVALID' })]),
        });
    });

    it('enforces the byte limit before resolving and verifies resolved UTF-8 bytes', async () => {
        const source = toolDocument();
        const { document, prepared } = await externalizedDocument(source, ['content'], {
            name: 'file.txt',
            content: '[stored]',
        });
        const resolver = vi.fn(async () => byteChunks(prepared.content));
        await expect(hydrateToolCallArguments(document, CALL_ID, resolver, { max_bytes: 1 })).rejects.toThrow(
            'exceeds max_bytes',
        );
        expect(resolver).not.toHaveBeenCalled();

        await expect(
            hydrateToolCallArguments(document, CALL_ID, async () => byteChunks('different content')),
        ).rejects.toThrow('byte length does not match');
    });

    it('copies resolver chunks before the producer can reuse its buffer', async () => {
        const source = toolDocument({ content: 'abcd' });
        const { document } = await externalizedDocument(source, ['content'], { content: '[stored]' });
        async function* reusedBuffer(): AsyncIterable<Uint8Array> {
            const shared = new Uint8Array(2);
            shared.set(new TextEncoder().encode('ab'));
            yield shared;
            shared.set(new TextEncoder().encode('cd'));
            yield shared;
        }
        await expect(hydrateToolCallArguments(document, CALL_ID, async () => reusedBuffer())).resolves.toEqual({
            content: 'abcd',
        });
    });

    it('bounds empty chunk streams and the total bytes across multiple assets', async () => {
        const source = toolDocument({ content: 'exact' });
        const { document } = await externalizedDocument(source, ['content'], { content: '[stored]' });
        async function* emptyChunks(): AsyncIterable<Uint8Array> {
            for (let index = 0; index < 4_097; index += 1) yield new Uint8Array();
        }
        await expect(hydrateToolCallArguments(document, CALL_ID, async () => emptyChunks())).rejects.toThrow(
            'exceeds the hydration chunk limit',
        );

        const first = 'a'.repeat(400);
        const second = 'b'.repeat(400);
        const firstHash = await hashUtf8Text(first);
        const secondHash = await hashUtf8Text(second);
        const multi = toolDocument();
        multi.assets.first = externalAsset(firstHash.content_hash, firstHash.byte_length, 'first');
        multi.assets.second = externalAsset(secondHash.content_hash, secondHash.byte_length, 'second');
        const call = multi.turns[0].blocks[0];
        if (call.type !== 'tool_call') throw new Error('Expected tool call');
        call.arguments = {
            type: 'externalized_json',
            value: {},
            model_value: { first: '[stored]', second: '[stored]' },
            exact_arguments_hash: await fingerprintJson({ first, second }),
            hydration: [
                { type: 'text_asset', input_path: ['first'], asset_id: 'first', content_hash: firstHash.content_hash },
                {
                    type: 'text_asset',
                    input_path: ['second'],
                    asset_id: 'second',
                    content_hash: secondHash.content_hash,
                },
            ],
        };
        const valid = parseConversationDocument(multi);
        await expect(
            hydrateToolCallArguments(
                valid,
                CALL_ID,
                async (asset) => byteChunks(asset.id === 'first' ? first : second),
                { max_bytes: 700 },
            ),
        ).rejects.toThrow('exceeds max_bytes before resolution');
    });

    it('rejects externalization when a retained replacement replay binds the call', async () => {
        const source = toolDocument();
        source.compactions.compaction = {
            id: 'compaction',
            operation_id: 'compaction-operation',
            strategy: { id: 'summarize', version: '1', configuration_fingerprint: 'sha256:config' },
            source: {
                turn_ids: ['agent-turn'],
                block_ids: [BLOCK_ID],
                source_fingerprint: 'sha256:source',
            },
            replacement_turns: [
                {
                    id: 'replacement-turn',
                    kind: 'agent',
                    authority: 'ordinary',
                    blocks: [
                        {
                            id: 'replacement-replay',
                            type: 'native_replay',
                            adapter: 'test-adapter',
                            protocol: 'test-protocol',
                            compatibility_scope: {
                                provider: 'test-provider',
                                protocol: 'test-protocol',
                                adapter_version: 'test-adapter',
                            },
                            payload: { evidence: true },
                            dependencies: {
                                turn_ids: ['agent-turn'],
                                block_ids: [],
                                call_ids: [],
                                request_ids: [],
                            },
                        },
                    ],
                    status: 'completed',
                    timestamps: { recorded_at: RECORDED_AT },
                    provenance: {
                        type: 'derived',
                        derivation_id: 'compaction',
                        source_turn_ids: ['agent-turn'],
                        source_block_ids: [BLOCK_ID],
                        source_hash: 'sha256:source',
                    },
                    model_visibility: 'include',
                },
            ],
            fidelity: 'reversible_representation',
            retained_asset_ids: [],
            generation_ids: [],
            created_at: RECORDED_AT,
        };
        source.context.entries = [
            {
                id: 'replacement-context',
                type: 'replacement_turn',
                compaction_id: 'compaction',
                turn_id: 'replacement-turn',
            },
        ];
        const valid = parseConversationDocument(source);
        await expect(prepareToolArgumentExternalization(valid, CALL_ID, ['content'])).rejects.toThrow(
            'protected by native replay replacement-replay',
        );
    });

    it('archives and invalidates only replay explicitly declared discardable', async () => {
        const source = toolDocument();
        const replayBlocks: NativeReplayBlock[] = ['Ω-replay', 'a-replay', 'A-replay'].map((id) => ({
            id,
            type: 'native_replay',
            adapter: 'test-adapter',
            protocol: 'test-protocol',
            compatibility_scope: {
                provider: 'test-provider',
                protocol: 'test-protocol',
                adapter_version: 'test-adapter',
            },
            payload: { id, raw_arguments: '{ "name": "file.txt", "content": "exact content" }' },
            dependencies: {
                turn_ids: [],
                block_ids: [BLOCK_ID],
                call_ids: [CALL_ID],
                request_ids: [],
            },
            dependency_policy: 'discard_on_dependency_change',
        }));
        source.turns[0] = {
            ...source.turns[0],
            blocks: [...source.turns[0].blocks, ...replayBlocks],
        } as (typeof source.turns)[number];
        const valid = parseConversationDocument(source);
        const prepared = await prepareToolArgumentExternalization(valid, CALL_ID, ['content']);
        expect(prepared.replay_archives.map((replay) => replay.replay_block_id)).toEqual([
            'A-replay',
            'a-replay',
            'Ω-replay',
        ]);
        const contentAsset = externalAsset(prepared.content_hash, prepared.byte_length);
        const replayArchives = prepared.replay_archives.map((replay, index) => ({
            replay_block_id: replay.replay_block_id,
            asset: {
                id: `asset-replay-${index}`,
                kind: 'document',
                mime_type: 'application/json',
                storage: {
                    type: 'external',
                    resolver: 'test.artifact',
                    locator: { storage_id: 'run-1', artifact_path: `tool-inputs/${index}.json` },
                },
                provenance: { type: 'imported', source: 'test' },
                byte_length: replay.byte_length,
                content_hash: replay.content_hash,
                created_at: RECORDED_AT,
            } satisfies Asset,
        }));
        const options = {
            operation_id: 'externalize-with-replay',
            expected_revision: valid.revision,
            recorded_at: RECORDED_AT,
            call_id: CALL_ID,
            input_path: ['content'] as [string],
            model_value: { name: 'file.txt', content: '[stored]' },
            exact_arguments_hash: prepared.exact_arguments_hash,
            asset: contentAsset,
            replay_archives: [...replayArchives].reverse(),
        };
        const first = await externalizeToolCallArguments(valid, options);
        const call = first.document.turns[0].blocks.find((block) => block.type === 'tool_call');
        expect(first.document.turns[0].blocks.some((block) => block.type === 'native_replay')).toBe(false);
        expect(call).toMatchObject({
            arguments: {
                type: 'externalized_json',
                invalidated_replay_archives: replayArchives.map((archive) => ({
                    replay_block_id: archive.replay_block_id,
                    asset_id: archive.asset.id,
                    content_hash: archive.asset.content_hash,
                })),
            },
        });
        for (const archive of replayArchives) expect(first.document.assets[archive.asset.id]).toEqual(archive.asset);

        const retry = await externalizeToolCallArguments(
            parseConversationDocument(JSON.parse(JSON.stringify(first.document))),
            options,
        );
        expect(retry.applied).toBe(false);
        expect(retry.document).toEqual(first.document);
    });

    it('uses SHA-256 of exact UTF-8 bytes for asset content binding', async () => {
        await expect(hashUtf8Text('é')).resolves.toMatchObject({ byte_length: 2 });
        const composed = await hashUtf8Text('é');
        const decomposed = await hashUtf8Text('e\u0301');
        expect(composed.content_hash).not.toBe(decomposed.content_hash);
    });

    it('rejects lossy UTF-8 externalization before changing the document', async () => {
        const source = toolDocument({ content: '\ud800' });
        const before = structuredClone(source);

        await expect(prepareToolArgumentExternalization(source, CALL_ID, ['content'])).rejects.toThrow(
            /unpaired surrogate/,
        );
        await expect(
            externalizeToolCallArguments(source, {
                operation_id: 'externalize-invalid-utf8',
                expected_revision: source.revision,
                recorded_at: RECORDED_AT,
                call_id: CALL_ID,
                input_path: ['content'],
                model_value: { content: '[stored]' },
                exact_arguments_hash: await fingerprintJson({ content: '\ud800' }),
                asset: externalAsset(`sha256:${'0'.repeat(64)}`, 3),
            }),
        ).rejects.toThrow(/unpaired surrogate/);
        expect(source).toEqual(before);
    });
});

it('hydrates an owned selected call without a synthetic document and rejects changed asset identity/bytes', async () => {
    const { document, prepared, asset } = await externalizedDocument(toolDocument(), ['content'], {
        name: 'file.txt',
        content: '[selected exact input]',
    });
    const call = document.turns[0]?.blocks[0];
    if (call?.type !== 'tool_call') throw new Error('Expected real externalized call');
    const assets = { [asset.id]: asset };
    await expect(
        hydrateSelectedToolCallArguments(call, assets, async () => byteChunks(prepared.content)),
    ).resolves.toEqual({ name: 'file.txt', content: 'exact content' });
    await expect(hydrateSelectedToolCallArguments(call, assets, async () => byteChunks('changed'))).rejects.toThrow();
    const wrong = { [asset.id]: { ...asset, id: 'foreign:asset' } };
    await expect(
        hydrateSelectedToolCallArguments(call, wrong, async () => byteChunks(prepared.content)),
    ).rejects.toThrow('key differs');
    let release: () => void = () => undefined;
    const gate = new Promise<void>((resolve) => {
        release = resolve;
    });
    const mutable = { [asset.id]: structuredClone(asset) };
    const pending = hydrateSelectedToolCallArguments(call, mutable, async () => {
        await gate;
        return byteChunks(prepared.content);
    });
    mutable[asset.id].content_hash = 'changed-after-read-start';
    release();
    await expect(pending).resolves.toEqual({ name: 'file.txt', content: 'exact content' });
});

it('preserves an explicit selected argument budget above the default source bound', async () => {
    const argumentLimit = 32 * 1024 * 1024 + 1;
    const value = {
        text: 'x'.repeat(argumentLimit - new TextEncoder().encode(JSON.stringify({ text: '' })).byteLength),
    };
    const call = {
        id: 'large:selected:call',
        type: 'tool_call' as const,
        call_id: 'large:selected:call-id',
        tool_name: 'write_artifact',
        executor: 'application' as const,
        arguments: { type: 'json' as const, value },
    };
    const resolver = vi.fn(async () => byteChunks('never resolved'));
    const hydrated = await hydrateSelectedToolCallArguments(call, {}, resolver, { max_bytes: argumentLimit });
    expect(hydrated.text).toBe(value.text);
    await expect(
        hydrateSelectedToolCallArguments(call, {}, resolver, { max_bytes: argumentLimit - 1 }),
    ).rejects.toThrow('inline arguments exceed max_bytes');
    await expect(hydrateSelectedToolCallArguments(call, {}, resolver)).rejects.toThrow(
        'inline arguments exceed max_bytes',
    );
    expect(resolver).not.toHaveBeenCalled();
});
