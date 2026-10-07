import { Ajv2020 } from 'ajv/dist/2020.js';
import formats from 'ajv-formats';
import { describe, expect, expectTypeOf, it } from 'vitest';
import type { z } from 'zod';
import {
    type AgentContentBlock,
    type Asset,
    type BlockSubselection,
    type ConversationDocument,
    type ConversationSelection,
    type ConversationSelectionRequest,
    ConversationSelectionRequestSchema,
    ConversationSelectionResultSchema,
    ConversationSelectionSchema,
    type ConversationSelector,
    ConversationSelectorSchema,
    ConversationSliceSchema,
    fingerprintAssetSelectionMetadata,
    fingerprintJson,
    parseConversationDocument,
    resolveContextSelection,
    resolveConversationSelection,
    sliceConversation,
} from '../src/index.js';
import * as JsonSchemas from '../src/json-schema.js';
import {
    CONVERSATION_JSON_SCHEMAS,
    ConversationSelectionRequestJsonSchema,
    ConversationSelectionResultJsonSchema,
    ConversationSelectorJsonSchema,
} from '../src/json-schema.js';
import { emptyDocument, RECORDED_AT, textBlock, toolCallBlock, userTurn } from './fixtures.js';

function source(
    blocks: AgentContentBlock[] = [
        textBlock('text', 'A😀e\u0301Z'),
        { id: 'json', type: 'json', value: { 'a/b': { '~': [null, 3] }, '': true, z: 1 } },
    ],
): ConversationDocument {
    const doc = emptyDocument();
    doc.turns.push({ ...userTurn('turn'), kind: 'agent', provenance: { type: 'received' }, blocks });
    doc.context.entries = [{ id: 'entry', type: 'source_turn', turn_id: 'turn' }];
    return parseConversationDocument(doc);
}
function query(doc: ConversationDocument, subselections?: BlockSubselection[]): ConversationSelectionRequest {
    return {
        conversation: { conversation_id: doc.id, revision: doc.revision },
        expected_context_revision: doc.context.revision,
        selector: { source: { kind: 'all' }, ...(subselections ? { subselections } : {}) },
    };
}
async function text(doc: ConversationDocument, start = 1, end = 2): Promise<BlockSubselection> {
    return {
        kind: 'text_range',
        entry_id: 'entry',
        block_id: 'text',
        expected_block_fingerprint: await fingerprintJson(doc.turns[0].blocks[0]),
        range: { start_code_point: start, end_code_point: end },
    };
}
async function pointer(doc: ConversationDocument, value: string): Promise<BlockSubselection> {
    return {
        kind: 'json_pointer',
        entry_id: 'entry',
        block_id: 'json',
        expected_block_fingerprint: await fingerprintJson(doc.turns[0].blocks.find((b) => b.id === 'json')),
        pointer: value,
    };
}
async function selected(doc: ConversationDocument, request = query(doc)): Promise<ConversationSelection> {
    const result = await resolveConversationSelection(doc, request);
    if (result.kind !== 'selected') throw new Error(JSON.stringify(result));
    return result.selection;
}
function asset(id: string, kind: Asset['kind']): Asset {
    return {
        id,
        kind,
        mime_type: `${kind}/example`,
        storage: { type: 'external', resolver: 'store', locator: { path: 'immutable' } },
        provenance: { type: 'received' },
        created_at: RECORDED_AT,
    };
}
async function media(
    doc: ConversationDocument,
    range: Extract<BlockSubselection, { kind: 'media_range' }>['range'],
): Promise<BlockSubselection> {
    const block = doc.turns[0].blocks[0];
    if (!('asset_id' in block)) throw new Error('Media fixture required');
    return {
        kind: 'media_range',
        entry_id: 'entry',
        block_id: block.id,
        expected_block_fingerprint: await fingerprintJson(block),
        expected_asset_metadata_fingerprint: await fingerprintAssetSelectionMetadata(doc.assets[block.asset_id]),
        range,
    };
}
function mediaSource(type: 'image' | 'document' | 'audio' | 'video') {
    const doc = source([]);
    doc.assets.asset = asset('asset', type);
    doc.turns[0].blocks = [{ id: 'media', type, asset_id: 'asset' }];
    return parseConversationDocument(doc);
}

describe('read-only materialized selection', () => {
    it('inspects protected code and pending/replay dependencies without changing edit eligibility or source', async () => {
        const doc = source([
            { ...textBlock('text', 'code()'), format: 'code' },
            toolCallBlock('call-block', 'call-id'),
            {
                id: 'replay',
                type: 'native_replay',
                adapter: 'adapter',
                protocol: 'protocol',
                compatibility_scope: { provider: 'provider', protocol: 'protocol', adapter_version: '1' },
                payload: { opaque: true },
                dependencies: { turn_ids: [], block_ids: ['text'], call_ids: [], request_ids: [] },
            },
        ]);
        doc.context.protected_entry_ids = ['entry'];
        const baseline = JSON.stringify(doc);
        const input = query(doc, [await text(doc)]);
        const result = await selected(doc, input);
        expect(result.entries[0].blocks.map((block) => block.kind)).toEqual(['text_range', 'whole', 'whole']);
        expect(JSON.stringify(doc)).toBe(baseline);
        expect(
            (
                await resolveContextSelection(doc, {
                    ...query(doc),
                    selector: { source: { kind: 'all' }, filters: { block_ids: ['text'] } },
                })
            ).kind,
        ).toBe('rejected');
        const opaque = {
            ...(await pointer(source(), '')),
            block_id: 'replay',
            expected_block_fingerprint: await fingerprintJson(doc.turns[0].blocks[2]),
        };
        expect((await resolveConversationSelection(doc, query(doc, [opaque]))).kind).toBe('rejected');
    });
    it('refines selected blocks only, leaves unspecified matched blocks whole, and returns a pinned ref view', async () => {
        const doc = source(),
            input = query(doc, [await text(doc)]);
        input.selector.filters = { block_ids: ['json'] };
        expect((await resolveConversationSelection(doc, input)).kind).toBe('rejected');
        delete input.selector.filters;
        const result = await sliceConversation(doc, input);
        expect(result).toMatchObject({
            kind: 'selected',
            view: {
                kind: 'context_view',
                selection: {
                    access: 'read_only',
                    entries: [{ entry: doc.context.entries[0], blocks: [{ kind: 'text_range' }, { kind: 'whole' }] }],
                },
            },
        });
        expect(JSON.stringify(result)).not.toContain('A😀');
        expect(ConversationSelectionResultSchema.parse(await resolveConversationSelection(doc, input))).toEqual(
            JSON.parse(JSON.stringify(await resolveConversationSelection(doc, input))),
        );
    });
    it('sorts disjoint Unicode code-point ranges and rejects empty, reversed, overlapping, stale and out-of-bounds ranges', async () => {
        const doc = source();
        const input = query(doc, [await text(doc, 3, 4), await text(doc, 1, 2)]);
        expect((await selected(doc, input)).entries[0].blocks.slice(0, 2)).toMatchObject([
            { range: { start_code_point: 1, end_code_point: 2 } },
            { range: { start_code_point: 3, end_code_point: 4 } },
        ]);
        expect(
            (await selected(doc, query(doc, [await text(doc, 1, 2), await text(doc, 2, 3)]))).entries[0].blocks,
        ).toHaveLength(3);
        for (const [start, end] of [
            [1, 1],
            [2, 1],
            [0, 6],
        ])
            expect((await resolveConversationSelection(doc, query(doc, [await text(doc, start, end)]))).kind).toBe(
                'rejected',
            );
        expect(
            (await resolveConversationSelection(doc, query(doc, [await text(doc, 0, 3), await text(doc, 2, 4)]))).kind,
        ).toBe('rejected');
        expect(
            (
                await resolveConversationSelection(
                    doc,
                    query(doc, [{ ...(await text(doc)), expected_block_fingerprint: 'sha256:stale' }]),
                )
            ).kind,
        ).toBe('rejected');
    });
    it('resolves RFC6901 root/escapes/empty keys and canonical array indices with own-property ordering and overlap checks', async () => {
        const doc = source();
        const paths = ['/z', '/a~1b/~0/1', '/'];
        const result = await selected(doc, query(doc, await Promise.all(paths.map((path) => pointer(doc, path)))));
        expect(
            result.entries[0].blocks.filter((block) => block.kind === 'json_pointer').map((block) => block.pointer),
        ).toEqual(['/', '/a~1b/~0/1', '/z']);
        expect((await selected(doc, query(doc, [await pointer(doc, '')]))).entries[0].blocks[1]).toMatchObject({
            pointer: '',
        });
        for (const path of ['/a~1b/~0/01', '/a~1b/~0/-', '/a~1b/~0/2', '/missing', '/constructor', '/~2'])
            expect((await resolveConversationSelection(doc, query(doc, [await pointer(doc, path)]))).kind).toBe(
                'rejected',
            );
        for (const paths of [
            ['', '/z'],
            ['/a~1b', '/a~1b/~0/1'],
            ['/z', '/z'],
        ])
            expect(
                (
                    await resolveConversationSelection(
                        doc,
                        query(doc, await Promise.all(paths.map((p) => pointer(doc, p)))),
                    )
                ).kind,
            ).toBe('rejected');
    });
    it('preserves split-entry order and legal dictionary-like IDs without treating them as map authority', async () => {
        const doc = source();
        doc.context.entries = [
            { id: '__proto__', type: 'source_turn', turn_id: 'turn', block_ids: ['text'] },
            { id: 'constructor', type: 'source_turn', turn_id: 'turn', block_ids: ['json'] },
        ];
        const refinement = { ...(await text(doc)), entry_id: '__proto__' };
        const result = await selected(doc, query(doc, [refinement]));
        expect(result.entries.map((entry) => entry.entry.id)).toEqual(['__proto__', 'constructor']);
        expect(result.entries[0].blocks[0].kind).toBe('text_range');
        const ambiguous = query(doc);
        ambiguous.selector.source = {
            kind: 'range',
            range: { from: { kind: 'turn', id: 'turn' }, through: { kind: 'entry', id: 'constructor' } },
        };
        expect((await resolveConversationSelection(doc, ambiguous)).kind).toBe('rejected');
        ambiguous.selector.source = {
            kind: 'range',
            range: { from: { kind: 'entry', id: '__proto__' }, through: { kind: 'entry', id: 'constructor' } },
        };
        expect((await resolveConversationSelection(doc, ambiguous)).kind).toBe('selected');
    });
    it('owns caller inputs before hashing and binds even unselected source changes to the fingerprint', async () => {
        const doc = source(),
            baseline = structuredClone(doc),
            input = query(doc, [await text(doc)]),
            saved = structuredClone(input);
        const pending = resolveConversationSelection(doc, input);
        doc.turns[0].blocks[0] = textBlock('text', 'changed');
        input.selector.subselections = [];
        expect(await pending).toEqual(await resolveConversationSelection(baseline, saved));
        const changed = structuredClone(baseline);
        changed.turns[0].metadata = { other: true };
        expect((await selected(changed, saved)).source_fingerprint).not.toBe(
            (await selected(baseline, saved)).source_fingerprint,
        );
        const orderA = query(baseline, [await text(baseline, 0, 1), await text(baseline, 2, 3)]);
        expect(await resolveConversationSelection(baseline, orderA)).toEqual(
            await resolveConversationSelection(baseline, {
                ...orderA,
                selector: { ...orderA.selector, subselections: orderA.selector.subselections?.toReversed() },
            }),
        );
    });
    it('distinguishes no matches, bad explicit IDs/pins and prototype/accessor input; bounds refinement work', async () => {
        const doc = source(),
            input = query(doc);
        expect(
            (
                await resolveConversationSelection(doc, {
                    ...input,
                    selector: { ...input.selector, filters: { actor_ids: ['absent'] } },
                })
            ).kind,
        ).toBe('no_match');
        for (const invalid of [
            { ...input, expected_context_revision: 1 },
            { ...input, selector: { source: { kind: 'turn_ids', turn_ids: ['missing'] } } },
        ])
            expect((await resolveConversationSelection(doc, invalid)).kind).toBe('rejected');
        const prototype = Object.create(input);
        expect((await resolveConversationSelection(doc, prototype)).kind).toBe('rejected');
        let calls = 0;
        Object.defineProperty(input.selector, 'filters', {
            enumerable: true,
            get: () => {
                calls++;
                return {};
            },
        });
        expect((await resolveConversationSelection(doc, input)).kind).toBe('rejected');
        expect(calls).toBe(0);
        const refinement = await text(doc);
        expect(
            (
                await resolveConversationSelection(
                    doc,
                    query(
                        doc,
                        Array.from({ length: 257 }, () => refinement),
                    ),
                )
            ).kind,
        ).toBe('rejected');
        expect(
            ConversationSelectorSchema.safeParse({
                source: { kind: 'all' },
                subselections: Array.from({ length: 4097 }, () => refinement),
            }).success,
        ).toBe(false);
    });
});

describe('typed media inspection evidence', () => {
    it('inspects unknown extents with unverified content, never treating a metadata hash as verified binary', async () => {
        const doc = mediaSource('image');
        const range = {
            type: 'image_region' as const,
            coordinate_space: 'pixels' as const,
            x: 10,
            y: 20,
            width: 5,
            height: 5,
        };
        const result = await selected(doc, query(doc, [await media(doc, range)]));
        expect(result.entries[0].blocks[0]).toMatchObject({
            kind: 'media_range',
            evidence: {
                extent: 'unknown',
                content: { status: 'unverified', reason: 'external_content' },
                mutation_ready: false,
            },
        });
        const mismatch = await media(doc, range);
        if (mismatch.kind !== 'media_range') throw new Error('fixture');
        mismatch.expected_asset_metadata_fingerprint = 'sha256:other';
        expect((await resolveConversationSelection(doc, query(doc, [mismatch]))).kind).toBe('rejected');
    });
    it('verifies intrinsic inline bytes separately; reports declared mismatches and unencodable text', async () => {
        const doc = mediaSource('image');
        doc.assets.asset.storage = { type: 'inline_base64', data: 'YWJj' };
        const range = {
            type: 'image_region' as const,
            coordinate_space: 'normalized' as const,
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        };
        const metadata = await fingerprintAssetSelectionMetadata(doc.assets.asset);
        expect((await selected(doc, query(doc, [await media(doc, range)]))).entries[0].blocks[0]).toMatchObject({
            evidence: {
                content: {
                    status: 'verified_inline',
                    byte_length: 3,
                    content_hash: 'sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
                },
            },
        });
        doc.assets.asset.storage = { type: 'inline_base64', data: 'ZGVm' };
        expect(await fingerprintAssetSelectionMetadata(doc.assets.asset)).toBe(metadata);
        doc.assets.asset.content_hash = 'sha256:incorrect';
        expect((await selected(doc, query(doc, [await media(doc, range)]))).entries[0].blocks[0]).toMatchObject({
            evidence: { content: { status: 'mismatch' }, mutation_ready: false },
        });
        delete doc.assets.asset.content_hash;
        doc.assets.asset.storage = { type: 'inline_text', text: '\ud800' };
        expect((await selected(doc, query(doc, [await media(doc, range)]))).entries[0].blocks[0]).toMatchObject({
            evidence: { content: { status: 'unverified', reason: 'unencodable_inline_text' } },
        });
    });
    it('checks retained dimensions, existing block selections, disjoint rectangles and mixed coordinates', async () => {
        const doc = mediaSource('image');
        doc.assets.asset.media = { width: 100, height: 100 };
        const first = {
            type: 'image_region' as const,
            coordinate_space: 'pixels' as const,
            x: 0,
            y: 0,
            width: 10,
            height: 10,
        };
        const adjacent = { ...first, x: 10 };
        expect(
            (await selected(doc, query(doc, [await media(doc, adjacent), await media(doc, first)]))).entries[0].blocks,
        ).toMatchObject([{ range: first }, { range: adjacent }]);
        for (const range of [
            { ...first, x: 95 },
            { ...first, coordinate_space: 'normalized' as const, width: 2 },
        ])
            expect((await resolveConversationSelection(doc, query(doc, [await media(doc, range)]))).kind).toBe(
                'rejected',
            );
        expect(
            (
                await resolveConversationSelection(
                    doc,
                    query(doc, [await media(doc, first), await media(doc, { ...first, x: 5 })]),
                )
            ).kind,
        ).toBe('rejected');
        const normalized = { ...first, coordinate_space: 'normalized' as const, x: 0.1, y: 0, width: 0.1, height: 0.1 };
        expect(
            (
                await resolveConversationSelection(
                    doc,
                    query(doc, [await media(doc, normalized), await media(doc, first)]),
                )
            ).kind,
        ).toBe('selected');
        delete doc.assets.asset.media;
        expect(
            (
                await resolveConversationSelection(
                    doc,
                    query(doc, [await media(doc, normalized), await media(doc, first)]),
                )
            ).kind,
        ).toBe('rejected');
        doc.assets.asset.media = { width: 100, height: 100 };
        doc.turns[0].blocks = [{ id: 'media', type: 'image', asset_id: 'asset', selection: adjacent }];
        expect((await resolveConversationSelection(doc, query(doc, [await media(doc, first)]))).kind).toBe('rejected');
    });
    it('owns inline content and metadata before any async digest and rejects finite-geometry overflow', async () => {
        const doc = mediaSource('image');
        doc.assets.asset.storage = { type: 'inline_base64', data: 'YWJj' };
        const range = {
            type: 'image_region' as const,
            coordinate_space: 'pixels' as const,
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        };
        const input = query(doc, [await media(doc, range)]),
            saved = structuredClone(input),
            baseline = structuredClone(doc);
        const pending = resolveConversationSelection(doc, input);
        doc.assets.asset.storage = { type: 'inline_base64', data: 'ZGVm' };
        doc.assets.asset.metadata = { changed: true };
        expect(await pending).toEqual(await resolveConversationSelection(baseline, saved));
        expect(
            (
                await resolveConversationSelection(
                    baseline,
                    query(baseline, [
                        await media(baseline, { ...range, x: Number.MAX_VALUE, width: Number.MAX_VALUE }),
                    ]),
                )
            ).kind,
        ).toBe('rejected');
    });
    it('orders pages and half-open time ranges while respecting actual source ranges and unknown duration', async () => {
        const pages = mediaSource('document');
        pages.assets.asset.media = { page_count: 6 };
        const page = (from_page: number, through_page: number) => ({
            type: 'page_range' as const,
            from_page,
            through_page,
        });
        expect(
            (await selected(pages, query(pages, [await media(pages, page(4, 5)), await media(pages, page(1, 2))])))
                .entries[0].blocks,
        ).toMatchObject([{ range: page(1, 2) }, { range: page(4, 5) }]);
        for (const ranges of [[page(3, 2)], [page(1, 7)], [page(1, 2), page(2, 3)]])
            expect(
                (
                    await resolveConversationSelection(
                        pages,
                        query(pages, await Promise.all(ranges.map((r) => media(pages, r)))),
                    )
                ).kind,
            ).toBe('rejected');
        const audio = mediaSource('audio'),
            time = (start_seconds: number, end_seconds: number) => ({
                type: 'time_range' as const,
                start_seconds,
                end_seconds,
            });
        expect(
            (await selected(audio, query(audio, [await media(audio, time(1, 2)), await media(audio, time(0, 1))])))
                .entries[0].blocks,
        ).toMatchObject([{ range: time(0, 1), evidence: { extent: 'unknown' } }, { range: time(1, 2) }]);
        audio.assets.asset.media = { duration_seconds: 2 };
        expect((await resolveConversationSelection(audio, query(audio, [await media(audio, time(1, 3))]))).kind).toBe(
            'rejected',
        );
        audio.turns[0].blocks = [{ id: 'media', type: 'audio', asset_id: 'asset', selection: time(1, 2) }];
        expect((await resolveConversationSelection(audio, query(audio, [await media(audio, time(0, 1))]))).kind).toBe(
            'rejected',
        );
    });
});

it('publishes inferred contracts and strict schema-derived JSON with catalog identity', async () => {
    expectTypeOf<ConversationSelector>().toEqualTypeOf<z.infer<typeof ConversationSelectorSchema>>();
    expectTypeOf<ConversationSelectionRequest>().toEqualTypeOf<z.infer<typeof ConversationSelectionRequestSchema>>();
    expectTypeOf<ConversationSelection>().toEqualTypeOf<z.infer<typeof ConversationSelectionSchema>>();
    const ajv = new Ajv2020({ strict: false });
    formats.default(ajv);
    const doc = source(),
        request = query(doc, [await text(doc)]),
        result = await resolveConversationSelection(doc, request);
    for (const [schema, json, valid, invalids] of [
        [
            ConversationSelectorSchema,
            ConversationSelectorJsonSchema,
            request.selector,
            [
                { ...request.selector, extra: true },
                {
                    ...request.selector,
                    subselections: [{ ...(await text(doc)), range: { start_code_point: -1, end_code_point: 2 } }],
                },
            ],
        ],
        [
            ConversationSelectionRequestSchema,
            ConversationSelectionRequestJsonSchema,
            request,
            [
                { ...request, expected_context_revision: -1 },
                { ...request, extra: true },
            ],
        ],
        [
            ConversationSelectionResultSchema,
            ConversationSelectionResultJsonSchema,
            result,
            [
                { kind: 'rejected', diagnostics: [] },
                {
                    kind: 'no_match',
                    conversation: request.conversation,
                    context_revision: 0,
                    source_fingerprint: 'sha256:test',
                    diagnostics: [],
                    extra: true,
                },
            ],
        ],
    ] as const) {
        const validate = ajv.compile(json);
        expect(validate(valid)).toBe(true);
        expect(schema.safeParse(valid).success).toBe(true);
        for (const invalid of invalids) {
            expect(validate(invalid)).toBe(false);
            expect(schema.safeParse(invalid).success).toBe(false);
        }
    }
    expect(CONVERSATION_JSON_SCHEMAS.conversation_selector).toBe(ConversationSelectorJsonSchema);
    expect(CONVERSATION_JSON_SCHEMAS.conversation_selection_request).toBe(ConversationSelectionRequestJsonSchema);
    expect(CONVERSATION_JSON_SCHEMAS.conversation_selection_result).toBe(ConversationSelectionResultJsonSchema);
    expect(ConversationSliceSchema.parse({ kind: 'context_view', selection: await selected(doc, request) })).toEqual({
        kind: 'context_view',
        selection: await selected(doc, request),
    });
});

it('keeps every new public JSON export discoverable in the catalog and rejects contradictory media result shapes', async () => {
    const names = [
        'TextCodePointRange',
        'JsonPointer',
        'MediaSelectionRange',
        'BlockSubselection',
        'ConversationSelector',
        'ConversationSelectionRequest',
        'SelectionBinaryEvidence',
        'SelectionMediaEvidence',
        'SelectedBlock',
        'SelectedContextEntry',
        'ConversationSelection',
        'ConversationSelectionResult',
        'ConversationSlice',
        'ConversationSliceResult',
    ] as const;
    for (const name of names) {
        const key = name.replace(/[A-Z]/g, (character, index) => `${index ? '_' : ''}${character.toLowerCase()}`);
        expect(Reflect.get(CONVERSATION_JSON_SCHEMAS, key)).toBe(JsonSchemas[`${name}JsonSchema`]);
    }
    const doc = mediaSource('image'),
        range = { type: 'image_region' as const, coordinate_space: 'pixels' as const, x: 0, y: 0, width: 1, height: 1 };
    const value = (await selected(doc, query(doc, [await media(doc, range)]))).entries[0].blocks[0];
    const ajv = new Ajv2020({ strict: false });
    formats.default(ajv);
    const validate = ajv.compile(JsonSchemas.SelectedBlockJsonSchema);
    expect(validate(value)).toBe(true);
    expect(
        ConversationSelectionSchema.safeParse(await selected(doc, query(doc, [await media(doc, range)]))).success,
    ).toBe(true);
    for (const invalid of [
        { ...value, block_type: 'audio' },
        { ...value, range: { type: 'page_range', from_page: 1, through_page: 2 } },
        {
            ...value,
            evidence: {
                asset_id: 'asset',
                source_metadata_fingerprint: 'sha256:metadata',
                extent: 'unknown',
                content: { status: 'verified_inline', byte_length: 3 },
                mutation_ready: false,
            },
        },
        {
            ...value,
            evidence: {
                asset_id: 'asset',
                source_metadata_fingerprint: 'sha256:metadata',
                extent: 'unknown',
                content: { status: 'unverified', reason: 'external_content' },
                mutation_ready: true,
            },
        },
    ]) {
        expect(validate(invalid)).toBe(false);
        const base = await selected(doc, query(doc, [await media(doc, range)]));
        expect(
            ConversationSelectionSchema.safeParse({ ...base, entries: [{ ...base.entries[0], blocks: [invalid] }] })
                .success,
        ).toBe(false);
    }
});
