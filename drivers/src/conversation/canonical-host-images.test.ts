import {
    appendConversationRecords,
    createConversationDocument,
    createTextBlock,
    createToolTurn,
    createUserTurn,
    fingerprintJson,
    hashContentBytes,
    type ResolveConversationAsset,
} from '@llumiverse/conversation';
import { resolveCanonicalExecutionContextOptions } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { prepareBedrockConverseCanonicalContext } from '../bedrock/bedrock-converse-conversation-adapter.js';
import { prepareClaudeCanonicalContext } from '../shared/claude-messages-conversation-adapter.js';
import {
    finalizeGeminiPreparedRequest,
    prepareGeminiCanonicalContext,
} from '../vertexai/models/gemini-conversation-adapter.js';
import { hydrateCanonicalHostImages, hydrateCanonicalSelectedMediaAssets } from './canonical-host-images.js';
import { providerJsonValue } from './canonical-runtime.js';

const at = '2026-10-04T00:00:00.000Z';
const png = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg==',
    'base64',
);

type MediaKind = 'image' | 'document' | 'audio' | 'video';

async function source(
    nested = false,
    media: { kind: MediaKind; mime_type: string; bytes: Buffer } = {
        kind: 'image',
        mime_type: 'image/png',
        bytes: png,
    },
) {
    const initial = createConversationDocument({ id: 'conversation:host-image', created_at: at });
    const asset = {
        id: 'asset:received-image',
        kind: media.kind,
        mime_type: media.mime_type,
        storage: {
            type: 'external' as const,
            resolver: 'vertesia.agent_artifact',
            locator: { storage_id: 'owner', artifact_path: 'archive/image' },
        },
        provenance: { type: 'received' as const },
        created_at: at,
        ...(await hashContentBytes(media.bytes)),
    };
    const mediaBlock =
        media.kind === 'image'
            ? { id: 'block:nested-image', type: 'image' as const, asset_id: asset.id }
            : media.kind === 'document'
              ? { id: 'block:nested-image', type: 'document' as const, asset_id: asset.id }
              : media.kind === 'audio'
                ? { id: 'block:nested-image', type: 'audio' as const, asset_id: asset.id }
                : { id: 'block:nested-image', type: 'video' as const, asset_id: asset.id };
    const user = createUserTurn({
        id: 'turn:user',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: at },
        provenance: { type: 'received' },
        model_visibility: 'include',
        blocks: [
            createTextBlock({ id: 'block:text', text: 'Inspect this image.', format: 'plain' }),
            ...(!nested ? [mediaBlock] : []),
        ],
    });
    const call = {
        id: 'turn:call',
        kind: 'agent' as const,
        authority: 'ordinary' as const,
        status: 'completed' as const,
        timestamps: { recorded_at: at },
        provenance: { type: 'received' as const },
        model_visibility: 'include' as const,
        blocks: [
            {
                id: 'block:call',
                type: 'tool_call' as const,
                call_id: 'call:inspect',
                tool_name: 'inspect',
                executor: 'application' as const,
                arguments: { type: 'json' as const, value: {} },
            },
        ],
    };
    const result = createToolTurn({
        id: 'turn:result',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: at },
        provenance: { type: 'received' },
        model_visibility: 'include',
        blocks: [
            {
                id: 'block:result',
                type: 'tool_result',
                call_id: 'call:inspect',
                status: 'success',
                content: [mediaBlock],
            },
        ],
    });
    return appendConversationRecords(
        initial,
        {
            turns: nested ? [user, call, result] : [user],
            assets: [asset],
            ...(nested
                ? {
                      tool_definitions: [
                          {
                              id: 'tool-definition:inspect:v1',
                              name: 'inspect',
                              version: 'v1',
                              input_schema: { type: 'object' },
                              result_capabilities: [media.kind],
                          },
                      ],
                      active_tool_definition_ids: ['tool-definition:inspect:v1'],
                  }
                : {}),
            context_entries: [
                { id: 'entry:user', type: 'source_turn', turn_id: user.id },
                ...(nested ? [{ id: 'entry:call', type: 'source_turn' as const, turn_id: call.id }] : []),
                ...(nested ? [{ id: 'entry:result', type: 'source_turn' as const, turn_id: result.id }] : []),
            ],
        },
        {
            expected_revision: initial.revision,
            operation_id: 'append:host-image',
            payload_fingerprint: await fingerprintJson({ nested }),
            recorded_at: at,
        },
    ).document;
}

function options(conversation: Awaited<ReturnType<typeof source>>, model: string) {
    return resolveCanonicalExecutionContextOptions({
        model,
        conversation,
        conversation_runtime: {
            conversation_id: conversation.id,
            request_id: 'request:host-image',
            attempt_id: 'attempt:host-image',
            input_operation_id: 'input:host-image',
            response_operation_id: 'response:host-image',
            recorded_at: at,
        },
    });
}

describe('owned canonical host image projection', () => {
    it.each([false, true])('rejects absent selected images before hydration (nested=%s)', async (nested) => {
        const document = await source(nested);
        delete document.assets['asset:received-image'];
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        await expect(
            hydrateCanonicalHostImages({
                document,
                label: 'Missing selected media',
                selection: { allow_interrupted_with_complete_tool_calls: true },
                hydrated: new Map(),
                native_external: () => false,
                resolve_asset: resolver,
            }),
        ).rejects.toThrow('selected image asset asset:received-image is missing');
        expect(resolver).not.toHaveBeenCalled();
    });

    it('checks every selected asset before reading an earlier external image', async () => {
        const document = await source();
        const user = document.turns.find((turn) => turn.kind === 'user');
        if (user?.kind !== 'user') throw new Error('Selected user fixture is absent');
        user.blocks.push({ id: 'block:missing-image', type: 'image', asset_id: 'asset:missing' });
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        await expect(
            hydrateCanonicalHostImages({
                document,
                label: 'Partially missing selected media',
                selection: {},
                hydrated: new Map(),
                native_external: () => false,
                resolve_asset: resolver,
            }),
        ).rejects.toThrow('selected image asset asset:missing is missing');
        expect(resolver).not.toHaveBeenCalled();
    });

    it('projects a selected nested PDF in the actual Gemini function response', async () => {
        const bytes = Buffer.from('%PDF-1.7\n1 0 obj\n');
        const mimeType = 'application/pdf';
        const document = await source(true, { kind: 'document', mime_type: mimeType, bytes });
        const original = structuredClone(document);
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield bytes;
        });
        const prepared = await prepareGeminiCanonicalContext({
            options: options(document, 'gemini-2.5-pro'),
            provider: 'google',
            resolve_asset: resolver,
        });
        expect(resolver).toHaveBeenCalledOnce();
        const encoded = JSON.stringify(prepared.native_conversation);
        expect(encoded).toContain(`"mimeType":"${mimeType}"`);
        expect(encoded.split(bytes.toString('base64'))).toHaveLength(2);
        expect(encoded).toContain('functionResponse');
        expect(document).toEqual(original);
        expect(prepared.document.assets['asset:received-image']?.storage.type).toBe('external');
    });

    it.each([
        ['audio', 'audio/wav', Buffer.from([82, 73, 70, 70, 4, 0, 0, 0, 87, 65, 86, 69])],
        ['video', 'video/mp4', Buffer.from([0, 0, 0, 16, 102, 116, 121, 112, 105, 115, 111, 109])],
    ] as const)(
        'projects selected ordinary %s input while rejecting it as a function response',
        async (kind, mimeType, bytes) => {
            const ordinary = await source(false, { kind, mime_type: mimeType, bytes });
            const resolve = vi.fn<ResolveConversationAsset>(async function* () {
                yield bytes;
            });
            const prepared = await prepareGeminiCanonicalContext({
                options: options(ordinary, 'gemini-2.5-pro'),
                provider: 'google',
                resolve_asset: resolve,
            });
            const encoded = JSON.stringify(prepared.native_conversation);
            expect(encoded).toContain(`"mimeType":"${mimeType}"`);
            expect(encoded).toContain(bytes.toString('base64'));
            expect(resolve).toHaveBeenCalledOnce();
            const nested = await source(true, { kind, mime_type: mimeType, bytes });
            resolve.mockClear();
            await expect(
                prepareGeminiCanonicalContext({
                    options: options(nested, 'gemini-2.5-pro'),
                    provider: 'google',
                    resolve_asset: resolve,
                }),
            ).rejects.toThrow('cannot project audio or video in a function response');
            expect(resolve).not.toHaveBeenCalled();
        },
    );

    it.each([
        ['document', 'application/pdf', true],
        ['audio', 'audio/wav', false],
        ['video', 'video/mp4', false],
    ] as const)(
        'rejects mismatched selected %s bytes before Gemini native preparation',
        async (kind, mimeType, nested) => {
            const document = await source(nested, { kind, mime_type: mimeType, bytes: png });
            const resolver = vi.fn<ResolveConversationAsset>(async function* () {
                yield png;
            });
            await expect(
                prepareGeminiCanonicalContext({
                    options: options(document, 'gemini-2.5-pro'),
                    provider: 'google',
                    resolve_asset: resolver,
                }),
            ).rejects.toThrow('invalid media bytes');
            expect(resolver).toHaveBeenCalledOnce();
        },
    );

    it.each([
        ['image', 'image/gif', Buffer.from('GIF89a'), 0],
        ['image', 'image/webp', Buffer.from([82, 73, 70, 70, 4, 0, 0, 0, 87, 69, 66, 80]), 8],
        ['document', 'application/pdf', Buffer.from('%PDF-1.7\n%%EOF\n'), 0],
        ['audio', 'audio/wav', Buffer.from([82, 73, 70, 70, 4, 0, 0, 0, 87, 65, 86, 69]), 8],
        ['audio', 'audio/mpeg', Buffer.from([73, 68, 51, 4, 0, 0, 0, 0, 0, 0]), 0],
        ['video', 'video/mp4', Buffer.from([0, 0, 0, 16, 102, 116, 121, 112, 105, 115, 111, 109]), 4],
    ] as const)(
        'rejects high-bit aliasing in %s %s signatures after integrity verification',
        async (kind, mimeType, valid, changedAt) => {
            const corrupted = Buffer.from(valid);
            corrupted[changedAt] |= 0x80;
            const document = await source(false, { kind, mime_type: mimeType, bytes: corrupted });
            const resolver = vi.fn<ResolveConversationAsset>(async function* () {
                yield corrupted;
            });
            await expect(
                hydrateCanonicalSelectedMediaAssets({
                    assets: document.assets,
                    selected_ids: new Set(['asset:received-image']),
                    media_kinds: [kind],
                    label: 'Exact media signature',
                    hydrated: new Map(),
                    native_external: () => false,
                    resolve_asset: resolver,
                }),
            ).rejects.toThrow('invalid media bytes');
            expect(resolver).toHaveBeenCalledOnce();
            expect(document.assets['asset:received-image']).toMatchObject(await hashContentBytes(corrupted));
        },
    );

    it('hydrates selected received images in Claude, Gemini and Bedrock without changing canonical refs', async () => {
        const document = await source();
        const original = structuredClone(document);
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        const claude = await prepareClaudeCanonicalContext({
            options: options(document, 'claude-sonnet-4-20250514'),
            provider: 'anthropic',
            resolve_asset: resolver,
        });
        const gemini = await prepareGeminiCanonicalContext({
            options: options(document, 'gemini-2.5-pro'),
            provider: 'google',
            resolve_asset: resolver,
        });
        const bedrock = await prepareBedrockConverseCanonicalContext({
            options: options(document, 'anthropic.claude-sonnet-4-6-v1:0'),
            provider: 'bedrock',
            resolve_asset: resolver,
        });
        expect(resolver).toHaveBeenCalledTimes(3);
        expect(JSON.stringify(claude.native_conversation)).toContain(png.toString('base64'));
        expect(JSON.stringify(gemini.native_conversation)).toContain(png.toString('base64'));
        expect(JSON.stringify(bedrock.native_conversation)).toContain('image');
        expect(bedrock.native_conversation.messages?.[0]?.content?.[1]).toMatchObject({ image: { format: 'png' } });
        expect(document).toEqual(original);
        expect(claude.document.assets['asset:received-image']?.storage.type).toBe('external');
        expect(gemini.document.assets['asset:received-image']?.storage.type).toBe('external');
        expect(bedrock.document.assets['asset:received-image']?.storage.type).toBe('external');
    });

    it('fails without a host resolver, rejects corrupted bytes, and ignores unselected image blocks', async () => {
        const document = await source();
        await expect(
            prepareClaudeCanonicalContext({
                options: options(document, 'claude-sonnet-4-20250514'),
                provider: 'anthropic',
            }),
        ).rejects.toThrow('no host resolver');
        const corrupt = vi.fn<ResolveConversationAsset>(async function* () {
            yield new Uint8Array(png.byteLength);
        });
        await expect(
            prepareGeminiCanonicalContext({
                options: options(document, 'gemini-2.5-pro'),
                provider: 'google',
                resolve_asset: corrupt,
            }),
        ).rejects.toThrow();
        const inactive = createConversationDocument({ id: 'conversation:inactive-image', created_at: at });
        const textTurn = createUserTurn({
            id: 'turn:visible-text',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            provenance: { type: 'received' },
            model_visibility: 'include',
            blocks: [createTextBlock({ id: 'block:visible-text', text: 'Visible text.', format: 'plain' })],
        });
        const imageTurn = createUserTurn({
            id: 'turn:inactive-image',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            provenance: { type: 'received' },
            model_visibility: 'include',
            blocks: [{ id: 'block:inactive-image', type: 'image', asset_id: 'asset:received-image' }],
        });
        const sourceAsset = document.assets['asset:received-image'];
        if (sourceAsset === undefined) throw new Error('Missing received image asset fixture');
        const selectedText = appendConversationRecords(
            inactive,
            {
                turns: [textTurn, imageTurn],
                assets: [sourceAsset],
                context_entries: [{ id: 'entry:visible-text', type: 'source_turn', turn_id: textTurn.id }],
            },
            {
                expected_revision: inactive.revision,
                operation_id: 'append:inactive-image',
                payload_fingerprint: await fingerprintJson({ selected: 'text' }),
                recorded_at: at,
            },
        ).document;
        const noRead = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        await prepareClaudeCanonicalContext({
            options: options(selectedText, 'claude-sonnet-4-20250514'),
            provider: 'anthropic',
            resolve_asset: noRead,
        });
        expect(noRead).not.toHaveBeenCalled();
    });

    it('selects nested tool-result images and rejects a source change after preparation', async () => {
        const document = await source(true);
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        const prepared = await prepareGeminiCanonicalContext({
            options: options(document, 'gemini-2.5-pro'),
            provider: 'google',
            resolve_asset: resolver,
        });
        expect(resolver).toHaveBeenCalled();
        expect(JSON.stringify(prepared.native_conversation)).toContain(png.toString('base64'));
        expect(prepared.document.assets['asset:received-image']?.storage.type).toBe('external');
        const nativePayload = {
            model: 'gemini-2.5-pro',
            contents: prepared.native_conversation.contents,
        };
        const finalized = await finalizeGeminiPreparedRequest(prepared, nativePayload);
        expect(finalized.receipt.request_fingerprint).toBe(await fingerprintJson(providerJsonValue(nativePayload)));
        expect(resolver).toHaveBeenCalledTimes(1);
        const firstTurn = prepared.document.turns[0];
        if (firstTurn?.kind !== 'user') throw new Error('Expected retained user turn');
        firstTurn.blocks.push(
            createTextBlock({ id: 'block:changed', text: 'Changed after preparation.', format: 'plain' }),
        );
        await expect(finalizeGeminiPreparedRequest(prepared, nativePayload)).rejects.toThrow(
            'source changed after native preparation',
        );
        expect(resolver).toHaveBeenCalledTimes(1);
    });
});
