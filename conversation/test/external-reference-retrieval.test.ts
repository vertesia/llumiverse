import { describe, expect, it } from 'vitest';
import { hashUtf8Content } from '../src/content-integrity.js';
import { resolveActiveTextExternalReference } from '../src/external-reference-retrieval.js';
import { appendConversationRecords } from '../src/runtime.js';
import type { ConversationDocument } from '../src/types.js';
import { emptyDocument, RECORDED_AT, userTurn } from './fixtures.js';

const PATH = 'archive/assets/asset-text.txt';

async function referencedDocument(): Promise<ConversationDocument> {
    const initial = emptyDocument('conversation:retrieval');
    const bytes = await hashUtf8Content('Exact original text');
    const retrieval = {
        capability: 'read_artifact',
        version: 1,
        arguments: { asset_id: 'asset:text', path: PATH },
        tool_definition_id: 'definition:read',
    } as const;
    const turn = userTurn('turn:source');
    turn.blocks = [
        {
            id: 'block:reference',
            type: 'external_reference',
            asset_id: 'asset:text',
            original_type: 'text',
            description: 'Archived source',
            content_hash: bytes.content_hash,
            preview: 'Exact original',
            retrieval,
        },
    ];
    const asset = {
        id: 'asset:text',
        kind: 'text',
        mime_type: 'text/plain',
        storage: {
            type: 'external',
            resolver: 'vertesia.agent_artifact',
            locator: { storage_id: 'run:owner', artifact_path: PATH },
        },
        provenance: { type: 'received' },
        byte_length: bytes.byte_length,
        content_hash: bytes.content_hash,
        created_at: RECORDED_AT,
    } as const;
    const document = appendConversationRecords(
        initial,
        {
            assets: [asset],
            tool_definitions: [
                {
                    id: 'definition:read',
                    name: 'read_artifact',
                    version: `sha256:${'b'.repeat(64)}`,
                    input_schema: true,
                },
            ],
            active_tool_definition_ids: ['definition:read'],
        },
        {
            operation_id: 'archive:text',
            expected_revision: 0,
            payload_fingerprint: 'sha256:archive-text',
            recorded_at: RECORDED_AT,
        },
    ).document;
    document.turns.push(turn);
    document.context.entries.push({ id: 'entry:reference', type: 'source_turn', turn_id: turn.id });
    document.context.retrieval_requirements.push({
        id: 'requirement:read',
        asset_id: 'asset:text',
        retrieval,
        accepted_asset_operation_id: 'archive:text',
    });
    return document;
}

describe('resolveActiveTextExternalReference', () => {
    it('returns only an exact active reference backed by an accepted requirement and asset', async () => {
        const document = await referencedDocument();
        const resolved = resolveActiveTextExternalReference(document, 'asset:text');
        expect(resolved.accepted_asset_operation_id).toBe('archive:text');
        expect(resolved.tool_definition.name).toBe('read_artifact');
        expect(resolved.tool_definition.version).toBe(`sha256:${'b'.repeat(64)}`);
        expect(resolved.block.retrieval.version).toBe(1);
        expect(resolved.asset.content_hash).toBe(document.assets['asset:text'].content_hash);
        resolved.block.description = 'mutated';
        expect(document.turns[0].blocks[0]).toHaveProperty('description', 'Archived source');
    });

    it('rejects an inactive reference, missing requirement, wrong block, and mismatched hash', async () => {
        const document = await referencedDocument();
        expect(() => resolveActiveTextExternalReference(document, 'asset:text', 'block:other')).toThrow();

        const unsupportedAbi = structuredClone(document);
        const unsupportedBlock = unsupportedAbi.turns[0].blocks[0];
        if (unsupportedBlock.type !== 'external_reference') throw new Error('Expected external reference fixture');
        unsupportedBlock.retrieval.version = 2;
        unsupportedAbi.context.retrieval_requirements[0].retrieval.version = 2;
        expect(() => resolveActiveTextExternalReference(unsupportedAbi, 'asset:text')).toThrow();

        const wrongDefinition = structuredClone(document);
        const wrongDefinitionBlock = wrongDefinition.turns[0].blocks[0];
        if (wrongDefinitionBlock.type !== 'external_reference') throw new Error('Expected external reference fixture');
        wrongDefinitionBlock.retrieval.tool_definition_id = 'definition:foreign';
        wrongDefinition.context.retrieval_requirements[0].retrieval.tool_definition_id = 'definition:foreign';
        expect(() => resolveActiveTextExternalReference(wrongDefinition, 'asset:text')).toThrow();

        const wrongName = structuredClone(document);
        wrongName.tool_definitions['definition:read'].name = 'foreign_reader';
        expect(() => resolveActiveTextExternalReference(wrongName, 'asset:text')).toThrow();

        const inactiveDefinition = structuredClone(document);
        inactiveDefinition.context.active_tool_definition_ids = [];
        expect(() => resolveActiveTextExternalReference(inactiveDefinition, 'asset:text')).toThrow();

        const noRequirement = structuredClone(document);
        noRequirement.context.retrieval_requirements = [];
        expect(() => resolveActiveTextExternalReference(noRequirement, 'asset:text')).toThrow();

        const inactive = structuredClone(document);
        inactive.context.entries = [];
        expect(() => resolveActiveTextExternalReference(inactive, 'asset:text')).toThrow();

        const hidden = structuredClone(document);
        hidden.turns[0].model_visibility = 'exclude';
        expect(() => resolveActiveTextExternalReference(hidden, 'asset:text')).toThrow();

        const wrongHash = structuredClone(document);
        wrongHash.assets['asset:text'].content_hash = `sha256:${'0'.repeat(64)}`;
        expect(() => resolveActiveTextExternalReference(wrongHash, 'asset:text')).toThrow();
    });
});
