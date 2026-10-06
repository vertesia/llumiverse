import {
    appendConversationRecords,
    appendConversationRecordsWithProcessing,
    type ConversationDocument,
    createConversationDocument,
    createTextBlock,
    createTextExternalizationProcessor,
    createUserTurn,
    fingerprintJson,
    hashUtf8Content,
    type ProcessingStore,
    parseConversationDocument,
    runProcessingJob,
    setProcessingPolicy,
    textExternalizationArchiveInputs,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import { compileBedrockConverseConversation } from '../bedrock/bedrock-converse-conversation-adapter.js';
import { compileOpenAIChatCompletionsConversation } from '../openai/openai-chat-conversation-adapter.js';
import { compileOpenAIResponsesConversation } from '../openai/openai-responses-conversation-adapter.js';
import { compileClaudeMessagesConversation } from '../shared/claude-messages-conversation-adapter.js';
import { compileGeminiConversation } from '../vertexai/models/gemini-conversation-adapter.js';
import { retrievableTextReference } from './retrievable-text-reference.js';

const AT = '2026-09-11T00:01:00.000Z';
const ORIGINAL = `Archived exact original: ${'secret original body '.repeat(100)}`;

class MemoryStore implements ProcessingStore {
    constructor(public current: ConversationDocument) {}

    async load(): Promise<ConversationDocument> {
        return structuredClone(this.current);
    }

    async commit(expectedRevision: number, document: ConversationDocument): Promise<boolean> {
        if (this.current.revision !== expectedRevision) return false;
        this.current = parseConversationDocument(structuredClone(document));
        return true;
    }
}

async function externalized(): Promise<ConversationDocument> {
    const source = await setProcessingPolicy(
        createConversationDocument({ id: 'conversation:external-reference-native', created_at: AT }),
        {
            operation_id: 'policy:externalize',
            expected_revision: 0,
            recorded_at: AT,
            enabled: true,
            processors: [
                {
                    id: 'externalize-text',
                    version: '1',
                    scope: 'on_append',
                    config: {},
                    required: true,
                    failure_behavior: 'block',
                },
            ],
        },
    );
    const turn = createUserTurn({
        id: 'turn:original',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: AT },
        model_visibility: 'include',
        provenance: { type: 'received' },
        blocks: [createTextBlock({ id: 'block:original', text: ORIGINAL, format: 'plain' })],
    });
    const readInputSchema = {
        type: 'object',
        properties: { asset_id: { type: 'string' } },
        required: ['asset_id'],
        additionalProperties: false,
    };
    const definitionVersion = await fingerprintJson(readInputSchema);
    const accepted = await appendConversationRecordsWithProcessing(
        source.document,
        {
            turns: [turn],
            context_entries: [{ id: 'entry:original', type: 'source_turn', turn_id: turn.id }],
            tool_definitions: [
                {
                    id: 'definition:read',
                    name: 'read_artifact',
                    version: definitionVersion,
                    input_schema: readInputSchema,
                },
            ],
            active_tool_definition_ids: ['definition:read'],
        },
        {
            operation_id: 'append:original',
            expected_revision: source.document.revision,
            payload_fingerprint: 'sha256:original',
            recorded_at: AT,
        },
    );
    const job = Object.values(accepted.document.processing.jobs ?? {})[0];
    if (!job) throw new Error('Expected durable text externalization job');
    const archive = await textExternalizationArchiveInputs(accepted.document, job);
    const integrity = archive.integrities[0];
    if (!integrity) throw new Error('Expected selected original text integrity');
    const asset = {
        id: 'asset:original',
        kind: 'text' as const,
        mime_type: 'text/plain',
        storage: {
            type: 'external' as const,
            resolver: 'vertesia.agent_artifact',
            locator: { storage_id: 'owner:run', artifact_path: 'archive/original.txt' },
        },
        provenance: { type: 'received' as const },
        content_hash: integrity.content_hash,
        byte_length: integrity.byte_length,
        created_at: AT,
    };
    const archived = await appendConversationRecordsWithProcessing(
        accepted.document,
        { assets: [asset] },
        {
            operation_id: `processing:archive:${job.id}`,
            expected_revision: accepted.document.revision,
            payload_fingerprint: archive.payload_fingerprint,
            recorded_at: AT,
        },
    );
    const store = new MemoryStore(archived.document);
    const processor = createTextExternalizationProcessor(({ asset: selected }) => ({
        capability: 'read_artifact',
        version: 1,
        arguments: { asset_id: selected.id },
        tool_definition_id: 'definition:read',
    }));
    const result = await runProcessingJob(store, { resolve: () => processor }, job.id, 'attempt:externalize', () => AT);
    if (result.status !== 'completed') throw new Error('Externalization did not complete');
    return store.current;
}

function selectedReference(document: ConversationDocument) {
    const replacement = Object.values(document.compactions)[0]?.replacement_turns[0];
    const block = replacement?.blocks[0];
    if (block?.type !== 'external_reference') throw new Error('Expected accepted external reference');
    return block;
}

const compilers = {
    'OpenAI Chat': (document: ConversationDocument) => compileOpenAIChatCompletionsConversation(document).conversation,
    'OpenAI Responses': (document: ConversationDocument) => compileOpenAIResponsesConversation(document).conversation,
    'Claude Messages': (document: ConversationDocument) => compileClaudeMessagesConversation(document).conversation,
    Gemini: (document: ConversationDocument) => compileGeminiConversation(document).conversation,
    'Bedrock Converse': (document: ConversationDocument) => compileBedrockConverseConversation(document).conversation,
};

describe('native projection of actual externalize-text output', () => {
    it('uses the accepted selected preview and rejects missing or oversized cues', async () => {
        const document = await externalized();
        const block = selectedReference(document);
        expect(retrievableTextReference(document, block)).toContain('Preview: Archived exact original:');
        delete block.preview;
        expect(() => retrievableTextReference(document, block)).toThrow('lacks a bounded preview');
        block.preview = 'x'.repeat(513);
        expect(() => retrievableTextReference(document, block)).toThrow('lacks a bounded preview');
    });

    it('requires the exact active retrieval ABI and bounds serialized arguments', async () => {
        const document = await externalized();
        const block = selectedReference(document);
        const requirement = document.context.retrieval_requirements[0];
        if (!requirement) throw new Error('Expected accepted retrieval requirement');
        document.context.active_tool_definition_ids = [];
        expect(() => retrievableTextReference(document, block)).toThrow('not required by the active context');
        document.context.active_tool_definition_ids = ['definition:read'];
        block.retrieval.version = 2;
        requirement.retrieval.version = 2;
        expect(() => retrievableTextReference(document, block)).toThrow('not required by the active context');
        block.retrieval.version = 1;
        requirement.retrieval.version = 1;
        block.retrieval.arguments = { asset_id: 'asset:original', extra: 'x'.repeat(2049) };
        requirement.retrieval.arguments = structuredClone(block.retrieval.arguments);
        expect(() => retrievableTextReference(document, block)).toThrow('oversized retrieval arguments');
    });

    it.each(Object.entries(compilers))(
        '%s keeps a bounded retrieval cue without hydrating archived text',
        async (_name, compile) => {
            const document = await externalized();
            const native = JSON.stringify(compile(document)) ?? '';
            expect(native).toContain('read_artifact');
            expect(native).toContain('asset:original');
            expect(native).toContain('Preview: Archived exact original:');
            expect(native).not.toContain(ORIGINAL);
            expect(native.length).toBeLessThan(4096);
            const retainedAsset = document.assets['asset:original'];
            if (!retainedAsset) throw new Error('Archived asset is unavailable');
            expect(retainedAsset.storage.type).toBe('external');
        },
    );

    it.each(Object.entries(compilers))(
        '%s refuses a reference with no active retrieval requirement',
        async (_name, compile) => {
            const document = await externalized();
            document.context.retrieval_requirements = [];
            expect(() => compile(document)).toThrow('not required by the active context');
        },
    );

    it.each(Object.entries(compilers))('%s refuses a changed archive hash', async (_name, compile) => {
        const document = await externalized();
        const retainedAsset = document.assets['asset:original'];
        if (!retainedAsset) throw new Error('Archived asset is unavailable');
        retainedAsset.content_hash = `sha256:${'0'.repeat(64)}`;
        expect(() => compile(document)).toThrow();
    });
});

async function jsonOriginal() {
    const original = JSON.stringify({
        count: 3,
        exact: 'private JSON original α'.repeat(10_000),
        flags: [false, null],
    });
    const integrity = await hashUtf8Content(original);
    const retrieval = {
        capability: 'read_artifact',
        version: 1,
        tool_definition_id: 'definition:json:read',
        arguments: { asset_id: 'asset:json', path: 'original.json', start_byte: 0, byte_count: 5000 },
    };
    const block = {
        id: 'block:json',
        type: 'external_reference' as const,
        asset_id: 'asset:json',
        original_type: 'json' as const,
        description: 'Exact JSON original',
        preview: 'Archived exact JSON original',
        content_hash: integrity.content_hash,
        retrieval,
    };
    const document = appendConversationRecords(
        createConversationDocument({ id: 'conversation:json:cue', created_at: AT }),
        {
            turns: [
                createUserTurn({
                    id: 'turn:json',
                    authority: 'ordinary',
                    status: 'completed',
                    model_visibility: 'include',
                    timestamps: { recorded_at: AT },
                    provenance: { type: 'received' },
                    blocks: [block],
                }),
            ],
            context_entries: [{ id: 'entry:json', type: 'source_turn', turn_id: 'turn:json' }],
            assets: [
                {
                    id: 'asset:json',
                    kind: 'json',
                    mime_type: 'application/json',
                    storage: {
                        type: 'external',
                        resolver: 'vertesia.agent_artifact',
                        locator: { storage_id: 'owner:run', artifact_path: 'original.json' },
                    },
                    provenance: { type: 'received' },
                    created_at: AT,
                    ...integrity,
                },
            ],
            tool_definitions: [
                {
                    id: 'definition:json:read',
                    name: 'read_artifact',
                    version: 'json:1',
                    input_schema: {
                        type: 'object',
                        properties: { asset_id: { type: 'string' } },
                        required: ['asset_id'],
                        additionalProperties: false,
                    },
                },
            ],
            active_tool_definition_ids: ['definition:json:read'],
            retrieval_requirements: [
                {
                    id: 'requirement:json',
                    asset_id: 'asset:json',
                    retrieval,
                    accepted_asset_operation_id: 'accept:json',
                },
            ],
        },
        {
            expected_revision: 0,
            operation_id: 'accept:json',
            payload_fingerprint: await fingerprintJson({ original, block }),
            recorded_at: AT,
        },
    ).document;
    return { document, block, original };
}

describe('native JSON-original retrieval cues', () => {
    it.each(Object.entries(compilers))(
        '%s keeps authentic typed JSON external without hydrating full original',
        async (_name, compile) => {
            const { document, original } = await jsonOriginal();
            const native = JSON.stringify(compile(document));
            expect(native).toContain('Full original JSON is available through read_artifact');
            expect(native).toContain('original.json');
            expect(native).toContain('Preview: Archived exact JSON original');
            expect(native).not.toContain(original);
            expect(native.length).toBeLessThan(4096);
            expect(document.assets['asset:json'].kind).toBe('json');
        },
    );

    it.each(Object.entries(compilers))(
        '%s rejects JSON reference/asset type mismatch and missing accepted requirement',
        async (_name, compile) => {
            const { document } = await jsonOriginal();
            const changed = structuredClone(document);
            changed.assets['asset:json'].kind = 'text';
            expect(() => compile(changed)).toThrow();
            document.context.retrieval_requirements = [];
            expect(() => compile(document)).toThrow();
        },
    );
});
