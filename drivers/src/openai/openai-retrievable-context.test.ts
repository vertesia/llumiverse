import {
    appendConversationRecords,
    applyContextChange,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    hashUtf8Content,
    planContextChange,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import { compileOpenAIChatCompletionsConversation } from './openai-chat-conversation-adapter.js';

const AT = '2026-10-03T00:00:00.000Z';

describe('OpenAI Chat retrievable canonical context', () => {
    it('projects a bounded preview and exact read tool selector only for a verified active requirement', async () => {
        const original = 'An exact original that remains archived after context replacement.';
        const integrity = await hashUtf8Content(original);
        const path = 'archive/assets/original.txt';
        const initial = createConversationDocument({ id: 'conversation:retrievable-openai', created_at: AT });
        const turn = createUserTurn({
            id: 'turn:original',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: AT },
            provenance: { type: 'received' },
            model_visibility: 'include',
            blocks: [createTextBlock({ id: 'block:original', text: original, format: 'plain' })],
        });
        const accepted = appendConversationRecords(
            initial,
            {
                turns: [turn],
                context_entries: [{ id: 'entry:original', type: 'source_turn', turn_id: turn.id }],
                tool_definitions: [{ id: 'definition:read', name: 'read_artifact', version: '1', input_schema: true }],
                active_tool_definition_ids: ['definition:read'],
            },
            {
                operation_id: 'input:original',
                expected_revision: 0,
                payload_fingerprint: 'sha256:input',
                recorded_at: AT,
            },
        ).document;
        const asset = {
            id: 'asset:original',
            kind: 'text' as const,
            mime_type: 'text/plain',
            storage: {
                type: 'external' as const,
                resolver: 'vertesia.agent_artifact',
                locator: { storage_id: 'owner:run', artifact_path: path },
            },
            provenance: { type: 'received' as const },
            content_hash: integrity.content_hash,
            byte_length: integrity.byte_length,
            created_at: AT,
        };
        const staged = appendConversationRecords(
            accepted,
            { assets: [asset] },
            {
                operation_id: 'archive:original',
                expected_revision: accepted.revision,
                payload_fingerprint: integrity.content_hash,
                recorded_at: AT,
            },
        ).document;
        const plan = await planContextChange(staged, {
            expected_revision: staged.revision,
            expected_context_revision: staged.context.revision,
            entry_ids: ['entry:original'],
        });
        const reference = {
            id: 'block:reference',
            type: 'external_reference' as const,
            asset_id: asset.id,
            original_type: 'text' as const,
            description: 'Archived original',
            content_hash: integrity.content_hash,
            preview: 'An exact original',
            retrieval: {
                capability: 'read_artifact',
                version: 1,
                arguments: { asset_id: asset.id, path },
                tool_definition_id: 'definition:read',
            },
        };
        const replacement = {
            ...turn,
            id: 'turn:reference',
            kind: 'agent' as const,
            blocks: [reference],
            provenance: {
                type: 'derived' as const,
                derivation_id: 'compaction:original',
                source_turn_ids: plan.source_turn_ids,
                source_hash: plan.source_fingerprint,
            },
        };
        const changed = await applyContextChange(staged, {
            operation_id: 'context:externalize',
            expected_revision: staged.revision,
            expected_context_revision: staged.context.revision,
            expected_source_fingerprint: plan.source_fingerprint,
            recorded_at: AT,
            entry_ids: plan.entry_ids,
            proposal: {
                kind: 'replace_with_compaction',
                compaction_id: 'compaction:original',
                strategy: { id: 'externalize-text', version: '1', configuration_fingerprint: 'sha256:config' },
                replacement_turns: [replacement],
                fidelity: 'retrievable',
                accepted_asset_operation_id: 'archive:original',
                retained_asset_ids: [asset.id],
                generation_ids: [],
                placement: { mode: 'first_selected', causal_order: 'contiguous' },
            },
        });
        const compiled = compileOpenAIChatCompletionsConversation(changed.document);
        const text = JSON.stringify(compiled.conversation.messages);
        expect(text).toContain('An exact original');
        expect(text).toContain('read_artifact');
        expect(text).toContain('asset:original');
        expect(text).not.toContain(original);

        const missingRequirement = structuredClone(changed.document);
        missingRequirement.context.retrieval_requirements = [];
        expect(() => compileOpenAIChatCompletionsConversation(missingRequirement)).toThrow(
            'not required by the active context',
        );
        const wrongAsset = structuredClone(changed.document);
        wrongAsset.assets[asset.id].content_hash = `sha256:${'0'.repeat(64)}`;
        expect(() => compileOpenAIChatCompletionsConversation(wrongAsset)).toThrow();

        const genericInput = appendConversationRecords(
            initial,
            {
                turns: [turn],
                context_entries: [{ id: 'entry:original', type: 'source_turn', turn_id: turn.id }],
                tool_definitions: [{ id: 'definition:blob', name: 'retrieve_blob', version: '1', input_schema: true }],
                active_tool_definition_ids: ['definition:blob'],
            },
            {
                operation_id: 'input:generic',
                expected_revision: 0,
                payload_fingerprint: 'sha256:generic',
                recorded_at: AT,
            },
        ).document;
        const genericAsset = {
            ...asset,
            storage: {
                type: 'external' as const,
                resolver: 'acme.blob_store',
                locator: { opaque_key: 'blob:original' },
            },
        };
        const genericStaged = appendConversationRecords(
            genericInput,
            { assets: [genericAsset] },
            {
                operation_id: 'archive:generic',
                expected_revision: genericInput.revision,
                payload_fingerprint: integrity.content_hash,
                recorded_at: AT,
            },
        ).document;
        const genericPlan = await planContextChange(genericStaged, {
            expected_revision: genericStaged.revision,
            expected_context_revision: genericStaged.context.revision,
            entry_ids: ['entry:original'],
        });
        const genericReplacement = {
            ...replacement,
            blocks: [
                {
                    ...reference,
                    retrieval: {
                        capability: 'retrieve_blob',
                        version: 1,
                        arguments: { opaque_key: 'blob:original' },
                        tool_definition_id: 'definition:blob',
                    },
                },
            ],
            provenance: {
                ...replacement.provenance,
                source_turn_ids: genericPlan.source_turn_ids,
                source_hash: genericPlan.source_fingerprint,
            },
        };
        const genericChanged = await applyContextChange(genericStaged, {
            operation_id: 'context:generic',
            expected_revision: genericStaged.revision,
            expected_context_revision: genericStaged.context.revision,
            expected_source_fingerprint: genericPlan.source_fingerprint,
            recorded_at: AT,
            entry_ids: genericPlan.entry_ids,
            proposal: {
                kind: 'replace_with_compaction',
                compaction_id: 'compaction:original',
                strategy: { id: 'externalize-text', version: '1', configuration_fingerprint: 'sha256:config' },
                replacement_turns: [genericReplacement],
                fidelity: 'retrievable',
                accepted_asset_operation_id: 'archive:generic',
                retained_asset_ids: [genericAsset.id],
                generation_ids: [],
                placement: { mode: 'first_selected', causal_order: 'contiguous' },
            },
        });
        const genericText = JSON.stringify(
            compileOpenAIChatCompletionsConversation(genericChanged.document).conversation.messages,
        );
        expect(genericText).toContain('retrieve_blob');
        expect(genericText).toContain('opaque_key');
        expect(genericText).not.toContain('read_artifact');
    });
});
