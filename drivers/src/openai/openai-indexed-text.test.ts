import {
    appendConversationRecords,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    hashContentBytes,
    type IndexedConversationRecordStore,
    loadIndexedSelectedTextContext,
    stageIndexedConversationSnapshot,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import { createRequestReceipt, providerJsonValue } from '../conversation/canonical-runtime.js';
import { OpenAISDKChatCompletionsProtocol } from './openai_chat_completions.js';
import {
    compileOpenAIChatCompletionsConversation,
    compileOpenAIChatIndexedSelectedText,
    OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
} from './openai-chat-conversation-adapter.js';

describe('indexed OpenAI Chat selected text', () => {
    it('compiles the exact full-document prompt from a bounded selected context', async () => {
        const recordedAt = '2026-10-02T00:00:00.000Z';
        const initial = createConversationDocument({ id: 'conversation:indexed-openai', created_at: recordedAt });
        const user = createUserTurn({
            id: 'turn:user',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: recordedAt },
            provenance: { type: 'received' },
            model_visibility: 'include',
            blocks: [
                createTextBlock({ id: 'block:small', text: 'Hello', format: 'plain' }),
                createTextBlock({ id: 'block:cold', text: 'Not selected', format: 'plain' }),
            ],
        });
        const document = appendConversationRecords(
            initial,
            {
                turns: [user],
                context_entries: [
                    {
                        id: 'entry:user',
                        type: 'source_turn',
                        turn_id: user.id,
                        block_ids: ['block:small'],
                    },
                ],
            },
            {
                expected_revision: initial.revision,
                operation_id: 'operation:user',
                payload_fingerprint: 'sha256:input',
                recorded_at: recordedAt,
            },
        ).document;
        const pages = new Map<string, Uint8Array>();
        const records = new Map<string, Uint8Array>();
        const reads: string[] = [];
        const store: IndexedConversationRecordStore = {
            async read(ref) {
                const bytes = pages.get(ref.content_hash);
                if (!bytes) throw new Error('missing page');
                return bytes;
            },
            async write(bytes, ref) {
                pages.set(ref.content_hash, Uint8Array.from(bytes));
            },
            async readRecord(value) {
                reads.push(`${value.kind}:${value.id}`);
                const bytes = records.get(value.content_hash);
                if (!bytes) throw new Error('missing record');
                return Uint8Array.from(bytes);
            },
            async writeRecord(value, bytes) {
                expect((await hashContentBytes(bytes)).content_hash).toBe(value.content_hash);
                records.set(value.content_hash, Uint8Array.from(bytes));
            },
        };
        const staged = await stageIndexedConversationSnapshot(document, undefined, store);
        reads.length = 0;
        const selected = await loadIndexedSelectedTextContext(store, staged.root, staged.locator);
        const target = { provider: 'openai', model: 'test-model' };
        expect(compileOpenAIChatIndexedSelectedText(selected, target)).toEqual(
            compileOpenAIChatCompletionsConversation(document, target),
        );
        const runtime = {
            conversation_id: document.id,
            request_id: 'request:fresh',
            attempt_id: 'attempt:fresh',
            input_operation_id: 'operation:user',
            response_operation_id: 'response:fresh',
            recorded_at: recordedAt,
            purpose: 'interaction',
        };
        const prepared = await new OpenAISDKChatCompletionsProtocol({}).prepareIndexedTextRequest({
            selection: selected,
            runtime,
            options: { model: target.model },
            provider: target.provider,
            stream: false,
        });
        const full = compileOpenAIChatCompletionsConversation(document, target);
        const expectedReceipt = await createRequestReceipt(
            document,
            runtime,
            {
                ...target,
                protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
                adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
            },
            providerJsonValue(prepared.payload),
            full.mappings,
            [],
        );
        expect(prepared.receipt).toEqual(expectedReceipt);
        expect(prepared.status).toBe('awaiting_durable_prepared_record');
        expect(reads).not.toContain('blocks:block:cold');
        expect(selected.completeness).toBe('selected_text_pending_admission');

        const mutableSelection = structuredClone(selected);
        const mutableRuntime = structuredClone(runtime);
        const mutableOptions = { model: target.model, model_options: { temperature: 0.25 } };
        const extraBody = { nested: { token: 'original' } };
        const pending = new OpenAISDKChatCompletionsProtocol({ extraBody }).prepareIndexedTextRequest({
            selection: mutableSelection,
            runtime: mutableRuntime,
            options: mutableOptions,
            provider: target.provider,
            stream: false,
        });
        mutableSelection.source.revision = 99;
        mutableSelection.context.active_tool_definition_ids.push('tool:late');
        mutableRuntime.request_id = 'request:late';
        mutableOptions.model = 'late-model';
        mutableOptions.model_options.temperature = 0.95;
        extraBody.nested.token = 'late';
        const owned = await pending;
        expect(owned.source).toEqual(selected.source);
        expect(owned.receipt.source).toEqual(selected.source);
        expect(owned.receipt.request_id).toBe(runtime.request_id);
        expect(owned.payload.model).toBe(target.model);
        expect(owned.payload.temperature).toBe(0.25);
        expect(owned.payload.extra_body).toEqual({ nested: { token: 'original' } });
    });
});
