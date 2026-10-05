import {
    appendConversationRecords,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
    deriveConversationId,
    fingerprintJson,
    INDEXED_MEASURED_NATIVE_PREPARED_VALIDATOR_PROFILE,
    INDEXED_PROCESSED_INPUT_PREPARED_VALIDATOR_PROFILE,
    type IndexedConversationRecordStore,
    indexedProcessingContextFingerprint,
    loadIndexedSelectedTextContext,
    parseConversationPreparedRequestRecord,
    stageIndexedConversationSnapshot,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import { OpenAIDriver } from '../openai/openai.js';
import { OpenAIChatCompletionsDriver } from '../openai/openai_chat_completions.js';
import { indexedPreparedReceiptMatches } from './indexed-prepared-receipt.js';

// These test adapter integrity constraints only. Actual epoch/phase authority and measured count
// are exercised by the installed handoff HTTP/Mongo/Temporal fixture, never conferred by this data.
async function fixture(protocol: 'chat' | 'responses') {
    const at = '2026-10-05T00:00:00.000Z';
    let document = createConversationDocument({ id: 'conversation:measured-indexed', created_at: at });
    for (let i = 0; i < 4; i++) {
        const turn = createUserTurn({
            id: `turn:${i}`,
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            provenance: { type: 'received' },
            model_visibility: 'include',
            blocks: [createTextBlock({ id: `block:${i}`, text: `text ${i}`, format: 'plain' })],
        });
        document = appendConversationRecords(
            document,
            {
                turns: [turn],
                context_entries: [{ id: `entry:${i}`, type: 'source_turn', turn_id: turn.id }],
            },
            {
                expected_revision: document.revision,
                operation_id: `operation:${i}`,
                payload_fingerprint: `sha256:input${i}`,
                recorded_at: at,
            },
        ).document;
    }
    const bytes = new Map<string, Uint8Array>();
    const store: IndexedConversationRecordStore = {
        async read(ref) {
            const value = bytes.get(ref.content_hash);
            if (!value) throw new Error('Missing page');
            return value;
        },
        async write(value, ref) {
            bytes.set(ref.content_hash, Uint8Array.from(value));
        },
        async readRecord(ref) {
            const value = bytes.get(ref.content_hash);
            if (!value) throw new Error('Missing record');
            return value;
        },
        async writeRecord(ref, value) {
            bytes.set(ref.content_hash, Uint8Array.from(value));
        },
    };
    const staged = await stageIndexedConversationSnapshot(document, undefined, store);
    const selection = await loadIndexedSelectedTextContext(store, staged.root, staged.locator);
    const runtime = {
        conversation_id: document.id,
        request_id: 'request:measured',
        attempt_id: 'attempt:measured',
        input_operation_id: 'operation:0',
        response_operation_id: 'operation:answer',
        recorded_at: at,
        purpose: 'interaction',
        materialized_input: { operation_id: 'operation:0', result_revision: 1 },
    };
    const driver =
        protocol === 'chat'
            ? new OpenAIChatCompletionsDriver({ apiKey: 'test', endpoint: 'https://indexed.test/v1' })
            : new OpenAIDriver({ apiKey: 'test' });
    const prepared = await driver.prepareIndexedTextRequest({
        selection,
        runtime,
        options: { model: 'gpt-4o-mini-2024-07-18' },
        stream: false,
    });
    const measurement = {
        input_tokens: 20,
        method: 'estimated' as const,
        tokenizer: 'fixture:bpe',
        tokenizer_version: '1',
        adapter: prepared.receipt.target.protocol,
        adapter_version: prepared.receipt.target.adapter_version,
        source_fingerprint: await indexedProcessingContextFingerprint({
            ...selection,
            completeness: 'active_processing_dependencies_verified',
        }),
        target_model: prepared.receipt.target.model,
        measured_at: at,
    };
    const record = parseConversationPreparedRequestRecord({
        source: selection.source,
        runtime,
        request_receipt: { ...prepared.receipt, measurement },
        generation_id: await deriveConversationId('generation', runtime.request_id, runtime.attempt_id),
        response_turn_id: await deriveConversationId('turn', runtime.response_operation_id, 'response', '0'),
        indexed_source: {
            version: 1,
            validator_profile: INDEXED_PROCESSED_INPUT_PREPARED_VALIDATOR_PROFILE,
            root: staged.locator,
            context_revision: selection.context.revision,
            processing_input: {
                version: 1,
                activation_operation_id: runtime.input_operation_id,
                original_source: { conversation_id: document.id, revision: 0 },
                original_root: staged.locator,
                accepted_input_revision: 1,
                accepted_input_receipt_fingerprint: 'sha256:accepted',
                settled_source: { conversation_id: document.id, revision: 3 },
                settled_root: staged.locator,
                coverage: {
                    operation_id: 'operation:coverage',
                    expected_revision: 3,
                    target_fingerprint: await fingerprintJson(prepared.receipt.target),
                    measured_input_tokens: 20,
                    tokenizer_id: 'fixture:bpe:1',
                    measurement_fingerprint: await fingerprintJson(measurement),
                    recorded_at: at,
                },
            },
        },
    });
    return { record, selection, receipt: prepared.receipt };
}

describe('indexed measured native receipt integrity', () => {
    it.each(['chat', 'responses'] as const)(
        'retains exact %s native receipt with its separate measured coverage',
        async (protocol) => {
            const { record, selection, receipt } = await fixture(protocol);
            expect(await indexedPreparedReceiptMatches(record, selection, receipt)).toBe(true);
            const oldProfile = parseConversationPreparedRequestRecord({
                ...record,
                indexed_source: {
                    version: 1,
                    validator_profile: 'llumiverse.conversation/indexed-selected-text/2026-10-03.v1',
                    root: record.indexed_source?.root,
                    context_revision: selection.context.revision,
                },
            });
            expect(await indexedPreparedReceiptMatches(oldProfile, selection, receipt)).toBe(false);
            for (const field of ['count', 'target', 'source', 'native', 'coverage'] as const) {
                const changed = structuredClone(record);
                const measured = changed.request_receipt.measurement;
                const witness = changed.indexed_source?.processing_input;
                if (!measured || !witness) throw new Error('Expected measured witness');
                if (field === 'count') measured.input_tokens++;
                if (field === 'target') measured.target_model = 'foreign-model';
                if (field === 'source') measured.source_fingerprint = 'sha256:foreign';
                if (field === 'native') changed.request_receipt.request_fingerprint = 'sha256:foreign-native';
                if (field === 'coverage') witness.coverage.target_fingerprint = 'sha256:foreign-target';
                expect(await indexedPreparedReceiptMatches(changed, selection, receipt), field).toBe(false);
            }
            const changedSelection = structuredClone(selection);
            const turn = changedSelection.turns[0];
            const block = turn?.selected_blocks[0];
            if (block?.type !== 'text') throw new Error('Expected selected text');
            block.text = 'foreign selected bytes';
            expect(await indexedPreparedReceiptMatches(record, changedSelection, receipt)).toBe(false);
        },
    );
});

describe('generic measured native prepared profile', () => {
    it.each(['chat', 'responses'] as const)(
        'binds %s count to the accepted input without borrowing an epoch',
        async (protocol) => {
            const { record: ordinary, selection, receipt } = await fixture(protocol);
            const epoch = ordinary.indexed_source?.processing_input;
            if (!epoch) throw new Error('Expected independent measured receipt fixture');
            const record = parseConversationPreparedRequestRecord({
                ...ordinary,
                indexed_source: {
                    version: 1,
                    validator_profile: INDEXED_MEASURED_NATIVE_PREPARED_VALIDATOR_PROFILE,
                    root: selection.root,
                    context_revision: selection.context.revision,
                    native_measurement: {
                        version: 1,
                        accepted_source: { conversation_id: selection.source.conversation_id, revision: 1 },
                        accepted_root: selection.root,
                        accepted_operation_id: ordinary.runtime.materialized_input?.operation_id,
                        runtime_input_operation_id: ordinary.runtime.input_operation_id,
                        accepted_receipt_fingerprint: epoch.accepted_input_receipt_fingerprint,
                        settled_source: epoch.settled_source,
                        settled_root: epoch.settled_root,
                        coverage: epoch.coverage,
                    },
                },
            });
            expect(await indexedPreparedReceiptMatches(record, selection, receipt)).toBe(true);
            for (const mutation of ['operation', 'accepted_revision', 'coverage', 'count', 'native'] as const) {
                const changed = structuredClone(record);
                const witness = changed.indexed_source?.native_measurement;
                const measurement = changed.request_receipt.measurement;
                if (!witness || !measurement) throw new Error('Expected measured native witness');
                if (mutation === 'operation') witness.accepted_operation_id = 'operation:unrelated';
                if (mutation === 'accepted_revision') witness.accepted_source.revision++;
                if (mutation === 'coverage') witness.coverage.expected_revision++;
                if (mutation === 'count') measurement.input_tokens++;
                if (mutation === 'native') changed.request_receipt.request_fingerprint = 'sha256:another-native-body';
                expect(await indexedPreparedReceiptMatches(changed, selection, receipt), mutation).toBe(false);
            }
            expect(() =>
                parseConversationPreparedRequestRecord({
                    ...record,
                    indexed_source: { ...record.indexed_source, processing_input: epoch },
                }),
            ).toThrow();
        },
    );
});
