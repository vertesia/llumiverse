import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { createRequire, registerHooks } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const provenance = JSON.parse(readFileSync(new URL('./provenance.json', import.meta.url), 'utf8'));
const dependencyRepository = process.argv[2];
if (!dependencyRepository) throw new TypeError('Historical capture requires an explicit local llumiverse checkout');
const installedRequire = createRequire(pathToFileURL(`${dependencyRepository}/conversation/package.json`));
const installedZod = installedRequire.resolve('zod');
const installedZodPackage = JSON.parse(readFileSync(installedRequire.resolve('zod/package.json'), 'utf8'));
if (installedZodPackage.version !== '4.6.5')
    throw new TypeError('Captured historical fixture requires reviewed compatible zod 4.6.5');
const base = new URL('../historical/conversation/src/', import.meta.url);
registerHooks({
    resolve(specifier, context, nextResolve) {
        if (context.parentURL?.startsWith(base.href) && specifier === 'zod')
            return nextResolve(pathToFileURL(installedZod).href, context);
        if (context.parentURL?.startsWith(base.href) && specifier.startsWith('.') && specifier.endsWith('.js')) {
            const source = new URL(`${specifier.slice(0, -3)}.ts`, context.parentURL);
            if (source.href.startsWith(base.href) && existsSync(fileURLToPath(source)))
                return nextResolve(source.href, context);
        }
        return nextResolve(specifier, context);
    },
});
const historical = await import(new URL('index.ts', base).href);
const at = '2026-10-06T00:00:00.000Z';
const pages = new Map();
const records = new Map();
const recordKey = (value) => `${value.kind}:${value.content_hash}`;
const put = (map, key, bytes) => {
    const encoded = Buffer.from(bytes).toString('base64');
    if (map.has(key) && map.get(key) !== encoded) throw new TypeError('Historical fixture immutable write conflicts');
    map.set(key, encoded);
};
const store = {
    async write(bytes, ref) {
        put(pages, ref.content_hash, bytes);
    },
    async read(ref) {
        const encoded = pages.get(ref.content_hash);
        if (!encoded) throw new TypeError('Missing captured page');
        return Buffer.from(encoded, 'base64');
    },
    async writeRecord(value, bytes) {
        put(records, recordKey(value), bytes);
    },
    async readRecord(value) {
        const encoded = records.get(recordKey(value));
        if (!encoded) throw new TypeError('Missing captured record');
        return Buffer.from(encoded, 'base64');
    },
};
const source = historical.createConversationDocument({ id: 'historical:fffdaef:paged', created_at: at });
const turns = Array.from({ length: 130 }, (_, ordinal) =>
    historical.createUserTurn({
        id: `turn:${ordinal}`,
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: at },
        model_visibility: 'include',
        provenance: { type: 'received' },
        blocks: [
            { type: 'text', format: 'plain', id: `block:${ordinal}`, text: `Original historical content ${ordinal}.` },
        ],
    }),
);
const command = {
    expected_revision: 0,
    operation_id: 'append:historical:received',
    payload_fingerprint: 'historical:received:130',
    recorded_at: at,
};
const appended = historical.appendConversationRecords(
    source,
    {
        turns,
        context_entries: turns
            .slice(0, 129)
            .map((turn) => ({ id: `entry:${turn.id}`, type: 'source_turn', turn_id: turn.id })),
    },
    command,
).document;
const captured = await historical.stageIndexedConversationSnapshot(appended, undefined, store);
const deletionCommand = {
    operation_id: 'delete:historical:one',
    source: captured.root.source,
    expected_source_root: captured.locator,
    recorded_at: at,
    dependency_policy: 'reject',
    turn_ids: ['turn:129'],
};
const deleted = await historical.stageIndexedConversationDelete(captured.root, deletionCommand, store);
if (!deleted.locator || !deleted.applied) throw new TypeError('Historical deletion was not actually staged');
const enabledSource = await historical.setProcessingPolicy(
    historical.createConversationDocument({ id: 'historical:fffdaef:pending', created_at: at }),
    {
        expected_revision: 0,
        operation_id: 'policy:historical:text',
        recorded_at: at,
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
const enabledDocument = (
    await historical.appendConversationRecordsWithProcessing(
        enabledSource.document,
        {
            turns: [
                historical.createUserTurn({
                    id: 'pending:user',
                    authority: 'ordinary',
                    status: 'completed',
                    timestamps: { recorded_at: at },
                    model_visibility: 'include',
                    provenance: { type: 'received' },
                    blocks: [{ id: 'pending:text', type: 'text', format: 'plain', text: 'Retained pending input.' }],
                }),
            ],
            context_entries: [{ id: 'pending:entry', type: 'source_turn', turn_id: 'pending:user' }],
        },
        {
            expected_revision: 1,
            operation_id: 'append:historical:pending',
            payload_fingerprint: 'historical:pending',
            recorded_at: at,
        },
    )
).document;
const pending = await historical.stageIndexedConversationSnapshot(enabledDocument, undefined, store);
const jsonEnabledSource = await historical.setProcessingPolicy(
    historical.createConversationDocument({ id: 'historical:fffdaef:pending-json', created_at: at }),
    {
        expected_revision: 0,
        operation_id: 'policy:historical:text',
        recorded_at: at,
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
const jsonEnabledDocument = (
    await historical.appendConversationRecordsWithProcessing(
        jsonEnabledSource.document,
        {
            turns: [
                historical.createUserTurn({
                    id: 'pending-json:user',
                    authority: 'ordinary',
                    status: 'completed',
                    timestamps: { recorded_at: at },
                    model_visibility: 'include',
                    provenance: { type: 'received' },
                    blocks: [
                        {
                            id: 'pending-json:body',
                            type: 'json',
                            value: { retained: 'Original received structured input.' },
                        },
                    ],
                }),
            ],
            context_entries: [{ id: 'pending-json:entry', type: 'source_turn', turn_id: 'pending-json:user' }],
        },
        {
            expected_revision: 1,
            operation_id: 'append:historical:pending-json-json',
            payload_fingerprint: 'historical:pending-json',
            recorded_at: at,
        },
    )
).document;
const pendingJson = await historical.stageIndexedConversationSnapshot(jsonEnabledDocument, undefined, store);
const nativeJsonEnabledSource = await historical.setProcessingPolicy(
    historical.createConversationDocument({ id: 'historical:fffdaef:pending-native-json', created_at: at }),
    {
        expected_revision: 0,
        operation_id: 'policy:historical:text',
        recorded_at: at,
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
const nativeJsonBase = await historical.stageIndexedConversationSnapshot(
    nativeJsonEnabledSource.document,
    undefined,
    store,
);
const nativeJsonInputAppend = {
    expected_revision: 1,
    operation_id: 'append:historical:pending-native-json',
    recorded_at: at,
    records: {
        turns: [
            historical.createUserTurn({
                id: 'pending-native-json:user',
                authority: 'ordinary',
                status: 'completed',
                timestamps: { recorded_at: at },
                model_visibility: 'include',
                provenance: { type: 'received' },
                blocks: [
                    {
                        id: 'pending-native-json:body',
                        type: 'json',
                        value: { retained: 'Original received structured input.' },
                    },
                ],
            }),
        ],
        context_entries: [
            { id: 'pending-native-json:entry', type: 'source_turn', turn_id: 'pending-native-json:user' },
        ],
    },
};
const nativeJsonEnabledDocument = (
    await historical.appendConversationRecordsWithProcessing(
        nativeJsonEnabledSource.document,
        nativeJsonInputAppend.records,
        {
            expected_revision: nativeJsonInputAppend.expected_revision,
            operation_id: nativeJsonInputAppend.operation_id,
            recorded_at: nativeJsonInputAppend.recorded_at,
            payload_fingerprint: await historical.fingerprintJson({
                purpose: 'resume_user',
                input_append: nativeJsonInputAppend,
            }),
        },
    )
).document;
const pendingNativeJson = await historical.stageIndexedConversationSnapshot(
    nativeJsonEnabledDocument,
    undefined,
    store,
);
const importedDocument = historical.appendConversationRecords(
    historical.createConversationDocument({ id: 'historical:fffdaef:imported', created_at: at }),
    {
        turns: [
            {
                id: 'imported:agent',
                kind: 'agent',
                authority: 'ordinary',
                status: 'completed',
                timestamps: { recorded_at: at },
                model_visibility: 'include',
                provenance: { type: 'imported', source: 'openai.responses', missing_metadata: ['generation'] },
                blocks: [
                    {
                        id: 'imported:text',
                        type: 'text',
                        format: 'plain',
                        text: 'Uninvented original imported history.',
                    },
                ],
            },
        ],
        context_entries: [{ id: 'imported:entry', type: 'source_turn', turn_id: 'imported:agent' }],
    },
    {
        expected_revision: 0,
        operation_id: 'append:historical:imported',
        payload_fingerprint: 'historical:imported',
        recorded_at: at,
    },
).document;
const imported = await historical.stageIndexedConversationSnapshot(importedDocument, undefined, store);
const executedSource = historical.createConversationDocument({ id: 'historical:fffdaef:executed', created_at: at });
const generation = {
    id: 'executed:generation',
    record_source: 'executed',
    request_id: 'executed:request',
    attempt_id: 'executed:attempt',
    purpose: 'interaction',
    requested_model: 'fixture:model',
    provider: 'openai',
    protocol: 'openai.responses',
    adapter_version: 'historical-fixture',
    status: 'completed',
    finish_reason: 'stop',
    timestamps: { recorded_at: at, completed_at: at },
    source: { conversation_id: executedSource.id, revision: 0 },
    request_receipt: {
        id: 'executed:request-receipt',
        request_id: 'executed:request',
        attempt_id: 'executed:attempt',
        source: { conversation_id: executedSource.id, revision: 0 },
        context_fingerprint: 'historical:context',
        tool_set_fingerprint: 'historical:tools',
        request_fingerprint: 'historical:request',
        target: {
            provider: 'openai',
            protocol: 'openai.responses',
            model: 'fixture:model',
            adapter_version: 'historical-fixture',
        },
        tool_definition_ids: [],
        asset_versions: [],
        item_mappings: [],
        recorded_at: at,
    },
};
const executedDocument = historical.appendConversationRecords(
    executedSource,
    {
        generations: [generation],
        turns: [
            historical.createGeneratedAgentTurn({
                id: 'executed:agent',
                generation_id: generation.id,
                authority: 'ordinary',
                status: 'completed',
                timestamps: { recorded_at: at, completed_at: at },
                model_visibility: 'include',
                provenance: { type: 'generated' },
                blocks: [
                    {
                        id: 'executed:text',
                        type: 'text',
                        format: 'plain',
                        text: 'Actually accepted historical canonical answer.',
                    },
                ],
            }),
        ],
        context_entries: [{ id: 'executed:entry', type: 'source_turn', turn_id: 'executed:agent' }],
    },
    {
        expected_revision: 0,
        operation_id: 'executed:response',
        payload_fingerprint: 'historical:executed',
        recorded_at: at,
    },
).document;
const executed = await historical.stageIndexedConversationSnapshot(executedDocument, 'executed:response', store);
// Actual old policy transition after original accepted output. Original generation and
// response bytes remain retained; this does not retroactively enqueue append work.
const retainedOutputPolicyCommand = {
    expected_revision: executedDocument.revision,
    operation_id: 'policy:historical:retained-output',
    recorded_at: at,
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
};
const retainedOutputDocument = await historical.setProcessingPolicy(executedDocument, retainedOutputPolicyCommand);
const retainedOutput = await historical.stageIndexedConversationSnapshot(
    retainedOutputDocument.document,
    undefined,
    store,
);

// Imported closed facts preserve their actual archival origin. No generated request or
// execution authority is fabricated; native dispatch eligibility remains separate.
const importedClosedSource = historical.createConversationDocument({
    id: 'historical:fffdaef:imported-closed',
    created_at: at,
});
const importedCall = {
    id: 'imported:call:block',
    type: 'tool_call',
    call_id: 'imported:call',
    tool_name: 'archived_provider_tool',
    executor: 'provider',
    arguments: { type: 'json', value: { query: 'original' } },
};
const importedResult = {
    id: 'imported:result:block',
    type: 'tool_result',
    call_id: 'imported:call',
    status: 'success',
    content: [
        { id: 'imported:result:nested', type: 'text', format: 'plain', text: 'Original archived provider result.' },
    ],
};
const importedClosedRecords = {
    turns: [
        {
            id: 'imported:call:turn',
            kind: 'agent',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            model_visibility: 'include',
            provenance: { type: 'imported', source: 'historical.provider', missing_metadata: ['generation'] },
            blocks: [importedCall],
        },
        {
            id: 'imported:result:turn',
            kind: 'tool',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            model_visibility: 'include',
            provenance: { type: 'received' },
            blocks: [importedResult],
        },
    ],
    execution_receipts: [
        {
            id: 'imported:execution',
            call_id: importedCall.call_id,
            executor: 'provider',
            status: 'success',
            result_turn_id: 'imported:result:turn',
            result_fingerprint: await historical.fingerprintJson(importedResult),
            recorded_at: at,
        },
    ],
};
const importedClosedDocument = historical.appendConversationRecords(importedClosedSource, importedClosedRecords, {
    expected_revision: 0,
    operation_id: 'append:historical:imported-closed',
    payload_fingerprint: await historical.fingerprintJson(importedClosedRecords),
    recorded_at: at,
}).document;
const importedClosed = await historical.stageIndexedConversationSnapshot(importedClosedDocument, undefined, store);
// Imported generation preserves absence of request ID and request receipt.
const importedGenerationSource = historical.createConversationDocument({
    id: 'historical:fffdaef:imported-generation',
    created_at: at,
});
const importedGeneration = {
    id: 'imported:optional-generation',
    record_source: 'imported',
    status: 'completed',
    timestamps: { recorded_at: at },
    source: { conversation_id: importedGenerationSource.id, revision: 0 },
    missing_metadata: ['request_id', 'request_receipt', 'attempt_id'],
};
const importedGenerationRecords = {
    generations: [importedGeneration],
    turns: [
        {
            id: 'imported:generation:turn',
            kind: 'agent',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            model_visibility: 'include',
            provenance: { type: 'imported', source: 'historical.provider' },
            generation_id: importedGeneration.id,
            blocks: [
                {
                    id: 'imported:generation:text',
                    type: 'text',
                    format: 'plain',
                    text: 'Original imported generation with unavailable request metadata.',
                },
            ],
        },
    ],
};
const importedGenerationDocument = historical.appendConversationRecords(
    importedGenerationSource,
    importedGenerationRecords,
    {
        expected_revision: 0,
        operation_id: 'append:historical:imported-generation',
        payload_fingerprint: await historical.fingerprintJson(importedGenerationRecords),
        recorded_at: at,
    },
).document;
const importedGenerationSnapshot = await historical.stageIndexedConversationSnapshot(
    importedGenerationDocument,
    undefined,
    store,
);
const output = {
    version: 1,
    provenance,
    dependencies: { zod: installedZodPackage.version },
    source_commands: {
        append: command,
        deletion: deletionCommand,
        native_json_input_append: nativeJsonInputAppend,
        retained_output_policy: retainedOutputPolicyCommand,
    },
    cases: {
        retained_executed_output: { root: retainedOutput.root, locator: retainedOutput.locator },
        imported_generation_no_request: {
            root: importedGenerationSnapshot.root,
            locator: importedGenerationSnapshot.locator,
        },
        imported_closed: { root: importedClosed.root, locator: importedClosed.locator },
        received: { root: captured.root, locator: captured.locator },
        deleted: { root: deleted.root, locator: deleted.locator },
        pending: { root: pending.root, locator: pending.locator },
        pending_json: { root: pendingJson.root, locator: pendingJson.locator },
        pending_native_json_base: { root: nativeJsonBase.root, locator: nativeJsonBase.locator },
        pending_native_json: { root: pendingNativeJson.root, locator: pendingNativeJson.locator },
        imported_missing_generation: { root: imported.root, locator: imported.locator },
        executed: { root: executed.root, locator: executed.locator },
    },
    pages: Object.fromEntries([...pages].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))),
    records: Object.fromEntries([...records].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))),
};
const bytes = Buffer.from(JSON.stringify(output));
writeFileSync(new URL('./historical-fffdaef-artifacts.json', import.meta.url), bytes);
writeFileSync(
    new URL('./capture-manifest.json', import.meta.url),
    `${JSON.stringify(
        {
            version: 1,
            provenance,
            generator_sha256: createHash('sha256')
                .update(readFileSync(new URL('./capture.mjs', import.meta.url)))
                .digest('hex'),
            command: 'node --experimental-transform-types capture.mjs <llumiverse-checkout>',
            artifact: 'historical-fffdaef-artifacts.json',
            sha256: createHash('sha256').update(bytes).digest('hex'),
            bytes: bytes.length,
        },
        null,
        2,
    )}\n`,
);
