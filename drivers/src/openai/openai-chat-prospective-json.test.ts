import {
    appendConversationRecords,
    type ConversationDocument,
    createConversationDocument,
    createProgramTurn,
    createTextBlock,
    createUserTurn,
    fingerprintJson,
    hashContentBytes,
    JSON_MINIFICATION_PROCESSOR_ID,
    type JsonMinificationProspectiveInput,
    jsonMinificationProcessor,
    type ModelTarget,
    type ProcessingStore,
    parseConversationDocument,
    queueProcessingForExisting,
    type ResolveConversationAsset,
    runProcessingJob,
    setProcessingPolicy,
} from '@llumiverse/conversation';
import { type ExecutionOptions, resolveCanonicalExecutionContextOptions } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIChatCompletionsDriver } from './openai_chat_completions.js';
import {
    compileOpenAIChatCompletionsConversation,
    compileOpenAIChatProspectiveJsonMinification,
    OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    prepareOpenAIChatCanonicalContext,
} from './openai-chat-conversation-adapter.js';

const at = '2026-10-02T00:00:00.000Z';
const png = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg==',
    'base64',
);
const target: ModelTarget = {
    provider: 'openai',
    protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    model: 'gpt-4o',
    adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
};
class Store implements ProcessingStore {
    constructor(public current: ConversationDocument) {}
    async load() {
        return structuredClone(this.current);
    }
    async commit(revision: number, next: ConversationDocument) {
        if (revision !== this.current.revision) return false;
        this.current = parseConversationDocument(next);
        return true;
    }
}
async function fixture(
    duplicate = false,
    historicalNoops = 0,
    withImage = false,
): Promise<JsonMinificationProspectiveInput> {
    const user = createUserTurn({
        id: 'user',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: at },
        model_visibility: 'include',
        provenance: { type: 'received' },
        blocks: [
            createTextBlock({
                id: 'raw',
                text: ' { "n": 900719925474099312345, "n": -0, "s": "\\u0061" } ',
                format: 'plain',
            }),
        ],
    });
    const program = createProgramTurn({
        id: 'program',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: at },
        model_visibility: 'include',
        presentation: 'transcript',
        provenance: { type: 'received' },
        blocks: [createTextBlock({ id: 'program-raw', text: ' [ 1e+999, true ] ', format: 'plain' })],
    });
    const image = {
        id: 'asset:unselected-json-image',
        kind: 'image' as const,
        mime_type: 'image/png',
        storage: {
            type: 'external' as const,
            resolver: 'vertesia.agent_artifact',
            locator: { storage_id: 'owner', artifact_path: 'image' },
        },
        provenance: { type: 'received' as const },
        created_at: at,
        ...(await hashContentBytes(png)),
    };
    const imageTurn = createUserTurn({
        id: 'image-user',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: at },
        model_visibility: 'include',
        provenance: { type: 'received' },
        blocks: [{ id: 'image-block', type: 'image', asset_id: image.id }],
    });
    const base = appendConversationRecords(
        createConversationDocument({ id: 'dry-json', created_at: at }),
        {
            turns: withImage ? [user, program, imageTurn] : [user, program],
            ...(withImage ? { assets: [image] } : {}),
            context_entries: [
                { id: 'first', type: 'source_turn', turn_id: user.id },
                { id: 'second', type: 'source_turn', turn_id: program.id },
                ...(withImage ? [{ id: 'image-entry', type: 'source_turn' as const, turn_id: imageTurn.id }] : []),
            ],
        },
        { operation_id: 'append', expected_revision: 0, recorded_at: at, payload_fingerprint: 'sha256:append' },
    ).document;
    const configured = (
        await setProcessingPolicy(base, {
            operation_id: 'policy',
            expected_revision: base.revision,
            recorded_at: at,
            enabled: true,
            processors: [
                {
                    id: JSON_MINIFICATION_PROCESSOR_ID,
                    version: '1',
                    scope: 'manual',
                    config: {
                        format: 'raw_json_text',
                        minimum_token_reduction: 1,
                        max_code_units: 1048576,
                        max_depth: 128,
                        max_lexical_tokens: 262144,
                    },
                    required: true,
                    failure_behavior: 'block',
                },
            ],
        })
    ).document;
    const targetFingerprint = await fingerprintJson(target);
    const store = new Store(configured);
    let captured: JsonMinificationProspectiveInput | undefined;
    for (let index = 0; index <= historicalNoops; index++) {
        const queued = await queueProcessingForExisting(
            store.current,
            {
                conversation: { conversation_id: store.current.id, revision: store.current.revision },
                expected_context_revision: store.current.context.revision,
                selector: duplicate
                    ? {
                          source: {
                              kind: 'range',
                              range: { from: { kind: 'entry', id: 'first' }, through: { kind: 'entry', id: 'first' } },
                          },
                      }
                    : withImage
                      ? {
                            source: {
                                kind: 'range',
                                range: {
                                    from: { kind: 'entry', id: 'first' },
                                    through: { kind: 'entry', id: 'second' },
                                },
                            },
                        }
                      : { source: { kind: 'all' } },
            },
            {
                operation_id: `queue:${index}`,
                expected_revision: store.current.revision,
                recorded_at: at,
                processor_id: JSON_MINIFICATION_PROCESSOR_ID,
                scope: 'manual',
                target_fingerprint: targetFingerprint,
            },
        );
        store.current = queued.document;
        await runProcessingJob(
            store,
            { resolve: () => jsonMinificationProcessor },
            queued.job_id,
            `attempt:${index}`,
            () => at,
            undefined,
            {
                json_minification: {
                    target_fingerprint: targetFingerprint,
                    identity: {
                        tokenizer: 'not-counted-test',
                        tokenizer_version: '1',
                        adapter: 'openai.chat.completions',
                        adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
                        target_model: target.model,
                        method: 'estimated',
                    },
                    measureProspective: async (input) => {
                        captured = structuredClone(input);
                        return { status: 'unavailable' };
                    },
                },
            },
        );
    }
    if (!captured) throw new Error('Expected real processing runner prospective input');
    return captured;
}
async function rebind(input: JsonMinificationProspectiveInput) {
    input.input_fingerprint = await fingerprintJson({
        processing_job_id: input.processing_job_id,
        resolved_input_fingerprint: input.resolved_input_fingerprint,
        context_fingerprint: input.context_fingerprint,
        candidate: input.candidate,
        target_fingerprint: input.target_fingerprint,
    });
}

describe('OpenAI Chat prospective JSON projection', () => {
    it('reuses actual native compilation without creating accepted records or readiness', async () => {
        const input = await fixture();
        const before = structuredClone(input);
        const result = await compileOpenAIChatProspectiveJsonMinification(input, target);
        expect(result.original.conversation).toEqual(
            compileOpenAIChatCompletionsConversation(input.source, target).conversation,
        );
        expect(result.original.mappings).toEqual(result.replacement.mappings);
        expect(result.replacement.conversation.messages.map((message) => message.content)).toEqual([
            '{"n":900719925474099312345,"n":-0,"s":"\\u0061"}',
            '[1e+999,true]',
        ]);
        expect(result.replacement.conversation.messages.map((message) => message.role)).toEqual(['user', 'user']);
        expect(result.original.projection_fingerprint).toBe(await fingerprintJson(result.original.conversation));
        expect(result.replacement.projection_fingerprint).toBe(await fingerprintJson(result.replacement.conversation));
        expect(input).toEqual(before);
        expect(input.source.processing.coverage).toBeUndefined();
    });
    it('projects selected JSON replacements with cached external image bytes without another asset read', async () => {
        const input = await fixture(false, 0, true);
        const original = structuredClone(input);
        const resolver = vi.fn<ResolveConversationAsset>(async function* () {
            yield png;
        });
        const options = resolveCanonicalExecutionContextOptions({
            model: target.model,
            conversation: input.source,
            conversation_runtime: {
                conversation_id: input.source.id,
                request_id: 'request:prospective-image',
                attempt_id: 'attempt:prospective-image',
                input_operation_id: 'input:prospective-image',
                response_operation_id: 'response:prospective-image',
                recorded_at: at,
            },
        });
        const prepared = await prepareOpenAIChatCanonicalContext({
            options,
            provider: target.provider,
            resolve_asset: resolver,
        });
        expect(resolver).toHaveBeenCalledOnce();
        const direct = await compileOpenAIChatProspectiveJsonMinification(
            input,
            target,
            undefined,
            prepared.native_conversation,
        );
        expect(JSON.stringify(direct.original.conversation)).toContain(png.toString('base64'));
        expect(JSON.stringify(direct.replacement.conversation)).toContain(png.toString('base64'));
        expect(direct.replacement.conversation.messages[0].content).toBe(
            '{"n":900719925474099312345,"n":-0,"s":"\\u0061"}',
        );
        expect(direct.original.mappings).toEqual(direct.replacement.mappings);
        expect(input).toEqual(original);
        expect(resolver).toHaveBeenCalledOnce();

        await expect(
            compileOpenAIChatProspectiveJsonMinification(
                input,
                target,
                undefined,
                structuredClone(prepared.native_conversation),
            ),
        ).rejects.toThrow('introduced an external image without prepared host bytes');
        const stale = structuredClone(input);
        const staleImage = stale.source.assets['asset:unselected-json-image'];
        if (!staleImage) throw new Error('Expected selected external image');
        staleImage.content_hash = `sha256:${'f'.repeat(64)}`;
        await expect(
            compileOpenAIChatProspectiveJsonMinification(stale, target, undefined, prepared.native_conversation),
        ).rejects.toThrow('external image asset asset:unselected-json-image changed during native preparation');
        expect(resolver).toHaveBeenCalledOnce();

        const driver = new OpenAIChatCompletionsDriver({ apiKey: 'test-only', endpoint: 'http://unused.invalid' });
        const projected = vi.fn<NonNullable<ExecutionOptions['on_canonical_request_projected']>>(
            async (projection, compiler) => {
                expect(projection.document).toEqual(input.source);
                const actual = structuredClone(input);
                actual.target_fingerprint = await fingerprintJson(projection.target);
                const resolution = actual.source.processing.resolved_inputs?.[actual.processing_job_id];
                if (!resolution) throw new Error('Expected retained image processing resolution');
                resolution.target_fingerprint = actual.target_fingerprint;
                actual.resolved_input_fingerprint = await fingerprintJson(resolution);
                await rebind(actual);
                const result = await compiler.compileProspective(actual);
                expect(JSON.stringify(result)).toContain(png.toString('base64'));
                throw new Error('Prospective callback reached');
            },
        );
        await expect(
            driver.executeCanonicalContext({ ...options, on_canonical_request_projected: projected }, undefined, {
                resolve_canonical_asset: resolver,
            }),
        ).rejects.toThrow('Prospective callback reached');
        expect(projected).toHaveBeenCalledOnce();
        expect(resolver).toHaveBeenCalledTimes(2);
    });
    it('changes only the selected entry and retains an unselected program turn', async () => {
        const input = await fixture(true);
        const result = await compileOpenAIChatProspectiveJsonMinification(input, target);
        expect(result.replacement.conversation.messages[0].content).toBe(
            '{"n":900719925474099312345,"n":-0,"s":"\\u0061"}',
        );
        expect(result.replacement.conversation.messages[1]).toEqual(result.original.conversation.messages[1]);
    });
    it('reruns the core scanner rather than accepting a self-consistent forged output', async () => {
        const input = await fixture();
        input.candidate.transforms[0].replacement_text = '{"n":1}';
        input.candidate.transforms[0].replacement_text_fingerprint = await fingerprintJson('{"n":1}');
        await rebind(input);
        await expect(compileOpenAIChatProspectiveJsonMinification(input, target)).rejects.toThrow(
            /verified selected source\/output/,
        );
    });
    it('rejects forged source-slice evidence even with a rebound outer fingerprint', async () => {
        const input = await fixture();
        input.candidate.transforms[0].source_slice.block_fingerprint = 'sha256:forged';
        await rebind(input);
        await expect(compileOpenAIChatProspectiveJsonMinification(input, target)).rejects.toThrow(
            /verified selected source\/output/,
        );
    });
    it('rejects a stale source context', async () => {
        const input = await fixture();
        input.source.context.entries.reverse();
        await expect(compileOpenAIChatProspectiveJsonMinification(input, target)).rejects.toThrow(/binding conflicts/);
    });
    it('rejects a changed resolved target', async () => {
        await expect(
            compileOpenAIChatProspectiveJsonMinification(await fixture(), { ...target, model: 'other-model' }),
        ).rejects.toThrow(/binding conflicts/);
    });
    it('rejects a changed retained processor configuration', async () => {
        const input = await fixture();
        const job = Object.values(input.source.processing.jobs ?? {})[0];
        job.configuration.max_depth = 1;
        await expect(compileOpenAIChatProspectiveJsonMinification(input, target)).rejects.toThrow(
            /configuration fingerprint conflicts|exceeds depth/,
        );
    });
    it('rejects missing retained processing resolution', async () => {
        const input = await fixture();
        input.source.processing.resolved_inputs = {};
        await expect(compileOpenAIChatProspectiveJsonMinification(input, target)).rejects.toThrow();
    });
    it('bounds owned input before hashing or compiling', async () => {
        const input = await fixture();
        input.source.metadata = { oversized: 'x'.repeat(16 * 1024 * 1024) };
        await expect(compileOpenAIChatProspectiveJsonMinification(input, target)).rejects.toThrow(
            /16MiB owned input bound/,
        );
    });
    it('bounds a separately supplied target before cloning or hashing', async () => {
        const input = await fixture();
        const oversized = { ...target, model: 'x'.repeat(17 * 1024 * 1024) };
        await expect(compileOpenAIChatProspectiveJsonMinification(input, oversized)).rejects.toThrow(
            '16MiB owned input bound',
        );
    });
    it('owns source, plan, and target before the first hash await', async () => {
        const input = await fixture();
        const selectedTarget = structuredClone(target);
        const pending = compileOpenAIChatProspectiveJsonMinification(input, selectedTarget);
        input.candidate.transforms[0].replacement_text = 'changed';
        input.source.context.entries.reverse();
        selectedTarget.model = 'changed';
        expect((await pending).replacement.conversation.messages[0].content).toBe(
            '{"n":900719925474099312345,"n":-0,"s":"\\u0061"}',
        );
    });
    it('rejects a rebound target with an unsupported adapter version', async () => {
        const input = await fixture();
        const wrong = { ...target, adapter_version: 'unsupported' };
        input.target_fingerprint = await fingerprintJson(wrong);
        const resolution = input.source.processing.resolved_inputs?.[input.processing_job_id];
        if (!resolution) throw new Error('Expected retained current resolution');
        resolution.target_fingerprint = input.target_fingerprint;
        input.resolved_input_fingerprint = await fingerprintJson(resolution);
        await rebind(input);
        await expect(compileOpenAIChatProspectiveJsonMinification(input, wrong)).rejects.toThrow('adapter version');
    });
    it('compiles the exact current job after ten real retained no-op jobs', async () => {
        const input = await fixture(false, 10);
        expect(Object.keys(input.source.processing.jobs ?? {})).toHaveLength(11);
        expect(Object.keys(input.source.processing.completions ?? {})).toHaveLength(10);
        const result = await compileOpenAIChatProspectiveJsonMinification(input, target);
        expect(result.replacement.conversation.messages[0].content).toBe(
            '{"n":900719925474099312345,"n":-0,"s":"\\u0061"}',
        );
    });
    it('rejects a rebound unknown job and a forged resolution fingerprint', async () => {
        const input = await fixture();
        input.processing_job_id = 'unavailable-current-job';
        await rebind(input);
        await expect(compileOpenAIChatProspectiveJsonMinification(input, target)).rejects.toThrow('job/resolution');
        const valid = await fixture();
        valid.resolved_input_fingerprint = 'sha256:forged';
        await rebind(valid);
        await expect(compileOpenAIChatProspectiveJsonMinification(valid, target)).rejects.toThrow('job/resolution');
    });
    it('rejects unsupported protocol and cancellation without invoking a provider', async () => {
        const input = await fixture();
        await expect(
            compileOpenAIChatProspectiveJsonMinification(input, { ...target, protocol: 'other' }),
        ).rejects.toThrow(/OpenAI Chat protocol/);
        const controller = new AbortController();
        controller.abort(new Error('cancelled'));
        await expect(compileOpenAIChatProspectiveJsonMinification(input, target, controller.signal)).rejects.toThrow(
            'cancelled',
        );
    });
});
