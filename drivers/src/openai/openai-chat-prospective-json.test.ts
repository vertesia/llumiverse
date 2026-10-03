import {
    appendConversationRecords,
    type ConversationDocument,
    createConversationDocument,
    createProgramTurn,
    createTextBlock,
    createUserTurn,
    fingerprintJson,
    JSON_MINIFICATION_PROCESSOR_ID,
    type JsonMinificationProspectiveInput,
    jsonMinificationProcessor,
    type ModelTarget,
    type ProcessingStore,
    parseConversationDocument,
    queueProcessingForExisting,
    runProcessingJob,
    setProcessingPolicy,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import {
    compileOpenAIChatCompletionsConversation,
    compileOpenAIChatProspectiveJsonMinification,
    OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
} from './openai-chat-conversation-adapter.js';

const at = '2026-10-02T00:00:00.000Z';
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
async function fixture(duplicate = false, historicalNoops = 0): Promise<JsonMinificationProspectiveInput> {
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
    const base = appendConversationRecords(
        createConversationDocument({ id: 'dry-json', created_at: at }),
        {
            turns: [user, program],
            context_entries: [
                { id: 'first', type: 'source_turn', turn_id: user.id },
                { id: 'second', type: 'source_turn', turn_id: program.id },
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
