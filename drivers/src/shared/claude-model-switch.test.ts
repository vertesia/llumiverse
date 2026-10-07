import { getEventListeners } from 'node:events';
import {
    appendConversationRecords,
    createConversationDocument,
    createGeneratedAgentTurn,
    createTextBlock,
    createUserTurn,
} from '@llumiverse/conversation';
import { JsonObjectSchema } from '@llumiverse/conversation/schemas';
import { describe, expect, it, vi } from 'vitest';
import { AnthropicDriver } from '../anthropic/index.js';
import { canonicalConversationTurnNumber, providerJsonValue } from '../conversation/canonical-runtime.js';
import { getClaudePayload, projectClaudeConversation } from './claude-messages.js';
import {
    CLAUDE_MESSAGES_ADAPTER_VERSION,
    CLAUDE_MESSAGES_PROTOCOL,
    compileClaudeMessagesConversation,
} from './claude-messages-conversation-adapter.js';
import { compileClaudeModelSwitchRequest } from './claude-model-switch.js';

const at = '2026-10-03T00:00:00.000Z';
const target = {
    provider: 'anthropic',
    protocol: CLAUDE_MESSAGES_PROTOCOL,
    model: 'claude-sonnet-4-20250514',
    adapter_version: CLAUDE_MESSAGES_ADAPTER_VERSION,
};

function document() {
    const source = createConversationDocument({ id: 'switch:claude', created_at: at });
    const turn = createUserTurn({
        id: 'turn:request',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: at },
        provenance: { type: 'received' },
        model_visibility: 'include',
        blocks: [createTextBlock({ id: 'block:request', text: 'Summarize this text', format: 'plain' })],
    });
    return appendConversationRecords(
        source,
        { turns: [turn], context_entries: [{ id: 'entry:request', type: 'source_turn', turn_id: turn.id }] },
        {
            operation_id: 'append:request',
            expected_revision: 0,
            payload_fingerprint: 'sha256:request',
            recorded_at: at,
        },
    ).document;
}

describe('Claude Messages compatible model switch projection', () => {
    it('does not retain an abort listener when prepared publication fails before native transport', async () => {
        const driver = new AnthropicDriver({ apiKey: 'never-network' });
        const controller = new AbortController();
        const before = getEventListeners(controller.signal, 'abort');
        const transport = vi.spyOn(driver.client.messages, 'stream');
        const publication = vi.fn(async () => {
            throw new Error('Prepared publication interrupted');
        });
        const source = document();
        await expect(
            driver.streamCanonicalContextEvents(
                {
                    model: target.model,
                    conversation: source,
                    conversation_runtime: {
                        conversation_id: source.id,
                        request_id: 'request:publication-failed',
                        attempt_id: 'attempt:publication-failed',
                        input_operation_id: 'input:publication-failed',
                        response_operation_id: 'response:publication-failed',
                        recorded_at: at,
                    },
                    on_canonical_request_prepared: publication,
                },
                controller.signal,
                { stream_id: 'stream:publication-failed' },
            ),
        ).rejects.toThrow('Prepared publication interrupted');
        expect(publication).toHaveBeenCalledOnce();
        expect(transport).not.toHaveBeenCalled();
        expect(getEventListeners(controller.signal, 'abort')).toEqual(before);
        controller.abort();
        expect(transport).not.toHaveBeenCalled();
    });

    it('owns configured-driver source and target before its dynamic compiler import', async () => {
        const driver = new AnthropicDriver({ apiKey: 'test-only' });
        const source = document();
        const original = structuredClone(source);
        const mutableTarget = { ...target };
        const pending = driver.projectCanonicalModelSwitchRequest(source, mutableTarget, 'execute');
        const firstBlock = source.turns[0]?.blocks[0];
        if (firstBlock?.type !== 'text') throw new Error('Expected mutable source text block');
        firstBlock.text = 'Mutated after invocation';
        mutableTarget.model = 'claude-other';
        expect(await pending).toEqual({
            status: 'compiled',
            native_request: await compileClaudeModelSwitchRequest({ document: original, target }),
        });
    });

    it('matches the actual native adapter, history projection and transport payload', async () => {
        const source = document();
        const compiled = compileClaudeMessagesConversation(source, target);
        const options = { model: target.model };
        const projected = projectClaudeConversation(
            compiled.conversation,
            options,
            canonicalConversationTurnNumber(source),
        );
        for (const operation of ['execute', 'stream'] as const) {
            const actual = await compileClaudeModelSwitchRequest({ document: source, target, operation });
            const { payload } = getClaudePayload(
                options,
                projected,
                target.provider,
                operation,
                { model: target.model },
                [],
            );
            expect(actual).toEqual(providerJsonValue(payload));
            expect(actual).toHaveProperty('messages', payload.messages);
        }
    });

    it('binds only the real Claude 4.6 trailing-assistant continuation to the dry native body', async () => {
        const source = document();
        const answer = createGeneratedAgentTurn({
            id: 'turn:prior-answer',
            generation_id: 'generation:prior-answer',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            provenance: { type: 'generated' },
            model_visibility: 'include',
            blocks: [createTextBlock({ id: 'block:prior-answer', text: 'Earlier answer.', format: 'plain' })],
        });
        const withAnswer = appendConversationRecords(
            source,
            {
                turns: [answer],
                generations: [
                    {
                        id: 'generation:prior-answer',
                        record_source: 'imported',
                        status: 'completed',
                        timestamps: { recorded_at: at },
                        source: { conversation_id: source.id, revision: source.revision },
                        missing_metadata: ['requested_model', 'provider'],
                    },
                ],
                context_entries: [{ id: 'entry:prior-answer', type: 'source_turn', turn_id: answer.id }],
            },
            {
                operation_id: 'append:prior-answer',
                expected_revision: source.revision,
                payload_fingerprint: 'sha256:prior-answer',
                recorded_at: at,
            },
        ).document;
        const original = structuredClone(withAnswer);
        const nextTarget = { ...target, model: 'claude-sonnet-4-6' };
        const compiled = await compileClaudeModelSwitchRequest({ document: withAnswer, target: nextTarget });
        expect(compiled).toHaveProperty('model', nextTarget.model);
        expect(compiled).toHaveProperty('messages');
        const messages = JsonObjectSchema.parse(compiled).messages;
        if (!Array.isArray(messages)) throw new Error('Claude dry body has no messages');
        expect(messages.at(-1)).toEqual({ role: 'user', content: [{ type: 'text', text: 'Continue.' }] });
        expect(messages.filter((message) => JSON.stringify(message).includes('Continue.'))).toHaveLength(1);
        expect(withAnswer).toEqual(original);
    });

    it('binds the host-resolved deployment model in the exact native body', async () => {
        const first = await compileClaudeModelSwitchRequest({
            document: document(),
            target,
            resolved_model: 'deployment:first',
        });
        const second = await compileClaudeModelSwitchRequest({
            document: document(),
            target,
            resolved_model: 'deployment:second',
        });
        expect(first).toHaveProperty('model', 'deployment:first');
        expect(second).toHaveProperty('model', 'deployment:second');
        expect(first).not.toEqual(second);
    });

    it('rejects incompatible adapter and media without silently rewriting history', async () => {
        await expect(
            compileClaudeModelSwitchRequest({
                document: document(),
                target: { ...target, adapter_version: 'other' },
            }),
        ).rejects.toThrow('unsupported protocol or adapter');
        const source = document();
        const imageTurn = createUserTurn({
            id: 'turn:image',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            provenance: { type: 'received' },
            model_visibility: 'include',
            blocks: [{ id: 'block:image', type: 'image', asset_id: 'asset:image' }],
        });
        const withImage = appendConversationRecords(
            source,
            {
                turns: [imageTurn],
                assets: [
                    {
                        id: 'asset:image',
                        kind: 'image',
                        mime_type: 'image/png',
                        storage: { type: 'inline_base64', data: 'AA==' },
                        provenance: { type: 'received', source_turn_id: imageTurn.id },
                        created_at: at,
                    },
                ],
                context_entries: [{ id: 'entry:image', type: 'source_turn', turn_id: imageTurn.id }],
            },
            {
                operation_id: 'append:image',
                expected_revision: source.revision,
                payload_fingerprint: 'sha256:image',
                recorded_at: at,
            },
        ).document;
        await expect(compileClaudeModelSwitchRequest({ document: withImage, target })).rejects.toThrow(
            'explicit image policy',
        );
    });

    it('requires an explicit native replay policy even for the same Claude adapter', async () => {
        const source = document();
        const agent = createGeneratedAgentTurn({
            id: 'turn:answer',
            generation_id: 'generation:answer',
            authority: 'ordinary',
            status: 'completed',
            timestamps: { recorded_at: at },
            provenance: { type: 'generated' },
            model_visibility: 'include',
            blocks: [
                createTextBlock({ id: 'block:answer', text: 'answer', format: 'plain' }),
                {
                    id: 'block:replay',
                    type: 'native_replay',
                    adapter: CLAUDE_MESSAGES_ADAPTER_VERSION,
                    protocol: CLAUDE_MESSAGES_PROTOCOL,
                    compatibility_scope: {
                        provider: target.provider,
                        protocol: target.protocol,
                        model: target.model,
                        adapter_version: target.adapter_version,
                    },
                    payload: { content: [{ type: 'thinking', thinking: 'private', signature: 'signed' }] },
                    dependencies: {
                        turn_ids: ['turn:answer'],
                        block_ids: ['block:answer'],
                        call_ids: [],
                        request_ids: [],
                    },
                },
            ],
        });
        const withReplay = appendConversationRecords(
            source,
            {
                turns: [agent],
                generations: [
                    {
                        id: 'generation:answer',
                        record_source: 'imported',
                        status: 'completed',
                        timestamps: { recorded_at: at },
                        source: { conversation_id: source.id, revision: source.revision },
                        missing_metadata: ['requested_model', 'provider'],
                    },
                ],
                context_entries: [{ id: 'entry:answer', type: 'source_turn', turn_id: agent.id }],
            },
            {
                operation_id: 'append:answer',
                expected_revision: source.revision,
                payload_fingerprint: 'sha256:answer',
                recorded_at: at,
            },
        ).document;
        await expect(compileClaudeModelSwitchRequest({ document: withReplay, target })).rejects.toThrow(
            'explicit native_replay policy',
        );
    });

    it('discards only foreign replay marked discardable and returns a blocked plan for protected replay', async () => {
        const source = document();
        const original = structuredClone(source);
        const replayTurn = (discardable: boolean) =>
            createGeneratedAgentTurn({
                id: 'turn:foreign-answer',
                generation_id: 'generation:foreign-answer',
                authority: 'ordinary',
                status: 'completed',
                timestamps: { recorded_at: at },
                provenance: { type: 'generated' },
                model_visibility: 'include',
                blocks: [
                    createTextBlock({ id: 'block:foreign-answer', text: 'Imported answer', format: 'plain' }),
                    {
                        id: 'block:foreign-replay',
                        type: 'native_replay',
                        adapter: 'openai.responses.canonical.1',
                        protocol: 'openai.responses',
                        compatibility_scope: {
                            provider: 'openai',
                            protocol: 'openai.responses',
                            model: 'gpt-5.4',
                            adapter_version: 'openai.responses.canonical.1',
                        },
                        payload: { type: 'openai_response', signature: 'protected-native-state' },
                        dependencies: {
                            turn_ids: ['turn:foreign-answer'],
                            block_ids: ['block:foreign-answer'],
                            call_ids: [],
                            request_ids: [],
                        },
                        ...(discardable ? { dependency_policy: 'discard_on_dependency_change' as const } : {}),
                    },
                ],
            });
        const append = (discardable: boolean) =>
            appendConversationRecords(
                source,
                {
                    turns: [replayTurn(discardable)],
                    generations: [
                        {
                            id: 'generation:foreign-answer',
                            record_source: 'imported',
                            status: 'completed',
                            timestamps: { recorded_at: at },
                            source: { conversation_id: source.id, revision: source.revision },
                            missing_metadata: ['requested_model', 'provider'],
                        },
                    ],
                    context_entries: [
                        { id: 'entry:foreign-answer', type: 'source_turn', turn_id: 'turn:foreign-answer' },
                    ],
                },
                {
                    operation_id: 'append:foreign-answer',
                    expected_revision: source.revision,
                    payload_fingerprint: 'sha256:foreign-answer',
                    recorded_at: at,
                },
            ).document;
        const discardable = append(true);
        const retained = structuredClone(discardable);
        const compiled = await compileClaudeModelSwitchRequest({ document: discardable, target });
        expect(JSON.stringify(compiled)).toContain('Imported answer');
        expect(JSON.stringify(compiled)).not.toContain('protected-native-state');
        expect(discardable).toEqual(retained);
        const protectedSource = append(false);
        await expect(compileClaudeModelSwitchRequest({ document: protectedSource, target })).rejects.toThrow(
            'requires an explicit native_replay policy',
        );
        const driver = new AnthropicDriver({ apiKey: 'test-only' });
        await expect(
            driver.projectCanonicalModelSwitchRequest(protectedSource, target, 'execute'),
        ).resolves.toMatchObject({
            status: 'unsupported',
            reason: 'Claude Messages model switch requires an explicit native_replay policy',
        });
        expect(source).toEqual(original);
    });

    it('owns the operation before asynchronous projection work', async () => {
        const input = {
            document: document(),
            target: { ...target, options: { tool_choice: 'required' } },
            operation: 'execute' as 'execute' | 'stream',
        };
        const pending = compileClaudeModelSwitchRequest(input);
        input.operation = 'stream';
        await expect(pending).rejects.toMatchObject({ context: { operation: 'execute' } });
    });
});
