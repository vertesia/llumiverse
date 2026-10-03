import {
    appendConversationRecords,
    createConversationDocument,
    createGeneratedAgentTurn,
    createTextBlock,
    createUserTurn,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
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
