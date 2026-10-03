import {
    appendConversationRecords,
    createConversationDocument,
    createTextBlock,
    createUserTurn,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import { canonicalConversationTurnNumber, providerJsonValue } from '../conversation/canonical-runtime.js';
import {
    buildOpenAIChatCompletionsPayload,
    prepareOpenAIChatCompletionsConversation,
    projectOpenAIChatCompletionsHistory,
    toOpenAINonStreamingPayload,
    toOpenAIStreamingPayload,
} from './openai_chat_completions.js';
import {
    compileOpenAIChatCompletionsConversation,
    OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
} from './openai-chat-conversation-adapter.js';
import { compileOpenAIChatModelSwitchRequest } from './openai-chat-model-switch.js';

const at = '2026-10-03T00:00:00.000Z';
const target = {
    provider: 'openai',
    protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    model: 'gpt-4o-2024-08-06',
    adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
};

function document() {
    const source = createConversationDocument({ id: 'switch:openai', created_at: at });
    const turn = createUserTurn({
        id: 'turn:request',
        authority: 'ordinary',
        status: 'completed',
        timestamps: { recorded_at: at },
        provenance: { type: 'received' },
        model_visibility: 'include',
        blocks: [createTextBlock({ id: 'block:request', text: 'Tell me about the source', format: 'plain' })],
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

describe('OpenAI Chat compatible model switch projection', () => {
    it('matches the actual native compiler, builder and SDK body for both transport modes', async () => {
        const source = document();
        const protocolOptions = { modelName: 'deployment:new', extraBody: { extension: { id: 'new' } } };
        const options = { model: target.model };
        const compiled = compileOpenAIChatCompletionsConversation(source, target);
        const projected = projectOpenAIChatCompletionsHistory(
            compiled.conversation,
            options,
            canonicalConversationTurnNumber(source),
        );
        const prepared = prepareOpenAIChatCompletionsConversation(projected, { model: target.model, tools: [] });
        for (const stream of [false, true]) {
            const actual = await compileOpenAIChatModelSwitchRequest({
                document: source,
                target,
                protocol_options: protocolOptions,
                stream,
            });
            const payload = buildOpenAIChatCompletionsPayload(
                prepared,
                options,
                protocolOptions,
                'deployment:new',
                stream,
                target.provider,
                [],
            );
            expect(actual).toEqual(
                providerJsonValue(stream ? toOpenAIStreamingPayload(payload) : toOpenAINonStreamingPayload(payload)),
            );
            expect(actual).toHaveProperty('extension', { id: 'new' });
        }
    });

    it('rejects an unrelated adapter identity before compiling a request', async () => {
        await expect(
            compileOpenAIChatModelSwitchRequest({
                document: document(),
                target: { ...target, adapter_version: 'unrelated' },
                protocol_options: {},
                stream: false,
            }),
        ).rejects.toThrow('unsupported protocol or adapter version');
    });

    it('requires an explicit media policy before switching an image-bearing context', async () => {
        const initial = createConversationDocument({ id: 'switch:image', created_at: at });
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
            initial,
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
                expected_revision: 0,
                payload_fingerprint: 'sha256:image',
                recorded_at: at,
            },
        ).document;
        await expect(
            compileOpenAIChatModelSwitchRequest({ document: withImage, target, protocol_options: {}, stream: false }),
        ).rejects.toThrow('explicit image media policy');
    });

    it('binds trusted deployment resolution into the native body without changing requested target identity', async () => {
        const source = document();
        const first = await compileOpenAIChatModelSwitchRequest({
            document: source,
            target,
            protocol_options: { modelName: 'deployment:first' },
            stream: false,
        });
        const second = await compileOpenAIChatModelSwitchRequest({
            document: source,
            target,
            protocol_options: { modelName: 'deployment:second' },
            stream: false,
        });
        expect(first).toHaveProperty('model', 'deployment:first');
        expect(second).toHaveProperty('model', 'deployment:second');
        expect(first).not.toEqual(second);
        expect(target.model).toBe('gpt-4o-2024-08-06');
    });

    it('rejects extraBody overrides of attested request fields', async () => {
        for (const field of ['model', 'messages', 'tools', 'stream', 'stream_options']) {
            await expect(
                compileOpenAIChatModelSwitchRequest({
                    document: document(),
                    target,
                    protocol_options: { extraBody: { [field]: 'injected' } },
                    stream: false,
                }),
            ).rejects.toThrow(`extraBody cannot override ${field}`);
        }
    });

    it('rejects unknown model options instead of silently stripping them', async () => {
        await expect(
            compileOpenAIChatModelSwitchRequest({
                document: document(),
                target: { ...target, options: { unrecognized_option: true } },
                protocol_options: {},
                stream: false,
            }),
        ).rejects.toThrow('unrecognized_option');
    });
});
