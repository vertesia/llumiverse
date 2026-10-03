import {
    type ConversationDocument,
    createConversationDocument,
    parseConversationDocument,
    setProcessingPolicy,
} from '@llumiverse/conversation';
import { describe, expect, it } from 'vitest';
import {
    appendOpenAIChatCanonicalResponseWithProcessing,
    OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    type PreparedOpenAIChatConversation,
} from '../openai/openai-chat-conversation-adapter.js';
import {
    appendCanonicalDecodedResponseWithProcessing,
    createExecutedGeneration,
    createRequestReceipt,
    type ResolvedConversationRuntimeContext,
} from './canonical-runtime.js';

const at = '2026-09-30T00:00:00.000Z';
const runtime: ResolvedConversationRuntimeContext = {
    conversation_id: 'response-processing',
    request_id: 'request',
    attempt_id: 'attempt',
    input_operation_id: 'input',
    response_operation_id: 'response',
    recorded_at: at,
    purpose: 'conversation',
};
const target = { provider: 'test', protocol: 'test.generate', model: 'model', adapter_version: '1' };

async function prepared(document: ConversationDocument, selectedTarget = target) {
    const receipt = await createRequestReceipt(document, runtime, selectedTarget, { messages: [] }, [], []);
    return {
        document,
        payload: { messages: [] },
        receipt,
        generation_id: 'generation',
        response_turn_id: 'response-turn',
        diagnostics: [],
    };
}

async function decoded(value: Awaited<ReturnType<typeof prepared>>, selectedTarget = target) {
    return {
        payload_fingerprint: 'sha256:response',
        diagnostics: [],
        generation: await createExecutedGeneration({
            id: value.generation_id,
            runtime,
            receipt: value.receipt,
            provider: selectedTarget.provider,
            protocol: selectedTarget.protocol,
            adapter_version: selectedTarget.adapter_version,
            requested_model: selectedTarget.model,
        }),
        turns: [
            {
                id: value.response_turn_id,
                kind: 'agent' as const,
                authority: 'ordinary' as const,
                blocks: [{ id: 'answer', type: 'text' as const, text: 'done', format: 'plain' as const }],
                status: 'completed' as const,
                timestamps: { recorded_at: at },
                generation_id: value.generation_id,
                provenance: { type: 'generated' as const },
                model_visibility: 'include' as const,
            },
        ],
    };
}

describe('canonical provider response processing handoff', () => {
    it('keeps selection validation before one accepted response and its on-append jobs', async () => {
        const source = createConversationDocument({ id: runtime.conversation_id, created_at: at });
        const enabled = await setProcessingPolicy(source, {
            operation_id: 'policy',
            expected_revision: source.revision,
            recorded_at: at,
            enabled: true,
            processors: [
                {
                    id: 'noop',
                    version: 'v1',
                    scope: 'on_append',
                    config: {},
                    required: true,
                    failure_behavior: 'block',
                },
            ],
        });
        const request = await prepared(enabled.document);
        const response = await decoded(request);
        const options = { operation_id: runtime.response_operation_id, recorded_at: at };
        await expect(
            appendCanonicalDecodedResponseWithProcessing(
                { ...request, response_selection_policy: { mode: 'required', tool_name: 'lookup' } },
                response,
                options,
            ),
        ).rejects.toThrow();
        const accepted = await appendCanonicalDecodedResponseWithProcessing(
            { ...request, response_selection_policy: { mode: 'none' } },
            response,
            options,
        );
        expect(accepted.acceptance.processing.status).toBe('pending');
        expect(accepted.accepted_generation_ids).toEqual([request.generation_id]);
        expect(accepted.accepted_turn_ids).toEqual([request.response_turn_id]);
        expect(Object.values(accepted.document.processing.jobs ?? {})).toHaveLength(1);
        const retry = await appendCanonicalDecodedResponseWithProcessing(
            {
                ...request,
                document: parseConversationDocument(JSON.parse(JSON.stringify(accepted.document))),
                response_selection_policy: { mode: 'none' },
            },
            response,
            options,
        );
        expect(retry.applied).toBe(false);
        expect(retry.document.processing.jobs).toEqual(accepted.document.processing.jobs);
    });

    it('uses the OpenAI Chat finalizer to accept a response with its processing job', async () => {
        const source = createConversationDocument({ id: runtime.conversation_id, created_at: at });
        const enabled = await setProcessingPolicy(source, {
            operation_id: 'policy:chat',
            expected_revision: source.revision,
            recorded_at: at,
            enabled: true,
            processors: [
                {
                    id: 'noop',
                    version: 'v1',
                    scope: 'on_append',
                    config: {},
                    required: true,
                    failure_behavior: 'block',
                },
            ],
        });
        const chatTarget = {
            provider: 'openai',
            protocol: OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            model: 'gpt-test',
            adapter_version: OPENAI_CHAT_COMPLETIONS_ADAPTER_VERSION,
        };
        const request = await prepared(enabled.document, chatTarget);
        const adapterPrepared: PreparedOpenAIChatConversation = {
            ...request,
            payload: { model: chatTarget.model, messages: [], stream: false },
            runtime,
            native_conversation: { messages: [] },
            tool_definitions: [],
            provider: chatTarget.provider,
            requested_model: chatTarget.model,
            prior_native_message_count: 0,
        };
        const response = await decoded(request, chatTarget);
        const document = await appendOpenAIChatCanonicalResponseWithProcessing(adapterPrepared, response);
        expect(document.operation_receipts[runtime.response_operation_id]?.accepted_generation_ids).toEqual([
            request.generation_id,
        ]);
        expect(Object.values(document.processing.jobs ?? {})).toHaveLength(1);
    });
});
