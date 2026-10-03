import { ModelOptionsSchema } from '@llumiverse/common/schemas';
import {
    type ConversationDocument,
    fingerprintJson,
    type ModelTarget,
    parseConversationDocument,
} from '@llumiverse/conversation';
import { ModelTargetSchema } from '@llumiverse/conversation/schemas';
import type { ExecutionOptions } from '@llumiverse/core';
import {
    canonicalConversationTurnNumber,
    providerJsonValue,
    selectedCanonicalTurns,
} from '../conversation/canonical-runtime.js';
import { getClaudePayload, projectClaudeConversation } from './claude-messages.js';
import {
    CLAUDE_MESSAGES_ADAPTER_VERSION,
    CLAUDE_MESSAGES_PROTOCOL,
    compileClaudeMessagesConversation,
} from './claude-messages-conversation-adapter.js';

/**
 * Dry compatible-only Claude projection through the transport's actual adapter and payload builder.
 * resolved_model is a trusted host deployment identity; the resulting body is what the switch plan hashes.
 */
export async function compileClaudeModelSwitchRequest(input: {
    document: ConversationDocument;
    target: ModelTarget;
    resolved_model?: string;
    operation?: 'execute' | 'stream';
}): Promise<ReturnType<typeof providerJsonValue>> {
    const document = parseConversationDocument(input.document);
    const target = ModelTargetSchema.parse(structuredClone(input.target));
    const resolvedModel = input.resolved_model ?? target.model;
    const operation = input.operation ?? 'execute';
    if (target.protocol !== CLAUDE_MESSAGES_PROTOCOL || target.adapter_version !== CLAUDE_MESSAGES_ADAPTER_VERSION)
        throw new Error('Claude Messages model switch target has an unsupported protocol or adapter version');
    const tools = document.context.active_tool_definition_ids.map((id) => {
        const definition = document.tool_definitions[id];
        if (!definition) throw new Error(`Claude Messages model switch tool ${id} is unavailable`);
        return definition;
    });
    const unsupported = new Set([
        'image',
        'document',
        'audio',
        'video',
        'extension',
        'native_replay',
        'external_reference',
    ]);
    for (const turn of selectedCanonicalTurns(document, { allow_interrupted_with_complete_tool_calls: true })) {
        for (const block of turn.blocks) {
            if (unsupported.has(block.type))
                throw new Error(`Claude Messages model switch requires an explicit ${block.type} policy`);
            if (block.type === 'tool_result' && block.content.some((item) => unsupported.has(item.type)))
                throw new Error('Claude Messages model switch requires an explicit nested media policy');
        }
    }
    const modelOptions = target.options === undefined ? undefined : ModelOptionsSchema.parse(target.options);
    if (
        target.options !== undefined &&
        (await fingerprintJson(modelOptions)) !== (await fingerprintJson(target.options))
    )
        throw new Error('Claude Messages model switch target contains unsupported model options');
    const options: ExecutionOptions = {
        model: target.model,
        ...(modelOptions === undefined ? {} : { model_options: modelOptions }),
    };
    const compiled = compileClaudeMessagesConversation(document, target);
    const projected = projectClaudeConversation(
        compiled.conversation,
        options,
        canonicalConversationTurnNumber(document),
    );
    if (
        (await fingerprintJson(providerJsonValue(projected))) !==
        (await fingerprintJson(providerJsonValue(compiled.conversation)))
    )
        throw new Error('Claude Messages model switch requires an explicit history transformation');
    const { payload, requestOptions } = getClaudePayload(
        options,
        projected,
        target.provider,
        operation,
        { model: resolvedModel },
        tools,
    );
    if (requestOptions !== undefined)
        throw new Error('Claude Messages model switch requires an explicit transport-header policy');
    if (
        (await fingerprintJson(providerJsonValue(payload.messages))) !==
            (await fingerprintJson(providerJsonValue(projected.messages))) ||
        (await fingerprintJson(providerJsonValue(payload.system ?? []))) !==
            (await fingerprintJson(providerJsonValue(projected.system ?? [])))
    )
        throw new Error('Claude Messages model switch requires an explicit tool or replay transformation');
    return providerJsonValue(payload);
}
