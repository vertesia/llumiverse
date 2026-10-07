import type Anthropic from '@anthropic-ai/sdk';
import {
    type ConversationAcceptedOutputFragment,
    type ConversationPreparedRequestRecord,
    type DecodedConversationResponse,
    type IndexedConversationSelectedContext,
    parseConversationPreparedRequestRecord,
    preflightJsonInput,
    type ResolvedConversationRuntimeContext,
} from '@llumiverse/conversation';
import {
    IndexedConversationSelectedContextSchema,
    JsonObjectSchema,
    ResolvedConversationRuntimeContextSchema,
} from '@llumiverse/conversation/schemas';
import {
    type CanonicalExecutionEventStream,
    type CanonicalHostCapabilities,
    type CanonicalStreamOpenOptions,
    canonicalToolSelectionPolicy,
    type ExecutionOptions,
    ownCanonicalHostCapabilities,
} from '@llumiverse/core';
import { hydrateCanonicalSelectedMediaAssets } from '../conversation/canonical-host-images.js';
import {
    canonicalToolSelectionTargetOptions,
    createRequestReceiptFromSelectedSource,
    providerJsonValue,
} from '../conversation/canonical-runtime.js';
import { indexedPreparedReceiptMatches } from '../conversation/indexed-prepared-receipt.js';
import { selectedWorkingSetSource } from '../conversation/selected-working-set-source.js';
import {
    normalizeDecodedStructuredOutputForSchema,
    rejectDecodedStructuredOutput,
} from '../conversation/structured-output.js';
import {
    getClaudePayload,
    projectClaudeContextResultSchema,
    projectClaudeConversation,
    streamPreparedClaudeNativeEvents,
} from '../shared/claude-messages.js';
import {
    CLAUDE_MESSAGES_ADAPTER_VERSION,
    CLAUDE_MESSAGES_PROTOCOL,
    compileClaudeIndexedSelection,
    decodeClaudeCanonicalResponse,
} from '../shared/claude-messages-conversation-adapter.js';

function ownedOptions(input: ExecutionOptions): ExecutionOptions {
    if ('resolve_canonical_asset' in input) throw new TypeError('Indexed Claude resolver must be a host capability');
    const keys = [
        'model',
        'model_options',
        'result_schema',
        'prompt_cache_key',
        'stripImagesAfterTurns',
        'stripHeartbeatsAfterTurns',
        'stripTextMaxTokens',
        'httpTimeout',
        'prompt_cache_schema_suffix',
    ] as const;
    const result = Object.fromEntries(
        keys.flatMap((key) => {
            const field = Object.getOwnPropertyDescriptor(input, key);
            if (field && !Object.hasOwn(field, 'value'))
                throw new TypeError('Indexed Claude options must be owned values');
            return field?.value === undefined ? [] : [[key, field.value]];
        }),
    );
    if (!preflightJsonInput(result).success) throw new TypeError('Indexed Claude controls are not bounded JSON');
    if (typeof result.model !== 'string') throw new TypeError('Indexed Claude model is missing');
    // Indexed selection lacks the materialized lifetime generation counter used by legacy age rewriting.
    // Never infer that counter from selected or live turn counts.
    if (
        result.stripImagesAfterTurns !== undefined ||
        result.stripHeartbeatsAfterTurns !== undefined ||
        result.stripTextMaxTokens !== undefined
    )
        throw new TypeError('Indexed Claude age rewriting requires an authenticated lifetime counter');
    return structuredClone(result) as ExecutionOptions;
}

export async function prepareAnthropicIndexedRequest(
    input: {
        selection: IndexedConversationSelectedContext;
        runtime: ResolvedConversationRuntimeContext;
        options: ExecutionOptions;
        stream: boolean;
        signal?: AbortSignal;
    },
    hostCapabilities?: CanonicalHostCapabilities,
) {
    const ownedHost = ownCanonicalHostCapabilities(hostCapabilities);
    if (!preflightJsonInput({ selection: input.selection, runtime: input.runtime }).success)
        throw new TypeError('Indexed Claude source/runtime is not bounded JSON');
    const selection = IndexedConversationSelectedContextSchema.parse(structuredClone(input.selection));
    const runtime = ResolvedConversationRuntimeContextSchema.parse(structuredClone(input.runtime));
    const options = ownedOptions(input.options);
    const signal = input.signal;
    const selectedIds = new Set<string>();
    for (const projection of [
        ...selection.turns,
        ...(selection.replacement_turns ?? []).map((value) => value.projection),
    ])
        for (const block of projection.selected_blocks)
            for (const nested of block.type === 'tool_result' ? block.content : [block])
                if (nested.type === 'image' || nested.type === 'document') selectedIds.add(nested.asset_id);
    const assets = await hydrateCanonicalSelectedMediaAssets({
        label: 'Indexed Claude',
        assets: selection.assets,
        selected_ids: selectedIds,
        media_kinds: ['image', 'document'],
        resolve_asset: ownedHost?.resolve_canonical_asset,
        signal,
        hydrated: new Map(),
        native_external: (asset) => asset.storage.type === 'external' && asset.storage.resolver === 'anthropic_file',
    });
    signal?.throwIfAborted();
    const compiled = compileClaudeIndexedSelection(
        { ...selection, assets },
        { provider: 'anthropic', model: options.model },
    );
    const tools = selection.context.active_tool_definition_ids.map((id) => {
        const tool = Object.hasOwn(selection.tool_definitions, id) ? selection.tool_definitions[id] : undefined;
        if (!tool) throw new Error(`Indexed Claude tool ${id} is unavailable`);
        return tool;
    });
    const conversation = projectClaudeContextResultSchema(
        projectClaudeConversation(compiled.conversation, options, 0),
        options,
        tools.length > 0,
    );
    const { payload, requestOptions } = getClaudePayload(
        options,
        conversation,
        'anthropic',
        input.stream ? 'stream' : 'execute',
        undefined,
        tools,
    );
    if (requestOptions?.headers !== undefined)
        throw new TypeError('Indexed Claude count profile does not support beta transport headers');
    const nativeRequest = JsonObjectSchema.parse(providerJsonValue(structuredClone(payload)));
    const targetOptions = canonicalToolSelectionTargetOptions(undefined, canonicalToolSelectionPolicy(options));
    const receipt = await createRequestReceiptFromSelectedSource(
        {
            ...selectedWorkingSetSource(selection),
            id: selection.source.conversation_id,
            revision: selection.source.revision,
            ...(selection.source_tail_turn_id === undefined
                ? {}
                : { source_tail_turn_id: selection.source_tail_turn_id }),
        },
        runtime,
        {
            provider: 'anthropic',
            protocol: CLAUDE_MESSAGES_PROTOCOL,
            model: options.model,
            adapter_version: CLAUDE_MESSAGES_ADAPTER_VERSION,
            ...(targetOptions === undefined ? {} : { options: targetOptions }),
        },
        nativeRequest,
        compiled.mappings,
        tools,
    );
    return {
        status: 'awaiting_durable_prepared_record' as const,
        source: selection.source,
        native_conversation: conversation,
        payload,
        native_request: structuredClone(nativeRequest),
        receipt,
    };
}

export async function executeAnthropicIndexedRequest(
    client: Anthropic,
    input: {
        selection: IndexedConversationSelectedContext;
        record: ConversationPreparedRequestRecord;
        options: ExecutionOptions;
        assert_committed: () => Promise<void>;
        signal?: AbortSignal;
    },
    hostCapabilities?: CanonicalHostCapabilities,
    transportOptions?: { signal?: AbortSignal; timeout?: number },
): Promise<DecodedConversationResponse> {
    const host = ownCanonicalHostCapabilities(hostCapabilities);
    const selection = IndexedConversationSelectedContextSchema.parse(structuredClone(input.selection));
    const record = parseConversationPreparedRequestRecord(structuredClone(input.record));
    const options = ownedOptions(input.options);
    const ownedTransport = transportOptions === undefined ? undefined : { ...transportOptions };
    const assertCommitted = input.assert_committed;
    const signal = input.signal;
    const prepared = await prepareAnthropicIndexedRequest(
        { selection, runtime: record.runtime, options, stream: false, signal },
        host,
    );
    if (!(await indexedPreparedReceiptMatches(record, selection, prepared.receipt)))
        throw new Error('Indexed Claude body differs from its durable prepared receipt');
    signal?.throwIfAborted();
    await assertCommitted();
    signal?.throwIfAborted();
    // Claude completion compilation uses stream:true. Consume the final message without
    // changing the native body already counted and bound by the durable receipt.
    const response = await client.messages.stream(prepared.payload, { ...ownedTransport, signal }).finalMessage();
    const tools = selection.context.active_tool_definition_ids.map((id) => selection.tool_definitions[id]);
    const responseContext = {
        runtime: record.runtime,
        receipt: record.request_receipt,
        tool_definitions: tools,
        provider: 'anthropic',
        requested_model: options.model,
        generation_id: record.generation_id,
        response_turn_id: record.response_turn_id,
    };
    const rawDecoded = await decodeClaudeCanonicalResponse(response, responseContext);
    const normalized =
        !response.content.some((block) => block.type === 'tool_use') && options.result_schema
            ? normalizeDecodedStructuredOutputForSchema(rawDecoded, options.result_schema)
            : undefined;
    const decoded =
        normalized?.status === 'valid'
            ? await decodeClaudeCanonicalResponse(response, responseContext, normalized.structured_output)
            : rawDecoded;
    return normalized?.status === 'invalid' ? rejectDecodedStructuredOutput(decoded, normalized.error) : decoded;
}

/** Actual Claude native SSE, retaining exactly the counted body and shared draft reconciliation. */
export async function streamAnthropicIndexedRequest(
    client: Anthropic,
    input: {
        selection: IndexedConversationSelectedContext;
        record: ConversationPreparedRequestRecord;
        options: ExecutionOptions;
        assert_committed: () => Promise<void>;
        accept_output(decoded: DecodedConversationResponse): Promise<ConversationAcceptedOutputFragment>;
        open: CanonicalStreamOpenOptions;
        signal?: AbortSignal;
    },
    hostCapabilities?: CanonicalHostCapabilities,
    transportOptions?: { signal?: AbortSignal; timeout?: number },
): Promise<CanonicalExecutionEventStream> {
    const host = ownCanonicalHostCapabilities(hostCapabilities);
    const selection = IndexedConversationSelectedContextSchema.parse(structuredClone(input.selection));
    const record = parseConversationPreparedRequestRecord(structuredClone(input.record));
    const options = ownedOptions(input.options);
    const assertCommitted = input.assert_committed;
    const acceptOutput = input.accept_output;
    const open = { ...input.open };
    const signal = input.signal;
    const prepared = await prepareAnthropicIndexedRequest(
        { selection, runtime: record.runtime, options, stream: true, signal },
        host,
    );
    if (!(await indexedPreparedReceiptMatches(record, selection, prepared.receipt)))
        throw new TypeError('Indexed Claude streaming body differs from its durable prepared receipt');
    signal?.throwIfAborted();
    await assertCommitted();
    signal?.throwIfAborted();
    if (prepared.payload.stream !== true)
        throw new TypeError('Indexed Claude stream requires an authoritative streaming body');
    const decoding: Parameters<typeof decodeClaudeCanonicalResponse>[1] = {
        runtime: record.runtime,
        receipt: record.request_receipt,
        tool_definitions: selection.context.active_tool_definition_ids.map((id) => {
            const definition = selection.tool_definitions[id];
            if (!definition) throw new TypeError('Indexed Claude selected tool definition is unavailable');
            return definition;
        }),
        provider: 'anthropic',
        requested_model: options.model,
        generation_id: record.generation_id,
        response_turn_id: record.response_turn_id,
    };
    return streamPreparedClaudeNativeEvents(client, {
        prepared: decoding,
        payload: { ...prepared.payload, stream: true },
        options,
        open,
        provider: 'anthropic',
        transportOptions: { ...transportOptions, signal },
        beforeTransport: assertCommitted,
        finalize_response: async (decoded) => ({ decoded, accept_output: acceptOutput }),
    });
}
