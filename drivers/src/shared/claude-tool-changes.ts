/**
 * Append-only tool changes for Claude conversations.
 *
 * Claude binds each thinking block, and the prompt cache, to the conversation before it, including the
 * `tools` array. Editing `tools` between turns therefore invalidates earlier thinking and restarts the cache.
 * Instead, the first request's tools stay in `tools` as sent; tools that appear later are appended with
 * `defer_loading: true`, and every change is announced by a `tool_addition` / `tool_removal` block in an
 * appended `role: "system"` message (beta `mid-conversation-tool-changes-2026-07-01`).
 *
 * TODO(inline-tools-2026-09-15): once that beta is available beyond the Claude API, define new and changed tools
 * by value in `tool_addition` (`{ type: 'tool_definition', definition }`) instead of appending them to `tools`.
 * That also removes the in-place replacement of a tool whose definition changes under the same name.
 */

import type {
    AnthropicBeta,
    BetaRequestToolAdditionBlock,
    BetaRequestToolRemovalBlock,
} from '@anthropic-ai/sdk/resources/beta.js';
import type { MessageParam } from '@anthropic-ai/sdk/resources/index.js';
import type { Tool } from '@anthropic-ai/sdk/resources/messages.js';
import {
    type ConversationMeta,
    getConversationMeta,
    Providers,
    setConversationMeta,
    supportsMidConversationToolChanges,
} from '@llumiverse/core';

/**
 * The user turn added after a trailing assistant turn, which Claude 4.6+ would otherwise treat as prefill. A new
 * object each time: cache breakpoints are written onto message blocks.
 */
export function claudeContinueTurn(): MessageParam {
    return { role: 'user', content: [{ type: 'text', text: 'Continue.' }] };
}

export const MID_CONVERSATION_TOOL_CHANGES_BETA: AnthropicBeta = 'mid-conversation-tool-changes-2026-07-01';

/** Providers whose Claude endpoint accepts mid-conversation tool changes (not Microsoft Foundry). */
const TOOL_CHANGE_PROVIDERS = new Set<string>([
    Providers.anthropic,
    Providers.vertexai,
    Providers.bedrock,
    Providers.bedrock_mantle,
]);

type ClaudeToolChangeBlock = BetaRequestToolAdditionBlock | BetaRequestToolRemovalBlock;

interface DeclaredClaudeTool {
    /** The definition as sent in `tools`, without request-only fields (`cache_control`, `defer_loading`). */
    definition: Tool;
    /** True for tools first offered after the first request; sent with `defer_loading: true`. */
    deferred: boolean;
}

/** Tool state stored with the conversation so later requests replay `tools` unchanged. */
export interface ClaudeToolState {
    /** Every tool declared in this conversation, in first-declared order. */
    declared: DeclaredClaudeTool[];
    /** Names the model can currently call. */
    visible: string[];
}

/** Conversation metadata with the Claude tool state (`_llumiverse_meta`). */
export interface ClaudeConversationMeta extends ConversationMeta {
    claude_tools?: ClaudeToolState;
}

export function canUseClaudeToolChanges(model: string, provider: string): boolean {
    return TOOL_CHANGE_PROVIDERS.has(provider) && supportsMidConversationToolChanges(model);
}

export function getClaudeToolState(conversation: unknown): ClaudeToolState | undefined {
    return (getConversationMeta(conversation) as ClaudeConversationMeta).claude_tools;
}

function withoutRequestFields(tool: Tool): Tool {
    const { cache_control: _cacheControl, defer_loading: _deferLoading, ...definition } = tool;
    return definition;
}

/** Whether two tool definitions are equal. Stored definitions keep their key order, so the plain JSON usually decides. */
function sameDefinition(a: Tool, b: Tool): boolean {
    return JSON.stringify(a) === JSON.stringify(b) || canonicalJson(a) === canonicalJson(b);
}

/** JSON with sorted object keys, so equal definitions compare equal regardless of key order. */
function canonicalJson(value: unknown): string {
    if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
    if (value && typeof value === 'object') {
        const entries = Object.entries(value as Record<string, unknown>)
            .filter(([, v]) => v !== undefined)
            .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
        return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${canonicalJson(v)}`).join(',')}}`;
    }
    return JSON.stringify(value);
}

function setClaudeToolState<T extends object>(conversation: T, state: ClaudeToolState): T {
    const meta: ClaudeConversationMeta = { ...getConversationMeta(conversation), claude_tools: state };
    return setConversationMeta(conversation, meta) as T;
}

/**
 * Forget the declared tools after a turn that sent tools the old way (a model or provider without tool
 * changes). Its tools are not tracked, so a later supported turn declares the tools again from scratch.
 */
export function clearClaudeToolState<T extends object>(conversation: T): T {
    if (!getClaudeToolState(conversation)) return conversation;
    // setConversationMeta merges, so drop the field on the object directly.
    const { claude_tools: _tools, ...meta } = getConversationMeta(conversation) as ClaudeConversationMeta;
    return { ...conversation, _llumiverse_meta: meta };
}

/**
 * Record this request's tool set on the conversation. The first request declares the tools as sent. Later
 * requests append new tools as deferred, announce additions and removals in one `role: "system"` message at
 * the end of the conversation, and keep every declared definition in place.
 *
 * Returns the conversation unchanged when the request has no tools; the declared tools are then sent with
 * tool_choice none (see getClaudePayload).
 */
export function applyClaudeToolChanges<T extends { messages: MessageParam[] }>(
    conversation: T,
    requestedTools: Tool[] | undefined,
): T {
    if (!requestedTools || requestedTools.length === 0) return conversation;
    // One definition per name; a later duplicate wins, as it does when replacing a declared tool.
    const requested = [...new Map(requestedTools.map((tool) => [tool.name, withoutRequestFields(tool)])).values()];
    const state = getClaudeToolState(conversation);

    if (!state) {
        // The first declaration loads its tools up front: the API rejects a `tools` array whose tools are all
        // deferred, so tools first declared after earlier turns (which ran without tracked tools) change the
        // tools those turns were bound to once. Tool-change blocks without stored state (e.g. a rebuilt
        // conversation that lost its metadata) may name tools this declaration does not include; drop them.
        return setClaudeToolState(
            { ...conversation, messages: stripClaudeToolChanges(conversation.messages) },
            {
                declared: requested.map((definition) => ({ definition, deferred: false })),
                visible: requested.map((tool) => tool.name),
            },
        );
    }

    const declared = [...state.declared];
    const declaredIndex = new Map(declared.map((tool, index) => [tool.definition.name, index]));
    const visible = new Set(state.visible);
    const additions: string[] = [];
    for (const definition of requested) {
        const index = declaredIndex.get(definition.name);
        if (index === undefined) {
            declaredIndex.set(definition.name, declared.length);
            declared.push({ definition, deferred: true });
        } else if (!sameDefinition(declared[index].definition, definition)) {
            // Same name, new definition (e.g. a refreshed MCP schema). There is no append-only form without the
            // inline-tools beta, so replace it in place: this invalidates earlier thinking and the prompt cache
            // once, and the model sees the current schema.
            declared[index] = { ...declared[index], definition };
        }
        if (!visible.has(definition.name)) additions.push(definition.name);
    }
    const requestedNames = new Set(requested.map((tool) => tool.name));
    const removals = state.visible.filter((name) => !requestedNames.has(name));

    const nextState: ClaudeToolState = { declared, visible: [...requestedNames] };
    if (additions.length === 0 && removals.length === 0) {
        return setClaudeToolState(conversation, nextState);
    }

    const changes: ClaudeToolChangeBlock[] = [
        ...additions.map(
            (name): BetaRequestToolAdditionBlock => ({
                type: 'tool_addition',
                tool: { type: 'tool_reference', name },
            }),
        ),
        ...removals.map(
            (name): BetaRequestToolRemovalBlock => ({
                type: 'tool_removal',
                tool: { type: 'tool_reference', name },
            }),
        ),
    ];
    // The SDK only types tool-change blocks on beta message params; the wire format is the same.
    // A system message must follow a user turn: prepareClaudeConversation stores the continuation turn after a
    // trailing assistant turn, and getClaudePayload places any change that still follows one.
    const changeMessage = { role: 'system', content: changes } as unknown as MessageParam;
    return setClaudeToolState({ ...conversation, messages: [...conversation.messages, changeMessage] }, nextState);
}

/** The `tools` array for a conversation: every declared tool, later ones deferred. */
export function renderClaudeTools(state: ClaudeToolState): Tool[] {
    return state.declared.map(({ definition, deferred }) =>
        deferred ? { ...definition, defer_loading: true } : { ...definition },
    );
}

export function isClaudeToolChangeBlock(block: unknown): boolean {
    const type = (block as { type?: unknown } | null)?.type;
    return type === 'tool_addition' || type === 'tool_removal';
}

/**
 * Remove tool-change blocks for a request that sends `tools` the old way (an unsupported model or provider,
 * e.g. a conversation continued elsewhere). System messages left empty are dropped.
 */
export function stripClaudeToolChanges(messages: MessageParam[]): MessageParam[] {
    if (!messages.some((message) => Array.isArray(message.content) && message.content.some(isClaudeToolChangeBlock))) {
        return messages;
    }
    const result: MessageParam[] = [];
    for (const message of messages) {
        if (!Array.isArray(message.content) || !message.content.some(isClaudeToolChangeBlock)) {
            result.push(message);
            continue;
        }
        const content = message.content.filter((block) => !isClaudeToolChangeBlock(block));
        if (content.length > 0) result.push({ ...message, content });
    }
    return result;
}
