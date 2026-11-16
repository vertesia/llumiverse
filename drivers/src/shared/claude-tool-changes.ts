/**
 * Append-only tool changes for Claude conversations.
 *
 * Claude binds each thinking block, and the prompt cache, to the conversation before it, including the
 * `tools` array. Editing `tools` between turns therefore invalidates earlier thinking and restarts the cache.
 * Instead, the first request's tools stay in `tools` as sent, and every later change is announced by a
 * `tool_addition` / `tool_removal` block in an appended `role: "system"` message. How a new or changed tool
 * reaches the model depends on the platform:
 *
 * - Inline tools (beta `inline-tools-2026-09-15`, see canUseClaudeInlineTools): `tool_addition` carries the new
 *   or changed definition by value (`{ type: 'tool_definition', definition }`) and `tools` never changes.
 * - By reference (beta `mid-conversation-tool-changes-2026-07-01`): new tools are appended to `tools` with
 *   `defer_loading: true` and added by name. A tool whose definition changes under the same name is replaced in
 *   place, which invalidates earlier thinking and the prompt cache once.
 *
 * A conversation can move between the two: a by-reference request moves the inline definitions into `tools` and
 * stores the `tool_definition` blocks as references (see applyClaudeToolChanges).
 *
 * Platforms and models: https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages
 */

import type {
    AnthropicBeta,
    BetaRequestToolAdditionBlock,
    BetaRequestToolRemovalBlock,
    BetaToolUnion,
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
/** Tools defined by value in `tool_addition`. Also covers changes by reference, so it replaces the beta above. */
export const INLINE_TOOLS_BETA: AnthropicBeta = 'inline-tools-2026-09-15';

/**
 * Providers whose Claude endpoint accepts mid-conversation tool changes. Anthropic lists Microsoft Foundry among the
 * platforms of `mid-conversation-tool-changes-2026-07-01`:
 * https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages
 */
const TOOL_CHANGE_PROVIDERS = new Set<string>([
    Providers.anthropic,
    Providers.vertexai,
    Providers.bedrock,
    Providers.bedrock_mantle,
    Providers.azure_foundry,
]);

/**
 * Providers whose Claude endpoint also accepts tools defined by value (`inline-tools-2026-09-15`). Amazon Bedrock and
 * Vertex AI (Claude Models on Google Cloud, Google managed) take tool changes by reference only (same page as
 * above). Anthropic also lists Microsoft Foundry, but not whether that covers Azure-hosted deployments, so Foundry
 * keeps tool changes by reference until that is confirmed.
 */
const INLINE_TOOL_PROVIDERS = new Set<string>([Providers.anthropic]);

type ClaudeToolChangeBlock = BetaRequestToolAdditionBlock | BetaRequestToolRemovalBlock;

interface DeclaredClaudeTool {
    /** The definition as sent in `tools`, without request-only fields (`cache_control`, `defer_loading`). */
    definition: Tool;
    /** True for tools first offered after the first request; sent with `defer_loading: true`. */
    deferred: boolean;
}

/** Tool state stored with the conversation so later requests replay `tools` unchanged. */
export interface ClaudeToolState {
    /** Every tool declared in `tools` in this conversation, in first-declared order. */
    declared: DeclaredClaudeTool[];
    /** Names the model can currently call. */
    visible: string[];
    /**
     * The current definition of each tool defined by value in a `tool_addition` block, by name: tools first offered
     * after the first request, and new definitions of declared tools. Inline tools only.
     */
    inline?: Record<string, Tool>;
}

/** Conversation metadata with the Claude tool state (`_llumiverse_meta`). */
export interface ClaudeConversationMeta extends ConversationMeta {
    claude_tools?: ClaudeToolState;
}

export function canUseClaudeToolChanges(model: string, provider: string): boolean {
    return TOOL_CHANGE_PROVIDERS.has(provider) && supportsMidConversationToolChanges(model);
}

/** Whether new and changed tools can be defined by value. The beta covers the models of tool changes. */
export function canUseClaudeInlineTools(model: string, provider: string): boolean {
    return INLINE_TOOL_PROVIDERS.has(provider) && supportsMidConversationToolChanges(model);
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
 * The state as a by-reference platform sends it: inline definitions move into `tools`, a new tool appended as
 * deferred and a changed one replaced in place. Deterministic, so every by-reference request renders the same
 * `tools`; returns the state itself when nothing is inline.
 */
function withInlineToolsDeclared(state: ClaudeToolState): ClaudeToolState {
    if (!state.inline || Object.keys(state.inline).length === 0) return state;
    const declared = [...state.declared];
    const declaredIndex = new Map(declared.map((tool, index) => [tool.definition.name, index]));
    for (const definition of Object.values(state.inline)) {
        const index = declaredIndex.get(definition.name);
        if (index === undefined) {
            declaredIndex.set(definition.name, declared.length);
            declared.push({ definition, deferred: true });
        } else {
            declared[index] = { ...declared[index], definition };
        }
    }
    return { declared, visible: state.visible };
}

function referenceAddition(name: string): BetaRequestToolAdditionBlock {
    return { type: 'tool_addition', tool: { type: 'tool_reference', name } };
}

function definitionAddition(definition: Tool): BetaRequestToolAdditionBlock {
    // `Tool` is a custom `tools` entry, which the beta `BetaToolUnion` accepts; the SDK types them separately.
    return { type: 'tool_addition', tool: { type: 'tool_definition', definition: definition as BetaToolUnion } };
}

/**
 * Record this request's tool set on the conversation. The first request declares the tools as sent. Later
 * requests announce additions and removals in one `role: "system"` message at the end of the conversation and
 * keep every declared definition in `tools` as sent. With inline tools, a new or changed tool is defined by
 * value in its `tool_addition` block; otherwise it is appended to `tools` as deferred or replaced in place.
 *
 * A request without inline tools first stores the conversation as it sends it (see withInlineToolsDeclared and
 * referenceInlineToolChanges), so a later request on any platform replays that request unchanged.
 *
 * Otherwise returns the conversation unchanged when the request has no tools; the declared tools are then sent
 * with tool_choice none (see getClaudePayload).
 */
export function applyClaudeToolChanges<T extends { messages: MessageParam[] }>(
    storedConversation: T,
    requestedTools: Tool[] | undefined,
    inlineTools = false,
): T {
    const stored = getClaudeToolState(storedConversation);
    const conversation =
        stored?.inline && !inlineTools
            ? setClaudeToolState(
                  { ...storedConversation, messages: referenceInlineToolChanges(storedConversation.messages, stored) },
                  withInlineToolsDeclared(stored),
              )
            : storedConversation;
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
    const inline: Record<string, Tool> = { ...state.inline };
    const visible = new Set(state.visible);
    const additions: BetaRequestToolAdditionBlock[] = [];
    for (const definition of requested) {
        const { name } = definition;
        const index = declaredIndex.get(name);
        const current = inline[name] ?? (index === undefined ? undefined : declared[index].definition);
        const changed = current !== undefined && !sameDefinition(current, definition);
        if (inlineTools && (current === undefined || changed)) {
            // A new tool, or a new definition that replaces the earlier one from here on: `tools` stays as sent.
            inline[name] = definition;
            additions.push(definitionAddition(definition));
            continue;
        }
        if (current === undefined) {
            declaredIndex.set(name, declared.length);
            declared.push({ definition, deferred: true });
        } else if (changed && index !== undefined) {
            // Same name, new definition (e.g. a refreshed MCP schema), on a platform without inline tools. There
            // is no append-only form, so replace it in place: this invalidates earlier thinking and the prompt
            // cache once, and the model sees the current schema.
            declared[index] = { ...declared[index], definition };
        }
        if (!visible.has(name)) {
            // Re-offer an inline tool by value: a tool_addition that defined it may have cancelled out with a
            // removal before it was sent (see mergeToolChangeMessages). An identical definition is a no-op.
            additions.push(name in inline ? definitionAddition(inline[name]) : referenceAddition(name));
        }
    }
    const requestedNames = new Set(requested.map((tool) => tool.name));
    const removals = state.visible
        .filter((name) => !requestedNames.has(name))
        .map((name): BetaRequestToolRemovalBlock => ({ type: 'tool_removal', tool: { type: 'tool_reference', name } }));

    const nextState: ClaudeToolState = {
        declared,
        visible: [...requestedNames],
        ...(Object.keys(inline).length > 0 && { inline }),
    };
    if (additions.length === 0 && removals.length === 0) {
        return setClaudeToolState(conversation, nextState);
    }

    const changes: ClaudeToolChangeBlock[] = [...additions, ...removals];
    // The SDK only types tool-change blocks on beta message params; the wire format is the same.
    // A system message must follow a user turn: prepareClaudeConversation stores the continuation turn after a
    // trailing assistant turn, and getClaudePayload places any change that still follows one.
    const changeMessage = { role: 'system', content: changes } as unknown as MessageParam;
    return setClaudeToolState({ ...conversation, messages: [...conversation.messages, changeMessage] }, nextState);
}

/**
 * The `tools` array for a conversation: every declared tool, later ones deferred. Without inline tools, the
 * inline definitions are sent here too (see withInlineToolsDeclared).
 */
export function renderClaudeTools(state: ClaudeToolState, inlineTools = false): Tool[] {
    const sent = inlineTools ? state : withInlineToolsDeclared(state);
    return sent.declared.map(({ definition, deferred }) =>
        deferred ? { ...definition, defer_loading: true } : { ...definition },
    );
}

export function isClaudeToolChangeBlock(block: unknown): boolean {
    const type = (block as { type?: unknown } | null)?.type;
    return type === 'tool_addition' || type === 'tool_removal';
}

/** Whether a block is a `tool_addition` that defines its tool by value. */
export function isClaudeToolDefinitionBlock(block: unknown): boolean {
    const change = block as { type?: unknown; tool?: { type?: unknown } } | null;
    return change?.type === 'tool_addition' && change.tool?.type === 'tool_definition';
}

/** The name of the tool a tool-change block adds or removes, by reference or by value. */
export function claudeToolChangeName(block: unknown): string | undefined {
    if (!isClaudeToolChangeBlock(block)) return undefined;
    const tool = (block as { tool?: { name?: unknown; definition?: { name?: unknown } } }).tool;
    const name = tool?.name ?? tool?.definition?.name;
    return typeof name === 'string' ? name : undefined;
}

/**
 * The messages as a by-reference platform accepts them: each `tool_definition` addition becomes a reference to
 * the tool, which renderClaudeTools sends in `tools`. A definition that changed a tool the model could already
 * call adds nothing by reference, so it is dropped. System messages left empty are dropped.
 */
export function referenceInlineToolChanges(messages: MessageParam[], state: ClaudeToolState): MessageParam[] {
    if (
        !messages.some((message) => Array.isArray(message.content) && message.content.some(isClaudeToolDefinitionBlock))
    ) {
        return messages;
    }
    // Stored tool changes all follow the first declaration (applyClaudeToolChanges strips older ones), whose tools
    // are the ones loaded up front.
    const visible = new Set(state.declared.filter((tool) => !tool.deferred).map((tool) => tool.definition.name));
    const result: MessageParam[] = [];
    for (const message of messages) {
        if (!Array.isArray(message.content) || !message.content.some(isClaudeToolChangeBlock)) {
            result.push(message);
            continue;
        }
        const content: unknown[] = [];
        for (const block of message.content as unknown[]) {
            const name = claudeToolChangeName(block);
            if (name === undefined) {
                content.push(block);
            } else if ((block as { type: string }).type === 'tool_removal') {
                visible.delete(name);
                content.push(block);
            } else if (!visible.has(name)) {
                visible.add(name);
                content.push(isClaudeToolDefinitionBlock(block) ? referenceAddition(name) : block);
            }
        }
        if (content.length > 0) result.push({ ...message, content } as MessageParam);
    }
    return result;
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
        // Only tool-change blocks are removed, so the copy stays valid for its role (SDK v1 types one per role).
        if (content.length > 0) result.push({ ...message, content } as MessageParam);
    }
    return result;
}
