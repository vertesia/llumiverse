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
 * New inline definitions are written while claudeInlineToolWrites is enabled (the default).
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
import { isClaudeCompactionBlock, isClaudeCompactionSummary } from './claude-compaction.js';

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

/**
 * Whether a request may start defining tools by value. A release that reads inline definitions must be deployed
 * wherever conversations are resumed before this is on: an earlier release sends a stored `tool_definition` block
 * under the by-reference beta, which the API rejects. When off, only a conversation that already holds inline
 * definitions keeps defining tools by value; every other one changes tools by reference.
 */
export const claudeInlineToolWrites = { enabled: true };

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

/**
 * Whether this request keeps the conversation's tool state and tool-change blocks. A conversation with a signed
 * compaction `tool_changes` field keeps them on any model: the block is replayed with the inline-tools beta anyway
 * (see usesClaudeInlineTools), and stripping the later changes would leave the tools the block signed in effect.
 */
export function tracksClaudeToolChanges(model: string, provider: string, conversation: unknown): boolean {
    if (canUseClaudeToolChanges(model, provider)) return true;
    const messages = (conversation as { messages?: unknown } | null | undefined)?.messages;
    return Array.isArray(messages) && containsSignedClaudeToolChanges(messages);
}

/** Whether the platform accepts tools defined by value. The beta covers the models of tool changes. */
export function canUseClaudeInlineTools(model: string, provider: string): boolean {
    return INLINE_TOOL_PROVIDERS.has(provider) && supportsMidConversationToolChanges(model);
}

/**
 * Whether this request defines new and changed tools by value (see claudeInlineToolWrites). Always true for a
 * conversation with a signed compaction `tool_changes` field: that block needs the inline-tools beta on every target
 * and cannot be rewritten, so the rest of the conversation is replayed as stored rather than converted to references,
 * which would lose a definition made after the compaction to the one the block signed.
 */
export function usesClaudeInlineTools(model: string, provider: string, conversation: unknown): boolean {
    const messages = (conversation as { messages?: unknown } | null | undefined)?.messages;
    if (Array.isArray(messages) && containsSignedClaudeToolChanges(messages)) return true;
    if (!canUseClaudeInlineTools(model, provider)) return false;
    return claudeInlineToolWrites.enabled || inlineToolMap(getClaudeToolState(conversation)).size > 0;
}

export function getClaudeToolState(conversation: unknown): ClaudeToolState | undefined {
    return (getConversationMeta(conversation) as ClaudeConversationMeta).claude_tools;
}

/** The stored inline definitions by name. A Map, so a tool named like an Object property (`constructor`) is not found on the prototype. */
function inlineToolMap(state: ClaudeToolState | undefined): Map<string, Tool> {
    return new Map(state?.inline ? Object.entries(state.inline) : []);
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
    if (!state.inline) return state;
    const declared = [...state.declared];
    const declaredIndex = new Map(declared.map((tool, index) => [tool.definition.name, index]));
    for (const definition of inlineToolMap(state).values()) {
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
    standbyTools?: Tool[],
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
        // By reference, standby tools are declared deferred after the loaded ones, so offering one later adds a
        // `tool_addition` block instead of changing `tools` (which rewrites the whole cached prefix). Inline
        // tools never change `tools`, so they need no standby declaration.
        const standby = inlineTools ? [] : standbyDeclarations(standbyTools, requested);
        return setClaudeToolState(
            { ...conversation, messages: stripClaudeToolChanges(conversation.messages) },
            {
                declared: [
                    ...requested.map((definition) => ({ definition, deferred: false })),
                    ...standby.map((definition) => ({ definition, deferred: true })),
                ],
                visible: requested.map((tool) => tool.name),
            },
        );
    }

    const declared = [...state.declared];
    const declaredIndex = new Map(declared.map((tool, index) => [tool.definition.name, index]));
    const inline = inlineToolMap(state);
    const visible = new Set(state.visible);
    const additions: BetaRequestToolAdditionBlock[] = [];
    for (const definition of requested) {
        const { name } = definition;
        const index = declaredIndex.get(name);
        const current = inline.get(name) ?? (index === undefined ? undefined : declared[index].definition);
        const changed = current !== undefined && !sameDefinition(current, definition);
        if (inlineTools && (current === undefined || changed)) {
            // A new tool, or a new definition that replaces the earlier one from here on: `tools` stays as sent.
            inline.set(name, definition);
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
            const inlineDefinition = inline.get(name);
            additions.push(inlineDefinition ? definitionAddition(inlineDefinition) : referenceAddition(name));
        }
    }
    const requestedNames = new Set(requested.map((tool) => tool.name));
    const removals = state.visible
        .filter((name) => !requestedNames.has(name))
        .map((name): BetaRequestToolRemovalBlock => ({ type: 'tool_removal', tool: { type: 'tool_reference', name } }));

    const nextState: ClaudeToolState = {
        declared,
        visible: [...requestedNames],
        // Object.fromEntries defines own properties, so a tool named `__proto__` is stored like any other.
        ...(inline.size > 0 && { inline: Object.fromEntries(inline) }),
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
 * The standby tools to declare deferred with the first declaration: one definition per name (a later duplicate
 * wins), without request-only fields, leaving out the requested tools. Declared only alongside at least one
 * requested tool (the caller returns earlier without one), since the API rejects a `tools` array whose tools are
 * all deferred.
 */
function standbyDeclarations(standbyTools: Tool[] | undefined, requested: Tool[]): Tool[] {
    if (!standbyTools?.length) return [];
    const requestedNames = new Set(requested.map((tool) => tool.name));
    return [...new Map(standbyTools.map((tool) => [tool.name, withoutRequestFields(tool)])).values()].filter(
        (tool) => !requestedNames.has(tool.name),
    );
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

/** The `tool_changes` a compaction block carries: the changes of the range it summarizes, sent back unchanged. */
function compactionToolChanges(block: unknown): unknown[] {
    if (!isClaudeCompactionBlock(block)) return [];
    const changes = (block as { tool_changes?: unknown }).tool_changes;
    return Array.isArray(changes) ? changes : [];
}

/**
 * Whether the messages change tools, in a tool-change block or in the `tool_changes` of a compaction block. Either
 * needs a tool-change beta to be replayed.
 */
export function containsClaudeToolChanges(messages: MessageParam[]): boolean {
    return messages.some(
        (message) =>
            Array.isArray(message.content) &&
            message.content.some((block) => isClaudeToolChangeBlock(block) || hasSignedToolChanges(block)),
    );
}

/**
 * Whether a compaction block carries a `tool_changes` field, even an empty one. The API returns that field when the
 * compaction request sent the inline-tools beta, and the block must be replayed with that beta on every request:
 * https://platform.claude.com/docs/en/build-with-claude/compaction-thinking-blocks#change-the-system-prompt-or-tools
 */
function hasSignedToolChanges(block: unknown): boolean {
    return isClaudeCompactionBlock(block) && Array.isArray((block as { tool_changes?: unknown }).tool_changes);
}

/** Whether a compaction block in the messages carries a signed `tool_changes` field (see hasSignedToolChanges). */
export function containsSignedClaudeToolChanges(messages: MessageParam[]): boolean {
    return messages.some((message) => Array.isArray(message.content) && message.content.some(hasSignedToolChanges));
}

/**
 * Whether the messages need the inline-tools beta: a tool defined by value in a tool-change block, or a compaction
 * block whose `tool_changes` field is replayed (see hasSignedToolChanges), whatever the field holds.
 */
export function needsClaudeInlineToolsBeta(messages: MessageParam[]): boolean {
    return messages.some(
        (message) =>
            Array.isArray(message.content) &&
            message.content.some((block) => isClaudeToolDefinitionBlock(block) || hasSignedToolChanges(block)),
    );
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
    const loaded = state.declared.filter((tool) => !tool.deferred).map((tool) => tool.definition.name);
    let visible = new Set(loaded);
    const result: MessageParam[] = [];
    for (const message of messages) {
        if (!Array.isArray(message.content)) {
            result.push(message);
            continue;
        }
        const content: unknown[] = [];
        for (const block of message.content as unknown[]) {
            const name = claudeToolChangeName(block);
            if (name !== undefined) {
                if (applyToolChange(visible, block)) {
                    content.push(isClaudeToolDefinitionBlock(block) ? referenceAddition(name) : block);
                }
                continue;
            }
            if (isClaudeCompactionSummary(block)) {
                // A compaction block is sent back unchanged. Its tool changes take the request's tools to the tool
                // set at the end of the range it replaces, so later changes apply to that set. The API ignores a
                // failed compaction (no summary), so the changes before it still stand.
                visible = new Set(loaded);
                for (const change of compactionToolChanges(block)) applyToolChange(visible, change);
            }
            content.push(block);
        }
        // Keep untouched messages as they were, so the rest of the history is replayed byte for byte.
        if (content.length === message.content.length && content.every((block, i) => block === message.content[i])) {
            result.push(message);
        } else if (content.length > 0) {
            result.push({ ...message, content } as MessageParam);
        }
    }
    return result;
}

/** Apply a tool change to the callable names. False for an addition of a tool that is already callable. */
function applyToolChange(visible: Set<string>, change: unknown): boolean {
    const name = claudeToolChangeName(change);
    if (name === undefined) return true;
    if ((change as { type: string }).type === 'tool_removal') {
        visible.delete(name);
        return true;
    }
    if (visible.has(name)) return false;
    visible.add(name);
    return true;
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
