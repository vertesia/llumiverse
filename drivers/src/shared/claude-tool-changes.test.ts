import type { Message, MessageParam, RawMessageStreamEvent, Tool } from '@anthropic-ai/sdk/resources/messages.js';
import type { ExecutionOptions, ToolDefinition } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import {
    type ClaudePrompt,
    executeClaudeCompletion,
    fixOrphanedToolResults,
    fixOrphanedToolUse,
    getClaudePayload,
} from './claude-messages.js';
import {
    applyClaudeToolChanges,
    getClaudeToolState,
    MID_CONVERSATION_TOOL_CHANGES_BETA,
    stripClaudeToolChanges,
} from './claude-tool-changes.js';

const MODEL = 'claude-sonnet-5-5';
const schema = { type: 'object', properties: { q: { type: 'string' } } } as const;
const tool = (name: string, description = `${name} tool`): ToolDefinition => ({
    name,
    description,
    input_schema: schema,
});

function sdkStream(finalMessage: Message) {
    return {
        async *[Symbol.asyncIterator](): AsyncGenerator<RawMessageStreamEvent> {},
        async finalMessage() {
            return finalMessage;
        },
        abort() {},
    };
}

function assistantToolUse(id: string, name: string): Message {
    return {
        id: `msg-${id}`,
        type: 'message',
        role: 'assistant',
        model: MODEL,
        content: [
            { type: 'thinking', thinking: '', signature: `sig-${id}` },
            { type: 'tool_use', id, name, input: { q: 'x' } },
        ],
        stop_reason: 'tool_use',
        stop_sequence: null,
        usage: { input_tokens: 1, output_tokens: 1 },
    } as unknown as Message;
}

const toolResult = (id: string): ClaudePrompt => ({
    messages: [{ role: 'user', content: [{ type: 'tool_result', tool_use_id: id, content: 'ok' }] }],
});

/** Runs one turn and returns the request the client received plus the stored (JSON round-tripped) conversation. */
async function turn(
    prompt: ClaudePrompt,
    tools: ToolDefinition[] | undefined,
    conversation: unknown,
    response: Message,
    provider = 'anthropic',
    model = MODEL,
) {
    const stream = vi.fn((_payload: unknown, _options: unknown) => sdkStream(response));
    const options: ExecutionOptions = { model, tools, conversation };
    const completion = await executeClaudeCompletion(
        { messages: { stream } } as never,
        prompt,
        options,
        undefined,
        provider,
    );
    const [payload, requestOptions] = stream.mock.calls[0] as [
        { tools?: Tool[]; messages: MessageParam[]; tool_choice?: unknown },
        { headers?: Record<string, string> } | undefined,
    ];
    return { payload, requestOptions, conversation: JSON.parse(JSON.stringify(completion.conversation)) };
}

const start = (): ClaudePrompt => ({ messages: [{ role: 'user', content: 'hi' }] });
const firstPrompt: ClaudePrompt = { messages: [{ role: 'user', content: 'Audit the project.' }] };

describe('append-only Claude tool changes', () => {
    it('declares the first tools as sent and replays them unchanged when a tool is added', async () => {
        const t1 = await turn(firstPrompt, [tool('learn_skill')], undefined, assistantToolUse('c1', 'learn_skill'));
        expect(t1.payload.tools).toEqual([tool('learn_skill')]);
        expect(t1.requestOptions).toBeUndefined();
        expect(getClaudeToolState(t1.conversation)?.visible).toEqual(['learn_skill']);

        const t2 = await turn(
            toolResult('c1'),
            [tool('learn_skill'), tool('list_items')],
            t1.conversation,
            assistantToolUse('c2', 'list_items'),
        );
        expect(JSON.stringify(t2.payload.tools?.[0])).toBe(JSON.stringify(t1.payload.tools?.[0]));
        expect(t2.payload.tools).toEqual([tool('learn_skill'), { ...tool('list_items'), defer_loading: true }]);
        expect(t2.payload.messages.at(-1)).toEqual({
            role: 'system',
            content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'list_items' } }],
        });
        expect(t2.payload.messages.at(-2)?.role).toBe('user');
        expect(t2.requestOptions?.headers?.['anthropic-beta']).toBe(MID_CONVERSATION_TOOL_CHANGES_BETA);

        // The change is stored and replayed verbatim; the next request adds nothing new.
        const t3 = await turn(
            toolResult('c2'),
            [tool('learn_skill'), tool('list_items')],
            t2.conversation,
            assistantToolUse('c3', 'list_items'),
        );
        expect(t3.payload.tools).toEqual(t2.payload.tools);
        const systemMessages = t3.payload.messages.filter((m) => m.role === 'system');
        expect(systemMessages).toHaveLength(1);
        expect(t3.payload.messages.slice(0, t2.payload.messages.length)).toMatchObject(
            t2.payload.messages.map((m) => ({ role: m.role })),
        );
    });

    it('withdraws a dropped tool without removing its definition, and re-offers it later', async () => {
        const t1 = await turn(firstPrompt, [tool('a'), tool('b')], undefined, assistantToolUse('c1', 'a'));
        const t2 = await turn(toolResult('c1'), [tool('a')], t1.conversation, assistantToolUse('c2', 'a'));
        expect(t2.payload.tools).toEqual([tool('a'), tool('b')]);
        expect(t2.payload.messages.at(-1)).toEqual({
            role: 'system',
            content: [{ type: 'tool_removal', tool: { type: 'tool_reference', name: 'b' } }],
        });

        const t3 = await turn(toolResult('c2'), [tool('a'), tool('b')], t2.conversation, assistantToolUse('c3', 'b'));
        expect(t3.payload.tools).toEqual([tool('a'), tool('b')]);
        expect(t3.payload.messages.at(-1)).toEqual({
            role: 'system',
            content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'b' } }],
        });
    });

    it('replaces a tool whose definition changed under the same name', () => {
        const declared = applyClaudeToolChanges(start(), [tool('mcp_search', 'v1') as Tool]);
        const next = applyClaudeToolChanges(
            { ...declared, messages: [...declared.messages, { role: 'user', content: 'again' }] },
            [tool('mcp_search', 'v2') as Tool],
        );
        expect(getClaudeToolState(next)?.declared).toEqual([{ definition: tool('mcp_search', 'v2'), deferred: false }]);
        expect(next.messages.some((m) => m.role === 'system')).toBe(false);
    });

    it('ignores key order when comparing definitions', () => {
        const declared = applyClaudeToolChanges(start(), [tool('a') as Tool]);
        const reordered = { input_schema: schema, description: 'a tool', name: 'a' } as Tool;
        const next = applyClaudeToolChanges(declared, [reordered]);
        expect(getClaudeToolState(next)?.declared[0].definition).toEqual(tool('a'));
    });

    it('keeps offering declared tools on a request without tools, with tool_choice none', async () => {
        const t1 = await turn(firstPrompt, [tool('a')], undefined, assistantToolUse('c1', 'a'));
        const final = await turn(toolResult('c1'), [], t1.conversation, assistantToolUse('c2', 'a'));
        expect(final.payload.tools).toEqual([tool('a')]);
        expect(final.payload.tool_choice).toEqual({ type: 'none' });
        expect(JSON.stringify(final.payload.messages)).toContain('"tool_use"');
        expect(getClaudeToolState(final.conversation)?.visible).toEqual(['a']);
    });

    it('sends tools the old way and strips tool changes where they are not supported', async () => {
        const t1 = await turn(firstPrompt, [tool('a')], undefined, assistantToolUse('c1', 'a'));
        const t2 = await turn(toolResult('c1'), [tool('a'), tool('b')], t1.conversation, assistantToolUse('c2', 'b'));

        for (const [provider, model] of [
            ['azure_foundry', MODEL],
            ['anthropic', 'claude-sonnet-5'],
        ]) {
            const legacy = await turn(
                toolResult('c2'),
                [tool('b')],
                t2.conversation,
                assistantToolUse('c3', 'b'),
                provider,
                model,
            );
            expect(legacy.payload.tools).toEqual([tool('b')]);
            expect(legacy.payload.messages.some((m) => m.role === 'system')).toBe(false);
            expect(legacy.requestOptions).toBeUndefined();
        }
    });

    it('adds a continuation turn before announcing a change after an assistant turn', () => {
        const declared = applyClaudeToolChanges(start(), [tool('a') as Tool]);
        const afterAssistant = { ...declared, messages: [...declared.messages, { role: 'assistant', content: 'ok' }] };
        const next = applyClaudeToolChanges(afterAssistant as ClaudePrompt, [tool('a') as Tool, tool('b') as Tool]);
        expect(next.messages.slice(-2)).toEqual([
            { role: 'user', content: [{ type: 'text', text: 'Continue.' }] },
            { role: 'system', content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'b' } }] },
        ]);
        expect(getClaudeToolState(next)?.visible).toEqual(['a', 'b']);
    });

    it('puts the continuation turn before a trailing tool change when orphan repair drops the user turn', () => {
        const declared = applyClaudeToolChanges(start(), [tool('a') as Tool]);
        const prompt = applyClaudeToolChanges(
            {
                ...declared,
                messages: [
                    ...declared.messages,
                    { role: 'assistant', content: 'ok' },
                    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'gone', content: 'x' }] },
                ],
            } as ClaudePrompt,
            [tool('a') as Tool, tool('b') as Tool],
        );
        const { payload } = getClaudePayload({ model: MODEL, tools: [tool('a'), tool('b')] }, prompt);
        expect(payload.messages.slice(-3).map((m) => m.role)).toEqual(['assistant', 'user', 'system']);
    });

    it('restores a user turn before a stored tool change whose user turn orphan repair dropped', () => {
        const declared = applyClaudeToolChanges(start(), [tool('a') as Tool]);
        const prompt: ClaudePrompt = {
            ...declared,
            messages: [
                ...declared.messages,
                { role: 'assistant', content: 'no tool call' },
                { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'gone', content: 'x' }] },
                {
                    role: 'system',
                    content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'b' } }],
                } as unknown as MessageParam,
                { role: 'assistant', content: 'done' },
                { role: 'user', content: 'next' },
            ],
        };
        const { payload } = getClaudePayload({ model: MODEL, tools: [tool('a'), tool('b')] }, prompt);
        const roles = payload.messages.map((m) => m.role);
        expect(roles.slice(1)).toEqual(['assistant', 'user', 'system', 'assistant', 'user']);
    });

    it('drops tool-change blocks left without stored state before declaring tools', () => {
        const rebuilt = {
            messages: [
                { role: 'user', content: 'hi' },
                {
                    role: 'system',
                    content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'gone' } }],
                },
                { role: 'assistant', content: 'ok' },
                { role: 'user', content: 'more' },
            ],
        } as unknown as ClaudePrompt;
        const next = applyClaudeToolChanges(rebuilt, [tool('a') as Tool]);
        expect(next.messages.some((m) => m.role === 'system')).toBe(false);
        expect(getClaudeToolState(next)?.visible).toEqual(['a']);
    });

    it('announces a tool listed twice once', () => {
        const declared = applyClaudeToolChanges(start(), [tool('a') as Tool]);
        const next = applyClaudeToolChanges(
            { ...declared, messages: [...declared.messages, { role: 'user', content: 'more' }] },
            [tool('a') as Tool, tool('b') as Tool, tool('b') as Tool],
        );
        expect(next.messages.at(-1)).toEqual({
            role: 'system',
            content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'b' } }],
        });
    });

    it('keeps agent cache breakpoints off tool-change messages', () => {
        const declared = applyClaudeToolChanges(start(), [tool('a') as Tool]);
        const prompt = applyClaudeToolChanges(
            { ...declared, messages: [...declared.messages, { role: 'user', content: 'more' }] },
            [tool('a') as Tool, tool('b') as Tool],
        );
        const { payload } = getClaudePayload(
            { model: MODEL, tools: [tool('a'), tool('b')], prompt_cache_key: 'agent-run' },
            prompt,
        );
        const system = payload.messages.find((m) => m.role === 'system');
        expect(JSON.stringify(system)).not.toContain('cache_control');
    });

    it('logs when declared tools override a caller tool choice', () => {
        const declared = applyClaudeToolChanges(start(), [tool('a') as Tool]);
        const logger = { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
        const { payload } = getClaudePayload(
            {
                model: MODEL,
                tools: [], // tool_choice is an internal execution hint, not a public option.
                model_options: {
                    _option_id: 'anthropic-claude',
                    tool_choice: 'auto',
                } as ExecutionOptions['model_options'],
            },
            declared,
            'anthropic',
            'execute',
            logger,
        );
        expect(payload.tool_choice).toEqual({ type: 'none' });
        expect(logger.warn).toHaveBeenCalledWith(
            expect.objectContaining({ option_names: ['tool_choice'], reason: 'claude_declared_tools' }),
            expect.any(String),
        );
    });

    it('caches the last tool loaded up front and never marks the caller tools', () => {
        const callerTools = [tool('a'), tool('b')];
        const declared = applyClaudeToolChanges(start(), [callerTools[0] as Tool]);
        const prompt = applyClaudeToolChanges(
            { ...declared, messages: [...declared.messages, { role: 'user', content: 'more' }] },
            callerTools as Tool[],
        );
        const { payload } = getClaudePayload(
            {
                model: MODEL,
                tools: callerTools,
                model_options: { _option_id: 'anthropic-claude', cache_enabled: true },
            },
            prompt,
        );
        expect(payload.tools?.[0]).toMatchObject({ name: 'a', cache_control: { type: 'ephemeral' } });
        expect(payload.tools?.[1]).toMatchObject({ name: 'b', defer_loading: true });
        expect(payload.tools?.[1]).not.toHaveProperty('cache_control');
        expect(callerTools[0]).not.toHaveProperty('cache_control');
        expect(getClaudeToolState(prompt)?.declared[0].definition).not.toHaveProperty('cache_control');
    });

    it('strips tool-change blocks and drops system messages left empty', () => {
        const messages = [
            { role: 'user', content: 'hi' },
            { role: 'system', content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'a' } }] },
        ] as MessageParam[];
        expect(stripClaudeToolChanges(messages)).toEqual([{ role: 'user', content: 'hi' }]);
    });
});

describe('orphan repair around system messages', () => {
    const toolUse = {
        role: 'assistant',
        content: [{ type: 'tool_use', id: 't1', name: 'a', input: {} }],
    } as MessageParam;
    const system = {
        role: 'system',
        content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'b' } }],
    } as unknown as MessageParam;
    const result = {
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 't1', content: 'ok' }],
    } as MessageParam;

    it('keeps tool results whose tool_use sits before a system message', () => {
        expect(fixOrphanedToolResults([toolUse, system, result])).toEqual([toolUse, system, result]);
    });

    it('adds interrupted results to the user turn after a system message', () => {
        const fixed = fixOrphanedToolUse([toolUse, system, { role: 'user', content: 'next' }]);
        expect(fixed[2].content).toEqual([
            expect.objectContaining({ type: 'tool_result', tool_use_id: 't1' }),
            { type: 'text', text: 'next' },
        ]);
    });
});
