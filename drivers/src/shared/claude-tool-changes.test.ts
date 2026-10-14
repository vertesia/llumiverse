import type { Message, MessageParam, RawMessageStreamEvent, Tool } from '@anthropic-ai/sdk/resources/messages.js';
import { type ExecutionOptions, Providers, type ToolDefinition } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import {
    buildClaudeStreamingConversation,
    type ClaudePrompt,
    executeClaudeCompletion,
    getClaudePayload,
    streamClaudeCompletion,
} from './claude-messages.js';
import {
    applyClaudeToolChanges,
    canUseClaudeToolChanges,
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
    extra: Partial<ExecutionOptions> = {},
) {
    const stream = vi.fn((_payload: unknown, _options: unknown) => sdkStream(response));
    const options: ExecutionOptions = { ...extra, model, tools, conversation };
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
const userText = (text: string): ClaudePrompt => ({ messages: [{ role: 'user', content: [{ type: 'text', text }] }] });

function assistantReply(content: unknown[], stop_reason = 'end_turn'): Message {
    return {
        id: 'msg-reply',
        type: 'message',
        role: 'assistant',
        model: MODEL,
        content,
        stop_reason,
        stop_sequence: null,
        usage: { input_tokens: 1, output_tokens: 1 },
    } as unknown as Message;
}

const addition = (name: string) => ({ type: 'tool_addition', tool: { type: 'tool_reference', name } });
const removal = (name: string) => ({ type: 'tool_removal', tool: { type: 'tool_reference', name } });

/** A copy without request-only `cache_control` markers, to compare what a request replays. */
function withoutCacheControl<T>(value: T): T {
    return JSON.parse(JSON.stringify(value, (key, v) => (key === 'cache_control' ? undefined : v)));
}
const firstPrompt: ClaudePrompt = { messages: [{ role: 'user', content: 'Audit the project.' }] };

describe('append-only Claude tool changes', () => {
    it('declares the first tools as sent and replays them unchanged when a tool is added', async () => {
        const t1 = await turn(firstPrompt, [tool('learn_skill')], undefined, assistantToolUse('c1', 'learn_skill'));
        expect(t1.payload.tools).toEqual([tool('learn_skill')]);
        expect(t1.requestOptions?.headers?.['anthropic-beta']).not.toContain(MID_CONVERSATION_TOOL_CHANGES_BETA);
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
        expect(t2.requestOptions?.headers?.['anthropic-beta']?.split(',')).toContain(
            MID_CONVERSATION_TOOL_CHANGES_BETA,
        );

        // The change is stored and replayed verbatim; the next request adds nothing new.
        const t3 = await turn(
            toolResult('c2'),
            [tool('learn_skill'), tool('list_items')],
            t2.conversation,
            assistantToolUse('c3', 'list_items'),
        );
        expect(JSON.stringify(t3.payload.tools)).toBe(JSON.stringify(t2.payload.tools));
        const systemMessages = t3.payload.messages.filter((m) => m.role === 'system');
        expect(systemMessages).toHaveLength(1);
        // Each request replays the previous one as a prefix.
        expect(withoutCacheControl(t2.payload.messages.slice(0, t1.payload.messages.length))).toEqual(
            withoutCacheControl(t1.payload.messages),
        );
        expect(withoutCacheControl(t3.payload.messages.slice(0, t2.payload.messages.length))).toEqual(
            withoutCacheControl(t2.payload.messages),
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
            expect(legacy.requestOptions?.headers?.['anthropic-beta'] ?? '').not.toContain(
                MID_CONVERSATION_TOOL_CHANGES_BETA,
            );
        }
    });

    it('adds a continuation turn before a change announced after an assistant turn', () => {
        const declared = applyClaudeToolChanges(start(), [tool('a') as Tool]);
        const afterAssistant = { ...declared, messages: [...declared.messages, { role: 'assistant', content: 'ok' }] };
        const next = applyClaudeToolChanges(afterAssistant as ClaudePrompt, [tool('a') as Tool, tool('b') as Tool]);
        expect(getClaudeToolState(next)?.visible).toEqual(['a', 'b']);
        const { payload } = getClaudePayload({ model: MODEL, tools: [tool('a'), tool('b')] }, next);
        expect(payload.messages.slice(-2)).toEqual([
            { role: 'user', content: [{ type: 'text', text: 'Continue.' }] },
            { role: 'system', content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'b' } }] },
        ]);
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

    it.each([
        [
            'a change left first after orphan repair',
            [
                { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'gone', content: 'x' }] },
                { role: 'system', content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'b' } }] },
                { role: 'assistant', content: 'ok' },
                { role: 'user', content: 'next' },
            ],
            ['user', 'system', 'assistant', 'user'],
        ],
        [
            'a change followed by a user turn after an empty reply',
            [
                { role: 'user', content: 'hi' },
                { role: 'system', content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'b' } }] },
                { role: 'user', content: 'again' },
            ],
            ['user', 'system'],
        ],
        [
            'back-to-back changes',
            [
                { role: 'user', content: 'hi' },
                { role: 'system', content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'b' } }] },
                { role: 'system', content: [{ type: 'tool_removal', tool: { type: 'tool_reference', name: 'b' } }] },
            ],
            ['user'],
        ],
    ])('keeps tool changes between a user and an assistant turn: %s', (_case, messages, roles) => {
        const state = getClaudeToolState(applyClaudeToolChanges(start(), [tool('b') as Tool]));
        const { payload } = getClaudePayload({ model: MODEL, tools: [tool('b')] }, {
            messages,
            _llumiverse_meta: { turnNumber: 0, claude_tools: state },
        } as unknown as ClaudePrompt);
        expect(payload.messages.map((m) => m.role)).toEqual(roles);
    });

    it('declares tools again, loaded up front, after a turn without tool-change support', async () => {
        const t1 = await turn(firstPrompt, [tool('a'), tool('b')], undefined, assistantToolUse('c1', 'a'));
        const other = await turn(
            toolResult('c1'),
            [tool('a'), tool('c')],
            t1.conversation,
            assistantToolUse('c2', 'c'),
            'anthropic',
            'claude-sonnet-5',
        );
        expect(getClaudeToolState(other.conversation)).toBeUndefined();

        // The API rejects a tools array whose tools are all deferred, so a new declaration loads them up front.
        const back = await turn(
            toolResult('c2'),
            [tool('a'), tool('c')],
            other.conversation,
            assistantToolUse('c3', 'a'),
        );
        expect(back.payload.tools).toEqual([tool('a'), tool('c')]);
        expect(back.payload.messages.some((m) => m.role === 'system')).toBe(false);
    });

    it('keeps a real tool result when a tool change sat between split result turns', () => {
        const declared = applyClaudeToolChanges(start(), [tool('a') as Tool]);
        const prompt = {
            ...declared,
            messages: [
                ...declared.messages,
                {
                    role: 'assistant',
                    content: [
                        { type: 'tool_use', id: 'x', name: 'a', input: {} },
                        { type: 'tool_use', id: 'y', name: 'a', input: {} },
                    ],
                },
                { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'x', content: 'x done' }] },
                { role: 'system', content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'b' } }] },
                { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'y', content: 'y done' }] },
            ],
        } as unknown as ClaudePrompt;
        const { payload } = getClaudePayload({ model: MODEL, tools: [tool('a'), tool('b')] }, prompt);
        const results = JSON.stringify(payload.messages);
        expect(results).toContain('y done');
        expect(results).not.toContain('Tool interrupted');
    });

    it('declares a tool listed twice once on the first request', () => {
        const declared = applyClaudeToolChanges(start(), [tool('a') as Tool, tool('a', 'v2') as Tool]);
        expect(getClaudeToolState(declared)).toEqual({
            declared: [{ definition: tool('a', 'v2'), deferred: false }],
            visible: ['a'],
        });
        expect(declared._llumiverse_meta?.turnNumber).toBe(0);
    });

    it('keeps tool results first when a continuation turn and tool results merge', () => {
        const declared = applyClaudeToolChanges(start(), [tool('a') as Tool]);
        const prompt = {
            ...declared,
            messages: [
                ...declared.messages,
                { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'a', input: {} }] },
                { role: 'user', content: [{ type: 'text', text: 'Continue.' }] },
                { role: 'system', content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'b' } }] },
                { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1', content: 'ok' }] },
            ],
        } as unknown as ClaudePrompt;
        const { payload } = getClaudePayload({ model: MODEL, tools: [tool('a'), tool('b')] }, prompt);
        const user = payload.messages.at(-2);
        expect(Array.isArray(user?.content) && user.content.map((b) => b.type)).toEqual(['tool_result', 'text']);
    });

    it('cancels a tool added and removed while its changes were pending', () => {
        const declared = applyClaudeToolChanges(start(), [tool('a') as Tool]);
        const prompt = {
            ...declared,
            messages: [
                ...declared.messages,
                { role: 'system', content: [{ type: 'tool_addition', tool: { type: 'tool_reference', name: 'b' } }] },
                { role: 'user', content: 'again' },
                { role: 'system', content: [{ type: 'tool_removal', tool: { type: 'tool_reference', name: 'b' } }] },
            ],
        } as unknown as ClaudePrompt;
        const { payload } = getClaudePayload({ model: MODEL, tools: [tool('a')] }, prompt);
        expect(payload.messages.map((m) => m.role)).toEqual(['user']);
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

describe('orphan repair around stored system messages', () => {
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

    const payloadMessages = (messages: MessageParam[]) => {
        const declared = applyClaudeToolChanges(start(), [tool('a') as Tool]);
        const prompt = { ...declared, messages: [...declared.messages, ...messages] };
        return getClaudePayload({ model: MODEL, tools: [tool('a'), tool('b')] }, prompt).payload.messages;
    };

    it('keeps tool results whose tool_use precedes a stored system message', () => {
        expect(payloadMessages([toolUse, system, result]).slice(1)).toEqual([toolUse, result, system]);
    });

    it('adds interrupted results to the user turn after a stored system message', () => {
        const messages = payloadMessages([toolUse, system, { role: 'user', content: 'next' }]);
        expect(messages[2].content).toEqual([
            expect.objectContaining({ type: 'tool_result', tool_use_id: 't1' }),
            { type: 'text', text: 'next' },
        ]);
        expect(messages[3]).toEqual(system);
    });
});

describe('canUseClaudeToolChanges', () => {
    it.each([
        [Providers.anthropic, MODEL, true],
        [Providers.vertexai, 'claude-sonnet-5-5@20260901', true],
        [Providers.bedrock, 'global.anthropic.claude-sonnet-5-5', true],
        [Providers.bedrock_mantle, 'anthropic.claude-opus-5-5', true],
        [Providers.azure_foundry, MODEL, false],
        [Providers.anthropic, 'claude-sonnet-5', false],
        [Providers.bedrock, 'us.anthropic.claude-sonnet-4-6', false],
    ])('%s with %s: %s', (provider, model, expected) => {
        expect(canUseClaudeToolChanges(model, provider)).toBe(expected);
    });
});

/** Streaming events for one assistant turn that calls a tool. */
function toolUseEvents(id: string, name: string): RawMessageStreamEvent[] {
    return [
        {
            type: 'message_start',
            message: { ...assistantReply([], 'tool_use'), id: `msg-${id}`, content: [] },
        },
        { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id, name, input: {} } },
        { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"q":"x"}' } },
        { type: 'content_block_stop', index: 0 },
        {
            type: 'message_delta',
            delta: { stop_reason: 'tool_use', stop_sequence: null },
            usage: { output_tokens: 1 },
        },
        { type: 'message_stop' },
    ] as unknown as RawMessageStreamEvent[];
}

async function streamTurn(
    prompt: ClaudePrompt,
    tools: ToolDefinition[],
    conversation: unknown,
    id: string,
    name: string,
) {
    const final = assistantToolUse(id, name);
    const stream = vi.fn((_payload: unknown, _options: unknown) => ({
        async *[Symbol.asyncIterator]() {
            yield* toolUseEvents(id, name);
        },
        finalMessage: async () => final,
        abort() {},
    }));
    const completion = await streamClaudeCompletion({ messages: { stream } } as never, prompt, {
        model: MODEL,
        tools,
        conversation,
    });
    const toolNames: string[] = [];
    for await (const chunk of completion) {
        if (typeof chunk === 'object') toolNames.push(...(chunk.tool_use ?? []).map((t) => t.tool_name));
    }
    const stored = JSON.parse(JSON.stringify(await completion.finalizeConversation?.()));
    const [payload] = stream.mock.calls[0] as [{ tools?: Tool[]; messages: MessageParam[] }, unknown];
    return { payload, conversation: stored, toolNames };
}

describe('Claude tool changes on the streaming path', () => {
    it('records the tool change in the conversation finalizeConversation returns', async () => {
        const t1 = await streamTurn(firstPrompt, [tool('a')], undefined, 'c1', 'a');
        expect(t1.toolNames).toContain('a');
        expect(getClaudeToolState(t1.conversation)).toEqual({
            declared: [{ definition: tool('a'), deferred: false }],
            visible: ['a'],
        });

        const t2 = await streamTurn(toolResult('c1'), [tool('a'), tool('b')], t1.conversation, 'c2', 'b');
        expect(t2.payload.tools).toEqual([tool('a'), { ...tool('b'), defer_loading: true }]);
        expect(t2.payload.messages.at(-1)).toEqual({ role: 'system', content: [addition('b')] });
        expect(t2.conversation.messages.map((m: MessageParam) => m.role)).toEqual([
            'user',
            'assistant',
            'user',
            'system',
            'assistant',
        ]);
        expect(getClaudeToolState(t2.conversation)).toEqual({
            declared: [
                { definition: tool('a'), deferred: false },
                { definition: tool('b'), deferred: true },
            ],
            visible: ['a', 'b'],
        });
    });

    it('records tool changes when the streamed conversation is rebuilt from chunks', async () => {
        const t1 = await turn(firstPrompt, [tool('a')], undefined, assistantToolUse('c1', 'a'));
        const options: ExecutionOptions = {
            model: MODEL,
            tools: [tool('a'), tool('b')],
            conversation: t1.conversation,
        };
        const streamed = buildClaudeStreamingConversation(
            toolResult('c1'),
            [],
            [{ id: 'c2', tool_name: 'b', tool_input: { q: 'x' } }],
            options,
            'anthropic',
        );
        const executed = await turn(toolResult('c1'), options.tools, t1.conversation, assistantToolUse('c2', 'b'));

        expect(streamed.messages.map((m) => m.role)).toEqual(['user', 'assistant', 'user', 'system', 'assistant']);
        expect(streamed.messages[3]).toEqual({ role: 'system', content: [addition('b')] });
        expect(executed.conversation.messages[3]).toEqual(streamed.messages[3]);
        expect(getClaudeToolState(streamed)).toEqual(getClaudeToolState(executed.conversation));
    });
});

describe('Claude tool changes across turns', () => {
    it('keeps tools and the message prefix byte-identical as the conversation moves between providers', async () => {
        const t1 = await turn(
            firstPrompt,
            [tool('a'), tool('b')],
            undefined,
            assistantToolUse('c1', 'a'),
            'anthropic',
            'claude-sonnet-5-5',
        );
        const t2 = await turn(
            toolResult('c1'),
            [tool('a'), tool('b'), tool('c')],
            t1.conversation,
            assistantToolUse('c2', 'c'),
            'vertexai',
            'claude-sonnet-5-5@20260901',
        );
        const t3 = await turn(
            toolResult('c2'),
            [tool('a'), tool('b'), tool('c')],
            t2.conversation,
            assistantToolUse('c3', 'b'),
            'bedrock',
            'global.anthropic.claude-sonnet-5-5',
        );

        expect(JSON.stringify(t2.payload.tools?.slice(0, 2))).toBe(JSON.stringify(t1.payload.tools));
        expect(JSON.stringify(t3.payload.tools)).toBe(JSON.stringify(t2.payload.tools));
        expect(JSON.stringify(t2.payload.messages.slice(0, t1.payload.messages.length))).toBe(
            JSON.stringify(t1.payload.messages),
        );
        expect(JSON.stringify(t3.payload.messages.slice(0, t2.payload.messages.length))).toBe(
            JSON.stringify(t2.payload.messages),
        );
        expect(t3.payload.messages.filter((m) => m.role === 'system')).toEqual([
            { role: 'system', content: [addition('c')] },
        ]);
    });

    it('announces an addition and a removal in one system message', async () => {
        const t1 = await turn(firstPrompt, [tool('a')], undefined, assistantToolUse('c1', 'a'));
        const t2 = await turn(toolResult('c1'), [tool('b')], t1.conversation, assistantToolUse('c2', 'b'));
        const systemMessages = t2.payload.messages.filter((m) => m.role === 'system');
        expect(systemMessages).toEqual([{ role: 'system', content: [addition('b'), removal('a')] }]);
        expect(t2.payload.tools).toEqual([tool('a'), { ...tool('b'), defer_loading: true }]);
        expect(getClaudeToolState(t2.conversation)?.visible).toEqual(['b']);
    });

    it('sends declared tools on a turn without tools, then announces only real differences', async () => {
        const t1 = await turn(firstPrompt, [tool('a')], undefined, assistantToolUse('c1', 'a'));
        const empty = await turn(
            toolResult('c1'),
            undefined,
            t1.conversation,
            assistantReply([{ type: 'text', text: 'Summary.' }]),
        );
        expect(empty.payload.tools).toEqual([tool('a')]);
        expect(empty.payload.tool_choice).toEqual({ type: 'none' });
        expect(empty.payload.messages.some((m) => m.role === 'system')).toBe(false);

        const same = await turn(userText('again'), [tool('a')], empty.conversation, assistantToolUse('c2', 'a'));
        expect(same.payload.messages.some((m) => m.role === 'system')).toBe(false);
        expect(same.payload.tool_choice).toBeUndefined();

        const more = await turn(
            userText('more'),
            [tool('a'), tool('b')],
            empty.conversation,
            assistantToolUse('c2', 'b'),
        );
        expect(more.payload.messages.filter((m) => m.role === 'system')).toEqual([
            { role: 'system', content: [addition('b')] },
        ]);
    });

    it('replays a thinking block from before a tool change unchanged', async () => {
        const thinking = { type: 'thinking', thinking: 'Look up the item first.', signature: 'sig-before' };
        const t1 = await turn(
            firstPrompt,
            [tool('a')],
            undefined,
            assistantReply([thinking, { type: 'tool_use', id: 'c1', name: 'a', input: { q: 'x' } }], 'tool_use'),
        );
        const t2 = await turn(toolResult('c1'), [tool('a'), tool('b')], t1.conversation, assistantToolUse('c2', 'b'));
        const t3 = await turn(toolResult('c2'), [tool('a'), tool('b')], t2.conversation, assistantToolUse('c3', 'b'));

        for (const { payload } of [t2, t3]) {
            expect(payload.messages[1]).toEqual(t1.conversation.messages[1]);
            expect(payload.messages[1].content).toContainEqual(thinking);
        }
        expect(t2.payload.messages.findIndex((m) => m.role === 'system')).toBeGreaterThan(1);
    });

    it('stores the continuation turn for a resumed run that ends with a tool call and replays it', async () => {
        const toolCall = assistantReply([{ type: 'tool_use', id: 'c1', name: 'a', input: { q: 'x' } }], 'tool_use');
        const t1 = await turn(firstPrompt, [tool('a')], undefined, toolCall);
        const resumed = await turn(
            { messages: [] },
            [tool('a')],
            t1.conversation,
            assistantReply([{ type: 'text', text: 'Resumed.' }]),
        );

        const last = resumed.payload.messages.at(-1);
        expect(resumed.payload.messages.at(-2)?.role).toBe('assistant');
        expect(last?.role).toBe('user');
        expect(last?.content).toEqual([
            expect.objectContaining({ type: 'tool_result', tool_use_id: 'c1' }),
            { type: 'text', text: 'Continue.' },
        ]);
        expect(resumed.conversation.messages.map((m: MessageParam) => m.role)).toEqual([
            'user',
            'assistant',
            'user',
            'assistant',
        ]);
        expect(resumed.conversation.messages[2]).toEqual({
            role: 'user',
            content: [{ type: 'text', text: 'Continue.' }],
        });

        const next = await turn(userText('next'), [tool('a')], resumed.conversation, assistantToolUse('c2', 'a'));
        expect(JSON.stringify(next.payload.messages.slice(0, resumed.payload.messages.length))).toBe(
            JSON.stringify(resumed.payload.messages),
        );
    });

    it('keeps the tool state and tool-change blocks through stored-conversation cleanup', async () => {
        const cleanup: Partial<ExecutionOptions> = {
            stripImagesAfterTurns: 0,
            stripTextMaxTokens: 2,
            stripHeartbeatsAfterTurns: 0,
        };
        const t1 = await turn(
            firstPrompt,
            [tool('a')],
            undefined,
            assistantToolUse('c1', 'a'),
            'anthropic',
            MODEL,
            cleanup,
        );
        const prompt = {
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'tool_result', tool_use_id: 'c1', content: 'ok' },
                        { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'iVBORw0KGgo=' } },
                        { type: 'text', text: '<heartbeat>status update</heartbeat>' },
                    ],
                },
            ],
        } as ClaudePrompt;
        const t2 = await turn(
            prompt,
            [tool('a'), tool('list_items')],
            t1.conversation,
            assistantReply([{ type: 'text', text: 'Listed.' }]),
            'anthropic',
            MODEL,
            cleanup,
        );

        const stored = JSON.stringify(t2.conversation);
        // The cleanup ran on this turn...
        expect(stored).not.toContain('iVBORw0KGgo=');
        expect(stored).not.toContain('status update');
        // ...without touching the tool state or the tool change.
        expect(t2.conversation.messages).toContainEqual({ role: 'system', content: [addition('list_items')] });
        expect(getClaudeToolState(t2.conversation)).toEqual({
            declared: [
                { definition: tool('a'), deferred: false },
                { definition: tool('list_items'), deferred: true },
            ],
            visible: ['a', 'list_items'],
        });

        const t3 = await turn(
            userText('again'),
            [tool('a'), tool('list_items')],
            t2.conversation,
            assistantToolUse('c3', 'a'),
            'anthropic',
            MODEL,
            cleanup,
        );
        expect(t3.payload.tools).toEqual([tool('a'), { ...tool('list_items'), defer_loading: true }]);
        expect(t3.payload.messages.filter((m) => m.role === 'system')).toEqual([
            { role: 'system', content: [addition('list_items')] },
        ]);
    });
});

describe('Claude cache breakpoints around tool changes', () => {
    /** A conversation of five turns whose last request adds tool b, ending with a tool-change system message. */
    function promptWithToolChange(): ClaudePrompt {
        const declared = applyClaudeToolChanges(start(), [tool('a') as Tool]);
        const text = (t: string) => [{ type: 'text' as const, text: t }];
        return applyClaudeToolChanges(
            {
                ...declared,
                system: [{ type: 'text', text: 'You are helpful.' }],
                messages: [
                    ...declared.messages,
                    { role: 'assistant', content: text('ok') },
                    { role: 'user', content: text('more') },
                    { role: 'assistant', content: text('sure') },
                    { role: 'user', content: [...text('stable source'), ...text('task')] },
                ],
            },
            [tool('a') as Tool, tool('b') as Tool],
        );
    }

    function expectNoBreakpointOnToolChanges(payload: { messages: MessageParam[]; tools?: unknown[] }) {
        const system = payload.messages.filter((m) => m.role === 'system');
        expect(system).toHaveLength(1);
        expect(JSON.stringify(system)).not.toContain('cache_control');
        const deferred = (payload.tools ?? []).filter((t) => (t as { defer_loading?: boolean }).defer_loading);
        expect(deferred).toHaveLength(1);
        expect(deferred[0]).not.toHaveProperty('cache_control');
    }

    it.each([
        ['the pivot turn', {}],
        ['the prompt_cache_key', { prompt_cache_key: 'document-prefix' }],
    ])('marks %s, never a tool-change message or a deferred tool, and leaves the input unchanged', (_case, extra) => {
        const prompt = promptWithToolChange();
        const tools = [tool('a'), tool('b')];
        const options: ExecutionOptions = {
            ...extra,
            model: MODEL,
            tools,
            model_options: { _option_id: 'anthropic-claude', cache_enabled: true },
        };
        const before = structuredClone({ prompt, tools });

        const { payload } = getClaudePayload(options, prompt);

        expectNoBreakpointOnToolChanges(payload);
        expect(payload.tools?.[0]).toMatchObject({ name: 'a', cache_control: { type: 'ephemeral' } });
        const turns = payload.messages.filter((m) => m.role !== 'system');
        // The routed key marks the stable block before the task; otherwise the pivot turn is marked.
        const markedTurn = turns['prompt_cache_key' in extra ? turns.length - 1 : turns.length - 2];
        const marked = (markedTurn.content as { text: string }[])[0];
        expect(marked).toMatchObject({ cache_control: { type: 'ephemeral' } });
        expect({ prompt, tools }).toEqual(before);
    });
});
