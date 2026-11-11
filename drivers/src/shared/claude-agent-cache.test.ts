import { createHash } from 'node:crypto';
import type { Message, MessageCreateParamsBase, MessageParam } from '@anthropic-ai/sdk/resources/messages.js';
import { keepsClaudePriorTurnThinking } from '@llumiverse/common';
import type { ExecutionOptions } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { claudeSdkStream } from '../../test/__helpers__/test-utils.js';
import { type ClaudePrompt, claudeHeartbeatKeepTurns, executeClaudeCompletion } from './claude-messages.js';

type Payload = MessageCreateParamsBase;
type Step = 'tools' | 'answer';

/** Small deterministic PRNG so every generated run is reproducible from its seed. */
function prng(seed: number) {
    let state = seed;
    return (limit: number) => {
        state = (state * 1103515245 + 12345) % 2 ** 31;
        return state % limit;
    };
}

function stripCacheControl<T>(value: T): T {
    return JSON.parse(JSON.stringify(value, (key, entry) => (key === 'cache_control' ? undefined : entry)));
}

function assistantMessage(model: string, content: unknown[], stop: 'tool_use' | 'end_turn'): Message {
    return {
        id: `msg-${Math.random()}`,
        type: 'message',
        role: 'assistant',
        model,
        content,
        stop_reason: stop,
        stop_sequence: null,
        usage: { input_tokens: 1, output_tokens: 1 },
    } as unknown as Message;
}

/** Cache markers as [message index, block index]; -1 for a system block or a tool. */
function markers(payload: Payload): [number, number][] {
    const found: [number, number][] = [];
    for (const block of (payload.system as { cache_control?: unknown }[] | undefined) ?? []) {
        if (block.cache_control) found.push([-1, -1]);
    }
    for (const tool of (payload.tools as { cache_control?: unknown }[] | undefined) ?? []) {
        if (tool.cache_control) found.push([-1, -1]);
    }
    payload.messages.forEach((message, messageIndex) => {
        if (!Array.isArray(message.content)) return;
        message.content.forEach((block, blockIndex) => {
            if ((block as { cache_control?: unknown }).cache_control) found.push([messageIndex, blockIndex]);
        });
    });
    return found;
}

function isPlainUserTurn(message: MessageParam): boolean {
    return (
        message.role === 'user' &&
        !(Array.isArray(message.content) && message.content.some((block) => block.type === 'tool_result'))
    );
}

interface CachedBlock {
    /** Identifies the whole prompt up to and including this block. */
    prefix: string;
    marked: boolean;
    /** Lookback position: a run of tool_use or tool_result blocks in one message shares a position. */
    position: number;
}

/** Flattens a request in render order (tools, system, messages) into cumulative prefix identities. */
function cachedBlocks(payload: Payload): CachedBlock[] {
    const blocks: CachedBlock[] = [];
    let prefix = '';
    let position = -1;
    const add = (scope: string, block: unknown, runType?: string, previousType?: string) => {
        prefix = createHash('sha256')
            .update(`${prefix}|${scope}|${JSON.stringify(stripCacheControl(block))}`)
            .digest('hex');
        if (!runType || runType !== previousType) position++;
        blocks.push({ prefix, marked: !!(block as { cache_control?: unknown }).cache_control, position });
    };
    for (const tool of payload.tools ?? []) add('tool', tool);
    for (const block of (payload.system as unknown[] | undefined) ?? []) add('system', block);
    for (const message of payload.messages) {
        const content = Array.isArray(message.content) ? message.content : [message.content];
        content.forEach((block, index) => {
            const type = (block as { type?: string }).type;
            const runType = type === 'tool_use' || type === 'tool_result' ? type : undefined;
            const previousType = index > 0 ? (content[index - 1] as { type?: string }).type : undefined;
            add(message.role, block, runType, previousType);
        });
    }
    return blocks;
}

/**
 * Replays requests against a model of the provider cache: each marker writes the prefix it ends, and reads the
 * longest prefix an earlier request wrote at most 20 positions behind it. Returns the read block index per request.
 */
function simulateCache(payloads: Payload[]): number[] {
    const written = new Set<string>();
    return payloads.map((payload) => {
        const blocks = cachedBlocks(payload);
        let read = -1;
        blocks.forEach((block, index) => {
            if (!block.marked) return;
            for (let back = index; back >= 0 && block.position - blocks[back].position < 20; back--) {
                if (written.has(blocks[back].prefix)) {
                    read = Math.max(read, back);
                    break;
                }
            }
        });
        for (const block of blocks) if (block.marked) written.add(block.prefix);
        return read;
    });
}

/**
 * Drives a generated agent run through the real request builder and returns every request payload. Each user turn
 * runs a tool loop of up to 25 iterations, with one to three parallel calls each, then answers.
 */
async function runAgent(model: string, provider: string, seed: number, requests: number) {
    const random = prng(seed);
    const options: ExecutionOptions = {
        model,
        prompt_cache_key: 'agent-generated-run',
        model_options: { effort: 'low' } as never,
        tools: [
            { name: 'lookup', description: 'Look a value up.', input_schema: { type: 'object', properties: {} } },
            { name: 'write', description: 'Write a value.', input_schema: { type: 'object', properties: {} } },
        ],
        stripHeartbeatsAfterTurns: 1,
        stripTextMaxTokens: 8_000,
    };
    let prompt: ClaudePrompt = {
        system: [{ type: 'text', text: 'You are a test agent.' }],
        messages: [{ role: 'user', content: [{ type: 'text', text: 'Start the task.' }] }],
    };
    let conversation: unknown;
    const payloads: Payload[] = [];
    const steps: Step[] = [];
    let loopLeft = random(26);

    for (let request = 0; request < requests; request++) {
        const step: Step = loopLeft > 0 ? 'tools' : 'answer';
        const calls = 1 + random(3);
        const content =
            step === 'tools'
                ? [
                      { type: 'thinking', thinking: `plan ${request}`, signature: `sig-${request}` },
                      ...Array.from({ length: calls }, (_, call) => ({
                          type: 'tool_use',
                          id: `call-${request}-${call}`,
                          name: call % 2 ? 'write' : 'lookup',
                          input: { n: request },
                      })),
                  ]
                : [
                      { type: 'thinking', thinking: `answer ${request}`, signature: `sig-${request}` },
                      { type: 'text', text: `Answer ${request}.`, citations: null },
                  ];
        const response = assistantMessage(model, content, step === 'tools' ? 'tool_use' : 'end_turn');
        const stream = vi.fn((_payload: Payload) => claudeSdkStream([], response));
        const completion = await executeClaudeCompletion(
            { messages: { stream } } as never,
            prompt,
            { ...options, conversation },
            undefined,
            provider,
        );
        payloads.push(stream.mock.calls[0][0]);
        steps.push(step);
        conversation = completion.conversation;

        if (step === 'tools') {
            loopLeft--;
            const results: MessageParam = {
                role: 'user',
                content: Array.from({ length: calls }, (_, call) => ({
                    type: 'tool_result' as const,
                    tool_use_id: `call-${request}-${call}`,
                    content: `result ${request}-${call} ${'x'.repeat(random(3) * 400)}`,
                })),
            };
            prompt = { messages: [results] };
        } else {
            loopLeft = random(26);
            const text =
                random(2) === 0
                    ? `<heartbeat>[System] Status heartbeat ${request}</heartbeat>`
                    : `Next question ${request}.`;
            prompt = { messages: [{ role: 'user', content: [{ type: 'text', text }] }] };
        }
    }
    return { payloads, steps };
}

describe('Claude agent prompt caching', () => {
    // Sonnet 4.5 strips earlier turns' thinking, so its history is pruned at turn boundaries; the others keep it.
    const runs = ['claude-sonnet-4-5', 'claude-sonnet-4-6', 'claude-sonnet-5-5'].flatMap((model) =>
        ['anthropic', 'vertexai', 'bedrock'].flatMap((provider) =>
            [1, 7, 42].map((seed) => ({ model, provider, seed })),
        ),
    );

    it.each(runs)(
        'keeps every request a prefix of the next ($model, $provider, seed $seed)',
        async ({ model, provider, seed }) => {
            const { payloads, steps } = await runAgent(model, provider, seed, 150);

            for (let index = 1; index < payloads.length; index++) {
                const previous = stripCacheControl(payloads[index - 1]);
                const current = stripCacheControl(payloads[index]);
                expect(current.tools).toEqual(previous.tools);
                expect(current.system).toEqual(previous.system);

                // Models that keep earlier turns' thinking only ever append. For the others, turn boundaries prune
                // thinking (see pruneClaudeThinking), but never before the earliest marked user turn.
                const earliest = markers(payloads[index]).filter(([message]) => message >= 0)[0];
                const appendOnly =
                    keepsClaudePriorTurnThinking(model) ||
                    (steps[index - 1] === 'tools' && !isPlainUserTurn(previous.messages.at(-1) as MessageParam));
                const unchanged = appendOnly ? previous.messages.length : earliest ? earliest[0] + 1 : 0;
                expect(
                    current.messages.slice(0, unchanged).map((message) => JSON.stringify(message)),
                    `request ${index}`,
                ).toEqual(previous.messages.slice(0, unchanged).map((message) => JSON.stringify(message)));
            }
        },
    );

    it.each(runs)(
        'reads the previous request from cache, and for pruned models at least the earlier user turns across turns ($model, $provider, seed $seed)',
        async ({ model, provider, seed }) => {
            const { payloads, steps } = await runAgent(model, provider, seed, 150);
            const reads = simulateCache(payloads);

            for (let index = 1; index < payloads.length; index++) {
                const previousBlocks = cachedBlocks(payloads[index - 1]).length;
                const appendOnly =
                    keepsClaudePriorTurnThinking(model) ||
                    (steps[index - 1] === 'tools' &&
                        !isPlainUserTurn(payloads[index - 1].messages.at(-1) as MessageParam));
                const blocks = cachedBlocks(payloads[index]);
                const prefixMarker = blocks.findIndex((block) => block.marked);
                const earliestUserMarker = blocks.findIndex(
                    (block, blockIndex) => block.marked && blockIndex > prefixMarker,
                );
                const expected = appendOnly ? previousBlocks - 1 : earliestUserMarker;
                expect(reads[index], `request ${index} (${steps[index - 1]} before)`).toBeGreaterThanOrEqual(expected);
            }
        },
    );

    it.each(runs)(
        'marks the static prefix, the tail, and for pruned models the two latest user turns ($model, $provider, seed $seed)',
        async ({ model, provider, seed }) => {
            const { payloads } = await runAgent(model, provider, seed, 60);

            for (const payload of payloads) {
                const found = markers(payload);
                expect(found.length).toBeLessThanOrEqual(4);
                expect(found.filter(([message]) => message < 0)).toHaveLength(1);

                const turns = payload.messages.map((message, index) => ({ message, index }));
                const last = turns.findLast(({ message }) => message.role !== 'system');
                expect(last).toBeDefined();
                const lastBlocks = last?.message.content as { type: string; cache_control?: unknown }[];
                expect(lastBlocks.findLast((block) => block.type !== 'thinking')?.cache_control).toBeDefined();

                const userTurns = keepsClaudePriorTurnThinking(model)
                    ? []
                    : turns
                          .filter(({ message, index }) => index < (last?.index ?? 0) && isPlainUserTurn(message))
                          .slice(-2)
                          .map(({ index }) => index);
                expect(found.filter(([message]) => message >= 0 && message !== last?.index).map(([m]) => m)).toEqual(
                    userTurns,
                );
            }
        },
    );

    it('keeps heartbeats in a cached conversation', async () => {
        const { payloads } = await runAgent('claude-sonnet-4-6', 'anthropic', 3, 150);
        const final = JSON.stringify(payloads.at(-1));

        expect(final).toContain('<heartbeat>');
        expect(final).not.toContain('[Heartbeat removed from conversation history]');
    });

    it('strips heartbeats as requested only when prompt caching is off', () => {
        expect(claudeHeartbeatKeepTurns({ model: 'claude-sonnet-4-6', stripHeartbeatsAfterTurns: 2 })).toBe(2);
        expect(claudeHeartbeatKeepTurns({ model: 'claude-sonnet-4-6' })).toBe(1);
        expect(
            claudeHeartbeatKeepTurns({
                model: 'claude-sonnet-4-6',
                prompt_cache_key: 'agent-x',
                stripHeartbeatsAfterTurns: 2,
            }),
        ).toBe(Infinity);
        expect(
            claudeHeartbeatKeepTurns({ model: 'claude-sonnet-4-6', model_options: { cache_enabled: true } as never }),
        ).toBe(Infinity);
    });

    it('marks the last loaded tool when an agent request has no system prompt', async () => {
        const stream = vi.fn((_payload: Payload) =>
            claudeSdkStream([], assistantMessage('claude-sonnet-4-6', [{ type: 'text', text: 'ok' }], 'end_turn')),
        );
        await executeClaudeCompletion(
            { messages: { stream } } as never,
            { messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] },
            {
                model: 'claude-sonnet-4-6',
                prompt_cache_key: 'agent-no-system',
                model_options: { cache_ttl: '1h' } as never,
                tools: [{ name: 'lookup', description: 'Look up.', input_schema: { type: 'object', properties: {} } }],
            },
        );
        const payload = stream.mock.calls[0][0];

        expect(payload.tools?.at(-1)).toMatchObject({ cache_control: { type: 'ephemeral', ttl: '1h' } });
        expect(payload.messages[0].content).toEqual([
            { type: 'text', text: 'hi', cache_control: { type: 'ephemeral', ttl: '1h' } },
        ]);
    });
});
