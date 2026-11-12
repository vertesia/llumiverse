/**
 * Some Responses-compatible endpoints stream a function call's arguments only in its completed events
 * (`response.function_call_arguments.done`, `response.output_item.done`, or the terminal response) and send no
 * argument deltas. The stream must still return the call's arguments instead of an empty input.
 */

import type { CompletionChunkObject, ToolUse } from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it } from 'vitest';
import { mapResponseStream } from './index.js';

type Event = OpenAI.Responses.ResponseStreamEvent;

async function* events(list: unknown[]): AsyncIterable<Event> {
    for (const event of list) {
        yield event as Event;
    }
}

/** Collects the streamed calls the way the core stream does: arguments concatenated per call id. */
async function streamedCalls(list: unknown[]): Promise<Record<string, { name: string; input: string }>> {
    const calls: Record<string, { name: string; input: string }> = {};
    for await (const chunk of mapResponseStream(events(list))) {
        for (const tool of (chunk as CompletionChunkObject).tool_use ?? []) {
            const call = calls[tool.id] ?? { name: '', input: '' };
            calls[tool.id] = call;
            if (tool.tool_name) call.name = tool.tool_name;
            if (typeof tool.tool_input === 'string') call.input += tool.tool_input;
        }
    }
    return calls;
}

function added(index: number, name: string): unknown {
    return {
        type: 'response.output_item.added',
        output_index: index,
        item: { type: 'function_call', id: `fc_${index}`, call_id: `call_${index}`, name, arguments: '' },
    };
}

function functionCall(index: number, name: string, args: string): OpenAI.Responses.ResponseFunctionToolCall {
    return { type: 'function_call', id: `fc_${index}`, call_id: `call_${index}`, name, arguments: args };
}

function completed(output: unknown[]): unknown {
    return {
        type: 'response.completed',
        response: { id: 'resp_1', status: 'completed', output, usage: null },
    };
}

describe('mapResponseStream function call arguments', () => {
    it('takes the arguments from function_call_arguments.done when no delta arrived', async () => {
        const calls = await streamedCalls([
            added(0, 'execute_shell'),
            added(1, 'discover_tools'),
            {
                type: 'response.function_call_arguments.done',
                item_id: 'fc_0',
                output_index: 0,
                arguments: '{"command":"date"}',
            },
            {
                type: 'response.function_call_arguments.done',
                item_id: 'fc_1',
                output_index: 1,
                arguments: '{"task":"find tools"}',
            },
            completed([
                functionCall(0, 'execute_shell', '{"command":"date"}'),
                functionCall(1, 'discover_tools', '{"task":"find tools"}'),
            ]),
        ]);

        expect(calls).toEqual({
            tool_0: { name: 'execute_shell', input: '{"command":"date"}' },
            tool_1: { name: 'discover_tools', input: '{"task":"find tools"}' },
        });
    });

    it('takes the arguments from output_item.done when no delta arrived', async () => {
        const calls = await streamedCalls([
            added(0, 'list_revisions'),
            {
                type: 'response.output_item.done',
                output_index: 0,
                item: functionCall(0, 'list_revisions', '{"id":"a1"}'),
            },
            completed([functionCall(0, 'list_revisions', '{"id":"a1"}')]),
        ]);

        expect(calls).toEqual({ tool_0: { name: 'list_revisions', input: '{"id":"a1"}' } });
    });

    it('takes the arguments from the terminal response when no other event carried them', async () => {
        const calls = await streamedCalls([added(0, 'search'), completed([functionCall(0, 'search', '{"q":"x"}')])]);

        expect(calls).toEqual({ tool_0: { name: 'search', input: '{"q":"x"}' } });
    });

    it('does not repeat arguments that already arrived as deltas', async () => {
        const calls = await streamedCalls([
            added(0, 'search'),
            { type: 'response.function_call_arguments.delta', item_id: 'fc_0', output_index: 0, delta: '{"q":' },
            { type: 'response.function_call_arguments.delta', item_id: 'fc_0', output_index: 0, delta: '"x"}' },
            { type: 'response.function_call_arguments.done', item_id: 'fc_0', output_index: 0, arguments: '{"q":"x"}' },
            { type: 'response.output_item.done', output_index: 0, item: functionCall(0, 'search', '{"q":"x"}') },
            completed([functionCall(0, 'search', '{"q":"x"}')]),
        ]);

        expect(calls).toEqual({ tool_0: { name: 'search', input: '{"q":"x"}' } });
    });

    it('keeps the provider call id for a call completed without deltas', async () => {
        const tools: ToolUse<unknown>[] = [];
        for await (const chunk of mapResponseStream(
            events([added(0, 'search'), completed([functionCall(0, 'search', '{"q":"x"}')])]),
        )) {
            tools.push(...((chunk as CompletionChunkObject).tool_use ?? []));
        }

        expect(tools.at(-1)).toMatchObject({ id: 'tool_0', _actual_id: 'call_0', tool_input: '{"q":"x"}' });
    });
});
