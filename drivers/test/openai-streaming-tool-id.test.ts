import type OpenAI from 'openai';
import { describe, expect, test } from 'vitest';
import { mapResponseStream } from '../src/openai/index.js';
import type { Tree } from './__helpers__/test-utils.js';

type ResponseStreamEvent = OpenAI.Responses.ResponseStreamEvent;

async function* streamEvents(events: ResponseStreamEvent[]): AsyncIterable<ResponseStreamEvent> {
    for (const event of events) {
        yield event;
    }
}

describe('OpenAI Responses streaming tool ids', () => {
    test('uses call_id, not response item id, as the final tool result id', async () => {
        const events = [
            {
                type: 'response.output_item.added',
                output_index: 0,
                item: {
                    type: 'function_call',
                    id: 'fc_response_item',
                    call_id: 'call_plan_tool',
                    name: 'plan',
                    arguments: '',
                },
            },
            {
                type: 'response.function_call_arguments.delta',
                output_index: 0,
                item_id: 'fc_response_item',
                delta: '{"plan":',
            },
            {
                type: 'response.function_call_arguments.delta',
                output_index: 0,
                item_id: 'fc_response_item',
                delta: '[]}',
            },
            {
                type: 'response.function_call_arguments.done',
                output_index: 0,
                item_id: 'fc_response_item',
                arguments: '{"plan":[]}',
                sequence_number: 3,
            } satisfies OpenAI.Responses.ResponseFunctionCallArgumentsDoneEvent,
        ] as ResponseStreamEvent[];

        const chunks = [];
        for await (const chunk of mapResponseStream(streamEvents(events))) {
            chunks.push(chunk);
        }

        const tools = chunks.flatMap((chunk) => chunk.tool_use ?? []);
        expect(tools).toHaveLength(3);
        expect(tools.every((tool) => tool.tool_name === 'plan')).toBe(true);
        expect(tools.map((tool) => tool.tool_input).join('')).toBe('{"plan":[]}');
        expect(tools.map((tool) => (tool as unknown as Tree as Tree)._actual_id)).toEqual([
            'call_plan_tool',
            'call_plan_tool',
            'call_plan_tool',
        ]);
    });

    test('falls back to response item id when call_id is absent', async () => {
        const events = [
            {
                type: 'response.output_item.added',
                output_index: 0,
                item: {
                    type: 'function_call',
                    id: 'fc_response_item',
                    name: 'plan',
                    arguments: '',
                },
            },
            {
                type: 'response.function_call_arguments.delta',
                output_index: 0,
                item_id: 'fc_response_item',
                delta: '{}',
            },
        ] as ResponseStreamEvent[];

        const chunks = [];
        for await (const chunk of mapResponseStream(streamEvents(events))) {
            chunks.push(chunk);
        }

        const tools = chunks.flatMap((chunk) => chunk.tool_use ?? []);
        expect((tools[0] as unknown as Tree)._actual_id).toBe('fc_response_item');
        expect((tools[1] as unknown as Tree)._actual_id).toBe('fc_response_item');
    });
});
