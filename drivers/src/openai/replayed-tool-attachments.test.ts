/**
 * A stored tool result is filtered for the model of each request, never in the conversation the driver returns: a turn
 * on a text-only model must not cost a later vision turn the image.
 */

import { type ExecutionOptions, Providers, type ToolDefinition } from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { OpenAIResponsesDriverBase } from './index.js';

const tools: ToolDefinition[] = [{ name: 'fetch', input_schema: { type: 'object' } }];
const IMAGE = { type: 'input_image', image_url: 'data:image/png;base64,AQID', detail: 'auto' } as const;
const NOTE = '[attachment omitted: image (image/png) — gpt-oss-120b cannot read this type in a tool result]';

class TestResponsesDriver extends OpenAIResponsesDriverBase {
    provider: OpenAIResponsesDriverBase['provider'] = Providers.openai;
    service: OpenAI;

    constructor(create: (request: unknown) => Promise<unknown>) {
        super({});
        this.service = { responses: { create } } as unknown as OpenAI;
    }
}

const answer = {
    id: 'r',
    status: 'completed',
    output: [
        {
            id: 'm',
            type: 'message',
            role: 'assistant',
            status: 'completed',
            content: [{ type: 'output_text', text: 'ok', annotations: [] }],
        },
    ],
    usage: { input_tokens: 1, output_tokens: 1 },
};

function history(): OpenAI.Responses.ResponseInputItem[] {
    return [
        { type: 'message', role: 'user', content: 'chart it' },
        { type: 'function_call', call_id: 'c1', name: 'fetch', arguments: '{}' },
        { type: 'function_call_output', call_id: 'c1', output: [{ type: 'input_text', text: 'Fetched' }, IMAGE] },
    ];
}

function sentToolOutput(create: ReturnType<typeof vi.fn>, call: number) {
    const input = (create.mock.calls[call][0] as { input: OpenAI.Responses.ResponseInputItem[] }).input;
    return input.find((item) => item.type === 'function_call_output')?.output;
}

describe('OpenAI Responses replayed tool attachments across a model switch', () => {
    it('omits the image for a text-only turn and sends it again on the next vision turn', async () => {
        const create = vi.fn(async (_request: unknown) => answer);
        const driver = new TestResponsesDriver(create);
        const prompt: OpenAI.Responses.ResponseInputItem[] = [{ type: 'message', role: 'user', content: 'again' }];

        const textOnly = await driver.requestTextCompletion(prompt, {
            model: 'gpt-oss-120b',
            tools,
            conversation: history(),
        });
        expect(sentToolOutput(create, 0)).toEqual([
            { type: 'input_text', text: 'Fetched' },
            { type: 'input_text', text: NOTE },
        ]);

        await driver.requestTextCompletion(prompt, {
            model: 'gpt-5.6',
            tools,
            conversation: textOnly.conversation,
        });
        expect(sentToolOutput(create, 1)).toEqual([{ type: 'input_text', text: 'Fetched' }, IMAGE]);
    });

    it('keeps the image in the conversation built after a streamed text-only turn', async () => {
        const create = vi.fn(async (_request: unknown) => (async function* () {})());
        const driver = new TestResponsesDriver(create);
        const prompt: OpenAI.Responses.ResponseInputItem[] = [{ type: 'message', role: 'user', content: 'again' }];
        const options: ExecutionOptions = { model: 'gpt-oss-120b', tools, conversation: history() };

        for await (const _chunk of await driver.requestTextCompletionStream(prompt, options)) {
            // drain
        }
        expect(sentToolOutput(create, 0)).toEqual([
            { type: 'input_text', text: 'Fetched' },
            { type: 'input_text', text: NOTE },
        ]);

        const conversation = driver.buildStreamingConversation(prompt, [{ type: 'text', value: 'ok' }], [], options);
        expect(JSON.stringify(conversation)).toContain(IMAGE.image_url);
    });
});
