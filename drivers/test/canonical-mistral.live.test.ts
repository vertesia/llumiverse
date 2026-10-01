import type { ConversationDocument, ConversationStreamEvent } from '@llumiverse/conversation';
import {
    type CanonicalExecutionEventStream,
    type CanonicalExecutionInputOptions,
    type ExecutionOptions,
    PromptRole,
} from '@llumiverse/core';
import 'dotenv/config';
import { describe, expect, test } from 'vitest';
import { MistralAIDriver } from '../src/index.js';
import { selectLiveTestDrivers } from './live-model-selection.js';

const TIMEOUT = 90_000;
const MODEL = 'mistral-small-latest';
const TOOLS = [
    {
        name: 'lookup',
        description: 'Return the value for a key.',
        input_schema: {
            type: 'object' as const,
            properties: { key: { type: 'string' as const } },
            required: ['key'],
            additionalProperties: false,
        },
    },
];

interface MistralCanonicalLiveDriver {
    name: string;
    models: string[];
    setup(): {
        driver: MistralAIDriver;
        transportCalls(): number;
        restore(): void;
    };
}

function countMethodCalls(target: object, key: PropertyKey): { calls(): number; restore(): void } {
    const original = Reflect.get(target, key) as unknown;
    if (typeof original !== 'function') throw new TypeError(`Expected ${String(key)} to be callable`);
    const ownDescriptor = Object.getOwnPropertyDescriptor(target, key);
    let count = 0;
    Object.defineProperty(target, key, {
        configurable: true,
        writable: true,
        value: function countedMethod(this: unknown, ...args: unknown[]) {
            count += 1;
            return Reflect.apply(original, this, args);
        },
    });
    return {
        calls: () => count,
        restore: () => {
            if (ownDescriptor === undefined) Reflect.deleteProperty(target, key);
            else Object.defineProperty(target, key, ownDescriptor);
        },
    };
}

const liveDrivers: MistralCanonicalLiveDriver[] = [];
if (process.env.MISTRAL_API_KEY) {
    liveDrivers.push({
        name: 'mistralai',
        models: [MODEL],
        setup: () => {
            const driver = new MistralAIDriver({
                apiKey: process.env.MISTRAL_API_KEY as string,
                endpoint_url: process.env.MISTRAL_ENDPOINT_URL,
            });
            const transport = countMethodCalls(driver.client.chat, 'stream');
            return {
                driver,
                transportCalls: transport.calls,
                restore: transport.restore,
            };
        },
    });
} else {
    console.warn('Canonical Mistral live coverage is unavailable: MISTRAL_API_KEY is not set');
}

const selection = {
    providers: process.env.LLUMIVERSE_LIVE_PROVIDERS,
    models: process.env.LLUMIVERSE_LIVE_MODELS,
};
const selectedDrivers = selectLiveTestDrivers(liveDrivers, selection);
if (
    process.env.CI === 'true' &&
    selection.providers === undefined &&
    selection.models === undefined &&
    liveDrivers.length !== 1
) {
    throw new Error('Canonical Mistral live coverage requires MISTRAL_API_KEY');
}

function options(attempt: string, conversation?: ConversationDocument): CanonicalExecutionInputOptions {
    return {
        model: MODEL,
        ...(conversation === undefined ? {} : { conversation }),
        model_options: {
            _option_id: 'mistral-text',
            max_tokens: 96,
            tool_choice: 'required',
            required_tool_name: 'lookup',
        } as ExecutionOptions['model_options'] & { required_tool_name: string },
        tools: TOOLS,
        conversation_runtime: {
            conversation_id: 'live:canonical:mistral',
            request_id: 'live:canonical:mistral:request',
            attempt_id: `live:canonical:mistral:attempt:${attempt}`,
            input_operation_id: 'live:canonical:mistral:input',
            response_operation_id: 'live:canonical:mistral:response',
            recorded_at: '2026-09-30T00:00:00.000Z',
            started_at: '2026-09-30T00:00:00.000Z',
            completed_at: '2026-09-30T00:00:00.000Z',
        },
    };
}

async function collect(stream: CanonicalExecutionEventStream): Promise<ConversationStreamEvent[]> {
    const events: ConversationStreamEvent[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

describe.each(selectedDrivers)('Canonical Mistral live execution', (live) => {
    test.each(live.models)(
        'streams a native tool call and exact-recovers without another transport for %s',
        { timeout: TIMEOUT, retry: 1 },
        async () => {
            const transport = live.setup();
            let firstPublishCount = 0;
            let retryPublishCount = 0;
            let first: CanonicalExecutionEventStream | undefined;
            let retry: CanonicalExecutionEventStream | undefined;
            try {
                first = await transport.driver.streamCanonicalEvents(
                    [{ role: PromptRole.user, content: 'Call lookup once with key exactly "typed".' }],
                    {
                        ...options('first'),
                        on_canonical_request_prepared: async () => {
                            firstPublishCount += 1;
                        },
                    },
                    undefined,
                    { stream_id: 'live:canonical:mistral:first' },
                );
                const events = await collect(first);
                expect(events[0]).toMatchObject({ type: 'draft_started', origin: 'live_transport' });
                expect(events.at(-1)).toMatchObject({ type: 'response_accepted' });
                expect(firstPublishCount).toBe(1);
                expect(transport.transportCalls()).toBe(1);
                const toolCall = first.completion?.accepted_output.turn.blocks.find(
                    (block) => block.type === 'tool_call',
                );
                expect(toolCall).toMatchObject({
                    type: 'tool_call',
                    executor: 'application',
                    tool_name: 'lookup',
                    arguments: { type: 'json', value: { key: 'typed' } },
                });
                expect(first.completion?.accepted_output.generation).toMatchObject({
                    provider: 'mistralai',
                    protocol: 'openai.chat.completions',
                    requested_model: MODEL,
                    status: 'completed',
                });

                const persisted = JSON.parse(JSON.stringify(first.completion?.conversation)) as ConversationDocument;
                retry = await transport.driver.streamCanonicalEvents(
                    [{ role: PromptRole.user, content: 'Call lookup once with key exactly "typed".' }],
                    {
                        ...options('retry', persisted),
                        on_canonical_request_prepared: async () => {
                            retryPublishCount += 1;
                        },
                    },
                    undefined,
                    { stream_id: 'live:canonical:mistral:retry' },
                );
                expect(await collect(retry)).toEqual([
                    expect.objectContaining({
                        type: 'response_accepted',
                        sequence: 0,
                        origin: 'accepted_recovery',
                    }),
                ]);
                expect(retry.completion?.accepted_output).toEqual(first.completion?.accepted_output);
                expect(retryPublishCount).toBe(0);
                expect(transport.transportCalls()).toBe(1);
            } finally {
                for (const stream of [retry, first]) {
                    if (stream === undefined) continue;
                    try {
                        await stream.cancel();
                    } finally {
                        await stream.closed;
                    }
                }
                transport.driver.destroy();
                transport.restore();
            }
        },
    );
});
