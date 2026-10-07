import type { ConversationDocument, ConversationStreamEvent } from '@llumiverse/conversation';
import {
    type AbstractDriver,
    type CanonicalExecutionEventStream,
    type CanonicalExecutionInputOptions,
    type ExecutionOptions,
    PromptRole,
    type PromptSegment,
} from '@llumiverse/core';
import 'dotenv/config';
import { describe, expect, test } from 'vitest';
import { BedrockDriver, OpenAIDriver, VertexAIDriver } from '../src/index.js';
import { OpenAIChatCompletionsDriver } from '../src/openai/openai_chat_completions.js';
import { selectLiveTestDrivers } from './live-model-selection.js';

const TIMEOUT = 90_000;
const RESULT_SCHEMA = {
    type: 'object' as const,
    properties: { answer: { type: 'string' as const } },
    required: ['answer'],
    additionalProperties: false,
};
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

type Coverage = 'structured' | 'tool';

interface TypedLiveDriver {
    name: string;
    models: string[];
    protocol: string;
    coverage: Coverage;
    option_id: string;
    setup(): Promise<{
        driver: AbstractDriver;
        transportCalls(): number;
        restore(): void;
    }>;
}

const liveDrivers: TypedLiveDriver[] = [];

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

if (process.env.OPENAI_API_KEY) {
    liveDrivers.push({
        name: 'openai',
        models: ['gpt-4o-mini'],
        protocol: 'openai.responses',
        coverage: 'structured',
        option_id: 'openai-text',
        setup: async () => {
            const driver = new OpenAIDriver({ apiKey: process.env.OPENAI_API_KEY });
            const transport = countMethodCalls(driver.service.responses, 'create');
            return {
                driver,
                transportCalls: transport.calls,
                restore: transport.restore,
            };
        },
    });
    liveDrivers.push({
        name: 'openai',
        models: ['gpt-4o-mini'],
        protocol: 'openai.chat.completions',
        coverage: 'tool',
        option_id: 'openai-text',
        setup: async () => {
            const driver = new OpenAIChatCompletionsDriver({
                apiKey: process.env.OPENAI_API_KEY as string,
                endpoint: 'https://api.openai.com/v1',
            });
            const transport = countMethodCalls(driver.service.chat.completions, 'create');
            return {
                driver,
                transportCalls: transport.calls,
                restore: transport.restore,
            };
        },
    });
}

if (process.env.GOOGLE_PROJECT_ID && process.env.GOOGLE_REGION) {
    liveDrivers.push({
        name: 'google-vertex',
        models: ['publishers/anthropic/models/claude-sonnet-4-5'],
        protocol: 'anthropic.messages',
        coverage: 'tool',
        option_id: 'vertexai-claude',
        setup: async () => {
            const driver = new VertexAIDriver({
                project: process.env.GOOGLE_PROJECT_ID as string,
                region: process.env.GOOGLE_REGION as string,
            });
            const client = await driver.getAnthropicClient();
            const transport = countMethodCalls(client.messages, 'stream');
            return {
                driver,
                transportCalls: transport.calls,
                restore: transport.restore,
            };
        },
    });
    liveDrivers.push({
        name: 'google-vertex',
        models: ['publishers/google/models/gemini-2.5-flash'],
        protocol: 'google.generate_content',
        coverage: 'structured',
        option_id: 'vertexai-gemini',
        setup: async () => {
            const driver = new VertexAIDriver({
                project: process.env.GOOGLE_PROJECT_ID as string,
                region: process.env.GOOGLE_REGION as string,
            });
            const client = driver.getGoogleGenAIClient();
            const transport = countMethodCalls(client.models, 'generateContentStream');
            return {
                driver,
                transportCalls: transport.calls,
                restore: transport.restore,
            };
        },
    });
}

if (process.env.BEDROCK_REGION) {
    liveDrivers.push({
        name: 'bedrock',
        models: ['global.anthropic.claude-haiku-4-5-20251001-v1:0'],
        protocol: 'aws.bedrock.converse',
        coverage: 'tool',
        option_id: 'bedrock-claude',
        setup: async () => {
            const driver = new BedrockDriver({ region: process.env.BEDROCK_REGION as string });
            const executor = driver.getExecutor();
            const transport = countMethodCalls(executor, 'converseStream');
            return {
                driver,
                transportCalls: transport.calls,
                restore: transport.restore,
            };
        },
    });
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
    liveDrivers.length !== 5
) {
    throw new Error(`Canonical typed live coverage requires all five protocols; configured ${liveDrivers.length}`);
}

function runtimeOptions(
    live: TypedLiveDriver,
    model: string,
    attempt: string,
    conversation: ConversationDocument | undefined,
    publish: () => Promise<void>,
): CanonicalExecutionInputOptions {
    const scope = live.protocol.replaceAll('.', '-');
    const toolOptions = live.coverage === 'tool' ? { required_tool_name: 'lookup', tool_choice: 'required' } : {};
    const modelOptions = {
        _option_id: live.option_id,
        max_tokens: 128,
        ...toolOptions,
    } as ExecutionOptions['model_options'];
    return {
        model,
        ...(conversation === undefined ? {} : { conversation }),
        model_options: modelOptions,
        ...(live.coverage === 'tool' ? { tools: TOOLS } : { result_schema: RESULT_SCHEMA }),
        on_canonical_request_prepared: publish,
        conversation_runtime: {
            conversation_id: `live:typed:${scope}`,
            request_id: `live:typed:${scope}:request`,
            attempt_id: `live:typed:${scope}:attempt:${attempt}`,
            input_operation_id: `live:typed:${scope}:input`,
            response_operation_id: `live:typed:${scope}:response`,
            recorded_at: '2026-09-30T00:00:00.000Z',
            started_at: '2026-09-30T00:00:00.000Z',
            completed_at: '2026-09-30T00:00:00.000Z',
        },
    };
}

function promptFor(coverage: Coverage): PromptSegment[] {
    return [
        {
            role: PromptRole.user,
            content:
                coverage === 'tool'
                    ? 'Call the lookup tool with key set to "typed". Do not answer in prose.'
                    : 'Return only a JSON object with answer set to "typed".',
        },
    ];
}

async function collect(stream: CanonicalExecutionEventStream): Promise<ConversationStreamEvent[]> {
    const events: ConversationStreamEvent[] = [];
    for await (const event of stream) events.push(event);
    return events;
}

// Closure-owned counters keep concurrent provider observations isolated from Vitest mock state.
describe.concurrent.each(selectedDrivers)('$protocol canonical typed live stream', (live) => {
    test.each(live.models)(
        '$protocol accepts authoritative output and exact-retries without transport for %s',
        { timeout: TIMEOUT, retry: 1, concurrent: false },
        async (model) => {
            const transport = await live.setup();
            const streams: CanonicalExecutionEventStream[] = [];
            try {
                const prompt = promptFor(live.coverage);
                let publishCount = 0;
                const publish = async () => {
                    publishCount += 1;
                };
                const first = await transport.driver.streamCanonicalEvents(
                    prompt,
                    runtimeOptions(live, model, 'first', undefined, publish),
                    undefined,
                    { stream_id: `live:typed:${live.protocol}:first` },
                );
                streams.push(first);
                const events = await collect(first);

                expect(events[0]).toMatchObject({ type: 'draft_started', sequence: 0 });
                expect(events.at(-1)).toMatchObject({ type: 'response_accepted', origin: 'live_transport' });
                expect(events).toContainEqual(
                    expect.objectContaining({ native_position: expect.objectContaining({ protocol: live.protocol }) }),
                );
                expect(first.completion).toBeDefined();
                expect(publishCount).toBe(1);
                expect(transport.transportCalls()).toBe(1);

                const acceptedBlocks = first.completion?.accepted_output.turn.blocks ?? [];
                if (live.coverage === 'tool') {
                    expect(acceptedBlocks).toContainEqual(
                        expect.objectContaining({
                            type: 'tool_call',
                            executor: 'application',
                            tool_name: 'lookup',
                            arguments: { type: 'json', value: { key: 'typed' } },
                        }),
                    );
                } else {
                    expect(acceptedBlocks).toContainEqual(
                        expect.objectContaining({ type: 'json', value: { answer: 'typed' } }),
                    );
                }

                if (first.completion === undefined) throw new Error('Expected an accepted canonical response');
                let retryPublishCount = 0;
                const retryPublish = async () => {
                    retryPublishCount += 1;
                };
                const recovered = await transport.driver.streamCanonicalEvents(
                    prompt,
                    runtimeOptions(
                        live,
                        model,
                        'retry',
                        JSON.parse(JSON.stringify(first.completion.conversation)) as ConversationDocument,
                        retryPublish,
                    ),
                    undefined,
                    { stream_id: `live:typed:${live.protocol}:retry` },
                );
                streams.push(recovered);
                const recoveredEvents = await collect(recovered);

                expect(recoveredEvents).toEqual([
                    expect.objectContaining({ type: 'response_accepted', sequence: 0, origin: 'accepted_recovery' }),
                ]);
                expect(recovered.completion?.accepted_output).toEqual(first.completion.accepted_output);
                expect(retryPublishCount).toBe(0);
                expect(transport.transportCalls()).toBe(1);
            } finally {
                await Promise.allSettled(
                    streams.map(async (stream) => {
                        try {
                            await stream.cancel();
                        } finally {
                            await stream.closed;
                        }
                    }),
                );
                transport.driver.destroy();
                transport.restore();
            }
        },
    );
});
