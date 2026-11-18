/**
 * Records what each driver on the shared OpenAI Responses protocol sends and returns, so changes to how drivers reach
 * the protocol can be checked for unchanged wire behaviour: the request (URL, method, non-auth headers, body), the
 * completion, the stored conversation and the driver's log calls, per driver and scenario.
 */
import { type AbstractDriver, type ExecutionOptions, type JSONSchema, PromptRole } from '@llumiverse/core';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AzureFoundryDriver } from '../src/azure/azure_foundry.js';
import { BedrockMantleDriver } from '../src/bedrock-mantle/index.js';
import { AzureOpenAIDriver } from '../src/openai/azure_openai.js';
import { OpenAIDriver } from '../src/openai/openai.js';
import { OpenAIResponsesDriver } from '../src/openai/openai_responses.js';
import { VertexAIDriver } from '../src/vertexai/index.js';
import { xAIDriver } from '../src/xai/index.js';

type LogCall = [level: string, ...args: unknown[]];

function recordingLogger(calls: LogCall[]) {
    const log =
        (level: string) =>
        (...args: unknown[]) =>
            void calls.push([level, ...args.map(serializable)]);
    const logger = {
        debug: log('debug'),
        info: log('info'),
        warn: log('warn'),
        error: log('error'),
        child: () => logger,
    };
    return logger;
}

function serializable(value: unknown): unknown {
    if (value instanceof Error) {
        const { code, retryable } = value as Error & { code?: unknown; retryable?: unknown };
        return { error: value.name, message: value.message, code, retryable };
    }
    try {
        return JSON.parse(JSON.stringify(value));
    } catch {
        return String(value);
    }
}

class TestVertexAIDriver extends VertexAIDriver {
    constructor(logger: unknown) {
        super({ project: 'test-project', region: 'us-central1', logger: logger as never });
        this.googleAuth = {
            getAccessToken: async () => 'test-token',
            getRequestHeaders: async () => new Headers({ authorization: 'Bearer test-token' }),
        } as unknown as VertexAIDriver['googleAuth'];
    }
}

type DriverCase = {
    name: string;
    model: string;
    create: (logger: unknown) => AbstractDriver;
    /** Scenarios a driver does not take part in, such as Azure's image-generation headers elsewhere. */
    skip?: readonly string[];
};

const DRIVERS: DriverCase[] = [
    { name: 'openai', model: 'gpt-6-luna', create: (logger) => new OpenAIDriver({ apiKey: 'key', logger } as never) },
    {
        name: 'azure-openai-v1',
        model: 'gpt-6-luna',
        create: (logger) =>
            new AzureOpenAIDriver({ apiKey: 'key', endpoint: 'https://resource.openai.azure.com', logger } as never),
    },
    {
        name: 'azure-openai-dated',
        model: 'gpt-6-luna',
        create: (logger) =>
            new AzureOpenAIDriver({
                apiKey: 'key',
                endpoint: 'https://resource.openai.azure.com',
                apiVersion: '2025-04-01-preview',
                logger,
            } as never),
    },
    {
        name: 'azure-foundry',
        model: 'luna::gpt-6-luna',
        create: (logger) =>
            new AzureFoundryDriver({
                endpoint: 'https://foundry.example.test/api/projects/project',
                // API-key auth: the protocol follows the model ID, without a deployment lookup.
                apiKey: 'key',
                logger,
            } as never),
    },
    { name: 'xai', model: 'grok-4.20', create: (logger) => new xAIDriver({ apiKey: 'key', logger } as never) },
    {
        name: 'compatible-minimax',
        model: 'minimax/minimax-m3',
        create: (logger) =>
            new OpenAIResponsesDriver({ apiKey: 'key', endpoint: 'https://compatible.example/v1', logger } as never),
    },
    {
        name: 'compatible-glm',
        model: 'z-ai/glm-5.3',
        create: (logger) =>
            new OpenAIResponsesDriver({ apiKey: 'key', endpoint: 'https://compatible.example/v1', logger } as never),
    },
    {
        name: 'bedrock-mantle',
        model: 'openai.gpt-6.1-terra',
        // Mantle's SDK client retries failed requests with backoff, so its error scenarios would wait on real timers.
        skip: ['error-rate-limit', 'error-rate-limit-stream', 'error-network'],
        create: (logger) =>
            new BedrockMantleDriver({
                region: 'us-west-2',
                credentials: { accessKeyId: 'AKIDTEST', secretAccessKey: 'secret' },
                logger,
            } as never),
    },
    {
        name: 'vertex-muse',
        model: 'publishers/meta/models/muse-spark-1.3',
        create: (logger) => new TestVertexAIDriver(logger),
    },
    {
        name: 'vertex-gpt',
        model: 'locations/us/publishers/openai/models/gpt-6-luna',
        create: (logger) => new TestVertexAIDriver(logger),
    },
];

const schema: JSONSchema = {
    type: 'object',
    properties: { color: { type: 'string' } },
    required: ['color'],
    additionalProperties: false,
};

const weatherTool = {
    name: 'get_weather',
    description: 'Get the weather for a city',
    input_schema: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
};

function responseBody(model: string, output: unknown[]) {
    return {
        id: 'resp_1',
        object: 'response',
        created_at: 0,
        model,
        status: 'completed',
        error: null,
        incomplete_details: null,
        output,
        usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    };
}

const textOutput = (text: string) => [
    { type: 'reasoning', id: 'rs_1', summary: [], encrypted_content: 'opaque' },
    {
        type: 'message',
        id: 'msg_1',
        role: 'assistant',
        status: 'completed',
        content: [{ type: 'output_text', text, annotations: [] }],
    },
];

const toolOutput = [
    {
        type: 'function_call',
        id: 'fc_1',
        call_id: 'call_1',
        name: 'get_weather',
        arguments: '{"city":"Lyon"}',
        status: 'completed',
    },
];

type Scenario = {
    name: string;
    stream?: boolean;
    options?: Partial<ExecutionOptions>;
    modelOptions?: Record<string, unknown>;
    output?: unknown[];
    /** Run a second turn with the conversation returned by the first. */
    secondTurn?: boolean;
    /** The second turn sends the result of the first turn's tool call, and is answered with this output. */
    toolResultOutput?: unknown[];
    compact?: boolean;
    /** Answer with this provider error, or fail the request at the network. */
    failure?: 'rate-limit' | 'network';
};

const SCENARIOS: Scenario[] = [
    { name: 'execute' },
    { name: 'stream', stream: true },
    { name: 'tools', options: { tools: [weatherTool] as never }, output: toolOutput },
    { name: 'tools-stream', stream: true, options: { tools: [weatherTool] as never }, output: toolOutput },
    { name: 'result-schema', options: { result_schema: schema }, output: textOutput('{"color":"green"}') },
    {
        name: 'result-schema-tools',
        options: { result_schema: schema, tools: [weatherTool] as never },
        output: textOutput('{"color":"green"}'),
    },
    {
        name: 'result-schema-suffix',
        options: { result_schema: schema, prompt_cache_schema_suffix: true },
        output: textOutput('{"color":"green"}'),
    },
    {
        name: 'multi-turn',
        options: { result_schema: schema },
        output: textOutput('{"color":"green"}'),
        secondTurn: true,
    },
    { name: 'effort', modelOptions: { effort: 'low' } },
    { name: 'prompt-cache-key', options: { prompt_cache_key: 'cache-key-1' }, secondTurn: true },
    { name: 'max-tokens-temperature', modelOptions: { max_tokens: 256, temperature: 0.3 } },
    { name: 'compact', compact: true },
    {
        name: 'image-generation-tool',
        modelOptions: { image_generation: { model: 'gpt-image-2' } },
    },
    {
        name: 'tool-result',
        options: { tools: [weatherTool] as never },
        output: toolOutput,
        secondTurn: true,
        toolResultOutput: textOutput('It is sunny in Lyon'),
    },
    {
        name: 'tool-result-stream',
        stream: true,
        options: { tools: [weatherTool] as never },
        output: toolOutput,
        secondTurn: true,
        toolResultOutput: textOutput('It is sunny in Lyon'),
    },
    { name: 'error-rate-limit', failure: 'rate-limit' },
    { name: 'error-rate-limit-stream', stream: true, failure: 'rate-limit' },
    { name: 'error-network', failure: 'network' },
];

/** Headers that identify the client or carry credentials, which vary by SDK version and are not wire behaviour. */
const IGNORED_HEADERS = /^(authorization|api-key|user-agent|x-stainless-|x-amz-|idempotency-key|content-length)/i;

type Recorded = { method: string; url: string; headers: Record<string, string>; body: unknown };

function sse(events: unknown[]): string {
    return events
        .map((event) => `event: ${(event as { type: string }).type}\ndata: ${JSON.stringify(event)}\n\n`)
        .join('');
}

function streamEvents(model: string, output: unknown[]) {
    const events: unknown[] = [];
    for (const [index, item] of output.entries()) {
        if ((item as { type: string }).type === 'message') {
            const text = (item as { content: Array<{ text: string }> }).content[0].text;
            events.push({
                type: 'response.output_text.delta',
                item_id: 'msg_1',
                output_index: index,
                content_index: 0,
                delta: text,
            });
        }
        events.push({ type: 'response.output_item.done', output_index: index, item });
    }
    events.push({ type: 'response.completed', response: responseBody(model, output) });
    return events;
}

describe('OpenAI Responses protocol baseline', () => {
    let requests: Recorded[];
    let nextOutput: unknown[];
    let compactResponse: boolean;
    let failure: Scenario['failure'];

    beforeEach(() => {
        requests = [];
        nextOutput = textOutput('Hello');
        compactResponse = false;
        failure = undefined;
        vi.stubGlobal(
            'fetch',
            vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
                const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
                const headers: Record<string, string> = {};
                for (const [key, value] of new Headers(init?.headers)) {
                    if (!IGNORED_HEADERS.test(key)) headers[key] = value;
                }
                const text = typeof init?.body === 'string' ? init.body : '';
                const body = text ? JSON.parse(text) : undefined;
                requests.push({ method: init?.method ?? 'GET', url: url.replace(/X-Amz-[^&]+&?/g, ''), headers, body });
                const model = (body as { model?: string } | undefined)?.model ?? 'model';
                if (failure === 'network') throw new TypeError('fetch failed');
                if (failure === 'rate-limit') {
                    return Response.json(
                        {
                            error: {
                                message: 'Rate limit reached',
                                type: 'rate_limit_error',
                                code: 'rate_limit_exceeded',
                            },
                        },
                        { status: 429 },
                    );
                }
                if (compactResponse) {
                    return Response.json({
                        id: 'cmp_1',
                        object: 'response.compaction',
                        created_at: 0,
                        output: [
                            { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'Say hello' }] },
                            { type: 'compaction', id: 'cmp_item_1', encrypted_content: 'opaque' },
                        ],
                        usage: { input_tokens: 10, output_tokens: 2, total_tokens: 12 },
                    });
                }
                if ((body as { stream?: boolean } | undefined)?.stream) {
                    return new Response(sse(streamEvents(model, nextOutput)), {
                        headers: { 'content-type': 'text/event-stream' },
                    });
                }
                return Response.json(responseBody(model, nextOutput));
            }),
        );
    });
    afterEach(() => vi.unstubAllGlobals());

    for (const driverCase of DRIVERS) {
        for (const scenario of SCENARIOS) {
            if (driverCase.skip?.includes(scenario.name)) continue;
            it(`${driverCase.name}: ${scenario.name}`, async () => {
                const logs: LogCall[] = [];
                const driver = driverCase.create(recordingLogger(logs));
                nextOutput = scenario.output ?? textOutput('Hello');
                compactResponse = scenario.compact === true;
                failure = scenario.failure;
                const options: ExecutionOptions = {
                    model: driverCase.model,
                    ...scenario.options,
                    ...(scenario.compact ? { compact_conversation: {} } : {}),
                    model_options: { _option_id: 'openai-text', ...scenario.modelOptions } as never,
                };
                const turns: unknown[] = [];
                let conversation: unknown;
                for (const turn of scenario.secondTurn ? [1, 2] : [1]) {
                    const toolResultTurn = turn === 2 && scenario.toolResultOutput !== undefined;
                    if (toolResultTurn) nextOutput = scenario.toolResultOutput ?? nextOutput;
                    const segments = toolResultTurn
                        ? [{ role: PromptRole.tool, tool_use_id: 'call_1', content: '{"weather":"sunny"}' }]
                        : [{ role: PromptRole.user, content: turn === 1 ? 'Say hello' : 'Again' }];
                    try {
                        if (scenario.stream) {
                            const stream = await driver.stream(segments, { ...options, conversation });
                            const chunks: unknown[] = [];
                            for await (const chunk of stream) chunks.push(chunk);
                            const completion = stream.completion;
                            conversation = completion?.conversation;
                            turns.push({
                                chunks,
                                result: completion?.result,
                                tool_use: completion?.tool_use,
                                finish_reason: completion?.finish_reason,
                                token_usage: completion?.token_usage,
                                conversation,
                            });
                        } else {
                            const response = await driver.execute(segments, { ...options, conversation });
                            conversation = response.conversation;
                            turns.push({
                                result: response.result,
                                tool_use: response.tool_use,
                                finish_reason: response.finish_reason,
                                token_usage: response.token_usage,
                                conversation,
                            });
                        }
                    } catch (error: unknown) {
                        turns.push({ error: serializable(error) });
                        break;
                    }
                }
                driver.destroy();
                expect({ requests, turns, logs }).toMatchSnapshot();
            });
        }
    }
});
