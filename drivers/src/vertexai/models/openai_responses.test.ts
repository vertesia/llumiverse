import { type JSONSchema, PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { VertexAIDriver } from '../index.js';
import { getModelDefinition } from '../models.js';
import { OpenAIChatCompletionsModelDefinition } from './openai_chat_completions.js';
import { VertexAIResponsesModelDefinition } from './openai_responses.js';

class TestVertexAIDriver extends VertexAIDriver {
    constructor(private readonly fetchMock: typeof fetch) {
        super({ project: 'test-project', region: 'us-central1' });
        // The SDK's Vertex provider authenticates each attempt through the Google client's request headers.
        this.googleAuth = {
            getAccessToken: async () => 'test-token',
            getRequestHeaders: async () => new Headers({ authorization: 'Bearer test-token' }),
        } as unknown as VertexAIDriver['googleAuth'];
    }

    protected override getDriverFetch(): typeof fetch {
        return this.fetchMock;
    }
}

function responseBody(model: string, text = 'Bonjour') {
    return {
        id: 'resp_1',
        object: 'response',
        created_at: 0,
        model: `meta/${model}`,
        status: 'completed',
        error: null,
        incomplete_details: null,
        output: [
            { type: 'reasoning', id: 'rs_1', summary: [], encrypted_content: 'opaque' },
            {
                type: 'message',
                id: 'msg_1',
                role: 'assistant',
                status: 'completed',
                content: [{ type: 'output_text', text, annotations: [] }],
            },
        ],
        usage: { input_tokens: 10, output_tokens: 20, total_tokens: 30 },
    };
}

async function captureRequest(model: string, modelOptions: Record<string, unknown>, resultSchema?: JSONSchema) {
    const fetchMock = vi.fn(
        async () =>
            new Response(JSON.stringify(responseBody(model.split('/').pop() ?? model)), {
                status: 200,
                headers: { 'Content-Type': 'application/json' },
            }),
    );
    const driver = new TestVertexAIDriver(fetchMock as unknown as typeof fetch);
    const completion = await driver.execute([{ role: PromptRole.user, content: 'Say hello' }], {
        model,
        result_schema: resultSchema,
        model_options: { _option_id: 'openai-text', ...modelOptions },
    });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    const headers = new Headers(init.headers);
    return {
        completion,
        url,
        headers,
        body: JSON.parse(String(init.body)),
        authorization: headers.get('authorization'),
    };
}

describe('Vertex Meta Muse routing', () => {
    it.each([
        'publishers/meta/models/muse-wonder-1.0',
        'locations/global/publishers/meta/models/muse-spark-1.3',
        'publishers/meta/models/muse-ember-1.0',
        // Later generations keep the Responses transport.
        'publishers/meta/models/muse-spark-1.4',
        'publishers/meta/models/muse-wonder-2.0@001',
        'muse-ember-1.0',
        // The `publisher/model` form Vertex model cards use.
        'meta/muse-wonder-1.0',
    ])('routes %s to the Responses API', (model) => {
        expect(getModelDefinition(model)).toBeInstanceOf(VertexAIResponsesModelDefinition);
    });

    it('keeps Llama on Chat Completions', () => {
        expect(getModelDefinition('publishers/meta/models/llama-4-scout-17b-16e-instruct-maas')).toBeInstanceOf(
            OpenAIChatCompletionsModelDefinition,
        );
    });
});

describe('Vertex OpenAI GPT routing', () => {
    it.each([
        ['locations/global/publishers/openai/models/gpt-6-luna', 'global'],
        ['publishers/openai/models/gpt-6.1-sol', 'global'],
        ['locations/us/publishers/openai/models/gpt-6-astra', 'us'],
        ['openai/gpt-6.1-terra', 'global'],
        ['gpt-6-luna', 'global'],
        // A later generation keeps the Responses transport.
        ['publishers/openai/models/gpt-7', 'global'],
    ])('routes %s to the OpenAI Responses API in %s', (model, region) => {
        const definition = getModelDefinition(model);
        expect(definition).toBeInstanceOf(VertexAIResponsesModelDefinition);
        expect(definition).toMatchObject({ publisher: 'openai', region });
    });

    it.each(['publishers/openai/models/gpt-oss-120b-maas', 'openai/gpt-oss-20b-maas'])(
        'keeps %s on Chat Completions',
        (model) => {
            expect(getModelDefinition(model)).toBeInstanceOf(OpenAIChatCompletionsModelDefinition);
        },
    );
});

describe('Vertex OpenAI GPT requests', () => {
    const schema: JSONSchema = {
        type: 'object',
        properties: { color: { type: 'string' } },
        required: ['color'],
        additionalProperties: false,
    };

    it('posts to the OpenAI publisher endpoint of the US multi-region with the bare model ID', async () => {
        const { url, body } = await captureRequest('locations/us/publishers/openai/models/gpt-6-luna', {
            effort: 'low',
            max_tokens: 512,
        });
        expect(url).toBe(
            'https://aiplatform.us.rep.googleapis.com/v1/projects/test-project/locations/us/publishers/openai/v1/responses',
        );
        expect(body).toMatchObject({ model: 'gpt-6-luna', max_output_tokens: 512, reasoning: { effort: 'low' } });
        // Vertex does not offer reasoning summaries for OpenAI models.
        expect(body.reasoning).not.toHaveProperty('summary');
        expect(body.include).toEqual(['reasoning.encrypted_content']);
    });

    it('leaves out the reasoning summary on streamed requests too', async () => {
        const events = [
            { type: 'response.output_text.delta', item_id: 'msg_1', output_index: 1, content_index: 0, delta: 'Hi' },
            { type: 'response.completed', response: responseBody('gpt-6-luna', 'Hi') },
        ];
        const fetchMock = vi.fn(
            async () =>
                new Response(
                    events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(''),
                    {
                        status: 200,
                        headers: { 'Content-Type': 'text/event-stream' },
                    },
                ),
        );
        const driver = new TestVertexAIDriver(fetchMock as unknown as typeof fetch);
        const stream = await driver.stream([{ role: PromptRole.user, content: 'Say hello' }], {
            model: 'publishers/openai/models/gpt-6-luna',
            model_options: { _option_id: 'openai-thinking', effort: 'low' },
        });
        let text = '';
        for await (const chunk of stream) text += chunk;
        const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
        const body = JSON.parse(String(init.body));
        expect(url).toBe(
            'https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/openai/v1/responses',
        );
        expect(body).toMatchObject({ stream: true, model: 'gpt-6-luna', reasoning: { effort: 'low' } });
        expect(body.reasoning).not.toHaveProperty('summary');
        expect(text).toBe('Hi');
    });

    it('sends the result schema natively, without a prompt copy', async () => {
        const { body } = await captureRequest('publishers/openai/models/gpt-6.1-sol', {}, schema);
        expect(body.text?.format).toMatchObject({ type: 'json_schema' });
        expect(JSON.stringify(body.input)).not.toContain('<response_schema>');
    });

    it('keeps the prompt copy for Muse', async () => {
        const { body } = await captureRequest('publishers/meta/models/muse-spark-1.3', {}, schema);
        expect(JSON.stringify(body.input)).toContain('<response_schema>');
    });
});

describe('Vertex Meta Muse requests', () => {
    it('posts a stateless Responses request with the bare model ID to the global Meta endpoint', async () => {
        const { completion, url, body, authorization } = await captureRequest(
            'publishers/meta/models/muse-wonder-1.0',
            { max_tokens: 4096, temperature: 0.5 },
        );
        expect(url).toBe(
            'https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/meta/v1/responses',
        );
        expect(authorization).toBe('Bearer test-token');
        expect(body).toMatchObject({ model: 'muse-wonder-1.0', max_output_tokens: 4096, temperature: 0.5 });
        expect(body).not.toHaveProperty('store');
        // Omitted effort stays omitted, so Meta's default applies.
        expect(body).not.toHaveProperty('reasoning');
        expect(completion.result).toEqual([{ type: 'text', value: 'Bonjour' }]);
    });

    it('sends a supplied effort with encrypted reasoning for stateless replay', async () => {
        const { body } = await captureRequest('publishers/meta/models/muse-ember-1.0', { effort: 'max' });
        expect(body.reasoning).toMatchObject({ effort: 'max' });
        expect(body.include).toEqual(['reasoning.encrypted_content']);
    });

    it('keeps an explicit model-path location', async () => {
        const { url } = await captureRequest('locations/us-central1/publishers/meta/models/muse-spark-1.3', {});
        expect(url).toBe(
            'https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/meta/v1/responses',
        );
    });

    it('sends the result schema both natively and in the prompt', async () => {
        const schema: JSONSchema = {
            type: 'object',
            properties: { color: { type: 'string' } },
            required: ['color'],
            additionalProperties: false,
        };
        const { body } = await captureRequest('publishers/meta/models/muse-spark-1.3', {}, schema);
        expect(body.text.format).toMatchObject({ type: 'json_schema', strict: true });
        expect(body.input[0].content[0].text).toContain(`<response_schema>${JSON.stringify(schema)}</response_schema>`);
    });
});

describe('Vertex Meta Muse transport', () => {
    it('sends no OpenAI organization or project from the environment to Vertex', async () => {
        vi.stubEnv('OPENAI_ORG_ID', 'org-from-env');
        vi.stubEnv('OPENAI_PROJECT_ID', 'proj_from_env');
        try {
            const { headers, authorization } = await captureRequest('publishers/meta/models/muse-spark-1.3', {});
            expect(authorization).toBe('Bearer test-token');
            expect(headers.get('openai-organization')).toBeNull();
            expect(headers.get('openai-project')).toBeNull();
        } finally {
            vi.unstubAllEnvs();
        }
    });
});

describe('Vertex Meta Muse result schema across turns', () => {
    const colour: JSONSchema = {
        type: 'object',
        properties: { color: { type: 'string' } },
        required: ['color'],
        additionalProperties: false,
    };
    const shade: JSONSchema = {
        type: 'object',
        properties: { shade: { type: 'string' } },
        required: ['shade'],
        additionalProperties: false,
    };

    function schemaCopies(input: Array<{ content?: unknown }>): string[] {
        return input.flatMap((item) => {
            const parts = Array.isArray(item.content) ? item.content : [{ text: item.content }];
            return parts
                .map((part: { text?: unknown }) => part.text)
                .filter((text): text is string => typeof text === 'string' && text.includes('<response_schema>'));
        });
    }

    async function runTurns(schemas: JSONSchema[]) {
        const fetchMock = vi.fn(
            async () =>
                new Response(JSON.stringify(responseBody('muse-spark-1.3', '{"color":"green","shade":"dark"}')), {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                }),
        );
        const driver = new TestVertexAIDriver(fetchMock as unknown as typeof fetch);
        let conversation: unknown;
        for (const schema of schemas) {
            const completion = await driver.execute([{ role: PromptRole.user, content: 'What colour is grass?' }], {
                model: 'publishers/meta/models/muse-spark-1.3',
                result_schema: schema,
                conversation,
                model_options: { _option_id: 'openai-text' },
            });
            conversation = completion.conversation;
        }
        const calls = fetchMock.mock.calls as unknown as Array<[string, RequestInit]>;
        return calls.map(([, init]) => JSON.parse(String(init.body)).input as Array<{ content?: unknown }>);
    }

    it('keeps one copy of an unchanged schema, at the position of its first turn', async () => {
        const inputs = await runTurns([colour, colour, colour]);
        for (const input of inputs) expect(schemaCopies(input)).toHaveLength(1);
        // The stored copy keeps its place, so the request prefix stays the same from turn to turn.
        expect(JSON.stringify(inputs[2].slice(0, 2))).toBe(JSON.stringify(inputs[1].slice(0, 2)));
    });

    it('replaces the stored copy when the schema changes', async () => {
        const inputs = await runTurns([colour, shade]);
        const copies = schemaCopies(inputs[1]);
        expect(copies).toHaveLength(1);
        expect(copies[0]).toContain('"shade"');
    });
});

describe('Vertex Responses errors', () => {
    const model = 'publishers/meta/models/muse-spark-1.3';

    it('keeps the OpenAI-compatible format for publisher endpoint errors', async () => {
        const fetchMock = vi.fn(
            async () =>
                new Response(
                    JSON.stringify({ error: { message: 'Quota exceeded', type: 'rate_limit', code: 'rate_limited' } }),
                    { status: 429, headers: { 'Content-Type': 'application/json', 'x-request-id': 'req_1' } },
                ),
        );
        const driver = new TestVertexAIDriver(fetchMock as unknown as typeof fetch);

        await expect(
            driver.execute([{ role: PromptRole.user, content: 'Say hello' }], {
                model,
                model_options: { _option_id: 'openai-text' },
            }),
        ).rejects.toMatchObject({
            code: 429,
            retryable: true,
            message: expect.stringMatching(/^\[vertexai\] \[429\] .*Quota exceeded.*\(Request ID: req_1\)/),
        });
        expect(fetchMock).toHaveBeenCalledTimes(1);
    });

    it("gives other errors the Vertex driver's default format", () => {
        const driver = new TestVertexAIDriver(vi.fn() as unknown as typeof fetch);
        const formatted = driver.formatLlumiverseError(new Error('socket closed'), {
            provider: 'vertexai',
            model,
            operation: 'execute',
        });

        expect(formatted).toMatchObject({ context: { provider: 'vertexai', model } });
        expect(formatted.message).toContain('socket closed');
    });
});

describe('Vertex OpenAI SDK auth errors', () => {
    const model = 'publishers/meta/models/muse-spark-1.3';
    const execute = (driver: VertexAIDriver) =>
        driver.execute([{ role: PromptRole.user, content: 'Say hello' }], {
            model,
            model_options: { _option_id: 'openai-text' },
        });

    function driverWithAuth(getRequestHeaders: () => Promise<Headers>) {
        const fetchMock = vi.fn(
            async () =>
                new Response(JSON.stringify(responseBody('muse-spark-1.3')), {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                }),
        );
        const driver = new TestVertexAIDriver(fetchMock as unknown as typeof fetch);
        driver.googleAuth = { getRequestHeaders } as unknown as VertexAIDriver['googleAuth'];
        return { driver, fetchMock };
    }

    it('reports the Google auth failure the SDK leaves out, and does not retry a revoked grant', async () => {
        const secret = { refresh_token: 'never-in-the-error' };
        const failure = Object.assign(new Error('invalid_grant: Token has been expired or revoked.'), {
            config: { data: secret },
        });
        const { driver, fetchMock } = driverWithAuth(async () => {
            throw failure;
        });

        const error = await execute(driver).then(
            () => undefined,
            (caught: unknown) => caught,
        );

        expect(fetchMock).not.toHaveBeenCalled();
        expect(error).toMatchObject({
            retryable: false,
            message: '[vertexai] Vertex authentication failed: invalid_grant: Token has been expired or revoked.',
        });
        expect(JSON.stringify(error, Object.getOwnPropertyNames(error))).not.toContain('never-in-the-error');
    });

    it('forgets the failure once authentication succeeds again', async () => {
        let fail = true;
        const { driver } = driverWithAuth(async () => {
            if (fail) throw new Error('invalid_grant: Bad Request');
            return new Headers({ authorization: 'Bearer test-token' });
        });
        await expect(execute(driver)).rejects.toMatchObject({ retryable: false });

        fail = false;
        await expect(execute(driver)).resolves.toMatchObject({ result: [{ type: 'text', value: 'Bonjour' }] });
        expect(Reflect.get(driver, 'lastOpenAIAuthFailure')).toBeUndefined();
    });
});
