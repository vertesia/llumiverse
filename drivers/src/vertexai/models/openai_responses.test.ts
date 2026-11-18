import { PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { VertexAIDriver } from '../index.js';
import { getModelDefinition } from '../models.js';
import { OpenAIChatCompletionsModelDefinition } from './openai_chat_completions.js';
import { VertexAIResponsesModelDefinition } from './openai_responses.js';

class TestVertexAIDriver extends VertexAIDriver {
    constructor(private readonly fetchMock: typeof fetch) {
        super({ project: 'test-project', region: 'us-central1' });
        this.googleAuth = {
            getAccessToken: async () => 'test-token',
        } as VertexAIDriver['googleAuth'];
    }

    protected override getDriverFetch(): typeof fetch {
        return this.fetchMock;
    }
}

function responseBody(model: string) {
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
                content: [{ type: 'output_text', text: 'Bonjour', annotations: [] }],
            },
        ],
        usage: { input_tokens: 10, output_tokens: 20, total_tokens: 30 },
    };
}

async function captureRequest(model: string, modelOptions: Record<string, unknown>) {
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
        model_options: { _option_id: 'openai-text', ...modelOptions },
    });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    const headers = new Headers(init.headers);
    return { completion, url, body: JSON.parse(String(init.body)), authorization: headers.get('authorization') };
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
    ])('routes %s to the Responses API', (model) => {
        expect(getModelDefinition(model)).toBeInstanceOf(VertexAIResponsesModelDefinition);
    });

    it('keeps Llama on Chat Completions', () => {
        expect(getModelDefinition('publishers/meta/models/llama-4-scout-17b-16e-instruct-maas')).toBeInstanceOf(
            OpenAIChatCompletionsModelDefinition,
        );
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
});
