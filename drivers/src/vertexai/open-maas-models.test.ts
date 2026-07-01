import { type Completion, type ExecutionOptions, PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import type { VertexAIDriver } from './index.js';
import { getModelDefinition } from './models.js';
import { getListedVertexOpenMaaSModels, VERTEX_OPEN_MAAS_MODELS } from './open-maas-models.js';

function createDriverStub() {
    const post = vi.fn(async () => ({
        id: 'chatcmpl-1',
        object: 'chat.completion',
        created: 1,
        model: 'test-model',
        choices: [
            {
                index: 0,
                message: { role: 'assistant', content: 'ok' },
                finish_reason: 'stop',
            },
        ],
        usage: {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        },
    }));
    const getFetchClient = vi.fn(() => ({ post }));
    const getFetchClientForRegion = vi.fn(() => ({ post }));

    return {
        driver: { getFetchClient, getFetchClientForRegion } as unknown as VertexAIDriver,
        getFetchClient,
        getFetchClientForRegion,
        post,
    };
}

async function requestForModel(model: string): Promise<{
    completion: Completion;
    post: ReturnType<typeof vi.fn>;
    getFetchClient: ReturnType<typeof vi.fn>;
    getFetchClientForRegion: ReturnType<typeof vi.fn>;
}> {
    const modelDefinition = getModelDefinition(model);
    const prompt = await modelDefinition.createPrompt(
        {} as VertexAIDriver,
        [{ role: PromptRole.user, content: 'hello' }],
        { model },
    );
    const stub = createDriverStub();
    const options: ExecutionOptions = {
        model,
        model_options: { _option_id: 'text-fallback' },
    };

    const completion = await modelDefinition.requestTextCompletion(stub.driver, prompt, options);
    return { completion, ...stub };
}

describe('Vertex open MaaS catalog', () => {
    it('catalogs the manually verified MaaS model and region matrix', () => {
        const catalog = Object.fromEntries(
            VERTEX_OPEN_MAAS_MODELS.map((entry) => [`${entry.publisher}/${entry.model}`, entry.regions]),
        );

        expect(catalog).toEqual({
            'deepseek-ai/deepseek-ocr-maas': ['us-central1'],
            'deepseek-ai/deepseek-r1-0528-maas': ['us-central1'],
            'deepseek-ai/deepseek-v3.1-maas': ['us-central1'],
            'deepseek-ai/deepseek-v3.2-maas': ['global'],
            'google/gemma-4-26b-a4b-it-maas': ['global'],
            'meta/llama-3.3-70b-instruct-maas': ['us-central1'],
            'meta/llama-4-maverick-17b-128e-instruct-maas': ['us-east5'],
            'meta/llama-4-scout-17b-16e-instruct-maas': ['us-east5'],
            'minimaxai/minimax-m2-maas': ['global'],
            'moonshotai/kimi-k2-thinking-maas': ['global'],
            'openai/gpt-oss-120b-maas': ['global', 'us-central1'],
            'openai/gpt-oss-20b-maas': ['us-central1'],
            'qwen/qwen3-235b-a22b-instruct-2507-maas': ['global', 'us-south1'],
            'qwen/qwen3-coder-480b-a35b-instruct-maas': ['global', 'us-south1'],
            'qwen/qwen3-next-80b-a3b-instruct-maas': ['global'],
            'qwen/qwen3-next-80b-a3b-thinking-maas': ['global'],
            'zai-org/glm-4.7-maas': ['global'],
            'zai-org/glm-5-maas': ['global'],
        });
    });

    it('lists MaaS models as static Vertex model ids', () => {
        const modelIds = getListedVertexOpenMaaSModels('us-east5').map((model) => model.id);

        expect(modelIds).toContain('locations/us-east5/publishers/meta/models/llama-4-maverick-17b-128e-instruct-maas');
        expect(modelIds).toContain('locations/global/publishers/qwen/models/qwen3-coder-480b-a35b-instruct-maas');
        expect(modelIds).toContain('locations/global/publishers/zai-org/models/glm-5-maas');
        expect(modelIds).toContain('locations/global/publishers/minimaxai/models/minimax-m2-maas');
        expect(modelIds).toContain('locations/global/publishers/openai/models/gpt-oss-120b-maas');
        expect(modelIds).toContain('locations/global/publishers/google/models/gemma-4-26b-a4b-it-maas');
        expect(modelIds).toContain('locations/us-central1/publishers/deepseek-ai/models/deepseek-ocr-maas');
        expect(modelIds).toContain('locations/us-central1/publishers/deepseek-ai/models/deepseek-v3.1-maas');
        expect(modelIds).not.toContain(
            'locations/global/publishers/meta/models/llama-4-maverick-17b-128e-instruct-maas',
        );
        expect(modelIds).not.toContain('locations/global/publishers/deepseek-ai/models/deepseek-ocr-maas');
        expect(modelIds).not.toContain('locations/us-east5/publishers/google/models/gemma-4-26b-a4b-it-maas');
    });

    it('lists regional MaaS models independently of the configured region', () => {
        const usCentralModels = getListedVertexOpenMaaSModels('us-central1');
        const usCentralIds = usCentralModels.map((model) => model.id);

        expect(usCentralIds).toContain(
            'locations/us-east5/publishers/meta/models/llama-4-maverick-17b-128e-instruct-maas',
        );
        expect(usCentralIds).toContain('locations/us-central1/publishers/deepseek-ai/models/deepseek-ocr-maas');
        expect(usCentralIds).toContain('locations/us-central1/publishers/openai/models/gpt-oss-120b-maas');
        expect(usCentralIds).toContain('locations/us-central1/publishers/openai/models/gpt-oss-20b-maas');
        expect(usCentralIds).toContain('locations/global/publishers/openai/models/gpt-oss-120b-maas');
        expect(usCentralIds).not.toContain('locations/global/publishers/openai/models/gpt-oss-20b-maas');

        const deepSeekOcr = usCentralModels.find(
            (model) => model.id === 'locations/us-central1/publishers/deepseek-ai/models/deepseek-ocr-maas',
        );
        expect(deepSeekOcr?.input_modalities).toEqual(['text', 'image']);
        expect(deepSeekOcr?.output_modalities).toEqual(['text']);
        expect(deepSeekOcr?.tool_support).toBe(false);
    });

    it('lists MaaS regional alternates for the configured region', () => {
        const modelIds = getListedVertexOpenMaaSModels('us-south1').map((model) => model.id);

        expect(modelIds).toContain('locations/global/publishers/qwen/models/qwen3-coder-480b-a35b-instruct-maas');
        expect(modelIds).toContain('locations/us-south1/publishers/qwen/models/qwen3-coder-480b-a35b-instruct-maas');
        expect(modelIds).toContain('locations/global/publishers/qwen/models/qwen3-235b-a22b-instruct-2507-maas');
        expect(modelIds).toContain('locations/us-south1/publishers/qwen/models/qwen3-235b-a22b-instruct-2507-maas');
    });

    it('routes Llama MaaS through the OpenAI-compatible endpoint with Llama-specific extra body', async () => {
        const { post, getFetchClientForRegion, completion } = await requestForModel(
            'locations/us-east5/publishers/meta/models/llama-4-maverick-17b-128e-instruct-maas',
        );

        expect(completion.result).toEqual([{ type: 'text', value: 'ok' }]);
        expect(getFetchClientForRegion).toHaveBeenCalledWith('us-east5', 'v1beta1');
        expect(post).toHaveBeenCalledWith('endpoints/openapi/chat/completions', {
            payload: expect.objectContaining({
                model: 'meta/llama-4-maverick-17b-128e-instruct-maas',
                extra_body: {
                    google: {
                        model_safety_settings: {
                            enabled: false,
                            llama_guard_settings: {},
                        },
                    },
                },
            }),
        });
    });

    it('routes dynamic no-location MaaS ids to the catalog region', async () => {
        const { post, getFetchClientForRegion } = await requestForModel(
            'publishers/meta/models/llama-4-maverick-17b-128e-instruct-maas',
        );

        expect(getFetchClientForRegion).toHaveBeenCalledWith('us-east5', 'v1beta1');
        expect(post).toHaveBeenCalledWith('endpoints/openapi/chat/completions', {
            payload: expect.objectContaining({
                model: 'meta/llama-4-maverick-17b-128e-instruct-maas',
            }),
        });
    });

    it('keeps old short Llama ids as routing aliases', async () => {
        const { post, getFetchClientForRegion } = await requestForModel(
            'locations/us-east5/publishers/meta/models/llama-4-maverick-17b-128e',
        );

        expect(getFetchClientForRegion).toHaveBeenCalledWith('us-east5', 'v1beta1');
        expect(post).toHaveBeenCalledWith('endpoints/openapi/chat/completions', {
            payload: expect.objectContaining({
                model: 'meta/llama-4-maverick-17b-128e-instruct-maas',
            }),
        });
    });

    it('routes new open MaaS families through the OpenAI-compatible endpoint', async () => {
        const cases = [
            ['locations/global/publishers/deepseek-ai/models/deepseek-v3.2-maas', 'deepseek-ai/deepseek-v3.2-maas'],
            ['locations/us-central1/publishers/deepseek-ai/models/deepseek-ocr-maas', 'deepseek-ai/deepseek-ocr-maas'],
            [
                'locations/global/publishers/qwen/models/qwen3-next-80b-a3b-instruct-maas',
                'qwen/qwen3-next-80b-a3b-instruct-maas',
            ],
            ['locations/global/publishers/zai-org/models/glm-4.7-maas', 'zai-org/glm-4.7-maas'],
            ['locations/global/publishers/moonshotai/models/kimi-k2-thinking-maas', 'moonshotai/kimi-k2-thinking-maas'],
            ['locations/global/publishers/minimaxai/models/minimax-m2-maas', 'minimaxai/minimax-m2-maas'],
            ['locations/global/publishers/openai/models/gpt-oss-120b-maas', 'openai/gpt-oss-120b-maas'],
            ['locations/us-central1/publishers/openai/models/gpt-oss-20b-maas', 'openai/gpt-oss-20b-maas'],
            ['locations/global/publishers/google/models/gemma-4-26b-a4b-it-maas', 'google/gemma-4-26b-a4b-it-maas'],
        ] as const;

        for (const [model, requestModel] of cases) {
            const { post, getFetchClientForRegion } = await requestForModel(model);

            if (model.includes('deepseek-ocr-maas')) {
                expect(getFetchClientForRegion).toHaveBeenCalledWith('us-central1', undefined, 'global');
            } else {
                expect(getFetchClientForRegion).toHaveBeenCalledWith(model.split('/')[1], undefined);
            }
            expect(post).toHaveBeenCalledWith('endpoints/openapi/chat/completions', {
                payload: expect.objectContaining({ model: requestModel }),
            });
        }
    });

    it('does not change default MaaS thinking behavior unless the caller opts in', async () => {
        const glm = await requestForModel('locations/global/publishers/zai-org/models/glm-5-maas');
        const payload = glm.post.mock.calls[0][1].payload as Record<string, unknown>;

        expect(payload.model).toBe('zai-org/glm-5-maas');
        expect(payload).not.toHaveProperty('chat_template_kwargs');
    });
});
