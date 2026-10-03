import {
    type CanonicalExecutionInputOptions,
    type ExecutionOptions,
    isCanonicalAcceptedRecovery,
    ModelType,
    PromptRole,
} from '@llumiverse/core';
import { AzureOpenAI } from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { AzureOpenAIDriver } from './azure_openai.js';

const prompt = [{ role: 'user' as const, content: 'Draw a garden' }];
const canonicalPrompt = [{ role: PromptRole.user, content: 'Draw a garden' }];
const options: ExecutionOptions = { model: 'garden::gpt-image-2.5-flare' };
const responseBody = {
    id: 'response-1',
    model: 'chat',
    status: 'completed',
    output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'done', annotations: [] }] }],
    usage: null,
};

function canonicalOptions(flow: string): CanonicalExecutionInputOptions {
    return {
        model: 'chat',
        conversation_runtime: {
            conversation_id: `conversation:azure:${flow}`,
            request_id: `request:azure:${flow}`,
            attempt_id: `attempt:azure:${flow}:first`,
            input_operation_id: `input:azure:${flow}`,
            response_operation_id: `response:azure:${flow}`,
            recorded_at: '2026-10-01T00:00:00.000Z',
        },
    };
}

describe('Azure image deployment routing', () => {
    it('preserves existing text model IDs and discovery probes', async () => {
        const service = new AzureOpenAI({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test',
            deployment: 'chat',
            apiVersion: 'custom-version',
        });
        const chat = vi.spyOn(service.chat.completions, 'create').mockResolvedValue({
            id: 'test',
            object: 'chat.completion',
            created: 0,
            model: 'gpt-5',
            choices: [],
        });
        const driver = new AzureOpenAIDriver(service);
        expect(await driver.listModels()).toEqual([expect.objectContaining({ id: 'gpt-5' })]);
        expect(chat).toHaveBeenCalledOnce();
        expect(driver.getResponsesRequestModel('chat::gpt-5')).toBe('chat');
        expect(driver.getResponsesRequestModel('gpt-5')).toBe('gpt-5');
    });

    it('preserves injected clients and uses deployment names for execution', async () => {
        const service = new AzureOpenAI({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test',
            deployment: 'garden',
            apiVersion: 'custom-version',
        });
        const generate = vi.fn().mockResolvedValue({ data: [{ b64_json: 'YQ==' }] });
        service.images.generate = generate;
        const driver = new AzureOpenAIDriver(service);
        await driver.requestImageGeneration(prompt, options);
        expect(driver.getImageService()).toBe(service);
        expect(generate.mock.calls[0][0].model).toBe('garden');
    });

    it('uses the injected Azure image service for canonical standalone generation', async () => {
        const service = new AzureOpenAI({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test',
            deployment: 'garden',
            apiVersion: 'custom-version',
        });
        const generate = vi.fn().mockResolvedValue({
            created: 1,
            data: [{ b64_json: 'iVBORw0KGgph' }],
        });
        service.images.generate = generate;
        const driver = new AzureOpenAIDriver(service);
        const result = await driver.executeCanonical([{ role: PromptRole.user, content: 'Draw a garden' }], {
            ...canonicalOptions('standalone-image'),
            model: options.model,
        });
        expect(result.accepted_output.generation).toMatchObject({
            provider: 'azure_openai',
            protocol: 'openai.images.generate',
            requested_model: options.model,
        });
        expect(generate).toHaveBeenCalledWith(expect.objectContaining({ model: 'garden', prompt: 'Draw a garden' }));
    });

    it('binds the Azure image-tool deployment into canonical request identity', async () => {
        const service = new AzureOpenAI({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test',
            deployment: 'chat',
            apiVersion: 'custom-version',
        });
        const create = vi
            .spyOn(service.responses, 'create')
            .mockResolvedValue(responseBody as unknown as Awaited<ReturnType<typeof service.responses.create>>);
        const driver = new AzureOpenAIDriver(service);
        const base = {
            ...canonicalOptions('image-tool-identity'),
            model_options: { image_generation: { model: 'garden-a::gpt-image-2.5-flare' } },
        } satisfies CanonicalExecutionInputOptions;
        const first = await driver.executeCanonical(canonicalPrompt, base);
        const retry = await driver.executeCanonical(canonicalPrompt, {
            ...base,
            conversation: JSON.parse(JSON.stringify(first.conversation)),
            conversation_runtime: {
                ...base.conversation_runtime,
                attempt_id: 'attempt:azure:image-tool-identity:retry',
            },
        });
        expect(isCanonicalAcceptedRecovery(retry)).toBe(true);
        expect(create).toHaveBeenCalledOnce();

        await expect(
            driver.executeCanonical(canonicalPrompt, {
                ...base,
                conversation: JSON.parse(JSON.stringify(first.conversation)),
                model_options: { image_generation: { model: 'garden-b::gpt-image-2.5-flare' } },
                conversation_runtime: {
                    ...base.conversation_runtime,
                    attempt_id: 'attempt:azure:image-tool-identity:changed-deployment',
                },
            }),
        ).rejects.toThrow('incompatible request identity');
        expect(create).toHaveBeenCalledOnce();
    });

    it('opens typed image-tool Responses on the v1 service with deployment headers', async () => {
        const driver = new AzureOpenAIDriver({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test',
            deployment: 'chat',
            apiVersion: 'custom-version',
        });
        const execution = {
            ...canonicalOptions('typed-image-tool'),
            model_options: { image_generation: { model: 'garden::gpt-image-2.5-flare' } },
        } satisfies CanonicalExecutionInputOptions;
        const ordinaryCreate = vi.spyOn(driver.service.responses, 'create');
        const imageService = driver.getResponsesService(execution);
        const create = vi.spyOn(imageService.responses, 'create').mockResolvedValue(
            (async function* () {
                yield {
                    type: 'response.output_text.delta' as const,
                    item_id: 'message-typed-image',
                    output_index: 0,
                    content_index: 0,
                    sequence_number: 1,
                    delta: 'done',
                    logprobs: [],
                };
                yield { type: 'response.completed' as const, sequence_number: 2, response: responseBody };
            })() as unknown as Awaited<ReturnType<typeof imageService.responses.create>>,
        );
        const stream = await driver.requestCanonicalTextCompletionEventStream(prompt, execution, undefined, {
            stream_id: 'stream:azure:typed-image-tool',
        });
        for await (const _event of stream) {
            // Consume through response acceptance so transport selection and finalization are both exercised.
        }
        expect(stream.completion?.accepted_output.generation.status).toBe('completed');
        expect(create).toHaveBeenCalledOnce();
        expect(ordinaryCreate).not.toHaveBeenCalled();
        expect(create.mock.calls[0][0]).toMatchObject({ stream: true });
        expect(create.mock.calls[0][1]).toMatchObject({
            headers: { 'x-ms-oai-image-generation-deployment': 'garden' },
        });
    });
    it.each([undefined, 'custom-version'])('routes only image tool Responses through v1 (%s)', async (apiVersion) => {
        const driver = new AzureOpenAIDriver({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test',
            deployment: 'chat',
            apiVersion,
        });
        const fetchMock = vi.fn(
            async (_url: RequestInfo | URL, _init?: RequestInit) =>
                new Response(JSON.stringify(responseBody), {
                    headers: { 'content-type': 'application/json' },
                }),
        );
        driver.service = driver.service.withOptions({ fetch: fetchMock });
        const images = driver.getImageService().withOptions({ fetch: fetchMock });
        vi.spyOn(driver, 'getImageService').mockReturnValue(images);
        await driver.requestTextCompletion(prompt, { model: 'chat' });
        await driver.requestTextCompletion(prompt, {
            model: 'chat',
            model_options: { image_generation: { model: 'garden::gpt-image-2.5-flare' } },
        });
        expect(String(fetchMock.mock.calls[0][0])).toContain(
            `/openai/responses?api-version=${apiVersion ?? '2024-10-21'}`,
        );
        expect(new Headers(fetchMock.mock.calls[0][1]?.headers).has('x-ms-oai-image-generation-deployment')).toBe(
            false,
        );
        expect(String(fetchMock.mock.calls[1][0])).toContain(
            `/openai/v1/responses?api-version=${apiVersion ?? 'preview'}`,
        );
        expect(new Headers(fetchMock.mock.calls[1][1]?.headers).get('x-ms-oai-image-generation-deployment')).toBe(
            'garden',
        );
        await driver.destroy();
    });

    it('preserves injected clients for image tool Responses', async () => {
        const service = new AzureOpenAI({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test',
            apiVersion: 'custom-version',
        });
        const create = vi
            .spyOn(service.responses, 'create')
            .mockResolvedValue(responseBody as unknown as Awaited<ReturnType<typeof service.responses.create>>);
        const driver = new AzureOpenAIDriver(service);
        const execution: ExecutionOptions = {
            model: 'chat',
            model_options: { image_generation: { model: 'garden' } },
        };
        expect(driver.getResponsesService(execution)).toBe(service);
        await driver.requestTextCompletion(prompt, execution);
        expect(create).toHaveBeenCalledOnce();
        await driver.destroy();
    });

    it('lists known image deployments without a Chat Completions probe', async () => {
        const driver = new AzureOpenAIDriver({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test',
            deployment: 'garden',
            sourceModel: 'gpt-image-2.5-flare',
            apiVersion: 'custom-version',
        });
        const chat = vi.spyOn(driver.service.chat.completions, 'create');
        expect(await driver.listModels()).toEqual([
            expect.objectContaining({
                id: options.model,
                type: ModelType.Image,
                output_modalities: ['image'],
            }),
        ]);
        expect(chat).not.toHaveBeenCalled();
        expect(driver.isImageModel('garden')).toBe(true);
        expect(driver.isImageModel('gpt-5')).toBe(false);
        expect(driver.getImageService().baseURL).toBe('https://example.openai.azure.com/openai/v1');
    });
    it('uses a separate v1 SDK transport and preserves explicit API versions', async () => {
        const driver = new AzureOpenAIDriver({
            endpoint: 'https://example.openai.azure.com',
            apiKey: 'test',
            deployment: 'garden',
            apiVersion: 'custom-version',
        });
        const fetchMock = vi.fn(
            async (_url: RequestInfo | URL, _init?: RequestInit) =>
                new Response(JSON.stringify({ data: [{ b64_json: 'YQ==' }], output_format: 'png' }), {
                    status: 200,
                    headers: { 'content-type': 'application/json' },
                }),
        );
        const service = driver.getImageService().withOptions({ fetch: fetchMock });
        vi.spyOn(driver, 'getImageService').mockReturnValue(service);
        await driver.requestImageGeneration(prompt, options);
        expect(String(fetchMock.mock.calls[0][0])).toContain(
            '/openai/v1/images/generations?api-version=custom-version',
        );
        expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body)).model).toBe('garden');
        const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
        expect(headers.get('api-key')).toBe('test');
        expect(headers.has('authorization')).toBe(false);
    });
});
