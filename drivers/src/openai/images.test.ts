import { Base64DataSource, type ExecutionOptions, type OpenAiGptImageOptions, PromptRole } from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { imageDataUrl, imageRequest } from './images.js';
import { OpenAIDriver } from './openai.js';
import { formatOpenAILikeMultimodalPrompt } from './openai_format.js';

const options: ExecutionOptions = {
    model: 'gpt-image-2.5-sunburst',
    model_options: {
        _option_id: 'openai-gpt-image',
        size: '2048x1024',
        image_quality: 'max',
        output_format: 'webp',
        n: 2,
        output_compression: 0,
        partial_images: 0,
        input_fidelity: 'high',
    },
};
const input: OpenAI.Responses.ResponseInputItem[] = [{ role: 'user', content: 'A garden' }];

describe('image requests', () => {
    it('serializes numeric dimensions and preserves size compatibility', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        for (const [config, size] of [
            [{ width: 2048, height: 1536, size: '1024x1024' }, '2048x1536'],
            [{ width: 2048 }, '2048x1024'],
            [{ size: 'auto' }, 'auto'],
            [{}, '1024x1024'],
        ] as const) {
            const request = await imageRequest(driver.service, input, options.model, config, options.model);
            expect(request.generate?.size).toBe(size);
        }
    });

    it('serializes generation options and zero values', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const generate = vi.fn().mockResolvedValue({
            data: [{ b64_json: 'YWJj' }],
            output_format: 'jpeg',
            usage: { input_tokens: 2, output_tokens: 3, total_tokens: 5 },
        });
        driver.service.images.generate = generate;
        const result = await driver.requestTextCompletion(input, options);
        expect(generate.mock.calls[0][0]).toMatchObject({
            model: options.model,
            quality: 'max',
            n: 2,
            size: '2048x1024',
            output_compression: 0,
            partial_images: 0,
            output_format: 'webp',
        });
        expect(result.result).toEqual([{ type: 'image', value: 'data:image/jpeg;base64,YWJj' }]);
        expect(result.token_usage).toEqual({ prompt: 2, result: 3, total: 5 });
    });

    it('retains ordered references and applies masks separately', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const image = (data: string) => new Base64DataSource('image.png', 'image/png', data);
        const prompt = await formatOpenAILikeMultimodalPrompt(
            [
                { role: PromptRole.user, content: 'Edit', files: [image('YQ=='), image('Yg==')] },
                { role: PromptRole.mask, content: '', files: [image('Yw==')] },
            ],
            options,
        );
        const request = await imageRequest(
            driver.service,
            prompt,
            options.model,
            options.model_options as OpenAiGptImageOptions,
            options.model,
        );
        expect(request.generate).toBeUndefined();
        if (!request.edit) throw new Error('Expected an edit request');
        const refs = request.edit.image as File[];
        expect(await Promise.all(refs.map((file) => file.text()))).toEqual(['a', 'b']);
        expect(await (request.edit.mask as File).text()).toBe('c');
        expect(request.edit?.input_fidelity).toBe('high');
        const gpt2 = await imageRequest(
            driver.service,
            prompt,
            'gpt-image-2',
            options.model_options as OpenAiGptImageOptions,
            'gpt-image-2',
        );
        expect(gpt2.edit?.input_fidelity).toBeUndefined();
    });

    it('preserves input fidelity for earlier models and rejects ambiguous masks', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const image = () => new Base64DataSource('image.png', 'image/png', 'YQ==');
        const prompt = await formatOpenAILikeMultimodalPrompt(
            [
                { role: PromptRole.user, content: 'Edit', files: [image()] },
                { role: PromptRole.mask, content: '', files: [image(), image()] },
            ],
            options,
        );
        await expect(imageRequest(driver.service, prompt, options.model, undefined, options.model)).rejects.toThrow(
            'Only one image mask',
        );
    });

    it('streams completed images only and fails if no completion arrives', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        driver.service.images.generate = vi.fn().mockResolvedValue(
            (async function* () {
                yield { type: 'image_generation.partial_image', b64_json: 'preview' };
                yield { type: 'image_generation.completed', b64_json: 'a', output_format: 'png' };
                yield { type: 'image_generation.completed', b64_json: 'b', output_format: 'webp' };
            })(),
        );
        const chunks = [];
        for await (const chunk of await driver.requestImageStream(input, options)) chunks.push(chunk);
        expect(chunks.flatMap((chunk) => chunk.result)).toEqual([
            { type: 'image', value: 'data:image/png;base64,a' },
            { type: 'image', value: 'data:image/webp;base64,b' },
        ]);
        driver.service.images.generate = vi.fn().mockResolvedValue(
            (async function* () {
                yield { type: 'image_generation.partial_image', b64_json: 'preview' };
            })(),
        );
        await expect(
            (async () => {
                for await (const _chunk of await driver.requestImageStream(input, options)) {
                    /* consume */
                }
            })(),
        ).rejects.toThrow('without a completed image');
    });

    it('does not disguise provider errors as validation failures', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        driver.service.images.generate = vi.fn().mockRejectedValue(new Error('provider unavailable'));
        await expect(driver.requestImageGeneration(input, options)).rejects.toThrow('provider unavailable');
    });

    it('derives MIME in provider, request, default order', () => {
        expect(imageDataUrl('a', 'webp', 'jpeg')).toContain('image/webp');
        expect(imageDataUrl('a', undefined, 'jpeg')).toContain('image/jpeg');
        expect(imageDataUrl('a')).toContain('image/png');
    });
});

describe('image stream lifecycle', () => {
    it('uses the managed stream cancellation and timeout path', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const closed = vi.fn();
        let requestSignal: AbortSignal | undefined;
        let requestTimeout: number | undefined;
        driver.service.images.generate = vi.fn().mockImplementation(async (_request, requestOptions) => {
            requestSignal = requestOptions.signal;
            requestTimeout = requestOptions.timeout;
            return (async function* () {
                try {
                    yield { type: 'image_generation.completed', b64_json: 'YQ==', output_format: 'png' };
                    yield { type: 'image_generation.completed', b64_json: 'Yg==', output_format: 'png' };
                } finally {
                    closed();
                }
            })();
        });
        const stream = await driver.stream([{ role: PromptRole.user, content: 'Draw' }], {
            ...options,
            httpTimeout: { headersTimeout: 1000, bodyTimeout: 2000 },
        });
        const iterator = stream[Symbol.asyncIterator]();
        await iterator.next();
        await stream.cancel();
        expect(requestSignal?.aborted).toBe(true);
        expect(requestTimeout).toBe(2000);
        expect(closed).toHaveBeenCalledOnce();
    });
    it('rejects pre-cancelled streams without calling the image API', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const generate = vi.spyOn(driver.service.images, 'generate');
        const controller = new AbortController();
        controller.abort();
        await expect(
            driver.stream([{ role: PromptRole.user, content: 'Draw' }], options, controller.signal),
        ).rejects.toThrow();
        expect(generate).not.toHaveBeenCalled();
    });
});

it('lets the SDK serialize ordered multipart files and one mask', async () => {
    const driver = new OpenAIDriver({ apiKey: 'test' });
    const fetchMock = vi.fn(
        async (_url: RequestInfo | URL, _init?: RequestInit) =>
            new Response(JSON.stringify({ data: [{ b64_json: 'YQ==' }], output_format: 'png' }), {
                status: 200,
                headers: { 'content-type': 'application/json' },
            }),
    );
    driver.service = driver.service.withOptions({ fetch: fetchMock });
    const mask = new Base64DataSource('mask.png', 'image/png', 'Yw==');
    const readMask = vi.spyOn(mask, 'getStream');
    const prompt = await driver.createPrompt(
        [
            {
                role: PromptRole.user,
                content: 'Edit',
                files: [new Base64DataSource('a.png', 'image/png', 'YQ==')],
            },
            {
                role: PromptRole.system,
                content: 'Reference',
                files: [new Base64DataSource('b.png', 'image/png', 'Yg==')],
            },
            { role: PromptRole.mask, content: '', files: [mask] },
        ],
        options,
    );
    await driver.requestImageGeneration(prompt, options);
    const editCall = fetchMock.mock.calls.find(([url]) => String(url).includes('/images/edits'));
    expect(editCall).toBeDefined();
    const form = editCall?.[1]?.body as FormData;
    expect(await Promise.all((form.getAll('image[]') as File[]).map((file) => file.text()))).toEqual(['a', 'b']);
    expect(await (form.get('mask') as File).text()).toBe('c');
    expect(form.get('model')).toBe(options.model);
    expect(readMask).toHaveBeenCalledOnce();
});
