import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { Base64DataSource, type ExecutionOptions, type OpenAiGptImageOptions, PromptRole } from '@llumiverse/core';
import type OpenAI from 'openai';
import { describe, expect, it, vi } from 'vitest';
import { canStreamImage, imageDataUrl, imageRequest } from './images.js';
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
            [{ size: 'auto' }, 'auto'],
            [{}, undefined],
        ] as const) {
            const request = await imageRequest(driver.service, input, options.model, config, options.model);
            expect(request.generate?.size).toBe(size);
        }
        const legacy = await imageRequest(driver.service, input, 'dall-e-3', {}, 'dall-e-3');
        expect(legacy.generate?.size).toBe('1024x1024');
    });

    it.each([{ width: 2048 }, { height: 1536 }, { width: 2048, size: '1024x1024' }])(
        'rejects a single dimension %o',
        async (config) => {
            const driver = new OpenAIDriver({ apiKey: 'test' });
            await expect(imageRequest(driver.service, input, options.model, config, options.model)).rejects.toThrow(
                'Set both width and height',
            );
        },
    );

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
        // Without output_tokens_details the image share of the output tokens is unknown.
        expect(result.token_usage).toEqual({ prompt: 2, prompt_new: 2, result: 3, total: 5 });
    });

    it.each([
        ['url', false],
        ['url', true],
        ['file', false],
        ['file', true],
    ] as const)('bounds %s downloads before image requests (stream=%s)', async (kind, streaming) => {
        const server = createServer((_request, response) => {
            setTimeout(() => {
                response.writeHead(200, { 'content-type': 'image/png' });
                response.end('image');
            }, 120);
        });
        await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
        const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
        const driver = new OpenAIDriver({ apiKey: 'test' });
        driver.service = driver.service.withOptions({ baseURL: url });
        const edit = vi.spyOn(driver.service.images, 'edit');
        const prompt: OpenAI.Responses.ResponseInputItem[] = [
            {
                role: 'user',
                content: [
                    { type: 'input_text', text: 'Edit' },
                    {
                        type: 'input_image',
                        detail: 'auto',
                        ...(kind === 'url' ? { image_url: url } : { file_id: 'file-ref' }),
                    },
                ],
            },
        ];
        const execution = { ...options, httpTimeout: { headersTimeout: 20, bodyTimeout: 20 } };
        try {
            await expect(
                streaming
                    ? driver.requestImageStream(prompt, execution)
                    : driver.requestImageGeneration(prompt, execution),
            ).rejects.toThrow();
            expect(edit).not.toHaveBeenCalled();
        } finally {
            await driver.destroy();
            server.closeAllConnections();
            await new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
        }
    });

    it('passes request options to file downloads and cancellation to reference loading', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const content = vi.spyOn(driver.service.files, 'content').mockResolvedValue(new Response('reference'));
        const fetcher = vi.fn(async (_url: RequestInfo | URL, _init?: RequestInit) => new Response('reference'));
        const controller = new AbortController();
        const prompt: OpenAI.Responses.ResponseInputItem[] = [
            {
                role: 'user',
                content: [
                    { type: 'input_image', file_id: 'file-ref', detail: 'auto' },
                    { type: 'input_image', image_url: 'https://example.com/image.png', detail: 'auto' },
                ],
            },
        ];
        await imageRequest(
            driver.service,
            prompt,
            options.model,
            undefined,
            options.model,
            { signal: controller.signal, timeout: 2000 },
            fetcher,
        );
        expect(content.mock.calls[0][1]?.timeout).toBe(2000);
        const fileSignal = content.mock.calls[0][1]?.signal;
        const urlSignal = fetcher.mock.calls[0][1]?.signal;
        controller.abort();
        expect(fileSignal?.aborted).toBe(true);
        expect(urlSignal?.aborted).toBe(true);
        await driver.destroy();
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

    it('counts the output tokens of a streamed completed image as image tokens', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        driver.service.images.generate = vi.fn().mockResolvedValue(
            (async function* () {
                yield {
                    type: 'image_generation.completed',
                    b64_json: 'a',
                    output_format: 'png',
                    usage: {
                        input_tokens: 12,
                        input_tokens_details: { image_tokens: 0, text_tokens: 12 },
                        output_tokens: 1056,
                        total_tokens: 1068,
                    },
                };
            })(),
        );
        const chunks = [];
        for await (const chunk of await driver.requestImageStream(input, options)) chunks.push(chunk);
        expect(chunks[0]?.token_usage).toEqual({
            prompt: 12,
            prompt_new: 12,
            result: 1056,
            result_image: 1056,
            total: 1068,
        });
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
        expect(imageDataUrl('a', 'svg')).toBe('data:image/svg+xml;base64,a');
        expect(imageDataUrl('a', undefined, 'gif')).toBe('data:image/gif;base64,a');
    });

    it('forwards GPT Image 2.6 complexity and GIF controls to generations and edits', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const config: OpenAiGptImageOptions = {
            complexity: 'low',
            output_format: 'gif',
            frame_count: 4,
            frame_duration_ms: 120,
        };
        const generate = await imageRequest(
            driver.service,
            input,
            'gpt-image-2.6-flare',
            config,
            'gpt-image-2.6-flare',
        );
        expect(generate.generate).toMatchObject(config);
        const reference = [
            {
                role: 'user',
                content: [{ type: 'input_image', image_url: 'data:image/png;base64,YQ==', detail: 'auto' }],
            },
        ] as OpenAI.Responses.ResponseInputItem[];
        const edit = await imageRequest(
            driver.service,
            reference,
            'gpt-image-2.6-sunburst',
            { ...config, output_format: 'svg' },
            'gpt-image-2.6-sunburst',
        );
        expect(edit.edit).toMatchObject({ ...config, output_format: 'svg' });
    });

    it('forwards SVG author settings to generations', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const svg = { model: 'gpt-6-astra', reasoning: { effort: 'low' as const } };
        const request = await imageRequest(
            driver.service,
            input,
            'gpt-image-2.6-flare',
            { output_format: 'svg', svg },
            'gpt-image-2.6-flare',
        );
        expect(request.generate).toMatchObject({ output_format: 'svg', svg });
    });

    it('leaves omitted 2.6 controls to the provider', async () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const request = await imageRequest(
            driver.service,
            input,
            'gpt-image-2.6-sunburst',
            {},
            'gpt-image-2.6-sunburst',
        );
        for (const name of ['complexity', 'frame_count', 'frame_duration_ms', 'svg'] as const) {
            expect(request.generate?.[name]).toBeUndefined();
        }
    });

    it('streams GPT Image except GIF and SVG outputs, and never legacy endpoints', () => {
        for (const model of ['gpt-image-2', 'gpt-image-2.5-flare', 'deployment::gpt-image-2.6-sunburst']) {
            expect(canStreamImage(model, undefined)).toBe(true);
            expect(canStreamImage(model, { _option_id: 'openai-gpt-image', output_format: 'webp' })).toBe(true);
            expect(canStreamImage(model, { _option_id: 'openai-gpt-image', output_format: 'gif' })).toBe(false);
            expect(canStreamImage(model, { _option_id: 'openai-gpt-image', output_format: 'svg' })).toBe(false);
        }
        expect(canStreamImage('dall-e-3', undefined)).toBe(false);
    });

    it.each(['gif', 'svg'] as const)('returns %s output without streaming', async (output_format) => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const generate = vi.fn().mockResolvedValue({ data: [{ b64_json: 'YQ==' }], output_format });
        driver.service.images.generate = generate;
        const stream = await driver.stream([{ role: PromptRole.user, content: 'Draw' }], {
            model: 'gpt-image-2.6-sunburst',
            model_options: { _option_id: 'openai-gpt-image', output_format },
        });
        for await (const _chunk of stream) {
            /* consume */
        }
        expect(generate.mock.calls[0][0].stream).toBeUndefined();
        expect(stream.completion?.result?.[0]?.type).toBe('image');
    });

    it.each(['gif', 'svg'] as const)('returns %s output without streaming from direct stream calls', async (format) => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        const generate = vi.fn().mockResolvedValue({ data: [{ b64_json: 'YQ==' }], output_format: format });
        driver.service.images.generate = generate;
        const options = {
            model: 'gpt-image-2.6-sunburst',
            model_options: { _option_id: 'openai-gpt-image' as const, output_format: format },
        };
        const prompt = [{ role: 'user' as const, content: 'Draw' }];
        for (const stream of [
            await driver.requestImageStream(prompt, options),
            await driver.requestTextCompletionStream(prompt, options),
        ]) {
            const chunks = [];
            for await (const chunk of stream) chunks.push(chunk);
            expect(chunks).toEqual([
                expect.objectContaining({ result: [{ type: 'image', value: expect.stringContaining(format) }] }),
            ]);
        }
        expect(generate).toHaveBeenCalledTimes(2);
        for (const [params] of generate.mock.calls) expect(params.stream).toBeUndefined();
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
