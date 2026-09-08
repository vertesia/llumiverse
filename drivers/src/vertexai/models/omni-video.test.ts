import {
    type CompletionResult,
    type DataSource,
    type ExecutionOptions,
    LlumiverseError,
    PromptRole,
} from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import type { VertexAIDriver } from '../index.js';
import { getModelDefinition } from '../models.js';
import {
    GEMINI_OMNI_1_1_VIDEO_MODEL,
    GEMINI_OMNI_VIDEO_MODEL,
    type GeminiOmniVideoModel,
    GeminiOmniVideoModelDefinition,
} from './omni-video.js';

const OUTPUT_PREFIX = 'gs://project-bucket/runs/run-1/media/';

function media(uri = 'gs://project-bucket/input/frame.png', mimeType = 'image/png'): DataSource {
    return {
        name: 'frame.png',
        mime_type: mimeType,
        getURI: vi.fn().mockResolvedValue(uri),
        getURL: vi.fn(),
        getStream: vi.fn(),
    };
}

function image(uri = 'gs://project-bucket/input/frame.png', mimeType = 'image/png'): DataSource {
    return media(uri, mimeType);
}

function video(uri = 'gs://project-bucket/input/clip.mp4', mimeType = 'video/mp4'): DataSource {
    return media(uri, mimeType);
}

function options(
    overrides: Partial<ExecutionOptions> = {},
    model: GeminiOmniVideoModel = GEMINI_OMNI_VIDEO_MODEL,
): ExecutionOptions {
    return {
        model,
        model_options: { _option_id: 'vertexai-gemini-omni-video' },
        output_storage_uri: OUTPUT_PREFIX,
        ...overrides,
    };
}

function completedResponse(...results: CompletionResult[]) {
    return {
        id: 'interaction-1',
        status: 'completed',
        usage: { total_tokens: 9, total_input_tokens: 3, total_output_tokens: 6 },
        steps: [
            { type: 'thought', summary: [{ type: 'text', text: 'hidden' }] },
            {
                type: 'model_output',
                content: results.map((result) =>
                    result.type === 'video'
                        ? { type: 'video', uri: result.value, mime_type: 'video/mp4' }
                        : { type: 'text', text: String(result.value) },
                ),
            },
        ],
    };
}

function driverWithResponse(response: unknown) {
    const post = vi.fn().mockResolvedValue(response);
    const getFetchClientForRegion = vi.fn(() => ({ post }));
    const getRequestTimeoutMs = vi.fn(() => 900_000);
    return {
        driver: { getFetchClientForRegion, getRequestTimeoutMs } as unknown as VertexAIDriver,
        getFetchClientForRegion,
        getRequestTimeoutMs,
        post,
    };
}

describe('GeminiOmniVideoModelDefinition', () => {
    it.each([GEMINI_OMNI_VIDEO_MODEL, GEMINI_OMNI_1_1_VIDEO_MODEL])(
        'routes %s to the dedicated non-streaming definition',
        (model) => {
            const definition = getModelDefinition(`publishers/google/models/${model}`);

            expect(definition).toBeInstanceOf(GeminiOmniVideoModelDefinition);
            expect(definition.model).toMatchObject({ id: model, type: 'video', can_stream: false });
        },
    );

    it('constructs a text-to-video request with defaults and the global beta endpoint', async () => {
        const definition = new GeminiOmniVideoModelDefinition();
        const prompt = await definition.createPrompt(
            {} as VertexAIDriver,
            [
                { role: PromptRole.system, content: 'Cinematic' },
                { role: PromptRole.user, content: 'A fox in snow' },
            ],
            options(),
        );
        const stub = driverWithResponse(completedResponse({ type: 'video', value: `${OUTPUT_PREFIX}video.mp4` }));

        const completion = await definition.requestTextCompletion(stub.driver, prompt, options());

        expect(stub.getFetchClientForRegion).toHaveBeenCalledWith('global', 'v1beta1');
        expect(stub.post).toHaveBeenCalledWith('interactions', {
            payload: {
                model: GEMINI_OMNI_VIDEO_MODEL,
                input: [{ type: 'text', text: 'Cinematic\nA fox in snow' }],
                response_format: [{ type: 'video', delivery: 'uri', gcs_uri: OUTPUT_PREFIX, duration: '5s' }],
                generation_config: { video_config: { task: 'text_to_video' } },
            },
            signal: undefined,
            timeoutMs: 900_000,
        });
        expect(completion).toEqual({
            result: [{ type: 'video', value: `${OUTPUT_PREFIX}video.mp4` }],
            token_usage: { total: 9, prompt: 3, result: 6 },
            finish_reason: 'stop',
        });
    });

    it('maps a GCS first frame without reading its bytes', async () => {
        const source = image();
        const definition = new GeminiOmniVideoModelDefinition();
        const configured = options({
            model_options: {
                _option_id: 'vertexai-gemini-omni-video',
                task: 'image_to_video',
                aspect_ratio: '9:16',
                duration_seconds: 10,
            },
        });
        const prompt = await definition.createPrompt(
            {} as VertexAIDriver,
            [{ role: PromptRole.user, content: 'Animate this', files: [source] }],
            configured,
        );
        const stub = driverWithResponse(
            completedResponse(
                { type: 'video', value: `${OUTPUT_PREFIX}one.mp4` },
                { type: 'video', value: `${OUTPUT_PREFIX}two.mp4` },
            ),
        );

        const completion = await definition.requestTextCompletion(stub.driver, prompt, configured);

        expect(source.getURI).toHaveBeenCalledOnce();
        expect(source.getStream).not.toHaveBeenCalled();
        expect(stub.post.mock.calls[0]?.[1]).toMatchObject({
            payload: {
                input: [
                    { type: 'text', text: 'Animate this' },
                    { type: 'image', uri: 'gs://project-bucket/input/frame.png', mime_type: 'image/png' },
                ],
                response_format: [expect.objectContaining({ aspect_ratio: '9:16', duration: '10s' })],
                generation_config: { video_config: { task: 'image_to_video' } },
            },
        });
        expect(completion.result).toHaveLength(2);
    });

    it('accepts references and requires an explicit media task', async () => {
        const definition = new GeminiOmniVideoModelDefinition();
        const references = [image('gs://inputs/1.png'), image('gs://inputs/2.png'), image('gs://inputs/3.png')];
        const configured = options({
            model_options: { _option_id: 'vertexai-gemini-omni-video', task: 'reference_to_video' },
        });

        const prompt = await definition.createPrompt(
            {} as VertexAIDriver,
            [{ role: PromptRole.user, content: 'Use these', files: references }],
            configured,
        );

        expect(prompt.media).toHaveLength(3);
        await expect(
            definition.createPrompt(
                {} as VertexAIDriver,
                [{ role: PromptRole.user, content: 'Use this', files: [image()] }],
                options(),
            ),
        ).rejects.toThrow('requires an explicit');
    });

    it('supports first and last frames and 1.1 output resolutions', async () => {
        const definition = new GeminiOmniVideoModelDefinition(GEMINI_OMNI_1_1_VIDEO_MODEL);
        const configured = options(
            {
                model_options: {
                    _option_id: 'vertexai-gemini-omni-video',
                    task: 'image_to_video',
                    resolution: '4k',
                },
            },
            GEMINI_OMNI_1_1_VIDEO_MODEL,
        );
        const prompt = await definition.createPrompt(
            {} as VertexAIDriver,
            [
                {
                    role: PromptRole.user,
                    content: 'Transition between these frames',
                    files: [image('gs://inputs/first.png'), image('gs://inputs/last.png')],
                },
            ],
            configured,
        );
        const stub = driverWithResponse(completedResponse({ type: 'video', value: `${OUTPUT_PREFIX}result.mp4` }));

        await definition.requestTextCompletion(stub.driver, prompt, configured);

        expect(stub.post.mock.calls[0]?.[1]).toMatchObject({
            payload: {
                model: GEMINI_OMNI_1_1_VIDEO_MODEL,
                input: [
                    { type: 'text', text: 'Transition between these frames' },
                    { type: 'image', uri: 'gs://inputs/first.png', mime_type: 'image/png' },
                    { type: 'image', uri: 'gs://inputs/last.png', mime_type: 'image/png' },
                ],
                response_format: [expect.objectContaining({ resolution: '4k' })],
                generation_config: { video_config: { task: 'image_to_video' } },
            },
        });
    });

    it.each([
        ['reference_to_video', [image('gs://inputs/reference.png'), video('gs://inputs/reference.mp4')]],
        ['edit', [video('gs://inputs/source.mp4')]],
        ['extend', [video('gs://inputs/source.mp4')]],
    ] as const)('maps media inputs for the 1.1 %s task', async (task, files) => {
        const definition = new GeminiOmniVideoModelDefinition(GEMINI_OMNI_1_1_VIDEO_MODEL);
        const configured = options(
            { model_options: { _option_id: 'vertexai-gemini-omni-video', task } },
            GEMINI_OMNI_1_1_VIDEO_MODEL,
        );

        const prompt = await definition.createPrompt(
            {} as VertexAIDriver,
            [{ role: PromptRole.user, content: 'Transform this', files: [...files] }],
            configured,
        );

        expect(prompt.media.map((item) => item.type)).toEqual(files.map((file) => file.mime_type.split('/')[0]));
    });

    it('supports video editing on the 1.0 model', async () => {
        const definition = new GeminiOmniVideoModelDefinition();
        const configured = options({
            model_options: { _option_id: 'vertexai-gemini-omni-video', task: 'edit' },
        });

        const prompt = await definition.createPrompt(
            {} as VertexAIDriver,
            [{ role: PromptRole.user, content: 'Remove the sign', files: [video()] }],
            configured,
        );

        expect(prompt.media).toEqual([
            { type: 'video', uri: 'gs://project-bucket/input/clip.mp4', mime_type: 'video/mp4' },
        ]);
    });

    it.each([
        [GEMINI_OMNI_VIDEO_MODEL, 'extend', [video()], 'does not support extend'],
        [GEMINI_OMNI_VIDEO_MODEL, 'image_to_video', [image(), image('gs://inputs/last.png')], 'exactly one'],
        [GEMINI_OMNI_1_1_VIDEO_MODEL, 'edit', [image()], 'exactly one video'],
        [GEMINI_OMNI_1_1_VIDEO_MODEL, 'extend', [video(), video('gs://inputs/second.mp4')], 'exactly one video'],
    ] as const)('rejects invalid %s %s media combinations', async (model, task, files, message) => {
        const configured = options({ model_options: { _option_id: 'vertexai-gemini-omni-video', task } }, model);

        await expect(
            new GeminiOmniVideoModelDefinition(model).createPrompt(
                {} as VertexAIDriver,
                [{ role: PromptRole.user, content: 'prompt', files: [...files] }],
                configured,
            ),
        ).rejects.toThrow(message);
    });

    it.each([
        [Array.from({ length: 11 }, (_, index) => image(`gs://inputs/${index}.png`)), 'ten images'],
        [Array.from({ length: 4 }, (_, index) => video(`gs://inputs/${index}.mp4`)), 'three videos'],
    ])('enforces documented 1.1 reference attachment limits', async (files, message) => {
        const configured = options(
            { model_options: { _option_id: 'vertexai-gemini-omni-video', task: 'reference_to_video' } },
            GEMINI_OMNI_1_1_VIDEO_MODEL,
        );

        await expect(
            new GeminiOmniVideoModelDefinition(GEMINI_OMNI_1_1_VIDEO_MODEL).createPrompt(
                {} as VertexAIDriver,
                [{ role: PromptRole.user, content: 'prompt', files }],
                configured,
            ),
        ).rejects.toThrow(message);
    });

    it('rejects non-720p output on 1.0', async () => {
        const configured = options({
            model_options: { _option_id: 'vertexai-gemini-omni-video', resolution: '1080p' },
        });

        await expect(
            new GeminiOmniVideoModelDefinition().createPrompt(
                {} as VertexAIDriver,
                [{ role: PromptRole.user, content: 'prompt' }],
                configured,
            ),
        ).rejects.toThrow('only supports 720p');
    });

    it('returns supported text output alongside video output', async () => {
        const definition = new GeminiOmniVideoModelDefinition(GEMINI_OMNI_1_1_VIDEO_MODEL);
        const configured = options({}, GEMINI_OMNI_1_1_VIDEO_MODEL);
        const prompt = await definition.createPrompt(
            {} as VertexAIDriver,
            [{ role: PromptRole.user, content: 'prompt' }],
            configured,
        );
        const stub = driverWithResponse(
            completedResponse(
                { type: 'text', value: 'Generated video' },
                { type: 'video', value: `${OUTPUT_PREFIX}result.mp4` },
            ),
        );

        const completion = await definition.requestTextCompletion(stub.driver, prompt, configured);

        expect(completion.result).toEqual([
            { type: 'text', value: 'Generated video' },
            { type: 'video', value: `${OUTPUT_PREFIX}result.mp4` },
        ]);
    });

    it.each([
        ['empty prompt', [{ role: PromptRole.user, content: '  ' }], options(), 'non-empty'],
        [
            'unsupported MIME type',
            [{ role: PromptRole.user, content: 'prompt', files: [image('gs://inputs/a.gif', 'image/gif')] }],
            options({ model_options: { _option_id: 'vertexai-gemini-omni-video', task: 'image_to_video' } }),
            'MIME type',
        ],
        [
            'audio reference',
            [{ role: PromptRole.user, content: 'prompt', files: [media('gs://inputs/a.wav', 'audio/wav')] }],
            options({ model_options: { _option_id: 'vertexai-gemini-omni-video', task: 'edit' } }),
            'MIME type',
        ],
        [
            'non-GCS image',
            [{ role: PromptRole.user, content: 'prompt', files: [image('https://example.com/a.png')] }],
            options({ model_options: { _option_id: 'vertexai-gemini-omni-video', task: 'image_to_video' } }),
            'GCS object URI',
        ],
    ])('rejects %s', async (_name, segments, configured, message) => {
        await expect(
            new GeminiOmniVideoModelDefinition().createPrompt({} as VertexAIDriver, segments, configured),
        ).rejects.toThrow(message);
    });

    it.each([
        ['tools', options({ tools: [{ name: 'lookup', input_schema: { type: 'object' } }] }), 'tools'],
        ['result schemas', options({ result_schema: { type: 'object' } }), 'result schemas'],
        ['conversation continuation', options({ conversation: { id: 'interaction-0' } }), 'conversation resume'],
    ])('rejects unsupported %s', async (_name, configured, message) => {
        await expect(
            new GeminiOmniVideoModelDefinition().createPrompt(
                {} as VertexAIDriver,
                [{ role: PromptRole.user, content: 'prompt' }],
                configured,
            ),
        ).rejects.toThrow(message);
    });

    it.each([
        [{ status: 'in_progress' }, 'did not complete'],
        [completedResponse(), 'without a video'],
        [
            completedResponse({ type: 'video', value: 'gs://foreign-bucket/video.mp4' }),
            'outside the requested output prefix',
        ],
        [completedResponse({ type: 'video', value: 'https://example.com/video.mp4' }), 'invalid GCS video URI'],
        [
            {
                id: 'interaction-1',
                status: 'completed',
                steps: [{ type: 'model_output', content: [{ type: 'video', data: 'AAAA', mime_type: 'video/mp4' }] }],
            },
            'inline video data',
        ],
    ])('rejects incomplete or invalid provider output', async (response, message) => {
        const definition = new GeminiOmniVideoModelDefinition();
        const prompt = await definition.createPrompt(
            {} as VertexAIDriver,
            [{ role: PromptRole.user, content: 'prompt' }],
            options(),
        );

        await expect(
            definition.requestTextCompletion(driverWithResponse(response).driver, prompt, options()),
        ).rejects.toThrow(message);
    });

    it('preserves the original response when requested', async () => {
        const response = completedResponse({ type: 'video', value: `${OUTPUT_PREFIX}result.mp4` });
        const definition = new GeminiOmniVideoModelDefinition();
        const configured = options({ include_original_response: true });
        const prompt = await definition.createPrompt(
            {} as VertexAIDriver,
            [{ role: PromptRole.user, content: 'prompt' }],
            configured,
        );

        const completion = await definition.requestTextCompletion(
            driverWithResponse(response).driver,
            prompt,
            configured,
        );

        expect(completion.original_response).toBe(response);
    });

    it('rejects streaming', async () => {
        await expect(new GeminiOmniVideoModelDefinition().requestTextCompletionStream()).rejects.toThrow(
            'does not support streaming',
        );
    });

    it('keeps deliberate cancellation terminal and delegates transport failures to shared retry handling', () => {
        const definition = new GeminiOmniVideoModelDefinition();
        const context = {
            provider: 'vertexai',
            model: GEMINI_OMNI_VIDEO_MODEL,
            operation: 'execute' as const,
        };
        const cancellation = Object.assign(new Error('cancelled'), { name: 'AbortError' });
        const formatted = definition.formatLlumiverseError({} as VertexAIDriver, cancellation, context);

        expect(formatted).toBeInstanceOf(LlumiverseError);
        expect(formatted.retryable).toBe(false);

        for (const error of [
            Object.assign(new Error('timed out'), { name: 'TimeoutError' }),
            Object.assign(new Error('gateway timeout'), { status: 504 }),
        ]) {
            expect(() => definition.formatLlumiverseError({} as VertexAIDriver, error, context)).toThrow(error);
        }
    });

    it.each([
        ['in_progress', undefined, true],
        ['incomplete', undefined, true],
        ['requires_action', undefined, false],
        ['cancelled', undefined, false],
        ['budget_exceeded', undefined, false],
        ['failed', [{ code: 'RESOURCE_EXHAUSTED', message: 'Try later' }], true],
        ['failed', [{ code: 'PERMISSION_DENIED', message: 'Forbidden' }], false],
        ['failed', [{ code: 'UNKNOWN', message: 'Unknown failure' }], undefined],
    ] as const)('classifies interaction status %s errors %j with retryable=%s', async (status, errors, retryable) => {
        const definition = new GeminiOmniVideoModelDefinition();
        const prompt = await definition.createPrompt(
            {} as VertexAIDriver,
            [{ role: PromptRole.user, content: 'prompt' }],
            options(),
        );
        let thrown: unknown;
        try {
            await definition.requestTextCompletion(
                driverWithResponse({ id: 'interaction-1', status, errors }).driver,
                prompt,
                options(),
            );
        } catch (error) {
            thrown = error;
        }

        const formatted = definition.formatLlumiverseError({} as VertexAIDriver, thrown, {
            provider: 'vertexai',
            model: GEMINI_OMNI_VIDEO_MODEL,
            operation: 'execute',
        });
        expect(formatted.retryable).toBe(retryable);
    });
});
