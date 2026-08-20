import { type DataSource, type ExecutionOptions, LlumiverseError, PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import type { VertexAIDriver } from '../index.js';
import { getModelDefinition } from '../models.js';
import { GEMINI_OMNI_VIDEO_MODEL, GeminiOmniVideoModelDefinition } from './omni-video.js';

const OUTPUT_PREFIX = 'gs://project-bucket/runs/run-1/media/';

function image(uri = 'gs://project-bucket/input/frame.png', mimeType = 'image/png'): DataSource {
    return {
        name: 'frame.png',
        mime_type: mimeType,
        getURI: vi.fn().mockResolvedValue(uri),
        getURL: vi.fn(),
        getStream: vi.fn(),
    };
}

function options(overrides: Partial<ExecutionOptions> = {}): ExecutionOptions {
    return {
        model: GEMINI_OMNI_VIDEO_MODEL,
        model_options: { _option_id: 'vertexai-gemini-omni-video' },
        output_storage_uri: OUTPUT_PREFIX,
        ...overrides,
    };
}

function completedResponse(...uris: string[]) {
    return {
        id: 'interaction-1',
        status: 'completed',
        usage: { total_tokens: 9, total_input_tokens: 3, total_output_tokens: 6 },
        steps: [
            { type: 'thought', summary: [{ type: 'text', text: 'hidden' }] },
            {
                type: 'model_output',
                content: uris.map((uri) => ({ type: 'video', uri, mime_type: 'video/mp4' })),
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
    it('routes the exact model to the dedicated non-streaming definition', () => {
        const definition = getModelDefinition('publishers/google/models/gemini-omni-flash-preview');

        expect(definition).toBeInstanceOf(GeminiOmniVideoModelDefinition);
        expect(definition.model).toMatchObject({ type: 'video', can_stream: false });
    });

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
        const stub = driverWithResponse(completedResponse(`${OUTPUT_PREFIX}video.mp4`));

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
        const stub = driverWithResponse(completedResponse(`${OUTPUT_PREFIX}one.mp4`, `${OUTPUT_PREFIX}two.mp4`));

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

    it('accepts one to three reference images and requires an explicit image task', async () => {
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

        expect(prompt.images).toHaveLength(3);
        await expect(
            definition.createPrompt(
                {} as VertexAIDriver,
                [{ role: PromptRole.user, content: 'Use this', files: [image()] }],
                options(),
            ),
        ).rejects.toThrow('requires an explicit');
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
        [{ status: 'in_progress' }, 'did not complete'],
        [completedResponse(), 'without a video'],
        [completedResponse('gs://foreign-bucket/video.mp4'), 'outside the requested output prefix'],
        [completedResponse('https://example.com/video.mp4'), 'invalid GCS video URI'],
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
