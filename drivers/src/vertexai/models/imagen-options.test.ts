import type { MaskReferenceConfig } from '@google/genai';
import { helpers, type protos } from '@google-cloud/aiplatform';
import { ImagenMaskMode, type ImagenOptions, ImagenTaskType } from '@llumiverse/common';
import { Base64DataSource, PromptRole } from '@llumiverse/core';
import { describe, expect, it, vi } from 'vitest';
import { VertexAIDriver } from '../index.js';
import { ImagenModelDefinition } from './imagen.js';

describe('Imagen person generation options', () => {
    it.each(['allow_adult', 'allow_adults'] satisfies ImagenOptions['person_generation'][])(
        'sends the provider spelling for %s',
        async (person_generation) => {
            const driver = new VertexAIDriver({ project: 'test-project', region: 'us-central1' });
            const predict = vi.fn().mockResolvedValue([{ predictions: [] }]);
            vi.spyOn(driver, 'getImagenClient').mockResolvedValue({ predict } as unknown as Awaited<
                ReturnType<VertexAIDriver['getImagenClient']>
            >);
            const model = 'imagen-3.0-generate-002';
            await new ImagenModelDefinition(model).requestImageGeneration(
                driver,
                { prompt: 'A person' },
                {
                    model,
                    model_options: { _option_id: 'vertexai-imagen', person_generation },
                },
            );
            expect(predict).toHaveBeenCalledWith(
                expect.objectContaining({
                    parameters: expect.objectContaining({
                        structValue: expect.objectContaining({
                            fields: expect.objectContaining({
                                personGeneration: expect.objectContaining({ stringValue: 'allow_adult' }),
                            }),
                        }),
                    }),
                }),
                expect.any(Object),
            );
        },
    );
});

// Wire names and numeric mask classes are specified by Google's Imagen REST API:
// https://cloud.google.com/vertex-ai/generative-ai/docs/image/edit-insert-objects
// and the installed @google/genai editImageConfigToVertex/maskReferenceConfigToVertex mappings.
async function imagenRequest(model_options: ImagenOptions, roles: PromptRole[] = [PromptRole.user]) {
    const driver = new VertexAIDriver({ project: 'test-project', region: 'us-central1' });
    const predict = vi
        .fn<
            (
                request: protos.google.cloud.aiplatform.v1.IPredictRequest,
                options: unknown,
            ) => Promise<[protos.google.cloud.aiplatform.v1.IPredictResponse]>
        >()
        .mockResolvedValue([{ predictions: [] }]);
    vi.spyOn(driver, 'getImagenClient').mockResolvedValue({ predict } as unknown as Awaited<
        ReturnType<VertexAIDriver['getImagenClient']>
    >);
    const options = { model: 'imagen-3.0-capability-001', model_options };
    const definition = new ImagenModelDefinition(options.model);
    const prompt = await definition.createPrompt(
        driver,
        roles.map((role) => ({
            role,
            content: 'Edit the image',
            files: [new Base64DataSource('image.png', 'image/png', 'AQID')],
        })),
        options,
    );
    await definition.requestImageGeneration(driver, prompt, options);
    const request = predict.mock.calls[0][0];
    if (!request.parameters || !request.instances?.[0]) throw new Error('Expected an Imagen prediction request');
    return {
        parameters: helpers.fromValue(request.parameters as Parameters<typeof helpers.fromValue>[0]),
        instance: helpers.fromValue(request.instances[0] as Parameters<typeof helpers.fromValue>[0]),
    };
}

describe('Imagen option serialization', () => {
    it.each([
        ImagenTaskType.EDIT_MODE_INPAINT_REMOVAL,
        ImagenTaskType.EDIT_MODE_INPAINT_INSERTION,
        ImagenTaskType.EDIT_MODE_BGSWAP,
        ImagenTaskType.EDIT_MODE_OUTPAINT,
    ] satisfies ImagenOptions['edit_mode'][])('forwards numeric semantic mask classes for %s', async (edit_mode) => {
        const mask_class = [175, 176] satisfies NonNullable<MaskReferenceConfig['segmentationClasses']>;
        const request = await imagenRequest({
            _option_id: 'vertexai-imagen',
            edit_mode,
            mask_mode: ImagenMaskMode.MASK_MODE_SEMANTIC,
            mask_class,
            mask_dilation: 0.01,
            edit_steps: 35,
            guidance_scale: 60,
        });
        expect(request.instance).toMatchObject({
            referenceImages: [
                { referenceType: 'REFERENCE_TYPE_RAW' },
                {
                    referenceType: 'REFERENCE_TYPE_MASK',
                    maskImageConfig: {
                        maskMode: ImagenMaskMode.MASK_MODE_SEMANTIC,
                        maskClasses: [175, 176],
                        dilation: 0.01,
                    },
                },
            ],
        });
        expect(request.parameters).toMatchObject({
            editMode: edit_mode,
            editConfig: { baseSteps: 35 },
            guidanceScale: 60,
        });
    });

    it.each([
        ImagenMaskMode.MASK_MODE_BACKGROUND,
        ImagenMaskMode.MASK_MODE_FOREGROUND,
        ImagenMaskMode.MASK_MODE_USER_PROVIDED,
    ] as const)('does not send stale semantic classes for %s', async (mask_mode) => {
        const request = await imagenRequest(
            {
                _option_id: 'vertexai-imagen',
                edit_mode: ImagenTaskType.EDIT_MODE_INPAINT_INSERTION,
                mask_mode,
                mask_class: [175],
                mask_dilation: 0.01,
            },
            mask_mode === ImagenMaskMode.MASK_MODE_USER_PROVIDED
                ? [PromptRole.user, PromptRole.mask]
                : [PromptRole.user],
        );
        expect(request.instance).toMatchObject({
            referenceImages: [
                { referenceType: 'REFERENCE_TYPE_RAW' },
                { referenceType: 'REFERENCE_TYPE_MASK', maskImageConfig: { maskMode: mask_mode, dilation: 0.01 } },
            ],
        });
        expect(JSON.stringify(request.instance)).not.toContain('maskClasses');
    });

    it.each([
        ImagenTaskType.TEXT_IMAGE,
        ImagenTaskType.EDIT_MODE_INPAINT_INSERTION,
    ] satisfies ImagenOptions['edit_mode'][])(
        'preserves output options and explicit zero/false values for %s',
        async (edit_mode) => {
            const request = await imagenRequest({
                _option_id: 'vertexai-imagen',
                edit_mode,
                number_of_images: 2,
                seed: 0,
                person_generation: 'allow_adults',
                safety_setting: 'block_medium_and_above',
                image_file_type: 'image/jpeg',
                jpeg_compression_quality: 0,
                guidance_scale: 0,
                ...(edit_mode === ImagenTaskType.TEXT_IMAGE
                    ? { add_watermark: false, enhance_prompt: false, aspect_ratio: '4:3' }
                    : {}),
            });
            expect(request.parameters).toMatchObject({
                sampleCount: 2,
                seed: 0,
                personGeneration: 'allow_adult',
                safetySetting: 'block_medium_and_above',
                guidanceScale: 0,
                outputOptions: { mimeType: 'image/jpeg', compressionQuality: 0 },
                ...(edit_mode === ImagenTaskType.TEXT_IMAGE
                    ? { addWatermark: false, enhancePrompt: false, aspectRatio: '4:3' }
                    : {}),
            });
        },
    );

    it('uses the advertised user-provided mask default and omits unspecified edit settings', async () => {
        const request = await imagenRequest(
            { _option_id: 'vertexai-imagen', edit_mode: ImagenTaskType.EDIT_MODE_INPAINT_INSERTION },
            [PromptRole.user, PromptRole.mask],
        );
        expect(request.instance).toMatchObject({
            referenceImages: [
                { referenceType: 'REFERENCE_TYPE_RAW' },
                {
                    referenceType: 'REFERENCE_TYPE_MASK',
                    referenceImage: { bytesBase64Encoded: 'AQID' },
                    maskImageConfig: { maskMode: ImagenMaskMode.MASK_MODE_USER_PROVIDED },
                },
            ],
        });
        expect(request.parameters).not.toHaveProperty('editConfig');
    });

    it('leaves unspecified output options to the provider default', async () => {
        const request = await imagenRequest({ _option_id: 'vertexai-imagen' });
        expect(request.parameters).not.toHaveProperty('outputOptions');
        expect(request.parameters).not.toHaveProperty('guidanceScale');
    });

    it.each([
        ImagenTaskType.CUSTOMIZATION_SUBJECT,
        ImagenTaskType.CUSTOMIZATION_CONTROLLED,
    ] satisfies ImagenOptions['edit_mode'][])(
        'preserves the Scribble control and false computation setting for %s',
        async (edit_mode) => {
            const request = await imagenRequest({
                _option_id: 'vertexai-imagen',
                edit_mode,
                controlType: 'CONTROL_TYPE_SCRIBBLE',
                controlImageComputation: false,
            });
            expect(request.instance).toMatchObject({
                referenceImages: [
                    {
                        referenceType: 'REFERENCE_TYPE_CONTROL',
                        controlImageConfig: {
                            controlType: 'CONTROL_TYPE_SCRIBBLE',
                            enableControlImageComputation: false,
                        },
                    },
                ],
            });
        },
    );
});
