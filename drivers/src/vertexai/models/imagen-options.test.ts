import type { ImagenOptions } from '@llumiverse/common';
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
