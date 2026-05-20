import { describe, expect, it } from 'vitest';
import { ModelType, Providers } from '../types.js';
import {
    EMBEDDING_MODEL_CATALOG,
    embeddingDescriptorToAIModel,
    enrichWithEmbeddingCatalog,
    getDefaultEmbeddingModel,
    getEmbeddingModelDescriptor,
    getEmbeddingModelsForProvider,
    isEmbeddingModel,
} from './embedding.js';

describe('embedding model catalog', () => {
    it('every provider with a default model has a matching catalog entry', () => {
        const providersWithDefaults: Providers[] = [
            Providers.openai,
            Providers.vertexai,
            Providers.bedrock,
            Providers.mistralai,
            Providers.watsonx,
        ];

        for (const provider of providersWithDefaults) {
            const defaultId = getDefaultEmbeddingModel(provider);
            expect(defaultId, `provider ${provider} has a default model`).toBeDefined();
            const descriptor = getEmbeddingModelDescriptor(provider, defaultId!);
            expect(descriptor, `provider ${provider} default ${defaultId} is in catalog`).toBeDefined();
        }
    });

    it('every catalog entry has a positive default_dimensions and consistent supported_dimensions', () => {
        for (const [provider, descriptors] of Object.entries(EMBEDDING_MODEL_CATALOG)) {
            for (const d of descriptors ?? []) {
                expect(d.embedding.default_dimensions, `${provider}/${d.id} default_dimensions > 0`)
                    .toBeGreaterThan(0);

                if (d.embedding.supported_dimensions) {
                    expect(
                        d.embedding.supported_dimensions,
                        `${provider}/${d.id} supported_dimensions includes default`,
                    ).toContain(d.embedding.default_dimensions);
                }

                expect(d.input_modalities.length, `${provider}/${d.id} has at least one input modality`)
                    .toBeGreaterThan(0);
            }
        }
    });

    it('embeddingDescriptorToAIModel produces a well-formed AIModel', () => {
        const descriptor = EMBEDDING_MODEL_CATALOG[Providers.openai]![0];
        const model = embeddingDescriptorToAIModel(Providers.openai, descriptor);

        expect(model.id).toBe(descriptor.id);
        expect(model.provider).toBe(Providers.openai);
        expect(model.type).toBe(ModelType.Embedding);
        expect(model.embedding).toEqual(descriptor.embedding);
        expect(model.input_modalities).toEqual(descriptor.input_modalities);
        expect(model.output_modalities).toEqual(['vector']);
    });

    it('getEmbeddingModelsForProvider returns the full provider catalog as AIModels', () => {
        const openaiModels = getEmbeddingModelsForProvider(Providers.openai);
        expect(openaiModels.length).toBe(EMBEDDING_MODEL_CATALOG[Providers.openai]!.length);
        for (const m of openaiModels) {
            expect(m.type).toBe(ModelType.Embedding);
            expect(m.embedding?.default_dimensions).toBeGreaterThan(0);
        }
    });

    it('getEmbeddingModelsForProvider returns [] for providers without a catalog entry', () => {
        expect(getEmbeddingModelsForProvider(Providers.anthropic)).toEqual([]);
    });

    it('enrichWithEmbeddingCatalog attaches capabilities to known ids', () => {
        const stub = [{
            id: 'text-embedding-3-small',
            name: 'text-embedding-3-small',
            provider: Providers.openai,
        }];

        const [enriched] = enrichWithEmbeddingCatalog(Providers.openai, stub);

        expect(enriched.type).toBe(ModelType.Embedding);
        expect(enriched.embedding?.default_dimensions).toBe(1536);
        expect(enriched.embedding?.supports_dimension_truncation).toBe(true);
        expect(enriched.input_modalities).toEqual(['text']);
    });

    it('enrichWithEmbeddingCatalog passes unknown ids through with type set', () => {
        const stub = [{
            id: 'unknown-embed-model',
            name: 'unknown-embed-model',
            provider: Providers.openai,
        }];

        const [enriched] = enrichWithEmbeddingCatalog(Providers.openai, stub);

        expect(enriched.id).toBe('unknown-embed-model');
        expect(enriched.type).toBe(ModelType.Embedding);
        expect(enriched.embedding).toBeUndefined();
    });

    it('enrichWithEmbeddingCatalog drops non-embedding leaks as a defensive backstop', () => {
        const stub = [
            { id: 'text-embedding-3-small', name: 'text-embedding-3-small', provider: Providers.openai },
            { id: 'gpt-4o', name: 'gpt-4o', provider: Providers.openai }, // not an embedding model
            { id: 'ibm/slate-rtrvr', name: 'IBM Slate Retriever', provider: Providers.openai, type: ModelType.Embedding },
        ];

        const enriched = enrichWithEmbeddingCatalog(Providers.openai, stub);

        const ids = enriched.map((m) => m.id);
        expect(ids).toContain('text-embedding-3-small');
        expect(ids).toContain('ibm/slate-rtrvr'); // kept because type was already set
        expect(ids).not.toContain('gpt-4o');
    });

    it('isEmbeddingModel prefers the type check, falls back to the id substring', () => {
        expect(isEmbeddingModel({ id: 'gpt-4o', name: 'gpt-4o', provider: 'openai', type: ModelType.Embedding })).toBe(true);
        expect(isEmbeddingModel({ id: 'text-embedding-3-small', name: 'text-embedding-3-small', provider: 'openai' })).toBe(true);
        expect(isEmbeddingModel({ id: 'gpt-4o', name: 'gpt-4o', provider: 'openai' })).toBe(false);
        expect(isEmbeddingModel({ id: 'ibm/slate-125m-english-rtrvr', name: 'Slate', provider: 'watsonx' })).toBe(false);
    });
});
