import { z } from 'zod';

// Runtime schemas for the embedding result types that studio-server publishes from
// `POST /environments/:envId/embeddings`.
//
// The request half of that endpoint is NOT here: the wire format a caller sends carries a
// JSON-friendly source (a URL or base64 text) rather than the `DataSource` a driver consumes, so it
// is a Vertesia type (`EmbeddingsApiRequest`) rather than a llumiverse one. The result half is
// identical on both sides — vectors and counts are already JSON — so it is defined once, here.
//
// `//` rather than `/** */` throughout: a JSDoc block immediately preceding an exported declaration
// is picked up by Vertesia's OpenAPI scanner and published as that component's `description`.

export const EmbeddingTaskTypeSchema = z.enum(['query', 'document']).meta({
    id: 'EmbeddingTaskType',
    description:
        'Semantic task type for embedding models. Drivers map these to provider- specific values ' +
        '(e.g. "RETRIEVAL_QUERY" for Vertex, "search_query" for Cohere).\n' +
        '- "query"    — a search query to find relevant documents\n' +
        '- "document" — a document to be indexed and retrieved',
});

export const EmbeddingOutputSchema = z
    .strictObject({
        values: z.array(z.number()),
        modality: z
            .enum(['text', 'image', 'video', 'audio'])
            .meta({ description: 'Which modality this vector represents (useful for joint-multimodal results).' })
            .optional(),
        start_sec: z.number().meta({ description: 'Segment start time for video/audio.' }).optional(),
        end_sec: z.number().meta({ description: 'Segment end time for video/audio.' }).optional(),
        embedding_option: z
            .string()
            .meta({ description: 'TwelveLabs Marengo: which view of the segment this vector represents.' })
            .optional(),
    })
    .meta({ id: 'EmbeddingOutput' });

export const EmbeddingResultItemSchema = z
    .strictObject({
        outputs: z.array(EmbeddingOutputSchema).meta({
            description:
                'One or more vectors produced for this input. Single vector for text/image; multiple for ' +
                'segmented video/audio or joint-multimodal models that return per-modality vectors.',
        }),
        input_tokens: z
            .number()
            .meta({ description: 'Token count attributed to this input, when reported by the provider.' })
            .optional(),
    })
    .meta({ id: 'EmbeddingResultItem' });

export const EmbeddingsTokenUsageSchema = z
    .strictObject({
        input_tokens: z.number().optional(),
        input_text_tokens: z.number().optional(),
        input_image_tokens: z.number().optional(),
    })
    .meta({ id: 'EmbeddingsTokenUsage' });

export const EmbeddingsResultSchema = z
    .strictObject({
        results: z.array(EmbeddingResultItemSchema).meta({
            description: 'One result item per input, in the same order as EmbeddingsOptions.inputs.',
        }),
        model: z.string().meta({ description: 'The provider model id that produced the result.' }),
        usage: EmbeddingsTokenUsageSchema.meta({
            description: 'Aggregate token usage when reported by the provider.',
        }).optional(),
    })
    .meta({ id: 'EmbeddingsResult' });
