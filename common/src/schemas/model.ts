import { z } from 'zod';
import { AIModelStatus, ModelType } from '../types.js';

// Runtime schemas for the model-description types that studio-server publishes: what a model IS
// (`AIModel`), the two closed vocabularies describing it, and the payload that searches for one.
//
// These are the SINGLE definition of each shape — the OpenAPI document publishes them, AJV enforces
// them, and the public types in `../types.ts` are `z.infer` of them.
//
// `//` rather than `/** */` throughout: a JSDoc block immediately preceding an exported declaration
// is picked up by Vertesia's OpenAPI scanner and published as that component's `description`.

// `z.enum` over the TypeScript enum rather than a second list of strings. Both are published
// identically, but the members stay one declaration: `ModelType.Text` is read in a dozen drivers and
// in the Studio UI, so the enum has to survive, and a hand-copied string list beside it would be the
// exact drift this migration removes.
export const AIModelStatusSchema = z.enum(AIModelStatus).meta({ id: 'AIModelStatus' });

export const ModelTypeSchema = z.enum(ModelType).meta({ id: 'ModelType' });

// `strictObject`: `additionalProperties: false` is what the component has always published, and
// `z.object` would have silently dropped an unknown key instead of rejecting it.
export const AIModelSchema = z
    .strictObject({
        id: z.string(),
        name: z.string(),
        // A driver's own provider id, and every caller in and out of this repo instantiates it as a
        // plain string. It used to be a type parameter (`AIModel<ProviderKeys = string>`) that was
        // only ever passed `string`; a canonical alias cannot carry one, and nothing was using it.
        provider: z.string(),
        description: z.string().optional(),
        version: z.string().optional(),
        type: ModelTypeSchema.optional(),
        tags: z.array(z.string()).optional(),
        owner: z.string().optional(),
        status: AIModelStatusSchema.optional(),
        can_stream: z.boolean().optional(),
        is_custom: z.boolean().optional(),
        is_multimodal: z.boolean().optional(),
        input_modalities: z.array(z.string()).optional(),
        output_modalities: z.array(z.string()).optional(),
        tool_support: z.boolean().optional(),
        environment: z.string().optional(),
    })
    .meta({ id: 'AIModel' });

export const AIModelArraySchema = z.array(AIModelSchema).meta({ id: 'AIModelArray' });

// `text` is OPTIONAL, where the derived parameter said `required: true`. The interface declared it
// non-optional, but `EnvironmentsApi.listModels(id, payload?)` takes the whole payload optionally and
// three Studio call sites pass nothing — so the document promised a parameter that its own client
// omits and the server never wanted. Enforcing it would have 400'd the UI; publishing it optional
// says what has always been true.
export const ModelSearchPayloadSchema = z
    .strictObject({
        text: z.string().optional(),
        type: ModelTypeSchema.optional(),
        tags: z.array(z.string()).optional(),
        owner: z.string().optional(),
    })
    .meta({ id: 'ModelSearchPayload' });
