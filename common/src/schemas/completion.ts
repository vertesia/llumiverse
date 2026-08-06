import { z } from 'zod';
import type { JSONObject, JSONValue } from '../types.js';
// biome-ignore lint/suspicious/noDeprecatedImports: publishing a deprecated field's schema requires the deprecated enum
import { Modalities, PromptRole } from '../types.js';
import { HttpTimeoutOptionsSchema } from './http-timeout.js';
import { JSONSchemaSchema } from './json-schema.js';
import { ModelOptionsSchema } from './model-options.js';

// Runtime schemas for the completion types that reach a public API contract: what a prompt is made
// of, what a model returns, and the execution options a caller may send.
//
// `//` rather than `/** */` throughout: a JSDoc block immediately preceding an exported declaration
// is picked up by Vertesia's OpenAPI scanner and published as that component's `description`.

// `z.any()` annotated as `ZodType<JSONValue>`, the same arrangement `JSONSchemaSchema` uses and for
// the same reason. `JSONValue` is recursive, so a faithful Zod schema would be a `z.lazy` union —
// and would publish that union, where the component has always been the unconstrained `{}`. So the
// RUNTIME schema stays unconstrained (a value that came out of `JSON.parse` is JSON by
// construction), and the annotation gives every field built from these the type the API documents:
// `test_data?: JSONObject` has to stay `JSONObject`, not decay to `{[k: string]: unknown}`.
export const JSONValueSchema: z.ZodType<JSONValue> = z.any().meta({ id: 'JSONValue' });

// A map, written as an object with a catchall rather than `z.record`: `z.record` also emits a
// `propertyNames: {type: 'string'}` that the index signature never published.
export const JSONObjectSchema: z.ZodType<JSONObject> = z
    .object({})
    .catchall(JSONValueSchema)
    .meta({ id: 'JSONObject' });

// `z.enum(PromptRole)` over the TS enum rather than a restated member list: the enum is a value that
// callers write (`{ role: PromptRole.user }`), and a string-literal union here would infer a type
// those calls no longer satisfy. Same for `Modalities`.
export const PromptRoleSchema = z.enum(PromptRole).meta({ id: 'PromptRole' });

export const ModalitiesSchema = z.enum(Modalities).meta({ id: 'Modalities' });

// The wire form of a data source: what it is called and what it contains. The BYTES are not here —
// a `DataSource` in memory can hand back a stream, and the published component has only ever
// described these two fields.
export const DataSourceSchema = z
    .strictObject({
        name: z.string(),
        mime_type: z.string(),
    })
    .meta({ id: 'DataSource' });

export const PromptSegmentSchema = z
    .strictObject({
        role: PromptRoleSchema,
        content: z.string(),
        tool_use_id: z.string().meta({ description: 'The tool use id if the segment is a tool response' }).optional(),
        thought_signature: z
            .string()
            .meta({
                description:
                    'Gemini thinking models require thought_signature to be passed back with tool results. ' +
                    'This should be copied from the ToolUse.thought_signature when sending tool responses.',
            })
            .optional(),
        files: z.array(DataSourceSchema).optional(),
    })
    .meta({ id: 'PromptSegment' });

export const ToolDefinitionSchema = z
    .strictObject({
        name: z.string(),
        description: z.string().optional(),
        // `looseObject`, matching the permissive `input_schema` the published component has always
        // carried: it holds either an AJV `JSONSchemaType<T>` or a plain object schema. The
        // `additionalProperties: true` is stated rather than left as Zod's equivalent `{}` because
        // `true` is the spelling the published document carries, and a component the registry owns
        // has to be published exactly as the registry defines it.
        input_schema: z.looseObject({}).meta({ additionalProperties: true }),
    })
    .meta({
        id: 'ToolDefinition',
        description:
            'Tool definition for LLM tool use. The input_schema uses a permissive type to support both:\n' +
            "- AJV's JSONSchemaType<T> for type-safe schema generation\n" +
            '- Plain object schemas for simpler cases',
    });

export const ToolUseSchema = z
    .strictObject({
        id: z.string(),
        tool_name: z.string(),
        tool_input: z.union([JSONObjectSchema, z.null()]),
        thought_signature: z
            .string()
            .meta({
                description:
                    'Gemini thinking models require thought_signature to be passed back with tool results. This ' +
                    "preserves the model's reasoning state during multi-turn tool use.",
            })
            .optional(),
    })
    .meta({
        id: 'ToolUse',
        description:
            'A tool use instance represents a call to a tool. The id property is used to identify the tool call.',
    });

export const TextResultSchema = z
    .strictObject({ type: z.literal('text'), value: z.string() })
    .meta({ id: 'TextResult' });

export const ThoughtsResultSchema = z
    .strictObject({ type: z.literal('thoughts'), value: z.string() })
    .meta({ id: 'ThoughtsResult' });

export const JsonResultSchema = z
    .strictObject({ type: z.literal('json'), value: JSONValueSchema })
    .meta({ id: 'JsonResult' });

export const ImageResultSchema = z
    .strictObject({ type: z.literal('image'), value: z.string() })
    .meta({ id: 'ImageResult' });

export const CompletionResultSchema = z
    .discriminatedUnion('type', [TextResultSchema, ThoughtsResultSchema, JsonResultSchema, ImageResultSchema])
    .meta({ id: 'CompletionResult' });

export const ExecutionTokenUsageSchema = z
    .strictObject({
        prompt: z.number().optional(),
        result: z.number().optional(),
        total: z.number().optional(),
        prompt_cached: z
            .number()
            .meta({ description: 'Number of input tokens read from prompt cache (discounted rate).' })
            .optional(),
        prompt_cache_write: z
            .number()
            .meta({ description: 'Number of input tokens written to prompt cache.' })
            .optional(),
        prompt_new: z.number().optional(),
    })
    .meta({ id: 'ExecutionTokenUsage' });

// `format` is NOT here, and its absence is the point. `PromptOptions.format` is a `PromptFormatter`
// — a FUNCTION — which the scanner published as an object with `namedArgs` and `returns` that no
// caller could ever send. It stays on `ExecutionOptions`, which is the in-process type the drivers
// take, so the wire component describes only what can cross a wire.
export const StatelessExecutionOptionsSchema = z
    .strictObject({
        model: z.string(),
        result_schema: JSONSchemaSchema.optional(),
        prompt_cache_schema_suffix: z
            .boolean()
            .meta({
                description:
                    'Provider-specific opt-in to put the result schema after the cached prompt prefix instead of ' +
                    'including it in native structured-output configuration. The returned JSON is still validated ' +
                    'against result_schema by Llumiverse.',
            })
            .optional(),
        include_original_response: z
            .boolean()
            .meta({
                description:
                    'If set to true the original response from the target LLM will be included in the response ' +
                    'under the original_response field. This is useful for debugging and for some advanced use ' +
                    'cases. It is ignored on streaming requests',
            })
            .optional(),
        model_options: ModelOptionsSchema.optional(),
        prompt_cache_key: z
            .string()
            .meta({
                description:
                    'Stable identity for prompt caching. Providers with cache routing keys receive the value ' +
                    'directly; providers with cache breakpoints use its presence to cache the stable prefix before ' +
                    'the final dynamic block. Providers with fully implicit caching still require an identical ' +
                    'prompt prefix.',
            })
            .optional(),
        httpTimeout: HttpTimeoutOptionsSchema.meta({
            description:
                "Per-call HTTP timeouts for upstream LLM-provider calls. These override the driver's default " +
                '`DriverOptions.httpTimeout` for this execution only.',
        }).optional(),
        output_modality: ModalitiesSchema.meta({
            description: 'Deprecated: This is deprecated. Use CompletionResult.type information instead.',
            deprecated: true,
            'x-deprecated-message': 'This is deprecated. Use CompletionResult.type information instead.',
        }).optional(),
    })
    .meta({ id: 'StatelessExecutionOptions' });
