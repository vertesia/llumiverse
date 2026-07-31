import { z } from 'zod';
import type {
    BedrockAI21Options,
    BedrockClaudeOptions,
    BedrockCohereCommandOptions,
    BedrockConverseOptions,
    BedrockGptOssOptions,
    BedrockMistralOptions,
    BedrockNovaOptions,
    BedrockPalmyraOptions,
    NovaCanvasOptions,
    TwelvelabsPegasusOptions,
} from '../options/bedrock.js';
import type {
    BedrockMantleChatCompletionsOptions,
    BedrockMantleClaudeOptions,
    BedrockMantleResponsesOptions,
} from '../options/bedrock_mantle.js';
import type { TextFallbackOptions } from '../options/fallback.js';
import type { GroqOptions } from '../options/groq.js';
import type {
    OpenAiDalleOptions,
    OpenAiGptImageOptions,
    OpenAiTextOptions,
    OpenAiThinkingOptions,
} from '../options/openai.js';
import type {
    ImagenOptions,
    VertexAIClaudeOptions,
    VertexAIGeminiOptions,
    VertexAIGrokOptions,
} from '../options/vertexai.js';
import { ImagenMaskMode, ImagenTaskType, ThinkingLevel } from '../options/vertexai.js';

// Runtime schemas for `ModelOptions` — the union of every driver's per-model options — and its
// twenty-seven members.
//
// `//` rather than `/** */` throughout: a JSDoc block immediately preceding an exported declaration is
// picked up by Vertesia's OpenAPI scanner and published as that component's `description`.
//
// These are BRIDGES in the same sense as `./json-schema.js`, and for the same reason: the scanner
// still derives components from TypeScript for every slot that has not converted, and it resolves a
// `z.infer<>` alias to nothing. The interfaces in `../options/*` therefore stay the public types.
// Unlike `JSONSchema`, none of these is recursive, so when nothing derives them any more the
// interfaces can be deleted outright and the public types become `z.infer` of the schemas below.
//
// Every schema is checked against its interface by {@link FieldSchemas}, which is what makes a
// generated schema honest: it requires the Zod shape to cover every field of the interface exactly,
// so a field on one side and not the other fails to compile here. The annotation alone would not —
// these interfaces have one required property and the rest optional, so a schema declaring only
// `_option_id` would satisfy `z.ZodType<T>`.

// `strictObject`, not `object`. The published component has always said `additionalProperties: false`,
// and Zod's default would have PARSED an unknown option key by silently dropping it — so the document
// promised rejection while the schema quietly discarded. `strictObject` rejects, and emits the
// `additionalProperties: false` the component already carried, so the published and enforced contracts
// are the same statement. (`JSONSchema` next door is the opposite case for the opposite reason: a JSON
// Schema legitimately carries keywords the type never enumerated.)

// Exact per-field coverage. `-?` is the load-bearing part: an optional key in the mapped type would
// make this a subset check and let a missing property through.
type FieldSchemas<T> = {
    [K in keyof Required<T>]-?: z.ZodType<T[K]>;
};

// The four option enums that are published as their own components.
export const ImagenMaskModeSchema = z.enum(ImagenMaskMode).meta({ id: 'ImagenMaskMode' });

export const ImagenTaskTypeSchema = z.enum(ImagenTaskType).meta({ id: 'ImagenTaskType' });

export const ThinkingLevelSchema = z.enum(ThinkingLevel).meta({ id: 'ThinkingLevel' });

export const ReasoningEffortSchema = z
    .enum(['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'])
    .meta({ id: 'ReasoningEffort' });

// ===== fallback =====

export const TextFallbackOptionsSchema = z
    .strictObject({
        _option_id: z.literal('text-fallback'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        top_k: z.number().optional(),
        presence_penalty: z.number().optional(),
        frequency_penalty: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
    } satisfies FieldSchemas<TextFallbackOptions>)
    .meta({ id: 'TextFallbackOptions' });

// ===== groq =====

export const GroqOptionsSchema = z
    .strictObject({
        _option_id: z.literal('groq-deepseek-thinking'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        presence_penalty: z.number().optional(),
        frequency_penalty: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        reasoning_format: z.enum(['parsed', 'raw', 'hidden']),
    } satisfies FieldSchemas<GroqOptions>)
    .meta({ id: 'GroqOptions' });

// ===== bedrock =====

export const BedrockConverseOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-converse'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
    } satisfies FieldSchemas<BedrockConverseOptions>)
    .meta({ id: 'BedrockConverseOptions' });

export const BedrockNovaOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-nova'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
    } satisfies FieldSchemas<BedrockNovaOptions>)
    .meta({ id: 'BedrockNovaOptions' });

export const BedrockMistralOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-mistral'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
    } satisfies FieldSchemas<BedrockMistralOptions>)
    .meta({ id: 'BedrockMistralOptions' });

export const BedrockAI21OptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-ai21'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
    } satisfies FieldSchemas<BedrockAI21Options>)
    .meta({ id: 'BedrockAI21Options' });

export const BedrockCohereCommandOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-cohere-command'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
    } satisfies FieldSchemas<BedrockCohereCommandOptions>)
    .meta({ id: 'BedrockCohereCommandOptions' });

export const BedrockClaudeOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-claude'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        top_k: z.number().optional(),
        thinking_budget_tokens: z.number().optional(),
        include_thoughts: z.boolean().optional(),
        effort: z.enum(['low', 'medium', 'high', 'xhigh', 'max']).optional(),
        cache_enabled: z.boolean().optional(),
        cache_ttl: z.enum(['5m', '1h']).optional(),
    } satisfies FieldSchemas<BedrockClaudeOptions>)
    .meta({ id: 'BedrockClaudeOptions' });

export const BedrockPalmyraOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-palmyra'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        min_tokens: z.number().optional(),
        seed: z.number().optional(),
        frequency_penalty: z.number().optional(),
        presence_penalty: z.number().optional(),
    } satisfies FieldSchemas<BedrockPalmyraOptions>)
    .meta({ id: 'BedrockPalmyraOptions' });

export const BedrockGptOssOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-gpt-oss'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        reasoning_effort: z.enum(['low', 'medium', 'high']).optional(),
        frequency_penalty: z.number().optional(),
        presence_penalty: z.number().optional(),
    } satisfies FieldSchemas<BedrockGptOssOptions>)
    .meta({ id: 'BedrockGptOssOptions' });

export const TwelvelabsPegasusOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-twelvelabs-pegasus'),
        temperature: z.number().optional(),
        max_tokens: z.number().optional(),
    } satisfies FieldSchemas<TwelvelabsPegasusOptions>)
    .meta({ id: 'TwelvelabsPegasusOptions' });

export const NovaCanvasOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-nova-canvas'),
        taskType: z.enum([
            'TEXT_IMAGE',
            'TEXT_IMAGE_WITH_IMAGE_CONDITIONING',
            'COLOR_GUIDED_GENERATION',
            'IMAGE_VARIATION',
            'INPAINTING',
            'OUTPAINTING',
            'BACKGROUND_REMOVAL',
        ]),
        width: z.number().optional(),
        height: z.number().optional(),
        quality: z.enum(['standard', 'premium']).optional(),
        cfgScale: z.number().optional(),
        seed: z.number().optional(),
        numberOfImages: z.number().optional(),
        controlMode: z.enum(['CANNY_EDGE', 'SEGMENTATION']).optional(),
        controlStrength: z.number().optional(),
        colors: z.array(z.string()).optional(),
        similarityStrength: z.number().optional(),
        outPaintingMode: z.enum(['DEFAULT', 'PRECISE']).optional(),
    } satisfies FieldSchemas<NovaCanvasOptions>)
    .meta({ id: 'NovaCanvasOptions' });

// ===== bedrock_mantle =====

export const BedrockMantleResponsesOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-mantle-responses'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        effort: z.enum(['none', 'low', 'medium', 'high', 'xhigh']).optional(),
        reasoning_effort: z.enum(['none', 'low', 'medium', 'high', 'xhigh']).optional(),
        verbosity: z.enum(['low', 'medium', 'high']).optional(),
        image_detail: z.enum(['low', 'high', 'auto']).optional(),
    } satisfies FieldSchemas<BedrockMantleResponsesOptions>)
    .meta({ id: 'BedrockMantleResponsesOptions' });

export const BedrockMantleChatCompletionsOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-mantle-chat-completions'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
    } satisfies FieldSchemas<BedrockMantleChatCompletionsOptions>)
    .meta({ id: 'BedrockMantleChatCompletionsOptions' });

export const BedrockMantleClaudeOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-mantle-claude'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        top_k: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        effort: z.enum(['low', 'medium', 'high', 'xhigh', 'max']).optional(),
        thinking_budget_tokens: z.number().optional(),
        include_thoughts: z.boolean().optional(),
        cache_enabled: z.boolean().optional(),
        cache_ttl: z.enum(['5m', '1h']).optional(),
    } satisfies FieldSchemas<BedrockMantleClaudeOptions>)
    .meta({ id: 'BedrockMantleClaudeOptions' });

// ===== openai =====

export const OpenAiThinkingOptionsSchema = z
    .strictObject({
        _option_id: z.literal('openai-thinking'),
        max_tokens: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        effort: ReasoningEffortSchema.optional(),
        reasoning_effort: ReasoningEffortSchema.optional(),
        image_detail: z.enum(['low', 'high', 'auto']).optional(),
    } satisfies FieldSchemas<OpenAiThinkingOptions>)
    .meta({ id: 'OpenAiThinkingOptions' });

export const OpenAiTextOptionsSchema = z
    .strictObject({
        _option_id: z.literal('openai-text'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        presence_penalty: z.number().optional(),
        frequency_penalty: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        image_detail: z.enum(['low', 'high', 'auto']).optional(),
    } satisfies FieldSchemas<OpenAiTextOptions>)
    .meta({ id: 'OpenAiTextOptions' });

export const OpenAiDalleOptionsSchema = z
    .strictObject({
        _option_id: z.literal('openai-dalle'),
        size: z.enum(['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792']).optional(),
        image_quality: z.enum(['standard', 'hd']).optional(),
        style: z.enum(['vivid', 'natural']).optional(),
        response_format: z.enum(['url', 'b64_json']).optional(),
        n: z.number().optional(),
    } satisfies FieldSchemas<OpenAiDalleOptions>)
    .meta({ id: 'OpenAiDalleOptions' });

export const OpenAiGptImageOptionsSchema = z
    .strictObject({
        _option_id: z.literal('openai-gpt-image'),
        size: z.enum(['1024x1024', '1024x1536', '1536x1024', 'auto']).optional(),
        image_quality: z.enum(['low', 'medium', 'high', 'auto']).optional(),
        background: z.enum(['transparent', 'opaque', 'auto']).optional(),
        output_format: z.enum(['png', 'webp', 'jpeg']).optional(),
    } satisfies FieldSchemas<OpenAiGptImageOptions>)
    .meta({ id: 'OpenAiGptImageOptions' });

// ===== vertexai =====

export const ImagenOptionsSchema = z
    .strictObject({
        _option_id: z.literal('vertexai-imagen'),
        number_of_images: z.number().optional(),
        seed: z.number().optional(),
        person_generation: z.enum(['dont_allow', 'allow_adults', 'allow_all']).optional(),
        safety_setting: z
            .enum(['block_none', 'block_only_high', 'block_medium_and_above', 'block_low_and_above'])
            .optional(),
        image_file_type: z.enum(['image/jpeg', 'image/png']).optional(),
        jpeg_compression_quality: z.number().optional(),
        aspect_ratio: z.enum(['1:1', '4:3', '3:4', '16:9', '9:16']).optional(),
        add_watermark: z.boolean().optional(),
        enhance_prompt: z.boolean().optional(),
        edit_mode: ImagenTaskTypeSchema.optional(),
        guidance_scale: z.number().optional(),
        edit_steps: z.number().optional(),
        mask_mode: ImagenMaskModeSchema.optional(),
        mask_dilation: z.number().optional(),
        mask_class: z.array(z.number()).optional(),
        controlType: z.enum(['CONTROL_TYPE_FACE_MESH', 'CONTROL_TYPE_CANNY', 'CONTROL_TYPE_SCRIBBLE']).optional(),
        controlImageComputation: z.boolean().optional(),
        subjectType: z
            .enum(['SUBJECT_TYPE_PERSON', 'SUBJECT_TYPE_ANIMAL', 'SUBJECT_TYPE_PRODUCT', 'SUBJECT_TYPE_DEFAULT'])
            .optional(),
    } satisfies FieldSchemas<ImagenOptions>)
    .meta({ id: 'ImagenOptions' });

export const VertexAIClaudeOptionsSchema = z
    .strictObject({
        _option_id: z.literal('vertexai-claude'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        top_k: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        effort: z.enum(['low', 'medium', 'high', 'xhigh', 'max']).optional(),
        thinking_budget_tokens: z.number().optional(),
        include_thoughts: z.boolean().optional(),
        cache_enabled: z.boolean().optional(),
        cache_ttl: z.enum(['5m', '1h']).optional(),
    } satisfies FieldSchemas<VertexAIClaudeOptions>)
    .meta({ id: 'VertexAIClaudeOptions' });

export const VertexAIGeminiOptionsSchema = z
    .strictObject({
        _option_id: z.literal('vertexai-gemini'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        top_k: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        presence_penalty: z.number().optional(),
        frequency_penalty: z.number().optional(),
        seed: z.number().optional(),
        effort: z.enum(['minimal', 'low', 'medium', 'high']).optional(),
        include_thoughts: z.boolean().optional(),
        thinking_budget_tokens: z.number().optional(),
        thinking_level: ThinkingLevelSchema.optional(),
        flex: z.boolean().optional(),
        image_aspect_ratio: z.enum(['1:1', '2:3', '3:2', '3:4', '4:3', '9:16', '16:9', '21:9']).optional(),
        image_size: z.enum(['1K', '2K', '4K']).optional(),
        person_generation: z.enum(['ALLOW_ALL', 'ALLOW_ADULT', 'ALLOW_NONE']).optional(),
        prominent_people: z
            .enum(['PROMINENT_PEOPLE_UNSPECIFIED', 'ALLOW_PROMINENT_PEOPLE', 'BLOCK_PROMINENT_PEOPLE'])
            .optional(),
        output_mime_type: z.enum(['image/png', 'image/jpeg']).optional(),
        output_compression_quality: z.number().optional(),
    } satisfies FieldSchemas<VertexAIGeminiOptions>)
    .meta({ id: 'VertexAIGeminiOptions' });

export const VertexAIGrokOptionsSchema = z
    .strictObject({
        _option_id: z.literal('vertexai-grok'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
    } satisfies FieldSchemas<VertexAIGrokOptions>)
    .meta({ id: 'VertexAIGrokOptions' });

// The discriminated union. Member order is the order the derived component lists, which the adapter
// turns back into a `discriminator` + `mapping` keyed on `_option_id`; a generated Java or Go client
// reads that mapping to pick the concrete subtype.
export const ModelOptionsSchema = z
    .discriminatedUnion('_option_id', [
        TextFallbackOptionsSchema,
        ImagenOptionsSchema,
        VertexAIClaudeOptionsSchema,
        VertexAIGeminiOptionsSchema,
        VertexAIGrokOptionsSchema,
        NovaCanvasOptionsSchema,
        BedrockConverseOptionsSchema,
        BedrockNovaOptionsSchema,
        BedrockMistralOptionsSchema,
        BedrockAI21OptionsSchema,
        BedrockCohereCommandOptionsSchema,
        BedrockClaudeOptionsSchema,
        BedrockPalmyraOptionsSchema,
        BedrockGptOssOptionsSchema,
        TwelvelabsPegasusOptionsSchema,
        BedrockMantleResponsesOptionsSchema,
        BedrockMantleChatCompletionsOptionsSchema,
        BedrockMantleClaudeOptionsSchema,
        OpenAiThinkingOptionsSchema,
        OpenAiTextOptionsSchema,
        OpenAiDalleOptionsSchema,
        OpenAiGptImageOptionsSchema,
        GroqOptionsSchema,
    ])
    .meta({ id: 'ModelOptions' });
