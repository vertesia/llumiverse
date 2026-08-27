import { z } from 'zod';
import { ImagenMaskMode, ImagenTaskType, ThinkingLevel } from '../options/vertexai.js';

// Runtime schemas for `ModelOptions` — the union of every driver's per-model options.
//
// These are the SINGLE definition of each option set. The OpenAPI document publishes them, AJV
// enforces them, and the public TypeScript types in `../options/*` are `z.infer` of them, so there
// is no second declaration to drift from. OpenAPI publishes these registered schemas directly.
//
// `//` rather than `/** */` throughout: a JSDoc block immediately preceding an exported declaration is
// picked up by Vertesia's OpenAPI scanner and published as that component's `description`.

// `strictObject`, not `object`. The published component has always said `additionalProperties: false`,
// and Zod's default would have PARSED an unknown option key by silently dropping it — so the document
// promised rejection while the schema quietly discarded. `strictObject` rejects, and emits the
// `additionalProperties: false` the component already carried, so the published and enforced contracts
// are the same statement. (`JSONSchema` next door is the opposite case for the opposite reason: a JSON
// Schema legitimately carries keywords the type never enumerated.)

// The four option enums that are published as their own components.
export const ImagenMaskModeSchema = z.enum(ImagenMaskMode).meta({ id: 'ImagenMaskMode' });

export const ImagenTaskTypeSchema = z.enum(ImagenTaskType).meta({ id: 'ImagenTaskType' });

export const ThinkingLevelSchema = z.enum(ThinkingLevel).meta({ id: 'ThinkingLevel' });

export const ReasoningEffortSchema = z
    .enum(['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'])
    .meta({ id: 'ReasoningEffort' });

const ServiceTierSchema = z
    .string()
    .min(1)
    .describe('Provider-defined processing tier. Unknown non-empty values are preserved for forward compatibility.');

const ExtraBodySchema = z
    .record(z.string(), z.unknown())
    .describe('Additional provider-specific fields merged into the OpenAI-compatible request body.');

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
        include_thoughts: z.boolean().optional(),
    })
    .meta({ id: 'TextFallbackOptions' });

// ===== azure_foundry =====

export const AzureFoundryChatOptionsSchema = z
    .strictObject({
        _option_id: z.literal('azure-foundry-chat'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        presence_penalty: z.number().optional(),
        frequency_penalty: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        seed: z.number().optional(),
        image_detail: z.enum(['low', 'high', 'auto']).optional(),
        include_thoughts: z.boolean().optional(),
    })
    .meta({ id: 'AzureFoundryChatOptions' });

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
    })
    .meta({ id: 'GroqOptions' });

// ===== mistral =====

export const MistralTextOptionsSchema = z
    .strictObject({
        _option_id: z.literal('mistral-text'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        presence_penalty: z.number().optional(),
        frequency_penalty: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        effort: z.enum(['none', 'high']).optional(),
        random_seed: z.number().int().optional(),
        safe_prompt: z.boolean().optional(),
        parallel_tool_calls: z.boolean().optional(),
        tool_choice: z.enum(['auto', 'none', 'any', 'required']).optional(),
        prompt_mode: z.literal('reasoning').optional(),
        include_thoughts: z.boolean().optional(),
    })
    .meta({ id: 'MistralTextOptions' });

// ===== bedrock =====

export const BedrockConverseOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-converse'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        include_thoughts: z.boolean().optional(),
        service_tier: ServiceTierSchema.optional(),
    })
    .meta({ id: 'BedrockConverseOptions' });

export const BedrockNovaOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-nova'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        include_thoughts: z.boolean().optional(),
        service_tier: ServiceTierSchema.optional(),
    })
    .meta({ id: 'BedrockNovaOptions' });

export const BedrockMistralOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-mistral'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        include_thoughts: z.boolean().optional(),
        service_tier: ServiceTierSchema.optional(),
    })
    .meta({ id: 'BedrockMistralOptions' });

export const BedrockAI21OptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-ai21'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        include_thoughts: z.boolean().optional(),
        service_tier: ServiceTierSchema.optional(),
    })
    .meta({ id: 'BedrockAI21Options' });

export const BedrockCohereCommandOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-cohere-command'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        include_thoughts: z.boolean().optional(),
        service_tier: ServiceTierSchema.optional(),
    })
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
        service_tier: ServiceTierSchema.optional(),
    })
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
        service_tier: ServiceTierSchema.optional(),
    })
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
        service_tier: ServiceTierSchema.optional(),
    })
    .meta({ id: 'BedrockGptOssOptions' });

export const TwelvelabsPegasusOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-twelvelabs-pegasus'),
        temperature: z.number().optional(),
        max_tokens: z.number().optional(),
        service_tier: ServiceTierSchema.optional(),
    })
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
    })
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
        include_thoughts: z.boolean().optional(),
    })
    .meta({ id: 'BedrockMantleResponsesOptions' });

export const BedrockMantleChatCompletionsOptionsSchema = z
    .strictObject({
        _option_id: z.literal('bedrock-mantle-chat-completions'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        effort: z.enum(['low', 'medium', 'high']).optional(),
        reasoning_effort: z.enum(['low', 'medium', 'high']).optional(),
        include_thoughts: z.boolean().optional(),
    })
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
    })
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
        include_thoughts: z.boolean().optional(),
        service_tier: ServiceTierSchema.optional(),
        extra_body: ExtraBodySchema.optional(),
    })
    .meta({ id: 'OpenAiThinkingOptions' });

export const OpenAiTextOptionsSchema = z
    .strictObject({
        _option_id: z.literal('openai-text'),
        max_tokens: z.number().optional(),
        effort: ReasoningEffortSchema.optional(),
        reasoning_effort: ReasoningEffortSchema.optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        presence_penalty: z.number().optional(),
        frequency_penalty: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
        image_detail: z.enum(['low', 'high', 'auto']).optional(),
        include_thoughts: z.boolean().optional(),
        service_tier: ServiceTierSchema.optional(),
        extra_body: ExtraBodySchema.optional(),
    })
    .meta({ id: 'OpenAiTextOptions' });

export const OpenRouterTextOptionsSchema = OpenAiTextOptionsSchema.omit({ _option_id: true, extra_body: true })
    .extend({
        _option_id: z.literal('openrouter-text'),
        provider_sort: z.enum(['price', 'throughput', 'latency', 'exacto']).optional(),
        provider_order: z.array(z.string()).optional(),
        provider_only: z.array(z.string()).optional(),
        provider_ignore: z.array(z.string()).optional(),
        provider_allow_fallbacks: z.boolean().optional(),
        provider_require_parameters: z.boolean().optional(),
        provider_data_collection: z.enum(['allow', 'deny']).optional(),
        provider_zdr: z.boolean().optional(),
        provider_quantizations: z
            .array(
                z.enum([
                    'int4',
                    'int8',
                    'fp4',
                    'mxfp4',
                    'nvfp4',
                    'fp6',
                    'fp8',
                    'mxfp8',
                    'fp16',
                    'bf16',
                    'fp32',
                    'unknown',
                ]),
            )
            .optional(),
    })
    .meta({ id: 'OpenRouterTextOptions' });

export const OpenAiDalleOptionsSchema = z
    .strictObject({
        _option_id: z.literal('openai-dalle'),
        size: z.enum(['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792']).optional(),
        image_quality: z.enum(['standard', 'hd']).optional(),
        style: z.enum(['vivid', 'natural']).optional(),
        response_format: z.enum(['url', 'b64_json']).optional(),
        n: z.number().optional(),
    })
    .meta({ id: 'OpenAiDalleOptions' });

export const OpenAiGptImageOptionsSchema = z
    .strictObject({
        _option_id: z.literal('openai-gpt-image'),
        size: z.enum(['1024x1024', '1024x1536', '1536x1024', 'auto']).optional(),
        image_quality: z.enum(['low', 'medium', 'high', 'auto']).optional(),
        background: z.enum(['transparent', 'opaque', 'auto']).optional(),
        output_format: z.enum(['png', 'webp', 'jpeg']).optional(),
    })
    .meta({ id: 'OpenAiGptImageOptions' });

// ===== xai =====

export const XAIGrokImageOptionsSchema = z
    .strictObject({
        _option_id: z.literal('xai-grok-image'),
        aspect_ratio: z
            .enum([
                '1:1',
                '16:9',
                '9:16',
                '4:3',
                '3:4',
                '3:2',
                '2:3',
                '2:1',
                '1:2',
                '19.5:9',
                '9:19.5',
                '20:9',
                '9:20',
                'auto',
            ])
            .optional(),
        resolution: z.enum(['1k', '2k']).optional(),
        quality: z.enum(['low', 'medium']).optional(),
        response_format: z.enum(['url', 'b64_json']).optional(),
        n: z.number().int().min(1).max(10).optional(),
    })
    .meta({ id: 'XAIGrokImageOptions' });

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
    })
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
    })
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
        service_tier: ServiceTierSchema.optional(),
        // TODO: Remove this deprecated alias after main's OpenAPI compatibility baseline advances past release/1.4.
        flex: z
            .boolean()
            .meta({
                description: 'Deprecated: Use service_tier="flex" instead.',
                deprecated: true,
                'x-deprecated-message': 'Use service_tier="flex" instead.',
            })
            .optional(),
        image_aspect_ratio: z.enum(['1:1', '2:3', '3:2', '3:4', '4:3', '9:16', '16:9', '21:9']).optional(),
        image_size: z.enum(['1K', '2K', '4K']).optional(),
        person_generation: z.enum(['ALLOW_ALL', 'ALLOW_ADULT', 'ALLOW_NONE']).optional(),
        prominent_people: z
            .enum(['PROMINENT_PEOPLE_UNSPECIFIED', 'ALLOW_PROMINENT_PEOPLE', 'BLOCK_PROMINENT_PEOPLE'])
            .optional(),
        output_mime_type: z.enum(['image/png', 'image/jpeg']).optional(),
        output_compression_quality: z.number().optional(),
    })
    .meta({ id: 'VertexAIGeminiOptions' });

export const VertexAIGeminiOmniVideoOptionsSchema = z
    .strictObject({
        _option_id: z.literal('vertexai-gemini-omni-video'),
        task: z.enum(['text_to_video', 'image_to_video', 'reference_to_video']).optional(),
        aspect_ratio: z.enum(['16:9', '9:16']).optional(),
        duration_seconds: z.number().int().min(3).max(10).optional(),
    })
    .meta({ id: 'VertexAIGeminiOmniVideoOptions' });

export const VertexAIGrokOptionsSchema = z
    .strictObject({
        _option_id: z.literal('vertexai-grok'),
        max_tokens: z.number().optional(),
        temperature: z.number().optional(),
        top_p: z.number().optional(),
        stop_sequence: z.array(z.string()).optional(),
    })
    .meta({ id: 'VertexAIGrokOptions' });

// The discriminated union. Member order is the order the derived component lists, which the adapter
// turns back into a `discriminator` + `mapping` keyed on `_option_id`; a generated Java or Go client
// reads that mapping to pick the concrete subtype.
export const ModelOptionsSchema = z
    .discriminatedUnion('_option_id', [
        TextFallbackOptionsSchema,
        AzureFoundryChatOptionsSchema,
        ImagenOptionsSchema,
        VertexAIClaudeOptionsSchema,
        VertexAIGeminiOptionsSchema,
        VertexAIGeminiOmniVideoOptionsSchema,
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
        OpenRouterTextOptionsSchema,
        OpenAiDalleOptionsSchema,
        OpenAiGptImageOptionsSchema,
        XAIGrokImageOptionsSchema,
        GroqOptionsSchema,
        MistralTextOptionsSchema,
    ])
    .meta({ id: 'ModelOptions' });
