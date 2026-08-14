import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import type { ModelOptions } from '../types.js';
import { ModelOptionsSchema } from './model-options.js';

/** Exact type identity — `extends` in both directions is too weak (`any`/`unknown` slip through). */
type Equals<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
function assertType<T extends true>(_ok: T): void {}

/**
 * `ModelOptions` is published by Vertesia as a discriminated-union component with twenty-four
 * members, each its own component. These pin the properties that a consumer of the published document
 * depends on and that a careless edit here would break silently — the union is large enough that a
 * dropped member reads as a normal diff.
 */
const emitted = z.toJSONSchema(ModelOptionsSchema, { target: 'draft-2020-12', io: 'input' }) as {
    anyOf?: { $ref: string }[];
    oneOf?: { $ref: string }[];
    $defs: Record<string, { properties?: Record<string, unknown>; required?: string[] }>;
};

const MEMBERS = (emitted.oneOf ?? emitted.anyOf ?? []).map((member) => member.$ref.replace('#/$defs/', ''));

describe('ModelOptionsSchema', () => {
    it('validates strict Gemini Omni video option boundaries', () => {
        expect(
            ModelOptionsSchema.safeParse({
                _option_id: 'vertexai-gemini-omni-video',
                task: 'reference_to_video',
                aspect_ratio: '16:9',
                duration_seconds: 3,
            }).success,
        ).toBe(true);
        expect(
            ModelOptionsSchema.safeParse({ _option_id: 'vertexai-gemini-omni-video', duration_seconds: 10 }).success,
        ).toBe(true);
        expect(
            ModelOptionsSchema.safeParse({ _option_id: 'vertexai-gemini-omni-video', duration_seconds: 2 }).success,
        ).toBe(false);
        expect(
            ModelOptionsSchema.safeParse({ _option_id: 'vertexai-gemini-omni-video', duration_seconds: 5.5 }).success,
        ).toBe(false);
        expect(ModelOptionsSchema.safeParse({ _option_id: 'vertexai-gemini-omni-video', unknown: true }).success).toBe(
            false,
        );
    });
    it('is the only definition of the union — the public type is inferred from it', () => {
        // Vacuous as an equality, and that is the point: `ModelOptions` in `../types.js` IS
        // `z.infer` of this schema, so there is no second declaration for it to disagree with. The
        // assertion is kept because it fails to COMPILE if the public type is ever redeclared as a
        // hand-written union — which is the regression, not an inequality at runtime.
        assertType<Equals<ModelOptions, z.infer<typeof ModelOptionsSchema>>>(true);
        // A real value, checked against the schema rather than only against the compiler.
        expect(
            ModelOptionsSchema.safeParse({ _option_id: 'vertexai-gemini', temperature: 0.2 } as ModelOptions).success,
        ).toBe(true);
    });

    it('carries every driver option set, in the published order', () => {
        // Order is significant: it becomes the `oneOf` order in the document, which decides the branch
        // order a generated Java or Go client tries.
        expect(MEMBERS).toEqual([
            'TextFallbackOptions',
            'AzureFoundryChatOptions',
            'ImagenOptions',
            'VertexAIClaudeOptions',
            'VertexAIGeminiOptions',
            'VertexAIGeminiOmniVideoOptions',
            'VertexAIGrokOptions',
            'NovaCanvasOptions',
            'BedrockConverseOptions',
            'BedrockNovaOptions',
            'BedrockMistralOptions',
            'BedrockAI21Options',
            'BedrockCohereCommandOptions',
            'BedrockClaudeOptions',
            'BedrockPalmyraOptions',
            'BedrockGptOssOptions',
            'TwelvelabsPegasusOptions',
            'BedrockMantleResponsesOptions',
            'BedrockMantleChatCompletionsOptions',
            'BedrockMantleClaudeOptions',
            'OpenAiThinkingOptions',
            'OpenAiTextOptions',
            'OpenAiDalleOptions',
            'OpenAiGptImageOptions',
            'GroqOptions',
            'MistralTextOptions',
        ]);
    });

    it('discriminates on a required, unique _option_id in every member', () => {
        // What makes the union a `discriminator` + `mapping` in the document rather than a bare
        // `oneOf`. Two members sharing a literal, or one leaving `_option_id` optional, silently
        // demotes it and generated clients fall back to a loose map.
        const ids = MEMBERS.map((name) => {
            const member = emitted.$defs[name];
            expect(member.required, `${name} must require _option_id`).toContain('_option_id');
            const discriminant = member.properties?._option_id as { const?: string } | undefined;
            expect(discriminant?.const, `${name} must pin a literal _option_id`).toBeTypeOf('string');
            return discriminant?.const;
        });
        expect(new Set(ids).size).toBe(ids.length);
    });

    it('closes every member, so an unknown option is rejected rather than dropped', () => {
        // `z.object` would publish `additionalProperties: false` — the component has always said so —
        // while silently STRIPPING an unknown option at parse time. `strictObject` makes the enforced
        // behaviour the published one, and since the public type is inferred from this schema, it is
        // also what the compiler enforces.
        for (const name of MEMBERS) {
            expect((emitted.$defs[name] as { additionalProperties?: unknown }).additionalProperties, name).toBe(false);
        }
    });

    it('parses a real option object and rejects a foreign key', () => {
        const options = { _option_id: 'text-fallback', max_tokens: 100, temperature: 0.7 } as const;
        expect(ModelOptionsSchema.parse(options)).toEqual(options);
        expect(ModelOptionsSchema.safeParse({ _option_id: 'text-fallback', nope: 1 }).success).toBe(false);
        expect(ModelOptionsSchema.safeParse({ _option_id: 'not-a-driver' }).success).toBe(false);
    });

    it('accepts current and future service tiers for provider option schemas', () => {
        expect(ModelOptionsSchema.safeParse({ _option_id: 'openai-text', service_tier: 'flex' }).success).toBe(true);
        expect(ModelOptionsSchema.safeParse({ _option_id: 'openai-thinking', service_tier: 'priority' }).success).toBe(
            true,
        );
        expect(
            ModelOptionsSchema.safeParse({ _option_id: 'vertexai-gemini', service_tier: 'future-tier' }).success,
        ).toBe(true);
        expect(ModelOptionsSchema.safeParse({ _option_id: 'vertexai-gemini', flex: true }).success).toBe(true);
        expect(
            ModelOptionsSchema.safeParse({ _option_id: 'bedrock-claude', service_tier: 'future-tier' }).success,
        ).toBe(true);
        expect(
            ModelOptionsSchema.safeParse({ _option_id: 'bedrock-twelvelabs-pegasus', service_tier: 'flex' }).success,
        ).toBe(true);
    });
});
