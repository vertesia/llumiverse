import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import type { ModelOptions } from '../types.js';
import { ModelOptionsSchema } from './model-options.js';

/** Exact type identity — `extends` in both directions is too weak (`any`/`unknown` slip through). */
type Equals<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
function assertType<T extends true>(_ok: T): void {}

/**
 * `ModelOptions` is published by Vertesia as a discriminated-union component with twenty-three
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
    it('infers the public union exactly, so the bridge can be deleted rather than reconciled', () => {
        // The per-member `satisfies FieldSchemas<T>` checks each option set field by field; this
        // checks the UNION, which nothing else does — a member present in the TypeScript union and
        // absent from `discriminatedUnion` type-checks everywhere else in the file.
        //
        // It is also the exit condition for the bridge. `ModelOptions` is not recursive, so once no
        // derived component references these the interfaces go away and the public type becomes
        // `z.infer<typeof ModelOptionsSchema>`. This asserts today that the swap is a rename.
        assertType<Equals<ModelOptions, z.infer<typeof ModelOptionsSchema>>>(true);
        expect(true).toBe(true);
    });

    it('carries every driver option set, in the published order', () => {
        // Order is significant: it becomes the `oneOf` order in the document, which decides the branch
        // order a generated Java or Go client tries.
        expect(MEMBERS).toEqual([
            'TextFallbackOptions',
            'ImagenOptions',
            'VertexAIClaudeOptions',
            'VertexAIGeminiOptions',
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
        // behaviour the published one.
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
});
