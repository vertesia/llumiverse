import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import { getOptions } from '../options.js';
import { type ModelOptions, type ModelOptionsInfo, Providers } from '../types.js';
import { ModelOptionsSchema } from './model-options.js';

/** Exact type identity — `extends` in both directions is too weak (`any`/`unknown` slip through). */
type Equals<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
function assertType<T extends true>(_ok: T): void {}

/**
 * `ModelOptions` is published by Vertesia as a union component with named
 * members, each its own component. These pin the properties that a consumer of the published document
 * depends on and that a careless edit here would break silently — the union is large enough that a
 * dropped member reads as a normal diff.
 */
const rawEmitted = z.toJSONSchema(ModelOptionsSchema, { target: 'draft-2020-12', io: 'input' }) as {
    $ref?: string;
    type?: string;
    anyOf?: { $ref: string }[];
    oneOf?: { $ref: string }[];
    $defs: Record<string, { properties?: Record<string, unknown>; required?: string[] }>;
};
const emitted = rawEmitted.$ref
    ? { ...rawEmitted.$defs[rawEmitted.$ref.replace('#/$defs/', '')], $defs: rawEmitted.$defs }
    : rawEmitted;

const MEMBERS = (emitted.oneOf ?? emitted.anyOf ?? []).map((member) => member.$ref.replace('#/$defs/', ''));

describe('ModelOptionsSchema', () => {
    it('validates strict Gemini Omni video option boundaries', () => {
        expect(
            ModelOptionsSchema.safeParse({
                _option_id: 'vertexai-gemini-omni-video',
                task: 'reference_to_video',
                aspect_ratio: '16:9',
                duration_seconds: 3,
                resolution: '4k',
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
        expect(ModelOptionsSchema.safeParse({ _option_id: 'vertexai-gemini-omni-video', task: 'edit' }).success).toBe(
            true,
        );
        expect(
            ModelOptionsSchema.safeParse({ _option_id: 'vertexai-gemini-omni-video', resolution: '1440p' }).success,
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

    it('publishes every registered schema in union order', () => {
        expect(MEMBERS).toEqual(ModelOptionsSchema.options.map((schema) => schema.meta()?.id));
    });

    it('requires factory IDs to belong to the schema union at compile time', () => {
        assertType<Equals<ModelOptionsInfo['_option_id'], NonNullable<ModelOptions['_option_id']>>>(true);
        // Factory metadata must still identify a family, unlike caller-authored payloads.
        // @ts-expect-error Metadata factories cannot omit their registered ID.
        const missing: ModelOptionsInfo = { options: [] };
        expect(missing._option_id).toBeUndefined();
        // This must fail compilation even if no model fixture exercises the new factory branch.
        // @ts-expect-error An unregistered option ID cannot be returned by a typed factory.
        const unregistered: ModelOptionsInfo['_option_id'] = 'unregistered-provider';
        expect(ModelOptionsSchema.safeParse({ _option_id: unregistered }).success).toBe(false);
    });

    it('validates Anthropic factory options and rejects invalid fields', () => {
        const { _option_id } = getOptions('claude-sonnet-4-6', Providers.anthropic);
        expect(_option_id).toBe('anthropic-claude');
        const options = {
            _option_id,
            max_tokens: 4096,
            temperature: 0.5,
            top_p: 0.9,
            top_k: 10,
            stop_sequence: ['STOP'],
            effort: 'high',
            thinking_budget_tokens: 1024,
            include_thoughts: true,
            cache_enabled: true,
            cache_ttl: '1h',
        };
        expect(ModelOptionsSchema.parse(options)).toEqual(options);
        for (const invalid of [{ effort: 'none' }, { cache_ttl: '2h' }, { max_tokens: '4096' }, { unknown: true }]) {
            expect(ModelOptionsSchema.safeParse({ ...options, ...invalid }).success).toBe(false);
        }
    });

    it('publishes optional unique IDs and anyOf for overlapping untagged objects', () => {
        expect(emitted.type).toBe('object');
        expect(emitted.oneOf).toBeUndefined();
        expect(emitted.anyOf).toBeDefined();
        const ids = MEMBERS.map((name) => {
            const member = emitted.$defs[name];
            expect(member.required ?? [], name).not.toContain('_option_id');
            const hint = member.properties?._option_id as { const?: string } | undefined;
            expect(hint?.const, name).toBeTypeOf('string');
            return hint?.const;
        });
        expect(new Set(ids).size).toBe(ids.length);
    });

    it.each([
        {},
        { temperature: 0.2, max_tokens: 1024 },
        { cache_enabled: true, cache_ttl: '1h', thinking_budget_tokens: 1024 },
        { extra_body: { provider_extension: true } },
    ] satisfies ModelOptions[])('accepts untagged options without adding a family: %j', (options) => {
        expect(ModelOptionsSchema.parse(options)).toEqual(options);
        expect(ModelOptionsSchema.parse(options)).not.toHaveProperty('_option_id');
    });

    it.each([
        { _option_id: null },
        { _option_id: 12 },
        { _option_id: 'unknown' },
        { max_tokens: '1024' },
        { cache_ttl: '2h' },
        { unknown_option: true },
        { _option_id: 'openai-text', cache_ttl: '1h' },
    ])('still rejects invalid fields and supplied IDs: %j', (options) => {
        expect(ModelOptionsSchema.safeParse(options).success).toBe(false);
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

    it('accepts provider-specific objects for OpenAI-compatible option schemas', () => {
        expect(
            ModelOptionsSchema.safeParse({
                _option_id: 'openai-text',
                extra_body: {
                    provider: { sort: 'throughput', allow_fallbacks: false },
                    baseten: { performance: 'max' },
                },
            }).success,
        ).toBe(true);
        expect(
            ModelOptionsSchema.safeParse({
                _option_id: 'openai-thinking',
                extra_body: { provider_extension: { enabled: true } },
            }).success,
        ).toBe(true);
        expect(ModelOptionsSchema.safeParse({ _option_id: 'openai-text', extra_body: [] }).success).toBe(false);
    });
});
