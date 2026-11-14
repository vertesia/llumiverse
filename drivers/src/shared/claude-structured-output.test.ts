import { JSON_SCHEMA_INSTRUCTION_PREFIX } from '@llumiverse/common';
import { type ExecutionOptions, type JSONSchema, PromptRole, type PromptSegment } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { BedrockDriver } from '../bedrock/index.js';
import type { VertexAIDriver } from '../vertexai/index.js';
import { ClaudeModelDefinition } from '../vertexai/models/claude.js';
import { formatClaudePrompt, getClaudePayload } from './claude-messages.js';
import {
    ClaudeStructuredOutputDenials,
    endsWithClaudePrefill,
    isClaudeStructuredOutputPolicyDenial,
    isClaudeStructuredOutputSchema,
    supportsClaudeNativeResultSchema,
    usesClaudeNativeResultSchema,
} from './claude-structured-output.js';

const schema = (extra: Record<string, unknown> = {}): JSONSchema =>
    ({
        type: 'object',
        properties: {
            name: { type: 'string' },
            tags: { type: 'array', items: { type: 'string', enum: ['a', 'b'] }, minItems: 1 },
            address: {
                type: 'object',
                properties: { city: { type: 'string' } },
                required: ['city'],
                additionalProperties: false,
            },
        },
        required: ['name', 'tags', 'address'],
        additionalProperties: false,
        ...extra,
    }) as JSONSchema;

const segments: PromptSegment[] = [{ role: PromptRole.user, content: 'Describe the object.' }];
const MODEL = 'us.anthropic.claude-sonnet-4-5-20250929-v1:0';

describe('supportsClaudeNativeResultSchema', () => {
    it.each([
        ['us.anthropic.claude-sonnet-4-5-20250929-v1:0', true],
        ['anthropic.claude-haiku-4-5-20251001-v1:0', true],
        ['global.anthropic.claude-opus-4-6-v1', true],
        ['us.anthropic.claude-fable-5-20260115-v1:0', true],
        ['anthropic.claude-sonnet-4-20250514-v1:0', false],
        ['anthropic.claude-3-7-sonnet-20250219-v1:0', false],
        ['anthropic.claude-3-5-sonnet-20241022-v2:0', false],
    ])('%s → %s', (model, expected) => {
        expect(supportsClaudeNativeResultSchema(model)).toBe(expected);
    });
});

describe('isClaudeStructuredOutputSchema', () => {
    it('accepts a schema within the Bedrock subset', () => {
        expect(isClaudeStructuredOutputSchema(schema())).toBe(true);
    });

    it('accepts internal non-recursive $refs', () => {
        expect(
            isClaudeStructuredOutputSchema(
                schema({
                    properties: { home: { $ref: '#/$defs/Place' } },
                    required: ['home'],
                    $defs: {
                        Place: {
                            type: 'object',
                            properties: { city: { type: 'string' } },
                            required: ['city'],
                            additionalProperties: false,
                        },
                    },
                }),
            ),
        ).toBe(true);
    });

    it.each<[string, Record<string, unknown>]>([
        ['a non-object root', { type: 'array' }],
        ['a root without additionalProperties: false', { additionalProperties: undefined }],
        [
            'a nested object without additionalProperties: false',
            { properties: { a: { type: 'object', properties: { b: { type: 'string' } } } } },
        ],
        ['a numeric bound', { properties: { n: { type: 'number', minimum: 0 } } }],
        ['a string length bound', { properties: { s: { type: 'string', maxLength: 10 } } }],
        ['minItems above 1', { properties: { l: { type: 'array', items: { type: 'string' }, minItems: 2 } } }],
        ['a non-primitive enum value', { properties: { e: { enum: [{ a: 1 }] } } }],
        ['oneOf', { properties: { o: { oneOf: [{ type: 'string' }, { type: 'number' }] } } }],
        ['maxItems', { properties: { l: { type: 'array', items: { type: 'string' }, maxItems: 3 } } }],
        ['uniqueItems', { properties: { l: { type: 'array', items: { type: 'string' }, uniqueItems: true } } }],
        ['contains', { properties: { l: { type: 'array', items: { type: 'string' }, contains: { const: 'a' } } } }],
        ['maxProperties', { maxProperties: 3 }],
        ['propertyNames', { propertyNames: { pattern: '^[a-z]+$' } }],
        ['a backreference in a pattern', { properties: { p: { type: 'string', pattern: '^(a)\\1$' } } }],
        ['a lookahead in a pattern', { properties: { p: { type: 'string', pattern: '^(?=a).*$' } } }],
        ['a word boundary in a pattern', { properties: { p: { type: 'string', pattern: '\\bword\\b' } } }],
        [
            'an allOf $ref branch',
            {
                properties: { a: { allOf: [{ $ref: '#/$defs/Place' }] } },
                $defs: { Place: { type: 'object', properties: {}, additionalProperties: false } },
            },
        ],
        ['an external $ref', { properties: { r: { $ref: 'https://example.com/schema.json' } } }],
        ['a $ref to the root', { properties: { self: { $ref: '#' } } }],
        ['a $ref to a missing definition', { properties: { r: { $ref: '#/$defs/Missing' } } }],
        [
            'recursive definitions',
            {
                properties: { node: { $ref: '#/$defs/Node' } },
                $defs: {
                    Node: {
                        type: 'object',
                        properties: { child: { $ref: '#/$defs/Node' } },
                        additionalProperties: false,
                    },
                },
            },
        ],
    ])('rejects %s', (_label, extra) => {
        expect(isClaudeStructuredOutputSchema(schema(extra))).toBe(false);
    });

    it.each(['date-time', 'time', 'date', 'duration', 'email', 'hostname', 'uri', 'ipv4', 'ipv6', 'uuid'])(
        'accepts the supported string format %s',
        (format) => {
            expect(isClaudeStructuredOutputSchema(schema({ properties: { v: { type: 'string', format } } }))).toBe(
                true,
            );
        },
    );

    it.each(['uri-reference', 'iri', 'regex', 'json-pointer', 'idn-email'])(
        'rejects the unsupported string format %s',
        (format) => {
            expect(isClaudeStructuredOutputSchema(schema({ properties: { v: { type: 'string', format } } }))).toBe(
                false,
            );
        },
    );

    it('accepts a simple pattern', () => {
        expect(
            isClaudeStructuredOutputSchema(
                schema({ properties: { p: { type: 'string', pattern: '^[A-Z]{2}\\d{3}$' } }, required: ['p'] }),
            ),
        ).toBe(true);
    });

    it('rejects more than 12 optional properties', () => {
        const properties = (count: number) =>
            Object.fromEntries(Array.from({ length: count }, (_, i) => [`p${i}`, { type: 'string' }]));
        expect(isClaudeStructuredOutputSchema(schema({ properties: properties(12), required: [] }))).toBe(true);
        expect(isClaudeStructuredOutputSchema(schema({ properties: properties(13), required: [] }))).toBe(false);
    });

    it('rejects more than 16 union-typed properties', () => {
        const properties = (count: number) =>
            Object.fromEntries(Array.from({ length: count }, (_, i) => [`p${i}`, { type: ['string', 'null'] }]));
        const required = (count: number) => Array.from({ length: count }, (_, i) => `p${i}`);
        expect(isClaudeStructuredOutputSchema(schema({ properties: properties(16), required: required(16) }))).toBe(
            true,
        );
        expect(isClaudeStructuredOutputSchema(schema({ properties: properties(17), required: required(17) }))).toBe(
            false,
        );
    });
});

describe('usesClaudeNativeResultSchema', () => {
    it('requires a schema, no tools, a recent Claude model and a compatible schema', () => {
        expect(usesClaudeNativeResultSchema({ model: MODEL, result_schema: schema() })).toBe(true);
        expect(usesClaudeNativeResultSchema({ model: MODEL })).toBe(false);
        expect(
            usesClaudeNativeResultSchema({
                model: MODEL,
                result_schema: schema(),
                tools: [{ name: 'lookup', input_schema: { type: 'object' } }],
            }),
        ).toBe(false);
        expect(
            usesClaudeNativeResultSchema({ model: 'anthropic.claude-sonnet-4-20250514-v1:0', result_schema: schema() }),
        ).toBe(false);
        expect(usesClaudeNativeResultSchema({ model: MODEL, result_schema: schema({ minLength: 1 }) })).toBe(false);
    });
});

describe('Claude native result schema on Bedrock and Vertex AI', () => {
    const options: ExecutionOptions = { model: MODEL, result_schema: schema() };

    it.each(['bedrock', 'vertexai'])(
        'sends output_config.format on %s and keeps the schema out of the prompt',
        async (provider) => {
            const prompt = await formatClaudePrompt(segments, options, undefined, { nativeResultSchema: true });
            const { payload } = getClaudePayload(options, prompt, provider);

            expect(JSON.stringify(prompt)).not.toContain('JSON Schema');
            expect(payload.output_config).toEqual({ format: { type: 'json_schema', schema: options.result_schema } });
            expect(payload.messages.at(-1)?.role).toBe('user');
        },
    );

    it('keeps the effort setting next to the format', async () => {
        const withEffort: ExecutionOptions = {
            model: 'global.anthropic.claude-opus-4-6-v1',
            result_schema: schema(),
            model_options: { _option_id: 'bedrock-claude', effort: 'low' },
        };
        const prompt = await formatClaudePrompt(segments, withEffort, undefined, { nativeResultSchema: true });
        const { payload } = getClaudePayload(withEffort, prompt, 'bedrock');

        expect(payload.output_config).toMatchObject({
            effort: 'low',
            format: { type: 'json_schema', schema: withEffort.result_schema },
        });
    });

    it.each(['anthropic', 'azure_foundry'])('leaves the %s provider on the prompt path', async (provider) => {
        const prompt = await formatClaudePrompt(segments, options);
        const { payload } = getClaudePayload(options, prompt, provider);

        expect(JSON.stringify(prompt.system)).toContain('JSON Schema');
        expect(payload.output_config?.format).toBeUndefined();
    });

    it('leaves an unsupported schema on the prompt path', async () => {
        const unsupported: ExecutionOptions = {
            model: MODEL,
            result_schema: schema({ maxProperties: 3, minLength: 1 }),
        };
        const prompt = await formatClaudePrompt(segments, unsupported);
        const { payload } = getClaudePayload(unsupported, prompt, 'bedrock');

        expect(JSON.stringify(prompt.system)).toContain('JSON Schema');
        expect(payload.output_config?.format).toBeUndefined();
    });

    it('keeps a closed schema with maxItems on the prompt path, with its instructions', async () => {
        const bounded: ExecutionOptions = {
            model: MODEL,
            result_schema: schema({
                properties: { tags: { type: 'array', items: { type: 'string' }, maxItems: 3 } },
                required: ['tags'],
            }),
        };
        const prompt = await formatClaudePrompt(segments, bounded, undefined, {
            nativeResultSchema: usesClaudeNativeResultSchema(bounded),
        });
        const { payload } = getClaudePayload(bounded, prompt, 'vertexai');

        expect(usesClaudeNativeResultSchema(bounded)).toBe(false);
        expect(JSON.stringify(prompt.system)).toContain('JSON Schema');
        expect(payload.output_config?.format).toBeUndefined();
    });

    it('keeps a closed schema with an unsupported string format on the prompt path, with its instructions', async () => {
        const linked: ExecutionOptions = {
            model: MODEL,
            result_schema: schema({
                properties: { link: { type: 'string', format: 'uri-reference' } },
                required: ['link'],
            }),
        };
        const prompt = await formatClaudePrompt(segments, linked, undefined, {
            nativeResultSchema: usesClaudeNativeResultSchema(linked),
        });
        const { payload } = getClaudePayload(linked, prompt, 'vertexai');

        expect(usesClaudeNativeResultSchema(linked)).toBe(false);
        expect(JSON.stringify(prompt.system)).toContain('JSON Schema');
        expect(payload.output_config?.format).toBeUndefined();
    });

    it.each(['bedrock', 'vertexai'])(
        'keeps a request that ends with an assistant prefill on the prompt path on %s',
        async (provider) => {
            const prefilled: PromptSegment[] = [...segments, { role: PromptRole.assistant, content: '{"name":' }];
            // Claude 4.5 continues a trailing assistant turn as a prefill; 4.6+ gets a continuation turn instead.
            const haiku: ExecutionOptions = { model: 'claude-haiku-4-5', result_schema: schema() };
            const prompt = await formatClaudePrompt(prefilled, haiku, undefined, {
                nativeResultSchema: usesClaudeNativeResultSchema(haiku),
            });
            const { payload } = getClaudePayload(haiku, prompt, provider);

            expect(payload.messages.at(-1)?.role).toBe('assistant');
            expect(payload.output_config?.format).toBeUndefined();
            expect(JSON.stringify(payload.system)).toContain('JSON Schema');
        },
    );

    it('adds the instructions only once when the prompt already carries them', async () => {
        const prefilled: PromptSegment[] = [...segments, { role: PromptRole.assistant, content: '{' }];
        const haiku: ExecutionOptions = { model: 'claude-haiku-4-5', result_schema: schema() };
        const prompt = await formatClaudePrompt(prefilled, haiku);
        const { payload } = getClaudePayload(haiku, prompt, 'bedrock');

        expect(JSON.stringify(payload.system).split('JSON Schema').length - 1).toBe(
            JSON.stringify(prompt.system).split('JSON Schema').length - 1,
        );
    });

    it("adds the current schema's instructions when an earlier turn carries another schema's", async () => {
        const previous = schema({ properties: { title: { type: 'string' } }, required: ['title'] });
        const current: ExecutionOptions = { model: MODEL, result_schema: schema() };
        const prompt = await formatClaudePrompt(segments, current, undefined, { nativeResultSchema: true });
        // A continuation: an earlier turn asked for the previous schema, the current request for another one.
        prompt.messages = [
            {
                role: 'user',
                content: [{ type: 'text', text: `${JSON_SCHEMA_INSTRUCTION_PREFIX}\n${JSON.stringify(previous)}` }],
            },
            { role: 'assistant', content: [{ type: 'text', text: '{"title":"An earlier answer"}' }] },
            ...prompt.messages,
        ];
        const { payload } = getClaudePayload(current, prompt, 'vertexai', 'execute', undefined, {
            allowNativeResultSchema: false,
        });

        expect(payload.output_config?.format).toBeUndefined();
        const systemText = (payload.system as { text: string }[]).map((block) => block.text).join('\n');
        expect(systemText).toContain(`${JSON_SCHEMA_INSTRUCTION_PREFIX}\n${JSON.stringify(current.result_schema)}`);
    });

    it('sends the schema as instructions when the native format is not allowed', async () => {
        const prompt = await formatClaudePrompt(segments, options, undefined, { nativeResultSchema: true });
        const { payload } = getClaudePayload(options, prompt, 'vertexai', 'execute', undefined, {
            allowNativeResultSchema: false,
        });

        expect(payload.output_config?.format).toBeUndefined();
        expect(JSON.stringify(payload.system)).toContain('JSON Schema');
    });

    it('makes the same decision in the Bedrock prompt as in the payload', async () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const native = await driver.createPrompt(segments, options);
        const legacy = await driver.createPrompt(segments, {
            model: 'us.anthropic.claude-sonnet-4-20250514-v1:0',
            result_schema: schema(),
        });

        expect(JSON.stringify(native)).not.toContain('JSON Schema');
        expect(JSON.stringify(legacy)).toContain('JSON Schema');
    });

    it('makes the same decision in the Vertex AI prompt as in the payload', async () => {
        const definition = new ClaudeModelDefinition('claude-haiku-4-5');
        const driver = {
            logger: undefined,
            options: { project: 'p', region: 'us-central1' },
        } as unknown as VertexAIDriver;
        const native = await definition.createPrompt(driver, segments, {
            model: 'locations/global/publishers/anthropic/models/claude-haiku-4-5',
            result_schema: schema(),
        });
        const legacy = await definition.createPrompt(driver, segments, {
            model: 'locations/us-east5/publishers/anthropic/models/claude-sonnet-4',
            result_schema: schema(),
        });
        const unsupported = await definition.createPrompt(driver, segments, {
            model: 'locations/global/publishers/anthropic/models/claude-haiku-4-5',
            result_schema: schema({ additionalProperties: true }),
        });

        expect(JSON.stringify(native)).not.toContain('JSON Schema');
        expect(JSON.stringify(legacy)).toContain('JSON Schema');
        expect(JSON.stringify(unsupported)).toContain('JSON Schema');
    });
});

describe('endsWithClaudePrefill', () => {
    it('is true only when the last non-system turn is the assistant', () => {
        const user = { role: 'user' as const, content: 'hi' };
        const assistant = { role: 'assistant' as const, content: '{' };
        expect(endsWithClaudePrefill([user, assistant])).toBe(true);
        expect(endsWithClaudePrefill([assistant, user])).toBe(false);
        expect(endsWithClaudePrefill([])).toBe(false);
    });
});

describe('isClaudeStructuredOutputPolicyDenial', () => {
    const denial = (status: number, message: string) => ({ status, message });

    it('matches the 400 FAILED_PRECONDITION policy refusal of structured_outputs', () => {
        expect(
            isClaudeStructuredOutputPolicyDenial(
                denial(
                    400,
                    '400 [{"error":{"code":400,"message":"Organization Policy constraint constraints/vertexai.allowedPartnerModelFeatures violated for a disallowed feature structured_outputs","status":"FAILED_PRECONDITION"}}]',
                ),
            ),
        ).toBe(true);
    });

    it.each([
        ['another 400', denial(400, 'max_tokens: too large')],
        [
            'a policy refusal of another feature',
            denial(400, 'FAILED_PRECONDITION allowedPartnerModelFeatures web_search'),
        ],
        ['another status', denial(403, 'FAILED_PRECONDITION allowedPartnerModelFeatures structured_outputs')],
        ['a non-object', 'FAILED_PRECONDITION allowedPartnerModelFeatures structured_outputs'],
    ])('does not match %s', (_label, error) => {
        expect(isClaudeStructuredOutputPolicyDenial(error)).toBe(false);
    });
});

describe('ClaudeStructuredOutputDenials', () => {
    it('remembers a bounded number of denials, dropping the oldest', () => {
        const denials = new ClaudeStructuredOutputDenials();
        const first = ClaudeStructuredOutputDenials.key('p', 'global', 'm0');
        denials.add(first);
        for (let i = 1; i <= 256; i++) denials.add(ClaudeStructuredOutputDenials.key('p', 'global', `m${i}`));

        expect(denials.has(first)).toBe(false);
        expect(denials.has(ClaudeStructuredOutputDenials.key('p', 'global', 'm256'))).toBe(true);
    });
});
