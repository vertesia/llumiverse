import { type ExecutionOptions, type JSONSchema, PromptRole, type PromptSegment } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { BedrockDriver } from '../bedrock/index.js';
import { formatClaudePrompt, getClaudePayload } from './claude-messages.js';
import {
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

describe('Claude native result schema on Bedrock', () => {
    const options: ExecutionOptions = { model: MODEL, result_schema: schema() };

    it('sends output_config.format and keeps the schema out of the prompt', async () => {
        const prompt = await formatClaudePrompt(segments, options, undefined, { nativeResultSchema: true });
        const { payload } = getClaudePayload(options, prompt, 'bedrock');

        expect(JSON.stringify(prompt)).not.toContain('JSON Schema');
        expect(payload.output_config).toEqual({ format: { type: 'json_schema', schema: options.result_schema } });
        expect(payload.messages.at(-1)?.role).toBe('user');
    });

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

    it.each(['anthropic', 'vertexai', 'azure_foundry'])(
        'leaves the %s provider on the prompt path',
        async (provider) => {
            const prompt = await formatClaudePrompt(segments, options);
            const { payload } = getClaudePayload(options, prompt, provider);

            expect(JSON.stringify(prompt.system)).toContain('JSON Schema');
            expect(payload.output_config?.format).toBeUndefined();
        },
    );

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
});
