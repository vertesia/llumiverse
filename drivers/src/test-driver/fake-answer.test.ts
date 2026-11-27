import type { JSONSchema, PromptSegment } from '@llumiverse/core';
import { PromptRole } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { createFakeAnswer, FAKE_ANSWER_TEXT, fakeValue } from './fake-answer.js';
import { TestDriver, TestDriverModels } from './index.js';

const segments: PromptSegment[] = [
    { role: PromptRole.system, content: 'Return the values of the slice that apply to the document.' },
    { role: PromptRole.user, content: 'Document Common Kingfisher: a small bird with a blue back.' },
];

// The result schema of the interaction of the load test of Penna (vertesia/experimental, penna/load).
const pickValuesSchema: JSONSchema = {
    type: 'object',
    properties: {
        values: { type: 'array', items: { type: 'string' } },
        text: { type: 'string' },
    },
    required: ['values'],
};

describe('fakeValue', () => {
    it('keeps only the required properties of an object', () => {
        expect(fakeValue(pickValuesSchema)).toEqual({ values: [] });
    });

    it('takes the first value of a const, a default or an enum', () => {
        expect(fakeValue({ const: 'fixed' })).toBe('fixed');
        expect(fakeValue({ type: 'string', default: 'kept' })).toBe('kept');
        expect(fakeValue({ type: 'string', enum: ['common', 'rare'] })).toBe('common');
    });

    it('respects the bounds of strings, numbers and arrays', () => {
        expect(fakeValue({ type: 'string', minLength: 3 })).toBe('xxx');
        expect(fakeValue({ type: 'string', format: 'date-time' })).toBe('1970-01-01T00:00:00.000Z');
        expect(fakeValue({ type: 'number', minimum: 0.5 })).toBe(0.5);
        expect(fakeValue({ type: 'integer', exclusiveMinimum: 0 })).toBe(1);
        expect(fakeValue({ type: 'integer', minimum: 10, maximum: 5 })).toBe(5);
        expect(fakeValue({ type: 'array', items: { type: 'boolean' }, minItems: 2 })).toEqual([false, false]);
    });

    it('follows the first branch of anyOf and oneOf, and merges the objects of allOf', () => {
        expect(fakeValue({ anyOf: [{ type: 'integer' }, { type: 'string' }] })).toBe(0);
        expect(fakeValue({ oneOf: [{ type: 'boolean' }] })).toBe(false);
        expect(
            fakeValue({
                allOf: [
                    { type: 'object', properties: { a: { type: 'string' } }, required: ['a'] },
                    { type: 'object', properties: { b: { type: 'integer' } }, required: ['b'] },
                ],
            }),
        ).toEqual({ a: '', b: 0 });
    });

    it('takes the first type other than null, and infers a missing type', () => {
        expect(fakeValue({ type: ['null', 'string'] })).toBe('');
        expect(fakeValue({ properties: { a: { type: 'boolean' } }, required: ['a'] })).toEqual({ a: false });
        expect(fakeValue({ items: { type: 'string' }, minItems: 1 })).toEqual(['']);
        expect(fakeValue({})).toBeNull();
    });

    it('stops a recursion that never ends', () => {
        const node: JSONSchema = { type: 'object', required: ['child'], properties: {} };
        node.properties = { child: node };
        expect(JSON.stringify(fakeValue(node))).toContain('"child":null');
    });
});

describe('createFakeAnswer', () => {
    it('answers in JSON with a schema, and in text without one', () => {
        expect(createFakeAnswer(segments, pickValuesSchema).result).toEqual([{ type: 'json', value: { values: [] } }]);
        expect(createFakeAnswer(segments, undefined).result).toEqual([{ type: 'text', value: FAKE_ANSWER_TEXT }]);
    });

    it('estimates the tokens from the size of the prompt and of the answer', () => {
        const usage = createFakeAnswer(segments, pickValuesSchema).token_usage;
        expect(usage?.prompt).toBeGreaterThan(20);
        expect(usage?.result).toBeGreaterThan(0);
        expect(usage?.total).toBe((usage?.prompt ?? 0) + (usage?.result ?? 0));
    });
});

describe('TestDriver with a fake model', () => {
    const driver = new TestDriver();

    it('lists the fake models', async () => {
        const ids = (await driver.listModels()).map((model) => model.id);
        expect(ids).toEqual(
            expect.arrayContaining([
                TestDriverModels.fakeAnswer,
                TestDriverModels.fakeAnswerInstant,
                TestDriverModels.fakeAnswerSlow,
            ]),
        );
    });

    it('executes through the validation of the result, which passes', async () => {
        const response = await driver.execute(segments, {
            model: TestDriverModels.fakeAnswerInstant,
            result_schema: pickValuesSchema,
        });
        expect(response.error).toBeUndefined();
        expect(response.result).toEqual([{ type: 'json', value: { values: [] } }]);
        expect(response.finish_reason).toBe('stop');
        expect(response.execution_time).toBeGreaterThanOrEqual(0);
    });

    it('streams the whole answer as one chunk', async () => {
        const stream = await driver.stream(segments, { model: TestDriverModels.fakeAnswerInstant });
        const chunks: string[] = [];
        for await (const chunk of stream) {
            chunks.push(chunk);
        }
        expect(chunks.join('')).toBe(FAKE_ANSWER_TEXT);
        expect(stream.completion?.error).toBeUndefined();
    });

    it('waits before it answers, and stops waiting when the signal aborts', async () => {
        const controller = new AbortController();
        const start = Date.now();
        const execution = driver.execute(segments, { model: TestDriverModels.fakeAnswerSlow }, controller.signal);
        setTimeout(() => controller.abort(new Error('cancelled')), 50);
        await expect(execution).rejects.toThrow();
        expect(Date.now() - start).toBeLessThan(5_000);
    });

    it('still rejects an unknown model', async () => {
        await expect(driver.execute(segments, { model: 'unknown-model' })).rejects.toThrow('Unknown model');
    });
});
