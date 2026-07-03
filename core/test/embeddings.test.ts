import type { EmbeddingsResult } from '@llumiverse/common';
import { describe, expect, test } from 'vitest';
import { firstVector, normalizeEmbeddingsOptions } from '../src/embeddings.js';

describe('normalizeEmbeddingsOptions', () => {
    test('throws on empty inputs', () => {
        expect(() => normalizeEmbeddingsOptions({ inputs: [] })).toThrow(
            'EmbeddingsOptions.inputs must contain at least one input',
        );
    });

    test('propagates top-level task_type to text inputs without their own', () => {
        const result = normalizeEmbeddingsOptions({
            inputs: [{ type: 'text', text: 'hello' }],
            task_type: 'query',
        });
        const input = result.inputs[0];
        expect(input.type).toBe('text');
        if (input.type === 'text') {
            expect(input.task_type).toBe('query');
        }
    });

    test('does not override a per-input task_type', () => {
        const result = normalizeEmbeddingsOptions({
            inputs: [{ type: 'text', text: 'hello', task_type: 'document' }],
            task_type: 'query',
        });
        const input = result.inputs[0];
        if (input.type === 'text') {
            expect(input.task_type).toBe('document');
        }
    });

    test('preserves non-text inputs unchanged', () => {
        const ds = { name: 'img.jpg', mime_type: 'image/jpeg' } as never;
        const result = normalizeEmbeddingsOptions({
            inputs: [{ type: 'image', source: ds }],
            task_type: 'query',
        });
        expect(result.inputs[0].type).toBe('image');
    });

    test('normalizes multiple inputs correctly', () => {
        const result = normalizeEmbeddingsOptions({
            inputs: [
                { type: 'text', text: 'a' },
                { type: 'text', text: 'b', task_type: 'document' },
            ],
            task_type: 'query',
        });
        const [a, b] = result.inputs;
        if (a.type === 'text') expect(a.task_type).toBe('query');
        if (b.type === 'text') expect(b.task_type).toBe('document');
    });

    test('passes through model and dimensions unchanged', () => {
        const result = normalizeEmbeddingsOptions({
            inputs: [{ type: 'text', text: 'x' }],
            model: 'my-model',
            dimensions: 512,
        });
        expect(result.model).toBe('my-model');
        expect(result.dimensions).toBe(512);
    });
});

describe('firstVector', () => {
    test('returns first vector from a valid result', () => {
        const result: EmbeddingsResult = {
            model: 'm',
            results: [{ outputs: [{ values: [1, 2, 3] }] }],
        };
        expect(firstVector(result)).toEqual([1, 2, 3]);
    });

    test('throws if no results', () => {
        const result: EmbeddingsResult = { model: 'm', results: [] };
        expect(() => firstVector(result)).toThrow('no vectors');
    });

    test('throws if no outputs', () => {
        const result: EmbeddingsResult = { model: 'm', results: [{ outputs: [] }] };
        expect(() => firstVector(result)).toThrow('no vectors');
    });
});
