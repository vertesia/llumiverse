import { setFlagsFromString } from 'node:v8';
import { runInNewContext } from 'node:vm';
import { Ajv } from 'ajv';
import { describe, expect, it, vi } from 'vitest';
import { SchemaValidatorCache } from './schema-validator-cache.js';

setFlagsFromString('--expose-gc');
const collectGarbage = runInNewContext('gc') as () => void;

function newAjv() {
    return new Ajv({ strict: false, useDefaults: true, coerceTypes: 'array', removeAdditional: 'failing' });
}

// Each call returns a new object, as a stored schema deserialized for every execution would be.
function schemaRequiring(field: string) {
    return { type: 'object', properties: { [field]: { type: 'string' } }, required: [field] };
}

function trackingCache(options = { maxEntries: 10, maxSchemaChars: 1_000_000 }) {
    const instances: Ajv[] = [];
    const createAjv = vi.fn(() => {
        const ajv = newAjv();
        instances.push(ajv);
        return ajv;
    });
    return { cache: new SchemaValidatorCache(createAjv, options), createAjv, instances };
}

describe('SchemaValidatorCache', () => {
    it('reuses one compiled validator for fresh objects with identical content', () => {
        const { cache, createAjv } = trackingCache();

        const first = cache.get(schemaRequiring('a'));
        const second = cache.get(schemaRequiring('a'));

        expect(second).toBe(first);
        expect(createAjv).toHaveBeenCalledTimes(1);
    });

    it('misses when the schema content changes', () => {
        const { cache } = trackingCache();
        const original = cache.get(schemaRequiring('a'));
        const edited = cache.get(schemaRequiring('b'));

        expect(edited).not.toBe(original);
        expect(edited({ a: 'x' })).toBe(false);
        expect(original({ a: 'x' })).toBe(true);
        // An edit deep inside the schema misses as well.
        const nested = cache.get({ ...schemaRequiring('a'), properties: { a: { type: 'number' } } });
        expect(nested).not.toBe(original);
        expect(nested({ a: 'x' })).toBe(false);
    });

    it('evicts the least recently used validator', () => {
        const { cache, createAjv } = trackingCache({ maxEntries: 2, maxSchemaChars: 1_000_000 });

        const validateA = cache.get(schemaRequiring('a'));
        const validateB = cache.get(schemaRequiring('b'));
        expect(cache.get(schemaRequiring('a'))).toBe(validateA); // A is now the most recently used
        cache.get(schemaRequiring('c')); // evicts B

        expect(cache.size).toBe(2);
        expect(cache.get(schemaRequiring('a'))).toBe(validateA);
        expect(createAjv).toHaveBeenCalledTimes(3);
        const recompiledB = cache.get(schemaRequiring('b'));
        expect(recompiledB).not.toBe(validateB);
        expect(createAjv).toHaveBeenCalledTimes(4);
    });

    it('evicts by total schema size', () => {
        const schemaChars = JSON.stringify(schemaRequiring('a')).length;
        const { cache } = trackingCache({ maxEntries: 100, maxSchemaChars: schemaChars * 3 });

        for (const field of ['a', 'b', 'c', 'd', 'e']) {
            cache.get(schemaRequiring(field));
        }

        expect(cache.size).toBe(3);
    });

    it('compiles a schema larger than the size budget without caching it', () => {
        const { cache, createAjv } = trackingCache({ maxEntries: 100, maxSchemaChars: 10 });

        expect(cache.get(schemaRequiring('a'))({ a: 'x' })).toBe(true);
        expect(cache.get(schemaRequiring('a'))({})).toBe(false);
        expect(cache.size).toBe(0);
        expect(createAjv).toHaveBeenCalledTimes(2);
    });

    // An Ajv instance keeps everything it compiled reachable from each of its validators, so memory
    // is only released when an evicted validator and its instance become unreachable together.
    it('lets an evicted validator be garbage collected', async () => {
        const cache = new SchemaValidatorCache(newAjv, { maxEntries: 1, maxSchemaChars: 1_000_000 });
        const evicted = new WeakRef(cache.get(schemaRequiring('a')));
        const cached = new WeakRef(cache.get(schemaRequiring('b')));

        await new Promise((resolve) => setTimeout(resolve, 0));
        collectGarbage();

        expect(evicted.deref()).toBeUndefined();
        expect(cached.deref()).toBe(cache.get(schemaRequiring('b')));
    });

    it('compiles each schema in its own Ajv instance', () => {
        const { cache, instances } = trackingCache({ maxEntries: 5, maxSchemaChars: 1_000_000 });

        for (let i = 0; i < 50; i++) {
            const validate = cache.get(schemaRequiring(`field_${i}`));
            expect(validate({ [`field_${i}`]: 'x' })).toBe(true);
        }

        expect(cache.size).toBe(5);
        expect(instances).toHaveLength(50);
        expect(new Set(instances).size).toBe(50);
    });

    it('compiles a schema edited under the same $id while keeping the previous version usable', () => {
        const { cache } = trackingCache();
        const id = 'https://example.test/schemas/edited.json';

        const original = cache.get({ $id: id, ...schemaRequiring('a') });
        const edited = cache.get({ $id: id, ...schemaRequiring('b') });

        expect(edited).not.toBe(original);
        expect(edited({ a: 'x' })).toBe(false);
        expect(original({ a: 'x' })).toBe(true);
        expect(cache.get({ $id: id, ...schemaRequiring('a') })).toBe(original);
    });
});
