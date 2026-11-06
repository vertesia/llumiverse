import type { Ajv, ValidateFunction } from 'ajv';
import { LRUCache } from 'lru-cache';

export interface SchemaValidatorCacheOptions {
    /** Most validators kept at once. */
    maxEntries: number;
    /**
     * Most schema JSON characters kept at once, summed over the cached schemas. A compiled
     * validator retains tens of times more memory than its schema text, so the cache is bounded by
     * schema size, not only by count. A larger schema is still cached, alone.
     */
    maxSchemaChars: number;
}

/**
 * Compiled Ajv validators keyed by the full content of their schema.
 *
 * Ajv caches compiled validators by object identity, but a stored result schema is deserialized into
 * a fresh object on every execution, so it compiled the schema again on every call (milliseconds to
 * over 100 ms for large schemas, blocking the event loop). The key is the schema's JSON text, so an
 * edited schema always misses. It keeps key order on purpose: Ajv generates code in property
 * order, which decides the order of errors and of default/coercion side effects, so only identical
 * text is guaranteed to compile to identical behavior. The schema compiled is parsed from that same
 * text, never the caller's object: compiled validators read some values (`const`, long `enum`s) from
 * their schema object when they run, so a caller editing its own object later must not change what a
 * cached validator enforces.
 *
 * Every schema is compiled in its own Ajv instance from `createAjv`. Ajv keeps every function it has
 * compiled in the instance's code scope for the life of the instance (`removeSchema` releases only
 * its lookup tables), so a shared instance grows without bound however its schemas are managed. With
 * one instance per entry, evicting the least recently used entry frees all of it. It also gives
 * every schema its own `$id` registry, so a schema edited under the same `$id` compiles without
 * `schema with key or id "..." already exists` and without affecting the cached previous version.
 *
 * A schema larger than `maxSchemaChars` counts as the whole budget: it evicts every other entry and
 * is cached on its own, since the largest schemas are the slowest to compile.
 */
export class SchemaValidatorCache {
    private readonly validators: LRUCache<string, ValidateFunction>;

    constructor(
        private readonly createAjv: () => Ajv,
        { maxEntries, maxSchemaChars }: SchemaValidatorCacheOptions,
    ) {
        this.validators = new LRUCache<string, ValidateFunction>({
            max: maxEntries,
            maxSize: maxSchemaChars,
            sizeCalculation: (_validate, key) => Math.min(key.length, maxSchemaChars),
        });
    }

    get(schema: object): ValidateFunction {
        const key = JSON.stringify(schema);
        let validate = this.validators.get(key);
        if (!validate) {
            validate = this.createAjv().compile(JSON.parse(key));
            this.validators.set(key, validate);
        }
        return validate;
    }

    get size(): number {
        return this.validators.size;
    }
}
