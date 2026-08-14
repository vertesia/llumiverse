import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import type { HttpTimeoutOptions } from '../types.js';
import { HttpTimeoutOptionsSchema } from './http-timeout.js';

/** Exact type identity — `extends` in both directions is too weak (`any`/`unknown` slip through). */
type Equals<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
function assertType<T extends true>(_ok: T): void {}

/**
 * `HttpTimeoutOptions` is published by Vertesia as a component of `InteractionExecutionConfiguration`,
 * and it converts here rather than there because this is where the type is declared. The emission is
 * asserted rather than described: the component was derived from this interface's TSDoc for several
 * releases, and the conversion is only sound if the schema reproduces it exactly — including the run
 * of spaces where the scanner collapsed the defaults list onto one line.
 */
const emitted = z.toJSONSchema(HttpTimeoutOptionsSchema, { target: 'draft-2020-12', io: 'input' }) as Record<
    string,
    unknown
>;

describe('HttpTimeoutOptionsSchema', () => {
    it('is the only definition — the public type is inferred from it', () => {
        // Vacuous as an equality, and that is the point: it fails to COMPILE if the interface is ever
        // restated beside the schema, which is the regression this guards.
        assertType<Equals<HttpTimeoutOptions, z.infer<typeof HttpTimeoutOptionsSchema>>>(true);
        assertType<Equals<HttpTimeoutOptions['bodyTimeout'], number | undefined>>(true);
    });

    it('emits the four optional millisecond fields and nothing else', () => {
        const properties = emitted.properties as Record<string, { type: string; description: string }>;
        expect(Object.keys(properties)).toEqual([
            'headersTimeout',
            'bodyTimeout',
            'connectTimeout',
            'keepAliveTimeout',
        ]);
        for (const property of Object.values(properties)) {
            expect(property.type).toBe('number');
            expect(property.description).toBeTruthy();
        }
        // No `required`: every field is optional, which is what the published component says.
        expect(emitted.required).toBeUndefined();
    });

    it('rejects an unknown timeout instead of discarding it', () => {
        // `strictObject`, not `object`. The published component has always said
        // `additionalProperties: false`; plain `z.object` would have PARSED this successfully by
        // dropping the key, so the document would promise a rejection the schema never performed.
        expect(emitted.additionalProperties).toBe(false);
        expect(HttpTimeoutOptionsSchema.safeParse({ bodyTimeout: 90_000 }).success).toBe(true);
        expect(HttpTimeoutOptionsSchema.safeParse({ socketTimeout: 90_000 }).success).toBe(false);
    });

    it('documents the runtime timeout defaults', () => {
        const description = emitted.description as string;
        expect(description.startsWith("HTTP timeouts applied to a driver's upstream LLM-provider calls.\n\n")).toBe(
            true,
        );
        expect(description).toContain(
            'the defaults applied in `@llumiverse/core/createDriverHttpAgent` are:   - headersTimeout:   900_000   ' +
                '- bodyTimeout:      900_000   - connectTimeout:   60_000   - keepAliveTimeout: 300_000',
        );
        expect(description.endsWith('driver timeouts are bounded-resource safety nets.')).toBe(true);
    });
});
