import { Ajv2020 } from 'ajv/dist/2020.js';
import { describe, expect, it } from 'vitest';
import { Base64Schema } from '../src/schemas/primitives.js';

describe('Base64Schema', () => {
    it('validates realistic multi-megabyte inline media without overflowing the regexp stack', () => {
        const encoded = 'QUJD'.repeat(2_000_000);
        expect(encoded).toHaveLength(8_000_000);
        expect(Base64Schema.parse(encoded)).toBe(encoded);
    });

    it.each(['A', 'AAA==', 'AA=A', 'AA-_'])('rejects malformed base64 %s', (encoded) => {
        expect(Base64Schema.safeParse(encoded).success).toBe(false);
    });

    it('keeps AJV and Zod validation behavior identical', () => {
        const validate = new Ajv2020({ strict: true }).compile(
            Base64Schema.toJSONSchema({ target: 'draft-2020-12', io: 'input' }),
        );
        for (const encoded of ['', 'TQ==', 'TWE=', 'QUJD', 'A', 'AAA==', 'AA=A', 'AA-_']) {
            expect(validate(encoded), encoded).toBe(Base64Schema.safeParse(encoded).success);
        }
        const large = 'QUJD'.repeat(2_000_000);
        expect(validate(large)).toBe(true);
        expect(Base64Schema.safeParse(large).success).toBe(true);
    });
});
