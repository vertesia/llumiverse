import { describe, expect, it } from 'vitest';
import { inspectGenerationUsageAccounting } from '../src/index.js';
import type { GenerationUsage } from '../src/types.js';

const reported = { method: 'reported', accounting_basis: 'provider' } as const;
const partition: GenerationUsage = {
    input_tokens: 110,
    input_new_tokens: 25,
    cache_read_tokens: 75,
    cache_write_tokens: 10,
    output_tokens: 5,
    reasoning_tokens: 2,
    total_tokens: 115,
    input_partition: { type: 'complete_disjoint', cache_write_bucket: 'included' },
    accounting_provenance: {
        input_tokens: reported,
        input_new_tokens: reported,
        cache_read_tokens: reported,
        cache_write_tokens: reported,
        output_tokens: reported,
        reasoning_tokens: reported,
        total_tokens: reported,
    },
};

describe('generation usage accounting inspection', () => {
    it('preserves exact disjoint accounting facts without adding reasoning twice or normalizing reporting payloads', () => {
        const source = { ...partition, reported_usage: [{ source: 'provider', payload: { arbitrary: 'opaque' } }] };
        expect(inspectGenerationUsageAccounting(source)).toEqual({ status: 'valid', usage: partition });
        expect(source.reported_usage).toEqual([{ source: 'provider', payload: { arbitrary: 'opaque' } }]);
    });
    it('keeps truthful partial output and known cache fields without inventing input or absent cache buckets', () => {
        const usage: GenerationUsage = {
            output_tokens: 5,
            cache_read_tokens: 75,
            accounting_provenance: { output_tokens: reported, cache_read_tokens: reported },
        };
        expect(inspectGenerationUsageAccounting(usage)).toEqual({ status: 'valid', usage });
        expect(
            inspectGenerationUsageAccounting({
                input_tokens: 110,
                accounting_provenance: { input_tokens: reported },
            }),
        ).toEqual({
            status: 'valid',
            usage: {
                input_tokens: 110,
                accounting_provenance: { input_tokens: reported },
            },
        });
    });
    it('distinguishes absent, reported-only and explicit known zero measurements', () => {
        expect(inspectGenerationUsageAccounting(undefined)).toEqual({ status: 'absent' });
        expect(inspectGenerationUsageAccounting({ reported_usage: [{ payload: {} }] })).toEqual({
            status: 'unmeasured',
            reason: 'no_normalized_usage',
        });
        expect(
            inspectGenerationUsageAccounting({ output_tokens: 0, accounting_provenance: { output_tokens: reported } }),
        ).toMatchObject({ status: 'valid', usage: { output_tokens: 0 } });
    });
    it('preserves reported versus estimated cost provenance and non-USD currency', () => {
        for (const provenance of ['reported', 'estimated'] as const) {
            const usage: GenerationUsage = { cost: { amount: '0.125', currency: 'EUR', provenance } };
            expect(inspectGenerationUsageAccounting(usage)).toEqual({ status: 'valid', usage });
        }
    });
    it('owns parsed cost and provenance after the caller mutates the source', () => {
        const source: GenerationUsage = {
            ...structuredClone(partition),
            cost: { amount: '0.125', currency: 'USD', provenance: 'reported' },
        };
        const inspected = inspectGenerationUsageAccounting(source);
        expect(inspected.status).toBe('valid');
        if (inspected.status !== 'valid' || !source.cost || !source.accounting_provenance?.input_tokens)
            throw new Error('Fixture needs valid source cost and provenance');
        source.cost.amount = '999';
        source.cost.currency = 'EUR';
        source.accounting_provenance.input_tokens.accounting_basis = 'changed';
        expect(inspected.usage.cost).toEqual({ amount: '0.125', currency: 'USD', provenance: 'reported' });
        expect(inspected.usage.accounting_provenance?.input_tokens).toEqual(reported);
        expect(inspected.usage.input_tokens).toBe(110);
    });
    it('reuses canonical partition validation rather than clamping inconsistent buckets', () => {
        expect(inspectGenerationUsageAccounting({ ...partition, input_new_tokens: 24 })).toMatchObject({
            status: 'invalid',
            reason: 'inconsistent',
            diagnostics: ['INPUT_PARTITION_INVALID'],
        });
        expect(inspectGenerationUsageAccounting({ ...partition, cache_write_tokens: undefined })).toMatchObject({
            status: 'invalid',
            reason: 'inconsistent',
        });
        expect(inspectGenerationUsageAccounting({ ...partition, reasoning_tokens: 6 })).toMatchObject({
            status: 'invalid',
            reason: 'inconsistent',
            diagnostics: ['USAGE_BREAKDOWN_INVALID'],
        });
    });
    it('rejects invalid provenance, incompatible totals and overflow using the shared semantics', () => {
        expect(inspectGenerationUsageAccounting({ output_tokens: 5 })).toMatchObject({
            status: 'invalid',
            reason: 'inconsistent',
            diagnostics: ['ACCOUNTING_PROVENANCE_MISMATCH'],
        });
        expect(
            inspectGenerationUsageAccounting({
                total_tokens: 10,
                accounting_provenance: { total_tokens: reported },
            }),
        ).toMatchObject({ status: 'invalid', reason: 'inconsistent', diagnostics: ['USAGE_TOTAL_INVALID'] });
        const { total_tokens: _total, ...withoutTotal } = partition;
        const { total_tokens: _totalProvenance, ...withoutTotalProvenance } = partition.accounting_provenance ?? {};
        expect(
            inspectGenerationUsageAccounting({
                ...withoutTotal,
                input_tokens: Number.MAX_SAFE_INTEGER,
                output_tokens: 5,
                accounting_provenance: withoutTotalProvenance,
            }),
        ).toMatchObject({
            status: 'invalid',
            reason: 'inconsistent',
            diagnostics: expect.arrayContaining(['USAGE_OVERFLOW']),
        });
    });
    it('bounds scalar inspection and never accesses an ignored reporting getter', () => {
        const source = { ...partition };
        Object.defineProperty(source, 'reported_usage', {
            get: () => {
                throw new Error('Must not inspect reporting');
            },
        });
        expect(inspectGenerationUsageAccounting(source)).toEqual({ status: 'valid', usage: partition });
        expect(
            inspectGenerationUsageAccounting({
                cost: { amount: '1'.repeat(70_000), currency: 'USD', provenance: 'reported' },
            }),
        ).toMatchObject({ status: 'invalid', reason: 'resource_limit' });
        const accessor = {};
        Object.defineProperty(accessor, 'output_tokens', {
            get: () => {
                throw new Error('Must not invoke accounting getter');
            },
        });
        expect(inspectGenerationUsageAccounting(accessor)).toMatchObject({ status: 'invalid', reason: 'malformed' });
        const provenance = {};
        Object.defineProperty(provenance, 'output_tokens', {
            enumerable: true,
            get: () => {
                throw new Error('Must not invoke nested getter');
            },
        });
        expect(inspectGenerationUsageAccounting({ output_tokens: 5, accounting_provenance: provenance })).toMatchObject(
            { status: 'invalid', reason: 'malformed' },
        );
    });
    it.each([null, [], 5, { input_tokens: -1 }])(
        'rejects malformed scalar input %s without inventing usage',
        (input) => {
            expect(inspectGenerationUsageAccounting(input)).toMatchObject({ status: 'invalid', reason: 'malformed' });
        },
    );
});
