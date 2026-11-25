import { describe, expect, it } from 'vitest';
import { Providers } from '../types.js';
import { CURATED_MODEL_LIFECYCLE, getModelLifecycleDates, toLifecycleDate } from './model-lifecycle.js';

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

describe('curated model lifecycle table', () => {
    it.each(Object.entries(CURATED_MODEL_LIFECYCLE))('%s entries are well formed', (_provider, entries) => {
        const keys = new Set<string>();
        for (const entry of entries ?? []) {
            expect(entry.id).toBe(entry.id.toLowerCase());
            expect(entry.retirement_date).toMatch(ISO_DATE);
            expect(Number.isNaN(Date.parse(entry.retirement_date))).toBe(false);
            if (entry.deprecation_date) {
                expect(entry.deprecation_date).toMatch(ISO_DATE);
                expect(entry.deprecation_date <= entry.retirement_date).toBe(true);
            }
            const key = `${entry.id}@${entry.version ?? ''}`;
            expect(keys.has(key), key).toBe(false);
            keys.add(key);
        }
    });

    it('covers only providers whose model listing has no lifecycle data', () => {
        expect(Object.keys(CURATED_MODEL_LIFECYCLE).sort()).toEqual(
            [Providers.azure_foundry, Providers.azure_openai, Providers.vertexai, Providers.xai].sort(),
        );
    });
});

describe('getModelLifecycleDates', () => {
    it('matches Vertex AI resource names and publisher model versions', () => {
        expect(
            getModelLifecycleDates('locations/global/publishers/google/models/gemini-3.6-flash', Providers.vertexai),
        ).toEqual({ retirement_date: '2026-11-19' });
        expect(
            getModelLifecycleDates('publishers/anthropic/models/claude-sonnet-4-5@20250929', Providers.vertexai),
        ).toEqual({ deprecation_date: '2026-09-29', retirement_date: '2026-11-30' });
    });

    it('does not match a longer model ID that shares a prefix', () => {
        expect(getModelLifecycleDates('gemini-2.5-flash', Providers.vertexai)?.retirement_date).toBe('2026-10-20');
        expect(getModelLifecycleDates('gemini-2.5-flash-preview-09-2025', Providers.vertexai)).toBeUndefined();
        expect(getModelLifecycleDates('gemini-3.8-flash', Providers.vertexai)).toBeUndefined();
    });

    it('leaves Bedrock to its regional listing', () => {
        // Bedrock dates differ by region, so a region-blind table would date models in regions AWS does not list.
        expect(getModelLifecycleDates('minimax.minimax-m2', Providers.bedrock)).toBeUndefined();
        expect(getModelLifecycleDates('moonshotai.kimi-k2-thinking', Providers.bedrock_mantle)).toBeUndefined();
    });

    it('resolves Azure models by name and version', () => {
        // Azure OpenAI reports the deployed model as `<name>-<version>`.
        expect(getModelLifecycleDates('gpt-4o-2024-05-13', Providers.azure_openai)).toEqual({
            retirement_date: '2027-01-14',
        });
        // Microsoft Foundry lists the name and the version separately.
        expect(getModelLifecycleDates('gpt-4o', Providers.azure_foundry, '2024-11-20')).toEqual({
            retirement_date: '2027-04-14',
        });
        // Versions of gpt-4o retire on different dates, so the bare name has no single date.
        expect(getModelLifecycleDates('gpt-4o', Providers.azure_foundry)).toBeUndefined();
        // A version the table does not know gets no date from another version.
        expect(getModelLifecycleDates('gpt-4o', Providers.azure_foundry, '2027-01-01')).toBeUndefined();
        // Partner models have one date for every version, and Foundry model names are matched case-insensitively.
        expect(getModelLifecycleDates('claude-sonnet-4-5', Providers.azure_foundry, '1')).toEqual({
            retirement_date: '2026-11-30',
        });
        expect(getModelLifecycleDates('Phi-4', Providers.azure_foundry)).toEqual({ retirement_date: '2027-01-17' });
    });

    it('resolves xAI models by ID', () => {
        expect(getModelLifecycleDates('grok-build-0.1', Providers.xai)).toEqual({ retirement_date: '2026-11-19' });
    });

    it('returns nothing for unknown models and providers without a curated table', () => {
        expect(getModelLifecycleDates('gemini-9-future', Providers.vertexai)).toBeUndefined();
        expect(getModelLifecycleDates('claude-sonnet-4-5-20250929', Providers.anthropic)).toBeUndefined();
        expect(getModelLifecycleDates('gpt-5', Providers.openai)).toBeUndefined();
    });
});

describe('toLifecycleDate', () => {
    it('normalizes provider date formats to a UTC calendar date', () => {
        expect(toLifecycleDate('2026-12-11')).toBe('2026-12-11');
        expect(toLifecycleDate('2026-11-30T00:00:00Z')).toBe('2026-11-30');
        expect(toLifecycleDate(new Date('2027-01-08T00:00:00Z'))).toBe('2027-01-08');
    });

    it('ignores missing and unparseable values', () => {
        expect(toLifecycleDate(null)).toBeUndefined();
        expect(toLifecycleDate(undefined)).toBeUndefined();
        expect(toLifecycleDate('')).toBeUndefined();
        expect(toLifecycleDate('soon')).toBeUndefined();
    });
});
