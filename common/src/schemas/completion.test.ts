import { describe, expect, it } from 'vitest';
import { CompletionResultSchema, PromptCacheDiagnosticSchema, StatelessExecutionOptionsSchema } from './completion.js';

describe('CompletionResultSchema', () => {
    it('accepts thoughts as a separate completion result type', () => {
        expect(CompletionResultSchema.parse({ type: 'thoughts', value: 'Reasoning summary' })).toEqual({
            type: 'thoughts',
            value: 'Reasoning summary',
        });
    });

    it('accepts a provider-neutral video result', () => {
        expect(CompletionResultSchema.parse({ type: 'video', value: 'gs://bucket/video.mp4' })).toEqual({
            type: 'video',
            value: 'gs://bucket/video.mp4',
        });
    });
});

describe('StatelessExecutionOptionsSchema', () => {
    it('requires explicit cache TTLs to meet the provider minimum', () => {
        expect(
            StatelessExecutionOptionsSchema.safeParse({ model: 'gemini-3.7-flash', prompt_cache_ttl_seconds: 60 })
                .success,
        ).toBe(true);
        expect(
            StatelessExecutionOptionsSchema.safeParse({ model: 'gemini-3.7-flash', prompt_cache_ttl_seconds: 59 })
                .success,
        ).toBe(false);
    });

    it('accepts required mode for explicit-cache diagnostics', () => {
        expect(
            StatelessExecutionOptionsSchema.parse({ model: 'gemini-3.7-flash', prompt_cache_mode: 'required' }),
        ).toMatchObject({ prompt_cache_mode: 'required' });
    });

    it('publishes safe cache-path diagnostics without provider resource data', () => {
        expect(
            PromptCacheDiagnosticSchema.parse({
                path: 'distributed_registry_hit',
                explicit_cache_used: true,
                content_hash_prefix: '0123456789ab',
                model: 'gemini-3.7-flash',
                scope: 'environment:project:us-central1',
                cacheable_part_count: 2,
                preparation_latency_ms: 4,
            }),
        ).not.toHaveProperty('resource_name');
    });
});
