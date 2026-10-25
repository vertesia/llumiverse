import { Providers } from '@llumiverse/core';
import { describe, expect, it } from 'vitest';
import { createDriver, isDriverFactoryProvider } from './factory.js';

describe('createDriver', () => {
    it('covers every provider', () => {
        for (const provider of Object.values(Providers)) {
            expect(isDriverFactoryProvider(provider), provider).toBe(true);
        }
    });

    it('creates the driver of the requested provider', async () => {
        await expect(createDriver('test', {})).resolves.toMatchObject({ provider: 'test' });
        await expect(createDriver('openai', { apiKey: 'key' })).resolves.toMatchObject({ provider: Providers.openai });
    });

    it('rejects an unknown provider', async () => {
        expect(isDriverFactoryProvider('toString')).toBe(false);
        await expect(createDriver('unknown' as 'test', {})).rejects.toThrow('Unknown driver provider: unknown');
    });
});
