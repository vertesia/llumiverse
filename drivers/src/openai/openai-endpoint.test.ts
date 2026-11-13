import OpenAI from 'openai';
import { describe, expect, it } from 'vitest';
import { OpenAIDriver } from './openai.js';

describe('OpenAI driver endpoint selection', () => {
    it('keeps the SDK default when neither an endpoint nor a residency is set', () => {
        const driver = new OpenAIDriver({ apiKey: 'test' });
        expect(driver.service.baseURL).toBe(new OpenAI({ apiKey: 'test' }).baseURL);
    });

    it.each([
        ['au', 'https://au.api.openai.com/v1'],
        ['in', 'https://in.api.openai.com/v1'],
        ['jp', 'https://jp.api.openai.com/v1'],
    ])('selects the %s residency host', (dataResidency, baseURL) => {
        expect(new OpenAIDriver({ apiKey: 'test', dataResidency }).service.baseURL).toBe(baseURL);
    });

    it('uses a supplied endpoint', () => {
        const driver = new OpenAIDriver({ apiKey: 'test', endpoint: 'https://proxy.example.com/v1' });
        expect(driver.service.baseURL).toBe('https://proxy.example.com/v1');
    });

    it('lets the SDK reject an endpoint combined with a residency', () => {
        expect(
            () => new OpenAIDriver({ apiKey: 'test', endpoint: 'https://proxy.example.com/v1', dataResidency: 'eu' }),
        ).toThrow('mutually exclusive');
    });
});
