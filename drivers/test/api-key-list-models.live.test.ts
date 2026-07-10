import { GoogleGenAI } from '@google/genai';
import 'dotenv/config';
import { describe, expect, test } from 'vitest';

const TIMEOUT = 30_000;

describe('List models using API keys', () => {
    test.skipIf(!process.env.GEMINI_API_KEY)('Gemini: list models with API key', { timeout: TIMEOUT }, async () => {
        const apiKey = process.env.GEMINI_API_KEY;
        if (!apiKey) {
            throw new Error('GEMINI_API_KEY is required');
        }
        const client = new GoogleGenAI({ apiKey });

        const pager = await client.models.list({ config: { pageSize: 100 } });
        const models = [];
        for await (const model of pager) {
            models.push(model);
        }
        console.log(`Gemini API key: found ${models.length} models`);
        expect(models.length).toBeGreaterThan(0);

        // Verify we get gemini models
        const geminiModels = models.filter((m) => m.name?.includes('gemini'));
        console.log(`  of which ${geminiModels.length} are Gemini models`);
        expect(geminiModels.length).toBeGreaterThan(0);
    });
});
