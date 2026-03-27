import { Modalities } from '@llumiverse/core';
import { GoogleGenAI } from '@google/genai';
import 'dotenv/config';
import { describe, expect, test } from 'vitest';
import { BedrockDriver } from '../src/index.js';

const TIMEOUT = 30_000;

describe('List models using API keys', () => {

    test.skipIf(!process.env.BEDROCK_API_KEY || !process.env.BEDROCK_REGION)(
        'Bedrock: converse with bearer token API key',
        { timeout: TIMEOUT },
        async () => {
            const driver = new BedrockDriver({
                region: process.env.BEDROCK_REGION!,
                token: { token: process.env.BEDROCK_API_KEY! },
            });

            const result = await driver.execute(
                [{ role: 'user', content: 'Say "hello" and nothing else.' }],
                {
                    model: 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                    output_modality: Modalities.text,
                    model_options: { _option_id: 'text-fallback', max_tokens: 64, temperature: 0 },
                },
            );
            console.log('Bedrock API key result:', result.result);
            expect(result.result).toBeDefined();
            expect(result.error).toBeFalsy();
        },
    );

    test.skipIf(!process.env.GEMINI_API_KEY)(
        'Gemini: list models with API key',
        { timeout: TIMEOUT },
        async () => {
            const client = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });

            const pager = await client.models.list({ config: { pageSize: 100 } });
            const models = [];
            for await (const model of pager) {
                models.push(model);
            }
            console.log(`Gemini API key: found ${models.length} models`);
            expect(models.length).toBeGreaterThan(0);

            // Verify we get gemini models
            const geminiModels = models.filter(m => m.name?.includes('gemini'));
            console.log(`  of which ${geminiModels.length} are Gemini models`);
            expect(geminiModels.length).toBeGreaterThan(0);
        },
    );
});
