import type { AbstractDriver, ExecutionOptions, PromptSegment } from '@llumiverse/core';
import { PromptRole } from '@llumiverse/core';
import 'dotenv/config';
import { describe, expect, test } from 'vitest';
import {
    AnthropicDriver,
    CloudflareAIGatewayDriver,
    OpenAIDriver,
    TogetherAIDriver,
    VercelAIGatewayDriver,
} from '../src/index.js';

const TIMEOUT = 120_000;

interface LiveModelCase {
    name: string;
    driver: AbstractDriver;
    model: string;
    options?: ExecutionOptions['model_options'];
}

const prompt: PromptSegment[] = [
    {
        role: PromptRole.user,
        content: 'Reply with one short sentence confirming that the model execution path is working.',
    },
];

const cases: LiveModelCase[] = [];
const openAiApiKey = env('OPENAI_API_KEY');
const anthropicApiKey = env('ANTHROPIC_API_KEY');
const togetherApiKey = env('TOGETHER_AI_API_KEY') ?? env('TOGETHER_API_KEY');
const cloudflareAccountId = env('CLOUDFLARE_ACCOUNT_ID');
const cloudflareGateway = env('CLOUDFLARE_AI_GATEWAY_NAME');
const cloudflareApiKey = env('CLOUDFLARE_AI_GATEWAY_API_KEY');
const vercelApiKey = env('VERCEL_AI_API_KEY');

function env(name: string): string | undefined {
    const value = process.env[name]?.trim();
    return value ? value : undefined;
}

function addCase(testCase: LiveModelCase | undefined): void {
    if (testCase) {
        cases.push(testCase);
    }
}

function textOptions(maxTokens = 32): ExecutionOptions['model_options'] {
    return {
        _option_id: 'text-fallback',
        max_tokens: maxTokens,
        temperature: 0,
    };
}

addCase(
    openAiApiKey
        ? {
              name: 'openai',
              driver: new OpenAIDriver({ apiKey: openAiApiKey }),
              model: env('OPENAI_LIVE_MODEL') ?? 'gpt-4o-mini',
              options: {
                  _option_id: 'openai-text',
                  max_tokens: 32,
                  temperature: 0,
              },
          }
        : undefined,
);

addCase(
    anthropicApiKey
        ? {
              name: 'anthropic',
              driver: new AnthropicDriver({ apiKey: anthropicApiKey }),
              model: env('ANTHROPIC_LIVE_MODEL') ?? 'claude-sonnet-4-5',
          }
        : undefined,
);

addCase(
    togetherApiKey
        ? {
              name: 'togetherai',
              driver: new TogetherAIDriver({ apiKey: togetherApiKey }),
              model: env('TOGETHER_LIVE_MODEL') ?? 'moonshotai/Kimi-K2.6',
              options: textOptions(),
          }
        : undefined,
);

addCase(
    cloudflareAccountId && cloudflareGateway && cloudflareApiKey
        ? {
              name: 'cloudflare-ai-gateway',
              driver: new CloudflareAIGatewayDriver({
                  accountId: cloudflareAccountId,
                  gateway: cloudflareGateway,
                  apiKey: cloudflareApiKey,
              }),
              model: env('CLOUDFLARE_LIVE_MODEL') ?? '@cf/zai-org/glm-5.2',
              options: textOptions(256),
          }
        : undefined,
);

addCase(
    vercelApiKey
        ? {
              name: 'vercel-ai-gateway',
              driver: new VercelAIGatewayDriver({ apiKey: vercelApiKey }),
              model: env('VERCEL_LIVE_MODEL') ?? 'openai/gpt-4o-mini',
              options: textOptions(),
          }
        : undefined,
);

if (cases.length === 0) {
    console.warn('Live model tests skipped: no provider API keys were configured.');
}

describe.concurrent.each(cases)('Live model driver $name', ({ driver, model, name, options }) => {
    test(`${name}: executes ${model}`, { timeout: TIMEOUT, retry: 1 }, async () => {
        const result = await driver.execute(prompt, {
            model,
            model_options: options,
        });
        const text = result.result.map((item) => ('value' in item ? String(item.value) : '')).join('');
        expect(result.error).toBeFalsy();
        expect(text.trim().length).toBeGreaterThan(0);
    });
});
