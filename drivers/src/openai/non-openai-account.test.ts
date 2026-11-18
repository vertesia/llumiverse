import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AzureFoundryDriver } from '../azure/azure_foundry.js';
import { xAIDriver } from '../xai/index.js';
import { AzureOpenAIDriver } from './azure_openai.js';
import { OpenAIDriver } from './openai.js';
import { OpenAIChatCompletionsDriver } from './openai_chat_completions.js';
import { OpenAIResponsesDriver } from './openai_responses.js';

type AccountFields = { organization: string | null; project: string | null };
type Built = { driver: { destroy(): void }; client: unknown };

function withClient(driver: { service: unknown; destroy(): void }): Built {
    return { driver, client: driver.service };
}

describe('OpenAI account settings from the environment', () => {
    beforeEach(() => {
        vi.stubEnv('OPENAI_ORG_ID', 'org-from-env');
        vi.stubEnv('OPENAI_PROJECT_ID', 'proj_from_env');
    });
    afterEach(() => vi.unstubAllEnvs());

    it.each([
        ['xAI', () => withClient(new xAIDriver({ apiKey: 'key' }))],
        [
            'an OpenAI-compatible Responses endpoint',
            () => withClient(new OpenAIResponsesDriver({ apiKey: 'key', endpoint: 'https://compatible.example/v1' })),
        ],
        [
            'an OpenAI-compatible Chat Completions endpoint',
            () =>
                withClient(
                    new OpenAIChatCompletionsDriver({ apiKey: 'key', endpoint: 'https://compatible.example/v1' }),
                ),
        ],
        [
            'Azure OpenAI image generation',
            () => {
                const driver = new AzureOpenAIDriver({ apiKey: 'key', endpoint: 'https://resource.openai.azure.com' });
                return { driver, client: (driver as unknown as { imageService: unknown }).imageService };
            },
        ],
        [
            'Azure OpenAI (dated API)',
            () =>
                withClient(
                    new AzureOpenAIDriver({
                        apiKey: 'key',
                        endpoint: 'https://resource.openai.azure.com',
                        apiVersion: '2024-10-21',
                    }),
                ),
        ],
        [
            'Azure Foundry inference',
            () => {
                const driver = new AzureFoundryDriver({
                    azureADTokenProvider: { getToken: async () => ({ token: 'token', expiresOnTimestamp: 0 }) },
                    endpoint: 'https://resource.services.ai.azure.com/api/projects/project',
                });
                return {
                    driver,
                    client: (driver as unknown as { getInferenceClient(): unknown }).getInferenceClient(),
                };
            },
        ],
    ])('are not sent to %s', (_name, build: () => Built) => {
        const { driver, client } = build();
        try {
            const { organization, project } = client as AccountFields;
            expect(organization).toBeNull();
            expect(project).toBeNull();
        } finally {
            driver.destroy();
        }
    });

    it('are still read by the OpenAI driver, whose account they describe', () => {
        const driver = new OpenAIDriver({ apiKey: 'key' });
        try {
            const { organization, project } = driver.service as unknown as AccountFields;
            expect(organization).toBe('org-from-env');
            expect(project).toBe('proj_from_env');
        } finally {
            driver.destroy();
        }
    });
});
