import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AzureFoundryDriver } from '../azure/azure_foundry.js';
import { xAIDriver } from '../xai/index.js';
import { AzureOpenAIDriver } from './azure_openai.js';
import { OpenAIDriver } from './openai.js';
import { OpenAIChatCompletionsDriver } from './openai_chat_completions.js';
import { OpenAIResponsesDriver } from './openai_responses.js';

type AccountFields = { organization: string | null; project: string | null };

describe('OpenAI account settings from the environment', () => {
    beforeEach(() => {
        vi.stubEnv('OPENAI_ORG_ID', 'org-from-env');
        vi.stubEnv('OPENAI_PROJECT_ID', 'proj_from_env');
    });
    afterEach(() => vi.unstubAllEnvs());

    it.each([
        ['xAI', () => new xAIDriver({ apiKey: 'key' }).service],
        [
            'an OpenAI-compatible Responses endpoint',
            () => new OpenAIResponsesDriver({ apiKey: 'key', endpoint: 'https://compatible.example/v1' }).service,
        ],
        [
            'an OpenAI-compatible Chat Completions endpoint',
            () => new OpenAIChatCompletionsDriver({ apiKey: 'key', endpoint: 'https://compatible.example/v1' }).service,
        ],
        [
            'Azure OpenAI (v1 API)',
            () => new AzureOpenAIDriver({ apiKey: 'key', endpoint: 'https://resource.openai.azure.com' }).service,
        ],
        [
            'Azure OpenAI (dated API)',
            () =>
                new AzureOpenAIDriver({
                    apiKey: 'key',
                    endpoint: 'https://resource.openai.azure.com',
                    apiVersion: '2024-10-21',
                }).service,
        ],
        [
            'Azure Foundry inference',
            () =>
                (
                    new AzureFoundryDriver({
                        apiKey: 'key',
                        endpoint: 'https://resource.services.ai.azure.com/api/projects/project',
                    }) as unknown as { getInferenceClient(): unknown }
                ).getInferenceClient(),
        ],
    ])('are not sent to %s', (_name, client) => {
        const { organization, project } = client() as AccountFields;
        expect(organization).toBeNull();
        expect(project).toBeNull();
    });

    it('are still read by the OpenAI driver, whose account they describe', () => {
        const { organization, project } = new OpenAIDriver({ apiKey: 'key' }).service as unknown as AccountFields;
        expect(organization).toBe('org-from-env');
        expect(project).toBe('proj_from_env');
    });
});
