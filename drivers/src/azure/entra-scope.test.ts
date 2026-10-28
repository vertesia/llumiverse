import { describe, expect, it } from 'vitest';
import { azureFoundryScope, cognitiveServicesScope } from './entra-scope.js';

describe('Azure Entra scopes', () => {
    it('keeps the public-cloud scopes for public and unknown endpoints', () => {
        expect(azureFoundryScope('https://acme.services.ai.azure.com/api/projects/p')).toBe(
            'https://ai.azure.com/.default',
        );
        expect(cognitiveServicesScope('https://acme.openai.azure.com')).toBe(
            'https://cognitiveservices.azure.com/.default',
        );
        expect(azureFoundryScope(undefined)).toBe('https://ai.azure.com/.default');
        expect(cognitiveServicesScope('not a url')).toBe('https://cognitiveservices.azure.com/.default');
    });

    it('uses the Azure Government scopes for .azure.us endpoints', () => {
        expect(azureFoundryScope('https://acme.services.ai.azure.us/api/projects/p')).toBe(
            'https://ai.azure.us/.default',
        );
        expect(cognitiveServicesScope('https://acme.openai.azure.us')).toBe(
            'https://cognitiveservices.azure.us/.default',
        );
    });

    it('does not treat a look-alike host as Azure Government', () => {
        expect(azureFoundryScope('https://acme.services.ai.azure.us.example.com')).toBe(
            'https://ai.azure.com/.default',
        );
    });
});
