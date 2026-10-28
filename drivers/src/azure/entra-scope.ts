/**
 * Entra token scopes of the Azure AI services. Azure Government serves them under `.azure.us`
 * (`<resource>.services.ai.azure.us`, `<resource>.openai.azure.us`) and only accepts tokens issued for
 * the matching `.us` resource, so the scope follows the endpoint's host. Any other host, including a
 * missing or unparsable endpoint, keeps the public-cloud scope.
 */
function isAzureGovernmentEndpoint(endpoint: string | undefined): boolean {
    if (!endpoint) return false;
    try {
        return new URL(endpoint).hostname.toLowerCase().endsWith('.azure.us');
    } catch {
        return false;
    }
}

/** Scope of an AI Foundry project endpoint. */
export function azureFoundryScope(endpoint: string | undefined): string {
    return isAzureGovernmentEndpoint(endpoint) ? 'https://ai.azure.us/.default' : 'https://ai.azure.com/.default';
}

/** Scope of an Azure OpenAI (Cognitive Services) endpoint. */
export function cognitiveServicesScope(endpoint: string | undefined): string {
    return isAzureGovernmentEndpoint(endpoint)
        ? 'https://cognitiveservices.azure.us/.default'
        : 'https://cognitiveservices.azure.com/.default';
}
