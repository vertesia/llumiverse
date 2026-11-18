import type { OpenAIResponsesPolicy } from './index.js';

/**
 * Azure serves the Responses image-generation tool from a separate image deployment, named in a header, on the preview
 * API. Shared by Azure OpenAI and Microsoft Foundry.
 */
export function azureResponsesPolicy(requestModel: (model: string) => string): OpenAIResponsesPolicy {
    return {
        requestHeaders: (options) => {
            const imageTool = (options.model_options as { image_generation?: { model: string } } | undefined)
                ?.image_generation;
            if (!imageTool) return undefined;
            return {
                'x-ms-oai-image-generation-deployment': requestModel(imageTool.model),
                api_version: 'preview',
            };
        },
    };
}
