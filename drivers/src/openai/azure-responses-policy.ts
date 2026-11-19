import type { OpenAiImageGenerationOptions } from '@llumiverse/core';
import type { OpenAIResponsesPolicy } from './index.js';

/**
 * Azure serves the Responses image-generation tool from a separate image deployment, named in the
 * `x-ms-oai-image-generation-deployment` header. Shared by Azure OpenAI and Microsoft Foundry. The API version is the
 * client's: Azure OpenAI's image-tool client sends `api-version=preview` as a query parameter, and Foundry uses v1.
 */
export function azureResponsesPolicy(requestModel: (model: string) => string): OpenAIResponsesPolicy {
    return {
        requestHeaders: (options) => {
            const imageTool = (options.model_options as { image_generation?: OpenAiImageGenerationOptions } | undefined)
                ?.image_generation;
            if (!imageTool) return undefined;
            return { 'x-ms-oai-image-generation-deployment': requestModel(imageTool.model) };
        },
    };
}
