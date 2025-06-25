import { DefaultAzureCredential, getBearerTokenProvider } from "@azure/identity";
import { AIModel, DriverOptions, getModelCapabilities, modelModalitiesToArray } from "@llumiverse/core";
import OpenAI, { AzureOpenAI } from "openai";
import { BaseOpenAIDriver } from "./index.js";

export interface AzureOpenAIDriverOptions extends DriverOptions {

    /**
     * The credentials to use to access Azure OpenAI
     */
    azureADTokenProvider?: any; //type with azure credentials
    
    apiKey?: string;

    endpoint?: string;

    apiVersion?: string

    deployment?: string;

}

export class AzureOpenAIDriver extends BaseOpenAIDriver {


    service: AzureOpenAI;
    provider: "azure_openai";

    constructor(opts: AzureOpenAIDriverOptions) {
        super(opts);

        if (!opts.azureADTokenProvider && !opts.apiKey) {
            opts.azureADTokenProvider = this.getDefaultCognitiveServicesAuth();
        }

        this.service = new AzureOpenAI({
            apiKey: opts.apiKey,
            azureADTokenProvider: opts.azureADTokenProvider,          
            endpoint: opts.endpoint,
            apiVersion: opts.apiVersion ?? "2024-10-21",
            deployment: opts.deployment
        });
        this.provider = "azure_openai";
    }

    /**
     * Get default authentication for Azure Cognitive Services API
     */
    getDefaultCognitiveServicesAuth() {
        const scope = "https://cognitiveservices.azure.com/.default";
        const azureADTokenProvider = getBearerTokenProvider(new DefaultAzureCredential(), scope);
        return azureADTokenProvider;
    }
    
    async listModels(): Promise<AIModel[]> {
        return this._listModels();
    }

    async _listModels(_filter?: (m: OpenAI.Models.Model) => boolean): Promise<AIModel[]> {
        if (this.service.deploymentName) {
            //Do a test execution to check if the model works
            let modelID = this.service.deploymentName;
            try {
                const testResponse = await this.service.chat.completions.create({
                    model: this.service.deploymentName,
                    messages: [{ role: "user", content: "Hi" }],
                    max_tokens: 1,
                });
                modelID = testResponse.model;
            } catch (error) {
                this.logger.error("Failed to test model for listing:", error);
            }
            const modelCapability = getModelCapabilities(modelID, "openai");
            return [{
                id: modelID,
                name: this.service.deploymentName,
                provider: this.provider,
                owner: "azure_openai", // Azure OpenAI does not expose ownership in the same way
                can_stream: false, // Streaming is supported, but we don't have a way to check if it works for this model.
                input_modalities: modelModalitiesToArray(modelCapability.input),
                output_modalities: modelModalitiesToArray(modelCapability.output),
                tool_support: modelCapability.tool_support,
            } satisfies AIModel<string>];
        } else {
            return [];
        }
    }
}