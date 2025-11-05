import { DefaultAzureCredential, getBearerTokenProvider } from "@azure/identity";
import { AIModel, DriverOptions, getModelCapabilities, modelModalitiesToArray, Providers } from "@llumiverse/core";
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
    readonly provider = Providers.azure_openai;

    //Overload to allow independent instantiation with AzureOpenAI service
    constructor(serviceOrOpts: AzureOpenAI | AzureOpenAIDriverOptions) {
        if (serviceOrOpts instanceof AzureOpenAI) {
            super({});
            this.service = serviceOrOpts;
            return;
        }
        const opts = serviceOrOpts ?? {};
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
        if (!this.service.deploymentName) {
            throw new Error("A specific deployment is not set. Azure OpenAI cannot list deployments. Update your endpoint URL to include the deployment name, e.g., https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions");
        }
            
        //Do a test execution to check if the model works and to get the model ID.
        let modelID = this.service.deploymentName;
        try {
            const testResponse = await this.service.chat.completions.create({
                model: this.service.deploymentName,
                messages: [{ role: "user", content: "Hi" }],
                max_tokens: 1,
            });
            modelID = testResponse.model;
        } catch (error) {
            this.logger.error({ error }, "Failed to test model for Azure OpenAI listing :");
        }
        const modelCapability = getModelCapabilities(modelID, "openai");
        return [{
            id: modelID,
            name: this.service.deploymentName,
            provider: this.provider,
            owner: "openai",
            input_modalities: modelModalitiesToArray(modelCapability.input),
            output_modalities: modelModalitiesToArray(modelCapability.output),
            tool_support: modelCapability.tool_support,
        } satisfies AIModel<string>];
    }
}