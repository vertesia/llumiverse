import { DefaultAzureCredential, getBearerTokenProvider } from "@azure/identity";
import { AIModel, DriverOptions, ExecutionOptions } from "@llumiverse/core";
import OpenAI, { AzureOpenAI } from "openai";
import { BaseOpenAIDriver } from "./index.js";

export interface AzureFoundryDriverOptions extends DriverOptions {

    /**
     * The credentials to use to access Azure OpenAI
     */
    azureADTokenProvider?: any; //type with azure credentials
    
    apiKey?: string;

    endpoint?: string;

    apiVersion?: string

    deployment?: string;

}

export class AzureFoundryDriver extends BaseOpenAIDriver {

    service: AzureOpenAI;
    provider: "azure_foundry";

    constructor(opts: AzureFoundryDriverOptions) {
        super(opts);

        if (!opts.azureADTokenProvider && !opts.apiKey) {
            opts.azureADTokenProvider = this.getDefaultCognitiveServicesAuth();
        }

        this.service = new AzureOpenAI({
            apiKey: opts.apiKey,
            azureADTokenProvider: opts.azureADTokenProvider,          
            endpoint: opts.endpoint,
            apiVersion: opts.apiVersion ?? "2024-10-21",
        });
        this.provider = "azure_foundry";
    }

    /**
     * Get default authentication for Azure AI Foundry API
     */
    getDefaultAIFoundryAuth() {
        const scope = "https://ai.azure.com/.default";
        const azureADTokenProvider = getBearerTokenProvider(new DefaultAzureCredential(), scope);
        return azureADTokenProvider;
    }

    /**
     * Get default authentication for Azure Cognitive Services API
     */
    getDefaultCognitiveServicesAuth() {
        const scope = "https://cognitiveservices.azure.com/.default";
        const azureADTokenProvider = getBearerTokenProvider(new DefaultAzureCredential(), scope);
        return azureADTokenProvider;
    }

    /**
     * Get default authentication for Azure Management API (for ARM operations)
     */
    getDefaultManagementAuth() {
        const scope = "https://management.azure.com/.default";
        const azureADTokenProvider = getBearerTokenProvider(new DefaultAzureCredential(), scope);
        return azureADTokenProvider;
    }

    protected canStream(_options: ExecutionOptions): Promise<boolean> {
        return Promise.resolve(false);
    }
    
    async listModels(): Promise<AIModel[]> {
        return this._listModels();
    }

    async _listModels(_filter?: (m: OpenAI.Models.Model) => boolean): Promise<AIModel[]> {

        let result = await this.listDeployments();
        //let result: OpenAI.Models.Model[] = [];
        console.debug("service.model.deployments(): ", JSON.stringify(result, null, 2));

        console.log("Azure client")
        console.log(this.service.apiKey)
        console.log(this.service.baseURL)

        return [];
        
        // const models = filter ? result.filter(filter) : result;
        // const aiModels = models.map((m) => {
        //     const modelCapability = getModelCapabilities(m.id, "openai");
        //     return {
        //         id: m.id,
        //         name: m.id,
        //         provider: this.provider,
        //         owner: m.owned_by,
        //         type: m.object === "model" ? ModelType.Text : ModelType.Unknown,
        //         can_stream: true,
        //         is_multimodal: m.id.includes("gpt-4"),
        //         input_modalities: modelModalitiesToArray(modelCapability.input),
        //         output_modalities: modelModalitiesToArray(modelCapability.output),
        //         tool_support: modelCapability.tool_support,
        //     } satisfies AIModel<string>;
        // }).sort((a, b) => a.id.localeCompare(b.id));

        // return aiModels;
    }

    async listDeployments() {
        // If using API key only, fall back to models.list()
        if (this.service.apiKey && !process.env.AZURE_SUBSCRIPTION_ID) {
            console.log("Using API key authentication - falling back to models.list()");
            try {
                const modelsResponse = await this.service.models.list();
                return { data: modelsResponse.data, source: 'models_api' };
            } catch (error: any) {
                console.error('Error fetching models:', error);
                return { error: { code: 'MODELS_API_ERROR', message: error.message } };
            }
        }

        // Try AI Foundry first if configured, fall back to Management API
        const useAIFoundry = process.env.AZURE_USE_AI_FOUNDRY === 'true';
        
        if (useAIFoundry) {
            return this.listDeploymentsViaAIFoundry();
        } else {
            return this.listDeploymentsViaManagementAPI();
        }
    }

    /**
     * List deployments using Azure Management API (recommended for existing Cognitive Services)
     * This matches what Azure CLI uses: az cognitiveservices account deployment list
     */
    async listDeploymentsViaManagementAPI() {
        const resourceName = process.env.AZURE_RESOURCE_NAME || "vertesia-test-resource";
        const resourceGroup = process.env.AZURE_RESOURCE_GROUP || "composable";
        const subscriptionId = process.env.AZURE_SUBSCRIPTION_ID;
        
        if (!subscriptionId) {
            throw new Error("AZURE_SUBSCRIPTION_ID environment variable is required");
        }
        
        const deploymentsUrl = `https://management.azure.com/subscriptions/${subscriptionId}/resourceGroups/${resourceGroup}/providers/Microsoft.CognitiveServices/accounts/${resourceName}/deployments?api-version=2023-05-01`;

        console.log(`Management API URL: ${deploymentsUrl}`);

        try {
            const headers: Record<string, string> = {
                'Content-Type': 'application/json',
            };

            // Use Azure Management API authentication (same as Azure CLI)
            const tokenProvider = this.getDefaultManagementAuth();
            const token = await tokenProvider();
            if (!token) {
                throw new Error("Failed to obtain Azure AD token for Azure Management API");
            }
            headers['Authorization'] = `Bearer ${token}`;
            console.log("Using Azure Management API authentication");

            const response = await fetch(deploymentsUrl, { headers });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
            }

            const deployments = await response.json();
            return deployments;
        } catch (error: any) {
            console.error('Error fetching deployments via Management API:', error);
            return { error: { code: 'MANAGEMENT_API_ERROR', message: error.message } };
        }
    }

    /**
     * List deployments using Azure AI Foundry API (for new AI Studio projects)
     */
    async listDeploymentsViaAIFoundry() {
        const hubName = process.env.AZURE_AI_HUB_NAME;
        const projectName = process.env.AZURE_AI_PROJECT_NAME || "vertesia-test";
        
        if (!hubName) {
            throw new Error("AZURE_AI_HUB_NAME environment variable is required for AI Foundry");
        }

        // AI Foundry uses ML workspace endpoints
        const deploymentsUrl = `https://${hubName}.api.azureml.ms/rp/workspaces/${projectName}/deployments?api-version=2024-04-01`;

        console.log(`AI Foundry URL: ${deploymentsUrl}`);

        try {
            const headers: Record<string, string> = {
                'Content-Type': 'application/json',
            };

            // Use AI Foundry authentication
            const tokenProvider = this.getDefaultAIFoundryAuth();
            const token = await tokenProvider();
            if (!token) {
                throw new Error("Failed to obtain Azure AD token for AI Foundry");
            }
            headers['Authorization'] = `Bearer ${token}`;
            console.log("Using AI Foundry authentication");

            const response = await fetch(deploymentsUrl, { headers });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
            }

            const deployments = await response.json();
            return deployments;
        } catch (error: any) {
            console.error('Error fetching deployments via AI Foundry:', error);
            return { error: { code: 'AI_FOUNDRY_ERROR', message: error.message } };
        }
    }
}