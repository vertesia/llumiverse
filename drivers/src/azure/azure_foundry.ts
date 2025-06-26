import { DefaultAzureCredential, getBearerTokenProvider } from "@azure/identity";
import { AbstractDriver, AIModel, Completion, CompletionChunk, DriverOptions, EmbeddingsOptions, EmbeddingsResult, ExecutionOptions, getModelCapabilities, modelModalitiesToArray, Providers } from "@llumiverse/core";
import { AIProjectClient, Deployment, DeploymentUnion, ModelDeployment } from '@azure/ai-projects';

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

export interface AzureFoundryPrompt {
    prompt: string;
}

export class AzureFoundryDriver extends AbstractDriver<AzureFoundryDriverOptions, AzureFoundryPrompt> {
    requestTextCompletion(_prompt: AzureFoundryPrompt, _options: ExecutionOptions): Promise<Completion> {
        throw new Error("Method not implemented.");
    }
    requestTextCompletionStream(_prompt: AzureFoundryPrompt, _options: ExecutionOptions): Promise<AsyncIterable<CompletionChunk>> {
        throw new Error("Method not implemented.");
    }
    validateConnection(): Promise<boolean> {
        throw new Error("Method not implemented.");
    }
    generateEmbeddings(_options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        throw new Error("Method not implemented.");
    }

    service: AIProjectClient;
    readonly provider = Providers.azure_foundry;

    constructor(opts: AzureFoundryDriverOptions) {
        super(opts);

        if (!opts.azureADTokenProvider && !opts.apiKey) {
            opts.azureADTokenProvider = this.getDefaultCognitiveServicesAuth();
        }

        this.service = new AIProjectClient(opts.endpoint ?? "", new DefaultAzureCredential());
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

    async _listModels(filter?: (m: any) => boolean): Promise<AIModel[]> {
        console.log(this.service.getEndpointUrl());

        const deploymentsIterable = this.service.deployments.list();
        const deployments: DeploymentUnion[] = [];
        for await (const deployment of deploymentsIterable) {
            deployments.push(deployment);
        }

        const filteredDeployments: Deployment[] = deployments.filter((d) => {
            if (d.type !== "ModelDeployment") {
                return false;
            }
            return true;
        });

        const models = filter ? filteredDeployments.filter(filter) : deployments;
        const aiModels = models.map((m) => {
            const model = m as ModelDeployment;
            const modelCapability = getModelCapabilities(model.modelName, "openai");
            return {
                id: model.name,
                name: model.name,
                description: model.modelName + " - " + model.modelVersion,
                version: model.modelVersion,
                provider: this.provider,
                owner: model.modelPublisher,
                input_modalities: modelModalitiesToArray(modelCapability.input),
                output_modalities: modelModalitiesToArray(modelCapability.output),
                tool_support: modelCapability.tool_support,
            } satisfies AIModel<string>;
        }).sort((a, b) => a.id.localeCompare(b.id));

        return aiModels;
    }
}
