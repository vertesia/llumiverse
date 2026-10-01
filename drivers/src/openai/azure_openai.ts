import { DefaultAzureCredential, getBearerTokenProvider } from '@azure/identity';
import {
    type AIModel,
    type DriverOptions,
    type ExecutionOptions,
    isEmbeddingModel,
    ModelType,
    Providers,
    resolveModelProfile,
} from '@llumiverse/core';
import OpenAI, { AzureOpenAI } from 'openai';
import { resolveModelListingMetadata } from '../shared/model-listing.js';
import { OpenAIResponsesDriverBase } from './index.js';

export interface AzureOpenAIDriverOptions extends DriverOptions {
    /**
     * The credentials to use to access Azure OpenAI
     */
    azureADTokenProvider?: (options?: unknown) => Promise<string>;

    apiKey?: string;

    endpoint?: string;

    apiVersion?: string;

    deployment?: string;

    /** Source model for deployments whose names do not identify the image family. */
    sourceModel?: string;
}

export class AzureOpenAIDriver extends OpenAIResponsesDriverBase {
    service: AzureOpenAI;
    private imageService?: OpenAI;
    private sourceModel?: string;
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
        this.sourceModel = opts.sourceModel;
        if (!opts.azureADTokenProvider && !opts.apiKey) {
            opts.azureADTokenProvider = this.getDefaultCognitiveServicesAuth();
        }

        this.service = new AzureOpenAI({
            apiKey: opts.apiKey,
            azureADTokenProvider: opts.azureADTokenProvider,
            endpoint: opts.endpoint,
            apiVersion: opts.apiVersion ?? '2024-10-21',
            deployment: opts.deployment,
            fetch: this.getDriverFetch(),
            maxRetries: 0,
            timeout: this.getDriverRequestTimeoutMs(),
        });
        if (opts.endpoint) {
            this.imageService = new OpenAI({
                baseURL: `${opts.endpoint.replace(/\/$/, '')}/openai/v1`,
                apiKey: opts.azureADTokenProvider ?? opts.apiKey,
                defaultHeaders: opts.apiKey ? { 'api-key': opts.apiKey, Authorization: null } : undefined,
                defaultQuery: { 'api-version': opts.apiVersion ?? 'preview' },
                fetch: this.getDriverFetch(),
                maxRetries: 0,
                timeout: this.getDriverRequestTimeoutMs(),
            });
        }
    }

    getResponsesRequestModel(model: string): string {
        return model.split('::')[0];
    }

    isImageModel(model: string): boolean {
        return resolveModelProfile(this.getImageSourceModel(model), this.provider).family === 'image';
    }

    protected canStream(options: ExecutionOptions): Promise<boolean> {
        if ((options.model_options as { image_generation?: unknown } | undefined)?.image_generation)
            return Promise.resolve(false);
        return super.canStream(options);
    }

    getImageSourceModel(model: string): string {
        if (model.includes('::') || resolveModelProfile(model, this.provider).family !== 'generic') return model;
        return this.sourceModel ?? model;
    }

    getImageService(): OpenAI {
        return this.imageService ?? this.service;
    }

    /**
     * Get default authentication for Azure Cognitive Services API
     */
    getDefaultCognitiveServicesAuth() {
        const scope = 'https://cognitiveservices.azure.com/.default';
        const azureADTokenProvider = getBearerTokenProvider(new DefaultAzureCredential(), scope);
        return azureADTokenProvider;
    }

    async listModels(): Promise<AIModel[]> {
        return this._listModels();
    }

    async _listModels(_filter?: (m: OpenAI.Models.Model) => boolean): Promise<AIModel[]> {
        if (!this.service.deploymentName) {
            throw new Error(
                'A specific deployment is not set. Azure OpenAI cannot list deployments. Update your endpoint URL to include the deployment name, e.g., https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions',
            );
        }

        //Do a test execution to check if the model works and to get the model ID.
        let modelID = this.sourceModel ?? this.service.deploymentName;
        if (!this.isImageModel(modelID))
            try {
                const testResponse = await this.service.chat.completions.create({
                    model: this.service.deploymentName,
                    messages: [{ role: 'user', content: 'Hi' }],
                    max_tokens: 1,
                });
                modelID = testResponse.model;
            } catch (error) {
                this.logger.error({ error }, 'Failed to test model for Azure OpenAI listing :');
            }
        const modelMetadata = resolveModelListingMetadata(modelID, this.provider);
        if (modelID.toLowerCase().includes('dall-e') || isEmbeddingModel({ id: modelID }, this.provider)) {
            return [];
        }
        return [
            {
                id: this.isImageModel(modelID) ? `${this.service.deploymentName}::${modelID}` : modelID,
                type: this.isImageModel(modelID) ? ModelType.Image : undefined,
                name: this.service.deploymentName,
                provider: this.provider,
                owner: 'openai',
                ...modelMetadata,
            } satisfies AIModel,
        ];
    }
}
