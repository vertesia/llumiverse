import { DefaultAzureCredential, getBearerTokenProvider } from '@azure/identity';
import {
    type AIModel,
    type ExecutionOptions,
    isEmbeddingModel,
    ModelType,
    Providers,
    resolveModelProfile,
} from '@llumiverse/core';
import OpenAI, { AzureOpenAI, type ClientOptions } from 'openai';
import { cognitiveServicesScope } from '../azure/entra-scope.js';
import { resolveModelListingMetadata } from '../shared/model-listing.js';
import { OpenAIResponsesDriverBase } from './index.js';

export type { AzureOpenAIDriverOptions } from '../driver-options.js';

import type { AzureOpenAIDriverOptions } from '../driver-options.js';

/** Endpoints that the legacy AzureOpenAI client sends to its configured deployment, whatever the request's model. */
const DEPLOYMENT_SCOPED_PATHS = new Set([
    '/completions',
    '/chat/completions',
    '/embeddings',
    '/audio/transcriptions',
    '/audio/translations',
    '/audio/speech',
    '/images/generations',
    '/images/edits',
]);

interface AzureOpenAIV1Options extends ClientOptions {
    /** Deployment to send as the model of deployment-scoped requests, as the legacy client did. */
    azureDeployment?: string;
}

/**
 * OpenAI client for the Azure OpenAI v1 API (`<endpoint>/openai/v1`), which takes no api-version and names the
 * deployment in the request's `model`. A configured deployment keeps the legacy client's behaviour: it is the model
 * of every deployment-scoped request (Chat Completions, embeddings, audio, images). Responses requests keep their
 * model, as they did on the legacy client.
 * https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle
 */
export class AzureOpenAIV1Client extends OpenAI {
    readonly deploymentName: string | undefined;

    constructor(options: AzureOpenAIV1Options) {
        super(options);
        this.deploymentName = options.azureDeployment;
    }

    override buildRequest(...args: Parameters<OpenAI['buildRequest']>): ReturnType<OpenAI['buildRequest']> {
        const [options, props] = args;
        const body = options.body;
        if (
            this.deploymentName &&
            options.method === 'post' &&
            DEPLOYMENT_SCOPED_PATHS.has(options.path) &&
            typeof body === 'object' &&
            body !== null &&
            Object.getPrototypeOf(body) === Object.prototype
        ) {
            return super.buildRequest({ ...options, body: { ...body, model: this.deploymentName } }, props);
        }
        return super.buildRequest(options, props);
    }
}

/** The v1 base URL of an Azure OpenAI or Microsoft Foundry resource endpoint. */
function azureOpenAIV1BaseURL(endpoint: string): string {
    return `${endpoint.replace(/\/+$/, '').replace(/\/openai(?:\/v1)?$/, '')}/openai/v1`;
}

export class AzureOpenAIDriver extends OpenAIResponsesDriverBase {
    /** The v1 client by default; the legacy AzureOpenAI client when an apiVersion is set or one is injected. */
    service: OpenAI;
    private imageService?: OpenAI;
    private sourceModel?: string;
    private readonly deploymentName?: string;
    readonly provider = Providers.azure_openai;

    //Overload to allow independent instantiation with AzureOpenAI service
    constructor(serviceOrOpts: AzureOpenAI | AzureOpenAIDriverOptions) {
        if (serviceOrOpts instanceof AzureOpenAI) {
            super({});
            this.service = serviceOrOpts;
            this.deploymentName = serviceOrOpts.deploymentName;
            return;
        }
        const opts = serviceOrOpts ?? {};
        super(opts);
        this.sourceModel = opts.sourceModel;
        this.deploymentName = opts.deployment;
        if (!opts.azureADTokenProvider && !opts.apiKey) {
            opts.azureADTokenProvider = this.getDefaultCognitiveServicesAuth(opts.endpoint);
        }
        const apiKeyHeaders = opts.apiKey ? { 'api-key': opts.apiKey, Authorization: null } : undefined;

        if (opts.endpoint && !opts.apiVersion) {
            // The v1 API is GA, needs no api-version, and is where Azure ships the Responses API.
            this.service = new AzureOpenAIV1Client({
                baseURL: azureOpenAIV1BaseURL(opts.endpoint),
                apiKey: opts.azureADTokenProvider ?? opts.apiKey,
                defaultHeaders: apiKeyHeaders,
                azureDeployment: opts.deployment,
                fetch: this.getDriverFetch(),
                maxRetries: 0,
                timeout: this.getDriverRequestTimeoutMs(),
            });
        } else {
            // An explicit api-version keeps the dated `/openai/...?api-version=` API. Without an endpoint, the
            // legacy client reads AZURE_OPENAI_ENDPOINT as it always has.
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
        }
        if (opts.endpoint) {
            this.imageService = new OpenAI({
                baseURL: azureOpenAIV1BaseURL(opts.endpoint),
                apiKey: opts.azureADTokenProvider ?? opts.apiKey,
                defaultHeaders: apiKeyHeaders,
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

    getResponsesService(options: ExecutionOptions): OpenAI {
        const imageTool = (options.model_options as { image_generation?: unknown } | undefined)?.image_generation;
        return imageTool ? this.getImageService() : this.service;
    }

    /**
     * Get default authentication for Azure Cognitive Services API
     */
    getDefaultCognitiveServicesAuth(endpoint?: string) {
        const azureADTokenProvider = getBearerTokenProvider(
            new DefaultAzureCredential(),
            cognitiveServicesScope(endpoint),
        );
        return azureADTokenProvider;
    }

    async listModels(): Promise<AIModel[]> {
        return this._listModels();
    }

    async _listModels(_filter?: (m: OpenAI.Models.Model) => boolean): Promise<AIModel[]> {
        const deploymentName = this.deploymentName;
        if (!deploymentName) {
            throw new Error(
                'A specific deployment is not set. Azure OpenAI cannot list deployments. Set the `deployment` option to the deployment name.',
            );
        }

        //Do a test execution to check if the model works and to get the model ID.
        let modelID = this.sourceModel ?? deploymentName;
        if (!this.isImageModel(modelID))
            try {
                const testResponse = await this.service.chat.completions.create({
                    model: deploymentName,
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
                id: this.isImageModel(modelID) ? `${deploymentName}::${modelID}` : modelID,
                type: this.isImageModel(modelID) ? ModelType.Image : undefined,
                name: deploymentName,
                provider: this.provider,
                owner: 'openai',
                ...modelMetadata,
            } satisfies AIModel,
        ];
    }
}
