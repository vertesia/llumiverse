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
import { withoutKeyRedirects } from '../azure/key-redirect.js';
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
    /** The v1 API is version-less; set for symmetry with the legacy `AzureOpenAI` client. */
    readonly apiVersion: undefined = undefined;

    constructor(options: AzureOpenAIV1Options) {
        super(options);
        const deployment = options.azureDeployment;
        this.deploymentName = deployment;
        if (deployment) {
            // Multipart uploads are encoded before `buildRequest`, which then cannot change their model, so the
            // deployment is set on the upload methods' parameters instead.
            const transcriptions = this.audio.transcriptions;
            const translations = this.audio.translations;
            transcriptions.create = withModel(transcriptions.create.bind(transcriptions), deployment);
            translations.create = withModel(translations.create.bind(translations), deployment);
            this.images.edit = withModel(this.images.edit.bind(this.images), deployment);
        }
    }

    override async buildRequest(...args: Parameters<OpenAI['buildRequest']>): ReturnType<OpenAI['buildRequest']> {
        const [options, props] = args;
        const body = options.body;
        const scoped =
            this.deploymentName &&
            options.method === 'post' &&
            DEPLOYMENT_SCOPED_PATHS.has(options.path) &&
            typeof body === 'object' &&
            body !== null &&
            Object.getPrototypeOf(body) === Object.prototype;
        const built = await super.buildRequest(
            scoped ? { ...options, body: { ...body, model: this.deploymentName } } : options,
            props,
        );
        // Fetch forwards custom headers across origins on a redirect, so a request that carries the resource key does
        // not follow one, as on the legacy AzureOpenAI client. https://fetch.spec.whatwg.org/#http-redirect-fetch
        if (built.req.headers.has('api-key')) built.req.redirect = 'manual';
        return built;
    }

    protected override fetchWithAuth(
        ...args: Parameters<OpenAI['fetchWithAuth']>
    ): ReturnType<OpenAI['fetchWithAuth']> {
        const [url, init, ...rest] = args;
        // The same guard for requests sent without `buildRequest`, as on the legacy AzureOpenAI client.
        const guarded = new Headers(init.headers).has('api-key') ? { ...init, redirect: 'manual' as const } : init;
        return super.fetchWithAuth(url, guarded, ...rest);
    }
}

/** A resource method that sends `model` in place of the model of its parameters. */
function withModel<M extends (body: never, ...rest: never[]) => unknown>(method: M, model: string): M {
    const call = method as unknown as (body: object, ...rest: unknown[]) => unknown;
    return ((body: object, ...rest: unknown[]) => call({ ...body, model }, ...rest)) as unknown as M;
}

/** The v1 base URL of an Azure OpenAI or Microsoft Foundry resource endpoint. */
function azureOpenAIV1BaseURL(endpoint: string): string {
    // Trim trailing slashes with a scan: a `/\/+$/` regex backtracks quadratically on a long run of slashes.
    let end = endpoint.length;
    while (end > 0 && endpoint[end - 1] === '/') end--;
    return `${endpoint.slice(0, end).replace(/\/openai(?:\/v1)?$/, '')}/openai/v1`;
}

/** The service an Azure OpenAI driver uses: the v1 client, or the legacy dated-API `AzureOpenAI` client. */
export type AzureOpenAIService = AzureOpenAI | AzureOpenAIV1Client;

/** A dated Azure OpenAI API version, such as `2024-10-21` or `2025-04-01-preview`. */
export type AzureOpenAIDatedApiVersion = `${number}-${string}`;

/** True when `S` is exactly `T`, not a subclass that adds members. */
type IsExactly<S, T> = [S] extends [T] ? ([T] extends [S] ? true : false) : false;

/**
 * Construction arguments for a driver whose service type is `S`: an injected client of that type, or the options that
 * build it. The legacy client needs a dated `apiVersion`, and the v1 client an endpoint and no `apiVersion`. Options
 * build the base classes only, so a subclass type can only be injected. Without a selected type, any options are
 * accepted and `service` is either client.
 */
export type AzureOpenAIDriverArgs<S extends AzureOpenAIService> =
    | S
    | (IsExactly<S, AzureOpenAIService> extends true
          ? AzureOpenAIDriverOptions
          : IsExactly<S, AzureOpenAI> extends true
            ? AzureOpenAIDriverOptions & { apiVersion: AzureOpenAIDatedApiVersion }
            : IsExactly<S, AzureOpenAIV1Client> extends true
              ? AzureOpenAIDriverOptions & { endpoint: `http${string}`; apiVersion?: undefined }
              : never);

/**
 * Azure OpenAI driver. `service` is typed by how the driver was built: an injected `AzureOpenAI` client infers
 * `AzureOpenAIDriver<AzureOpenAI>`, and `new AzureOpenAIDriver<AzureOpenAI>({ apiVersion, ... })` selects the legacy
 * client for a pinned API version. Otherwise it is either client, the v1 client by default.
 */
export class AzureOpenAIDriver<S extends AzureOpenAIService = AzureOpenAIService> extends OpenAIResponsesDriverBase {
    /**
     * The v1 client by default; the legacy AzureOpenAI client when an apiVersion is set or one is injected. Both expose
     * `deploymentName` and `apiVersion` (undefined on the version-less v1 API).
     */
    service: S;
    private imageService?: OpenAI;
    private sourceModel?: string;
    private readonly deploymentName?: string;
    private keyFetch?: typeof fetch;
    readonly provider = Providers.azure_openai;

    // Accepts an injected AzureOpenAI or v1 client, for independent instantiation, or the driver options.
    constructor(serviceOrOpts: AzureOpenAIDriverArgs<S>) {
        if (serviceOrOpts instanceof AzureOpenAI || serviceOrOpts instanceof AzureOpenAIV1Client) {
            super({});
            this.service = serviceOrOpts as S;
            this.deploymentName = serviceOrOpts.deploymentName;
            return;
        }
        const opts: AzureOpenAIDriverOptions = (serviceOrOpts as AzureOpenAIDriverOptions | undefined) ?? {};
        super(opts);
        this.sourceModel = opts.sourceModel;
        this.deploymentName = opts.deployment;
        if (!opts.azureADTokenProvider && !opts.apiKey) {
            opts.azureADTokenProvider = this.getDefaultCognitiveServicesAuth(opts.endpoint);
        }
        const apiKeyHeaders = opts.apiKey ? { 'api-key': opts.apiKey, Authorization: null } : undefined;

        if (opts.endpoint && !opts.apiVersion) {
            // The v1 API is GA, needs no api-version, and is where Azure ships the Responses API.
            // S is the v1 client or the union here: the argument type requires a dated apiVersion for the legacy type.
            this.service = new AzureOpenAIV1Client({
                baseURL: azureOpenAIV1BaseURL(opts.endpoint),
                apiKey: opts.azureADTokenProvider ?? opts.apiKey,
                defaultHeaders: apiKeyHeaders,
                azureDeployment: opts.deployment,
                fetch: this.getDriverFetch(),
                maxRetries: 0,
                timeout: this.getDriverRequestTimeoutMs(),
            }) as S;
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
            }) as S;
        }
        if (opts.endpoint) {
            this.imageService = new AzureOpenAIV1Client({
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

    protected override getDriverFetch(): typeof fetch {
        this.keyFetch ??= withoutKeyRedirects(super.getDriverFetch());
        return this.keyFetch;
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
     * The model whose prompt-cache support applies. A composite ID names its source model last. A configured
     * `sourceModel` describes the configured deployment, whose name is only an alias and may look like another model's
     * ID; it applies to that deployment alone, or to any bare name when no deployment is configured.
     */
    override getPromptCacheSourceModel(model: string): string {
        if (model.includes('::')) return model.split('::').pop() ?? model;
        if (!this.sourceModel) return model;
        return !this.deploymentName || model === this.deploymentName ? this.sourceModel : model;
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
