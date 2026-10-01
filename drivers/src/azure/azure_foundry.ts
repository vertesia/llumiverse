import { AIProjectClient, type DeploymentUnion, type ModelDeployment } from '@azure/ai-projects';
import { DefaultAzureCredential, getBearerTokenProvider } from '@azure/identity';
import {
    type AIModel,
    type Completion,
    type CompletionStream,
    type DriverCompletionStream,
    type DriverOptions,
    dataSourceToBase64,
    type EmbeddingResultItem,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    type ExecutionResponse,
    type ImageEmbeddingInput,
    LlumiverseError,
    type LlumiverseErrorContext,
    ModelType,
    normalizeEmbeddingsOptions,
    type PromptSegment,
    Providers,
    resolveModelProfile,
    type TextEmbeddingInput,
} from '@llumiverse/core';
import { AbstractDriver } from '@llumiverse/core/driver';
import OpenAI from 'openai';
import type { AzureFoundryDriverOptions } from '../driver-options.js';
import { openAIAudioTask } from '../openai/audio.js';
import { OpenAIResponsesDriverBase } from '../openai/index.js';
import {
    normalizeOpenAIChatCompletionsResponse,
    normalizeOpenAIChatCompletionsStream,
    type OpenAIChatCompletionsContentPart,
    OpenAIChatCompletionsDriverBase,
    type OpenAIChatCompletionsDriverOptions,
    type OpenAIChatCompletionsPayload,
    type OpenAIChatCompletionsPrompt,
    type OpenAIChatCompletionsResponse,
    openAIChatCompletionsStreamToSSE,
    preserveOpenAIChatCompletionsOriginalResponse,
    toOpenAINonStreamingPayload,
    toOpenAIStreamingPayload,
} from '../openai/openai_chat_completions.js';
import {
    convertResponseItemsToChatMessages,
    formatOpenAIDebugPrompt,
    formatOpenAILikeMultimodalPrompt,
} from '../openai/openai_format.js';
import { resolveModelListingMetadata } from '../shared/model-listing.js';

export type { AzureFoundryDriverOptions } from '../driver-options.js';

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;

function resourceInferenceURL(baseURL: string): string {
    const url = new URL(baseURL);
    // Images and embeddings use the resource endpoint, rather than the project proxy.
    url.pathname = url.pathname.replace(/\/api\/projects\/[^/]+\/openai\/v1\/?$/, '/openai/v1');
    return url.toString();
}
class AzureFoundryOpenAIProtocolDriver extends OpenAIResponsesDriverBase {
    service: OpenAI;
    readonly provider = Providers.azure_foundry;

    constructor(
        service: OpenAI,
        private readonly foundryOptions: AzureFoundryDriverOptions,
    ) {
        super(foundryOptions);
        this.service = service;
    }

    private imageService?: OpenAI;

    getImageService(): OpenAI {
        this.imageService ??= this.service.withOptions({
            baseURL: resourceInferenceURL(this.service.baseURL),
            fetch: this.getDriverFetch(),
            maxRetries: 0,
        });
        return this.imageService;
    }

    getImageSourceModel(model: string): string {
        if (model.includes('::') || resolveModelProfile(model, this.provider).family !== 'generic') return model;
        return this.foundryOptions.sourceModel ?? model;
    }

    async listModels(): Promise<AIModel[]> {
        return [];
    }

    getResponsesRequestModel(model: string): string {
        return parseAzureFoundryModelId(model).deploymentName;
    }
}

class AzureFoundryInferenceProtocolDriver extends OpenAIChatCompletionsDriverBase<OpenAIChatCompletionsDriverOptions> {
    readonly provider = Providers.azure_foundry;
    readonly service: OpenAI;

    constructor(service: OpenAI, options: DriverOptions) {
        super({ ...options, resultSchemaMode: 'response_format', toolSchemaMode: 'compatible' });
        this.service = service;
    }

    async _postChatCompletion(
        payload: OpenAIChatCompletionsPayload,
        _options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<OpenAIChatCompletionsResponse> {
        const response = await this.service.chat.completions.create(
            toOpenAINonStreamingPayload(payload),
            this.getDriverRequestOptions(_options, signal),
        );
        return preserveOpenAIChatCompletionsOriginalResponse(
            normalizeOpenAIChatCompletionsResponse(response),
            response,
        );
    }

    async _postChatCompletionStream(
        payload: OpenAIChatCompletionsPayload,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ReadableStream> {
        const request = toOpenAIStreamingPayload(payload);
        // Preserve the inference transport contract: some Foundry models reject OpenAI usage options.
        delete request.stream_options;
        const stream = await this.service.chat.completions.create(
            request,
            this.getDriverRequestOptions(options, signal),
        );
        return openAIChatCompletionsStreamToSSE(normalizeOpenAIChatCompletionsStream(stream), () =>
            stream.controller.abort(),
        );
    }

    async listModels(): Promise<AIModel[]> {
        return [];
    }

    async validateConnection(): Promise<boolean> {
        return true;
    }

    async generateEmbeddings(_options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        throw new Error('Azure Foundry embeddings are provided by the parent driver transport.');
    }
}

export interface AzureFoundryInferencePrompt {
    messages: OpenAIChatCompletionsPayload['messages'];
}

export interface AzureFoundryOpenAIPrompt {
    messages: ResponseInputItem[];
}

export type AzureFoundryPrompt = AzureFoundryInferencePrompt | AzureFoundryOpenAIPrompt;

export class AzureFoundryDriver extends AbstractDriver<AzureFoundryDriverOptions, ResponseInputItem[]> {
    service: AIProjectClient;
    private inferenceClient?: OpenAI;
    private resourceClient?: OpenAI;
    private inferenceProtocolDriver?: AzureFoundryInferenceProtocolDriver;
    private openAIProtocolDriver?: AzureFoundryOpenAIProtocolDriver;
    private readonly deploymentProtocols = new Map<string, 'responses' | 'chat_completions'>();
    readonly provider = Providers.azure_foundry;

    override async execute(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ExecutionResponse<ResponseInputItem[]>> {
        if (
            openAIAudioTask(options.model) &&
            (await this.isOpenAIDeployment(options.model, signal, options.httpTimeout))
        ) {
            return this.getOpenAIProtocolDriver().execute(segments, options, signal);
        }
        return super.execute(segments, options, signal);
    }

    override async stream(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CompletionStream<ResponseInputItem[]>> {
        if (
            openAIAudioTask(options.model) &&
            (await this.isOpenAIDeployment(options.model, signal, options.httpTimeout))
        ) {
            return this.getOpenAIProtocolDriver().stream(segments, options, signal);
        }
        return super.stream(segments, options, signal);
    }

    OPENAI_API_VERSION = '2025-01-01-preview';

    constructor(opts: AzureFoundryDriverOptions) {
        super(opts);

        this.formatPrompt = (segments, options) =>
            formatOpenAILikeMultimodalPrompt(segments, {
                ...options,
                imageGeneration: this.isImageModel(options.model),
                result_schema: this.isImageModel(options.model) ? undefined : options.result_schema,
            });

        if (!opts.endpoint) {
            throw new Error('Azure AI Foundry endpoint is required');
        }

        try {
            if (!opts.azureADTokenProvider) {
                // Using Microsoft Entra ID (Azure AD) for authentication
                opts.azureADTokenProvider = new DefaultAzureCredential();
            }
        } catch (error) {
            this.logger.error({ error }, 'Failed to initialize Azure AD token provider:');
            throw new Error('Failed to initialize Azure AD token provider');
        }

        if (opts.apiVersion) {
            this.OPENAI_API_VERSION = opts.apiVersion;
            this.logger.info(`[Azure Foundry] Overriding default API version, using API version: ${opts.apiVersion}`);
        }

        this.service = new AIProjectClient(opts.endpoint, opts.azureADTokenProvider);
    }

    /**
     * Get default authentication for Azure AI Foundry API
     */
    getDefaultAIFoundryAuth() {
        const scope = 'https://ai.azure.com/.default';
        const azureADTokenProvider = getBearerTokenProvider(new DefaultAzureCredential(), scope);
        return azureADTokenProvider;
    }

    async isOpenAIDeployment(
        model: string,
        signal?: AbortSignal,
        httpTimeout?: ExecutionOptions['httpTimeout'],
    ): Promise<boolean> {
        const { deploymentName } = parseAzureFoundryModelId(model);
        const cached = this.deploymentProtocols.get(deploymentName);
        if (cached) {
            return cached === 'responses';
        }
        const deployment = (await this.service.deployments.get(deploymentName, {
            ...(signal ? { abortSignal: signal } : {}),
            requestOptions: { timeout: this.getDriverRequestTimeoutMs(httpTimeout) },
        })) as ModelDeployment;
        const protocol = deployment.modelPublisher.toLowerCase() === 'openai' ? 'responses' : 'chat_completions';
        this.deploymentProtocols.set(deploymentName, protocol);
        this.logger.debug(`[Azure Foundry] Deployment ${deploymentName} uses ${protocol}`);
        return protocol === 'responses';
    }

    protected canStream(_options: ExecutionOptions): Promise<boolean> {
        if ((_options.model_options as { image_generation?: unknown } | undefined)?.image_generation) {
            return Promise.resolve(false);
        }
        if (this.isImageModel(_options.model)) {
            return Promise.resolve(
                this.getOpenAIProtocolDriver().getImageSourceModel(_options.model).toLowerCase().includes('gpt-image'),
            );
        }
        return Promise.resolve(true);
    }

    private getInferenceClient(): OpenAI {
        if (!this.inferenceClient) {
            const configured = this.service.getOpenAIClient();
            // Projects resolves the project URL; inference uses our current SDK independently.
            this.inferenceClient = new OpenAI({
                baseURL: configured.baseURL,
                apiKey: getBearerTokenProvider(
                    this.options.azureADTokenProvider ?? new DefaultAzureCredential(),
                    'https://ai.azure.com/.default',
                ),
                defaultQuery: this.options.apiVersion ? { 'api-version': this.options.apiVersion } : undefined,
                fetch: this.getDriverFetch(),
                timeout: this.getDriverRequestTimeoutMs(),
                maxRetries: 0,
            });
        }
        return this.inferenceClient;
    }

    private getResourceClient(): OpenAI {
        const inference = this.getInferenceClient();
        this.resourceClient ??= inference.withOptions({ baseURL: resourceInferenceURL(inference.baseURL) });
        return this.resourceClient;
    }

    private getOpenAIProtocolDriver(): AzureFoundryOpenAIProtocolDriver {
        this.openAIProtocolDriver ??= new AzureFoundryOpenAIProtocolDriver(this.getInferenceClient(), this.options);
        return this.openAIProtocolDriver;
    }

    private getInferenceProtocolDriver(): AzureFoundryInferenceProtocolDriver {
        this.inferenceProtocolDriver ??= new AzureFoundryInferenceProtocolDriver(
            this.getInferenceClient(),
            this.options,
        );
        return this.inferenceProtocolDriver;
    }

    protected destroyProviderResources(): void {
        this.openAIProtocolDriver?.destroy();
        this.inferenceProtocolDriver?.destroy();
    }

    public formatDebugPrompt(prompt: ResponseInputItem[]): ResponseInputItem[] {
        return formatOpenAIDebugPrompt(prompt);
    }

    protected isImageModel(model: string): boolean {
        const family = resolveModelProfile(model, this.provider).family;
        const source = model.includes('::') || family !== 'generic' ? model : (this.options.sourceModel ?? model);
        return resolveModelProfile(source, this.provider).family === 'image';
    }

    requestImageGeneration(
        prompt: ResponseInputItem[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        return this.getOpenAIProtocolDriver().requestImageGeneration(prompt, options, signal);
    }

    async requestTextCompletion(
        prompt: ResponseInputItem[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        if (this.isImageModel(options.model)) return this.requestImageGeneration(prompt, options, signal);
        const { deploymentName } = parseAzureFoundryModelId(options.model);
        const isOpenAI = await this.isOpenAIDeployment(options.model, signal, options.httpTimeout);

        if (isOpenAI) {
            return this.getOpenAIProtocolDriver().requestTextCompletion(prompt, options, signal);
        }
        const chatPrompt = toAzureFoundryChatPrompt(prompt);
        return this.getInferenceProtocolDriver().requestTextCompletion(
            chatPrompt,
            toAzureFoundryChatOptions(options, deploymentName),
            signal,
        );
    }

    async requestTextCompletionStream(
        prompt: ResponseInputItem[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        if (this.isImageModel(options.model)) {
            return this.getOpenAIProtocolDriver().requestImageStream(prompt, options, signal);
        }
        const { deploymentName } = parseAzureFoundryModelId(options.model);
        const isOpenAI = await this.isOpenAIDeployment(options.model, signal, options.httpTimeout);

        if (isOpenAI) {
            return this.getOpenAIProtocolDriver().requestTextCompletionStream(prompt, options, signal);
        }
        const chatPrompt = toAzureFoundryChatPrompt(prompt);
        return this.getInferenceProtocolDriver().requestTextCompletionStream(
            chatPrompt,
            toAzureFoundryChatOptions(options, deploymentName),
            signal,
        );
    }

    buildStreamingConversation(
        prompt: ResponseInputItem[],
        result: unknown[],
        toolUse: unknown[] | undefined,
        options: ExecutionOptions,
    ): unknown | undefined {
        const { deploymentName } = parseAzureFoundryModelId(options.model);
        const protocol = this.deploymentProtocols.get(deploymentName);
        if (protocol === 'responses' && this.openAIProtocolDriver) {
            return this.openAIProtocolDriver.buildStreamingConversation(prompt, result, toolUse, options);
        }
        if (protocol === 'chat_completions') {
            return this.getInferenceProtocolDriver().buildStreamingConversation(
                toAzureFoundryChatPrompt(prompt),
                result,
                toolUse,
                toAzureFoundryChatOptions(options, deploymentName),
            );
        }
        return undefined;
    }

    validateResult(result: Completion, options: ExecutionOptions): void {
        const { deploymentName } = parseAzureFoundryModelId(options.model);
        const protocol = this.deploymentProtocols.get(deploymentName);
        if (protocol === 'responses' && this.openAIProtocolDriver) {
            this.openAIProtocolDriver.validateResult(result, options);
            return;
        }
        if (protocol === 'chat_completions') {
            this.getInferenceProtocolDriver().validateResult(
                result,
                toAzureFoundryChatOptions(options, deploymentName),
            );
            return;
        }
        super.validateResult(result, options);
    }

    formatLlumiverseError(error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        const { deploymentName } = parseAzureFoundryModelId(context.model);
        const protocol = this.deploymentProtocols.get(deploymentName);
        if ((protocol === 'responses' || this.isImageModel(context.model)) && this.openAIProtocolDriver) {
            return this.openAIProtocolDriver.formatLlumiverseError(error, context);
        }
        if (protocol === 'chat_completions') {
            return this.getInferenceProtocolDriver().formatLlumiverseError(error, {
                ...context,
                model: deploymentName,
            });
        }
        return super.formatLlumiverseError(error, context);
    }

    async validateConnection(): Promise<boolean> {
        try {
            // Test the AI Projects client by listing deployments
            const deploymentsIterable = this.service.deployments.list({
                requestOptions: { timeout: this.getDriverRequestTimeoutMs() },
            });
            let hasDeployments = false;

            for await (const deployment of deploymentsIterable) {
                hasDeployments = true;
                this.logger.debug(`[Azure Foundry] Found deployment: ${deployment.name} (${deployment.type})`);
                break; // Just check if we can get at least one deployment
            }

            if (!hasDeployments) {
                this.logger.warn('[Azure Foundry] No deployments found in the project');
            }

            return true;
        } catch (error) {
            this.logger.error({ error }, 'Azure Foundry connection validation failed:');
            return false;
        }
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        const normalized = normalizeEmbeddingsOptions(options);
        if (!normalized.model) {
            throw new Error(
                'Default embedding model selection not supported for Azure Foundry. Please specify a model.',
            );
        }

        const textInputs: { index: number; input: TextEmbeddingInput }[] = [];
        const imageInputs: { index: number; input: ImageEmbeddingInput }[] = [];
        normalized.inputs.forEach((input, index) => {
            if (input.type === 'text') textInputs.push({ index, input });
            else if (input.type === 'image') imageInputs.push({ index, input });
            else {
                throw new Error(`Provider 'azure_foundry' does not support '${input.type}' embeddings.`);
            }
        });

        const items = new Array<EmbeddingResultItem>(normalized.inputs.length);

        if (textInputs.length > 0) {
            const vectors = await this.callAzureEmbeddings(
                textInputs.map((t) => t.input.text),
                normalized.model,
                'text',
            );
            textInputs.forEach((entry, i) => {
                items[entry.index] = { outputs: [{ values: vectors[i], modality: 'text' }] };
            });
        }

        if (imageInputs.length > 0) {
            const base64Images = await Promise.all(imageInputs.map((entry) => dataSourceToBase64(entry.input.source)));
            const vectors = await this.callAzureEmbeddings(base64Images, normalized.model, 'image');
            imageInputs.forEach((entry, i) => {
                items[entry.index] = { outputs: [{ values: vectors[i], modality: 'image' }] };
            });
        }

        return { model: normalized.model, results: items };
    }

    private async callAzureEmbeddings(input: string[], model: string, kind: 'text' | 'image'): Promise<number[][]> {
        const { deploymentName } = parseAzureFoundryModelId(model);
        try {
            const response = await this.getResourceClient().embeddings.create(
                { input, model: deploymentName, encoding_format: 'float' },
                { timeout: this.getDriverRequestTimeoutMs() },
            );
            const data = response.data;
            if (!Array.isArray(data) || data.length === 0) {
                throw new Error(`No embeddings found in Azure Foundry ${kind} response`);
            }
            const ordered = [...data].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
            return ordered.map((entry) => {
                const embedding = entry.embedding;
                if (!Array.isArray(embedding) || embedding.length === 0) {
                    throw new Error(
                        `Empty or non-array embedding in Azure Foundry ${kind} response (got ${typeof embedding})`,
                    );
                }
                return embedding;
            });
        } catch (error) {
            if (LlumiverseError.isLlumiverseError(error)) throw error;
            this.logger.error({ error }, `Azure Foundry ${kind} embeddings error:`);
            throw this.getOpenAIProtocolDriver().formatLlumiverseError(error, {
                provider: this.provider,
                model,
                operation: 'execute',
            });
        }
    }

    async listModels(): Promise<AIModel[]> {
        return this._listModels(isStandardInferenceDeployment);
    }

    async _listModels(filter?: (m: ModelDeployment) => boolean): Promise<AIModel[]> {
        let deploymentsIterable: ReturnType<typeof this.service.deployments.list>;
        try {
            // List all deployments in the Azure AI Foundry project
            deploymentsIterable = this.service.deployments.list({
                requestOptions: { timeout: this.getDriverRequestTimeoutMs() },
            });
        } catch (error) {
            this.logger.error({ error }, 'Failed to list deployments:');
            throw new Error('Failed to list deployments in Azure AI Foundry project');
        }
        const deployments: DeploymentUnion[] = [];

        for await (const page of deploymentsIterable.byPage()) {
            for (const deployment of page) {
                deployments.push(deployment);
            }
        }

        let modelDeployments: ModelDeployment[] = deployments.filter((d): d is ModelDeployment => {
            return d.type === 'ModelDeployment';
        });

        if (filter) {
            modelDeployments = modelDeployments.filter(filter);
        }

        const aiModels = modelDeployments
            .map((model) => {
                // Create composite ID: deployment_name::base_model
                const compositeId = `${model.name}::${model.modelName}`;

                const modelMetadata = resolveModelListingMetadata(model.modelName, Providers.azure_foundry);
                return {
                    id: compositeId,
                    name: model.name,
                    description: `${model.modelName} - ${model.modelVersion}`,
                    version: model.modelVersion,
                    provider: this.provider,
                    owner: model.modelPublisher,
                    type:
                        resolveModelProfile(model.modelName, this.provider).family === 'image'
                            ? ModelType.Image
                            : ModelType.Text,
                    ...modelMetadata,
                } satisfies AIModel;
            })
            .sort((modelA, modelB) => modelA.id.localeCompare(modelB.id));

        return aiModels;
    }
}

function toAzureFoundryChatPrompt(items: ResponseInputItem[]): OpenAIChatCompletionsPrompt {
    const messages = convertResponseItemsToChatMessages(items).map((message) => {
        switch (message.role) {
            case 'assistant':
                return {
                    role: message.role,
                    content:
                        typeof message.content === 'string' || message.content === null
                            ? message.content
                            : message.content?.flatMap((part) => (part.type === 'text' ? [part.text] : [])).join('') ||
                              null,
                    tool_calls: message.tool_calls?.flatMap((toolCall) =>
                        toolCall.type === 'function' ? [toolCall] : [],
                    ),
                };
            case 'tool':
                return {
                    role: message.role,
                    content: typeof message.content === 'string' ? message.content : '',
                    tool_call_id: message.tool_call_id,
                };
            case 'user':
                return {
                    role: message.role,
                    content:
                        typeof message.content === 'string'
                            ? message.content
                            : message.content.flatMap((part): OpenAIChatCompletionsContentPart[] => {
                                  if (part.type === 'text') {
                                      return [{ type: 'text' as const, text: part.text }];
                                  }
                                  if (part.type === 'image_url') {
                                      return [{ type: 'image_url' as const, image_url: part.image_url }];
                                  }
                                  return [];
                              }),
                };
            default:
                return {
                    role: message.role,
                    content: typeof message.content === 'string' ? message.content : '',
                };
        }
    });
    return { _is_openai_chat_completions: true, messages };
}

function toAzureFoundryChatOptions(options: ExecutionOptions, deploymentName: string): ExecutionOptions {
    return {
        ...options,
        model: deploymentName,
        conversation: Array.isArray(options.conversation)
            ? toAzureFoundryChatPrompt(options.conversation as ResponseInputItem[])
            : options.conversation,
    };
}

function parseCapabilityFlag(value: unknown): boolean | undefined {
    if (typeof value === 'boolean') return value;
    if (typeof value !== 'string') return undefined;
    switch (value.trim().toLowerCase()) {
        case 'true':
        case '1':
        case 'yes':
            return true;
        case 'false':
        case '0':
        case 'no':
            return false;
        default:
            return undefined;
    }
}

function isStandardInferenceDeployment(deployment: ModelDeployment): boolean {
    if (deployment.modelPublisher?.toLowerCase() === 'openai' && openAIAudioTask(deployment.modelName)) return true;
    // Foundry Anthropic deployments require the Messages API rather than this driver's OpenAI transport.
    if (deployment.modelPublisher.toLowerCase() === 'anthropic') return false;
    const profile = resolveModelProfile(deployment.modelName, Providers.azure_foundry);
    const sourceModel = deployment.modelName.toLowerCase();
    if (sourceModel.includes('dall-e')) return false;
    if (profile.family === 'image' && deployment.modelPublisher.toLowerCase() === 'openai') return true;
    // These source families use dedicated endpoint contracts, not Foundry chat or Responses inference.
    if (
        ['embedding', 'image', 'transcription', 'speech', 'realtime', 'video', 'moderation'].includes(profile.family) ||
        /(?:^|[-_.:/])(?:embed|embeddings?|flux|whisper|transcribe|tts|realtime|moderation|sora)(?:[-_.:/]|$)/.test(
            sourceModel,
        )
    ) {
        return false;
    }

    // Foundry capability values are strings. Only an explicit false is deterministic enough to hide a deployment;
    // omitted and future capability values remain visible so the provider listing does not become an allow-list.
    return parseCapabilityFlag(deployment.capabilities.chat_completion) !== false;
}

// Helper functions to parse the composite ID
export function parseAzureFoundryModelId(compositeId: string): { deploymentName: string; baseModel: string } {
    const parts = compositeId.split('::');
    if (parts.length === 2) {
        return {
            deploymentName: parts[0],
            baseModel: parts[1],
        };
    }

    // Backwards compatibility: if no delimiter found, treat as deployment name
    return {
        deploymentName: compositeId,
        baseModel: compositeId,
    };
}

export function isCompositeModelId(modelId: string): boolean {
    return modelId.includes('::');
}
