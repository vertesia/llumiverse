import { AIProjectClient, type DeploymentUnion, type ModelDeployment } from '@azure/ai-projects';
import { createSseStream, type NodeJSReadableStream } from '@azure/core-sse';
import { DefaultAzureCredential, getBearerTokenProvider, type TokenCredential } from '@azure/identity';
import type {
    ModelClient as AzureInferenceClient,
    ChatCompletionsOutput,
    ChatCompletionsResponseFormat,
    ChatCompletionsToolCall,
    ChatCompletionsToolDefinition,
    ChatRequestMessage,
    GetChatCompletionsParameters,
} from '@azure-rest/ai-inference';
import ModelClient, { isUnexpected } from '@azure-rest/ai-inference';
import {
    type AIModel,
    type Completion,
    type CompletionChunkObject,
    type DriverOptions,
    dataSourceToBase64,
    type EmbeddingResultItem,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    getModelCapabilities,
    type ImageEmbeddingInput,
    LlumiverseError,
    type LlumiverseErrorContext,
    modelModalitiesToArray,
    normalizeEmbeddingsOptions,
    Providers,
    type TextEmbeddingInput,
} from '@llumiverse/core';
import { AbstractDriver } from '@llumiverse/core/driver';
import type OpenAI from 'openai';
import { OpenAIResponsesDriverBase } from '../openai/index.js';
import {
    type OpenAIChatCompletionsContentPart,
    OpenAIChatCompletionsDriverBase,
    type OpenAIChatCompletionsDriverOptions,
    type OpenAIChatCompletionsPayload,
    type OpenAIChatCompletionsPrompt,
    type OpenAIChatCompletionsResponse,
    type OpenAIChatCompletionsStreamResponse,
    openAIChatCompletionsStreamToSSE,
    preserveOpenAIChatCompletionsOriginalResponse,
} from '../openai/openai_chat_completions.js';
import {
    convertResponseItemsToChatMessages,
    formatOpenAIDebugPrompt,
    formatOpenAILikeMultimodalPrompt,
} from '../openai/openai_format.js';

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;
type SSEMessage = { data?: string };

class AzureFoundryHTTPError extends Error {
    readonly status: number;
    readonly body: unknown;

    constructor(message: string, status: string, body?: unknown) {
        super(message);
        this.name = 'AzureFoundryHTTPError';
        this.status = Number(status);
        this.body = body;
    }
}

class AzureFoundryOpenAIProtocolDriver extends OpenAIResponsesDriverBase {
    service: OpenAI;
    readonly provider = Providers.azure_foundry;

    constructor(service: OpenAI) {
        super({});
        this.service = service;
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
    readonly service: AzureInferenceClient;

    constructor(service: AzureInferenceClient, options: DriverOptions) {
        super({ ...options, resultSchemaMode: 'response_format', toolSchemaMode: 'compatible' });
        this.service = service;
    }

    async _postChatCompletion(payload: OpenAIChatCompletionsPayload): Promise<OpenAIChatCompletionsResponse> {
        const response = await this.service.path('/chat/completions').post({
            body: toAzureInferenceRequest(payload, false),
        });
        if (response.status !== '200') {
            throw new AzureFoundryHTTPError(
                `Chat completion request failed with status ${response.status}: ${JSON.stringify(response.body)}`,
                response.status,
                response.body,
            );
        }
        const original = response.body as ChatCompletionsOutput;
        return preserveOpenAIChatCompletionsOriginalResponse(normalizeAzureInferenceResponse(original), original);
    }

    async _postChatCompletionStream(payload: OpenAIChatCompletionsPayload): Promise<ReadableStream> {
        const response = await this.service
            .path('/chat/completions')
            .post({ body: toAzureInferenceRequest(payload, true) })
            .asNodeStream();
        const stream = response.body as NodeJSReadableStream;
        if (!stream) {
            throw new Error('The Azure Foundry response stream is undefined');
        }
        if (response.status !== '200') {
            stream.destroy();
            throw new AzureFoundryHTTPError(
                `Failed to get chat completions, HTTP operation failed with ${response.status} code`,
                response.status,
            );
        }
        return openAIChatCompletionsStreamToSSE(normalizeAzureInferenceStream(createSseStream(stream)));
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

export interface AzureFoundryDriverOptions extends DriverOptions {
    /**
     * The credentials to use to access Azure AI Foundry
     */
    azureADTokenProvider?: TokenCredential;

    endpoint?: string;

    apiVersion?: string;
}

export interface AzureFoundryInferencePrompt {
    messages: ChatRequestMessage[];
}

export interface AzureFoundryOpenAIPrompt {
    messages: ResponseInputItem[];
}

export type AzureFoundryPrompt = AzureFoundryInferencePrompt | AzureFoundryOpenAIPrompt;

export class AzureFoundryDriver extends AbstractDriver<AzureFoundryDriverOptions, ResponseInputItem[]> {
    service: AIProjectClient;
    private readonly inferenceClient: AzureInferenceClient;
    private readonly inferenceProtocolDriver: AzureFoundryInferenceProtocolDriver;
    private openAIProtocolDriver?: AzureFoundryOpenAIProtocolDriver;
    private readonly deploymentProtocols = new Map<string, 'responses' | 'chat_completions'>();
    readonly provider = Providers.azure_foundry;

    OPENAI_API_VERSION = '2025-01-01-preview';
    INFERENCE_API_VERSION = '2024-05-01-preview';

    constructor(opts: AzureFoundryDriverOptions) {
        super(opts);

        this.formatPrompt = formatOpenAILikeMultimodalPrompt;

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
            this.INFERENCE_API_VERSION = opts.apiVersion;
            this.logger.info(`[Azure Foundry] Overriding default API version, using API version: ${opts.apiVersion}`);
        }

        this.service = new AIProjectClient(opts.endpoint, opts.azureADTokenProvider);
        this.inferenceClient = ModelClient(opts.endpoint, opts.azureADTokenProvider, {
            apiVersion: this.INFERENCE_API_VERSION,
        });
        this.inferenceProtocolDriver = new AzureFoundryInferenceProtocolDriver(this.inferenceClient, opts);
    }

    /**
     * Get default authentication for Azure AI Foundry API
     */
    getDefaultAIFoundryAuth() {
        const scope = 'https://ai.azure.com/.default';
        const azureADTokenProvider = getBearerTokenProvider(new DefaultAzureCredential(), scope);
        return azureADTokenProvider;
    }

    async isOpenAIDeployment(model: string): Promise<boolean> {
        const { deploymentName } = parseAzureFoundryModelId(model);
        const cached = this.deploymentProtocols.get(deploymentName);
        if (cached) {
            return cached === 'responses';
        }
        const deployment = (await this.service.deployments.get(deploymentName)) as ModelDeployment;
        const protocol = deployment.modelPublisher === 'OpenAI' ? 'responses' : 'chat_completions';
        this.deploymentProtocols.set(deploymentName, protocol);
        this.logger.debug(`[Azure Foundry] Deployment ${deploymentName} uses ${protocol}`);
        return protocol === 'responses';
    }

    protected canStream(_options: ExecutionOptions): Promise<boolean> {
        return Promise.resolve(true);
    }

    public formatDebugPrompt(prompt: ResponseInputItem[]): ResponseInputItem[] {
        return formatOpenAIDebugPrompt(prompt);
    }

    async requestTextCompletion(prompt: ResponseInputItem[], options: ExecutionOptions): Promise<Completion> {
        const { deploymentName } = parseAzureFoundryModelId(options.model);
        const isOpenAI = await this.isOpenAIDeployment(options.model);

        if (isOpenAI) {
            this.openAIProtocolDriver ??= new AzureFoundryOpenAIProtocolDriver(this.service.getOpenAIClient());
            return this.openAIProtocolDriver.requestTextCompletion(prompt, options);
        }
        const chatPrompt = toAzureFoundryChatPrompt(prompt);
        return this.inferenceProtocolDriver.requestTextCompletion(
            chatPrompt,
            toAzureFoundryChatOptions(options, deploymentName),
        );
    }

    async requestTextCompletionStream(
        prompt: ResponseInputItem[],
        options: ExecutionOptions,
    ): Promise<AsyncIterable<CompletionChunkObject>> {
        const { deploymentName } = parseAzureFoundryModelId(options.model);
        const isOpenAI = await this.isOpenAIDeployment(options.model);

        if (isOpenAI) {
            this.openAIProtocolDriver ??= new AzureFoundryOpenAIProtocolDriver(this.service.getOpenAIClient());
            return this.openAIProtocolDriver.requestTextCompletionStream(prompt, options);
        }
        const chatPrompt = toAzureFoundryChatPrompt(prompt);
        return this.inferenceProtocolDriver.requestTextCompletionStream(
            chatPrompt,
            toAzureFoundryChatOptions(options, deploymentName),
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
            return this.inferenceProtocolDriver.buildStreamingConversation(
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
            this.inferenceProtocolDriver.validateResult(result, toAzureFoundryChatOptions(options, deploymentName));
            return;
        }
        super.validateResult(result, options);
    }

    formatLlumiverseError(error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        const { deploymentName } = parseAzureFoundryModelId(context.model);
        const protocol = this.deploymentProtocols.get(deploymentName);
        if (protocol === 'responses' && this.openAIProtocolDriver) {
            return this.openAIProtocolDriver.formatLlumiverseError(error, context);
        }
        if (protocol === 'chat_completions') {
            return this.inferenceProtocolDriver.formatLlumiverseError(error, { ...context, model: deploymentName });
        }
        return super.formatLlumiverseError(error, context);
    }

    async validateConnection(): Promise<boolean> {
        try {
            // Test the AI Projects client by listing deployments
            const deploymentsIterable = this.service.deployments.list();
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
            const embeddingsClient = this.inferenceClient.path('/embeddings');
            const response = await embeddingsClient.post({ body: { input, model: deploymentName } });
            if (isUnexpected(response)) {
                throw new AzureFoundryHTTPError(
                    `${kind} embeddings request failed: ${response.status} ${response.body?.error?.message || 'Unknown error'}`,
                    response.status,
                    response.body,
                );
            }

            const data = response.body.data;
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
            throw this.formatLlumiverseError(error, {
                provider: this.provider,
                model,
                operation: 'execute',
            });
        }
    }

    async listModels(): Promise<AIModel[]> {
        const filter = (m: ModelDeployment) => {
            // Only include models that support chat completions
            return !!m.capabilities.chat_completion;
        };
        return this._listModels(filter);
    }

    async _listModels(filter?: (m: ModelDeployment) => boolean): Promise<AIModel[]> {
        let deploymentsIterable: ReturnType<typeof this.service.deployments.list>;
        try {
            // List all deployments in the Azure AI Foundry project
            deploymentsIterable = this.service.deployments.list();
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

                const modelCapability = getModelCapabilities(model.modelName, Providers.azure_foundry);
                return {
                    id: compositeId,
                    name: model.name,
                    description: `${model.modelName} - ${model.modelVersion}`,
                    version: model.modelVersion,
                    provider: this.provider,
                    owner: model.modelPublisher,
                    input_modalities: modelModalitiesToArray(modelCapability.input),
                    output_modalities: modelModalitiesToArray(modelCapability.output),
                    tool_support: modelCapability.tool_support,
                } satisfies AIModel<string>;
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

function toAzureInferenceRequest(
    payload: OpenAIChatCompletionsPayload,
    stream: boolean,
): GetChatCompletionsParameters['body'] {
    const responseFormat = payload.response_format
        ? ({ ...payload.response_format } satisfies ChatCompletionsResponseFormat)
        : undefined;
    const tools = payload.tools?.flatMap((tool): ChatCompletionsToolDefinition[] =>
        tool.type === 'function'
            ? [
                  {
                      type: 'function',
                      function: {
                          name: tool.function.name,
                          description: tool.function.description,
                          parameters: tool.function.parameters,
                      },
                  },
              ]
            : [],
    );
    return {
        model: payload.model,
        messages: payload.messages.map(toAzureInferenceMessage),
        max_tokens: payload.max_tokens ?? undefined,
        temperature: payload.temperature ?? undefined,
        top_p: payload.top_p ?? undefined,
        frequency_penalty: payload.frequency_penalty ?? undefined,
        presence_penalty: payload.presence_penalty ?? undefined,
        stop: Array.isArray(payload.stop) ? payload.stop : payload.stop ? [payload.stop] : undefined,
        response_format: responseFormat,
        tools,
        stream,
    } satisfies GetChatCompletionsParameters['body'];
}

function toAzureInferenceMessage(message: OpenAIChatCompletionsPayload['messages'][number]): ChatRequestMessage {
    const textContent = typeof message.content === 'string' || message.content === null ? message.content : undefined;
    switch (message.role) {
        case 'system':
        case 'developer':
            return { role: message.role, content: textContent ?? '' };
        case 'assistant':
            return {
                role: 'assistant',
                content: textContent ?? undefined,
                tool_calls: message.tool_calls?.map((toolCall) => ({
                    id: toolCall.id,
                    type: 'function',
                    function: {
                        name: toolCall.function.name,
                        arguments: toolCall.function.arguments,
                    },
                })),
            };
        case 'tool':
            return { role: 'tool', content: textContent ?? '', tool_call_id: message.tool_call_id ?? '' };
        default:
            return {
                role: 'user',
                content:
                    typeof message.content === 'string'
                        ? message.content
                        : (message.content?.map((part) =>
                              part.type === 'text'
                                  ? { type: 'text' as const, text: part.text }
                                  : { type: 'image_url' as const, image_url: part.image_url },
                          ) ?? ''),
            };
    }
}

function normalizeAzureInferenceResponse(response: ChatCompletionsOutput): OpenAIChatCompletionsResponse {
    return {
        id: response.id,
        object: 'chat.completion',
        created: response.created,
        model: response.model,
        choices: response.choices.map((choice) => ({
            index: choice.index,
            finish_reason: choice.finish_reason,
            message: {
                role: 'assistant',
                content: choice.message.content,
                tool_calls: choice.message.tool_calls?.map((toolCall) => ({
                    id: toolCall.id,
                    type: 'function',
                    function: toolCall.function,
                })),
            },
        })),
        usage: response.usage,
    };
}

type AzureInferenceStreamChunk = Pick<ChatCompletionsOutput, 'id' | 'created' | 'model'> & {
    choices: Array<{
        index: number;
        delta: {
            role?: string;
            content?: string | null;
            tool_calls?: Array<ChatCompletionsToolCall & { index?: number }>;
        };
        finish_reason?: string | null;
    }>;
    usage?: ChatCompletionsOutput['usage'];
};

async function* normalizeAzureInferenceStream(
    stream: AsyncIterable<SSEMessage>,
): AsyncIterable<OpenAIChatCompletionsStreamResponse> {
    for await (const event of stream) {
        if (!event.data || event.data === '[DONE]') {
            continue;
        }
        const chunk = JSON.parse(event.data) as AzureInferenceStreamChunk;
        yield {
            id: chunk.id,
            object: 'chat.completion.chunk',
            created: chunk.created,
            model: chunk.model,
            choices: chunk.choices.map((choice) => ({
                index: choice.index,
                finish_reason: choice.finish_reason,
                delta: {
                    role: choice.delta.role,
                    content: choice.delta.content,
                    tool_calls: choice.delta.tool_calls?.map((toolCall) => ({
                        index: toolCall.index,
                        id: toolCall.id,
                        type: toolCall.type,
                        function: toolCall.function,
                    })),
                },
            })),
            usage: chunk.usage,
        };
    }
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
