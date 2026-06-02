import {
    type AIModel,
    type Completion,
    type CompletionChunkObject,
    type CompletionResult,
    type DriverOptions,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    getConversationMeta,
    getModelCapabilities,
    incrementConversationTurn,
    type LlumiverseError,
    type LlumiverseErrorContext,
    type ModelSearchPayload,
    modelModalitiesToArray,
    type PromptSegment,
    stripBase64ImagesFromConversation,
    stripHeartbeatsFromConversation,
    type ToolUse,
    truncateLargeTextInConversation,
} from '@llumiverse/core';
import { AbstractDriver } from '@llumiverse/core/driver';
import { type FETCH_FN, FetchClient, type ServerSentEvent } from '@vertesia/api-fetch-client';
import { GoogleAuth, type GoogleAuthOptions } from 'google-auth-library';
import type { ClaudePrompt } from '../shared/claude-messages.js';
import { generateVertexAiEmbeddings } from './embeddings/embed.js';
import { NON_GLOBAL_ANTHROPIC_MODELS } from './models/claude.js';
import { ImagenModelDefinition, type ImagenPrompt } from './models/imagen.js';
import type { LLamaPrompt } from './models/llama.js';
import { getModelDefinition, trimModelName } from './models.js';
import type { GenerateContentPrompt, VertexContent, VertexListedModel, VertexListModelsResponse } from './types.js';

export type { GenerateContentPrompt } from './types.js';

export interface VertexAIDriverOptions extends DriverOptions {
    project: string;
    region: string;
    googleAuthOptions?: GoogleAuthOptions;
}

type ClaudeStreamingPrompt = { messages: unknown[]; system?: unknown[] };
type ConversationWrapper = { messages?: unknown[]; system?: unknown[] };

function isClaudeStreamingPrompt(prompt: unknown): prompt is ClaudeStreamingPrompt {
    return (
        prompt !== null &&
        typeof prompt === 'object' &&
        'messages' in prompt &&
        Array.isArray((prompt as ConversationWrapper).messages)
    );
}

//General Prompt type for VertexAI
export type VertexAIPrompt = ImagenPrompt | GenerateContentPrompt | ClaudePrompt | LLamaPrompt;

export { trimModelName };

export class VertexAIDriver extends AbstractDriver<VertexAIDriverOptions, VertexAIPrompt> {
    static PROVIDER = 'vertexai';
    provider = VertexAIDriver.PROVIDER;

    googleAuth: GoogleAuth;
    private fetchClients = new Map<string, FetchClient>();

    constructor(options: VertexAIDriverOptions) {
        super(options);

        this.googleAuth = new GoogleAuth(options.googleAuthOptions);
    }

    /**
     * Cleanup HTTP resources when the driver is evicted from the cache.
     * `super.destroy()` releases the HTTP agent socket pool created by
     * {@link AbstractDriver.getHttpAgent} / {@link AbstractDriver.getDriverFetch}.
     */
    destroy(): void {
        super.destroy();
    }

    private async getAuthorizationHeader(): Promise<string> {
        const token = await this.googleAuth.getAccessToken();
        if (!token) {
            throw new Error('Google Application Default Credentials did not return an access token');
        }
        return `Bearer ${token}`;
    }

    public getFetchClient(region: string = this.options.region, apiVersion: string = 'v1'): FetchClient {
        const key = `${apiVersion}:${region}`;
        const existing = this.fetchClients.get(key);
        if (existing) {
            return existing;
        }

        const client = createFetchClient({
            region: region,
            project: this.options.project,
            apiVersion,
            fetchImpl: this.getDriverFetch(),
        }).withAuthCallback(async () => {
            return this.getAuthorizationHeader();
        });

        this.fetchClients.set(key, client);
        return client;
    }

    public getLLamaClient(region: string = 'us-central1'): FetchClient {
        return this.getFetchClient(region, 'v1beta1');
    }

    public resolveVertexModelPath(model: string, regionOverride?: string) {
        let region = regionOverride ?? this.options.region;
        let modelPath = model;

        const splits = model.split('/');
        if (splits[0] === 'locations' && splits.length >= 3) {
            region = regionOverride ?? splits[1];
            modelPath = splits.slice(2).join('/');
        }

        if (!modelPath.includes('publishers/')) {
            const modelName = trimModelName(modelPath.split('/').pop() ?? modelPath);
            const publisher = modelName.includes('claude') || modelName.includes('anthropic') ? 'anthropic' : 'google';
            modelPath = `publishers/${publisher}/models/${modelName}`;
        }

        return { region, modelPath };
    }

    public vertexModelPath(model: string, method: string, regionOverride?: string): { region: string; path: string } {
        const { region, modelPath } = this.resolveVertexModelPath(model, regionOverride);
        return { region, path: `${modelPath}:${method}` };
    }

    public async postVertexModel<T>(
        model: string,
        method: string,
        payload: object,
        options?: {
            region?: string;
            apiVersion?: string;
            headers?: Record<string, string>;
            query?: Record<string, string | number | boolean>;
        },
    ): Promise<T> {
        const endpoint = this.vertexModelPath(model, method, options?.region);
        return this.getFetchClient(endpoint.region, options?.apiVersion).post(endpoint.path, {
            payload,
            headers: options?.headers,
            query: options?.query,
        }) as Promise<T>;
    }

    public async streamVertexModel(
        model: string,
        method: string,
        payload: object,
        options?: {
            region?: string;
            apiVersion?: string;
            headers?: Record<string, string>;
            query?: Record<string, string | number | boolean>;
        },
    ): Promise<ReadableStream<ServerSentEvent>> {
        const endpoint = this.vertexModelPath(model, method, options?.region);
        return this.getFetchClient(endpoint.region, options?.apiVersion).post(endpoint.path, {
            payload,
            headers: options?.headers,
            query: options?.query,
            reader: 'sse',
        }) as Promise<ReadableStream<ServerSentEvent>>;
    }

    validateResult(result: Completion, options: ExecutionOptions) {
        // Optionally preprocess the result before validation
        const modelDef = getModelDefinition(options.model);
        if (typeof modelDef.preValidationProcessing === 'function') {
            const processed = modelDef.preValidationProcessing(result, options);
            result = processed.result;
            options = processed.options;
        }

        super.validateResult(result, options);
    }

    protected canStream(options: ExecutionOptions): Promise<boolean> {
        if (this.isImageModel(options.model)) {
            return Promise.resolve(false);
        }
        return Promise.resolve(getModelDefinition(options.model).model.can_stream === true);
    }

    protected isImageModel(model: string): boolean {
        return model.includes('imagen');
    }

    public createPrompt(segments: PromptSegment[], options: ExecutionOptions): Promise<VertexAIPrompt> {
        if (this.isImageModel(options.model)) {
            return new ImagenModelDefinition(options.model).createPrompt(this, segments, options);
        }
        return getModelDefinition(options.model).createPrompt(this, segments, options);
    }

    async requestTextCompletion(prompt: VertexAIPrompt, options: ExecutionOptions): Promise<Completion> {
        return getModelDefinition(options.model).requestTextCompletion(this, prompt, options);
    }
    async requestTextCompletionStream(
        prompt: VertexAIPrompt,
        options: ExecutionOptions,
    ): Promise<AsyncIterable<CompletionChunkObject>> {
        return getModelDefinition(options.model).requestTextCompletionStream(this, prompt, options);
    }

    /**
     * Build conversation context after streaming completion.
     * Reconstructs the assistant message from accumulated results and applies stripping.
     * Handles both Gemini (Content[]) and Claude (ClaudePrompt) formats.
     */
    buildStreamingConversation(
        prompt: VertexAIPrompt,
        result: unknown[],
        toolUse: unknown[] | undefined,
        options: ExecutionOptions,
    ): VertexContent[] | unknown | undefined {
        // Handle Claude-style prompts (has 'messages' array)
        if (isClaudeStreamingPrompt(prompt)) {
            return this.buildClaudeStreamingConversation(prompt, result, toolUse, options);
        }

        // Only handle Gemini-style prompts with contents array
        if (!('contents' in prompt) || !Array.isArray(prompt.contents)) {
            return undefined;
        }

        const completionResults = result as CompletionResult[];

        // Convert accumulated results to text content for assistant message
        const textContent = completionResults
            .map((r) => {
                switch (r.type) {
                    case 'text':
                        return r.value;
                    case 'json':
                        return typeof r.value === 'string' ? r.value : JSON.stringify(r.value);
                    case 'image':
                        // Skip images in conversation - they're in the result
                        return '';
                    default: {
                        const _exhaustive: never = r;
                        return String(_exhaustive);
                    }
                }
            })
            .join('');

        // Build parts array for assistant message
        const parts: unknown[] = [];
        if (textContent) {
            parts.push({ text: textContent });
        }
        // Add function calls if present (Gemini format)
        if (toolUse && toolUse.length > 0) {
            for (const tool of toolUse as ToolUse<unknown>[]) {
                const functionCallPart: Record<string, unknown> = {
                    functionCall: {
                        name: tool.tool_name,
                        args: tool.tool_input,
                    },
                };
                // Include thought_signature for Gemini thinking models (2.5+/3.0+)
                // This must be preserved in the conversation for subsequent API calls
                if (tool.thought_signature) {
                    functionCallPart.thoughtSignature = tool.thought_signature;
                }
                parts.push(functionCallPart);
            }
        }

        // prompt.contents already includes the conversation history
        // (merged in requestTextCompletionStream via updateConversation),
        // so we use it directly — do NOT prepend options.conversation again.
        let conversation: VertexContent[] = [...prompt.contents];

        // Only add assistant message if there's actual content
        // (Empty text parts can cause API errors)
        if (parts.length > 0) {
            conversation.push({
                role: 'model',
                parts: parts as VertexContent['parts'],
            });
        }

        // Increment turn counter
        conversation = incrementConversationTurn(conversation) as VertexContent[];

        // Apply stripping based on options
        const currentTurn = getConversationMeta(conversation).turnNumber;
        const stripOptions = {
            keepForTurns: options.stripImagesAfterTurns ?? Infinity,
            currentTurn,
            textMaxTokens: options.stripTextMaxTokens,
        };
        let processedConversation = stripBase64ImagesFromConversation(conversation, stripOptions);
        processedConversation = truncateLargeTextInConversation(processedConversation, stripOptions);
        processedConversation = stripHeartbeatsFromConversation(processedConversation, {
            keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
            currentTurn,
        });

        // Preserve system instruction in conversation for Gemini multi-turn support.
        // The Gemini API takes system as a separate parameter (not in contents),
        // so we must store it in the conversation wrapper to survive serialization.
        const geminiPrompt = prompt as GenerateContentPrompt;
        if (geminiPrompt.system) {
            if (typeof processedConversation === 'object' && processedConversation !== null) {
                processedConversation = {
                    ...(processedConversation as object),
                    _llumiverse_system: geminiPrompt.system,
                };
            }
        }

        return processedConversation;
    }

    /**
     * Build conversation for Claude streaming.
     * Creates assistant message with tool_use blocks in Claude's ContentBlock format.
     */
    private buildClaudeStreamingConversation(
        prompt: { messages: unknown[]; system?: unknown[] },
        result: unknown[],
        toolUse: unknown[] | undefined,
        options: ExecutionOptions,
    ): unknown {
        const completionResults = result as CompletionResult[];

        // Convert accumulated results to text content
        const textContent = completionResults
            .map((r) => {
                switch (r.type) {
                    case 'text':
                        return r.value;
                    case 'json':
                        return typeof r.value === 'string' ? r.value : JSON.stringify(r.value);
                    case 'image':
                        return '';
                    default: {
                        const _exhaustive: never = r;
                        return String(_exhaustive);
                    }
                }
            })
            .join('');

        // Build Claude-style ContentBlock array for assistant message
        const content: unknown[] = [];

        // Add text block if there's text content
        if (textContent) {
            content.push({
                type: 'text',
                text: textContent,
            });
        }

        // Add tool_use blocks in Claude format
        if (toolUse && toolUse.length > 0) {
            for (const tool of toolUse as ToolUse<unknown>[]) {
                content.push({
                    type: 'tool_use',
                    id: tool.id,
                    name: tool.tool_name,
                    input: tool.tool_input ?? {},
                });
            }
        }

        // Claude's requestTextCompletionStream does NOT mutate prompt.messages
        // to include history, so we must prepend options.conversation here.
        const existingConversation = options.conversation as ConversationWrapper | undefined;
        const existingMessages = existingConversation?.messages ?? [];
        const existingSystem = existingConversation?.system ?? prompt.system;

        // Build the new messages array
        const newMessages = [...existingMessages, ...prompt.messages];

        // Only add assistant message if there's actual content
        // (Claude API rejects empty text content blocks)
        if (content.length > 0) {
            newMessages.push({
                role: 'assistant',
                content: content,
            });
        }

        // Build the new conversation in ClaudePrompt format
        const conversation = {
            messages: newMessages,
            system: existingSystem,
        };

        // Increment turn counter
        const withTurn = incrementConversationTurn(conversation);

        // Apply stripping based on options
        const currentTurn = getConversationMeta(withTurn).turnNumber;
        const stripOptions = {
            keepForTurns: options.stripImagesAfterTurns ?? Infinity,
            currentTurn,
            textMaxTokens: options.stripTextMaxTokens,
        };
        let processedConversation = stripBase64ImagesFromConversation(withTurn, stripOptions);
        processedConversation = truncateLargeTextInConversation(processedConversation, stripOptions);
        processedConversation = stripHeartbeatsFromConversation(processedConversation, {
            keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
            currentTurn,
        });

        return processedConversation;
    }

    async requestImageGeneration(_prompt: ImagenPrompt, _options: ExecutionOptions): Promise<Completion> {
        const splits = _options.model.split('/');
        const modelName = trimModelName(splits[splits.length - 1]);
        return new ImagenModelDefinition(modelName).requestImageGeneration(this, _prompt, _options);
    }

    private async listVertexModels(path: string, query?: Record<string, string | number | boolean>) {
        const models: VertexListedModel[] = [];
        let pageToken: string | undefined;

        do {
            const response = (await this.getFetchClient().get(path, {
                query: {
                    ...(query ?? {}),
                    ...(pageToken ? { pageToken } : {}),
                },
            })) as VertexListModelsResponse;
            models.push(...(response.publisherModels ?? response.models ?? []));
            pageToken = response.nextPageToken;
        } while (pageToken);

        return models;
    }

    private async listPublisherModels(publisher: string, query?: Record<string, string | number | boolean>) {
        const models: VertexListedModel[] = [];
        let pageToken: string | undefined;
        const client = this.getFetchClient(this.options.region, 'v1beta1');
        const path = `${vertexApiBaseUrl(this.options.region, 'v1beta1')}/publishers/${publisher}/models`;

        try {
            do {
                const response = (await client.get(path, {
                    query: {
                        ...(query ?? {}),
                        ...(pageToken ? { pageToken } : {}),
                    },
                })) as VertexListModelsResponse;
                models.push(...(response.publisherModels ?? response.models ?? []));
                pageToken = response.nextPageToken;
            } while (pageToken);
        } catch (err: unknown) {
            if (isRequestStatus(err, 403) || isRequestStatus(err, 404)) {
                return [];
            }
            throw err;
        }

        return models;
    }

    async listModels(params?: ModelSearchPayload): Promise<AIModel<string>[]> {
        let models: AIModel<string>[] = [];

        //Model Garden Publisher models - Pretrained models
        const publishers = ['google', 'anthropic', 'meta'] as const;
        // Meta "maas" models are LLama Models-As-A-Service. Non-maas models are not pre-deployed.
        const supportedModels = { google: ['gemini', 'imagen'], anthropic: ['claude'], meta: ['maas'] };
        // Additional models not in the listings, but we want to include
        // TODO: Remove once the models are available in the listing API, or no longer needed
        const additionalModels = {
            google: ['imagen-3.0-fast-generate-001'],
            anthropic: [],
            meta: [
                'llama-4-maverick-17b-128e-instruct-maas',
                'llama-4-scout-17b-16e-instruct-maas',
                'llama-3.3-70b-instruct-maas',
                'llama-3.2-90b-vision-instruct-maas',
                'llama-3.1-405b-instruct-maas',
                'llama-3.1-70b-instruct-maas',
                'llama-3.1-8b-instruct-maas',
            ],
        };

        //Used to exclude retired models that are still in the listing API but not available for use.
        //Or models we do not support yet
        const unsupportedModelsByPublisher = {
            google: [
                'gemini-pro',
                'gemini-ultra',
                'imagen-product-recontext-preview',
                'embedding',
                'gemini-live-2.5-flash-preview-native-audio',
                'computer-use-preview',
            ],
            anthropic: [],
            meta: [],
        };

        const publisherPromises = publishers.map(async (publisher) => ({
            publisher,
            response: await this.listPublisherModels(publisher, {
                orderBy: 'name',
                listAllVersions: true,
                pageSize: 100,
            }),
        }));

        const [projectModels, ...publisherResults] = await Promise.all([
            this.listVertexModels('models', { pageSize: 100 }),
            ...publisherPromises,
        ]);

        // Process aiplatform models, project specific models
        models = models.concat(
            projectModels.map((model) => ({
                id: model.name?.split('/').pop() ?? '',
                name: model.displayName ?? '',
                provider: 'vertexai',
            })),
        );

        // Process publisher models
        for (const result of publisherResults) {
            const { publisher, response } = result;
            const modelFamily = supportedModels[publisher as keyof typeof supportedModels];
            const retiredModels = unsupportedModelsByPublisher[publisher as keyof typeof unsupportedModelsByPublisher];

            models = models.concat(
                response
                    .filter((model) => {
                        const modelName = model.name ?? '';
                        // Exclude retired models
                        if (retiredModels.some((retiredModel) => modelName.includes(retiredModel))) {
                            return false;
                        }
                        // Check if the model belongs to the supported model families
                        if (modelFamily.some((family) => modelName.includes(family))) {
                            return true;
                        }
                        return false;
                    })
                    .map((model) => {
                        const modelCapability = getModelCapabilities(model.name ?? '', 'vertexai');
                        return {
                            id: model.name ?? '',
                            name: model.name?.split('/').pop() ?? '',
                            provider: 'vertexai',
                            owner: publisher,
                            input_modalities: modelModalitiesToArray(modelCapability.input),
                            output_modalities: modelModalitiesToArray(modelCapability.output),
                            tool_support: modelCapability.tool_support,
                        } satisfies AIModel<string>;
                    }),
            );

            // Create global google gemini models for Gemini 2.5 and later, if missing from GenAI listing
            if (publisher === 'google') {
                const globalGeminiModels = response
                    .filter((model) => {
                        const modelName = model.name ?? '';
                        if (retiredModels.some((retiredModel) => modelName.includes(retiredModel))) {
                            return false;
                        }
                        if (modelFamily.some((family) => modelName.includes(family))) {
                            const versionMatch = modelName.match(/gemini-(\d+(?:\.\d+)?)/);
                            if (versionMatch) {
                                const version = parseFloat(versionMatch[1]);
                                if (version >= 2.5) {
                                    // Check if already present
                                    const shortName = modelName.split('/').pop();
                                    const globalName = `Global ${shortName}`;
                                    if (models.some((m) => m.name === globalName)) {
                                        return false;
                                    }
                                    return true;
                                }
                            }
                            return false;
                        }
                        return false;
                    })
                    .map((model) => {
                        const modelCapability = getModelCapabilities(model.name ?? '', 'vertexai');
                        return {
                            id: `locations/global/${model.name}`,
                            name: `Global ${model.name?.split('/').pop()}`,
                            provider: 'vertexai',
                            owner: publisher,
                            input_modalities: modelModalitiesToArray(modelCapability.input),
                            output_modalities: modelModalitiesToArray(modelCapability.output),
                            tool_support: modelCapability.tool_support,
                        } satisfies AIModel<string>;
                    });

                models = models.concat(globalGeminiModels);
            }

            // Create global anthropic models for those not in NON_GLOBAL_ANTHROPIC_MODELS
            if (publisher === 'anthropic') {
                const globalAnthropicModels = response
                    .filter((model) => {
                        const modelName = model.name ?? '';
                        if (retiredModels.some((retiredModel) => modelName.includes(retiredModel))) {
                            return false;
                        }
                        if (modelFamily.some((family) => modelName.includes(family))) {
                            if (modelName.includes('claude-3-7')) {
                                return true;
                            }
                            return !NON_GLOBAL_ANTHROPIC_MODELS.some((nonGlobalModel) =>
                                modelName.includes(nonGlobalModel),
                            );
                        }
                        return false;
                    })
                    .map((model) => {
                        const modelCapability = getModelCapabilities(model.name ?? '', 'vertexai');
                        return {
                            id: `locations/global/${model.name}`,
                            name: `Global ${model.name?.split('/').pop()}`,
                            provider: 'vertexai',
                            owner: publisher,
                            input_modalities: modelModalitiesToArray(modelCapability.input),
                            output_modalities: modelModalitiesToArray(modelCapability.output),
                            tool_support: modelCapability.tool_support,
                        } satisfies AIModel<string>;
                    });

                models = models.concat(globalAnthropicModels);
            }

            // Add additional models that are not in the listing
            for (const additionalModel of additionalModels[publisher as keyof typeof additionalModels]) {
                const publisherModelName = `publishers/${publisher}/models/${additionalModel}`;
                const modelCapability = getModelCapabilities(additionalModel, 'vertexai');
                models.push({
                    id: publisherModelName,
                    name: additionalModel,
                    provider: 'vertexai',
                    owner: publisher,
                    input_modalities: modelModalitiesToArray(modelCapability.input),
                    output_modalities: modelModalitiesToArray(modelCapability.output),
                    tool_support: modelCapability.tool_support,
                } satisfies AIModel<string>);
            }
        }

        const text = params?.text?.toLowerCase();
        if (text) {
            models = models.filter((model) => {
                return model.id.toLowerCase().includes(text) || model.name.toLowerCase().includes(text);
            });
        }

        //Remove duplicates
        const uniqueModels = Array.from(new Set(models.map((a) => a.id)))
            .map((id) => {
                return models.find((a) => a.id === id) ?? ({} as AIModel<string>);
            })
            .sort((a, b) => a.id.localeCompare(b.id));

        return uniqueModels;
    }

    async validateConnection(): Promise<boolean> {
        await this.getAuthorizationHeader();
        return true;
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        return generateVertexAiEmbeddings(this, options);
    }

    /**
     * Format VertexAI errors by routing to model-specific error handlers.
     * Each model definition (Gemini, Claude, Llama) can provide custom error parsing
     * based on their specific REST error structures.
     *
     * @param error - The error from the VertexAI/model endpoint
     * @param context - Context about where the error occurred
     * @returns A standardized LlumiverseError
     */
    public formatLlumiverseError(error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        // Get the model definition for this request
        const modelDef = getModelDefinition(context.model);

        // If the model definition provides custom error handling, use it
        if (modelDef.formatLlumiverseError) {
            try {
                return modelDef.formatLlumiverseError(this, error, context);
            } catch {
                // If model-specific handler throws, fall through to default handling
                // This allows model handlers to explicitly opt out for certain errors
            }
        }

        // Fall back to default AbstractDriver error handling
        return super.formatLlumiverseError(error, context);
    }
}

const API_BASE_PATH = 'aiplatform.googleapis.com';
function vertexApiBaseUrl(region: string, apiVersion: string) {
    const vertexBaseEndpoint = region === 'global' ? API_BASE_PATH : `${region}-${API_BASE_PATH}`;
    return `https://${vertexBaseEndpoint}/${apiVersion}`;
}

function createFetchClient({
    region,
    project,
    apiEndpoint,
    apiVersion = 'v1',
    fetchImpl,
}: {
    region: string;
    project: string;
    apiEndpoint?: string;
    apiVersion?: string;
    fetchImpl?: FETCH_FN;
}) {
    const vertexBaseEndpoint = apiEndpoint ?? (region === 'global' ? API_BASE_PATH : `${region}-${API_BASE_PATH}`);
    return new FetchClient(
        `https://${vertexBaseEndpoint}/${apiVersion}/projects/${project}/locations/${region}`,
        fetchImpl,
    ).withHeaders({
        'Content-Type': 'application/json',
    });
}

function isRequestStatus(err: unknown, status: number) {
    return err !== null && typeof err === 'object' && 'status' in err && err.status === status;
}
