import { AnthropicVertex } from '@anthropic-ai/vertex-sdk';
import { type Content, GoogleGenAI, type Model } from '@google/genai';
import { PredictionServiceClient, v1beta1 } from '@google-cloud/aiplatform';
import {
    type AIModel,
    type Completion,
    type CompletionResult,
    type DriverCompletionStream,
    type DriverOptions,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    getConversationMeta,
    type HttpTimeoutOptions,
    incrementConversationTurn,
    type LlumiverseError,
    type LlumiverseErrorContext,
    type ModelSearchPayload,
    type PromptSegment,
    Providers,
    stripBase64ImagesFromConversation,
    stripHeartbeatsFromConversation,
    type ToolUse,
    truncateLargeTextInConversation,
} from '@llumiverse/core';
import { AbstractDriver } from '@llumiverse/core/driver';
import { type FETCH_FN, FetchClient } from '@vertesia/api-fetch-client';
import { type AuthClient, GoogleAuth, type GoogleAuthOptions } from 'google-auth-library';
import {
    buildOpenAIChatCompletionsStreamingConversation,
    type OpenAIChatCompletionsPrompt,
} from '../openai/openai_chat_completions.js';
import { type ClaudePrompt, formatClaudeDebugPrompt, isClaudePromptCacheEnabled } from '../shared/claude-messages.js';
import { resolveModelListingMetadata } from '../shared/model-listing.js';
import { generateVertexAiEmbeddings } from './embeddings/embed.js';
import { ANTHROPIC_REGIONS, NON_GLOBAL_ANTHROPIC_MODELS } from './models/claude.js';
import { formatGeminiDebugPrompt } from './models/gemini.js';
import { GeminiContextCacheManager } from './models/gemini-context-cache.js';
import { formatImagenDebugPrompt, ImagenModelDefinition, type ImagenPrompt } from './models/imagen.js';
import { GEMINI_OMNI_VIDEO_MODEL, type OmniVideoPrompt } from './models/omni-video.js';
import { getModelDefinition, trimModelName } from './models.js';
import { getListedVertexOpenMaaSModels } from './open-maas-models.js';

export interface VertexAIDriverOptions extends DriverOptions {
    project: string;
    region: string;
    googleAuthOptions?: GoogleAuthOptions;
    /**
     * Kill switch for explicit Gemini context caching (Vertex `cachedContents`). Caching is normally
     * decided per execution — see `ExecutionOptions.prompt_cache_mode`, which defaults to caching the
     * static prefix whenever `prompt_cache_key` is set. Setting this to `false` disables the whole
     * path for every execution this driver runs, whatever the execution options say.
     */
    geminiContextCache?: boolean;
    /**
     * Default lifetime, in seconds, of the `cachedContents` resources this driver creates.
     * Defaults to 1800 (30 minutes). `ExecutionOptions.prompt_cache_ttl_seconds` overrides it per call.
     */
    geminiContextCacheTtlSeconds?: number;
}

export interface GenerateContentPrompt {
    contents: Content[];
    system?: Content;
}
type ClaudeStreamingPrompt = { messages: unknown[]; system?: unknown[] };
type ConversationWrapper = { messages?: unknown[]; system?: unknown[] };

function isClaudePrompt(prompt: VertexAIPrompt): prompt is ClaudePrompt {
    return (
        'messages' in prompt &&
        ('system' in prompt || prompt.messages.some((message) => Array.isArray(message.content)))
    );
}

function isClaudeStreamingPrompt(prompt: unknown): prompt is ClaudeStreamingPrompt {
    return (
        prompt !== null &&
        typeof prompt === 'object' &&
        'messages' in prompt &&
        Array.isArray((prompt as ConversationWrapper).messages)
    );
}

//General Prompt type for VertexAI
export type VertexAIPrompt =
    | ImagenPrompt
    | OmniVideoPrompt
    | GenerateContentPrompt
    | ClaudePrompt
    | OpenAIChatCompletionsPrompt;

export { trimModelName };

export class VertexAIDriver extends AbstractDriver<VertexAIDriverOptions, VertexAIPrompt> {
    static readonly PROVIDER = Providers.vertexai;
    provider = VertexAIDriver.PROVIDER;

    aiplatform: v1beta1.ModelServiceClient | undefined;
    anthropicClient: AnthropicVertex | undefined;
    fetchClient: FetchClient | undefined;
    private regionOverrideClients: Map<string, FetchClient> = new Map();
    googleGenAI: GoogleGenAI | undefined;
    googleGenAIRegion: string | undefined;
    googleGenAIServiceTier: string | undefined;
    modelGarden: v1beta1.ModelGardenServiceClient | undefined;
    imagenClient: PredictionServiceClient | undefined;
    predictionClient: PredictionServiceClient | undefined;

    googleAuth: GoogleAuth<AuthClient>;
    private authClientPromise: Promise<AuthClient> | undefined;
    private geminiContextCacheManager: GeminiContextCacheManager | undefined;

    constructor(options: VertexAIDriverOptions) {
        super(options);

        this.aiplatform = undefined;
        this.anthropicClient = undefined;
        this.fetchClient = undefined;
        this.googleGenAI = undefined;
        this.googleGenAIRegion = undefined;
        this.googleGenAIServiceTier = undefined;
        this.modelGarden = undefined;
        this.imagenClient = undefined;
        this.predictionClient = undefined;

        this.googleAuth = new GoogleAuth(options.googleAuthOptions);
        this.authClientPromise = undefined;
        this.geminiContextCacheManager = undefined;
    }

    /**
     * Registry of the Vertex `cachedContents` resources this driver instance has created, used by the
     * Gemini execution path to find-or-create the cache for an execution's `prompt_cache_key`.
     * Returns `undefined` when the driver-level kill switch is off, which keeps the Gemini payload
     * exactly as it is built today.
     *
     * The registry is per instance on purpose. Two instances create two resources for the same
     * prefix; the duplicate expires on its own TTL and costs only storage rent for that window,
     * which is far cheaper than the coordination a shared registry would need.
     */
    public getGeminiContextCacheManager(): GeminiContextCacheManager | undefined {
        if (this.options.geminiContextCache === false) return undefined;
        if (!this.geminiContextCacheManager) {
            this.geminiContextCacheManager = new GeminiContextCacheManager({
                ttlSeconds: this.options.geminiContextCacheTtlSeconds,
            });
        }
        return this.geminiContextCacheManager;
    }

    /**
     * Cleanup Google Cloud clients when the driver is evicted from the cache.
     * AbstractDriver invokes this hook only after active executions finish.
     */
    protected override destroyProviderResources(): Promise<void> {
        const clients = [this.aiplatform, this.modelGarden, this.imagenClient, this.predictionClient];
        // Dropping the registry only forfeits reuse: the cachedContents resources it names live
        // server-side and expire on their own TTL.
        this.geminiContextCacheManager = undefined;
        this.aiplatform = undefined;
        this.modelGarden = undefined;
        this.imagenClient = undefined;
        this.predictionClient = undefined;
        return Promise.allSettled(clients.map((client) => client?.close())).then((results) => {
            for (const result of results) {
                if (result.status === 'rejected') {
                    this.logger.warn({ error: result.reason }, 'Failed to close Vertex AI client');
                }
            }
        });
    }

    private async getAuthClient(): Promise<AuthClient> {
        if (!this.authClientPromise) {
            this.authClientPromise = this.googleAuth.getClient();
        }
        return this.authClientPromise;
    }

    private getGoogleGenAIHttpOptions(serviceTier?: string, httpTimeout?: HttpTimeoutOptions) {
        return {
            timeout: this.getDriverRequestTimeoutMs(httpTimeout),
            ...(serviceTier && serviceTier !== 'default'
                ? {
                      headers: {
                          'X-Vertex-AI-LLM-Request-Type': 'shared',
                          'X-Vertex-AI-LLM-Shared-Request-Type': serviceTier,
                      },
                  }
                : {}),
        };
    }

    private getAnthropicVertexClientOptions(region: string, authClient: AuthClient, httpTimeout?: HttpTimeoutOptions) {
        return {
            timeout: this.getDriverRequestTimeoutMs(httpTimeout),
            region,
            projectId: this.options.project,
            authClient: authClient,
            fetch: this.getDriverFetch(),
        };
    }

    public getGoogleGenAIClient(
        region: string = this.options.region,
        serviceTier?: string,
        httpTimeout?: HttpTimeoutOptions,
    ): GoogleGenAI {
        if (httpTimeout) {
            return this.buildGoogleGenAIClient(region, serviceTier, httpTimeout);
        }
        if (this.googleGenAI && this.googleGenAIRegion === region && this.googleGenAIServiceTier === serviceTier) {
            // Return existing client if region and service tier settings match
            return this.googleGenAI;
        }
        this.googleGenAI = this.buildGoogleGenAIClient(region, serviceTier);
        this.googleGenAIRegion = region;
        this.googleGenAIServiceTier = serviceTier;
        return this.googleGenAI;
    }

    private buildGoogleGenAIClient(
        region: string,
        serviceTier?: string,
        httpTimeout?: HttpTimeoutOptions,
    ): GoogleGenAI {
        return new GoogleGenAI({
            project: this.options.project,
            location: region,
            vertexai: true,
            googleAuthOptions: this.options.googleAuthOptions || {
                scopes: ['https://www.googleapis.com/auth/cloud-platform'],
            },
            httpOptions: this.getGoogleGenAIHttpOptions(serviceTier, httpTimeout),
        });
    }

    public getFetchClient(region: string = this.options.region): FetchClient {
        //Lazy initialization
        if (!this.fetchClient) {
            this.fetchClient = createFetchClient({
                region: region,
                project: this.options.project,
                fetchImpl: this.getDriverFetch(),
            }).withAuthCallback(async () => {
                const token = await this.googleAuth.getAccessToken();
                return `Bearer ${token}`;
            });
        }
        return this.fetchClient;
    }

    /**
     * Get a fetch client with an overridden region. Useful when a model only exists
     * in a specific region (e.g., "global" for xAI models).
     */
    public getFetchClientForRegion(region: string, apiVersion = 'v1', endpointRegion?: string): FetchClient {
        const cacheKey = `${region}:${endpointRegion ?? region}:${apiVersion}:${this.options.project}`;
        let client = this.regionOverrideClients.get(cacheKey);
        if (!client) {
            client = createFetchClient({
                region: region,
                project: this.options.project,
                apiVersion,
                endpointRegion,
                fetchImpl: this.getDriverFetch(),
            }).withAuthCallback(async () => {
                const token = await this.googleAuth.getAccessToken();
                return `Bearer ${token}`;
            });
            this.regionOverrideClients.set(cacheKey, client);
        }
        return client;
    }

    public getRequestTimeoutMs(httpTimeout?: HttpTimeoutOptions): number {
        return this.getDriverRequestTimeoutMs(httpTimeout);
    }

    public async getAnthropicClient(
        region: string = this.options.region,
        httpTimeout?: HttpTimeoutOptions,
    ): Promise<AnthropicVertex> {
        // Extract region prefix and map if it exists in ANTHROPIC_REGIONS, otherwise use as-is
        const getRegionPrefix = (r: string) => r.split('-')[0];
        const regionPrefix = getRegionPrefix(region);
        const mappedRegion = ANTHROPIC_REGIONS[regionPrefix] || region;

        const defaultRegionPrefix = getRegionPrefix(this.options.region);
        const defaultMappedRegion = ANTHROPIC_REGIONS[defaultRegionPrefix] || this.options.region;

        // Get auth client to avoid version mismatch with GoogleAuth generic types
        const authClient = await this.getAuthClient();

        // If mapped region is different from default mapped region, create one-off client
        if (httpTimeout || mappedRegion !== defaultMappedRegion) {
            return new AnthropicVertex(this.getAnthropicVertexClientOptions(mappedRegion, authClient, httpTimeout));
        }

        //Lazy initialization for default region
        if (!this.anthropicClient) {
            this.anthropicClient = new AnthropicVertex(this.getAnthropicVertexClientOptions(mappedRegion, authClient));
        }
        return this.anthropicClient;
    }

    public async getAIPlatformClient(): Promise<v1beta1.ModelServiceClient> {
        //Lazy initialization
        if (!this.aiplatform) {
            const authClient = await this.getAuthClient();
            this.aiplatform = new v1beta1.ModelServiceClient({
                projectId: this.options.project,
                apiEndpoint: `${this.options.region}-${API_BASE_PATH}`,
                authClient,
            });
        }
        return this.aiplatform;
    }

    public async getModelGardenClient(): Promise<v1beta1.ModelGardenServiceClient> {
        //Lazy initialization
        if (!this.modelGarden) {
            const authClient = await this.getAuthClient();
            this.modelGarden = new v1beta1.ModelGardenServiceClient({
                projectId: this.options.project,
                apiEndpoint: `${this.options.region}-${API_BASE_PATH}`,
                authClient,
            });
        }
        return this.modelGarden;
    }

    public async getImagenClient(): Promise<PredictionServiceClient> {
        //Lazy initialization
        if (!this.imagenClient) {
            // TODO: make location configurable, fixed to us-central1 for now
            const authClient = await this.getAuthClient();
            this.imagenClient = new PredictionServiceClient({
                projectId: this.options.project,
                apiEndpoint: `us-central1-${API_BASE_PATH}`,
                authClient,
            });
        }
        return this.imagenClient;
    }

    public async getPredictionServiceClient(): Promise<PredictionServiceClient> {
        //Lazy initialization using the driver's configured region
        if (!this.predictionClient) {
            const authClient = await this.getAuthClient();
            this.predictionClient = new PredictionServiceClient({
                projectId: this.options.project,
                apiEndpoint: `${this.options.region}-${API_BASE_PATH}`,
                authClient,
            });
        }
        return this.predictionClient;
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

    public formatDebugPrompt(prompt: VertexAIPrompt): VertexAIPrompt {
        if (isClaudePrompt(prompt)) {
            return formatClaudeDebugPrompt(prompt);
        }
        if ('contents' in prompt) {
            return formatGeminiDebugPrompt(prompt);
        }
        if ('messages' in prompt) {
            return prompt;
        }
        if ('text' in prompt && 'images' in prompt) {
            return prompt;
        }
        return formatImagenDebugPrompt(prompt);
    }

    async requestTextCompletion(
        prompt: VertexAIPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        return getModelDefinition(options.model).requestTextCompletion(this, prompt, options, signal);
    }
    async requestTextCompletionStream(
        prompt: VertexAIPrompt,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        return getModelDefinition(options.model).requestTextCompletionStream(this, prompt, options, signal);
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
    ): Content[] | unknown | undefined {
        // Handle OpenAI Chat Completions prompts (xAI Grok, Meta Llama via Vertex AI MaaS)
        // IMPORTANT: check this BEFORE the Claude check — both have a `messages` array
        if ('_is_openai_chat_completions' in prompt && prompt._is_openai_chat_completions) {
            return buildOpenAIChatCompletionsStreamingConversation(
                prompt as unknown as OpenAIChatCompletionsPrompt,
                result,
                toolUse,
                options,
            );
        }

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
                    case 'thoughts':
                        return '';
                    case 'json':
                        return typeof r.value === 'string' ? r.value : JSON.stringify(r.value);
                    case 'image':
                        // Skip images in conversation - they're in the result
                        return '';
                    case 'video':
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
        let conversation: Content[] = [...prompt.contents];

        // Only add assistant message if there's actual content
        // (Empty text parts can cause API errors)
        if (parts.length > 0) {
            conversation.push({
                role: 'model',
                parts: parts as Content['parts'],
            });
        }

        // Increment turn counter
        conversation = incrementConversationTurn(conversation) as Content[];

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
                    case 'thoughts':
                        return '';
                    case 'json':
                        return typeof r.value === 'string' ? r.value : JSON.stringify(r.value);
                    case 'image':
                        return '';
                    case 'video':
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
            // Cached Claude histories must remain byte-stable between semantic
            // checkpoints. Studio already bounds and artifact-backs model-facing
            // tool results before they reach this driver.
            textMaxTokens: isClaudePromptCacheEnabled(options) ? undefined : options.stripTextMaxTokens,
        };
        let processedConversation = stripBase64ImagesFromConversation(withTurn, stripOptions);
        processedConversation = truncateLargeTextInConversation(processedConversation, stripOptions);
        processedConversation = stripHeartbeatsFromConversation(processedConversation, {
            keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
            currentTurn,
        });

        return processedConversation;
    }

    async requestImageGeneration(
        _prompt: ImagenPrompt,
        _options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        const splits = _options.model.split('/');
        const modelName = trimModelName(splits[splits.length - 1]);
        return new ImagenModelDefinition(modelName).requestImageGeneration(this, _prompt, _options, signal);
    }

    async getGenAIModelsArray(client: GoogleGenAI): Promise<Model[]> {
        const models: Model[] = [];
        const pager = await client.models.list();
        for await (const item of pager) {
            models.push(item);
        }
        return models;
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel[]> {
        // Get clients
        const modelGarden = await this.getModelGardenClient();
        const aiplatform = await this.getAIPlatformClient();
        const globalGenAiClient = this.getGoogleGenAIClient('global');

        let models: AIModel[] = [];

        // Model Garden publisher listings for families that are reliably returned by the API.
        // Open MaaS-only families are appended from VERTEX_OPEN_MAAS_MODELS below to avoid
        // extra listPublisherModels calls for publishers whose MaaS models do not appear there.
        // Intentional allow-list: Vertex Model Garden spans endpoint contracts this driver does not implement. Future
        // versions within these known executable families flow through automatically, but do not broaden this to every
        // publisher merely because it appears in listing metadata; add a publisher only with a compatible request path.
        const publisherConfig = {
            google: {
                families: ['gemini', 'imagen'],
                excluded: [
                    'gemini-pro',
                    'gemini-ultra',
                    'imagen-product-recontext-preview',
                    'embedding',
                    'embed',
                    'gemini-live',
                    'native-audio',
                    '-tts',
                    'computer-use-preview',
                ],
                /** Additional models not in the listings, but we want to include.
                 * TODO: Remove once available in listing API. */
                additional: ['imagen-3.0-fast-generate-001', 'gemini-omni-flash-preview'],
            },
            anthropic: {
                families: ['claude'],
                excluded: [],
                additional: [],
            },
            xai: {
                families: ['grok'],
                excluded: [],
                additional: [],
            },
        } as const;

        type Publisher = keyof typeof publisherConfig;
        const publishers: readonly Publisher[] = Object.keys(publisherConfig) as Publisher[];

        // Start all network requests in parallel
        const aiplatformPromise = aiplatform.listModels({
            parent: `projects/${this.options.project}/locations/${this.options.region}`,
        });
        const publisherPromises = publishers.map(async (publisher) => {
            const [response] = await modelGarden.listPublisherModels({
                parent: `publishers/${publisher}`,
                orderBy: 'name',
                listAllVersions: true,
            });
            return { publisher, response };
        });

        const globalGooglePromise = this.getGenAIModelsArray(globalGenAiClient);
        // Await all network requests
        const [aiplatformResult, globalGoogleResult, ...publisherResults] = await Promise.all([
            aiplatformPromise,
            globalGooglePromise,
            ...publisherPromises,
        ]);

        models = models.concat(getListedVertexOpenMaaSModels(this.options.region));

        // Process aiplatform models, project specific models
        const [response] = aiplatformResult;
        models = models.concat(
            response.map((model) => {
                const id = model.name?.split('/').pop() ?? '';
                return {
                    id,
                    name: model.displayName ?? '',
                    provider: 'vertexai',
                    ...resolveModelListingMetadata(id, Providers.vertexai),
                };
            }),
        );

        // Process global google models from GenAI
        // Exclude embedding, retired and or unsupported models
        const excludedModels = publisherConfig.google.excluded;
        models = models.concat(
            globalGoogleResult
                .filter(
                    (model) =>
                        !excludedModels.some((excludedModel) => (model.name ?? '').includes(excludedModel)) &&
                        isExecutableGoogleModel(model),
                )
                .map((model) => {
                    const modelMetadata = resolveModelListingMetadata(model.name ?? '', Providers.vertexai);
                    return {
                        id: `locations/global/${model.name}`,
                        name: `Global ${model.name?.split('/').pop()}`,
                        provider: 'vertexai',
                        owner: 'google',
                        ...modelMetadata,
                    };
                }),
        );

        // Process publisher models
        for (const result of publisherResults) {
            const { publisher, response } = result as { publisher: string; response: Array<{ name?: string }> };
            const config = publisherConfig[publisher as Publisher];
            const modelFamily = config.families;
            const excludedModels = config.excluded;

            models = models.concat(
                response
                    .filter((model) => {
                        const modelName = model.name ?? '';
                        // Exclude embedding, retired and or unsupported models
                        if (excludedModels.some((retiredModel) => modelName.includes(retiredModel))) {
                            return false;
                        }
                        // Check if the model belongs to the supported model families
                        if (modelFamily.some((family) => modelName.includes(family))) {
                            return true;
                        }
                        return false;
                    })
                    .map((model) => {
                        const rawModelId = model.name ?? '';
                        const isGlobalOnlyPublisher = isGlobalOnlyPublisherModel(publisher, rawModelId);
                        const listedModelId = isGlobalOnlyPublisher ? `locations/global/${rawModelId}` : rawModelId;
                        const modelMetadata = resolveModelListingMetadata(listedModelId, Providers.vertexai);
                        return {
                            id: listedModelId,
                            name: isGlobalOnlyPublisher
                                ? `Global ${rawModelId.split('/').pop()}`
                                : (rawModelId.split('/').pop() ?? ''),
                            provider: 'vertexai',
                            owner: publisher,
                            ...modelMetadata,
                        } satisfies AIModel;
                    }),
            );

            // Create global google gemini models for Gemini 2.5 and later, if missing from GenAI listing
            if (publisher === 'google') {
                const globalGeminiModels = response
                    .filter((model) => {
                        const modelName = model.name ?? '';
                        if (excludedModels.some((retiredModel) => modelName.includes(retiredModel))) {
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
                        const modelMetadata = resolveModelListingMetadata(model.name ?? '', Providers.vertexai);
                        return {
                            id: `locations/global/${model.name}`,
                            name: `Global ${model.name?.split('/').pop()}`,
                            provider: 'vertexai',
                            owner: publisher,
                            ...modelMetadata,
                        } satisfies AIModel;
                    });

                models = models.concat(globalGeminiModels);
            }

            // Create global anthropic models for those not in NON_GLOBAL_ANTHROPIC_MODELS
            if (publisher === 'anthropic') {
                const globalAnthropicModels = response
                    .filter((model) => {
                        const modelName = model.name ?? '';
                        if (excludedModels.some((retiredModel) => modelName.includes(retiredModel))) {
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
                        const modelMetadata = resolveModelListingMetadata(model.name ?? '', Providers.vertexai);
                        return {
                            id: `locations/global/${model.name}`,
                            name: `Global ${model.name?.split('/').pop()}`,
                            provider: 'vertexai',
                            owner: publisher,
                            ...modelMetadata,
                        } satisfies AIModel;
                    });

                models = models.concat(globalAnthropicModels);
            }

            // Add additional models that are not in the listing
            for (const additionalModel of publisherConfig[publisher as Publisher].additional) {
                const rawModelId = `publishers/${publisher}/models/${additionalModel}`;
                const isGlobalOnlyPublisher = isGlobalOnlyPublisherModel(publisher, rawModelId);
                const modelMetadata = resolveModelListingMetadata(additionalModel, Providers.vertexai);
                models.push({
                    id: isGlobalOnlyPublisher ? `locations/global/${rawModelId}` : rawModelId,
                    name: isGlobalOnlyPublisher ? `Global ${additionalModel}` : additionalModel,
                    provider: 'vertexai',
                    owner: publisher,
                    ...modelMetadata,
                } satisfies AIModel);
            }
        }

        //Remove duplicates
        const uniqueModels = Array.from(new Set(models.map((a) => a.id)))
            .map((id) => {
                return models.find((a) => a.id === id) ?? ({} as AIModel);
            })
            .sort((a, b) => a.id.localeCompare(b.id));

        return uniqueModels;
    }

    validateConnection(): Promise<boolean> {
        throw new Error('Method not implemented.');
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        return generateVertexAiEmbeddings(this, options);
    }

    /**
     * Format VertexAI errors by routing to model-specific error handlers.
     * Each model definition (Gemini, Claude, Llama) can provide custom error parsing
     * based on their specific SDK error structures.
     *
     * @param error - The error from the VertexAI/model SDK
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

function isGlobalOnlyPublisherModel(publisher: string, modelId: string): boolean {
    return publisher === 'xai' || (publisher === 'google' && modelId.endsWith(`/models/${GEMINI_OMNI_VIDEO_MODEL}`));
}

function isExecutableGoogleModel(model: Model): boolean {
    const modelName = (model.name ?? '').toLowerCase();
    if (modelName.endsWith('/gemini-omni-flash-preview') || modelName === 'gemini-omni-flash-preview') return true;
    // Intentional execution-path allow-list: Vertex uses separate methods for embeddings, Live/TTS, music and video.
    // This driver currently implements generateContent and generateImages. Unknown actions are excluded only when
    // Google supplies them; absent action metadata falls back to the known-family/name policy above.
    if (!modelName.includes('gemini') && !modelName.includes('imagen')) return false;

    if (model.supportedActions?.length) {
        const actions = model.supportedActions.map((action) => action.toLowerCase().replace(/[^a-z]/g, ''));
        return actions.some((action) => action === 'generatecontent' || action === 'generateimages');
    }

    return !/(?:embedding|embed|tts|live|native-audio|veo|lyria)/.test(modelName);
}

//'us-central1-aiplatform.googleapis.com',
const API_BASE_PATH = 'aiplatform.googleapis.com';
export function createFetchClient({
    region,
    project,
    apiEndpoint,
    apiVersion = 'v1',
    endpointRegion,
    fetchImpl,
}: {
    region: string;
    project: string;
    apiEndpoint?: string;
    apiVersion?: string;
    endpointRegion?: string;
    fetchImpl?: FETCH_FN;
}): FetchClient {
    // For the "global" region, use aiplatform.googleapis.com without any prefix.
    // Regional endpoints use ${region}-aiplatform.googleapis.com (e.g., us-central1-aiplatform.googleapis.com).
    const hostRegion = endpointRegion ?? region;
    const vertexBaseEndpoint =
        apiEndpoint ?? (hostRegion === 'global' ? API_BASE_PATH : `${hostRegion}-${API_BASE_PATH}`);
    return new FetchClient(
        `https://${vertexBaseEndpoint}/${apiVersion}/projects/${project}/locations/${region}`,
        fetchImpl,
    ).withHeaders({
        'Content-Type': 'application/json',
    });
}
