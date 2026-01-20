import type { ClientOptions as AnthropicVertexClientOptions } from "@anthropic-ai/vertex-sdk";
import { AnthropicVertex } from "@anthropic-ai/vertex-sdk";
import { PredictionServiceClient, v1beta1 } from "@google-cloud/aiplatform";
import { Content, GoogleGenAI, Model } from "@google/genai";
import {
    AIModel,
    AbstractDriver,
    BatchJob,
    BatchJobType,
    Completion,
    CompletionChunkObject,
    CompletionResult,
    CreateBatchJobOptions,
    DriverOptions,
    EmbeddingsOptions,
    EmbeddingsResult,
    ExecutionOptions,
    GCSBatchDestination,
    GCSBatchSource,
    ListBatchJobsOptions,
    ListBatchJobsResult,
    ModelSearchPayload,
    PromptSegment,
    getConversationMeta,
    getModelCapabilities,
    incrementConversationTurn,
    modelModalitiesToArray,
    stripBase64ImagesFromConversation,
    truncateLargeTextInConversation,
    unwrapConversationArray,
} from "@llumiverse/core";
import { FetchClient } from "@vertesia/api-fetch-client";
import { AuthClient, GoogleAuth, GoogleAuthOptions } from "google-auth-library";
import {
    cancelClaudeBatchJob,
    createClaudeBatchJob,
    deleteClaudeBatchJob,
    getClaudeBatchJob,
    listClaudeBatchJobs,
} from "./batch/claude-batch.js";
import {
    cancelEmbeddingsBatchJob,
    createEmbeddingsBatchJobSDK,
    deleteEmbeddingsBatchJob,
    getEmbeddingsBatchJob,
} from "./batch/embeddings-batch.js";
import {
    cancelGeminiBatchJob,
    createGeminiBatchJob,
    deleteGeminiBatchJob,
    getGeminiBatchJob,
    isTerminalState,
    listGeminiBatchJobs,
} from "./batch/gemini-batch.js";
import {
    deleteGeminiFile,
    getGeminiFile,
    listGeminiFiles,
    uploadFileToGemini,
    waitForFileActive,
} from "./batch/gemini-files.js";
import { GeminiFileResource, decodeBatchJobId } from "./batch/types.js";
import { getEmbeddingsForImages } from "./embeddings/embeddings-image.js";
import { TextEmbeddingsOptions, getEmbeddingsForText } from "./embeddings/embeddings-text.js";
import { getModelDefinition } from "./models.js";
import { ANTHROPIC_REGIONS, NON_GLOBAL_ANTHROPIC_MODELS } from "./models/claude.js";
import { ImagenModelDefinition, ImagenPrompt } from "./models/imagen.js";

export interface VertexAIDriverOptions extends DriverOptions {
    project: string;
    region: string;
    googleAuthOptions?: GoogleAuthOptions;
    geminiApiKey?: string;
}

export interface GenerateContentPrompt {
    contents: Content[];
    system?: Content;
}

//General Prompt type for VertexAI
export type VertexAIPrompt = ImagenPrompt | GenerateContentPrompt;

export function trimModelName(model: string) {
    const i = model.lastIndexOf("@");
    return i > -1 ? model.substring(0, i) : model;
}

export class VertexAIDriver extends AbstractDriver<
    VertexAIDriverOptions,
    VertexAIPrompt,
    GCSBatchSource,
    GCSBatchDestination> {
    static PROVIDER = "vertexai";
    provider = VertexAIDriver.PROVIDER;

    aiplatform: v1beta1.ModelServiceClient | undefined;
    anthropicClient: AnthropicVertex | undefined;
    fetchClient: FetchClient | undefined;
    geminiApiClient: FetchClient | undefined;
    googleGenAI: GoogleGenAI | undefined;
    llamaClient: FetchClient & { region?: string } | undefined;
    modelGarden: v1beta1.ModelGardenServiceClient | undefined;
    imagenClient: PredictionServiceClient | undefined;

    googleAuth: GoogleAuth<any>;
    private authClientPromise: Promise<AuthClient> | undefined;

    constructor(options: VertexAIDriverOptions) {
        super(options);

        this.aiplatform = undefined;
        this.anthropicClient = undefined;
        this.fetchClient = undefined;
        this.geminiApiClient = undefined;
        this.googleGenAI = undefined;
        this.modelGarden = undefined;
        this.llamaClient = undefined;
        this.imagenClient = undefined;

        if (options.googleAuthOptions) {
            options.googleAuthOptions.scopes = [
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/generative-language",
                "https://www.googleapis.com/auth/devstorage.read_only"
            ];
        }
        this.googleAuth = new GoogleAuth(options.googleAuthOptions);
        this.authClientPromise = undefined;

        this.logger.debug({ options: this.options }, "VertexAIDriver initialized with options:");
    }

    private async getAuthClient(): Promise<AuthClient> {
        if (!this.authClientPromise) {
            this.authClientPromise = this.googleAuth.getClient();
        }
        return this.authClientPromise;
    }

    public async getGoogleGenAIClient(region: string = this.options.region, api: "VERTEXAI" | "GEMINI" = "VERTEXAI"): Promise<GoogleGenAI> {
        //Lazy initialization

        //Gemini API - sometimes called Gemini Developer API
        if (api == "GEMINI") {
            //Prefer OAuth if available
            if (this.options.googleAuthOptions) {
                this.logger.info("Using OAuth credentials for Gemini API client");
                const auth = await this.getAuthClient();
                return new GoogleGenAI({
                    vertexai: false,
                    googleAuthOptions: { authClient: auth },
                });
            }
            this.logger.info("Using API Key for Gemini API client");
            return new GoogleGenAI({
                vertexai: false,
                apiKey: this.options.geminiApiKey,
            });
        }

        //Vertex AI API
        if (region !== this.options.region) {
            //Get one off client for different region
            return new GoogleGenAI({
                project: this.options.project,
                location: region,
                vertexai: true,
                googleAuthOptions: this.options.googleAuthOptions || {
                    scopes: ["https://www.googleapis.com/auth/cloud-platform"],
                }
            });
        }
        if (!this.googleGenAI) {
            this.googleGenAI = new GoogleGenAI({
                project: this.options.project,
                location: region,
                vertexai: true,
                googleAuthOptions: this.options.googleAuthOptions || {
                    scopes: ["https://www.googleapis.com/auth/cloud-platform"],
                }
            });
        }
        return this.googleGenAI;
    }

    public getFetchClient(): FetchClient {
        //Lazy initialization
        if (!this.fetchClient) {
            this.fetchClient = createFetchClient({
                region: this.options.region,
                project: this.options.project,
            }).withAuthCallback(async () => {
                const token = await this.googleAuth.getAccessToken();
                return `Bearer ${token}`;
            });
        }
        return this.fetchClient;
    }

    /**
     * Gets a FetchClient configured for the Generative Language API (Gemini API).
     * Supports both API key (via x-goog-api-key) and OAuth (via Bearer token).
     * If an API key is provided in options or env, it is used.
     * Otherwise, it falls back to the GoogleAuth strategy (ADC, Service Account, etc.).
     */
    public getGeminiApiFetchClient(useAPIkey: boolean = false): FetchClient {
        if (!this.geminiApiClient) {
            const apiKey = this.options.geminiApiKey;

            this.logger.debug({
                useAPIkey,
                hasApiKey: !!apiKey,
                project: this.options.project,
                region: this.options.region,
            }, "getGeminiApiFetchClient: Initializing Gemini API client");

            const client = new FetchClient("https://generativelanguage.googleapis.com/v1beta")
                .withHeaders({
                    "Content-Type": "application/json",
                });

            if (apiKey && useAPIkey) {
                client.withHeaders({
                    "x-goog-api-key": apiKey,
                });
                this.logger.debug("getGeminiApiFetchClient: Using API key authentication");
            } else {
                // Use OAuth via Google Auth library (ADC / Service Account / User Credentials)
                if (this.options.project) {
                    client.withHeaders({
                        "x-goog-user-project": this.options.project,
                    });
                    this.logger.info({
                        project: this.options.project,
                    }, "getGeminiApiFetchClient: Set x-goog-user-project header for OAuth");
                } else {
                    this.logger.warn("getGeminiApiFetchClient: No project ID available for x-goog-user-project header");
                }

                client.withAuthCallback(async () => {
                    const token = await this.googleAuth.getAccessToken();
                    return `Bearer ${token}`;
                });
            }
            this.geminiApiClient = client;
        }
        return this.geminiApiClient;
    }

    public getLLamaClient(region: string = "us-central1"): FetchClient {
        //Lazy initialization
        if (!this.llamaClient || this.llamaClient["region"] !== region) {
            this.llamaClient = createFetchClient({
                region: region,
                project: this.options.project,
                apiVersion: "v1beta1",
            }).withAuthCallback(async () => {
                const token = await this.googleAuth.getAccessToken();
                return `Bearer ${token}`;
            });
            // Store the region for potential client reuse
            this.llamaClient["region"] = region;
        }
        return this.llamaClient;
    }

    public async getAnthropicClient(region: string = this.options.region): Promise<AnthropicVertex> {
        // Extract region prefix and map if it exists in ANTHROPIC_REGIONS, otherwise use as-is
        const getRegionPrefix = (r: string) => r.split('-')[0];
        const regionPrefix = getRegionPrefix(region);
        const mappedRegion = ANTHROPIC_REGIONS[regionPrefix] || region;

        const defaultRegionPrefix = getRegionPrefix(this.options.region);
        const defaultMappedRegion = ANTHROPIC_REGIONS[defaultRegionPrefix] || this.options.region;

        // Get auth client to avoid version mismatch with GoogleAuth generic types
        const authClient = await this.getAuthClient();

        // If mapped region is different from default mapped region, create one-off client
        if (mappedRegion !== defaultMappedRegion) {
            return new AnthropicVertex({
                timeout: 20 * 60 * 10000, // Set to 20 minutes, 10 minute default, setting this disables long request error: https://github.com/anthropics/anthropic-sdk-typescript?#long-requests
                region: mappedRegion,
                projectId: this.options.project,
                authClient: authClient as unknown as AnthropicVertexClientOptions["authClient"],
            });
        }

        //Lazy initialization for default region
        if (!this.anthropicClient) {
            this.anthropicClient = new AnthropicVertex({
                timeout: 20 * 60 * 10000, // Set to 20 minutes, 10 minute default, setting this disables long request error: https://github.com/anthropics/anthropic-sdk-typescript?#long-requests
                region: mappedRegion,
                projectId: this.options.project,
                authClient: authClient as unknown as AnthropicVertexClientOptions["authClient"],
            });
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

    validateResult(result: Completion, options: ExecutionOptions) {
        // Optionally preprocess the result before validation
        const modelDef = getModelDefinition(options.model);
        if (typeof modelDef.preValidationProcessing === "function") {
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
        return model.includes("imagen");
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
        options: ExecutionOptions
    ): Content[] | unknown | undefined {
        // Handle Claude-style prompts (has 'messages' array)
        if ('messages' in prompt && Array.isArray((prompt as any).messages)) {
            return this.buildClaudeStreamingConversation(prompt as any, result, toolUse, options);
        }

        // Only handle Gemini-style prompts with contents array
        if (!('contents' in prompt) || !Array.isArray(prompt.contents)) {
            return undefined;
        }

        const completionResults = result as CompletionResult[];

        // Convert accumulated results to text content for assistant message
        const textContent = completionResults
            .map(r => {
                switch (r.type) {
                    case 'text':
                        return r.value;
                    case 'json':
                        return typeof r.value === 'string' ? r.value : JSON.stringify(r.value);
                    case 'image':
                        // Skip images in conversation - they're in the result
                        return '';
                    default:
                        return String((r as any).value || '');
                }
            })
            .join('');

        // Build parts array for assistant message
        const parts: any[] = [];
        if (textContent) {
            parts.push({ text: textContent });
        }
        // Add function calls if present (Gemini format)
        if (toolUse && toolUse.length > 0) {
            for (const tool of toolUse as any[]) {
                parts.push({
                    functionCall: {
                        name: tool.tool_name,
                        args: tool.tool_input,
                    }
                });
            }
        }

        // Build assistant message in Gemini Content format
        const assistantContent: Content = {
            role: 'model',
            parts: parts.length > 0 ? parts : [{ text: '' }]
        };

        // Unwrap array if wrapped, otherwise treat as array
        const unwrapped = unwrapConversationArray<Content>(options.conversation);
        const existingConversation = unwrapped ?? (options.conversation as Content[] || []);

        // Combine existing conversation + prompt contents + assistant response
        let conversation: Content[] = [
            ...existingConversation,
            ...prompt.contents,
            assistantContent
        ];

        // Increment turn counter
        conversation = incrementConversationTurn(conversation) as Content[];

        // Apply stripping based on options
        const currentTurn = getConversationMeta(conversation).turnNumber;
        const stripOptions = {
            keepForTurns: options.stripImagesAfterTurns ?? Infinity,
            currentTurn,
            textMaxTokens: options.stripTextMaxTokens
        };
        let processedConversation = stripBase64ImagesFromConversation(conversation, stripOptions);
        processedConversation = truncateLargeTextInConversation(processedConversation, stripOptions);

        return processedConversation as Content[];
    }

    /**
     * Build conversation for Claude streaming.
     * Creates assistant message with tool_use blocks in Claude's ContentBlock format.
     */
    private buildClaudeStreamingConversation(
        prompt: { messages: unknown[]; system?: unknown[] },
        result: unknown[],
        toolUse: unknown[] | undefined,
        options: ExecutionOptions
    ): unknown {
        const completionResults = result as CompletionResult[];

        // Convert accumulated results to text content
        const textContent = completionResults
            .map(r => {
                switch (r.type) {
                    case 'text':
                        return r.value;
                    case 'json':
                        return typeof r.value === 'string' ? r.value : JSON.stringify(r.value);
                    case 'image':
                        return '';
                    default:
                        return String((r as any).value || '');
                }
            })
            .join('');

        // Build Claude-style ContentBlock array for assistant message
        const content: unknown[] = [];

        // Add text block if there's text content
        if (textContent) {
            content.push({
                type: 'text',
                text: textContent
            });
        }

        // Add tool_use blocks in Claude format
        if (toolUse && toolUse.length > 0) {
            for (const tool of toolUse as any[]) {
                content.push({
                    type: 'tool_use',
                    id: tool.id,
                    name: tool.tool_name,
                    input: tool.tool_input ?? {}
                });
            }
        }

        // Build assistant message
        const assistantMessage = {
            role: 'assistant',
            content: content.length > 0 ? content : [{ type: 'text', text: '' }]
        };

        // Get existing conversation or start fresh
        const existingMessages = (options.conversation as any)?.messages ?? [];
        const existingSystem = (options.conversation as any)?.system ?? prompt.system;

        // Combine: existing conversation + new prompt messages + assistant response
        const newMessages = [
            ...existingMessages,
            ...prompt.messages,
            assistantMessage
        ];

        // Build the new conversation in ClaudePrompt format
        const conversation = {
            messages: newMessages,
            system: existingSystem
        };

        // Increment turn counter
        const withTurn = incrementConversationTurn(conversation);

        // Apply stripping based on options
        const currentTurn = getConversationMeta(withTurn).turnNumber;
        const stripOptions = {
            keepForTurns: options.stripImagesAfterTurns ?? Infinity,
            currentTurn,
            textMaxTokens: options.stripTextMaxTokens
        };
        let processedConversation = stripBase64ImagesFromConversation(withTurn, stripOptions);
        processedConversation = truncateLargeTextInConversation(processedConversation, stripOptions);

        return processedConversation;
    }

    async requestImageGeneration(
        _prompt: ImagenPrompt,
        _options: ExecutionOptions,
    ): Promise<Completion> {
        const splits = _options.model.split("/");
        const modelName = trimModelName(splits[splits.length - 1]);
        return new ImagenModelDefinition(modelName).requestImageGeneration(this, _prompt, _options);
    }

    async getGenAIModelsArray(client: GoogleGenAI): Promise<Model[]> {
        const models: Model[] = [];
        const pager = await client.models.list();
        for await (const item of pager) {
            models.push(item);
        }
        return models;
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel<string>[]> {
        // Get clients
        const modelGarden = await this.getModelGardenClient();
        const aiplatform = await this.getAIPlatformClient();
        const globalGenAiClient = await this.getGoogleGenAIClient("global");

        let models: AIModel<string>[] = [];

        //Model Garden Publisher models - Pretrained models
        const publishers = ["google", "anthropic", "meta"];
        // Meta "maas" models are LLama Models-As-A-Service. Non-maas models are not pre-deployed.
        const supportedModels = { google: ["gemini", "imagen"], anthropic: ["claude"], meta: ["maas"] };
        // Additional models not in the listings, but we want to include
        // TODO: Remove once the models are available in the listing API, or no longer needed
        const additionalModels = {
            google: [
                "imagen-3.0-fast-generate-001",
            ],
            anthropic: [],
            meta: [
                "llama-4-maverick-17b-128e-instruct-maas",
                "llama-4-scout-17b-16e-instruct-maas",
                "llama-3.3-70b-instruct-maas",
                "llama-3.2-90b-vision-instruct-maas",
                "llama-3.1-405b-instruct-maas",
                "llama-3.1-70b-instruct-maas",
                "llama-3.1-8b-instruct-maas",
            ],
        }

        //Used to exclude retired models that are still in the listing API but not available for use.
        //Or models we do not support yet
        const unsupportedModelsByPublisher = {
            google: ["gemini-pro", "gemini-ultra", "imagen-product-recontext-preview", "embedding", "gemini-live-2.5-flash-preview-native-audio", "computer-use-preview"],
            anthropic: [],
            meta: [],
        };

        // Start all network requests in parallel
        const aiplatformPromise = aiplatform.listModels({
            parent: `projects/${this.options.project}/locations/${this.options.region}`,
        });
        const publisherPromises = publishers.map(async (publisher) => {
            const [response] = await modelGarden.listPublisherModels({
                parent: `publishers/${publisher}`,
                orderBy: "name",
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

        // Process aiplatform models, project specific models
        const [response] = aiplatformResult;
        models = models.concat(
            response.map((model) => ({
                id: model.name?.split("/").pop() ?? "",
                name: model.displayName ?? "",
                provider: "vertexai"
            }))
        );

        // Process global google models from GenAI
        models = models.concat(
            globalGoogleResult.map((model) => {
                const modelCapability = getModelCapabilities(model.name ?? '', "vertexai");
                return {
                    id: "locations/global/" + model.name,
                    name: "Global " + model.name?.split('/').pop(),
                    provider: "vertexai",
                    owner: "google",
                    input_modalities: modelModalitiesToArray(modelCapability.input),
                    output_modalities: modelModalitiesToArray(modelCapability.output),
                    tool_support: modelCapability.tool_support,
                };
            })
        );

        // Process publisher models
        for (const result of publisherResults) {
            const { publisher, response } = result;
            const modelFamily = supportedModels[publisher as keyof typeof supportedModels];
            const retiredModels = unsupportedModelsByPublisher[publisher as keyof typeof unsupportedModelsByPublisher];

            models = models.concat(response.filter((model) => {
                const modelName = model.name ?? "";
                // Exclude retired models
                if (retiredModels.some(retiredModel => modelName.includes(retiredModel))) {
                    return false;
                }
                // Check if the model belongs to the supported model families
                if (modelFamily.some(family => modelName.includes(family))) {
                    return true;
                }
                return false;
            }).map(model => {
                const modelCapability = getModelCapabilities(model.name ?? '', "vertexai");
                return {
                    id: model.name ?? '',
                    name: model.name?.split('/').pop() ?? '',
                    provider: 'vertexai',
                    owner: publisher,
                    input_modalities: modelModalitiesToArray(modelCapability.input),
                    output_modalities: modelModalitiesToArray(modelCapability.output),
                    tool_support: modelCapability.tool_support,
                } satisfies AIModel<string>;
            }));

            // Create global google gemini models for Gemini 2.5 and later, if missing from GenAI listing
            if (publisher === 'google') {
                const globalGeminiModels = response.filter((model) => {
                    const modelName = model.name ?? "";
                    if (retiredModels.some(retiredModel => modelName.includes(retiredModel))) {
                        return false;
                    }
                    if (modelFamily.some(family => modelName.includes(family))) {
                        const versionMatch = modelName.match(/gemini-(\d+(?:\.\d+)?)/);
                        if (versionMatch) {
                            const version = parseFloat(versionMatch[1]);
                            if (version >= 2.5) {
                                // Check if already present
                                const shortName = modelName.split('/').pop();
                                const globalName = "Global " + shortName;
                                if (models.some(m => m.name === globalName)) {
                                    return false;
                                }
                                return true;
                            }
                        }
                        return false;
                    }
                    return false;
                }).map(model => {
                    const modelCapability = getModelCapabilities(model.name ?? '', "vertexai");
                    return {
                        id: "locations/global/" + model.name,
                        name: "Global " + model.name?.split('/').pop(),
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
                const globalAnthropicModels = response.filter((model) => {
                    const modelName = model.name ?? "";
                    if (retiredModels.some(retiredModel => modelName.includes(retiredModel))) {
                        return false;
                    }
                    if (modelFamily.some(family => modelName.includes(family))) {
                        if (modelName.includes("claude-3-7")) {
                            return true;
                        }
                        return !NON_GLOBAL_ANTHROPIC_MODELS.some(nonGlobalModel => modelName.includes(nonGlobalModel));
                    }
                    return false;
                }).map(model => {
                    const modelCapability = getModelCapabilities(model.name ?? '', "vertexai");
                    return {
                        id: "locations/global/" + model.name,
                        name: "Global " + model.name?.split('/').pop(),
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
                const modelCapability = getModelCapabilities(additionalModel, "vertexai");
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

        //Remove duplicates
        const uniqueModels = Array.from(new Set(models.map(a => a.id)))
            .map(id => {
                return models.find(a => a.id === id) ?? {} as AIModel<string>;
            }).sort((a, b) => a.id.localeCompare(b.id));

        return uniqueModels;
    }

    validateConnection(): Promise<boolean> {
        throw new Error("Method not implemented.");
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        if (options.image || options.model?.includes("multimodal")) {
            if (options.text && options.image) {
                throw new Error("Text and Image simultaneous embedding not implemented. Submit separately");
            }
            return getEmbeddingsForImages(this, options);
        }
        const text_options: TextEmbeddingsOptions = {
            content: options.text ?? "",
            model: options.model,
        };
        return getEmbeddingsForText(this, text_options);
    }

    // ============== Batch Operations ==============

    /**
     * Creates a new batch job for inference or embeddings.
     * Routes to the appropriate implementation based on model and job type.
     *
     * Requires source and destination to be configured with GCS URIs.
     *
     * @example
     * ```typescript
     * const job = await driver.createBatchJob({
     *     model: "gemini-2.5-flash-lite",
     *     type: BatchJobType.inference,
     *     source: { gcsUris: ["gs://bucket/input.jsonl"] },
     *     destination: { gcsUri: "gs://bucket/output/" },
     * });
     * ```
     */
    async createBatchJob(options: CreateBatchJobOptions<GCSBatchSource, GCSBatchDestination>): Promise<BatchJob<GCSBatchSource, GCSBatchDestination>> {
        // Route to appropriate implementation
        if (options.type === BatchJobType.embeddings) {
            return createEmbeddingsBatchJobSDK(this, options);
        }
        const modelLower = options.model.toLowerCase();
        if (modelLower.includes("claude") || modelLower.includes("anthropic")) {
            return createClaudeBatchJob(this, options);
        }
        return createGeminiBatchJob(this, options);
    }

    /**
     * Gets a batch job by ID.
     * The job ID encodes the provider for routing (e.g., "gemini:...", "claude:...").
     */
    async getBatchJob(jobId: string): Promise<BatchJob<GCSBatchSource, GCSBatchDestination>> {
        const { provider, providerJobId } = decodeBatchJobId(jobId);
        switch (provider) {
            case "claude":
                return getClaudeBatchJob(this, providerJobId);
            case "embeddings":
                return getEmbeddingsBatchJob(this, providerJobId);
            case "gemini":
            default:
                return getGeminiBatchJob(this, providerJobId);
        }
    }

    /**
     * Lists batch jobs from all providers (Gemini and Claude).
     */
    async listBatchJobs(options?: ListBatchJobsOptions): Promise<ListBatchJobsResult<GCSBatchSource, GCSBatchDestination>> {
        const [geminiResult, claudeResult] = await Promise.all([
            listGeminiBatchJobs(this, options),
            listClaudeBatchJobs(this, options).catch(() => ({ jobs: [], nextPageToken: undefined })),
        ]);

        return {
            jobs: [...geminiResult.jobs, ...claudeResult.jobs],
            nextPageToken: geminiResult.nextPageToken || claudeResult.nextPageToken,
        };
    }

    /**
     * Cancels a running batch job.
     */
    async cancelBatchJob(jobId: string): Promise<BatchJob<GCSBatchSource, GCSBatchDestination>> {
        const { provider, providerJobId } = decodeBatchJobId(jobId);
        switch (provider) {
            case "claude":
                return cancelClaudeBatchJob(this, providerJobId);
            case "embeddings":
                return cancelEmbeddingsBatchJob(this, providerJobId);
            case "gemini":
            default:
                return cancelGeminiBatchJob(this, providerJobId);
        }
    }

    /**
     * Deletes a batch job.
     */
    async deleteBatchJob(jobId: string): Promise<void> {
        const { provider, providerJobId } = decodeBatchJobId(jobId);
        switch (provider) {
            case "claude":
                return deleteClaudeBatchJob(this, providerJobId);
            case "embeddings":
                return deleteEmbeddingsBatchJob(this, providerJobId);
            case "gemini":
            default:
                return deleteGeminiBatchJob(this, providerJobId);
        }
    }

    // ============== Gemini File API Methods ===============

    /**
     * Uploads a file to the Gemini File API.
     *
     * Files uploaded through this API can be used in batch operations
     * like batch embeddings. Files are automatically deleted after 48 hours.
     *
     * @param content - The file content (string, Buffer, or Blob)
     * @param mimeType - MIME type of the file (e.g., "application/jsonl")
     * @param displayName - Optional display name for the file
     * @returns The uploaded file resource with name in format "files/{fileId}"
     *
     * @example
     * ```typescript
     * const file = await driver.uploadGeminiFile(
     *     '{"content": "Hello"}\n{"content": "World"}',
     *     'application/jsonl',
     *     'my-batch-input.jsonl'
     * );
     * console.log(file.name); // "files/abc123xyz"
     * ```
     */
    async uploadGeminiFile(
        content: string | Blob,
        mimeType: string,
        displayName?: string
    ): Promise<GeminiFileResource> {
        return uploadFileToGemini(this, content, mimeType, displayName);
    }

    /**
     * Gets a file resource from the Gemini File API.
     *
     * @param fileId - The file ID (format: "files/{fileId}" or just "{fileId}")
     * @returns The file resource
     */
    async getGeminiFile(fileId: string): Promise<GeminiFileResource> {
        return getGeminiFile(this, fileId);
    }

    /**
     * Lists files from the Gemini File API.
     *
     * @param pageSize - Optional number of files to return per page
     * @param pageToken - Optional token for pagination
     * @returns List of file resources and optional next page token
     */
    async listGeminiFiles(
        pageSize?: number,
        pageToken?: string
    ): Promise<{ files: GeminiFileResource[]; nextPageToken?: string }> {
        return listGeminiFiles(this, pageSize, pageToken);
    }

    /**
     * Deletes a file from the Gemini File API.
     *
     * @param fileId - The file ID (format: "files/{fileId}" or just "{fileId}")
     */
    async deleteGeminiFile(fileId: string): Promise<void> {
        return deleteGeminiFile(this, fileId);
    }

    /**
     * Waits for a file to reach ACTIVE state.
     *
     * Files may be in PROCESSING state immediately after upload.
     * This function polls until the file is ACTIVE or FAILED.
     *
     * @param fileId - The file ID
     * @param maxWaitMs - Maximum time to wait in milliseconds (default: 60000)
     * @param pollIntervalMs - Polling interval in milliseconds (default: 1000)
     * @returns The file resource in ACTIVE state
     * @throws Error if file fails processing or timeout is reached
     */
    async waitForGeminiFileActive(
        fileId: string,
        maxWaitMs: number = 60000,
        pollIntervalMs: number = 1000
    ): Promise<GeminiFileResource> {
        return waitForFileActive(this, fileId, maxWaitMs, pollIntervalMs);
    }

    /**
     * Waits for a batch job to complete (reach a terminal state).
     *
     * @param jobId - The batch job ID to wait for
     * @param pollIntervalMs - Polling interval in milliseconds (default: 30000)
     * @param maxWaitMs - Maximum wait time in milliseconds (default: 24 hours)
     * @returns The completed batch job
     * @throws Error if timeout is reached
     */
    async waitForBatchJobCompletion(
        jobId: string,
        pollIntervalMs: number = 30000,
        maxWaitMs: number = 24 * 60 * 60 * 1000
    ): Promise<BatchJob<GCSBatchSource, GCSBatchDestination>> {
        const startTime = Date.now();

        while (true) {
            const job = await this.getBatchJob(jobId);

            if (isTerminalState(job.status)) {
                return job;
            }

            if (Date.now() - startTime > maxWaitMs) {
                throw new Error(`Batch job ${jobId} did not complete within ${maxWaitMs}ms`);
            }

            await this.sleep(pollIntervalMs);
        }
    }

    private sleep(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

//'us-central1-aiplatform.googleapis.com',
const API_BASE_PATH = "aiplatform.googleapis.com";
function createFetchClient({
    region,
    project,
    apiEndpoint,
    apiVersion = "v1",
}: {
    region: string;
    project: string;
    apiEndpoint?: string;
    apiVersion?: string;
}) {
    const vertexBaseEndpoint = apiEndpoint ?? `${region}-${API_BASE_PATH}`;
    return new FetchClient(
        `https://${vertexBaseEndpoint}/${apiVersion}/projects/${project}/locations/${region}`,
    ).withHeaders({
        "Content-Type": "application/json",
    });
}
