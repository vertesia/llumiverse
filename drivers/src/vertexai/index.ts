import type { ClientOptions as AnthropicVertexClientOptions } from "@anthropic-ai/vertex-sdk";
import { AnthropicVertex } from "@anthropic-ai/vertex-sdk";
import { Content, GoogleGenAI, Model } from "@google/genai";
import {
    AIModel,
    AbstractDriver,
    Completion,
    CompletionChunkObject,
    CompletionResult,
    DriverOptions,
    EmbeddingsOptions,
    EmbeddingsResult,
    ExecutionOptions,
    LlumiverseError,
    LlumiverseErrorContext,
    ModelSearchPayload,
    PromptSegment,
    getConversationMeta,
    getModelCapabilities,
    incrementConversationTurn,
    modelModalitiesToArray,
    stripBase64ImagesFromConversation,
    stripHeartbeatsFromConversation,
    truncateLargeTextInConversation,
} from "@llumiverse/core";
import { FetchClient } from "@vertesia/api-fetch-client";
import { AuthClient, GoogleAuth, GoogleAuthOptions } from "google-auth-library";
import { getEmbeddingsForImages } from "./embeddings/embeddings-image.js";
import { TextEmbeddingsOptions, getEmbeddingsForText } from "./embeddings/embeddings-text.js";
import { getModelDefinition } from "./models.js";
import { ANTHROPIC_REGIONS, NON_GLOBAL_ANTHROPIC_MODELS } from "./models/claude.js";
import { ImagenModelDefinition, ImagenPrompt } from "./models/imagen.js";

export interface VertexAIDriverOptions extends DriverOptions {
    project: string;
    region: string;
    googleAuthOptions?: GoogleAuthOptions;
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

export class VertexAIDriver extends AbstractDriver<VertexAIDriverOptions, VertexAIPrompt> {
    static PROVIDER = "vertexai";
    provider = VertexAIDriver.PROVIDER;

    anthropicClient: AnthropicVertex | undefined;
    fetchClient: FetchClient | undefined;
    googleGenAI: GoogleGenAI | undefined;
    llamaClient: FetchClient & { region?: string } | undefined;

    googleAuth: GoogleAuth<any>;
    private authClientPromise: Promise<AuthClient> | undefined;

    constructor(options: VertexAIDriverOptions) {
        super(options);

        this.anthropicClient = undefined;
        this.fetchClient = undefined
        this.googleGenAI = undefined;
        this.llamaClient = undefined;

        this.googleAuth = new GoogleAuth(options.googleAuthOptions) as GoogleAuth<any>;
        this.authClientPromise = undefined;
    }

    private async getAuthClient(): Promise<AuthClient> {
        if (!this.authClientPromise) {
            this.authClientPromise = this.googleAuth.getClient();
        }
        return this.authClientPromise;
    }

    public getGoogleGenAIClient(region: string = this.options.region): GoogleGenAI {
        //Lazy initialization
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

    /**
     * List models from the AI Platform (project-specific models).
     * Uses REST API instead of gRPC to avoid the heavy @google-cloud/aiplatform package (108MB).
     */
    public async listAIPlatformModels(): Promise<{ name?: string; displayName?: string }[]> {
        const client = this.getRestClient(this.options.region);
        const response = await client.get(`/models`) as { models?: { name?: string; displayName?: string }[] };
        return response.models ?? [];
    }

    /**
     * List publisher models from the Model Garden.
     * Uses REST API (v1beta1) instead of gRPC.
     */
    public async listPublisherModels(publisher: string): Promise<{ name?: string }[]> {
        const token = await this.googleAuth.getAccessToken();
        const url = `https://${this.options.region}-${API_BASE_PATH}/v1beta1/publishers/${publisher}/models?orderBy=name&view=PUBLISHER_MODEL_VIEW_BASIC`;
        const response = await fetch(url, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
            },
        });
        if (!response.ok) {
            throw new Error(`Failed to list publisher models for ${publisher}: ${response.status} ${response.statusText}`);
        }
        const data = await response.json() as { publisherModels?: { name?: string }[] };
        return data.publisherModels ?? [];
    }

    /**
     * Send a predict request (used for Imagen image generation).
     * Uses REST API instead of gRPC PredictionServiceClient.
     */
    public async predict(endpoint: string, instances: unknown[], parameters: unknown, timeoutMs?: number): Promise<{ predictions?: unknown[] }> {
        const location = 'us-central1'; // TODO: make configurable
        const token = await this.googleAuth.getAccessToken();
        const url = `https://${location}-${API_BASE_PATH}/v1/${endpoint}:predict`;
        const controller = new AbortController();
        const timeout = timeoutMs ? setTimeout(() => controller.abort(), timeoutMs) : undefined;
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ instances, parameters }),
                signal: controller.signal,
            });
            if (!response.ok) {
                const body = await response.text();
                throw new Error(`Predict request failed: ${response.status} ${response.statusText}: ${body}`);
            }
            return await response.json() as { predictions?: unknown[] };
        } finally {
            if (timeout) clearTimeout(timeout);
        }
    }

    /**
     * Get a REST client for a specific region (reuses getFetchClient pattern).
     */
    private getRestClient(region: string): FetchClient {
        return createFetchClient({
            region,
            project: this.options.project,
        }).withAuthCallback(async () => {
            const token = await this.googleAuth.getAccessToken();
            return `Bearer ${token}`;
        });
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
                const functionCallPart: any = {
                    functionCall: {
                        name: tool.tool_name,
                        args: tool.tool_input,
                    }
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
        let conversation: Content[] = [
            ...prompt.contents,
        ];

        // Only add assistant message if there's actual content
        // (Empty text parts can cause API errors)
        if (parts.length > 0) {
            conversation.push({
                role: 'model',
                parts: parts
            });
        }

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
                processedConversation = { ...processedConversation as object, _llumiverse_system: geminiPrompt.system };
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

        // Claude's requestTextCompletionStream does NOT mutate prompt.messages
        // to include history, so we must prepend options.conversation here.
        const existingMessages = (options.conversation as any)?.messages ?? [];
        const existingSystem = (options.conversation as any)?.system ?? prompt.system;

        // Build the new messages array
        const newMessages = [
            ...existingMessages,
            ...prompt.messages,
        ];

        // Only add assistant message if there's actual content
        // (Claude API rejects empty text content blocks)
        if (content.length > 0) {
            newMessages.push({
                role: 'assistant',
                content: content
            });
        }

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
        processedConversation = stripHeartbeatsFromConversation(processedConversation, {
            keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
            currentTurn,
        });

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
        const globalGenAiClient = this.getGoogleGenAIClient("global");

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

        // Start all network requests in parallel using REST APIs
        const aiplatformPromise = this.listAIPlatformModels();
        const publisherPromises = publishers.map(async (publisher) => {
            const response = await this.listPublisherModels(publisher);
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
        models = models.concat(
            aiplatformResult.map((model) => ({
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

    /**
     * Cleanup Google Cloud clients when the driver is evicted from the cache.
     * REST-based clients don't need explicit cleanup.
     */
    destroy(): void {
        // No gRPC clients to close — all API calls use REST/fetch
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
    public formatLlumiverseError(
        error: unknown,
        context: LlumiverseErrorContext
    ): LlumiverseError {
        // Get the model definition for this request
        const modelDef = getModelDefinition(context.model);

        // If the model definition provides custom error handling, use it
        if (modelDef.formatLlumiverseError) {
            try {
                return modelDef.formatLlumiverseError(this, error, context);
            } catch (formattingError) {
                // If model-specific handler throws, fall through to default handling
                // This allows model handlers to explicitly opt out for certain errors
            }
        }

        // Fall back to default AbstractDriver error handling
        return super.formatLlumiverseError(error, context);
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
