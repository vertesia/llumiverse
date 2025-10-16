import {
    AIModel,
    AbstractDriver,
    Completion,
    CompletionChunkObject,
    DriverOptions,
    EmbeddingsResult,
    ExecutionOptions,
    Modalities,
    ModelSearchPayload,
    PromptSegment,
    getModelCapabilities,
    modelModalitiesToArray,
} from "@llumiverse/core";
import { FetchClient } from "@vertesia/api-fetch-client";
import { GoogleAuth, GoogleAuthOptions } from "google-auth-library";
import { JSONClient } from "google-auth-library/build/src/auth/googleauth.js";
import { TextEmbeddingsOptions, getEmbeddingsForText } from "./embeddings/embeddings-text.js";
import { getModelDefinition } from "./models.js";
import { EmbeddingsOptions } from "@llumiverse/core";
import { getEmbeddingsForImages } from "./embeddings/embeddings-image.js";
import { PredictionServiceClient, v1beta1 } from "@google-cloud/aiplatform";
import { AnthropicVertex } from "@anthropic-ai/vertex-sdk";
import { ImagenModelDefinition, ImagenPrompt } from "./models/imagen.js";
import { GoogleGenAI, Content, Model } from "@google/genai";
import { NON_GLOBAL_ANTHROPIC_MODELS, ANTHROPIC_REGIONS } from "./models/claude.js";

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

    aiplatform: v1beta1.ModelServiceClient | undefined;
    anthropicClient: AnthropicVertex | undefined;
    fetchClient: FetchClient | undefined;
    googleGenAI: GoogleGenAI | undefined;
    llamaClient: FetchClient & { region?: string } | undefined;
    modelGarden: v1beta1.ModelGardenServiceClient | undefined;
    imagenClient: PredictionServiceClient | undefined;

    authClient: JSONClient | GoogleAuth<JSONClient>;

    constructor(options: VertexAIDriverOptions) {
        super(options);

        this.aiplatform = undefined;
        this.anthropicClient = undefined;
        this.fetchClient = undefined
        this.googleGenAI = undefined;
        this.modelGarden = undefined;
        this.llamaClient = undefined;
        this.imagenClient = undefined;

        this.authClient = options.googleAuthOptions?.authClient ?? new GoogleAuth(options.googleAuthOptions);
    }

    public getGoogleGenAIClient(region: string = this.options.region): GoogleGenAI {
        //Lazy initialization
        if (region !== this.options.region) {
            //Get one off client for different region
            return new GoogleGenAI({
                project: this.options.project,
                location: region,
                vertexai: true,
                googleAuthOptions: {
                    authClient: this.authClient as JSONClient,
                }
            });
        }
        if (!this.googleGenAI) {
            this.googleGenAI = new GoogleGenAI({
                project: this.options.project,
                location: region,
                vertexai: true,
                googleAuthOptions: {
                    authClient: this.authClient as JSONClient,
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
                const accessTokenResponse = await this.authClient.getAccessToken();
                const token = typeof accessTokenResponse === 'string' ? accessTokenResponse : accessTokenResponse?.token;
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
                const accessTokenResponse = await this.authClient.getAccessToken();
                const token = typeof accessTokenResponse === 'string' ? accessTokenResponse : accessTokenResponse?.token;
                return `Bearer ${token}`;
            });
            // Store the region for potential client reuse
            this.llamaClient["region"] = region;
        }
        return this.llamaClient;
    }

    public getAnthropicClient(region: string = this.options.region): AnthropicVertex {
        // Extract region prefix and map if it exists in ANTHROPIC_REGIONS, otherwise use as-is
        const getRegionPrefix = (r: string) => r.split('-')[0];
        const regionPrefix = getRegionPrefix(region);
        const mappedRegion = ANTHROPIC_REGIONS[regionPrefix] || region;

        const defaultRegionPrefix = getRegionPrefix(this.options.region);
        const defaultMappedRegion = ANTHROPIC_REGIONS[defaultRegionPrefix] || this.options.region;

        // If mapped region is different from default mapped region, create one-off client
        if (mappedRegion !== defaultMappedRegion) {
            return new AnthropicVertex({
                timeout: 20 * 60 * 10000, // Set to 20 minutes, 10 minute default, setting this disables long request error: https://github.com/anthropics/anthropic-sdk-typescript?#long-requests
                region: mappedRegion,
                projectId: this.options.project,
                googleAuth: new GoogleAuth({
                    scopes: ["https://www.googleapis.com/auth/cloud-platform"],
                    authClient: this.authClient as JSONClient,
                    projectId: this.options.project,
                }),
            });
        }

        //Lazy initialization for default region
        if (!this.anthropicClient) {
            this.anthropicClient = new AnthropicVertex({
                timeout: 20 * 60 * 10000, // Set to 20 minutes, 10 minute default, setting this disables long request error: https://github.com/anthropics/anthropic-sdk-typescript?#long-requests
                region: mappedRegion,
                projectId: this.options.project,
                googleAuth: new GoogleAuth({
                    scopes: ["https://www.googleapis.com/auth/cloud-platform"],
                    authClient: this.authClient as JSONClient,
                    projectId: this.options.project,
                }),
            });
        }
        return this.anthropicClient;
    }

    public getAIPlatformClient(): v1beta1.ModelServiceClient {
        //Lazy initialization
        if (!this.aiplatform) {
            this.aiplatform = new v1beta1.ModelServiceClient({
                projectId: this.options.project,
                apiEndpoint: `${this.options.region}-${API_BASE_PATH}`,
                authClient: this.authClient as JSONClient,
            });
        }
        return this.aiplatform;
    }

    public getModelGardenClient(): v1beta1.ModelGardenServiceClient {
        //Lazy initialization
        if (!this.modelGarden) {
            this.modelGarden = new v1beta1.ModelGardenServiceClient({
                projectId: this.options.project,
                apiEndpoint: `${this.options.region}-${API_BASE_PATH}`,
                authClient: this.authClient as JSONClient,
            });
        }
        return this.modelGarden;
    }

    public getImagenClient(): PredictionServiceClient {
        //Lazy initialization
        if (!this.imagenClient) {
            // TODO: make location configurable, fixed to us-central1 for now
            this.imagenClient = new PredictionServiceClient({
                projectId: this.options.project,
                apiEndpoint: `us-central1-${API_BASE_PATH}`,
                authClient: this.authClient as JSONClient,
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
        if (options.output_modality == Modalities.image) {
            return Promise.resolve(false);
        }
        return Promise.resolve(getModelDefinition(options.model).model.can_stream === true);
    }

    public createPrompt(segments: PromptSegment[], options: ExecutionOptions): Promise<VertexAIPrompt> {
        if (options.model.includes("imagen")) {
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
        const modelGarden = this.getModelGardenClient();
        const aiplatform = this.getAIPlatformClient();
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
            google: ["gemini-pro", "gemini-ultra", "imagen-product-recontext-preview", "embedding"],
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
