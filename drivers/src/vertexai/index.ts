import {
    AIModel,
    AbstractDriver,
    Completion,
    CompletionChunkObject,
    DriverOptions,
    EmbeddingsResult,
    ExecutionOptions,
    ImageGeneration,
    Modalities,
    ModelSearchPayload,
    PromptSegment,
    getModelCapabilities,
    modelModalitiesToArray,
} from "@llumiverse/core";
import { FetchClient } from "api-fetch-client";
import { GoogleAuth, GoogleAuthOptions } from "google-auth-library";
import { JSONClient } from "google-auth-library/build/src/auth/googleauth.js";
import { TextEmbeddingsOptions, getEmbeddingsForText } from "./embeddings/embeddings-text.js";
import { getModelDefinition } from "./models.js";
import { EmbeddingsOptions } from "@llumiverse/core";
import { getEmbeddingsForImages } from "./embeddings/embeddings-image.js";
import { v1beta1 } from "@google-cloud/aiplatform";
import { AnthropicVertex } from "@anthropic-ai/vertex-sdk";
import { ImagenModelDefinition, ImagenPrompt } from "./models/imagen.js";
import { GoogleGenAI, Content, Tool } from "@google/genai";

export interface VertexAIDriverOptions extends DriverOptions {
    project: string;
    region: string;
    googleAuthOptions?: GoogleAuthOptions;
}

export interface GenerateContentPrompt {
    contents: Content[];
    system?: string;
    tools?: Tool[];
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

    authClient: JSONClient | GoogleAuth<JSONClient>;

    constructor(options: VertexAIDriverOptions) {
        super(options);

        this.aiplatform = undefined;
        this.anthropicClient = undefined;
        this.fetchClient = undefined
        this.googleGenAI = undefined;
        this.modelGarden = undefined;
        this.llamaClient = undefined;

        this.authClient = options.googleAuthOptions?.authClient ?? new GoogleAuth(options.googleAuthOptions);
    }

    public getGoogleGenAIClient(): GoogleGenAI {
        //Lazy initialisation
        if (!this.googleGenAI) {
            this.googleGenAI = new GoogleGenAI({
                project: this.options.project,
                location: this.options.region,
                vertexai: true,
                googleAuthOptions: {
                    authClient: this.authClient as JSONClient,
                }
            });
        }
        return this.googleGenAI;
    }

    public getFetchClient(): FetchClient {
        //Lazy initialisation
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
        //Lazy initialisation
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

    public getAnthropicClient(): AnthropicVertex {
        //Lazy initialisation
        if (!this.anthropicClient) {
            this.anthropicClient = new AnthropicVertex({
                region: "us-east5",
                projectId: process.env.GOOGLE_PROJECT_ID,
            });
        }
        return this.anthropicClient;
    }

    public getAIPlatformClient(): v1beta1.ModelServiceClient {
        //Lazy initialisation
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
        //Lazy initialisation
        if (!this.modelGarden) {
            this.modelGarden = new v1beta1.ModelGardenServiceClient({
                projectId: this.options.project,
                apiEndpoint: `${this.options.region}-${API_BASE_PATH}`,
                authClient: this.authClient as JSONClient,
            });
        }
        return this.modelGarden;
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

    async requestTextCompletion(prompt: VertexAIPrompt, options: ExecutionOptions): Promise<Completion<any>> {
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
    ): Promise<Completion<ImageGeneration>> {
        const splits = _options.model.split("/");
        const modelName = trimModelName(splits[splits.length - 1]);
        return new ImagenModelDefinition(modelName).requestImageGeneration(this, _prompt, _options);
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel<string>[]> {
        // Get clients
        const modelGarden = this.getModelGardenClient();
        const aiplatform = this.getAIPlatformClient();

        let models: AIModel<string>[] = [];

        //Project specific deployed models
        const [response] = await aiplatform.listModels({
            parent: `projects/${this.options.project}/locations/${this.options.region}`,
        });
        models = models.concat(
            response.map((model) => ({
                id: model.name?.split("/").pop() ?? "",
                name: model.displayName ?? "",
                provider: "vertexai"
            })),
        );

        //Model Garden Publisher models - Pretrained models
        const publishers = ["google", "anthropic", "meta"];
        // Meta "maas" models are LLama Models-As-A-Service. Non-maas models are not pre-deployed.
        const supportedModels = { google: ["gemini", "imagen"], anthropic: ["claude"], meta: ["maas"] };
        // Additional models not in the listings, but we want to include
        // TODO: Remove once the models are available in the listing API, or no longer needed
        const additionalModels = {
            google: [],
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
            google: ["gemini-pro", "gemini-ultra"],
            anthropic: [],
            meta: [],
        };

        for (const publisher of publishers) {
            let [response] = await modelGarden.listPublisherModels({
                parent: `publishers/${publisher}`,
                orderBy: "name",
                listAllVersions: true,
            });

            // Filter out the 100+ long list coming from Google models
            if (publisher === "google") {
                response = response.filter((model) => {
                    return (model.supportedActions?.openGenerationAiStudio || undefined) !== undefined;
                });
            }

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
            });

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
