import { GenerateContentRequest, VertexAI } from "@google-cloud/vertexai";
import { AIModel, AbstractDriver, Completion, CompletionChunkObject, DriverOptions, EmbeddingsResult, ExecutionOptions, ImageGeneration, Modalities, ModelSearchPayload, PromptSegment } from "@llumiverse/core";
import { FetchClient } from "api-fetch-client";
import { GoogleAuth, GoogleAuthOptions } from "google-auth-library";
import { JSONClient } from "google-auth-library/build/src/auth/googleauth.js";
import { TextEmbeddingsOptions, getEmbeddingsForText } from "./embeddings/embeddings-text.js";
import { getModelDefinition } from "./models.js";
import { EmbeddingsOptions } from "@llumiverse/core";
import { getEmbeddingsForImages } from "./embeddings/embeddings-image.js";
import { v1beta1 } from '@google-cloud/aiplatform';
import { AnthropicVertex } from '@anthropic-ai/vertex-sdk';
import { ImagenModelDefinition, ImagenPrompt } from "./models/imagen.js";
import { GoogleGenAI } from "@google/genai";
import * as aiplatform from '@google-cloud/aiplatform';
const { PredictionServiceClient } = aiplatform.v1;

export interface VertexAIDriverOptions extends DriverOptions {
    project: string;
    region: string;
    googleAuthOptions?: GoogleAuthOptions;
}

//General Prompt type for VertexAI
export type VertexAIPrompt = GenerateContentRequest | ImagenPrompt;

export function trimModelName(model: string) {
    const i = model.lastIndexOf('@');
    return i > -1 ? model.substring(0, i) : model;
}

export class VertexAIDriver extends AbstractDriver<VertexAIDriverOptions, VertexAIPrompt> {
    static PROVIDER = "vertexai";
    provider = VertexAIDriver.PROVIDER;

    genai: GoogleGenAI | undefined;
    aiplatform: v1beta1.ModelServiceClient | undefined;
    modelgarden: v1beta1.ModelGardenServiceClient | undefined;
    vertexai: VertexAI | undefined;
    imagenprediction: aiplatform.v1.PredictionServiceClient | undefined;
    fetchClient: FetchClient;
    authClient: JSONClient | GoogleAuth<JSONClient>;
    anthropicClient: AnthropicVertex | undefined;
    
    constructor( options: VertexAIDriverOptions) {
        super(options);

        //Lazy initialised
        this.anthropicClient = undefined;
        this.vertexai = undefined;
        this.aiplatform = undefined;
        this.modelgarden = undefined;
        this.genai = undefined;
        this.imagenprediction = undefined;

        this.authClient = options.googleAuthOptions?.authClient ?? new GoogleAuth(options.googleAuthOptions);

        this.fetchClient = createFetchClient({
            region: this.options.region,
            project: this.options.project,
        }).withAuthCallback(async () => {
            //@ts-ignore
            const token = await this.authClient.getAccessToken();
            return `Bearer ${token}`;
        });
    }

    public getAIPlatformClient() : v1beta1.ModelServiceClient {
        //Lazy initialisation
        if (!this.aiplatform) {
            this.aiplatform = new v1beta1.ModelServiceClient({
                projectId: this.options.project,
                apiEndpoint: `${this.options.region}-${API_BASE_PATH}`,
            });
        }
        return this.aiplatform;
    }

    public getModelGardenClient() : v1beta1.ModelGardenServiceClient {
        //Lazy initialisation
        if (!this.modelgarden) {
            this.modelgarden = new v1beta1.ModelGardenServiceClient({
                projectId: this.options.project,
                apiEndpoint: `${this.options.region}-${API_BASE_PATH}`,
            });
        }
        return this.modelgarden;
    }

    public getVertexAIClient() : VertexAI {
        //Lazy initialisation
        if (!this.vertexai) {
            this.vertexai = new VertexAI({
                project: this.options.project,
                location: this.options.region,
                googleAuthOptions: this.options.googleAuthOptions,
            });
        }
        return this.vertexai;
    }

    public getGenAIClient() : GoogleGenAI {
        //Lazy initialisation
        if (!this.genai) {
            this.genai = new GoogleGenAI({
                vertexai: true,
                project: this.options.project,
                location: this.options.region,
                googleAuthOptions: this.options.googleAuthOptions,
            });
        }
        return this.genai;
    }

    public getAnthropicClient() : AnthropicVertex {
        //Lazy initialisation
        if (!this.anthropicClient) {
            this.anthropicClient = new AnthropicVertex({region: "us-east5", projectId: process.env.GOOGLE_PROJECT_ID});
        }
        return this.anthropicClient;
    }

    public getImagenClient(): aiplatform.v1.PredictionServiceClient {
        //Lazy initialisation
        if (!this.imagenprediction) {
            const location = 'us-central1';
            const clientOptions = {
                apiEndpoint: `${location}-aiplatform.googleapis.com`,
            };

            // Instantiates a client
            this.imagenprediction = new PredictionServiceClient(clientOptions);
        }
        return this.imagenprediction;
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
    async requestTextCompletionStream(prompt: VertexAIPrompt, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        return getModelDefinition(options.model).requestTextCompletionStream(this, prompt, options);
    }

    async requestImageGeneration(_prompt: ImagenPrompt, _options: ExecutionOptions): Promise <Completion<ImageGeneration>> {
        const splits = _options.model.split("/");
        const modelName = trimModelName(splits[splits.length - 1]);
        return new ImagenModelDefinition(modelName).requestImageGeneration(this, _prompt, _options);
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel<string>[]> {
        this.getAIPlatformClient();
        this.getModelGardenClient();
        let models: AIModel<string>[] = [];
    
        //Project specific deployed models
        const [response] = await this.aiplatform!.listModels({
            parent: `projects/${this.options.project}/locations/${this.options.region}`,
        });

        models = models.concat(response.map(model => ({
            id: model.name?.split('/').pop() ?? '',
            name: model.displayName ?? '',
            provider: 'vertexai',
        })));

        //Model Garden Publisher models - Pretrained models
        const publishers = ['google', 'anthropic']
        const supportedModels = {google: ['gemini','imagen'], anthropic: ['claude']}

        for (const publisher of publishers) {
            const [response] = await this.modelgarden!.listPublisherModels({
                parent: `publishers/${publisher}`,
                orderBy: 'name',
                //filter: `name eq name`,
                listAllVersions: true,
            });

            models = models.concat(response.map(model => ({
                id: model.name ?? '',
                name: model.name?.split('/').pop() ?? '',
                provider: 'vertexai',
                owner: publisher,
            })).filter(model => {
                const modelFamily = supportedModels[publisher as keyof typeof supportedModels];
                for (const family of modelFamily) {
                    if (model.name.includes(family)) {
                        return true;
                    }
                }
            }));
        }

        return models;
    }

    validateConnection(): Promise<boolean> {
        throw new Error("Method not implemented.");
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        if (options.image || options.model?.includes("multimodal")) {
            if (options.text && options.image) {
                throw new Error("Text and Image simultaneous embedding not implemented. Submit seperately");
            }
            return getEmbeddingsForImages(this, options);
        }
        const text_options: TextEmbeddingsOptions = {
            content: options.text ?? '',
            model: options.model,
        }
        return getEmbeddingsForText(this, text_options);
    }

}

//'us-central1-aiplatform.googleapis.com',
const API_BASE_PATH = 'aiplatform.googleapis.com';
function createFetchClient({ region, project, apiEndpoint, apiVersion = 'v1' }: {
    region: string;
    project: string;
    apiEndpoint?: string;
    apiVersion?: string;
}) {
    const vertexBaseEndpoint = apiEndpoint ?? `${region}-${API_BASE_PATH}`;
    return new FetchClient(`https://${vertexBaseEndpoint}/${apiVersion}/projects/${project}/locations/${region}`).withHeaders({
        'Content-Type': 'application/json',
    });
}
