import {
    AbstractDriver, AIModel,
    Completion,
    CompletionChunk, DriverOptions, EmbeddingsOptions,
    EmbeddingsResult,
    ExecutionOptions,
    ImageGeneration,
    ModelSearchPayload,
    PromptSegment
} from "@llumiverse/core";

interface FireflyImageSource {
    url?: string;
    uploadId?: string;
}

interface FireflyImageReference {
    source: FireflyImageSource;
}

interface FireflyStyle {
    presets?: string[];
    strength?: number;
    imageReference?: FireflyImageReference;
}

interface FireflyStructure {
    strength: number;
    imageReference: FireflyImageReference;
}

interface FireflySize {
    width: number;
    height: number;
}

interface FireflyGenerateRequest {
    numVariations?: number;
    seeds?: number[];
    size?: FireflySize;
    prompt: string;
    negativePrompt?: string;
    contentClass?: 'photo' | 'art' | 'graphic';
    visualIntensity?: number;
    style?: FireflyStyle;
    promptBiasingLocaleCode?: string;
    tileable?: boolean;
    structure?: FireflyStructure;
}

interface FireflyOutput {
    seed: number;
    image: {
        url: string;
    };
}

interface FireflyGenerateResponse {
    size: FireflySize;
    outputs: FireflyOutput[];
    promptHasDeniedWords?: boolean;
    promptHasBlockedArtists?: boolean;
    contentClass?: string;
}

export interface FireflyDriverOptions extends DriverOptions {
    /**
     * Adobe Firefly API key
     */
    apiKey: string;

    /**
     * Optional API endpoint override
     */
    endpoint?: string;
}

export class FireflyDriver extends AbstractDriver<FireflyDriverOptions> {
    static PROVIDER = "firefly";
    provider = FireflyDriver.PROVIDER;

    private readonly endpoint: string;

    constructor(options: FireflyDriverOptions) {
        super(options);
        
        if (!options.apiKey) {
            throw new Error("No API key provided for Firefly driver");
        }

        this.endpoint = options.endpoint || "https://firefly-api.adobe.io/v3";
    }

    async requestTextCompletion(_prompt: string, _options: ExecutionOptions): Promise<Completion> {
        throw new Error("Text completion not supported by Firefly");
    }

    async requestTextCompletionStream(_prompt: string, _options: ExecutionOptions): Promise<AsyncIterable<CompletionChunk>> {
        throw new Error("Text completion streaming not supported by Firefly");
    }

    async requestImageGeneration(segments: PromptSegment[], options: ExecutionOptions): Promise<Completion<ImageGeneration>> {
        this.logger.debug(`[${this.provider}] Generating image with model ${options.model}`);
        const prompt = segments.map(s => s.content).join("\n\n");
    

        try {
            const payload: FireflyGenerateRequest = {
                prompt: prompt as string,
            };

            const response = await fetch(`${this.endpoint}/images/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-key': this.options.apiKey,
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(`Firefly API error: ${error.message || response.statusText}`);
            }

            const result: FireflyGenerateResponse = await response.json();

            if (result.promptHasDeniedWords || result.promptHasBlockedArtists) {
                return {
                    result: {},
                    error: {
                        message: "Prompt contains denied words or blocked artists",
                        code: "content_policy_violation"
                    }
                };
            }

            return {
                result: {
                    images: result.outputs.map(output => output.image.url)
                }
            };

        } catch (error: any) {
            this.logger.error("[Firefly] Image generation failed", { error });
            return {
                result: {},
                error: {
                    message: error.message,
                    code: error.code || 'GENERATION_FAILED'
                }
            };
        }
    }

    mapSize(size?: string): FireflySize {
        // Default to 1024x1024 if no size specified
        if (!size) return { width: 1024, height: 1024 };

        const [width, height] = size.split('x').map(Number);
        return { width, height };
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel[]> {
        return [
            {
                id: "firefly-v3-text-to-image",
                name: "Firefly v3 Text to Image",
                provider: this.provider,
                description: "Adobe Firefly v3 text to image generation model",
                tags: ["image-generation"]
            },
            {
                id: "firefly-v3-image-to-image", 
                name: "Firefly v3 Image to Image",
                provider: this.provider,
                description: "Adobe Firefly v3 image to image generation model",
                tags: ["image-generation"]
            },
            {
                id: "firefly-v3-inpainting",
                name: "Firefly v3 Inpainting",
                provider: this.provider,
                description: "Adobe Firefly v3 inpainting model",
                tags: ["image-generation"]
            }
        ];
    }

    async listEmbeddingModels(): Promise<AIModel[]> {
        //Not supported
        return [];
    }

    async validateConnection(): Promise<boolean> {
        try {
            const response = await fetch(`${this.endpoint}/auth/validate`, {
                headers: {
                    'x-api-key': this.options.apiKey
                }
            });
            return response.ok;
        } catch (error) {
            this.logger.error("[Firefly] Connection validation failed", { error });
            return false;
        }
    }

    async generateEmbeddings(_options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        throw new Error("Embeddings not supported by Firefly");
    }
}