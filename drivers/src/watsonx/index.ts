import { AbstractDriver, type AIModel, type Completion, type CompletionChunkObject, type DriverOptions, type EmbeddingsOptions, type EmbeddingsResult, type ExecutionOptions, LlumiverseError, normalizeEmbeddingsOptions, type TextFallbackOptions, WATSONX_DEFAULT_EMBEDDING_MODEL } from "@llumiverse/core";
import { transformSSEStream } from "@llumiverse/core/async";
import { FetchClient } from "@vertesia/api-fetch-client";
import type { GenerateEmbeddingPayload, GenerateEmbeddingResponse, WatsonAuthToken, WatsonxListModelResponse, WatsonxModelSpec, WatsonxTextGenerationPayload, WatsonxTextGenerationResponse } from "./interfaces.js";

interface WatsonxDriverOptions extends DriverOptions {
    apiKey: string;
    projectId: string;
    endpointUrl: string;
}

const API_VERSION = "2024-03-14"

export class WatsonxDriver extends AbstractDriver<WatsonxDriverOptions, string> {
    static PROVIDER = "watsonx";
    provider = WatsonxDriver.PROVIDER;
    apiKey: string;
    endpoint_url: string;
    projectId: string;
    authToken?: WatsonAuthToken;
    fetcher?: FetchClient;
    fetchClient: FetchClient

    constructor(options: WatsonxDriverOptions) {
        super(options);
        this.apiKey = options.apiKey;
        this.projectId = options.projectId;
        this.endpoint_url = options.endpointUrl;
        this.fetchClient = new FetchClient(this.endpoint_url).withAuthCallback(async () => this.getAuthToken().then(token => `Bearer ${token}`));
    }

    async requestTextCompletion(prompt: string, options: ExecutionOptions): Promise<Completion> {
        if (options.model_options?._option_id !== undefined && options.model_options?._option_id !== "text-fallback") {
            this.logger.debug({ options: options.model_options }, "Unexpected option id");
        }
        options.model_options = options.model_options as TextFallbackOptions | undefined;

        const payload: WatsonxTextGenerationPayload = {
            model_id: options.model,
            input: prompt + "\n",
            parameters: {
                max_new_tokens: options.model_options?.max_tokens,
                temperature: options.model_options?.temperature,
                top_k: options.model_options?.top_k,
                top_p: options.model_options?.top_p,
                stop_sequences: options.model_options?.stop_sequence,
            },
            project_id: this.projectId,
        }

        const res = await this.fetchClient.post(`/ml/v1/text/generation?version=${API_VERSION}`, { payload }) as WatsonxTextGenerationResponse;

        const result = res.results[0];

        return {
            result: [{ type: "text", value: result.generated_text }],
            token_usage: {
                prompt: result.input_token_count,
                result: result.generated_token_count,
                total: result.input_token_count + result.generated_token_count,
            },
            finish_reason: watsonFinishReason(result.stop_reason),
            original_response: options.include_original_response ? res : undefined,
        }
    }

    async requestTextCompletionStream(prompt: string, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        if (options.model_options?._option_id !== undefined && options.model_options?._option_id !== "text-fallback") {
            this.logger.debug({ options: options.model_options }, "Unexpected option id");
        }
        options.model_options = options.model_options as TextFallbackOptions | undefined;
        const payload: WatsonxTextGenerationPayload = {
            model_id: options.model,
            input: prompt + "\n",
            parameters: {
                max_new_tokens: options.model_options?.max_tokens,
                temperature: options.model_options?.temperature,
                top_k: options.model_options?.top_k,
                top_p: options.model_options?.top_p,
                stop_sequences: options.model_options?.stop_sequence,
            },
            project_id: this.projectId,
        }

        const stream = await this.fetchClient.post(`/ml/v1/text/generation_stream?version=${API_VERSION}`, {
            payload: payload,
            reader: 'sse'
        })

        return transformSSEStream(stream, (data: string) => {
            const json = JSON.parse(data) as WatsonxTextGenerationResponse;
            return {
                result: json.results[0]?.generated_text ? [{ type: "text", value: json.results[0].generated_text }] : [],
                finish_reason: watsonFinishReason(json.results[0]?.stop_reason),
                token_usage: {
                    prompt: json.results[0].input_token_count,
                    result: json.results[0].generated_token_count,
                    total: json.results[0].input_token_count + json.results[0].generated_token_count,
                },
            };
        });

    }



    async listModels(): Promise<AIModel<string>[]> {



        const res = await this.fetchClient.get(`/ml/v1/foundation_model_specs?version=${API_VERSION}`)
            .catch(err => this.logger.warn("Can't list models on Watsonx: " + err)) as WatsonxListModelResponse;

        const aiModels = res.resources.map((m: WatsonxModelSpec) => {
            return {
                id: m.model_id,
                name: m.label,
                description: m.short_description,
                provider: this.provider,
            }
        });

        return aiModels;

    }

    async getAuthToken(): Promise<string> {


        if (this.authToken) {
            const now = Date.now() / 1000;
            if (now < this.authToken.expiration) {
                return this.authToken.access_token;
            } else {
                this.logger.debug("Token expired, refetching")
            }
        }
        const authToken = await fetch('https://iam.cloud.ibm.com/identity/token', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=${this.apiKey}`,
        }).then(response => response.json()) as WatsonAuthToken;

        this.authToken = authToken;
        return this.authToken.access_token;

    }

    async validateConnection(): Promise<boolean> {
        return this.listModels()
            .then(() => true)
            .catch((err) => {
                this.logger.warn({ error: err }, "Failed to connect to WatsonX");
                return false
            });
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        const normalized = normalizeEmbeddingsOptions(options);
        const texts = normalized.inputs.map((input) => {
            if (input.type !== "text") {
                throw new Error(`Provider 'watsonx' does not support '${input.type}' embeddings; only 'text' is supported.`);
            }
            return input.text;
        });

        const payload: GenerateEmbeddingPayload = {
            inputs: texts,
            model_id: normalized.model ?? WATSONX_DEFAULT_EMBEDDING_MODEL,
            project_id: this.projectId,
        };

        const model = payload.model_id;
        try {
            const res = await this.fetchClient.post(`/ml/v1/text/embeddings?version=${API_VERSION}`, { payload }) as GenerateEmbeddingResponse;
            return {
                model: res.model_id,
                results: res.results.map((entry) => ({
                    outputs: [{ values: entry.embedding, modality: "text" }],
                })),
            };
        } catch (error) {
            if (LlumiverseError.isLlumiverseError(error)) throw error;
            if (error instanceof Error && typeof (error as any).status !== 'number') throw error;
            throw this.formatLlumiverseError(error, {
                provider: this.provider,
                model,
                operation: 'execute',
            });
        }
    }

}

function watsonFinishReason(reason: string | undefined) {
    if (!reason) return undefined;
    switch (reason) {
        case 'eos_token': return "stop";
        case 'max_tokens': return "length";
        default: return reason;
    }
}



/*interface ListModelsParams extends ModelSearchPayload {
    limit?: number;
}*/
