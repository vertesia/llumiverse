import { DefaultAzureCredential, getBearerTokenProvider, TokenCredential } from "@azure/identity";
import { AbstractDriver, AIModel, Completion, CompletionChunk, DriverOptions, EmbeddingsOptions, EmbeddingsResult, ExecutionOptions, getModelCapabilities, modelModalitiesToArray, Providers } from "@llumiverse/core";
import { AIProjectClient, DeploymentUnion, ModelDeployment } from '@azure/ai-projects';
import { isUnexpected } from "@azure-rest/ai-inference";
import { ChatCompletionMessageParam } from "openai/resources";
import type {
    ChatCompletionsOutput,
    ChatCompletionsToolCall,
    ChatRequestMessage,
} from "@azure-rest/ai-inference";
import { AzureOpenAIDriver } from "../openai/azure_openai.js";
import { createSseStream, NodeJSReadableStream } from "@azure/core-sse";
import { formatOpenAILikeMultimodalPrompt } from "../openai/openai_format.js";
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
    messages: ChatCompletionMessageParam[]
}

export type AzureFoundryPrompt = AzureFoundryInferencePrompt | AzureFoundryOpenAIPrompt

export class AzureFoundryDriver extends AbstractDriver<AzureFoundryDriverOptions, ChatCompletionMessageParam[]> {
    service: AIProjectClient;
    readonly provider = Providers.azure_foundry;

    OPENAI_API_VERSION = "2025-01-01-preview";
    INFERENCE_API_VERSION = "2024-05-01-preview";

    constructor(opts: AzureFoundryDriverOptions) {
        super(opts);

        this.formatPrompt = formatOpenAILikeMultimodalPrompt;

        if (!opts.endpoint) {
            throw new Error("Azure AI Foundry endpoint is required");
        }

        try {
            if (!opts.azureADTokenProvider) {
                // Using Microsoft Entra ID (Azure AD) for authentication
                opts.azureADTokenProvider = new DefaultAzureCredential();
            }
        } catch (error) {
            this.logger.error("Failed to initialize Azure AD token provider:", error);
            throw new Error("Failed to initialize Azure AD token provider");
        }

        // Initialize AI Projects client which provides access to inference operations
        this.service = new AIProjectClient(
            opts.endpoint,
            opts.azureADTokenProvider
        );

        if (opts.apiVersion) {
            this.OPENAI_API_VERSION = opts.apiVersion;
            this.INFERENCE_API_VERSION = opts.apiVersion;
            this.logger.info(`[Azure Foundry] Overriding default API version, using API version: ${opts.apiVersion}`);
        }
    }

    /**
     * Get default authentication for Azure AI Foundry API
     */
    getDefaultAIFoundryAuth() {
        const scope = "https://ai.azure.com/.default";
        const azureADTokenProvider = getBearerTokenProvider(new DefaultAzureCredential(), scope);
        return azureADTokenProvider;
    }

    async isOpenAIDeployment(model: string): Promise<boolean> {
        const { deploymentName } = parseAzureFoundryModelId(model);

        let deployment = undefined;
        // First, verify the deployment exists
        try {
            deployment = await this.service.deployments.get(deploymentName);
            this.logger.debug(`[Azure Foundry] Deployment ${deploymentName} found`);
        } catch (deploymentError) {
            this.logger.error(`[Azure Foundry] Deployment ${deploymentName} not found:`, deploymentError);
        }

        return (deployment as ModelDeployment).modelPublisher == "OpenAI";
    }

    protected canStream(_options: ExecutionOptions): Promise<boolean> {
        return Promise.resolve(true);
    }

    async requestTextCompletion(prompt: ChatCompletionMessageParam[], options: ExecutionOptions): Promise<Completion> {
        const { deploymentName } = parseAzureFoundryModelId(options.model);
        const model_options = options.model_options as any;
        const isOpenAI = await this.isOpenAIDeployment(options.model);

        let response;
        if (isOpenAI) {
            // Use the Azure OpenAI client for OpenAI models
            const azureOpenAI = await this.service.getAzureOpenAIClient({ apiVersion: this.OPENAI_API_VERSION });
            const subDriver = new AzureOpenAIDriver(azureOpenAI);
            // Use deployment name for API calls
            const modifiedOptions = { ...options, model: deploymentName };
            const response = await subDriver.requestTextCompletion(prompt, modifiedOptions);
            return response;

        } else {
            // Use the chat completions client from the inference operations
            const chatClient = this.service.inference.chatCompletions({ apiVersion: this.INFERENCE_API_VERSION });
            response = await chatClient.post({
                body: {
                    messages: prompt,
                    max_tokens: model_options?.max_tokens,
                    model: deploymentName,
                    stream: true,
                    temperature: model_options?.temperature,
                    top_p: model_options?.top_p,
                    frequency_penalty: model_options?.frequency_penalty,
                    presence_penalty: model_options?.presence_penalty,
                    stop: model_options?.stop_sequence,
                }
            });
            if (response.status !== "200") {
                this.logger.error(`[Azure Foundry] Chat completion request failed:`, response);
                throw new Error(`Chat completion request failed with status ${response.status}: ${response.body}`);
            }

            return this.extractDataFromResponse(response.body as ChatCompletionsOutput);
        }
    }

    async requestTextCompletionStream(prompt: ChatCompletionMessageParam[], options: ExecutionOptions): Promise<AsyncIterable<CompletionChunk>> {
        const { deploymentName } = parseAzureFoundryModelId(options.model);
        const model_options = options.model_options as any;
        const isOpenAI = await this.isOpenAIDeployment(options.model);

        if (isOpenAI) {
            const azureOpenAI = await this.service.getAzureOpenAIClient({ apiVersion: this.OPENAI_API_VERSION });
            const subDriver = new AzureOpenAIDriver(azureOpenAI);
            const modifiedOptions = { ...options, model: deploymentName };
            const stream = await subDriver.requestTextCompletionStream(prompt, modifiedOptions);
            return stream;
        } else {
            const chatClient = this.service.inference.chatCompletions({ apiVersion: this.INFERENCE_API_VERSION });
            const response = await chatClient.post({
                body: {
                    messages: prompt,
                    max_tokens: model_options?.max_tokens,
                    model: deploymentName,
                    stream: true,
                    temperature: model_options?.temperature,
                    top_p: model_options?.top_p,
                    frequency_penalty: model_options?.frequency_penalty,
                    presence_penalty: model_options?.presence_penalty,
                    stop: model_options?.stop_sequence,
                }
            }).asNodeStream();

            // We type assert from NodeJS.ReadableStream to NodeJSReadableStream
            // The Azure Examples, expect a .destroy() method on the stream
            const stream = response.body as NodeJSReadableStream;
            if (!stream) {
                throw new Error("The response stream is undefined");
            }

            if (response.status !== "200") {
                stream.destroy();
                throw new Error(`Failed to get chat completions, http operation failed with ${response.status} code`);
            }

            const sseStream = createSseStream(stream);

            return this.processStreamResponse(sseStream);
        }
    }

    private async *processStreamResponse(sseStream: any): AsyncIterable<CompletionChunk> {
        try {
            for await (const event of sseStream) {
                if (event.data === "[DONE]") {
                    break;
                }

                try {
                    const data = JSON.parse(event.data);
                    if (!data) {
                        this.logger.warn(`[Azure Foundry] Received empty data in streaming response`);
                        continue;
                    }
                    const choice = data.choices?.[0];
                    if (!choice) {
                        continue;
                    }
                    const chunk: CompletionChunk = {
                        result: choice.delta?.content || "",
                        finish_reason: this.convertFinishReason(choice.finish_reason),
                        token_usage: {
                            prompt: data.usage?.prompt_tokens,
                            result: data.usage?.completion_tokens,
                            total: data.usage?.total_tokens,
                        },
                    };

                    yield chunk;
                } catch (parseError) {
                    this.logger.warn(`[Azure Foundry] Failed to parse streaming response:`, parseError);
                    continue;
                }
            }
        } catch (error) {
            this.logger.error(`[Azure Foundry] Streaming error:`, error);
            throw error;
        }
    }


    private extractDataFromResponse(result: ChatCompletionsOutput): Completion {
        const tokenInfo = {
            prompt: result.usage?.prompt_tokens,
            result: result.usage?.completion_tokens,
            total: result.usage?.total_tokens,
        };

        const choice = result.choices?.[0];
        if (!choice) {
            this.logger?.error("[Azure Foundry] No choices in response", result);
            throw new Error("No choices in response");
        }

        const data = choice.message?.content;
        const toolCalls = choice.message?.tool_calls;

        if (!data && !toolCalls) {
            this.logger?.error("[Azure Foundry] Response is not valid", result);
            throw new Error("Response is not valid: no content or tool calls");
        }

        const completion: Completion = {
            result: data,
            token_usage: tokenInfo,
            finish_reason: this.convertFinishReason(choice.finish_reason),
        };

        if (toolCalls && toolCalls.length > 0) {
            completion.tool_use = toolCalls.map((call: ChatCompletionsToolCall) => ({
                id: call.id,
                tool_name: call.function?.name,
                tool_input: call.function?.arguments ? JSON.parse(call.function.arguments) : {}
            }));
        }

        return completion;
    }

    private convertFinishReason(reason: string | null | undefined): string | undefined {
        if (!reason) return undefined;
        // Map Azure AI finish reasons to standard format
        switch (reason) {
            case 'stop': return 'stop';
            case 'length': return 'length';
            case 'tool_calls': return 'tool_use';
            default: return reason;
        }
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
                this.logger.warn("[Azure Foundry] No deployments found in the project");
            }

            return true;
        } catch (error) {
            this.logger.error("Azure Foundry connection validation failed:", error);
            return false;
        }
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        if (!options.model) {
            throw new Error("Default embedding model selection not supported for Azure Foundry. Please specify a model.");
        }

        if (options.text) {
            return this.generateTextEmbeddings(options);
        } else if (options.image) {
            return this.generateImageEmbeddings(options);
        } else {
            throw new Error("No text or images provided for embeddings");
        }
    }

    async generateTextEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        if (!options.text) {
            throw new Error("No text provided for text embeddings");
        }

        const { deploymentName } = parseAzureFoundryModelId(options.model || "");

        let response;
        try {
            // Use the embeddings client from the inference operations
            const embeddingsClient = this.service.inference.embeddings({ apiVersion: this.INFERENCE_API_VERSION });
            response = await embeddingsClient.post({
                body: {
                    input: Array.isArray(options.text) ? options.text : [options.text],
                    model: deploymentName
                }
            });
        } catch (error) {
            this.logger.error("Azure Foundry text embeddings error:", error);
            throw error;
        }

        if (isUnexpected(response)) {
            throw new Error(`Text embeddings request failed: ${response.status} ${response.body?.error?.message || 'Unknown error'}`);
        }

        const embeddings = response.body.data?.[0]?.embedding;
        if (!embeddings || !Array.isArray(embeddings) || embeddings.length === 0) {
            throw new Error("No valid embedding array found in response");
        }

        return {
            values: embeddings,
            model: options.model ?? ""
        };
    }

    async generateImageEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        if (!options.image) {
            throw new Error("No images provided for image embeddings");
        }

        const { deploymentName } = parseAzureFoundryModelId(options.model || "");

        let response;
        try {
            // Use the embeddings client from the inference operations
            const embeddingsClient = this.service.inference.embeddings({ apiVersion: this.INFERENCE_API_VERSION });
            response = await embeddingsClient.post({
                body: {
                    input: Array.isArray(options.image) ? options.image : [options.image],
                    model: deploymentName
                }
            });
        } catch (error) {
            this.logger.error("Azure Foundry image embeddings error:", error);
            throw error;
        }
        if (isUnexpected(response)) {
            throw new Error(`Image embeddings request failed: ${response.status} ${response.body?.error?.message || 'Unknown error'}`);
        }
        const embeddings = response.body.data?.[0]?.embedding;
        if (!embeddings || !Array.isArray(embeddings) || embeddings.length === 0) {
            throw new Error("No valid embedding array found in response");
        }
        return {
            values: embeddings,
            model: options.model ?? ""
        };
    }

    async listModels(): Promise<AIModel[]> {
        const filter = (m: ModelDeployment) => {
            // Only include models that support chat completions
            return !!m.capabilities.chat_completion;
        };
        return this._listModels(filter);
    }

    async _listModels(filter?: (m: ModelDeployment) => boolean): Promise<AIModel[]> {
        let deploymentsIterable;
        try {
            // List all deployments in the Azure AI Foundry project
            deploymentsIterable = this.service.deployments.list();
        } catch (error) {
            this.logger.error("Failed to list deployments:", error);
            throw new Error("Failed to list deployments in Azure AI Foundry project");
        }
        const deployments: DeploymentUnion[] = [];

        for await (const page of deploymentsIterable.byPage()) {
            for (const deployment of page) {
                deployments.push(deployment);
            }
        }

        let modelDeployments: ModelDeployment[] = deployments.filter((d): d is ModelDeployment => {
            return d.type === "ModelDeployment";
        });

        if (filter) {
            modelDeployments = modelDeployments.filter(filter);
        }

        const aiModels = modelDeployments.map((model) => {
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
        }).sort((modelA, modelB) => modelA.id.localeCompare(modelB.id));

        return aiModels;
    }
}

// Helper functions to parse the composite ID
export function parseAzureFoundryModelId(compositeId: string): { deploymentName: string; baseModel: string } {
    const parts = compositeId.split('::');
    if (parts.length === 2) {
        return {
            deploymentName: parts[0],
            baseModel: parts[1]
        };
    }

    // Backwards compatibility: if no delimiter found, treat as deployment name
    return {
        deploymentName: compositeId,
        baseModel: compositeId
    };
}

export function isCompositeModelId(modelId: string): boolean {
    return modelId.includes('::');
}
