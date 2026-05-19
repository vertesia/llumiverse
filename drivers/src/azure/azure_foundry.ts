import type {
    ChatCompletionsOutput,
    ChatCompletionsToolCall,
    ChatRequestMessage,
} from "@azure-rest/ai-inference";
import { isUnexpected } from "@azure-rest/ai-inference";
import { AIProjectClient, type DeploymentUnion, type ModelDeployment } from '@azure/ai-projects';
import { createSseStream, type NodeJSReadableStream } from "@azure/core-sse";
import { DefaultAzureCredential, getBearerTokenProvider, type TokenCredential } from "@azure/identity";
import { AbstractDriver, type AIModel, type Completion, type CompletionChunkObject, dataSourceToBase64, type DriverOptions, type EmbeddingResultItem, type EmbeddingsOptions, type EmbeddingsResult, type ExecutionOptions, getModelCapabilities, type ImageEmbeddingInput, LlumiverseError, modelModalitiesToArray, normalizeEmbeddingsOptions, Providers, type TextEmbeddingInput } from "@llumiverse/core";
import type OpenAI from "openai";
import { AzureOpenAIDriver } from "../openai/azure_openai.js";
import { formatOpenAILikeMultimodalPrompt } from "../openai/openai_format.js";

type ResponseInputItem = OpenAI.Responses.ResponseInputItem;
type EasyInputMessage = OpenAI.Responses.EasyInputMessage;
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
    messages: ResponseInputItem[]
}

export type AzureFoundryPrompt = AzureFoundryInferencePrompt | AzureFoundryOpenAIPrompt

export class AzureFoundryDriver extends AbstractDriver<AzureFoundryDriverOptions, ResponseInputItem[]> {
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
            this.logger.error({ error }, "Failed to initialize Azure AD token provider:");
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
            this.logger.error({ deploymentError }, `[Azure Foundry] Deployment ${deploymentName} not found:`);
        }

        return (deployment as ModelDeployment).modelPublisher === "OpenAI";
    }

    protected canStream(_options: ExecutionOptions): Promise<boolean> {
        return Promise.resolve(true);
    }

    async requestTextCompletion(prompt: ResponseInputItem[], options: ExecutionOptions): Promise<Completion> {
        const { deploymentName } = parseAzureFoundryModelId(options.model);
        const model_options = options.model_options as any;
        const isOpenAI = await this.isOpenAIDeployment(options.model);

        if (isOpenAI) {
            // Use the Azure OpenAI client for OpenAI models
            const azureOpenAI = await this.service.inference.azureOpenAI({ apiVersion: this.OPENAI_API_VERSION });
            const subDriver = new AzureOpenAIDriver(azureOpenAI);
            // Use deployment name for API calls
            const modifiedOptions = { ...options, model: deploymentName };
            return subDriver.requestTextCompletion(prompt, modifiedOptions);

        } else {
            // Use the chat completions client from the inference operations
            // Convert ResponseInputItem[] to ChatRequestMessage[] for non-OpenAI inference
            const messages = convertToInferenceMessages(prompt);
            const chatClient = this.service.inference.chatCompletions({ apiVersion: this.INFERENCE_API_VERSION });
            const response = await chatClient.post({
                body: {
                    messages,
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
                this.logger.error({ response }, `[Azure Foundry] Chat completion request failed:`);
                throw new Error(`Chat completion request failed with status ${response.status}: ${response.body}`);
            }

            return this.extractDataFromResponse(response.body as ChatCompletionsOutput);
        }
    }

    async requestTextCompletionStream(prompt: ResponseInputItem[], options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const { deploymentName } = parseAzureFoundryModelId(options.model);
        const model_options = options.model_options as any;
        const isOpenAI = await this.isOpenAIDeployment(options.model);

        if (isOpenAI) {
            const azureOpenAI = await this.service.inference.azureOpenAI({ apiVersion: this.OPENAI_API_VERSION });
            const subDriver = new AzureOpenAIDriver(azureOpenAI);
            const modifiedOptions = { ...options, model: deploymentName };
            const stream = await subDriver.requestTextCompletionStream(prompt, modifiedOptions);
            return stream;
        } else {
            // Convert ResponseInputItem[] to ChatRequestMessage[] for non-OpenAI inference
            const messages = convertToInferenceMessages(prompt);
            const chatClient = this.service.inference.chatCompletions({ apiVersion: this.INFERENCE_API_VERSION });
            const response = await chatClient.post({
                body: {
                    messages,
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

    private async *processStreamResponse(sseStream: any): AsyncIterable<CompletionChunkObject> {
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
                    const chunk: CompletionChunkObject = {
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
                    this.logger.warn({ parseError }, `[Azure Foundry] Failed to parse streaming response:`);
                    continue;
                }
            }
        } catch (error) {
            this.logger.error({ error }, `[Azure Foundry] Streaming error:`);
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
            this.logger.error({ result }, "[Azure Foundry] No choices in response");
            throw new Error("No choices in response");
        }

        const data = choice.message?.content;
        const toolCalls = choice.message?.tool_calls;

        if (!data && !toolCalls) {
            this.logger.error({ result }, "[Azure Foundry] Response is not valid");
            throw new Error("Response is not valid: no content or tool calls");
        }

        const completion: Completion = {
            result: data ? [{ type: "text", value: data }] : [],
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
            this.logger.error({ error }, "Azure Foundry connection validation failed:");
            return false;
        }
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        const normalized = normalizeEmbeddingsOptions(options);
        if (!normalized.model) {
            throw new Error("Default embedding model selection not supported for Azure Foundry. Please specify a model.");
        }

        const textInputs: { index: number; input: TextEmbeddingInput }[] = [];
        const imageInputs: { index: number; input: ImageEmbeddingInput }[] = [];
        normalized.inputs.forEach((input, index) => {
            if (input.type === "text") textInputs.push({ index, input });
            else if (input.type === "image") imageInputs.push({ index, input });
            else {
                throw new Error(`Provider 'azure_foundry' does not support '${input.type}' embeddings.`);
            }
        });

        const items = new Array<EmbeddingResultItem>(normalized.inputs.length);

        if (textInputs.length > 0) {
            const vectors = await this.callAzureEmbeddings(
                textInputs.map((t) => t.input.text),
                normalized.model,
                "text",
            );
            textInputs.forEach((entry, i) => {
                items[entry.index] = { outputs: [{ values: vectors[i], modality: "text" }] };
            });
        }

        if (imageInputs.length > 0) {
            const base64Images = await Promise.all(imageInputs.map((entry) => dataSourceToBase64(entry.input.source)));
            const vectors = await this.callAzureEmbeddings(base64Images, normalized.model, "image");
            imageInputs.forEach((entry, i) => {
                items[entry.index] = { outputs: [{ values: vectors[i], modality: "image" }] };
            });
        }

        return { model: normalized.model, results: items };
    }

    private async callAzureEmbeddings(input: string[], model: string, kind: "text" | "image"): Promise<number[][]> {
        const { deploymentName } = parseAzureFoundryModelId(model);
        try {
            const embeddingsClient = this.service.inference.embeddings({ apiVersion: this.INFERENCE_API_VERSION });
            const response = await embeddingsClient.post({ body: { input, model: deploymentName } });
            if (isUnexpected(response)) {
                throw new Error(`${kind} embeddings request failed: ${response.status} ${response.body?.error?.message || 'Unknown error'}`);
            }

            const data = response.body.data;
            if (!Array.isArray(data) || data.length === 0) {
                throw new Error(`No embeddings found in Azure Foundry ${kind} response`);
            }
            const ordered = [...data].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
            return ordered.map((entry) => {
                const embedding = entry.embedding;
                if (!Array.isArray(embedding) || embedding.length === 0) {
                    throw new Error(`Empty or non-array embedding in Azure Foundry ${kind} response (got ${typeof embedding})`);
                }
                return embedding;
            });
        } catch (error) {
            if (LlumiverseError.isLlumiverseError(error)) throw error;
            if (error instanceof Error && typeof (error as any).status !== 'number') throw error;
            this.logger.error({ error }, `Azure Foundry ${kind} embeddings error:`);
            throw this.formatLlumiverseError(error, {
                provider: this.provider,
                model,
                operation: 'execute',
            });
        }
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
            this.logger.error({ error }, "Failed to list deployments:");
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

/**
 * Convert ResponseInputItem[] to ChatRequestMessage[] for Azure AI Inference API
 */
function convertToInferenceMessages(items: ResponseInputItem[]): ChatRequestMessage[] {
    const messages: ChatRequestMessage[] = [];

    for (const item of items) {
        // Handle EasyInputMessage (has role and content)
        if ('role' in item && 'content' in item) {
            const msg = item as EasyInputMessage;
            let content: string;
            if (typeof msg.content === 'string') {
                content = msg.content;
            } else if (Array.isArray(msg.content)) {
                // Extract text from content array
                content = msg.content
                    .filter((part): part is OpenAI.Responses.ResponseInputText => part.type === 'input_text')
                    .map(part => part.text)
                    .join('\n');
            } else {
                content = '';
            }

            messages.push({
                role: msg.role as 'system' | 'user' | 'assistant',
                content
            });
        }
        // Handle function_call_output
        else if ('type' in item && item.type === 'function_call_output') {
            const output = item as OpenAI.Responses.ResponseInputItem.FunctionCallOutput;
            messages.push({
                role: 'tool',
                content: typeof output.output === 'string' ? output.output : JSON.stringify(output.output),
                tool_call_id: output.call_id,
            } as ChatRequestMessage);
        }
    }

    return messages;
}
