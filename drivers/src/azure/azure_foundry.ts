import { DefaultAzureCredential, getBearerTokenProvider } from "@azure/identity";
import { AbstractDriver, AIModel, Completion, CompletionChunk, DriverOptions, EmbeddingsOptions, EmbeddingsResult, ExecutionOptions, getModelCapabilities, modelModalitiesToArray, PromptSegment, Providers, PromptRole } from "@llumiverse/core";
import { AIProjectClient, DeploymentUnion, ModelDeployment } from '@azure/ai-projects';
import { isUnexpected } from "@azure-rest/ai-inference";
import { ChatCompletionMessageParam } from "openai/resources";
import type {
    ChatCompletionsOutput,
    ChatCompletionsToolCall,
    ChatRequestMessage,
} from "@azure-rest/ai-inference";
import { AzureOpenAIDriver } from "../openai/azure_openai.js";
import { createSseStream } from "@azure/core-sse";
export interface AzureFoundryDriverOptions extends DriverOptions {
    /**
     * The credentials to use to access Azure AI Foundry
     */
    azureADTokenProvider?: any;

    apiKey?: string;

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

export class AzureFoundryDriver extends AbstractDriver<AzureFoundryDriverOptions, AzureFoundryPrompt> {
    service: AIProjectClient;
    readonly provider = Providers.azure_foundry;

    constructor(opts: AzureFoundryDriverOptions) {
        super(opts);

        if (!opts.endpoint) {
            throw new Error("Azure AI Foundry endpoint is required");
        }

        try {
            if (!opts.azureADTokenProvider) {
                if (opts.apiKey) {
                    // Using API key for authentication
                    //opts.azureADTokenProvider = new AzureKeyCredential(opts.apiKey);
                    opts.azureADTokenProvider = new DefaultAzureCredential();
                } else {
                    // Using Microsoft Entra ID (Azure AD) for authentication
                    opts.azureADTokenProvider = new DefaultAzureCredential();
                }
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
        let deployment = undefined;
        // First, verify the deployment exists
        try {
            deployment = await this.service.deployments.get(model);
            this.logger.debug(`[Azure Foundry] Deployment ${model} found`);
        } catch (deploymentError) {
            this.logger.error(`[Azure Foundry] Deployment ${model} not found:`, deploymentError);
        }

        return (deployment as ModelDeployment).modelPublisher == "OpenAI";
    }

    protected canStream(_options: ExecutionOptions): Promise<boolean> {
        return Promise.resolve(false);
    }

    protected async formatPrompt(segments: PromptSegment[], _opts: ExecutionOptions): Promise<AzureFoundryPrompt> {
        const messages: ChatRequestMessage[] = [];

        for (const segment of segments) {
            switch (segment.role) {
                case PromptRole.system:
                    messages.push({
                        role: 'system',
                        content: segment.content
                    });
                    break;
                case PromptRole.user:
                    // Azure AI Foundry supports multimodal inputs, but we'll handle simple text for now
                    // In the future, can be extended to handle segment.files for images
                    messages.push({
                        role: 'user',
                        content: segment.content
                    });
                    break;
                case PromptRole.assistant:
                    messages.push({
                        role: 'assistant',
                        content: segment.content
                    });
                    break;
                case PromptRole.tool:
                    // Azure AI supports tool responses, can be implemented later
                    messages.push({
                        role: 'assistant',
                        content: segment.content
                    });
                    break;
            }
        }

        return { messages };
    }

    async requestTextCompletion(prompt: AzureFoundryPrompt, options: ExecutionOptions): Promise<Completion> {
        const model_options = options.model_options as any;
        const isOpenAI = await this.isOpenAIDeployment(options.model);

        
        let response;
        if (isOpenAI) {
            // Use the Azure OpenAI client for OpenAI models
            const azureOpenAI = await this.service.inference.azureOpenAI({ apiVersion: "preview" });
            const subDriver = new AzureOpenAIDriver(azureOpenAI);
            const response = await subDriver.requestTextCompletion((prompt as AzureFoundryOpenAIPrompt).messages, options);
            return response;

        } else {
            // Use the chat completions client from the inference operations
            const chatClient = this.service.inference.chatCompletions({ apiVersion: "2024-05-01-preview"});
            response = await chatClient.post({
                body: {
                    messages: prompt.messages,
                    max_tokens: model_options?.max_tokens,
                    model: options.model,
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

            console.log(`[Azure Foundry] Chat completion response:`, JSON.stringify(response, null, 2));
            return this.extractDataFromResponse(response.body as ChatCompletionsOutput);
        }
    }

    async requestTextCompletionStream(prompt: AzureFoundryPrompt, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunk>> {
        const model_options = options.model_options as any;
        const isOpenAI = await this.isOpenAIDeployment(options.model);

        if (isOpenAI) {
            const azureOpenAI = await this.service.inference.azureOpenAI({ apiVersion: "preview" });
            const subDriver = new AzureOpenAIDriver(azureOpenAI);
            const stream = await subDriver.requestTextCompletionStream((prompt as AzureFoundryOpenAIPrompt).messages, options);
            return stream;
        } else {
            const chatClient = this.service.inference.chatCompletions({ apiVersion: "2024-05-01-preview" });
            const response = await chatClient.post({
                body: {
                    messages: prompt.messages,
                    max_tokens: model_options?.max_tokens,
                    model: options.model,
                    stream: true,
                    temperature: model_options?.temperature,
                    top_p: model_options?.top_p,
                    frequency_penalty: model_options?.frequency_penalty,
                    presence_penalty: model_options?.presence_penalty,
                    stop: model_options?.stop_sequence,
                }
            }).asNodeStream();

            if (!response) {
                throw new Error("The response stream is undefined");
            }

            if (response.status !== "200") {
                throw new Error(`Failed to get chat completions, http operation failed with ${response.status} code`);
            }

            const stream = response.body as NodeJS.ReadableStream;

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
                    const choice = data.choices?.[0];

                    if (!choice) {
                        continue;
                    }

                    const chunk: CompletionChunk = {
                        result: choice.delta?.content || "",
                        finish_reason: this.convertFinishReason(choice.finish_reason),
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
            case 'content_filter': return 'content_filter';
            case 'tool_calls': return 'tool_calls';
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
        if (!options.text) {
            throw new Error("No text provided for embeddings");
        }

        try {
            // Use the embeddings client from the inference operations
            const embeddingsClient = this.service.inference.embeddings();
            const response = await embeddingsClient.post({
                body: {
                    input: Array.isArray(options.text) ? options.text : [options.text],
                    model: options.model || "text-embedding-ada-002"
                }
            });

            if (isUnexpected(response)) {
                throw new Error(`Embeddings request failed: ${response.status} ${response.body?.error?.message || 'Unknown error'}`);
            }

            const embeddings = response.body.data?.[0]?.embedding;
            if (!embeddings || !Array.isArray(embeddings) || embeddings.length === 0) {
                throw new Error("No valid embedding array found in response");
            }

            return {
                values: embeddings as number[],
                model: options.model || "text-embedding-ada-002"
            };
        } catch (error) {
            this.logger.error("Azure Foundry embeddings error:", error);
            throw error;
        }
    }

    async listModels(): Promise<AIModel[]> {
        return this._listModels();
    }

    async _listModels(filter?: (m: ModelDeployment) => boolean): Promise<AIModel[]> {
        const deploymentsIterable = this.service.deployments.list();
        const deployments: DeploymentUnion[] = [];

        for await (const page of deploymentsIterable.byPage()) {
            for (const deployment of page) {
                deployments.push(deployment);
            }
        }

        const modelDeployments: ModelDeployment[] = deployments.filter((d): d is ModelDeployment => {
            return d.type === "ModelDeployment";
        });

        //Azure OpenAI has additional information about the models
        const filteredModels = modelDeployments.filter((m) => {
            return !m.capabilities.embeddings;
        }).filter((m) => {
            // Apply passed in filter function if provided
            return filter && filter(m);
        });

        const aiModels = filteredModels.map((model) => {
            const capabilitiesProvider = model.modelPublisher === "OpenAI" ? "openai" : "azure_foundry";
            const modelCapability = getModelCapabilities(model.modelName, capabilitiesProvider);
            return {
                id: model.name,
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
