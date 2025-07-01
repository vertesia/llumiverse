import {
    Bedrock, CreateModelCustomizationJobCommand, FoundationModelSummary, GetModelCustomizationJobCommand,
    GetModelCustomizationJobCommandOutput, ModelCustomizationJobStatus, ModelModality, StopModelCustomizationJobCommand
} from "@aws-sdk/client-bedrock";
import { BedrockRuntime, ConverseRequest, ConverseResponse, ConverseStreamOutput, InferenceConfiguration, Tool } from "@aws-sdk/client-bedrock-runtime";
import { S3Client } from "@aws-sdk/client-s3";
import { AwsCredentialIdentity, Provider } from "@aws-sdk/types";
import {
    AbstractDriver, AIModel, Completion, CompletionChunkObject, DataSource, DriverOptions, EmbeddingsOptions, EmbeddingsResult,
    ExecutionOptions, ExecutionTokenUsage, ImageGeneration, Modalities, PromptSegment,
    TextFallbackOptions, ToolDefinition, ToolUse, TrainingJob, TrainingJobStatus, TrainingOptions,
    BedrockClaudeOptions, BedrockPalmyraOptions, getMaxTokensLimitBedrock, NovaCanvasOptions,
    modelModalitiesToArray, getModelCapabilities,
    StatelessExecutionOptions,
    ModelOptions
} from "@llumiverse/core";
import { transformAsyncIterator } from "@llumiverse/core/async";
import { formatNovaPrompt, NovaMessagesPrompt } from "@llumiverse/core/formatters";
import { LRUCache } from "mnemonist";
import { converseConcatMessages, converseJSONprefill, converseSystemToMessages, formatConversePrompt } from "./converse.js";
import { formatNovaImageGenerationPayload, NovaImageGenerationTaskType } from "./nova-image-payload.js";
import { forceUploadFile } from "./s3.js";

const supportStreamingCache = new LRUCache<string, boolean>(4096);

enum BedrockModelType {
    FoundationModel = "foundation-model",
    InferenceProfile = "inference-profile",
    CustomModel = "custom-model",
    Unknown = "unknown",
};

function converseFinishReason(reason: string | undefined) {
    //Possible values:
    //end_turn | tool_use | max_tokens | stop_sequence | guardrail_intervened | content_filtered
    if (!reason) return undefined;
    switch (reason) {
        case 'end_turn': return "stop";
        case 'max_tokens': return "length";
        default: return reason;
    }
}

export interface BedrockModelCapabilities {
    name: string;
    canStream: boolean;
}

export interface BedrockDriverOptions extends DriverOptions {
    /**
     * The AWS region
     */
    region: string;
    /**
     * The bucket name to be used for training.
     * It will be created if does not already exist.
     */
    training_bucket?: string;

    /**
     * The role ARN to be used for training
     */
    training_role_arn?: string;

    /**
     * The credentials to use to access AWS
     */
    credentials?: AwsCredentialIdentity | Provider<AwsCredentialIdentity>;
}

//Used to get a max_token value when not specified in the model options. Claude requires it to be set.
function maxTokenFallbackClaude(option: StatelessExecutionOptions): number {
    const modelOptions = option.model_options as BedrockClaudeOptions | undefined;
    if (modelOptions && typeof modelOptions.max_tokens === "number") {
        return modelOptions.max_tokens;
    } else {
        // Fallback to the default max tokens limit for the model
        if (option.model.includes('claude-3-7-sonnet') && (modelOptions?.thinking_budget_tokens ?? 0) < 64000) {
            return 64000; // Claude 3.7 can go up to 128k with a beta header, but when no max tokens is specified, we default to 64k.
        }
        return getMaxTokensLimitBedrock(option.model) ?? 8192; // Should always return a number for claude, 8192 is to satisfy the TypeScript type checker
    }
}

export type BedrockPrompt = NovaMessagesPrompt | ConverseRequest;

export class BedrockDriver extends AbstractDriver<BedrockDriverOptions, BedrockPrompt> {

    static PROVIDER = "bedrock";

    provider = BedrockDriver.PROVIDER;

    private _executor?: BedrockRuntime;
    private _service?: Bedrock;
    private _service_region?: string;

    constructor(options: BedrockDriverOptions) {
        super(options);
        if (!options.region) {
            throw new Error("No region found. Set the region in the environment's endpoint URL.");
        }
    }

    getExecutor() {
        if (!this._executor) {
            this._executor = new BedrockRuntime({
                region: this.options.region,
                credentials: this.options.credentials,

            });
        }
        return this._executor;
    }

    getService(region: string = this.options.region) {
        if (!this._service || this._service_region != region) {
            this._service = new Bedrock({
                region: region,
                credentials: this.options.credentials,
            });
            this._service_region = region;
        }
        return this._service;
    }

    protected async formatPrompt(segments: PromptSegment[], opts: ExecutionOptions): Promise<BedrockPrompt> {
        if (opts.model.includes("canvas")) {
            return await formatNovaPrompt(segments, opts.result_schema);
        }
        return await formatConversePrompt(segments, opts);
    }

    static getExtractedExecution(result: ConverseResponse, _prompt?: BedrockPrompt, options?: ExecutionOptions): CompletionChunkObject {
        let resultText = "";
        let reasoning = "";

        if (result.output?.message?.content) {
            for (const content of result.output.message.content) {
                // Get text output
                if (content.text) {
                    resultText += content.text;
                }
                // Get reasoning content only if include_thoughts is true
                if (content.reasoningContent && options) {
                    const claudeOptions = options.model_options as BedrockClaudeOptions;
                    if (claudeOptions?.include_thoughts) {
                        if (content.reasoningContent.reasoningText) {
                            reasoning += content.reasoningContent.reasoningText.text;
                        } else if (content.reasoningContent.redactedContent) {
                            // Handle redacted thinking content
                            const redactedData = new TextDecoder().decode(content.reasoningContent.redactedContent);
                            reasoning += `[Redacted thinking: ${redactedData}]`;
                        }
                    }
                }
            }

            // Add spacing if we have reasoning content
            if (reasoning) {
                reasoning += '\n\n';
            }
        }

        const completionResult: CompletionChunkObject = {
            result: reasoning + resultText,
            token_usage: {
                prompt: result.usage?.inputTokens,
                result: result.usage?.outputTokens,
                total: result.usage?.totalTokens,
            },
            finish_reason: converseFinishReason(result.stopReason),
        };

        return completionResult;
    };

    static getExtractedStream(result: ConverseStreamOutput, _prompt?: BedrockPrompt, options?: ExecutionOptions): CompletionChunkObject {
        let output: string = "";
        let reasoning: string = "";
        let stop_reason = "";
        let token_usage: ExecutionTokenUsage | undefined;

        // Check if we should include thoughts
        const shouldIncludeThoughts = options && (options.model_options as BedrockClaudeOptions)?.include_thoughts;

        // Handle content block start events (for reasoning blocks)
        if (result.contentBlockStart) {
            // Handle redacted content at block start
            if (result.contentBlockStart.start && 'reasoningContent' in result.contentBlockStart.start && shouldIncludeThoughts) {
                const reasoningStart = result.contentBlockStart.start as any;
                if (reasoningStart.reasoningContent?.redactedContent) {
                    const redactedData = new TextDecoder().decode(reasoningStart.reasoningContent.redactedContent);
                    reasoning = `[Redacted thinking: ${redactedData}]`;
                }
            }
        }

        // Handle content block deltas (text and reasoning)
        if (result.contentBlockDelta) {
            const delta = result.contentBlockDelta.delta;
            if (delta?.text) {
                output = delta.text;
            } else if (delta?.reasoningContent && shouldIncludeThoughts) {
                if (delta.reasoningContent.text) {
                    reasoning = delta.reasoningContent.text;
                } else if (delta.reasoningContent.redactedContent) {
                    const redactedData = new TextDecoder().decode(delta.reasoningContent.redactedContent);
                    reasoning = `[Redacted thinking: ${redactedData}]`;
                } else if (delta.reasoningContent.signature) {
                    // Handle signature updates for reasoning content - end of thinking
                    reasoning = "\n\n";
                }
            }
        }

        // Handle content block stop events
        if (result.contentBlockStop) {
            // Content block ended - could be end of reasoning or text block
            // Add minimal spacing for reasoning blocks if not already present
            if (reasoning && !reasoning.endsWith('\n\n') && shouldIncludeThoughts) {
                reasoning += '\n\n';
            }
        }

        if (result.messageStop) {
            stop_reason = result.messageStop.stopReason ?? "";
        }

        if (result.metadata) {
            token_usage = {
                prompt: result.metadata.usage?.inputTokens,
                result: result.metadata.usage?.outputTokens,
                total: result.metadata.usage?.totalTokens,
            }
        }

        const completionResult: CompletionChunkObject = {
            result: reasoning + output,
            token_usage: token_usage,
            finish_reason: converseFinishReason(stop_reason),
        };

        return completionResult;
    };

    extractRegion(modelString: string, defaultRegion: string): string {
        // Match region in full ARN pattern
        const arnMatch = modelString.match(/arn:aws[^:]*:bedrock:([^:]+):/);
        if (arnMatch) {
            return arnMatch[1];
        }

        // Match common AWS regions directly in string
        const regionMatch = modelString.match(/(?:us|eu|ap|sa|ca|me|af)[-](east|west|central|south|north|southeast|southwest|northeast|northwest)[-][1-9]/);
        if (regionMatch) {
            return regionMatch[0];
        }

        return defaultRegion;
    }

    private async getCanStream(model: string, type: BedrockModelType): Promise<boolean> {
        let canStream: boolean = false;
        let error: any = null;
        const region = this.extractRegion(model, this.options.region);
        if (type == BedrockModelType.FoundationModel || type == BedrockModelType.Unknown) {
            try {
                const response = await this.getService(region).getFoundationModel({
                    modelIdentifier: model
                });
                canStream = response.modelDetails?.responseStreamingSupported ?? false;
                return canStream;
            } catch (e) {
                error = e;
            }
        }
        if (type == BedrockModelType.InferenceProfile || type == BedrockModelType.Unknown) {
            try {
                const response = await this.getService(region).getInferenceProfile({
                    inferenceProfileIdentifier: model
                });
                canStream = await this.getCanStream(response.models?.[0].modelArn ?? "", BedrockModelType.FoundationModel);
                return canStream;
            } catch (e) {
                error = e;
            }
        }
        if (type == BedrockModelType.CustomModel || type == BedrockModelType.Unknown) {
            try {
                const response = await this.getService(region).getCustomModel({
                    modelIdentifier: model
                });
                canStream = await this.getCanStream(response.baseModelArn ?? "", BedrockModelType.FoundationModel);
                return canStream;
            } catch (e) {
                error = e;
            }
        }
        if (error) {
            console.warn("Error on canStream check for model: " + model + " region detected: " + region, error);
        }
        return canStream;
    }

    protected async canStream(options: ExecutionOptions): Promise<boolean> {
        let canStream = supportStreamingCache.get(options.model);
        if (canStream == null) {
            let type = BedrockModelType.Unknown;
            if (options.model.includes("foundation-model")) {
                type = BedrockModelType.FoundationModel;
            } else if (options.model.includes("inference-profile")) {
                type = BedrockModelType.InferenceProfile;
            } else if (options.model.includes("custom-model")) {
                type = BedrockModelType.CustomModel;
            }
            canStream = await this.getCanStream(options.model, type);
            supportStreamingCache.set(options.model, canStream);
        }
        return canStream;
    }

    async requestTextCompletion(prompt: ConverseRequest, options: ExecutionOptions): Promise<Completion> {
        let conversation = updateConversation(options.conversation as ConverseRequest, prompt);

        const payload = this.preparePayload(conversation, options);
        const executor = this.getExecutor();

        const res = await executor.converse({
            ...payload,
        });

        conversation = updateConversation(conversation, {
            messages: [res.output?.message ?? { content: [{ text: "" }], role: "assistant" }],
            modelId: prompt.modelId,
        });

        let tool_use: ToolUse[] | undefined = undefined;
        //Get tool requests
        if (res.stopReason == "tool_use") {
            tool_use = res.output?.message?.content?.reduce((tools: ToolUse[], c) => {
                if (c.toolUse) {
                    tools.push({
                        tool_name: c.toolUse.name ?? "",
                        tool_input: c.toolUse.input as any,
                        id: c.toolUse.toolUseId ?? "",
                    } satisfies ToolUse);
                }
                return tools;
            }, []);
            //If no tools were used, set to undefined
            if (tool_use && tool_use.length == 0) {
                tool_use = undefined;
            }
        }

        const completion = {
            ...BedrockDriver.getExtractedExecution(res, prompt, options),
            original_response: options.include_original_response ? res : undefined,
            conversation: conversation,
            tool_use: tool_use,
        };

        return completion;
    }

    async requestTextCompletionStream(prompt: ConverseRequest, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const payload = this.preparePayload(prompt, options);
        const executor = this.getExecutor();
        return executor.converseStream({
            ...payload,
        }).then((res) => {
            const stream = res.stream;

            if (!stream) {
                throw new Error("[Bedrock] Stream not found in response");
            }

            return transformAsyncIterator(stream, (streamSegment: ConverseStreamOutput) => {
                return BedrockDriver.getExtractedStream(streamSegment, prompt, options);
            });

        }).catch((err) => {
            this.logger.error("[Bedrock] Failed to stream", { error: err });
            throw err;
        });
    }

    preparePayload(prompt: ConverseRequest, options: ExecutionOptions) {
        const model_options: TextFallbackOptions = options.model_options as TextFallbackOptions ?? { _option_id: "text-fallback" };

        let additionalField = {};
        let supportsJSONPrefill = false;

        if (options.model.includes("amazon")) {
            supportsJSONPrefill = true;
            //Titan models also exists but does not support any additional options
            if (options.model.includes("nova")) {
                additionalField = { inferenceConfig: { topK: model_options.top_k } };
            }
        } else if (options.model.includes("claude")) {
            const claude_options = model_options as ModelOptions as BedrockClaudeOptions;
            const thinking = claude_options.thinking_mode ?? false;
            supportsJSONPrefill = !thinking

            if (options.model.includes("claude-3-7") || options.model.includes("-4-")) {
                additionalField = {
                    ...additionalField,
                    reasoning_config: {
                        type: thinking ? "enabled" : "disabled",
                        budget_tokens: thinking ? (claude_options.thinking_budget_tokens ?? 1024) : undefined,
                    }
                };
                if (thinking && options.model.includes("claude-3-7-sonnet") &&
                    ((claude_options.max_tokens ?? 0) > 64000 || (claude_options.thinking_budget_tokens ?? 0) > 64000)) {
                    additionalField = {
                        ...additionalField,
                        anthropic_beta: ["output-128k-2025-02-19"]
                    };
                }
            }
            //Needs max_tokens to be set
            if (!model_options.max_tokens) {
                model_options.max_tokens = maxTokenFallbackClaude(options);
            }
            additionalField = { ...additionalField, top_k: model_options.top_k };
        } else if (options.model.includes("meta")) {
            //LLaMA models support no additional options
        } else if (options.model.includes("mistral")) {
            //7B instruct and 8x7B instruct
            if (options.model.includes("7b")) {
                supportsJSONPrefill = true;
                additionalField = { top_k: model_options.top_k };
                //Does not support system messages
                if (prompt.system && prompt.system?.length != 0) {
                    prompt.messages?.push(converseSystemToMessages(prompt.system));
                    prompt.system = undefined;
                    prompt.messages = converseConcatMessages(prompt.messages);
                }
            } else {
                //Other models such as Mistral Small,Large and Large 2
                //Support no additional fields.
            }
        } else if (options.model.includes("ai21")) {
            //Jamba models support no additional options
            //Jurassic 2 models do.
            if (options.model.includes("j2")) {
                additionalField = {
                    presencePenalty: { scale: model_options.presence_penalty },
                    frequencyPenalty: { scale: model_options.frequency_penalty },
                };
                //Does not support system messages
                if (prompt.system && prompt.system?.length != 0) {
                    prompt.messages?.push(converseSystemToMessages(prompt.system));
                    prompt.system = undefined;
                    prompt.messages = converseConcatMessages(prompt.messages);
                }
            }
        } else if (options.model.includes("cohere.command")) {
            // If last message is "```json", remove it.
            //Command R and R plus
            if (options.model.includes("cohere.command-r")) {
                additionalField = {
                    k: model_options.top_k,
                    frequency_penalty: model_options.frequency_penalty,
                    presence_penalty: model_options.presence_penalty,
                };
            } else {
                // Command non-R
                additionalField = { k: model_options.top_k };
                //Does not support system messages
                if (prompt.system && prompt.system?.length != 0) {
                    prompt.messages?.push(converseSystemToMessages(prompt.system));
                    prompt.system = undefined;
                    prompt.messages = converseConcatMessages(prompt.messages);
                }
            }
        } else if (options.model.includes("palmyra")) {
            const palmyraOptions = model_options as ModelOptions as BedrockPalmyraOptions;
            additionalField = {
                seed: palmyraOptions?.seed,
                presence_penalty: palmyraOptions?.presence_penalty,
                frequency_penalty: palmyraOptions?.frequency_penalty,
                min_tokens: palmyraOptions?.min_tokens,
            }
        } else if (options.model.includes("deepseek")) {
            //DeepSeek models support no additional options
        }

        //If last message is "```json", add corresponding ``` as a stop sequence.
        if (prompt.messages && prompt.messages.length > 0) {
            if (prompt.messages[prompt.messages.length - 1].content?.[0].text === "```json") {
                let stopSeq = model_options.stop_sequence;
                if (!stopSeq) {
                    model_options.stop_sequence = ["```"];
                } else if (!stopSeq.includes("```")) {
                    stopSeq.push("```");
                    model_options.stop_sequence = stopSeq;
                }
            }
        }

        const tool_defs = getToolDefinitions(options.tools);

        // Use prefill when there is a schema and tools are not being used
        if (supportsJSONPrefill && options.result_schema && !tool_defs) {
            prompt.messages = converseJSONprefill(prompt.messages);
        }

        const request: ConverseRequest = {
            messages: prompt.messages,
            system: prompt.system,
            modelId: options.model,
            inferenceConfig: {
                maxTokens: model_options.max_tokens,
                temperature: model_options.temperature,
                topP: model_options.top_p,
                stopSequences: model_options.stop_sequence,
            } satisfies InferenceConfiguration,
            additionalModelRequestFields: {
                ...additionalField,
            }
        };

        //Only add tools if they are defined
        if (tool_defs) {
            request.toolConfig = {
                tools: tool_defs,
            }
        }

        return request;
    }


    async requestImageGeneration(prompt: NovaMessagesPrompt, options: ExecutionOptions): Promise<Completion<ImageGeneration>> {
        if (options.output_modality !== Modalities.image) {
            throw new Error(`Image generation requires image output_modality`);
        }
        if (options.model_options?._option_id !== "bedrock-nova-canvas") {
            this.logger.warn("Invalid model options", { options: options.model_options });
        }
        const model_options = options.model_options as NovaCanvasOptions;

        const executor = this.getExecutor();
        const taskType = model_options.taskType ?? NovaImageGenerationTaskType.TEXT_IMAGE;

        this.logger.info("Task type: " + taskType);

        if (typeof prompt === "string") {
            throw new Error("Bad prompt format");
        }

        const payload = await formatNovaImageGenerationPayload(taskType, prompt, options);

        const res = await executor.invokeModel({
            modelId: options.model,
            contentType: "application/json",
            accept: "application/json",
            body: JSON.stringify(payload),
        },
            {
                requestTimeout: 60000 * 5
            });

        const decoder = new TextDecoder();
        const body = decoder.decode(res.body);
        const result = JSON.parse(body);

        return {
            error: result.error,
            result: {
                images: result.images,
            }
        }
    }

    async startTraining(dataset: DataSource, options: TrainingOptions): Promise<TrainingJob> {

        //convert options.params to Record<string, string>
        const params: Record<string, string> = {};
        for (const [key, value] of Object.entries(options.params || {})) {
            params[key] = String(value);
        }

        if (!this.options.training_bucket) {
            throw new Error("Training cannot nbe used since the 'training_bucket' property was not specified in driver options")
        }

        const s3 = new S3Client({ region: this.options.region, credentials: this.options.credentials });
        const stream = await dataset.getStream();
        const upload = await forceUploadFile(s3, stream, this.options.training_bucket, dataset.name);

        const service = this.getService();
        const response = await service.send(new CreateModelCustomizationJobCommand({
            jobName: options.name + "-job",
            customModelName: options.name,
            roleArn: this.options.training_role_arn || undefined,
            baseModelIdentifier: options.model,
            clientRequestToken: "llumiverse-" + Date.now(),
            trainingDataConfig: {
                s3Uri: `s3://${upload.Bucket}/${upload.Key}`,
            },
            outputDataConfig: undefined,
            hyperParameters: params,
            //TODO not supported?
            //customizationType: "FINE_TUNING",
        }));

        const job = await service.send(new GetModelCustomizationJobCommand({
            jobIdentifier: response.jobArn
        }));

        return jobInfo(job, response.jobArn!);
    }

    async cancelTraining(jobId: string): Promise<TrainingJob> {
        const service = this.getService();
        await service.send(new StopModelCustomizationJobCommand({
            jobIdentifier: jobId
        }));
        const job = await service.send(new GetModelCustomizationJobCommand({
            jobIdentifier: jobId
        }));

        return jobInfo(job, jobId);
    }

    async getTrainingJob(jobId: string): Promise<TrainingJob> {
        const service = this.getService();
        const job = await service.send(new GetModelCustomizationJobCommand({
            jobIdentifier: jobId
        }));

        return jobInfo(job, jobId);
    }

    // ===================== management API ==================

    async validateConnection(): Promise<boolean> {
        const service = this.getService();
        this.logger.debug("[Bedrock] validating connection", service.config.credentials.name);
        //return true as if the client has been initialized, it means the connection is valid
        return true;
    }


    async listTrainableModels(): Promise<AIModel<string>[]> {
        this.logger.debug("[Bedrock] listing trainable models");
        return this._listModels(m => m.customizationsSupported ? m.customizationsSupported.includes("FINE_TUNING") : false);
    }

    async listModels(): Promise<AIModel[]> {
        this.logger.debug("[Bedrock] listing models");
        // exclude trainable models since they are not executable
        // exclude embedding models, not to be used for typical completions.
        const filter = (m: FoundationModelSummary) => (m.inferenceTypesSupported?.includes("ON_DEMAND") && !m.outputModalities?.includes("EMBEDDING")) ?? false;
        return this._listModels(filter);
    }

    async _listModels(foundationFilter?: (m: FoundationModelSummary) => boolean): Promise<AIModel[]> {
        const service = this.getService();
        const [foundationModelsList, customModelsList, inferenceProfilesList] = await Promise.all([
            service.listFoundationModels({}).catch(() => {
                this.logger.warn("[Bedrock] Can't list foundation models. Check if the user has the right permissions.");
                return undefined
            }),
            service.listCustomModels({}).catch(() => {
                this.logger.warn("[Bedrock] Can't list custom models. Check if the user has the right permissions.");
                return undefined
            }),
            service.listInferenceProfiles({}).catch(() => {
                this.logger.warn("[Bedrock] Can't list inference profiles. Check if the user has the right permissions.");
                return undefined
            }),
        ]);

        if (!foundationModelsList?.modelSummaries) {
            throw new Error("Foundation models not found");
        }

        let foundationModels = foundationModelsList.modelSummaries || [];
        if (foundationFilter) {
            foundationModels = foundationModels.filter(foundationFilter);
        }

        const supportedPublishers = ["amazon", "anthropic", "cohere", "ai21", "mistral", "meta", "deepseek", "writer"];
        const unsupportedModelsByPublisher = {
            amazon: ["titan-image-generator", "nova-reel", "nova-sonic", "rerank"],
            anthropic: [],
            cohere: ["rerank"],
            ai21: [],
            mistral: [],
            meta: [],
            deepseek: [],
            writer: [],
        };

        // Helper function to check if model should be filtered out
        const shouldIncludeModel = (modelId?: string, providerName?: string): boolean => {
            if (!modelId || !providerName) return false;

            const normalizedProvider = providerName.toLowerCase();

            // Check if provider is supported
            const isProviderSupported = supportedPublishers.some(provider =>
                normalizedProvider.includes(provider)
            );

            if (!isProviderSupported) return false;

            // Check if model is in the unsupported list for its provider
            for (const provider of supportedPublishers) {
                if (normalizedProvider.includes(provider)) {
                    const unsupportedModels = unsupportedModelsByPublisher[provider as keyof typeof unsupportedModelsByPublisher] || [];
                    return !unsupportedModels.some(unsupported =>
                        modelId.toLowerCase().includes(unsupported)
                    );
                }
            }

            return true;
        };

        foundationModels = foundationModels.filter(m =>
            shouldIncludeModel(m.modelId, m.providerName)
        );

        const aiModels: AIModel[] = foundationModels.map((m) => {

            if (!m.modelId) {
                throw new Error("modelId not found");
            }

            const modelCapability = getModelCapabilities(m.modelArn ?? m.modelId, this.provider);

            const model: AIModel = {
                id: m.modelArn ?? m.modelId,
                name: `${m.providerName} ${m.modelName}`,
                provider: this.provider,
                //description: ``,
                owner: m.providerName,
                can_stream: m.responseStreamingSupported ?? false,
                input_modalities: m.inputModalities ? formatAmazonModalities(m.inputModalities) : modelModalitiesToArray(modelCapability.input),
                output_modalities: m.outputModalities ? formatAmazonModalities(m.outputModalities) : modelModalitiesToArray(modelCapability.input),
                tool_support: modelCapability.tool_support,
            };

            return model;
        });

        //add custom models
        if (customModelsList?.modelSummaries) {
            customModelsList.modelSummaries.forEach((m) => {

                if (!m.modelArn) {
                    throw new Error("Model ID not found");
                }

                const modelCapability = getModelCapabilities(m.modelArn, this.provider);

                const model: AIModel = {
                    id: m.modelArn,
                    name: m.modelName ?? m.modelArn,
                    provider: this.provider,
                    description: `Custom model from ${m.baseModelName}`,
                    is_custom: true,
                    input_modalities: modelModalitiesToArray(modelCapability.input),
                    output_modalities: modelModalitiesToArray(modelCapability.output),
                    tool_support: modelCapability.tool_support,
                };

                aiModels.push(model);
                this.validateConnection;
            });
        }

        //add inference profiles
        if (inferenceProfilesList?.inferenceProfileSummaries) {
            inferenceProfilesList.inferenceProfileSummaries.forEach((p) => {
                if (!p.inferenceProfileArn) {
                    throw new Error("Profile ARN not found");
                }

                // Apply the same filtering logic to inference profiles based on their name
                const profileId = p.inferenceProfileId || "";
                const profileName = p.inferenceProfileName || "";

                // Extract provider name from profile name or ID
                let providerName = "";
                for (const provider of supportedPublishers) {
                    if (profileName.toLowerCase().includes(provider) || profileId.toLowerCase().includes(provider)) {
                        providerName = provider;
                        break;
                    }
                }

                const modelCapability = getModelCapabilities(p.inferenceProfileArn ?? p.inferenceProfileId, this.provider);

                if (providerName && shouldIncludeModel(profileId, providerName)) {
                    const model: AIModel = {
                        id: p.inferenceProfileArn ?? p.inferenceProfileId,
                        name: p.inferenceProfileName ?? p.inferenceProfileArn,
                        provider: this.provider,
                        input_modalities: modelModalitiesToArray(modelCapability.input),
                        output_modalities: modelModalitiesToArray(modelCapability.output),
                        tool_support: modelCapability.tool_support,
                    };

                    aiModels.push(model);
                }
            });
        }

        return aiModels;
    }

    async generateEmbeddings({ text, image, model }: EmbeddingsOptions): Promise<EmbeddingsResult> {

        this.logger.info("[Bedrock] Generating embeddings with model " + model);
        const defaultModel = image ? "amazon.titan-embed-image-v1" : "amazon.titan-embed-text-v2:0";
        const modelID = model ?? defaultModel;

        const invokeBody = {
            inputText: text,
            inputImage: image
        }

        const executor = this.getExecutor();
        const res = await executor.invokeModel(
            {
                modelId: modelID,
                contentType: "application/json",
                body: JSON.stringify(invokeBody),
            }
        );

        const decoder = new TextDecoder();
        const body = decoder.decode(res.body);

        const result = JSON.parse(body);

        if (!result.embedding) {
            throw new Error("Embeddings not found");
        }

        return {
            values: result.embedding,
            model: modelID,
            token_count: result.inputTextTokenCount
        };
    }
}

function jobInfo(job: GetModelCustomizationJobCommandOutput, jobId: string): TrainingJob {
    const jobStatus = job.status;
    let status = TrainingJobStatus.running;
    let details: string | undefined;
    if (jobStatus === ModelCustomizationJobStatus.COMPLETED) {
        status = TrainingJobStatus.succeeded;
    } else if (jobStatus === ModelCustomizationJobStatus.FAILED) {
        status = TrainingJobStatus.failed;
        details = job.failureMessage || "error";
    } else if (jobStatus === ModelCustomizationJobStatus.STOPPED) {
        status = TrainingJobStatus.cancelled;
    } else {
        status = TrainingJobStatus.running;
        details = jobStatus;
    }
    job.baseModelArn
    return {
        id: jobId,
        model: job.outputModelArn,
        status,
        details
    }
}

function getToolDefinitions(tools?: ToolDefinition[]): Tool[] | undefined {
    return tools ? tools.map(getToolDefinition) : undefined;
}

function getToolDefinition(tool: ToolDefinition): Tool.ToolSpecMember {
    return {
        toolSpec: {
            name: tool.name,
            description: tool.description,
            inputSchema: {
                json: tool.input_schema as any,
            }
        }
    }
}

/**
 * Update the conversation messages
 * @param prompt
 * @param response
 * @returns
 */
function updateConversation(conversation: ConverseRequest, prompt: ConverseRequest): ConverseRequest {
    const combinedMessages = [...(conversation?.messages || []), ...(prompt.messages || [])];
    const combinedSystem = prompt.system || conversation?.system;

    return {
        modelId: prompt?.modelId || conversation?.modelId,
        messages: combinedMessages.length > 0 ? combinedMessages : [],
        system: combinedSystem && combinedSystem.length > 0 ? combinedSystem : undefined,
    };
}

function formatAmazonModalities(modalities: ModelModality[]): string[] {
    const standardizedModalities: string[] = [];
    for (const modality of modalities) {
        if (modality === ModelModality.TEXT) {
            standardizedModalities.push("text");
        } else if (modality === ModelModality.IMAGE) {
            standardizedModalities.push("image");
        } else if (modality === ModelModality.EMBEDDING) {
            standardizedModalities.push("embedding");
        } else if (modality == "SPEECH") {
            standardizedModalities.push("audio");
        } else if (modality == "VIDEO") {
            standardizedModalities.push("video");
        } else {
            // Handle other modalities as needed
            standardizedModalities.push((modality as string).toString().toLowerCase());
        }
    }
    return standardizedModalities;
}