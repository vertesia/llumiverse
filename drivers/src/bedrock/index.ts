import {
    Bedrock, CreateModelCustomizationJobCommand, FoundationModelSummary, GetModelCustomizationJobCommand,
    GetModelCustomizationJobCommandOutput, ModelCustomizationJobStatus, ModelModality, StopModelCustomizationJobCommand
} from "@aws-sdk/client-bedrock";
import { BedrockRuntime, ContentBlock, ConverseRequest, ConverseResponse, ConverseStreamOutput, InferenceConfiguration, Message, Tool } from "@aws-sdk/client-bedrock-runtime";
import { S3Client } from "@aws-sdk/client-s3";
import { AwsCredentialIdentity, Provider } from "@aws-sdk/types";
import {
    AbstractDriver, AIModel,
    BedrockClaudeOptions,
    BedrockGptOssOptions,
    BedrockPalmyraOptions,
    Completion, CompletionChunkObject,
    CompletionResult,
    DataSource,
    deserializeBinaryFromStorage,
    DriverOptions, EmbeddingsOptions, EmbeddingsResult,
    ExecutionOptions, ExecutionTokenUsage,
    getConversationMeta,
    getMaxTokensLimitBedrock,
    getModelCapabilities,
    incrementConversationTurn,
    LlumiverseError, LlumiverseErrorContext,
    modelModalitiesToArray,
    ModelOptions,
    NovaCanvasOptions,
    PromptSegment,
    StatelessExecutionOptions,
    stripBinaryFromConversation,
    stripHeartbeatsFromConversation,
    TextFallbackOptions, ToolDefinition, ToolUse, TrainingJob, TrainingJobStatus, TrainingOptions,
    truncateLargeTextInConversation
} from "@llumiverse/core";
import { transformAsyncIterator } from "@llumiverse/core/async";
import { formatNovaPrompt, NovaMessagesPrompt } from "@llumiverse/core/formatters";
import { LRUCache } from "mnemonist";
import { converseConcatMessages, converseJSONprefill, converseSystemToMessages, formatConversePrompt } from "./converse.js";
import { formatNovaImageGenerationPayload, NovaImageGenerationTaskType } from "./nova-image-payload.js";
import { forceUploadFile } from "./s3.js";
import {
    formatTwelvelabsPegasusPrompt,
    TwelvelabsMarengoRequest,
    TwelvelabsMarengoResponse,
    TwelvelabsPegasusRequest
} from "./twelvelabs.js";

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
        let maxSupportedTokens = getMaxTokensLimitBedrock(option.model) ?? 8192; // Should always return a number for claude, 8192 is to satisfy the TypeScript type checker;
        // Fallback to the default max tokens limit for the model
        if (option.model.includes('claude-3-7-sonnet') && (modelOptions?.thinking_budget_tokens ?? 0) < 48000) {
            maxSupportedTokens = 64000; // Claude 3.7 can go up to 128k with a beta header, but when no max tokens is specified, we default to 64k.
        }
        return maxSupportedTokens;
    }
}

/**
 * Parse Claude model version from model string.
 * @param modelString - The model identifier string
 * @returns An object with major and minor version numbers, or null if not parseable
 */
function parseClaudeVersion(modelString: string): { major: number; minor: number } | null {
    // Match pattern: claude-[optional variant]-{major}-[optional 1-2 digit minor]
    // The minor version is limited to 1-2 digits to avoid matching dates (YYYYMMDD format)
    const match = modelString.match(/claude-(?:[a-z]+-)?(\d+)(?:-(\d{1,2}))?(?:-|\b)/);
    if (match) {
        return {
            major: parseInt(match[1], 10),
            minor: match[2] ? parseInt(match[2], 10) : 0
        };
    }
    return null;
}

/**
 * Check if a Claude model version is greater than or equal to a target version.
 * @returns true if the model version is >= target version, false otherwise
 */
function isClaudeVersionGTE(modelString: string, targetMajor: number, targetMinor: number): boolean {
    const version = parseClaudeVersion(modelString);
    if (!version) {
        return false;
    }
    if (version.major > targetMajor) {
        return true;
    }
    if (version.major === targetMajor && version.minor >= targetMinor) {
        return true;
    }
    return false;
}

export type BedrockPrompt = NovaMessagesPrompt | ConverseRequest | TwelvelabsPegasusRequest;

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
        if (opts.model.includes("twelvelabs.pegasus")) {
            return await formatTwelvelabsPegasusPrompt(segments, opts);
        }
        return await formatConversePrompt(segments, opts);
    }

    /**
     * Format AWS Bedrock errors into LlumiverseError with proper status codes and retryability.
     * 
     * AWS SDK errors provide:
     * - error.name: The exception type (e.g., "ThrottlingException")
     * - error.$metadata.httpStatusCode: The HTTP status code
     * - error.$metadata.requestId: The AWS request ID for tracking
     * - error.$fault: "client" or "server" indicating error category
     * 
     * @param error - The AWS SDK error
     * @param context - Context about where the error occurred
     * @returns A standardized LlumiverseError
     */
    public formatLlumiverseError(
        error: unknown,
        context: LlumiverseErrorContext
    ): LlumiverseError {
        // Check if it's an AWS SDK error with $metadata
        const awsError = error as any;
        const hasMetadata = awsError?.$metadata !== undefined;

        if (!hasMetadata) {
            // Not an AWS SDK error, use default handling
            return super.formatLlumiverseError(error, context);
        }

        // Extract AWS-specific fields
        const errorName = awsError.name || 'UnknownError';
        const httpStatusCode = awsError.$metadata?.httpStatusCode;
        const requestId = awsError.$metadata?.requestId;
        const fault = awsError.$fault; // "client" or "server"

        // Extract error message - handle both Error instances and plain objects
        let message: string;
        if (error instanceof Error) {
            message = error.message;
        } else if (typeof awsError.message === 'string') {
            message = awsError.message;
        } else {
            message = String(error);
        }

        // Build user-facing message with error name and status code
        let userMessage = message;

        // Include status code in message if available (for end-user visibility)
        if (httpStatusCode) {
            userMessage = `[${httpStatusCode}] ${userMessage}`;
        }

        // Prefix with error name if it's meaningful (not just "Error")
        if (errorName && errorName !== 'Error' && errorName !== 'UnknownError') {
            userMessage = `${errorName}: ${userMessage}`;
        }

        // Add request ID if available (useful for AWS support)
        if (requestId) {
            userMessage += ` (Request ID: ${requestId})`;
        }

        // Determine retryability based on AWS error types
        const retryable = this.isBedrockErrorRetryable(errorName, httpStatusCode, fault);

        return new LlumiverseError(
            `[${this.provider}] ${userMessage}`,
            retryable,
            context,
            error,
            httpStatusCode, // Only set code if we have numeric status code
            errorName       // Preserve AWS error name
        );
    }

    /**
     * Determine if a Bedrock error is retryable based on error type and status.
     * 
     * Retryable errors:
     * - ThrottlingException: Rate limit exceeded, retry with backoff
     * - ServiceUnavailableException: Service temporarily down
     * - InternalServerException: Server-side error
     * - ServiceQuotaExceededException: Quota exhausted, may recover
     * - 5xx status codes: Server errors
     * - 429, 408 status codes: Rate limit, timeout
     * 
     * Non-retryable errors:
     * - ValidationException: Invalid request parameters
     * - AccessDeniedException: Authentication/authorization failure
     * - ResourceNotFoundException: Resource doesn't exist
     * - ConflictException: Resource state conflict
     * - ResourceInUseException: Resource locked by another operation
     * - 4xx status codes (except 429, 408): Client errors
     * 
     * @param errorName - The AWS error name (e.g., "ThrottlingException")
     * @param httpStatusCode - The HTTP status code if available
     * @param fault - The fault type ("client" or "server")
     * @returns True if retryable, false if not retryable, undefined if unknown
     */
    private isBedrockErrorRetryable(
        errorName: string,
        httpStatusCode: number | undefined,
        fault: string | undefined
    ): boolean | undefined {
        // Check specific AWS error types first
        switch (errorName) {
            // Retryable errors
            case 'ThrottlingException':
            case 'ServiceUnavailableException':
            case 'InternalServerException':
            case 'ServiceQuotaExceededException':
                return true;

            // Non-retryable errors
            case 'ValidationException':
            case 'AccessDeniedException':
            case 'ResourceNotFoundException':
            case 'ConflictException':
            case 'ResourceInUseException':
            case 'TooManyTagsException':
                return false;
        }

        // If we have HTTP status code, use it
        if (httpStatusCode !== undefined) {
            if (httpStatusCode === 429 || httpStatusCode === 408) return true; // Rate limit, timeout
            if (httpStatusCode === 529) return true; // Overloaded
            if (httpStatusCode >= 500 && httpStatusCode < 600) return true; // Server errors
            if (httpStatusCode >= 400 && httpStatusCode < 500) return false; // Client errors
        }

        // Fall back to fault type
        if (fault === 'server') return true;
        if (fault === 'client') return false;

        // Unknown error type - let consumer decide retry strategy
        return undefined;
    }

    getExtractedExecution(result: ConverseResponse, _prompt?: BedrockPrompt, options?: ExecutionOptions): CompletionChunkObject {
        let resultText = "";
        let reasoning = "";

        if (result.output?.message?.content) {
            for (const content of result.output.message.content) {
                // Get text output
                if (content.text) {
                    resultText += content.text;
                } else if (content.reasoningContent) {
                    // Extract reasoning content if include_thoughts is true, or if it's a
                    // reasoning-only model (e.g. DeepSeek R1) that returns no text blocks
                    const claudeOptions = options?.model_options as BedrockClaudeOptions;
                    const isReasoningModel = options?.model?.includes('deepseek') && options?.model?.includes('r1');
                    if (claudeOptions?.include_thoughts || isReasoningModel) {
                        if (content.reasoningContent.reasoningText) {
                            reasoning += content.reasoningContent.reasoningText.text;
                        } else if (content.reasoningContent.redactedContent) {
                            // Handle redacted thinking content
                            const redactedData = new TextDecoder().decode(content.reasoningContent.redactedContent);
                            reasoning += `[Redacted thinking: ${redactedData}]`;
                        }
                    } else {
                        this.logger.info("[Bedrock] Not outputting reasoning content as include_thoughts is false");
                    }
                } else {
                    // Get content block type
                    const type = Object.keys(content).find(
                        key => key !== '$unknown' && content[key as keyof typeof content] !== undefined
                    );
                    this.logger.info({ type }, "[Bedrock] Unsupported content response type:");
                }
            }

            // Add spacing if we have reasoning content
            if (reasoning) {
                reasoning += '\n\n';
            }
        }

        const completionResult: CompletionChunkObject = {
            result: reasoning + resultText ? [{ type: "text", value: reasoning + resultText }] : [],
            token_usage: {
                prompt: result.usage?.inputTokens,
                result: result.usage?.outputTokens,
                total: result.usage?.totalTokens,
            },
            finish_reason: converseFinishReason(result.stopReason),
        };

        return completionResult;
    };

    getExtractedStream(result: ConverseStreamOutput, _prompt?: BedrockPrompt, options?: ExecutionOptions): CompletionChunkObject {
        let output: string = "";
        let reasoning: string = "";
        let stop_reason = "";
        let token_usage: ExecutionTokenUsage | undefined;

        // Check if we should include thoughts (always true for reasoning-only models like DeepSeek R1)
        const isReasoningModel = options?.model?.includes('deepseek') && options?.model?.includes('r1');
        const shouldIncludeThoughts = isReasoningModel || (options && (options.model_options as BedrockClaudeOptions)?.include_thoughts);

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
                    // Putting logging here so it only triggers once.
                    this.logger.info("[Bedrock] Not outputting reasoning content as include_thoughts is false");
                }
            } else if (delta) {
                // Get content block type
                const type = Object.keys(delta).find(
                    key => key !== '$unknown' && (delta as any)[key] !== undefined
                );
                this.logger.info({ type }, "[Bedrock] Unsupported content response type:");
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
            result: reasoning + output ? [{ type: "text", value: reasoning + output }] : [],
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
        // // TwelveLabs Pegasus supports streaming according to the documentation
        // if (options.model.includes("twelvelabs.pegasus")) {
        //     return true;
        // }

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

    /**
     * Build conversation context after streaming completion.
     * Reconstructs the assistant message from accumulated results and applies stripping.
     */
    buildStreamingConversation(
        prompt: BedrockPrompt,
        result: unknown[],
        toolUse: unknown[] | undefined,
        options: ExecutionOptions
    ): ConverseRequest | undefined {
        // Only handle ConverseRequest prompts (not NovaMessagesPrompt or TwelvelabsPegasusRequest)
        if (options.model.includes("canvas") || options.model.includes("twelvelabs.pegasus")) {
            return undefined;
        }

        const conversePrompt = prompt as ConverseRequest;
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

        // Deserialize any base64-encoded binary data back to Uint8Array
        const incomingConversation = deserializeBinaryFromStorage(options.conversation) as ConverseRequest;

        // Start with the conversation from options combined with the prompt
        let conversation = updateConversation(incomingConversation, conversePrompt);

        // Build assistant message content
        const messageContent: any[] = [];
        if (textContent) {
            messageContent.push({ text: textContent });
        }
        // Add tool use blocks if present
        if (toolUse && toolUse.length > 0) {
            for (const tool of toolUse as ToolUse[]) {
                messageContent.push({
                    toolUse: {
                        toolUseId: tool.id,
                        name: tool.tool_name,
                        input: tool.tool_input,
                    }
                });
            }
        }

        // Add assistant message
        const assistantMessage: ConverseRequest = {
            messages: [{
                content: messageContent.length > 0 ? messageContent : [{ text: '' }],
                role: "assistant"
            }],
            modelId: conversePrompt.modelId,
        };
        conversation = updateConversation(conversation, assistantMessage);

        // Increment turn counter
        conversation = incrementConversationTurn(conversation) as ConverseRequest;

        // Apply stripping based on options
        const currentTurn = getConversationMeta(conversation).turnNumber;
        const stripOptions = {
            keepForTurns: options.stripImagesAfterTurns ?? Infinity,
            currentTurn,
            textMaxTokens: options.stripTextMaxTokens
        };
        let processedConversation = stripBinaryFromConversation(conversation, stripOptions);
        processedConversation = truncateLargeTextInConversation(processedConversation, stripOptions);
        processedConversation = stripHeartbeatsFromConversation(processedConversation, {
            keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
            currentTurn,
        });

        return processedConversation as ConverseRequest;
    }

    async requestTextCompletion(prompt: BedrockPrompt, options: ExecutionOptions): Promise<Completion> {
        // Handle Twelvelabs Pegasus models
        if (options.model.includes("twelvelabs.pegasus")) {
            return this.requestTwelvelabsPegasusCompletion(prompt as TwelvelabsPegasusRequest, options);
        }

        // Handle other Bedrock models that use Converse API
        const conversePrompt = prompt as ConverseRequest;

        // Deserialize any base64-encoded binary data back to Uint8Array before API call
        const incomingConversation = deserializeBinaryFromStorage(options.conversation) as ConverseRequest;
        let conversation = updateConversation(incomingConversation, conversePrompt);

        const payload = this.preparePayload(conversation, options);
        const executor = this.getExecutor();

        const res = await executor.converse({
            ...payload,
        });

        // Strip reasoningContent from assistant messages before storing in conversation
        // (DeepSeek R1 returns reasoning blocks but rejects them in subsequent user turns)
        const assistantMsg = res.output?.message ?? { content: [{ text: "" }], role: "assistant" };
        if (assistantMsg.content) {
            assistantMsg.content = assistantMsg.content.filter((c: any) => !c.reasoningContent);
        }

        conversation = updateConversation(conversation, {
            messages: [assistantMsg],
            modelId: conversePrompt.modelId,
        });

        // Increment turn counter for deferred stripping
        conversation = incrementConversationTurn(conversation) as ConverseRequest;

        let tool_use: ToolUse[] | undefined = undefined;
        //Get tool requests, we check tool use regardless of finish reason, as you can hit length and still get a valid response.
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

        // Strip/serialize binary data based on options.stripImagesAfterTurns
        const currentTurn = getConversationMeta(conversation).turnNumber;
        const stripOptions = {
            keepForTurns: options.stripImagesAfterTurns ?? Infinity,
            currentTurn,
            textMaxTokens: options.stripTextMaxTokens
        };
        let processedConversation = stripBinaryFromConversation(conversation, stripOptions);

        // Truncate large text content if configured
        processedConversation = truncateLargeTextInConversation(processedConversation, stripOptions);

        // Strip old heartbeat status messages
        processedConversation = stripHeartbeatsFromConversation(processedConversation, {
            keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
            currentTurn,
        });

        const completion = {
            ...this.getExtractedExecution(res, conversePrompt, options),
            original_response: options.include_original_response ? res : undefined,
            conversation: processedConversation,
            tool_use: tool_use,
        };

        return completion;
    }

    private async requestTwelvelabsPegasusCompletion(prompt: TwelvelabsPegasusRequest, options: ExecutionOptions): Promise<Completion> {
        const executor = this.getExecutor();

        const res = await executor.invokeModel({
            modelId: options.model,
            contentType: "application/json",
            accept: "application/json",
            body: JSON.stringify(prompt),
        });

        const decoder = new TextDecoder();
        const body = decoder.decode(res.body);
        const result = JSON.parse(body);

        // Extract the response according to TwelveLabs Pegasus format
        let finishReason: string | undefined;
        switch (result.finishReason) {
            case "stop":
                finishReason = "stop";
                break;
            case "length":
                finishReason = "length";
                break;
            default:
                finishReason = result.finishReason;
        }

        return {
            result: result.message ? [{ type: "text" as const, value: result.message }] : [],
            finish_reason: finishReason,
            original_response: options.include_original_response ? result : undefined,
        };
    }

    private async requestTwelvelabsPegasusCompletionStream(prompt: TwelvelabsPegasusRequest, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const executor = this.getExecutor();

        const res = await executor.invokeModelWithResponseStream({
            modelId: options.model,
            contentType: "application/json",
            accept: "application/json",
            body: JSON.stringify(prompt),
        });

        if (!res.body) {
            throw new Error("[Bedrock] Stream not found in response");
        }

        return transformAsyncIterator(res.body, (chunk: any) => {
            if (chunk.chunk?.bytes) {
                const decoder = new TextDecoder();
                const body = decoder.decode(chunk.chunk.bytes);

                try {
                    const result = JSON.parse(body);

                    // Extract streaming response according to TwelveLabs Pegasus format
                    let finishReason: string | undefined;
                    if (result.finishReason) {
                        switch (result.finishReason) {
                            case "stop":
                                finishReason = "stop";
                                break;
                            case "length":
                                finishReason = "length";
                                break;
                            default:
                                finishReason = result.finishReason;
                        }
                    }

                    return {
                        result: result.delta || result.message ? [{ type: "text" as const, value: result.delta || result.message || "" }] : [],
                        finish_reason: finishReason,
                    } satisfies CompletionChunkObject;
                } catch (error) {
                    // If JSON parsing fails, return empty chunk
                    return {
                        result: [],
                    } satisfies CompletionChunkObject;
                }
            }

            return {
                result: [],
            } satisfies CompletionChunkObject;
        });
    }

    async requestTextCompletionStream(prompt: BedrockPrompt, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        // Handle Twelvelabs Pegasus models
        if (options.model.includes("twelvelabs.pegasus")) {
            return this.requestTwelvelabsPegasusCompletionStream(prompt as TwelvelabsPegasusRequest, options);
        }

        // Handle other Bedrock models that use Converse API
        const conversePrompt = prompt as ConverseRequest;

        // Include conversation history (same as non-streaming)
        // Deserialize any base64-encoded binary data back to Uint8Array before API call
        const incomingConversation = deserializeBinaryFromStorage(options.conversation) as ConverseRequest;
        const conversation = updateConversation(incomingConversation, conversePrompt);

        const payload = this.preparePayload(conversation, options);
        const executor = this.getExecutor();
        return executor.converseStream({
            ...payload,
        }).then((res) => {
            const stream = res.stream;

            if (!stream) {
                throw new Error("[Bedrock] Stream not found in response");
            }

            return transformAsyncIterator(stream, (streamSegment: ConverseStreamOutput) => {
                return this.getExtractedStream(streamSegment, conversePrompt, options);
            });

        }).catch((err) => {
            this.logger.error({ error: err }, "[Bedrock] Failed to stream");
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
            // Claude 4.6 and later versions don't support JSON prefill
            if (isClaudeVersionGTE(options.model, 4, 6)) {
                supportsJSONPrefill = false;
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
            // DeepSeek models: no additional options, no stopSequences, only one of temperature/top_p
            model_options.stop_sequence = undefined;
            model_options.top_p = undefined;
        } else if (options.model.includes("gpt-oss")) {
            const gptOssOptions = model_options as ModelOptions as BedrockGptOssOptions;
            additionalField = {
                reasoning_effort: gptOssOptions?.reasoning_effort,
            };
        }

        //If last message is "```json", add corresponding ``` as a stop sequence.
        if (prompt.messages && prompt.messages.length > 0) {
            if (prompt.messages[prompt.messages.length - 1].content?.[0].text === "```json") {
                const stopSeq = model_options.stop_sequence;
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

        // Clean undefined values from additionalField since AWS Bedrock requires valid JSON
        // and will throw an exception for unrecognized parameters
        const cleanedAdditionalFields = removeUndefinedValues(additionalField);
        const cleanedModelOptions = removeUndefinedValues({
            maxTokens: model_options.max_tokens,
            temperature: model_options.temperature,
            topP: model_options.temperature != null ? undefined : model_options.top_p,
            stopSequences: model_options.stop_sequence,
        } satisfies InferenceConfiguration);

        //Construct the final request payload
        // We only add fields that are defined to avoid AWS errors
        const request: ConverseRequest = {
            modelId: options.model,
        };

        if (prompt.messages) {
            request.messages = prompt.messages;
        }

        if (prompt.system) {
            request.system = prompt.system;
        }

        if (Object.keys(cleanedModelOptions).length > 0) {
            request.inferenceConfig = cleanedModelOptions
        }

        if (Object.keys(cleanedAdditionalFields).length > 0) {
            request.additionalModelRequestFields = cleanedAdditionalFields;
        }

        if (tool_defs?.length) {
            request.toolConfig = {
                tools: tool_defs,
            }
        } else if (request.messages && messagesContainToolBlocks(request.messages)) {
            // Bedrock requires toolConfig when conversation contains toolUse/toolResult blocks.
            // When no tools are provided (e.g. checkpoint summary calls), convert tool blocks
            // to text representations so the conversation data is preserved while satisfying
            // Bedrock's API requirements without making tools callable.
            request.messages = convertToolBlocksToText(request.messages);
        }

        return request;
    }


    protected isImageModel(model: string): boolean {
        return model.includes("titan-image") || model.includes("stable-diffusion") || model.includes("nova-canvas");
    }

    async requestImageGeneration(prompt: NovaMessagesPrompt, options: ExecutionOptions): Promise<Completion> {
        if (options.model_options?._option_id !== "bedrock-nova-canvas") {
            this.logger.warn({ options: options.model_options }, "Invalid model options");
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
        const bedrockResult = JSON.parse(body);

        return {
            error: bedrockResult.error,
            result: bedrockResult.images.map((image: any) => ({
                type: "image" as const,
                value: image
            }))
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

        const supportedPublishers = ["amazon", "anthropic", "cohere", "ai21",
            "mistral", "meta", "deepseek", "writer",
            "openai", "twelvelabs", "qwen"];
        const unsupportedModelsByPublisher = {
            amazon: ["titan-image-generator", "nova-reel", "nova-sonic", "rerank"],
            anthropic: [],
            cohere: ["rerank", "embed"],
            ai21: [],
            mistral: [],
            meta: [],
            deepseek: [],
            writer: [],
            openai: [],
            twelvelabs: ["marengo"],
            qwen: [],
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
                    owner: "custom",
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
                        owner: providerName,
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

        // Handle TwelveLabs Marengo models
        if (model?.includes("twelvelabs.marengo")) {
            return this.generateTwelvelabsMarengoEmbeddings({ text, image, model });
        }

        // Handle other Bedrock embedding models
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

    private async generateTwelvelabsMarengoEmbeddings({ text, image, model }: EmbeddingsOptions): Promise<EmbeddingsResult> {
        const executor = this.getExecutor();

        // Prepare the request payload for TwelveLabs Marengo
        let invokeBody: TwelvelabsMarengoRequest = {
            inputType: "text"
        };

        if (text) {
            invokeBody.inputText = text;
            invokeBody.inputType = "text";
        }

        if (image) {
            // For the embeddings interface, image is expected to be base64
            invokeBody.mediaSource = {
                base64String: image
            };
            invokeBody.inputType = "image";
        }

        const res = await executor.invokeModel({
            modelId: model!,
            contentType: "application/json",
            accept: "application/json",
            body: JSON.stringify(invokeBody),
        });

        const decoder = new TextDecoder();
        const body = decoder.decode(res.body);
        const result: TwelvelabsMarengoResponse = JSON.parse(body);

        // TwelveLabs Marengo returns embedding data
        if (!result.embedding) {
            throw new Error("Embeddings not found in TwelveLabs Marengo response");
        }

        return {
            values: result.embedding,
            model: model!,
            // TwelveLabs Marengo doesn't return token count in the same way
            token_count: undefined
        };
    }

    /**
     * Cleanup AWS SDK clients when the driver is evicted from the cache.
     */
    destroy(): void {
        this._executor?.destroy();
        this._service?.destroy();
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
 * Checks whether any message contains toolUse or toolResult content blocks.
 */
function messagesContainToolBlocks(messages: Message[]): boolean {
    for (const msg of messages) {
        if (!msg.content) continue;
        for (const block of msg.content) {
            if ((block as ContentBlock.ToolUseMember).toolUse ||
                (block as ContentBlock.ToolResultMember).toolResult) {
                return true;
            }
        }
    }
    return false;
}

/**
 * Converts toolUse and toolResult content blocks to text representations.
 * This preserves the tool call information in the conversation while removing
 * the structured tool blocks that require Bedrock's toolConfig to be set.
 *
 * Used when no tools are provided (e.g. checkpoint summary calls) but the
 * conversation history contains tool interactions from prior turns.
 */
function convertToolBlocksToText(messages: Message[]): Message[] {
    return messages.map(msg => {
        if (!msg.content) return msg;
        let hasToolBlocks = false;
        for (const block of msg.content) {
            if ((block as ContentBlock.ToolUseMember).toolUse ||
                (block as ContentBlock.ToolResultMember).toolResult) {
                hasToolBlocks = true;
                break;
            }
        }
        if (!hasToolBlocks) return msg;

        const newContent: ContentBlock[] = [];
        for (const block of msg.content) {
            const toolUse = (block as ContentBlock.ToolUseMember).toolUse;
            const toolResult = (block as ContentBlock.ToolResultMember).toolResult;
            if (toolUse) {
                const inputStr = toolUse.input ? JSON.stringify(toolUse.input) : '';
                const truncatedInput = inputStr.length > 500 ? inputStr.substring(0, 500) + '...' : inputStr;
                newContent.push({
                    text: `[Tool call: ${toolUse.name}(${truncatedInput})]`,
                } as ContentBlock.TextMember);
            } else if (toolResult) {
                const resultTexts: string[] = [];
                if (toolResult.content) {
                    for (const c of toolResult.content) {
                        if ((c as any).text) {
                            const text = (c as any).text as string;
                            resultTexts.push(text.length > 500 ? text.substring(0, 500) + '...' : text);
                        }
                    }
                }
                const resultStr = resultTexts.length > 0 ? resultTexts.join('\n') : 'No text content';
                newContent.push({
                    text: `[Tool result: ${resultStr}]`,
                } as ContentBlock.TextMember);
            } else {
                newContent.push(block);
            }
        }
        return { ...msg, content: newContent };
    });
}

/**
 * Recursively removes undefined values from an object.
 * AWS Bedrock's additionalModelRequestFields must be valid JSON, and undefined is not valid JSON.
 * Any unrecognized parameters will cause an exception.
 */
function removeUndefinedValues<T extends Record<string, any>>(obj: T): Partial<T> {
    if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) {
        return obj;
    }

    const cleaned: any = {};
    for (const [key, value] of Object.entries(obj)) {
        if (value !== undefined) {
            if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
                const cleanedNested = removeUndefinedValues(value);
                // Only include nested objects if they have properties after cleaning
                if (Object.keys(cleanedNested).length > 0) {
                    cleaned[key] = cleanedNested;
                }
            } else {
                cleaned[key] = value;
            }
        }
    }
    return cleaned;
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

    // Fix orphaned toolUse blocks before returning
    const fixedMessages = fixOrphanedToolUse(combinedMessages);

    return {
        modelId: prompt?.modelId || conversation?.modelId,
        messages: fixedMessages.length > 0 ? fixedMessages : [],
        system: combinedSystem && combinedSystem.length > 0 ? combinedSystem : undefined,
    };
}

/**
 * Fix orphaned toolUse blocks in the conversation.
 *
 * When an agent is stopped mid-tool-execution, the assistant message contains toolUse blocks
 * but no corresponding toolResult was added. The AWS Converse API requires that every toolUse
 * must be followed by a toolResult in the next user message.
 *
 * This function detects such cases and injects synthetic toolResult blocks indicating
 * the tools were interrupted, allowing the conversation to continue.
 */
export function fixOrphanedToolUse(messages: Message[]): Message[] {
    if (messages.length < 2) return messages;

    const result: Message[] = [];

    for (let i = 0; i < messages.length; i++) {
        const current = messages[i];
        result.push(current);

        // Check if this is an assistant message with toolUse blocks
        if (current.role === 'assistant' && current.content) {
            // Extract toolUse blocks using simple property check (same pattern as existing Bedrock code)
            const toolUseBlocks: Array<{ toolUseId: string; name: string }> = [];
            for (const block of current.content) {
                if (block.toolUse?.toolUseId) {
                    toolUseBlocks.push({
                        toolUseId: block.toolUse.toolUseId,
                        name: block.toolUse.name ?? 'unknown'
                    });
                }
            }

            if (toolUseBlocks.length > 0) {
                // Check if the next message is a user message with matching toolResults
                const nextMessage = messages[i + 1];

                if (nextMessage && nextMessage.role === 'user' && nextMessage.content) {
                    // Get toolResult IDs from the next message using simple property check
                    const toolResultIds = new Set<string>();
                    for (const block of nextMessage.content) {
                        if (block.toolResult?.toolUseId) {
                            toolResultIds.add(block.toolResult.toolUseId);
                        }
                    }

                    // Find orphaned toolUse blocks (no matching toolResult)
                    const orphanedToolUse = toolUseBlocks.filter(tu => !toolResultIds.has(tu.toolUseId));

                    if (orphanedToolUse.length > 0) {
                        // Inject synthetic toolResults for orphaned toolUse
                        const syntheticResults: ContentBlock[] = orphanedToolUse.map(tu => ({
                            toolResult: {
                                toolUseId: tu.toolUseId,
                                content: [{
                                    text: `[Tool interrupted: The user stopped the operation before "${tu.name}" could execute.]`
                                }]
                            }
                        }));

                        // Prepend synthetic results to the next user message
                        const updatedNextMessage: Message = {
                            ...nextMessage,
                            content: [...syntheticResults, ...nextMessage.content]
                        };

                        // Replace the next message in our iteration
                        messages[i + 1] = updatedNextMessage;
                    }
                } else if (nextMessage && nextMessage.role === 'user' && !nextMessage.content) {
                    // Next message is a user message but has no content
                    // We need to add toolResults
                    const syntheticResults: ContentBlock[] = toolUseBlocks.map(tu => ({
                        toolResult: {
                            toolUseId: tu.toolUseId,
                            content: [{
                                text: `[Tool interrupted: The user stopped the operation before "${tu.name}" could execute.]`
                            }]
                        }
                    }));

                    const updatedNextMessage: Message = {
                        role: 'user',
                        content: syntheticResults
                    };

                    messages[i + 1] = updatedNextMessage;
                }
                // Note: If there's no nextMessage, we leave the conversation as-is.
                // The toolUse blocks are expected to be there - the next turn will provide toolResults.
            }
        }
    }

    return result;
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