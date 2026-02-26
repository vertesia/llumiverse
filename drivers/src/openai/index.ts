import {
    AIModel,
    AbstractDriver,
    Completion,
    CompletionChunkObject,
    CompletionResult,
    DataSource,
    DriverOptions,
    EmbeddingsOptions,
    EmbeddingsResult,
    ExecutionOptions,
    ExecutionTokenUsage,
    JSONSchema,
    LlumiverseError,
    LlumiverseErrorContext,
    ModelType,
    OpenAiDalleOptions,
    OpenAiGptImageOptions,
    Providers,
    ToolDefinition,
    ToolUse,
    TrainingJob,
    TrainingJobStatus,
    TrainingOptions,
    TrainingPromptOptions,
    getConversationMeta,
    getModelCapabilities,
    incrementConversationTurn,
    modelModalitiesToArray,
    stripBase64ImagesFromConversation,
    stripHeartbeatsFromConversation,
    supportsToolUse,
    truncateLargeTextInConversation,
    unwrapConversationArray,
} from "@llumiverse/core";
import OpenAI, { AzureOpenAI } from "openai";
import {
    APIConnectionError,
    APIConnectionTimeoutError,
    APIError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ContentFilterFinishReasonError,
    InternalServerError,
    LengthFinishReasonError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
} from 'openai/error';
import { formatOpenAILikeMultimodalPrompt } from "./openai_format.js";

// Response API types
type ResponseInputItem = OpenAI.Responses.ResponseInputItem;
type EasyInputMessage = OpenAI.Responses.EasyInputMessage;

// Helper function to convert string to CompletionResult[]
function textToCompletionResult(text: string): CompletionResult[] {
    return text ? [{ type: "text", value: text }] : [];
}

//TODO: Do we need a list?, replace with if statements and modernize?
const supportFineTunning = new Set([
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
    "babbage-002",
    "davinci-002",
    "gpt-4-0613"
]);

export interface BaseOpenAIDriverOptions extends DriverOptions {
}

export abstract class BaseOpenAIDriver extends AbstractDriver<
    BaseOpenAIDriverOptions,
    ResponseInputItem[]
> {
    abstract provider: Providers.openai | Providers.azure_openai | Providers.xai | Providers.azure_foundry | Providers.openai_compatible;
    abstract service: OpenAI | AzureOpenAI;

    constructor(opts: BaseOpenAIDriverOptions) {
        super(opts);
        this.formatPrompt = formatOpenAILikeMultimodalPrompt;
    }

    extractDataFromResponse(
        _options: ExecutionOptions,
        result: OpenAI.Responses.Response
    ): Completion {
        const tokenInfo = mapUsage(result.usage);

        const tools = collectTools(result.output);
        // Collect all parts in order (text and images)
        const allResults = extractCompletionResults(result.output);

        if (allResults.length === 0 && !tools) {
            this.logger.error({ result }, "[OpenAI] Response is not valid");
            throw new Error("Response is not valid: no data");
        }

        return {
            result: allResults,
            token_usage: tokenInfo,
            finish_reason: responseFinishReason(result, tools),
            tool_use: tools,
        };
    }

    async requestTextCompletionStream(prompt: ResponseInputItem[], options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        if (options.model_options?._option_id !== "openai-text" && options.model_options?._option_id !== "openai-thinking") {
            this.logger.warn({ options: options.model_options }, "Invalid model options");
        }

        // Include conversation history (same as non-streaming)
        // Fix orphaned function_call items (can occur when agent is stopped mid-tool-execution)
        let conversation = fixOrphanedToolUse(updateConversation(options.conversation, prompt));

        const toolDefs = getToolDefinitions(options.tools);
        const useTools: boolean = toolDefs ? supportsToolUse(options.model, this.provider, true) : false;

        // When no tools are provided but conversation contains function_call/function_call_output
        // items (e.g. checkpoint summary calls), convert them to text to avoid API errors
        if (!useTools) {
            conversation = convertOpenAIFunctionItemsToText(conversation);
        }

        convertRoles(prompt, options.model);

        const model_options = options.model_options as any;
        insert_image_detail(prompt, model_options?.image_detail ?? "auto");

        let parsedSchema: JSONSchema | undefined = undefined;
        let strictMode = false;
        if (options.result_schema && supportsSchema(options.model)) {
            try {
                parsedSchema = openAISchemaFormat(options.result_schema);
                strictMode = true;
            }
            catch (e) {
                parsedSchema = limitedSchemaFormat(options.result_schema);
                strictMode = false;
            }
        }

        const reasoning = model_options?.reasoning_effort ? { effort: model_options.reasoning_effort } : undefined;
        const isReasoningModel = /\b(o1|o3|o4)\b/.test(options.model);

        const stream = await this.service.responses.create({
            stream: true,
            model: options.model,
            input: conversation,
            reasoning,
            temperature: isReasoningModel ? undefined : model_options?.temperature,
            top_p: isReasoningModel ? undefined : model_options?.top_p,
            max_output_tokens: model_options?.max_tokens,
            tools: useTools ? toolDefs : undefined,
            text: parsedSchema ? {
                format: {
                    type: "json_schema",
                    name: "format_output",
                    schema: parsedSchema,
                    strict: strictMode,
                }
            } : undefined,
        });

        return mapResponseStream(stream);
    }

    async requestTextCompletion(prompt: ResponseInputItem[], options: ExecutionOptions): Promise<Completion> {
        if (options.model_options?._option_id !== "openai-text" && options.model_options?._option_id !== "openai-thinking") {
            this.logger.warn({ options: options.model_options }, "Invalid model options");
        }

        convertRoles(prompt, options.model);

        const model_options = options.model_options as any;
        insert_image_detail(prompt, model_options?.image_detail ?? "auto");

        const toolDefs = getToolDefinitions(options.tools);
        const useTools: boolean = toolDefs ? supportsToolUse(options.model, this.provider) : false;

        // Fix orphaned function_call items (can occur when agent is stopped mid-tool-execution)
        let conversation = fixOrphanedToolUse(updateConversation(options.conversation, prompt));

        // When no tools are provided but conversation contains function_call/function_call_output
        // items (e.g. checkpoint summary calls), convert them to text to avoid API errors
        if (!useTools) {
            conversation = convertOpenAIFunctionItemsToText(conversation);
        }

        let parsedSchema: JSONSchema | undefined = undefined;
        let strictMode = false;
        if (options.result_schema && supportsSchema(options.model)) {
            try {
                parsedSchema = openAISchemaFormat(options.result_schema);
                strictMode = true;
            }
            catch (e) {
                parsedSchema = limitedSchemaFormat(options.result_schema);
                strictMode = false;
            }
        }

        const reasoning = model_options?.reasoning_effort ? { effort: model_options.reasoning_effort } : undefined;
        const isReasoningModel = /\b(o1|o3|o4)\b/.test(options.model);

        const res = await this.service.responses.create({
            stream: false,
            model: options.model,
            input: conversation,
            reasoning,
            temperature: isReasoningModel ? undefined : model_options?.temperature,
            top_p: isReasoningModel ? undefined : model_options?.top_p,
            max_output_tokens: model_options?.max_tokens, //TODO: use max_tokens for older models, currently relying on OpenAI to handle it
            tools: useTools ? toolDefs : undefined,
            text: parsedSchema ? {
                format: {
                    type: "json_schema",
                    name: "format_output",
                    schema: parsedSchema,
                    strict: strictMode,
                }
            } : undefined,
        });

        const completion = this.extractDataFromResponse(options, res);
        if (options.include_original_response) {
            completion.original_response = res;
        }

        conversation = updateConversation(conversation, createAssistantMessageFromCompletion(completion));

        // Increment turn counter for deferred stripping
        conversation = incrementConversationTurn(conversation) as ResponseInputItem[];

        // Strip large base64 image data based on options.stripImagesAfterTurns
        const currentTurn = getConversationMeta(conversation).turnNumber;
        const stripOptions = {
            keepForTurns: options.stripImagesAfterTurns ?? Infinity,
            currentTurn,
            textMaxTokens: options.stripTextMaxTokens
        };
        let processedConversation = stripBase64ImagesFromConversation(conversation, stripOptions);

        // Truncate large text content if configured
        processedConversation = truncateLargeTextInConversation(processedConversation, stripOptions);

        // Strip old heartbeat status messages
        processedConversation = stripHeartbeatsFromConversation(processedConversation, {
            keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
            currentTurn,
        });

        completion.conversation = processedConversation;

        return completion;
    }

    protected canStream(_options: ExecutionOptions): Promise<boolean> {
        // Image generation models don't support streaming
        if (_options.model.includes("dall-e")
            || _options.model.includes("gpt-image")
            || _options.model.includes("chatgpt-image")) {
            return Promise.resolve(false);
        }

        if (_options.model.includes("o1")
            && !(_options.model.includes("mini") || _options.model.includes("preview"))) {
            //o1 full does not support streaming
            //TODO: Update when OpenAI adds support for streaming, last check 16/02/2025
            return Promise.resolve(false);
        }
        return Promise.resolve(true);
    }

    /**
     * Build conversation context after streaming completion.
     * Reconstructs the assistant message from accumulated results and applies stripping.
     */
    buildStreamingConversation(
        prompt: ResponseInputItem[],
        result: unknown[],
        toolUse: unknown[] | undefined,
        options: ExecutionOptions
    ): ResponseInputItem[] | undefined {
        // Build assistant message from accumulated CompletionResult[]
        const completionResults = result as CompletionResult[];

        const textContent = completionResultsToText(completionResults);

        // Start with the conversation from options or the prompt
        let conversation = updateConversation(options.conversation, prompt);

        // Add assistant message as EasyInputMessage
        if (textContent) {
            const assistantMessage: EasyInputMessage = {
                role: 'assistant',
                content: textContent,
            };
            conversation = updateConversation(conversation, [assistantMessage]);
        }

        // Add function calls as separate items (Response API format)
        if (toolUse && toolUse.length > 0) {
            const functionCalls: OpenAI.Responses.ResponseFunctionToolCall[] = (toolUse as ToolUse[]).map(t => ({
                type: 'function_call' as const,
                call_id: t.id,
                name: t.tool_name,
                arguments: typeof t.tool_input === 'string' ? t.tool_input : JSON.stringify(t.tool_input ?? {}),
            }));
            conversation = updateConversation(conversation, functionCalls);
        }

        // Increment turn counter
        conversation = incrementConversationTurn(conversation) as ResponseInputItem[];

        // Apply stripping based on options
        const currentTurn = getConversationMeta(conversation).turnNumber;
        const stripOptions = {
            keepForTurns: options.stripImagesAfterTurns ?? Infinity,
            currentTurn,
            textMaxTokens: options.stripTextMaxTokens
        };
        let processedConversation = stripBase64ImagesFromConversation(conversation, stripOptions);
        processedConversation = truncateLargeTextInConversation(processedConversation, stripOptions);
        processedConversation = stripHeartbeatsFromConversation(processedConversation, {
            keepForTurns: options.stripHeartbeatsAfterTurns ?? 1,
            currentTurn,
        });

        return processedConversation as ResponseInputItem[];
    }

    createTrainingPrompt(options: TrainingPromptOptions): Promise<string> {
        if (options.model.includes("gpt")) {
            return super.createTrainingPrompt(options);
        } else {
            // babbage, davinci not yet implemented
            throw new Error("Unsupported model for training: " + options.model);
        }
    }

    async startTraining(dataset: DataSource, options: TrainingOptions): Promise<TrainingJob> {
        const url = await dataset.getURL();
        const file = await this.service.files.create({
            file: await fetch(url),
            purpose: "fine-tune",
        });

        const job = await this.service.fineTuning.jobs.create({
            training_file: file.id,
            model: options.model,
            hyperparameters: options.params
        })

        return jobInfo(job);
    }

    async cancelTraining(jobId: string): Promise<TrainingJob> {
        const job = await this.service.fineTuning.jobs.cancel(jobId);
        return jobInfo(job);
    }

    async getTrainingJob(jobId: string): Promise<TrainingJob> {
        const job = await this.service.fineTuning.jobs.retrieve(jobId);
        return jobInfo(job);
    }

    // ========= management API =============

    async validateConnection(): Promise<boolean> {
        try {
            await this.service.models.list();
            return true;
        } catch (error) {
            return false;
        }
    }

    listTrainableModels(): Promise<AIModel<string>[]> {
        return this._listModels((m) => supportFineTunning.has(m.id));
    }

    async listModels(): Promise<AIModel[]> {
        return this._listModels();
    }

    async _listModels(filter?: (m: OpenAI.Models.Model) => boolean): Promise<AIModel[]> {
        let result = (await this.service.models.list()).data;

        //Some of these use the completions API instead of the chat completions API.
        //Others are for non-text input modalities. Therefore common to both.
        const wordBlacklist = ["embed", "whisper", "transcribe", "audio", "moderation", "tts",
            "realtime", "babbage", "davinci", "codex", "o1-pro", "computer-use", "sora"];


        //OpenAI has very little information, filtering based on name.
        result = result.filter((m) => {
            return !wordBlacklist.some((word) => m.id.includes(word));
        });

        const models = filter ? result.filter(filter) : result;
        const aiModels = models.map((m) => {
            const modelCapability = getModelCapabilities(m.id, "openai");
            let owner = m.owned_by;
            if (owner == "system") {
                owner = "openai";
            }

            // Determine model type based on capabilities
            let modelType = ModelType.Text;
            if (m.id.includes("dall-e") || m.id.includes("gpt-image")) {
                modelType = ModelType.Image;
            }


            return {
                id: m.id,
                name: m.id,
                provider: this.provider,
                owner: owner,
                type: modelType,
                input_modalities: modelModalitiesToArray(modelCapability.input),
                output_modalities: modelModalitiesToArray(modelCapability.output),
                tool_support: modelCapability.tool_support,
            } satisfies AIModel<string>;
        }).sort((a, b) => a.id.localeCompare(b.id));

        return aiModels;
    }


    async generateEmbeddings({ text, image, model = "text-embedding-3-small" }: EmbeddingsOptions): Promise<EmbeddingsResult> {

        if (image) {
            throw new Error("Image embeddings not supported by OpenAI");
        }

        if (!text) {
            throw new Error("No text provided");
        }

        const res = await this.service.embeddings.create({
            input: text,
            model: model,
        });

        const embeddings = res.data[0].embedding;

        if (!embeddings || embeddings.length === 0) {
            throw new Error("No embedding found");
        }

        return { values: embeddings, model } satisfies EmbeddingsResult;
    }

    imageModels = ["dall-e", "gpt-image", "chatgpt-image"];

    /**
     * Determine if a model is specifically an image generation model (not conversational image model)
     */
    isImageModel(model: string): boolean {
        // DALL-E models are standalone image generation
        // gpt-image models can generate images in conversations, not standalone
        return this.imageModels.some(imageModel => model.includes(imageModel));
    }

    /**
     * Request image generation from standalone Images API
     * Supports: DALL-E 2, DALL-E 3, GPT-image models (for edit/variation)
     */
    async requestImageGeneration(prompt: ResponseInputItem[], options: ExecutionOptions): Promise<Completion> {
        this.logger.debug(`[${this.provider}] Generating image with model ${options.model}`);

        const model_options = options.model_options as OpenAiDalleOptions | OpenAiGptImageOptions | undefined;

        // Extract prompt text from ResponseInputItem[]
        let promptText = "";
        for (const item of prompt) {
            if ('content' in item && typeof item.content === 'string') {
                promptText += item.content + "\\n";
            } else if ('content' in item && Array.isArray(item.content)) {
                // Extract text from content array
                for (const part of item.content) {
                    if ('type' in part && part.type === 'input_text' && 'text' in part) {
                        promptText += part.text + "\\n";
                    }
                }
            }
        }
        promptText = promptText.trim();

        try {
            const generateParams: OpenAI.Images.ImageGenerateParamsNonStreaming = {
                model: options.model,
                prompt: promptText,
                size: model_options?.size || "1024x1024",
            };

            // Add DALL-E specific options
            if (options.model.includes("dall-e") || model_options?._option_id === "openai-dalle") {
                const dalleOptions = model_options as OpenAiDalleOptions | undefined;
                generateParams.n = dalleOptions?.n || 1;
                generateParams.response_format = dalleOptions?.response_format || "b64_json";

                if (options.model.includes("dall-e-3")) {
                    generateParams.quality = dalleOptions?.image_quality || "standard";
                    if (dalleOptions?.style) {
                        generateParams.style = dalleOptions.style;
                    }
                }
            } else {
                // Default for other models
                generateParams.n = 1;
            }

            const response = await this.service.images.generate(generateParams);

            // Convert response to CompletionResults
            const results: CompletionResult[] = [];

            if (response.data) {
                for (const image of response.data) {
                    let imageValue: string;

                    if (image.b64_json) {
                        // Base64 format
                        imageValue = `data:image/png;base64,${image.b64_json}`;
                    } else if (image.url) {
                        // URL format
                        imageValue = image.url;
                    } else {
                        continue;
                    }

                    results.push({
                        type: "image",
                        value: imageValue
                    });
                }
            }

            return {
                result: results
            };

        } catch (error: any) {
            this.logger.error({ error }, `[${this.provider}] Image generation failed`);
            return {
                result: [],
                error: {
                    message: error.message,
                    code: error.code || 'GENERATION_FAILED'
                }
            };
        }
    }

    /**
     * Format OpenAI API errors into LlumiverseError with proper status codes and retryability.
     * 
     * OpenAI API errors have a specific structure:
     * - APIError.status: HTTP status code (400, 401, 403, 404, 409, 422, 429, 500+)
     * - APIError.error: Error object with type, message, param, code
     * - APIError.requestID: Request ID for support
     * - APIError.code: Error code (e.g., 'invalid_api_key', 'rate_limit_exceeded')
     * - APIError.param: Parameter that caused the error (optional)
     * - APIError.type: Error type (optional)
     * 
     * Common error types:
     * - BadRequestError (400): Invalid request parameters
     * - AuthenticationError (401): Invalid API key
     * - PermissionDeniedError (403): Insufficient permissions
     * - NotFoundError (404): Resource not found
     * - ConflictError (409): Resource conflict
     * - UnprocessableEntityError (422): Validation error
     * - RateLimitError (429): Rate limit exceeded
     * - InternalServerError (500+): Server-side errors
     * - APIConnectionError: Connection issues (no status code)
     * - APIConnectionTimeoutError: Request timeout (no status code)
     * - LengthFinishReasonError: Response truncated due to length
     * - ContentFilterFinishReasonError: Content filtered
     * 
     * This implementation works for:
     * - OpenAI API
     * - Azure OpenAI
     * - xAI (uses OpenAI-compatible API)
     * - Azure Foundry (OpenAI-compatible)
     * - Other OpenAI-compatible APIs
     * 
     * @see https://platform.openai.com/docs/guides/error-codes
     */
    public formatLlumiverseError(
        error: unknown,
        context: LlumiverseErrorContext
    ): LlumiverseError {
        // Check if it's an OpenAI API error
        const isOpenAIError = this.isOpenAIApiError(error);

        if (!isOpenAIError) {
            // Not an OpenAI API error, use default handling
            throw error;
        }

        const apiError = error as APIError;
        const httpStatusCode = apiError.status;

        // Extract error message
        let message = apiError.message || String(error);

        // Extract additional error details (only available on APIError)
        const errorCode = apiError.code;
        const errorParam = apiError.param;
        const errorType = apiError.type;

        // Build user-facing message with status code
        let userMessage = message;

        // Include status code in message (for end-user visibility)
        if (httpStatusCode) {
            userMessage = `[${httpStatusCode}] ${userMessage}`;
        }

        // Add error code if available and not already in message
        if (errorCode && !userMessage.includes(errorCode)) {
            userMessage += ` (code: ${errorCode})`;
        }

        // Add parameter info if available and helpful
        if (errorParam && !userMessage.toLowerCase().includes(errorParam.toLowerCase())) {
            userMessage += ` [param: ${errorParam}]`;
        }

        // Add request ID if available (useful for OpenAI support)
        if (apiError.requestID) {
            userMessage += ` (Request ID: ${apiError.requestID})`;
        }

        // Determine retryability based on OpenAI error types
        const retryable = this.isOpenAIErrorRetryable(error, httpStatusCode, errorCode, errorType);

        // Use the error constructor name as the error name
        const errorName = error.constructor?.name || 'OpenAIError';

        return new LlumiverseError(
            `[${context.provider}] ${userMessage}`,
            retryable,
            context,
            error,
            httpStatusCode,
            errorName
        );
    }

    /**
     * Type guard to check if error is an OpenAI API error or OpenAI-specific error.
     */
    private isOpenAIApiError(error: unknown): error is APIError | OpenAIError {
        return (
            error !== null &&
            typeof error === 'object' &&
            (error instanceof APIError || error instanceof OpenAIError)
        );
    }

    /**
     * Determine if an OpenAI API error is retryable.
     * 
     * Retryable errors:
     * - RateLimitError (429): Rate limit exceeded, retry with backoff
     * - InternalServerError (500+): Server-side errors
     * - APIConnectionTimeoutError: Request timeout
     * - Error codes: 'timeout', 'server_error', 'service_unavailable'
     * - Status codes: 408, 429, 502, 503, 504, 529, 5xx
     * 
     * Non-retryable errors:
     * - BadRequestError (400): Invalid request parameters
     * - AuthenticationError (401): Invalid API key
     * - PermissionDeniedError (403): Insufficient permissions
     * - NotFoundError (404): Resource not found
     * - ConflictError (409): Resource conflict
     * - UnprocessableEntityError (422): Validation error
     * - LengthFinishReasonError: Length limit reached
     * - ContentFilterFinishReasonError: Content filtered
     * - Error codes: 'invalid_api_key', 'invalid_request_error', 'model_not_found'
     * - Other 4xx client errors
     * 
     * @param error - The error object
     * @param httpStatusCode - The HTTP status code if available
     * @param errorCode - The error code if available
     * @param errorType - The error type if available
     * @returns True if retryable, false if not retryable, undefined if unknown
     */
    private isOpenAIErrorRetryable(
        error: unknown,
        httpStatusCode: number | undefined,
        errorCode: string | null | undefined,
        errorType: string | undefined
    ): boolean | undefined {
        // Check specific OpenAI error types by class
        if (error instanceof RateLimitError) return true;
        if (error instanceof InternalServerError) return true;
        if (error instanceof APIConnectionTimeoutError) return true;

        // Non-retryable by error type
        if (error instanceof BadRequestError) return false;
        if (error instanceof AuthenticationError) return false;
        if (error instanceof PermissionDeniedError) return false;
        if (error instanceof NotFoundError) return false;
        if (error instanceof ConflictError) return false;
        if (error instanceof UnprocessableEntityError) return false;
        if (error instanceof LengthFinishReasonError) return false;
        if (error instanceof ContentFilterFinishReasonError) return false;

        // Check error codes (OpenAI specific)
        if (errorCode) {
            // Retryable error codes
            if (errorCode === 'timeout') return true;
            if (errorCode === 'server_error') return true;
            if (errorCode === 'service_unavailable') return true;
            if (errorCode === 'rate_limit_exceeded') return true;

            // Non-retryable error codes
            if (errorCode === 'invalid_api_key') return false;
            if (errorCode === 'invalid_request_error') return false;
            if (errorCode === 'model_not_found') return false;
            if (errorCode === 'insufficient_quota') return false;
            if (errorCode === 'invalid_model') return false;
            if (errorCode.includes('invalid_')) return false;
        }

        // Check error type
        if (errorType === 'invalid_request_error') return false;
        if (errorType === 'authentication_error') return false;

        // Use HTTP status code
        if (httpStatusCode !== undefined) {
            if (httpStatusCode === 429) return true; // Rate limit
            if (httpStatusCode === 408) return true; // Request timeout
            if (httpStatusCode === 502) return true; // Bad gateway
            if (httpStatusCode === 503) return true; // Service unavailable
            if (httpStatusCode === 504) return true; // Gateway timeout
            if (httpStatusCode === 529) return true; // Overloaded
            if (httpStatusCode >= 500 && httpStatusCode < 600) return true; // Server errors
            if (httpStatusCode >= 400 && httpStatusCode < 500) return false; // Client errors
        }

        // Connection errors without status codes
        if (error instanceof APIConnectionError && !(error instanceof APIConnectionTimeoutError)) {
            // Generic connection errors might be retryable (network issues)
            return true;
        }

        // Unknown error type - let consumer decide retry strategy
        return undefined;
    }

}


function jobInfo(job: OpenAI.FineTuning.Jobs.FineTuningJob): TrainingJob {
    //validating_files`, `queued`, `running`, `succeeded`, `failed`, or `cancelled`.
    const jobStatus = job.status;
    let status = TrainingJobStatus.running;
    let details: string | undefined;
    if (jobStatus === 'succeeded') {
        status = TrainingJobStatus.succeeded;
    } else if (jobStatus === 'failed') {
        status = TrainingJobStatus.failed;
        details = job.error ? `${job.error.code} - ${job.error.message} ${job.error.param ? " [" + job.error.param + "]" : ""}` : "error";
    } else if (jobStatus === 'cancelled') {
        status = TrainingJobStatus.cancelled;
    } else {
        status = TrainingJobStatus.running;
        details = jobStatus;
    }
    return {
        id: job.id,
        model: job.fine_tuned_model || undefined,
        status,
        details
    }
}

function mapUsage(usage?: OpenAI.Responses.ResponseUsage | null): ExecutionTokenUsage | undefined {
    if (!usage) {
        return undefined;
    }
    return {
        prompt: usage.input_tokens,
        result: usage.output_tokens,
        total: usage.total_tokens,
    };
}

function completionResultsToText(completionResults: CompletionResult[] | undefined): string {
    if (!completionResults) {
        return '';
    }
    return completionResults
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
}

function createAssistantMessageFromCompletion(completion: Completion): ResponseInputItem[] {
    const textContent = completionResultsToText(completion.result);
    const result: ResponseInputItem[] = [];

    // Add assistant text message if present
    if (textContent) {
        const assistantMessage: EasyInputMessage = {
            role: 'assistant',
            content: textContent,
        };
        result.push(assistantMessage);
    }

    // Add function calls as separate items (Response API format)
    if (completion.tool_use && completion.tool_use.length > 0) {
        for (const t of completion.tool_use) {
            const functionCall: OpenAI.Responses.ResponseFunctionToolCall = {
                type: 'function_call',
                call_id: t.id,
                name: t.tool_name,
                arguments: typeof t.tool_input === 'string'
                    ? t.tool_input
                    : JSON.stringify(t.tool_input ?? {}),
            };
            result.push(functionCall);
        }
    }

    return result;
}

function mapResponseStream(stream: AsyncIterable<OpenAI.Responses.ResponseStreamEvent>): AsyncIterable<CompletionChunkObject> {
    const toolCallMetadata = new Map<string, { syntheticId: string, name?: string }>();

    return {
        async *[Symbol.asyncIterator]() {
            for await (const event of stream) {
                if (event.type === 'response.output_item.added' && event.item.type === 'function_call') {
                    const syntheticId = `tool_${event.output_index}`;
                    const actualId = event.item.id ?? event.item.call_id;
                    if (actualId) {
                        toolCallMetadata.set(actualId, { syntheticId, name: event.item.name });
                    }
                    const toolUse: ToolUse & { _actual_id?: string } = {
                        id: syntheticId,
                        _actual_id: actualId,
                        tool_name: event.item.name,
                        tool_input: '' as any,
                    };
                    yield {
                        result: [],
                        tool_use: [toolUse],
                    } satisfies CompletionChunkObject;
                } else if (event.type === 'response.function_call_arguments.delta') {
                    const metadata = toolCallMetadata.get(event.item_id);
                    const syntheticId = metadata?.syntheticId ?? `tool_${event.output_index}`;
                    const toolUse: ToolUse & { _actual_id?: string } = {
                        id: syntheticId,
                        _actual_id: event.item_id,
                        tool_name: metadata?.name ?? '',
                        tool_input: event.delta as any,
                    };
                    yield {
                        result: [],
                        tool_use: [toolUse],
                    } satisfies CompletionChunkObject;
                }
                // Note: We don't emit response.function_call_arguments.done because the arguments were already
                // streamed via delta events. Emitting it again would duplicate the tool_input content.
                // We only update the metadata to ensure the tool name is captured.
                else if (event.type === 'response.function_call_arguments.done') {
                    // Just update metadata, don't yield (arguments already accumulated from delta events)
                    const metadata = toolCallMetadata.get(event.item_id);
                    const syntheticId = metadata?.syntheticId ?? `tool_${event.output_index}`;
                    const tool_name = metadata?.name ?? event.name ?? '';
                    if (event.item_id) {
                        toolCallMetadata.set(event.item_id, { syntheticId, name: tool_name });
                    }
                } else if (event.type === 'response.output_text.delta') {
                    yield {
                        result: textToCompletionResult(event.delta),
                    } satisfies CompletionChunkObject;
                }
                // Note: We don't emit response.output_text.done because the text was already
                // streamed via delta events. Emitting it again would duplicate the content.
                else if (event.type === 'response.completed' || event.type === 'response.incomplete' || event.type === 'response.failed') {
                    const finalTools = collectTools(event.response.output);
                    yield {
                        result: [],
                        finish_reason: responseFinishReason(event.response, finalTools),
                        token_usage: mapUsage(event.response.usage),
                    } satisfies CompletionChunkObject;
                }
            }
        }
    };
}

function insert_image_detail(items: ResponseInputItem[], detail_level: string): ResponseInputItem[] {
    if (detail_level === "auto" || detail_level === "low" || detail_level === "high") {
        for (const item of items) {
            // Check if it's an EasyInputMessage or Message with content array
            if ('role' in item && 'content' in item && item.role !== 'assistant') {
                const content = (item as EasyInputMessage).content;
                if (Array.isArray(content)) {
                    for (const part of content) {
                        if (typeof part === 'object' && part.type === 'input_image') {
                            (part as any).detail = detail_level;
                        }
                    }
                }
            }
        }
    }
    return items;
}

function convertRoles(items: ResponseInputItem[], model: string): ResponseInputItem[] {
    //New openai models use developer role instead of system
    if (model.includes("o1") || model.includes("o3")) {
        if (model.includes("o1-mini") || model.includes("o1-preview")) {
            //o1-mini and o1-preview support neither system nor developer
            for (const item of items) {
                if ('role' in item && (item as EasyInputMessage).role === 'system') {
                    (item as any).role = 'user';
                }
            }
        } else {
            //Models newer than o1 use developer role
            for (const item of items) {
                if ('role' in item && (item as EasyInputMessage).role === 'system') {
                    (item as any).role = 'developer';
                }
            }
        }
    }
    return items;
}

//Structured output support is typically aligned with tool use support
//Not true for realtime models, which do not support structured output, but do support tool use.
function supportsSchema(model: string): boolean {
    const realtimeModel = model.includes("realtime");
    if (realtimeModel) {
        return false;
    }
    return supportsToolUse(model, "openai");
}

/**
 * Converts function_call and function_call_output items to text messages in OpenAI conversation.
 * Preserves tool call information while removing structured items that require
 * tools to be defined in the API request.
 */
function convertOpenAIFunctionItemsToText(items: ResponseInputItem[]): ResponseInputItem[] {
    const hasFunctionItems = items.some(item => {
        const type = (item as any).type;
        return type === 'function_call' || type === 'function_call_output';
    });
    if (!hasFunctionItems) return items;

    return items.map(item => {
        const typed = item as any;
        if (typed.type === 'function_call') {
            const argsStr = typed.arguments || '';
            const truncated = argsStr.length > 500 ? argsStr.substring(0, 500) + '...' : argsStr;
            return {
                role: 'assistant' as const,
                content: `[Tool call: ${typed.name}(${truncated})]`,
            };
        }
        if (typed.type === 'function_call_output') {
            const output = typed.output || 'No output';
            const truncated = output.length > 500 ? output.substring(0, 500) + '...' : output;
            return {
                role: 'user' as const,
                content: `[Tool result: ${truncated}]`,
            };
        }
        return item;
    });
}

function getToolDefinitions(tools: ToolDefinition[] | undefined | null): OpenAI.Responses.Tool[] | undefined {
    return tools ? tools.map(getToolDefinition) : undefined;
}
function getToolDefinition(toolDef: ToolDefinition): OpenAI.Responses.FunctionTool {
    let parsedSchema: JSONSchema | undefined = undefined;
    let strictMode = false;
    if (toolDef.input_schema) {
        try {
            //TODO: type assertion here is not safe, does not work with satisfies
            parsedSchema = openAISchemaFormat(toolDef.input_schema as JSONSchema);
            strictMode = true;
        }
        catch (e) {
            //TODO: type assertion here is not safe, does not work with satisfies
            parsedSchema = limitedSchemaFormat(toolDef.input_schema as JSONSchema);
            strictMode = false;
        }
    }

    return {
        type: "function",
        name: toolDef.name,
        description: toolDef.description,
        parameters: parsedSchema ?? null,
        strict: strictMode,
    };
}

function updateConversation(conversation: unknown, items: ResponseInputItem[]): ResponseInputItem[] {
    if (!items) {
        // Unwrap array if wrapped, otherwise treat as array
        const unwrapped = unwrapConversationArray<ResponseInputItem>(conversation);
        return unwrapped ?? (conversation as ResponseInputItem[] || []);
    }
    if (!conversation) {
        return items;
    }
    // Unwrap array if wrapped, otherwise treat as array
    const unwrapped = unwrapConversationArray<ResponseInputItem>(conversation);
    const convArray = unwrapped ?? (conversation as ResponseInputItem[]);
    return [...convArray, ...items];
}

export function collectTools(output?: OpenAI.Responses.ResponseOutputItem[]): ToolUse[] | undefined {
    if (!output) {
        return undefined;
    }

    const tools: ToolUse[] = [];
    for (const item of output) {
        if (item.type === 'function_call') {
            const id = item.call_id || item.id;
            if (!id) {
                continue;
            }
            tools.push({
                id,
                tool_name: item.name ?? '',
                tool_input: safeJsonParse(item.arguments),
            });
        }
    }
    return tools.length > 0 ? tools : undefined;
}

/**
 * Collect all parts (text and images) from response output in order.
 * This preserves the original ordering of text and image parts.
 */
function extractCompletionResults(output?: OpenAI.Responses.ResponseOutputItem[]): CompletionResult[] {
    if (!output) {
        return [];
    }

    const results: CompletionResult[] = [];
    for (const item of output) {
        if (item.type === 'message') {
            // Extract text from message content
            for (const part of item.content) {
                if (part.type === 'output_text' && part.text) {
                    results.push({
                        type: "text",
                        value: part.text
                    });
                }
            }
        } else if (item.type === 'image_generation_call' && 'result' in item && item.result) {
            // GPT-image models return base64 encoded images in result field
            const base64Data = item.result;
            // Format as data URL for consistency with other image outputs
            const imageUrl = base64Data.startsWith('data:')
                ? base64Data
                : `data:image/png;base64,${base64Data}`;
            results.push({
                type: "image",
                value: imageUrl
            });
        }
    }
    return results;
}

//For strict mode false
function limitedSchemaFormat(schema: JSONSchema): JSONSchema {
    const formattedSchema = { ...schema };

    // Defaults not supported
    delete formattedSchema.default;

    // OpenAI requires type field even in non-strict mode
    // If no type is specified, default to 'object' for properties with format/editor hints,
    // otherwise 'string' as a safe fallback
    if (!formattedSchema.type && formattedSchema.description) {
        // Properties with format: "document" or editor hints are typically objects
        if (formattedSchema.format === 'document' || formattedSchema.editor) {
            formattedSchema.type = 'object';
        } else {
            formattedSchema.type = 'string';
        }
    }

    if (formattedSchema?.properties) {
        // Process each property recursively
        for (const propName of Object.keys(formattedSchema.properties)) {
            const property = formattedSchema.properties[propName];

            // Recursively process properties
            formattedSchema.properties[propName] = limitedSchemaFormat(property);

            // Process arrays with items of type object
            if (property?.type === 'array' && property.items && property.items?.type === 'object') {
                formattedSchema.properties[propName] = {
                    ...property,
                    items: limitedSchemaFormat(property.items),
                };
            }
        }
    }

    return formattedSchema;
}

//For strict mode true
function openAISchemaFormat(schema: JSONSchema, nesting: number = 0): JSONSchema {
    if (nesting > 5) {
        throw new Error("OpenAI schema nesting too deep");
    }

    const formattedSchema = { ...schema };

    // Defaults not supported
    delete formattedSchema.default;

    // Additional properties not supported, required to be set.
    if (formattedSchema?.type === "object") {
        formattedSchema.additionalProperties = false;
    }

    if (formattedSchema?.properties) {
        // Set all properties as required
        formattedSchema.required = Object.keys(formattedSchema.properties);

        // Process each property recursively
        for (const propName of Object.keys(formattedSchema.properties)) {
            const property = formattedSchema.properties[propName];

            // OpenAI strict mode requires all properties to have a type
            if (!property?.type) {
                throw new Error(`Property '${propName}' is missing required 'type' field for OpenAI strict mode`);
            }

            // Recursively process properties
            formattedSchema.properties[propName] = openAISchemaFormat(property, nesting + 1);

            // Process arrays with items of type object
            if (property?.type === 'array' && property.items && property.items?.type === 'object') {
                formattedSchema.properties[propName] = {
                    ...property,
                    items: openAISchemaFormat(property.items, nesting + 1),
                };
            }
        }
    }
    if (formattedSchema?.type === 'object' && (!formattedSchema?.properties || Object.keys(formattedSchema?.properties ?? {}).length == 0)) {
        //If no properties are defined, then additionalProperties: true was set or the object would be empty.
        //OpenAI does not support this on structured output/ strict mode.
        throw new Error("OpenAI does not support empty objects or objects with additionalProperties set to true");
    }

    return formattedSchema
}

function responseFinishReason(response: OpenAI.Responses.Response, tools?: ToolUse[] | undefined): string | undefined {
    if (tools && tools.length > 0) {
        return "tool_use";
    }
    if (response.status === 'incomplete') {
        if (response.incomplete_details?.reason === 'max_output_tokens') {
            return 'length';
        }
        return response.incomplete_details?.reason ?? 'incomplete';
    }
    if (response.status && response.status !== 'completed') {
        return response.status;
    }
    return 'stop';
}

/**
 * Fix orphaned function_call items in the OpenAI Responses API conversation.
 *
 * When an agent is stopped mid-tool-execution, the conversation may contain
 * function_call items without matching function_call_output items. The OpenAI
 * Responses API requires every function_call to have a matching function_call_output.
 *
 * This function detects such cases and injects synthetic function_call_output items
 * indicating the tools were interrupted, allowing the conversation to continue.
 */
export function fixOrphanedToolUse(items: ResponseInputItem[]): ResponseInputItem[] {
    if (items.length < 2) return items;

    // First pass: collect all function_call_output call_ids
    const outputCallIds = new Set<string>();
    for (const item of items) {
        if ('type' in item && item.type === 'function_call_output') {
            outputCallIds.add((item as OpenAI.Responses.ResponseInputItem.FunctionCallOutput).call_id);
        }
    }

    // Second pass: build result, injecting synthetic outputs for orphaned function_calls
    const result: ResponseInputItem[] = [];
    const pendingCalls = new Map<string, string>(); // call_id -> tool name

    for (const item of items) {
        if ('type' in item && item.type === 'function_call') {
            const fc = item as OpenAI.Responses.ResponseFunctionToolCall;
            // Only track if there's no matching output anywhere in the conversation
            if (!outputCallIds.has(fc.call_id)) {
                pendingCalls.set(fc.call_id, fc.name ?? 'unknown');
            }
            result.push(item);
        } else if ('type' in item && item.type === 'function_call_output') {
            result.push(item);
        } else {
            // Before any non-function item, flush pending orphaned calls
            if (pendingCalls.size > 0) {
                for (const [callId, toolName] of pendingCalls) {
                    result.push({
                        type: 'function_call_output',
                        call_id: callId,
                        output: `[Tool interrupted: The user stopped the operation before "${toolName}" could execute.]`,
                    });
                }
                pendingCalls.clear();
            }
            result.push(item);
        }
    }

    // Handle trailing orphans at the end of the conversation
    if (pendingCalls.size > 0) {
        for (const [callId, toolName] of pendingCalls) {
            result.push({
                type: 'function_call_output',
                call_id: callId,
                output: `[Tool interrupted: The user stopped the operation before "${toolName}" could execute.]`,
            });
        }
    }

    return result;
}

function safeJsonParse(value: unknown): any {
    if (typeof value !== 'string') {
        return value;
    }
    try {
        return JSON.parse(value);
    } catch {
        return value;
    }
}
