import {
    type AIModel,
    type Completion,
    type CompletionChunkObject,
    type CompletionResult,
    type DataSource,
    type DriverOptions,
    type EmbeddingResultItem,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    type ExecutionTokenUsage,
    getConversationMeta,
    getModelCapabilities,
    incrementConversationTurn,
    type JSONSchema,
    LlumiverseError,
    ModelType,
    modelModalitiesToArray,
    normalizeEmbeddingsOptions,
    OPENAI_DEFAULT_EMBEDDING_MODEL,
    type OpenAiDalleOptions,
    type OpenAiGptImageOptions,
    type PromptOptions,
    type PromptSegment,
    type Providers,
    stripBase64ImagesFromConversation,
    stripHeartbeatsFromConversation,
    supportsToolUse,
    type TextFallbackOptions,
    type ToolDefinition,
    type ToolUse,
    type TrainingJob,
    TrainingJobStatus,
    type TrainingOptions,
    type TrainingPromptOptions,
    truncateLargeTextInConversation,
    unwrapConversationArray,
} from '@llumiverse/core';
import type OpenAI from 'openai';
import type { AzureOpenAI } from 'openai';
import { OpenAICompatibleDriverBase } from './openai_compatible.js';
import { formatOpenAILikeMultimodalPrompt } from './openai_format.js';
import { formatOpenAISchema } from './schema.js';

// Response API types
type ResponseInputItem = OpenAI.Responses.ResponseInputItem;
type EasyInputMessage = OpenAI.Responses.EasyInputMessage;
type OpenAIRequestOptions = Partial<TextFallbackOptions> & {
    image_detail?: 'low' | 'high' | 'auto';
    effort?: string;
    reasoning_effort?: string;
    verbosity?: 'low' | 'medium' | 'high';
};
type OpenAIErrorWithStatus = Error & { status?: unknown };
type OpenAIUsageWithProviderDetails = OpenAI.Responses.ResponseUsage & {
    cached_tokens?: number | null;
    cache_write_tokens?: number | null;
    prompt_tokens_details?: {
        cached_tokens?: number | null;
        cache_write_tokens?: number | null;
    } | null;
};
type MutableRoleItem = { role: 'user' | 'developer' | 'system' | 'assistant' };
type MutableInputImagePart = { type: string; detail?: string };
type OpenAIFunctionItem = ResponseInputItem & {
    type?: string;
    name?: string;
    arguments?: string;
    output?: string;
};

// Helper function to convert string to CompletionResult[]
function textToCompletionResult(text: string): CompletionResult[] {
    return text ? [{ type: 'text', value: text }] : [];
}

function hasNumericStatus(error: unknown): boolean {
    return error instanceof Error && typeof (error as OpenAIErrorWithStatus).status === 'number';
}

function isOpenAIReasoningModel(model: string): boolean {
    const normalized = model.toLowerCase();
    return (
        normalized.includes('o1') ||
        normalized.includes('o3') ||
        normalized.includes('o4') ||
        normalized.includes('gpt-5')
    );
}

function openAIReasoning(effort: string | undefined): OpenAI.Responses.ResponseCreateParams['reasoning'] {
    if (!effort) {
        return undefined;
    }
    // Forward provider-native values unchanged so the provider can return an authoritative validation error.
    return { effort } as OpenAI.Responses.ResponseCreateParams['reasoning'];
}

//TODO: Do we need a list?, replace with if statements and modernize?
const supportFineTunning = new Set([
    'gpt-3.5-turbo-1106',
    'gpt-3.5-turbo-0613',
    'babbage-002',
    'davinci-002',
    'gpt-4-0613',
]);

export interface OpenAIResponsesDriverBaseOptions extends DriverOptions {}

/** Reusable Responses text protocol shared by OpenAI-compatible transports. */
export class OpenAIResponsesProtocol {
    async requestTextCompletionStream(
        driver: OpenAIResponsesDriverBase,
        prompt: ResponseInputItem[],
        options: ExecutionOptions,
    ): Promise<AsyncIterable<CompletionChunkObject>> {
        if (
            options.model_options?._option_id !== undefined &&
            options.model_options?._option_id !== 'openai-text' &&
            options.model_options?._option_id !== 'openai-thinking' &&
            options.model_options?._option_id !== 'bedrock-mantle-responses' &&
            options.model_options?._option_id !== 'text-fallback'
        ) {
            driver.logger.debug({ options: options.model_options }, 'Unexpected option id');
        }

        // Include conversation history (same as non-streaming)
        // Fix orphaned function_call items (can occur when agent is stopped mid-tool-execution)
        let conversation = fixOrphanedToolResults(fixOrphanedToolUse(updateConversation(options.conversation, prompt)));

        const toolDefs = getToolDefinitions(options.tools);
        const useTools: boolean = toolDefs ? supportsToolUse(options.model, driver.provider, true) : false;

        // When no tools are provided but conversation contains function_call/function_call_output
        // items (e.g. checkpoint summary calls), convert them to text to avoid API errors
        if (!useTools) {
            conversation = convertOpenAIFunctionItemsToText(conversation);
        }

        convertRoles(prompt, options.model);

        const model_options = options.model_options as OpenAIRequestOptions | undefined;
        insert_image_detail(prompt, model_options?.image_detail ?? 'auto');

        let parsedSchema: JSONSchema | undefined;
        let strictMode = false;
        if (options.result_schema && supportsSchema(options.model, driver.provider)) {
            const formattedSchema = formatOpenAISchema(options.result_schema);
            parsedSchema = formattedSchema.schema;
            strictMode = formattedSchema.strict;
        }

        const isReasoningModel = isOpenAIReasoningModel(options.model);
        const reasoning = openAIReasoning(model_options?.effort ?? model_options?.reasoning_effort);

        const stream = await driver.service.responses.create({
            stream: true,
            model: driver.getResponsesRequestModel(options.model),
            prompt_cache_key: options.prompt_cache_key,
            input: conversation,
            reasoning,
            temperature: isReasoningModel ? undefined : model_options?.temperature,
            top_p: isReasoningModel ? undefined : model_options?.top_p,
            max_output_tokens: model_options?.max_tokens,
            tools: useTools ? toolDefs : undefined,
            text: buildResponseTextConfig(parsedSchema, strictMode, model_options?.verbosity),
        });

        return mapResponseStream(stream);
    }

    async requestTextCompletion(
        driver: OpenAIResponsesDriverBase,
        prompt: ResponseInputItem[],
        options: ExecutionOptions,
    ): Promise<Completion> {
        if (
            options.model_options?._option_id !== undefined &&
            options.model_options?._option_id !== 'openai-text' &&
            options.model_options?._option_id !== 'openai-thinking' &&
            options.model_options?._option_id !== 'bedrock-mantle-responses' &&
            options.model_options?._option_id !== 'text-fallback'
        ) {
            driver.logger.debug({ options: options.model_options }, 'Unexpected option id');
        }

        convertRoles(prompt, options.model);

        const model_options = options.model_options as OpenAIRequestOptions | undefined;
        insert_image_detail(prompt, model_options?.image_detail ?? 'auto');

        const toolDefs = getToolDefinitions(options.tools);
        const useTools: boolean = toolDefs ? supportsToolUse(options.model, driver.provider) : false;

        // Fix orphaned function_call items (can occur when agent is stopped mid-tool-execution)
        let conversation = fixOrphanedToolResults(fixOrphanedToolUse(updateConversation(options.conversation, prompt)));

        // When no tools are provided but conversation contains function_call/function_call_output
        // items (e.g. checkpoint summary calls), convert them to text to avoid API errors
        if (!useTools) {
            conversation = convertOpenAIFunctionItemsToText(conversation);
        }

        let parsedSchema: JSONSchema | undefined;
        let strictMode = false;
        if (options.result_schema && supportsSchema(options.model, driver.provider)) {
            const formattedSchema = formatOpenAISchema(options.result_schema);
            parsedSchema = formattedSchema.schema;
            strictMode = formattedSchema.strict;
        }

        const isReasoningModel = isOpenAIReasoningModel(options.model);
        const reasoning = openAIReasoning(model_options?.effort ?? model_options?.reasoning_effort);

        const res = await driver.service.responses.create({
            stream: false,
            model: driver.getResponsesRequestModel(options.model),
            prompt_cache_key: options.prompt_cache_key,
            input: conversation,
            reasoning,
            temperature: isReasoningModel ? undefined : model_options?.temperature,
            top_p: isReasoningModel ? undefined : model_options?.top_p,
            max_output_tokens: model_options?.max_tokens, //TODO: use max_tokens for older models, currently relying on OpenAI to handle it
            tools: useTools ? toolDefs : undefined,
            text: buildResponseTextConfig(parsedSchema, strictMode, model_options?.verbosity),
        });

        const completion = driver.extractDataFromResponse(options, res);
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
            textMaxTokens: options.stripTextMaxTokens,
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
}

export abstract class OpenAIResponsesDriverBase extends OpenAICompatibleDriverBase<
    OpenAIResponsesDriverBaseOptions,
    ResponseInputItem[]
> {
    abstract provider:
        | Providers.openai
        | Providers.azure_openai
        | Providers.xai
        | Providers.azure_foundry
        | Providers.bedrock
        | Providers.bedrock_mantle
        | Providers.openai_compatible;
    abstract service: OpenAI | AzureOpenAI;
    private readonly responsesProtocol: OpenAIResponsesProtocol;

    constructor(opts: OpenAIResponsesDriverBaseOptions) {
        super(opts);
        this.responsesProtocol = new OpenAIResponsesProtocol();
    }

    protected async formatPrompt(segments: PromptSegment[], options: PromptOptions): Promise<ResponseInputItem[]> {
        const resultSchema = supportsSchema(options.model, this.provider) ? undefined : options.result_schema;
        return formatOpenAILikeMultimodalPrompt(segments, { ...options, result_schema: resultSchema });
    }

    /** @internal Resolve the model identifier sent to the Responses transport. */
    getResponsesRequestModel(model: string): string {
        return model;
    }

    extractDataFromResponse(_options: ExecutionOptions, result: OpenAI.Responses.Response): Completion {
        const tokenInfo = mapUsage(result.usage);

        const tools = collectTools(result.output);
        // Collect all parts in order (text and images)
        const allResults = extractCompletionResults(result.output);

        if (allResults.length === 0 && !tools) {
            this.logger.error({ result }, '[OpenAI] Response is not valid');
            throw new Error('Response is not valid: no data');
        }

        return {
            result: allResults,
            token_usage: tokenInfo,
            finish_reason: responseFinishReason(result, tools),
            tool_use: tools,
        };
    }

    requestTextCompletionStream(
        prompt: ResponseInputItem[],
        options: ExecutionOptions,
    ): Promise<AsyncIterable<CompletionChunkObject>> {
        return this.responsesProtocol.requestTextCompletionStream(this, prompt, options);
    }

    requestTextCompletion(prompt: ResponseInputItem[], options: ExecutionOptions): Promise<Completion> {
        return this.responsesProtocol.requestTextCompletion(this, prompt, options);
    }

    protected canStream(_options: ExecutionOptions): Promise<boolean> {
        // Image generation models don't support streaming
        if (
            _options.model.includes('dall-e') ||
            _options.model.includes('gpt-image') ||
            _options.model.includes('chatgpt-image')
        ) {
            return Promise.resolve(false);
        }

        if (_options.model.includes('o1') && !(_options.model.includes('mini') || _options.model.includes('preview'))) {
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
        options: ExecutionOptions,
    ): ResponseInputItem[] | undefined {
        // Build assistant message from accumulated CompletionResult[]
        const completionResults = result as CompletionResult[];

        const textContent = completionResultsToText(completionResults);

        // Start with the conversation from options or the prompt
        let conversation = updateConversation(options.conversation, prompt);

        // Add assistant message as EasyInputMessage
        if (textContent) {
            const assistantMessage: EasyInputMessage = {
                type: 'message',
                role: 'assistant',
                content: textContent,
            };
            conversation = updateConversation(conversation, [assistantMessage]);
        }

        // Add function calls as separate items (Response API format)
        if (toolUse && toolUse.length > 0) {
            const functionCalls: OpenAI.Responses.ResponseFunctionToolCall[] = (toolUse as ToolUse[]).map((t) => ({
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
            textMaxTokens: options.stripTextMaxTokens,
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
        if (options.model.includes('gpt')) {
            return super.createTrainingPrompt(options);
        } else {
            // babbage, davinci not yet implemented
            throw new Error(`Unsupported model for training: ${options.model}`);
        }
    }

    async startTraining(dataset: DataSource, options: TrainingOptions): Promise<TrainingJob> {
        const url = await dataset.getURL();
        const file = await this.service.files.create({
            file: await fetch(url),
            purpose: 'fine-tune',
        });

        const job = await this.service.fineTuning.jobs.create({
            training_file: file.id,
            model: options.model,
            hyperparameters: options.params,
        });

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
        } catch {
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
        const wordBlacklist = [
            'embed',
            'whisper',
            'transcribe',
            'audio',
            'moderation',
            'tts',
            'realtime',
            'babbage',
            'davinci',
            'codex',
            'o1-pro',
            'computer-use',
            'sora',
        ];

        //OpenAI has very little information, filtering based on name.
        result = result.filter((m) => {
            return !wordBlacklist.some((word) => m.id.includes(word));
        });

        const models = filter ? result.filter(filter) : result;
        const aiModels = models
            .map((m) => {
                const modelCapability = getModelCapabilities(m.id, this.provider);
                let owner = m.owned_by;
                if (owner === 'system') {
                    owner = 'openai';
                }

                // Determine model type based on capabilities
                let modelType = ModelType.Text;
                if (m.id.includes('dall-e') || m.id.includes('gpt-image')) {
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
            })
            .sort((a, b) => a.id.localeCompare(b.id));

        return aiModels;
    }

    async generateEmbeddings(options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        const normalized = normalizeEmbeddingsOptions(options);
        const model = normalized.model ?? OPENAI_DEFAULT_EMBEDDING_MODEL;

        const texts: string[] = normalized.inputs.map((input) => {
            if (input.type !== 'text') {
                throw new Error(
                    `Provider 'openai' does not support '${input.type}' embeddings; only 'text' is supported.`,
                );
            }
            return input.text;
        });

        try {
            const res = await this.service.embeddings.create({
                input: texts,
                model,
                ...(normalized.dimensions ? { dimensions: normalized.dimensions } : {}),
                encoding_format: 'float',
            });

            // OpenAI does not guarantee data is returned in the same order as the input,
            // but does return a stable `index` per item. Sort by index to align with inputs.
            const ordered = [...res.data].sort((a, b) => a.index - b.index);
            const items = ordered.map((entry): EmbeddingResultItem => {
                if (!entry.embedding || entry.embedding.length === 0) {
                    throw new Error(`OpenAI embedding empty for input index ${entry.index}`);
                }
                return { outputs: [{ values: entry.embedding, modality: 'text' }] };
            });

            const usage = res.usage
                ? { input_tokens: res.usage.prompt_tokens, input_text_tokens: res.usage.prompt_tokens }
                : undefined;

            return { model, results: items, usage };
        } catch (error) {
            if (LlumiverseError.isLlumiverseError(error)) throw error;
            if (error instanceof Error && !hasNumericStatus(error)) throw error;
            throw this.formatLlumiverseError(error, {
                provider: this.provider,
                model,
                operation: 'execute',
            });
        }
    }

    imageModels = ['dall-e', 'gpt-image', 'chatgpt-image'];

    /**
     * Determine if a model is specifically an image generation model (not conversational image model)
     */
    isImageModel(model: string): boolean {
        // DALL-E models are standalone image generation
        // gpt-image models can generate images in conversations, not standalone
        return this.imageModels.some((imageModel) => model.includes(imageModel));
    }

    /**
     * Request image generation from standalone Images API
     * Supports: DALL-E 2, DALL-E 3, GPT-image models (for edit/variation)
     */
    async requestImageGeneration(prompt: ResponseInputItem[], options: ExecutionOptions): Promise<Completion> {
        this.logger.debug(`[${this.provider}] Generating image with model ${options.model}`);

        const model_options = options.model_options as OpenAiDalleOptions | OpenAiGptImageOptions | undefined;

        // Extract prompt text from ResponseInputItem[]
        let promptText = '';
        for (const item of prompt) {
            if ('content' in item && typeof item.content === 'string') {
                promptText += `${item.content}\\n`;
            } else if ('content' in item && Array.isArray(item.content)) {
                // Extract text from content array
                for (const part of item.content) {
                    if ('type' in part && part.type === 'input_text' && 'text' in part) {
                        promptText += `${part.text}\\n`;
                    }
                }
            }
        }
        promptText = promptText.trim();

        try {
            const generateParams: OpenAI.Images.ImageGenerateParamsNonStreaming = {
                model: options.model,
                prompt: promptText,
                size: model_options?.size || '1024x1024',
            };

            // Add DALL-E specific options
            if (options.model.includes('dall-e') || model_options?._option_id === 'openai-dalle') {
                const dalleOptions = model_options as OpenAiDalleOptions | undefined;
                generateParams.n = dalleOptions?.n || 1;
                generateParams.response_format = dalleOptions?.response_format || 'b64_json';

                if (options.model.includes('dall-e-3')) {
                    generateParams.quality = dalleOptions?.image_quality || 'standard';
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
                        type: 'image',
                        value: imageValue,
                    });
                }
            }

            return {
                result: results,
            };
        } catch (error: unknown) {
            this.logger.error({ error }, `[${this.provider}] Image generation failed`);
            const generationError = error instanceof Error ? error : new Error(String(error));
            const errorCode =
                (error as { code?: unknown })?.code === 'content_policy_violation'
                    ? 'content_policy_violation'
                    : 'validation_error';
            return {
                result: [],
                error: {
                    message: generationError.message,
                    code: errorCode,
                },
            };
        }
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
        details = job.error
            ? `${job.error.code} - ${job.error.message} ${job.error.param ? ` [${job.error.param}]` : ''}`
            : 'error';
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
        details,
    };
}

function mapUsage(usage?: OpenAIUsageWithProviderDetails | null): ExecutionTokenUsage | undefined {
    if (!usage) {
        return undefined;
    }
    const cachedTokens =
        usage.input_tokens_details?.cached_tokens ?? usage.prompt_tokens_details?.cached_tokens ?? usage.cached_tokens;
    const cacheWriteTokens =
        usage.input_tokens_details?.cache_write_tokens ??
        usage.prompt_tokens_details?.cache_write_tokens ??
        usage.cache_write_tokens;
    return {
        prompt: usage.input_tokens,
        result: usage.output_tokens,
        total: usage.total_tokens,
        prompt_cached: cachedTokens ?? undefined,
        prompt_cache_write: cacheWriteTokens ?? undefined,
        prompt_new: usage.input_tokens - (cachedTokens ?? 0),
    };
}

function completionResultsToText(completionResults: CompletionResult[] | undefined): string {
    if (!completionResults) {
        return '';
    }
    return completionResults
        .map((r) => {
            switch (r.type) {
                case 'text':
                    return r.value;
                case 'json':
                    return typeof r.value === 'string' ? r.value : JSON.stringify(r.value);
                case 'image':
                    // Skip images in conversation - they're in the result
                    return '';
                default: {
                    const _exhaustive: never = r;
                    return String(_exhaustive);
                }
            }
        })
        .join('');
}

function buildResponseTextConfig(
    schema: JSONSchema | undefined,
    strict: boolean,
    verbosity: OpenAIRequestOptions['verbosity'] | undefined,
): OpenAI.Responses.ResponseTextConfig | undefined {
    if (!schema && !verbosity) {
        return undefined;
    }
    return {
        ...(schema
            ? {
                  format: {
                      type: 'json_schema' as const,
                      name: 'format_output',
                      schema,
                      strict,
                  },
              }
            : {}),
        ...(verbosity ? { verbosity } : {}),
    };
}

function createAssistantMessageFromCompletion(completion: Completion): ResponseInputItem[] {
    const textContent = completionResultsToText(completion.result);
    const result: ResponseInputItem[] = [];

    // Add assistant text message if present
    if (textContent) {
        const assistantMessage: EasyInputMessage = {
            type: 'message',
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
                arguments: typeof t.tool_input === 'string' ? t.tool_input : JSON.stringify(t.tool_input ?? {}),
            };
            result.push(functionCall);
        }
    }

    return result;
}

export function mapResponseStream(
    stream: AsyncIterable<OpenAI.Responses.ResponseStreamEvent>,
): AsyncIterable<CompletionChunkObject> {
    const toolCallMetadata = new Map<string, { syntheticId: string; callId?: string; name?: string }>();

    return {
        async *[Symbol.asyncIterator]() {
            let hasTextDeltas = false;
            let refusalText = '';
            for await (const event of stream) {
                if (event.type === 'response.output_item.added' && event.item.type === 'function_call') {
                    const syntheticId = `tool_${event.output_index}`;
                    const callId = event.item.call_id ?? event.item.id;
                    const metadata = { syntheticId, callId, name: event.item.name };
                    if (event.item.id) {
                        toolCallMetadata.set(event.item.id, metadata);
                    }
                    if (event.item.call_id) {
                        toolCallMetadata.set(event.item.call_id, metadata);
                    }
                    const toolUse: ToolUse<unknown> & { _actual_id?: string } = {
                        id: syntheticId,
                        _actual_id: callId,
                        tool_name: event.item.name,
                        tool_input: '',
                    };
                    yield {
                        result: [],
                        tool_use: [toolUse],
                    } satisfies CompletionChunkObject;
                } else if (event.type === 'response.function_call_arguments.delta') {
                    const metadata = toolCallMetadata.get(event.item_id);
                    const syntheticId = metadata?.syntheticId ?? `tool_${event.output_index}`;
                    const callId = metadata?.callId ?? event.item_id;
                    const toolUse: ToolUse<unknown> & { _actual_id?: string } = {
                        id: syntheticId,
                        _actual_id: callId,
                        tool_name: metadata?.name ?? '',
                        tool_input: event.delta,
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
                        toolCallMetadata.set(event.item_id, { syntheticId, callId: metadata?.callId, name: tool_name });
                    }
                } else if (event.type === 'response.output_text.delta') {
                    hasTextDeltas = true;
                    yield {
                        result: textToCompletionResult(event.delta),
                    } satisfies CompletionChunkObject;
                } else if (event.type === 'response.output_text.done') {
                    // Fallback: some models (e.g. gpt-5 with json_schema structured output) buffer
                    // the entire output and never emit delta events — only the done event with the
                    // full text. Emit it here only when no deltas arrived to avoid duplication.
                    if (!hasTextDeltas && event.text) {
                        yield {
                            result: textToCompletionResult(event.text),
                        } satisfies CompletionChunkObject;
                    }
                } else if (event.type === 'response.refusal.delta') {
                    refusalText += event.delta;
                } else if (event.type === 'response.refusal.done') {
                    throw new Error(`[OpenAI] Model refused: ${event.refusal || refusalText}`);
                } else if ((event as { type: string }).type === 'response.error') {
                    const errEvent = event as unknown as { message: string; code?: string | null };
                    throw new Error(
                        `[OpenAI Responses API] ${errEvent.message}${errEvent.code ? ` (${errEvent.code})` : ''}`,
                    );
                } else if (
                    event.type === 'response.completed' ||
                    event.type === 'response.incomplete' ||
                    event.type === 'response.failed'
                ) {
                    const finalTools = collectTools(event.response.output);
                    yield {
                        result: [],
                        finish_reason: responseFinishReason(event.response, finalTools),
                        token_usage: mapUsage(event.response.usage),
                    } satisfies CompletionChunkObject;
                }
            }
        },
    };
}

function insert_image_detail(items: ResponseInputItem[], detail_level: string): ResponseInputItem[] {
    if (detail_level === 'auto' || detail_level === 'low' || detail_level === 'high') {
        for (const item of items) {
            // Check if it's an EasyInputMessage or Message with content array
            if ('role' in item && 'content' in item && item.role !== 'assistant') {
                const content = (item as EasyInputMessage).content;
                if (Array.isArray(content)) {
                    for (const part of content) {
                        if (typeof part === 'object' && part.type === 'input_image') {
                            (part as MutableInputImagePart).detail = detail_level;
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
    if (model.includes('o1') || model.includes('o3')) {
        if (model.includes('o1-mini') || model.includes('o1-preview')) {
            //o1-mini and o1-preview support neither system nor developer
            for (const item of items) {
                if ('role' in item && (item as EasyInputMessage).role === 'system') {
                    (item as MutableRoleItem).role = 'user';
                }
            }
        } else {
            //Models newer than o1 use developer role
            for (const item of items) {
                if ('role' in item && (item as EasyInputMessage).role === 'system') {
                    (item as MutableRoleItem).role = 'developer';
                }
            }
        }
    }
    return items;
}

//Structured output support is typically aligned with tool use support
//Not true for realtime models, which do not support structured output, but do support tool use.
function supportsSchema(model: string, provider: string | Providers): boolean {
    const realtimeModel = model.includes('realtime');
    if (realtimeModel) {
        return false;
    }
    return supportsToolUse(model, provider);
}

/**
 * Converts function_call and function_call_output items to text messages in OpenAI conversation.
 * Preserves tool call information while removing structured items that require
 * tools to be defined in the API request.
 */
export function convertOpenAIFunctionItemsToText(items: ResponseInputItem[]): ResponseInputItem[] {
    const hasFunctionItems = items.some((item) => {
        const type = (item as OpenAIFunctionItem).type;
        return type === 'function_call' || type === 'function_call_output';
    });
    if (!hasFunctionItems) return items;

    return items.map((item) => {
        const typed = item as OpenAIFunctionItem;
        if (typed.type === 'function_call') {
            const argsStr = typed.arguments || '';
            const truncated = argsStr.length > 500 ? `${argsStr.substring(0, 500)}...` : argsStr;
            return {
                role: 'assistant' as const,
                content: `[Tool call: ${typed.name}(${truncated})]`,
            };
        }
        if (typed.type === 'function_call_output') {
            const output = typed.output || 'No output';
            const truncated = output.length > 500 ? `${output.substring(0, 500)}...` : output;
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
    let parsedSchema: JSONSchema | undefined;
    let strictMode = false;
    if (toolDef.input_schema) {
        // TODO: type assertion here is not safe, does not work with satisfies.
        const formattedSchema = formatOpenAISchema(toolDef.input_schema as JSONSchema);
        parsedSchema = formattedSchema.schema;
        strictMode = formattedSchema.strict;
    }

    return {
        type: 'function',
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
        return unwrapped ?? ((conversation as ResponseInputItem[]) || []);
    }
    if (!conversation) {
        return items;
    }
    // Unwrap array if wrapped, otherwise treat as array
    const unwrapped = unwrapConversationArray<ResponseInputItem>(conversation);
    const convArray = unwrapped ?? (conversation as ResponseInputItem[]);
    return [...convArray, ...items];
}

export function collectTools(output?: OpenAI.Responses.ResponseOutputItem[]): ToolUse<unknown>[] | undefined {
    if (!output) {
        return undefined;
    }

    const tools: ToolUse<unknown>[] = [];
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
                        type: 'text',
                        value: part.text,
                    });
                }
            }
        } else if (item.type === 'image_generation_call' && 'result' in item && item.result) {
            // GPT-image models return base64 encoded images in result field
            const base64Data = item.result;
            // Format as data URL for consistency with other image outputs
            const imageUrl = base64Data.startsWith('data:') ? base64Data : `data:image/png;base64,${base64Data}`;
            results.push({
                type: 'image',
                value: imageUrl,
            });
        }
    }
    return results;
}

function responseFinishReason(
    response: OpenAI.Responses.Response,
    tools?: ToolUse<unknown>[] | undefined,
): string | undefined {
    if (tools && tools.length > 0) {
        return 'tool_use';
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

/**
 * Drop function_call_output items whose call_id has no matching function_call
 * item anywhere in the input. Mirror of {@link fixOrphanedToolUse}: that
 * synthesizes outputs for calls left unanswered (e.g. a cancelled run); this
 * removes outputs left dangling after their function_call was dropped (e.g. by
 * conversation compaction/trimming). The OpenAI Responses API requires every
 * function_call_output to reference a prior function_call's call_id.
 */
export function fixOrphanedToolResults(items: ResponseInputItem[]): ResponseInputItem[] {
    if (items.length === 0) return items;
    const callIds = new Set<string>();
    for (const item of items) {
        if ('type' in item && item.type === 'function_call') {
            callIds.add(item.call_id);
        }
    }
    return items.filter((item) => {
        if ('type' in item && item.type === 'function_call_output') {
            return callIds.has(item.call_id);
        }
        return true;
    });
}

function safeJsonParse(value: unknown): unknown {
    if (typeof value !== 'string') {
        return value;
    }
    try {
        return JSON.parse(value);
    } catch {
        return value;
    }
}
