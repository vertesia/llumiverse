import {
    type AIModel,
    type Completion,
    type CompletionChunkObject,
    type DriverOptions,
    type EmbeddingsOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    LlumiverseError,
    type LlumiverseErrorContext,
    type ModelSearchPayload,
    ModelType,
    type PromptOptions,
    type PromptSegment,
    Providers,
} from '@llumiverse/core';
import { AbstractDriver } from '@llumiverse/core/driver';

export interface CloudflareAIGatewayDriverOptions extends DriverOptions {
    /** Cloudflare API token or AI Gateway token. */
    apiKey: string;
    /** Cloudflare account id. */
    accountId: string;
    /** AI Gateway name/id. Defaults to Cloudflare's `default` gateway. */
    gateway?: string;
    /**
     * Override the Workers AI provider endpoint.
     * Defaults to `https://gateway.ai.cloudflare.com/v1/{accountId}/{gateway}/workers-ai/v1`.
     */
    endpoint?: string;
    default_headers?: Record<string, string>;
}

interface CloudflareChatMessage {
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: string;
}

interface CloudflareChatCompletion {
    choices?: Array<{
        finish_reason?: string;
        message?: {
            content?: string;
            reasoning_content?: string;
        };
    }>;
    usage?: {
        prompt_tokens?: number;
        completion_tokens?: number;
        total_tokens?: number;
    };
}

interface TextFallbackOptions {
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
}

export class CloudflareAIGatewayDriver extends AbstractDriver<
    CloudflareAIGatewayDriverOptions,
    CloudflareChatMessage[]
> {
    static readonly PROVIDER = Providers.cloudflare_ai_gateway;
    readonly provider = Providers.cloudflare_ai_gateway;
    private readonly endpoint: string;

    constructor(options: CloudflareAIGatewayDriverOptions) {
        super(options);
        if (!options.apiKey) {
            throw new Error('apiKey is required');
        }
        if (!options.accountId) {
            throw new Error('accountId is required');
        }
        const gateway = options.gateway ?? 'default';
        this.endpoint =
            options.endpoint ?? `https://gateway.ai.cloudflare.com/v1/${options.accountId}/${gateway}/workers-ai/v1`;
    }

    protected async formatPrompt(segments: PromptSegment[], _opts: PromptOptions): Promise<CloudflareChatMessage[]> {
        return segments.map((segment) => ({
            role: toCloudflareRole(segment.role),
            content: String(segment.content),
        }));
    }

    async requestTextCompletion(prompt: CloudflareChatMessage[], options: ExecutionOptions): Promise<Completion> {
        const modelOptions = options.model_options as TextFallbackOptions | undefined;
        const response = await this.getDriverFetch()(`${trimTrailingSlash(this.endpoint)}/chat/completions`, {
            method: 'POST',
            headers: {
                authorization: `Bearer ${this.options.apiKey}`,
                'content-type': 'application/json',
                ...this.options.default_headers,
            },
            body: JSON.stringify({
                model: options.model,
                messages: prompt,
                max_tokens: modelOptions?.max_tokens,
                temperature: modelOptions?.temperature,
                top_p: modelOptions?.top_p,
                top_k: modelOptions?.top_k,
            }),
        });

        const text = await response.text();
        if (!response.ok) {
            throw new CloudflareAIGatewayError(response.status, safeCloudflareError(text));
        }

        const result = JSON.parse(text) as CloudflareChatCompletion;
        const choice = result.choices?.[0];
        const content = choice?.message?.content || choice?.message?.reasoning_content || '';
        return {
            result: content ? [{ type: 'text', value: content }] : [],
            finish_reason: choice?.finish_reason,
            token_usage: {
                prompt: result.usage?.prompt_tokens,
                result: result.usage?.completion_tokens,
                total: result.usage?.total_tokens,
            },
            original_response: options.include_original_response ? result : undefined,
        };
    }

    async requestTextCompletionStream(
        _prompt: CloudflareChatMessage[],
        _options: ExecutionOptions,
    ): Promise<AsyncIterable<CompletionChunkObject>> {
        throw new Error('Cloudflare AI Gateway streaming is not implemented yet.');
    }

    protected canStream(_options: ExecutionOptions): Promise<boolean> {
        return Promise.resolve(false);
    }

    async listModels(_params?: ModelSearchPayload): Promise<AIModel[]> {
        return [
            {
                id: '@cf/zai-org/glm-5.2',
                name: 'GLM 5.2',
                provider: this.provider,
                type: ModelType.Text,
                can_stream: false,
            },
        ];
    }

    async validateConnection(): Promise<boolean> {
        return Boolean(this.options.apiKey && this.options.accountId);
    }

    async generateEmbeddings(_options: EmbeddingsOptions): Promise<EmbeddingsResult> {
        throw new Error('Cloudflare AI Gateway embeddings are not implemented yet.');
    }

    public formatLlumiverseError(error: unknown, context: LlumiverseErrorContext): LlumiverseError {
        if (error instanceof CloudflareAIGatewayError) {
            return new LlumiverseError(
                `[${this.provider}] ${error.message}`,
                error.status >= 500 || error.status === 429,
                context,
                undefined,
                error.status,
                'CloudflareAIGatewayError',
            );
        }
        return super.formatLlumiverseError(error, context);
    }
}

class CloudflareAIGatewayError extends Error {
    constructor(
        readonly status: number,
        message: string,
    ) {
        super(`Cloudflare AI Gateway request failed with ${status}: ${message}`);
        this.name = 'CloudflareAIGatewayError';
    }
}

function trimTrailingSlash(value: string): string {
    return value.replace(/\/+$/u, '');
}

function toCloudflareRole(role: PromptSegment['role']): CloudflareChatMessage['role'] {
    switch (role) {
        case 'system':
        case 'user':
        case 'assistant':
        case 'tool':
            return role;
        default:
            return 'user';
    }
}

function safeCloudflareError(text: string): string {
    try {
        const payload = JSON.parse(text) as { message?: unknown; error?: unknown; errors?: unknown };
        return String(payload.message ?? payload.error ?? payload.errors ?? text).slice(0, 500);
    } catch {
        return text.slice(0, 500);
    }
}
