import { InferenceClient, type TextGenerationStreamOutput } from '@huggingface/inference';
import {
    type AIModel,
    AIModelStatus,
    type CanonicalExecutionEventStream,
    type CanonicalExecutionInputOptions,
    type CanonicalExecutionResponse,
    type CanonicalStreamOpenOptions,
    type CompletionChunkObject,
    type DriverCompletionStream,
    type EmbeddingsResult,
    type ExecutionOptions,
    type PromptSegment,
    type TextFallbackOptions,
} from '@llumiverse/core';
import { transformAsyncIterator } from '@llumiverse/core/async';
import { AbstractDriver } from '@llumiverse/core/driver';
import { FetchClient } from '@vertesia/api-fetch-client';
import type { HuggingFaceIEDriverOptions } from './driver-options.js';
import { executeHuggingFaceCanonical, streamHuggingFaceCanonicalEvents } from './huggingface_ie.canonical.js';

export type { HuggingFaceIEDriverOptions } from './driver-options.js';

export class HuggingFaceIEDriver extends AbstractDriver<HuggingFaceIEDriverOptions, string> {
    static PROVIDER = 'huggingface_ie';
    provider = HuggingFaceIEDriver.PROVIDER;
    service: FetchClient;
    private readonly executors = new Map<string, Promise<{ executor: InferenceClient; url: string }>>();

    constructor(options: HuggingFaceIEDriverOptions) {
        super(options);
        if (!options.endpoint_url) {
            throw new Error(`Endpoint URL is required for ${this.provider}`);
        }
        this.service = new FetchClient(this.options.endpoint_url, this.getDriverFetch());
        this.service.headers.Authorization = `Bearer ${this.options.apiKey}`;
    }

    async getModelURLEndpoint(modelId: string): Promise<{ url: string; status: string }> {
        const res = (await this.service.get(`/${modelId}`)) as HuggingFaceIEModel;
        return {
            url: res.status.url,
            status: getStatus(res),
        };
    }

    async getExecutorTarget(model: string): Promise<{ executor: InferenceClient; url: string }> {
        let pending = this.executors.get(model);
        if (pending === undefined) {
            pending = (async () => {
                const endpoint = await this.getModelURLEndpoint(model);
                if (!endpoint.url) throw new Error(`Endpoint URL not found for model ${model}`);
                if (endpoint.status !== AIModelStatus.Available)
                    throw new Error(`Endpoint ${model} is not running - current status: ${endpoint.status}`);

                // Use the new InferenceClient and bind it to the endpoint URL
                return {
                    executor: new InferenceClient(this.options.apiKey, { fetch: this.getDriverFetch() }).endpoint(
                        endpoint.url,
                    ),
                    url: endpoint.url,
                };
            })();
            this.executors.set(model, pending);
            const registered = pending;
            void registered.catch(() => {
                if (this.executors.get(model) === registered) this.executors.delete(model);
            });
        }
        return pending;
    }

    async getExecutor(model: string): Promise<InferenceClient> {
        return (await this.getExecutorTarget(model)).executor;
    }

    protected supportsCanonicalConversation(_options: ExecutionOptions): boolean {
        return true;
    }

    private validateCanonicalPromptSegments(segments: PromptSegment[]): void {
        for (const segment of segments) {
            if (segment.files?.length) throw new TypeError('Hugging Face text generation does not support media input');
            if (segment.role === 'tool' || segment.role === 'negative' || segment.role === 'mask') {
                throw new TypeError(`Hugging Face text generation does not support ${segment.role} prompt segments`);
            }
        }
    }

    override executeCanonical(
        segments: PromptSegment[],
        options: CanonicalExecutionInputOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        this.validateCanonicalPromptSegments(segments);
        return super.executeCanonical(segments, options, signal);
    }

    override streamCanonicalEvents(
        segments: PromptSegment[],
        options: CanonicalExecutionInputOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        this.validateCanonicalPromptSegments(segments);
        return super.streamCanonicalEvents(segments, options, signal, open);
    }

    async requestCanonicalTextCompletion(
        prompt: string,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CanonicalExecutionResponse> {
        return executeHuggingFaceCanonical({ driver: this, prompt, options, signal });
    }

    async requestCanonicalTextCompletionEventStream(
        prompt: string,
        options: ExecutionOptions,
        signal: AbortSignal | undefined,
        open: CanonicalStreamOpenOptions,
    ): Promise<CanonicalExecutionEventStream> {
        return streamHuggingFaceCanonicalEvents({ driver: this, prompt, options, signal, open });
    }

    async requestTextCompletionStream(
        prompt: string,
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<DriverCompletionStream> {
        if (options.model_options?._option_id !== undefined && options.model_options?._option_id !== 'text-fallback') {
            this.logger.debug({ options: options.model_options }, 'Unexpected option id');
        }
        options.model_options = options.model_options as TextFallbackOptions;

        const executor = await this.getExecutor(options.model);
        const req = executor.textGenerationStream(
            {
                inputs: prompt,
                parameters: {
                    temperature: options.model_options?.temperature,
                    max_new_tokens: options.model_options?.max_tokens,
                },
            },
            { signal },
        );

        return transformAsyncIterator(req, (val: TextGenerationStreamOutput): CompletionChunkObject => {
            let finish_reason = val.details?.finish_reason as string;
            if (finish_reason === 'eos_token') {
                finish_reason = 'stop';
            }
            return {
                // Special tokens such as </s> are protocol evidence, not display text.
                result: !val.token.special && val.token.text ? [{ type: 'text' as const, value: val.token.text }] : [],
                finish_reason,
                token_usage:
                    val.details?.generated_tokens === undefined ? undefined : { result: val.details.generated_tokens },
            };
        });
    }

    async requestTextCompletion(prompt: string, options: ExecutionOptions, signal?: AbortSignal) {
        if (options.model_options?._option_id !== undefined && options.model_options?._option_id !== 'text-fallback') {
            this.logger.debug({ options: options.model_options }, 'Unexpected option id');
        }
        options.model_options = options.model_options as TextFallbackOptions;

        const executor = await this.getExecutor(options.model);
        const request = {
            inputs: prompt,
            parameters: {
                temperature: options.model_options?.temperature,
                max_new_tokens: options.model_options?.max_tokens,
            },
        };
        const res = signal
            ? await executor.textGeneration(request, { signal })
            : await executor.textGeneration(request);

        let finish_reason = res.details?.finish_reason as string;
        if (finish_reason === 'eos_token') {
            finish_reason = 'stop';
        }
        return {
            result: [{ type: 'text' as const, value: res.generated_text }],
            finish_reason: finish_reason,
            token_usage: {
                result: res.details?.generated_tokens,
            },
            original_response: options.include_original_response ? res : undefined,
        };
    }

    // ============== management API ==============

    async listModels(): Promise<AIModel[]> {
        const res = (await this.service.get('/')) as { items: HuggingFaceIEModel[] };
        const hfModels = res.items;
        if (!hfModels?.length) return [];

        const models: AIModel[] = hfModels.map((model: HuggingFaceIEModel) => ({
            id: model.name,
            name: `${model.name} [${model.model.repository}:${model.model.task}]`,
            provider: this.provider,
            tags: [model.model.task],
            status: getStatus(model),
        }));

        return models;
    }

    async validateConnection(): Promise<boolean> {
        try {
            await this.service.get('/models');
            return true;
        } catch {
            return false;
        }
    }

    async generateEmbeddings(): Promise<EmbeddingsResult> {
        throw new Error('Method not implemented.');
    }
}

//get status from HF status
function getStatus(hfModel: HuggingFaceIEModel): AIModelStatus {
    //[ pending, initializing, updating, updateFailed, running, paused, failed, scaledToZero ]
    switch (hfModel.status.state) {
        case 'running':
            return AIModelStatus.Available;
        case 'initializing':
            return AIModelStatus.Pending;
        case 'updating':
            return AIModelStatus.Pending;
        case 'updateFailed':
            return AIModelStatus.Unavailable;
        case 'paused':
            return AIModelStatus.Stopped;
        case 'failed':
            return AIModelStatus.Unavailable;
        case 'scaledToZero':
            return AIModelStatus.Available;
        default:
            return AIModelStatus.Unknown;
    }
}

interface HuggingFaceIEModel {
    accountId: string;
    compute: {
        accelerator: string;
        instanceSize: string;
        instanceType: string;
        scaling: {
            maxReplica: number;
            minReplica: number;
        };
    };
    model: {
        framework: string;
        image: {
            huggingface: Record<string, never>;
        };
        repository: string;
        revision: string;
        task: string;
    };
    name: string;
    provider: {
        region: string;
        vendor: string;
    };
    status: {
        createdAt: string;
        createdBy: {
            id: string;
            name: string;
        };
        message: string;
        private: {
            serviceName: string;
        };
        readyReplica: number;
        state: string;
        targetReplica: number;
        updatedAt: string;
        updatedBy: {
            id: string;
            name: string;
        };
        url: string;
    };
    type: string;
}

/*
Example of model returned by the API
{
    "items": [
      {
        "accountId": "string",
        "compute": {
          "accelerator": "cpu",
          "instanceSize": "large",
          "instanceType": "c6i",
          "scaling": {
            "maxReplica": 8,
            "minReplica": 2
          }
        },
        "model": {
          "framework": "custom",
          "image": {
            "huggingface": {}
          },
          "repository": "gpt2",
          "revision": "6c0e6080953db56375760c0471a8c5f2929baf11",
          "task": "text-classification"
        },
        "name": "my-endpoint",
        "provider": {
          "region": "us-east-1",
          "vendor": "aws"
        },
        "status": {
          "createdAt": "2023-10-19T05:04:17.305Z",
          "createdBy": {
            "id": "string",
            "name": "string"
          },
          "message": "Endpoint is ready",
          "private": {
            "serviceName": "string"
          },
          "readyReplica": 2,
          "state": "pending",
          "targetReplica": 4,
          "updatedAt": "2023-10-19T05:04:17.305Z",
          "updatedBy": {
            "id": "string",
            "name": "string"
          },
          "url": "https://endpoint-id.region.vendor.endpoints.huggingface.cloud"
        },
        "type": "public"
      }
    ]
  }
*/
