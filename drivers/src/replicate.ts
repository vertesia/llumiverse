import {
    AIModel,
    AbstractDriver,
    Completion,
    CompletionChunk,
    DataSource,
    DriverOptions,
    EmbeddingsResult,
    ModelSearchPayload,
    ExecutionOptions,
    TrainingJob,
    TrainingJobStatus,
    TrainingOptions,
    TextFallbackOptions,
} from "@llumiverse/core";
import { EventStream } from "@llumiverse/core/async";
import EventSource from "eventsource";
import Replicate, { Prediction } from "replicate";

let cachedTrainableModels: AIModel[] | undefined;
let cachedTrainableModelsTimestamp: number = 0;

const supportFineTunning = new Set([
    "meta/llama-2-70b-chat",
    "meta/llama-2-13b-chat",
    "meta/llama-2-7b-chat",
    "meta/llama-2-7b",
    "meta/llama-2-70b",
    "meta/llama-2-13b",
    "mistralai/mistral-7b-v0.1"
]);

export interface ReplicateDriverOptions extends DriverOptions {
    apiKey: string;
}

export class ReplicateDriver extends AbstractDriver<DriverOptions, string> {
    static PROVIDER = "replicate";
    provider = ReplicateDriver.PROVIDER;
    service: Replicate;

    static parseModelId(modelId: string) {
        const [owner, modelPart] = modelId.split("/");
        const i = modelPart.indexOf(':');
        if (i === -1) {
            throw new Error("Invalid model id. Expected format: owner/model:version");
        }
        return {
            owner, model: modelPart.slice(0, i), version: modelPart.slice(i + 1)
        }
    }

    constructor(options: ReplicateDriverOptions) {
        super(options);
        this.service = new Replicate({
            auth: options.apiKey,
        });
    }

    extractDataFromResponse(response: Prediction): Completion {
        const text = response.output.join("");
        return {
            result: text,
        };
    }

    async requestTextCompletionStream(prompt: string, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunk>> {
        if (options.model_options?._option_id !== "text-fallback") {
            this.logger.warn("Invalid model options", options.model_options);
        }
        options.model_options = options.model_options as TextFallbackOptions;
        
        const model = ReplicateDriver.parseModelId(options.model);
        const predictionData = {
            input: {
                prompt: prompt,
                max_new_tokens: options.model_options?.max_tokens,
                temperature: options.model_options?.temperature,
            },
            version: model.version,
            stream: true, //streaming described here https://replicate.com/blog/streaming
        };

        const prediction =
            await this.service.predictions.create(predictionData);

        const stream = new EventStream<CompletionChunk>();

        const source = new EventSource(prediction.urls.stream!);
        source.addEventListener("output", (e: any) => {
            stream.push(e.data);
        });
        source.addEventListener("error", (e: any) => {
            let error: any;
            try {
                error = JSON.parse(e.data);
            } catch (error) {
                error = JSON.stringify(e);
            }
            this.logger?.error(e, error, "Error in SSE stream");
        });
        source.addEventListener("done", () => {
            try {
                stream.close(""); // not using e.data which is {}
            } finally {
                source.close();
            }
        });
        return stream;
    }

    async requestTextCompletion(prompt: string, options: ExecutionOptions) {
        if (options.model_options?._option_id !== "text-fallback") {
            this.logger.warn("Invalid model options", options.model_options);
        }
        options.model_options = options.model_options as TextFallbackOptions;
        const model = ReplicateDriver.parseModelId(options.model);
        const predictionData = {
            input: {
                prompt: prompt,
                max_new_tokens: options.model_options?.max_tokens,
                temperature: options.model_options?.temperature,
            },
            version: model.version,
            //TODO stream
            //stream: stream,     //streaming described here https://replicate.com/blog/streaming
        };

        const prediction =
            await this.service.predictions.create(predictionData);

        //TODO stream
        //if we're streaming, return right away for the stream handler to handle
        //        if (stream) return prediction;

        //not streaming, wait for the result
        const res = await this.service.wait(prediction, {});

        const text = res.output.join("");
        return {
            result: text,
            original_response: options.include_original_response ? res : undefined,
        };
    }

    async startTraining(dataset: DataSource, options: TrainingOptions): Promise<TrainingJob> {
        if (options.name.indexOf("/") === -1) {
            throw new Error("Invalid target model name. Expected format: owner/model");
        }
        const { owner, model, version } = ReplicateDriver.parseModelId(options.model);
        const job = await this.service.trainings.create(owner, model, version, {
            destination: options.name as any,
            input: {
                train_data: await dataset.getURL(),
            },
        })
        return jobInfo(job, options.name);
    }

    /**
     * This method is not returning a consistent TrainingJob like the one returned by startTraining
     * Instead of returning the full model name `owner/model:version` it returns only the version `version
     * @param jobId
     * @returns
     */
    async cancelTraining(jobId: string): Promise<TrainingJob> {
        const job = await this.service.trainings.cancel(jobId);
        return jobInfo(job);
    }

    /**
     * This method is not returning a consistent TrainingJob like the one returned by startTraining
     * Instead of returning the full model name `owner/model:version` it returns only the version `version
     * @param jobId
     * @returns
     */
    async getTrainingJob(jobId: string): Promise<TrainingJob> {
        const job = await this.service.trainings.get(jobId);
        return jobInfo(job);
    }

    // ========= management API =============

    async validateConnection(): Promise<boolean> {
        try {
            await this.service.predictions.list();
            return true;
        } catch (error) {
            return false;
        }
    }

    async _listTrainableModels(): Promise<AIModel[]> {
        const promises = Array.from(supportFineTunning).map(id => {
            const [owner, model] = id.split('/');
            return this.service.models.get(owner, model)
        });
        const results = await Promise.all(promises);
        return results.filter(m => !!m.latest_version).map(m => {
            const fullName = m.owner + '/' + m.name;
            const v = m.latest_version!;
            return {
                id: fullName + ':' + v.id,
                name:
                    fullName + "@" + v.cog_version + ":" + v.id.slice(0, 6),
                provider: this.provider,
                owner: m.owner,
                description: m.description,
            } as AIModel;
        });
    }

    async listTrainableModels(): Promise<AIModel[]> {
        if (!cachedTrainableModels || Date.now() > cachedTrainableModelsTimestamp + 12 * 3600 * 1000) { // 12 hours
            cachedTrainableModels = await this._listTrainableModels();
            cachedTrainableModelsTimestamp = Date.now();
        }
        return cachedTrainableModels;
    }

    async listModels(params: ModelSearchPayload = { text: '' }): Promise<AIModel[]> {
        if (!params.text) {
            return this.listTrainableModels();
        }
        const [owner, model] = params.text.split("/");
        if (!owner || !model) {
            throw new Error("Invalid model name. Expected format: owner/model");
        }

        return this.listModelVersions(owner, model);
    }

    async listModelVersions(owner: string, model: string): Promise<AIModel[]> {
        const [rModel, versions] = await Promise.all([
            this.service.models.get(owner, model),
            this.service.models.versions.list(owner, model),
        ]);

        if (!rModel || !versions || versions.length === 0) {
            throw new Error("Model not found or no versions avaialble");
        }

        const models: AIModel[] = (versions as any).results.map((v: any) => {
            const fullName = rModel.owner + '/' + rModel.name;
            return {
                id: fullName + ':' + v.id,
                name:
                    fullName + "@" + v.cog_version + ":" + v.id.slice(0, 6),
                provider: this.provider,
                owner: rModel.owner,
                description: rModel.description,
                canTrain: supportFineTunning.has(fullName),
            } as AIModel;
        });

        //set latest version
        //const idx = models.findIndex(m => m.id === rModel.latest_version?.id);
        //models[idx].name = rModel.name + "@latest"

        return models;
    }

    async searchModels(params: ModelSearchPayload): Promise<AIModel[]> {
        const res = await this.service.request("models/search", {
            params: {
                query: params.text,
            },
        });

        const rModels = ((await res.json()) as any).models;

        const models: AIModel[] = rModels.map((v: any) => {
            return {
                id: v.name,
                name: v.name,
                provider: this.provider,
                owner: v.username,
                description: v.description,
                has_versions: true,
            };
        });

        return models;
    }

    async generateEmbeddings(): Promise<EmbeddingsResult> {
        throw new Error("Method not implemented.");
    }

}

function jobInfo(job: Prediction, modelName?: string): TrainingJob {
    // 'starting' | 'processing' | 'succeeded' | 'failed' | 'canceled'
    const jobStatus = job.status;
    let details: string | undefined;
    let status = TrainingJobStatus.running;
    if (jobStatus === 'succeeded') {
        status = TrainingJobStatus.succeeded;
    } else if (jobStatus === 'failed') {
        status = TrainingJobStatus.failed;
        const error = job.error;
        if (typeof error === 'string') {
            details = error;
        } else {
            const parts = [];
            if (error.code) {
                parts.push(error.code + ' - ');
            }
            if (error.message) {
                parts.push(error.message);
            }
            if (parts.length) {
                details = parts.join(' ');
            } else {
                details = JSON.stringify(error);
            }
        }
        details = job.error ? `${job.error.code} - ${job.error.message} ${job.error.param ? " [" + job.error.param + "]" : ""}` : "error";
    } else if (jobStatus === 'canceled') {
        status = TrainingJobStatus.cancelled;
    } else {
        status = TrainingJobStatus.running;
        details = job.status;
    }

    return {
        id: job.id,
        status,
        details,
        model: modelName ? modelName + ':' + job.version : job.version
    } as TrainingJob;

}