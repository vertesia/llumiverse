import { Bedrock, CreateModelCustomizationJobCommand, FoundationModelSummary, GetModelCustomizationJobCommand, GetModelCustomizationJobCommandOutput, ModelCustomizationJobStatus, StopModelCustomizationJobCommand } from "@aws-sdk/client-bedrock";
import { BedrockRuntime, ConverseRequest, ConverseResponse, ConverseStreamOutput, InferenceConfiguration, Message, SystemContentBlock } from "@aws-sdk/client-bedrock-runtime";
import { S3Client } from "@aws-sdk/client-s3";
import { AbstractDriver, AIModel, Completion, CompletionChunkObject, DataSource, DriverOptions, EmbeddingsOptions, EmbeddingsResult, ExecutionOptions, ExecutionTokenUsage, ImageGeneration, ImageGenExecutionOptions, Modalities, PromptOptions, PromptSegment, TrainingJob, TrainingJobStatus, TrainingOptions } from "@llumiverse/core";
import { transformAsyncIterator } from "@llumiverse/core/async";
import { ClaudeMessagesPrompt, formatClaudePrompt, formatNovaPrompt, NovaMessagesPrompt } from "@llumiverse/core/formatters";
import { AwsCredentialIdentity, Provider } from "@smithy/types";
import mnemonist from "mnemonist";
import { formatNovaImageGenerationPayload, NovaImageGenerationTaskType } from "./nova-image-payload.js";
import { forceUploadFile } from "./s3.js";
import { fortmatConversePrompt } from "./converse.js";

const { LRUCache } = mnemonist;

const supportStreamingCache = new LRUCache<string, boolean>(4096);

export interface ConversePrompt {
    messages: Message[];
    system?: SystemContentBlock[];
}

enum BedrockModelType {
    FoundationModel = "foundation-model",
    InferenceProfile = "inference-profile",
    CustomModel = "custom-model",
    Unknown = "unknown",
};

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
     * Tthe bucket name to be used for training.
     * It will be created oif nto already exixts
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

export type BedrockPrompt = string | ClaudeMessagesPrompt | NovaMessagesPrompt | PromptSegment[] | ConversePrompt | ConverseRequest;

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

    protected async formatPrompt(segments: PromptSegment[], opts: PromptOptions): Promise<BedrockPrompt> {
        return await fortmatConversePrompt(segments, opts.result_schema);
        //TODO move the anthropic test in abstract driver?
        if (opts.model.includes('anthropic')) {
            //TODO: need to type better the types aren't checked properly by TS
            return await formatClaudePrompt(segments, opts.result_schema);
        } else if (opts.model.includes('nova')) {
            //TODO: need to type better the types aren't checked properly by TS
            return await formatNovaPrompt(segments, opts.result_schema);
        } else {
            return await super.formatPrompt(segments, opts) as string;
        }
    }

    static getExtractedExecuton(result: ConverseResponse, _prompt?: BedrockPrompt): CompletionChunkObject {
        return {
            result: result.output,
            token_usage: {
                prompt: result.usage?.inputTokens,
                result: result.usage?.outputTokens,
                total: result.usage?.totalTokens,
             },
            finish_reason: result.stopReason,
        }
    };

    static getExtractedStream(result: ConverseStreamOutput, _prompt?: BedrockPrompt): CompletionChunkObject {
        let output: string = "";
        let stop_reason = "";
        let token_usage: ExecutionTokenUsage | undefined;
        if (result.contentBlockDelta) {
            output = result.contentBlockDelta.delta?.text ?? "";
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
        return {
            result: result,
            token_usage: token_usage,
            finish_reason: stop_reason,
        }
    };

    async requestCompletion(prompt: ConversePrompt, options: ExecutionOptions): Promise<Completion> {

        const payload = this.preparePayload(prompt, options);
        const executor = this.getExecutor();

        const res = await executor.converse({
            ...payload,
        });

        const completion = BedrockDriver.getExtractedExecuton(res, prompt) as Completion;
        if (options.include_original_response) {
            completion.original_response = res;
        }
        return completion;
    }

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

    async requestCompletionStream(prompt: ConversePrompt, options: ExecutionOptions): Promise<AsyncIterable<CompletionChunkObject>> {
        const payload = this.preparePayload(prompt, options);
        const executor = this.getExecutor();
        return executor.converseStream({
            ...payload,
        }).then((res) => {
            const stream = res.stream;

            if (!stream) {
                throw new Error("[Bedrock] Stream not found in response");
            }

            return transformAsyncIterator(stream, (stream: ConverseStreamOutput) => {
                //const segment = JSON.parse(decoder.decode(stream.chunk?.bytes));
                //console.log("Debug Segment for model " + options.model, JSON.stringify(segment));
                return BedrockDriver.getExtractedStream(stream, prompt);
            });

        }).catch((err) => {
            this.logger.error("[Bedrock] Failed to stream", err);
            throw err;
        });
    }



    preparePayload(prompt: ConversePrompt, options: ExecutionOptions) {
        return {
            messages: prompt.messages,
            system: prompt.system,
            modelId: options.model,
            inferenceConfig: {
                maxTokens: options.max_tokens,
                temperature: options.temperature,
                topP: options.top_p,
                stopSequences: options.stop_sequence,
            } as InferenceConfiguration,
            /*
            additionalModelRequestFields: {
                "top_k": options.top_k,
                "presence_penalty": options.presence_penalty,
                "frequency_penalty": options.frequency_penalty,
            },
            */
        } as ConverseRequest;
    }


    async requestImageGeneration(prompt: NovaMessagesPrompt, options: ImageGenExecutionOptions): Promise<Completion<ImageGeneration>> {

        if (options.output_modality !== Modalities.image) {
            throw new Error(`Image generation requires image output_modality`);
        }

        const executor = this.getExecutor();
        const taskType = () => {
            switch (options.generation_type) {
                case "text-to-image":
                    if (options.input_image_use === "variation") {
                        return NovaImageGenerationTaskType.IMAGE_VARIATION;
                    } else {
                        return NovaImageGenerationTaskType.TEXT_IMAGE
                    }
                default:
                    return NovaImageGenerationTaskType.TEXT_IMAGE
            }
        }
        this.logger.info("Task type: " + taskType());

        if (typeof prompt === "string" ) {
            throw  new Error( "Bad prompt format");
        }

        const payload = await formatNovaImageGenerationPayload(taskType(), prompt, options);

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
        const filter = (m: FoundationModelSummary) => m.inferenceTypesSupported?.includes("ON_DEMAND") ?? false;
        return this._listModels(filter);
    }

    async _listModels(foundationFilter?: (m: FoundationModelSummary) => boolean): Promise<AIModel[]> {
        const service = this.getService();
        const [foundationals, customs, inferenceProfiles] = await Promise.all([
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

        if (!foundationals?.modelSummaries) {
            throw new Error("Foundation models not found");
        }

        let fmodels = foundationals.modelSummaries || [];
        if (foundationFilter) {
            fmodels = fmodels.filter(foundationFilter);
        }

        const aimodels: AIModel[] = fmodels.map((m) => {

            if (!m.modelId) {
                throw new Error("modelId not found");
            }

            const model: AIModel = {
                id: m.modelArn ?? m.modelId,
                name: `${m.providerName} ${m.modelName}`,
                provider: this.provider,
                //description: ``,
                owner: m.providerName,
                can_stream: m.responseStreamingSupported ?? false,
                is_multimodal: m.inputModalities?.includes("IMAGE") ?? false,
                tags: m.outputModalities ?? [],
            };

            return model;
        });

        //add custom models
        if (customs?.modelSummaries) {
            customs.modelSummaries.forEach((m) => {

                if (!m.modelArn) {
                    throw new Error("Model ID not found");
                }

                const model: AIModel = {
                    id: m.modelArn,
                    name: m.modelName ?? m.modelArn,
                    provider: this.provider,
                    description: `Custom model from ${m.baseModelName}`,
                    is_custom: true,
                };

                aimodels.push(model);
                this.validateConnection;
            });
        }

        //add inference profiles
        if (inferenceProfiles?.inferenceProfileSummaries) {
            inferenceProfiles.inferenceProfileSummaries.forEach((p) => {
                if (!p.inferenceProfileArn) {
                    throw new Error("Profile ARN not found");
                }

                const model: AIModel = {
                    id: p.inferenceProfileArn ?? p.inferenceProfileId,
                    name: p.inferenceProfileName ?? p.inferenceProfileArn,
                    provider: this.provider,
                };

                aimodels.push(model);
            });
        }

        return aimodels;
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