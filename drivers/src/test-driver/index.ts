import { AIModel, AIModelStatus, CompletionStream, Driver, EmbeddingsResult, ExecutionOptions, ExecutionResponse, ModelType, PromptSegment, TrainingJob } from "@llumiverse/core";
import { TestErrorCompletionStream } from "./TestErrorCompletionStream.js";
import { TestValidationErrorCompletionStream } from "./TestValidationErrorCompletionStream.js";
import { createValidationErrorCompletion, sleep, throwError } from "./utils.js";

export * from "./TestErrorCompletionStream.js";
export * from "./TestValidationErrorCompletionStream.js";

export enum TestDriverModels {
    executionError = "execution-error",
    validationError = "validation-error",
}

export class TestDriver implements Driver<PromptSegment[]> {
    provider = "test";

    createTrainingPrompt(): Promise<string> {
        throw new Error("Method not implemented.");
    }

    startTraining(): Promise<TrainingJob> {
        throw new Error("Method not implemented.");
    }

    cancelTraining(): Promise<TrainingJob> {
        throw new Error("Method not implemented.");
    }

    getTrainingJob(_jobId: string): Promise<TrainingJob> {
        throw new Error("Method not implemented.");
    }

    async createPrompt(segments: PromptSegment[], _opts: ExecutionOptions): Promise<PromptSegment[]> {
        return segments;
    }
    
    execute(segments: PromptSegment[], options: ExecutionOptions): Promise<ExecutionResponse<PromptSegment[]>> {
        switch (options.model) {
            case TestDriverModels.executionError:
                return this.executeError(segments, options);
            case TestDriverModels.validationError:
                return this.executeValidationError(segments, options);
            default:
                throwError("[test driver] Unknown model: " + options.model, segments)
        }
    }
    async stream(segments: PromptSegment[], options: ExecutionOptions): Promise<CompletionStream<PromptSegment[]>> {
        switch (options.model) {
            case TestDriverModels.executionError:
                return new TestErrorCompletionStream(segments, options);
            case TestDriverModels.validationError:
                return new TestValidationErrorCompletionStream(segments, options);
            default:
                throwError("[test driver] Unknown model: " + options.model, segments)
        }
    }

    async listTrainableModels(): Promise<AIModel<string>[]> {
        return [];
    }

    async listModels(): Promise<AIModel<string>[]> {
        return [
            {
                id: TestDriverModels.executionError,
                name: "Execution Error",
                type: ModelType.Test,
                provider: this.provider,
                status: AIModelStatus.Available,
                description: "Test execution errors",
                tags: [],
            },
            {
                id: TestDriverModels.validationError,
                name: "Validation Error",
                type: ModelType.Test,
                provider: this.provider,
                status: AIModelStatus.Available,
                description: "Test validation errors",
                tags: [],
            },
        ]
    }

    async listEmbeddingModels(): Promise<AIModel[]> {
        return [];
    }
    
    validateConnection(): Promise<boolean> {
        throw new Error("Method not implemented.");
    }
    generateEmbeddings(): Promise<EmbeddingsResult> {
        throw new Error("Method not implemented.");
    }

    // ============== execution error ==================
    async executeError(segments: PromptSegment[], _options: ExecutionOptions): Promise<ExecutionResponse<PromptSegment[]>> {
        await sleep(1000);
        throwError("Testing stream completion error.", segments);
    }
    // ============== validation error ==================
    async executeValidationError(segments: PromptSegment[], _options: ExecutionOptions): Promise<ExecutionResponse<PromptSegment[]>> {
        await sleep(3000);
        return createValidationErrorCompletion(segments);
    }

}
