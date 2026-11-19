import {
    type AIModel,
    AIModelStatus,
    type Completion,
    type CompletionStream,
    type DriverOptions,
    type EmbeddingsResult,
    type ExecutionOptions,
    type ExecutionResponse,
    ModelType,
    type PromptSegment,
    type TrainingJob,
} from '@llumiverse/core';
import { AbstractDriver } from '@llumiverse/core/driver';
import { createFakeAnswer } from './fake-answer.js';
import { TestErrorCompletionStream } from './TestErrorCompletionStream.js';
import { TestValidationErrorCompletionStream } from './TestValidationErrorCompletionStream.js';
import { createValidationErrorCompletion, sleep, throwError } from './utils.js';

export * from './fake-answer.js';
export * from './TestErrorCompletionStream.js';
export * from './TestValidationErrorCompletionStream.js';

export enum TestDriverModels {
    executionError = 'execution-error',
    validationError = 'validation-error',
    fakeAnswer = 'fake-answer',
    fakeAnswerInstant = 'fake-answer-instant',
    fakeAnswerSlow = 'fake-answer-slow',
}

/**
 * The fake models: they answer with the smallest value that matches the result schema, without calling any model,
 * after a delay that holds the caller like a real call. A load test reaches the whole path of an execution with
 * them, at no cost: the API, the limiters, the queues, the workers and the metering.
 */
const FAKE_ANSWER_MODELS: Record<string, { name: string; delayMs: number; description: string }> = {
    [TestDriverModels.fakeAnswer]: {
        name: 'Fake Answer',
        delayMs: 2_000,
        description: 'Answers after 2 s with a value made from the result schema, without calling any model',
    },
    [TestDriverModels.fakeAnswerInstant]: {
        name: 'Fake Answer (Instant)',
        delayMs: 0,
        description: 'Answers at once with a value made from the result schema, without calling any model',
    },
    [TestDriverModels.fakeAnswerSlow]: {
        name: 'Fake Answer (Slow)',
        delayMs: 20_000,
        description:
            'Answers after 20 s, like a reasoning model, with a value made from the result schema, without calling any model',
    },
};

function isFakeAnswerModel(model: string): boolean {
    return Object.hasOwn(FAKE_ANSWER_MODELS, model);
}

export class TestDriver extends AbstractDriver<DriverOptions, PromptSegment[]> {
    provider = 'test';

    constructor() {
        super({});
    }

    async requestTextCompletion(
        prompt: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<Completion> {
        const fake = FAKE_ANSWER_MODELS[options.model];
        if (!fake) {
            throwError(`[test driver] Unknown model: ${options.model}`, prompt);
        }
        await sleep(fake.delayMs, signal);
        return createFakeAnswer(prompt, options.result_schema);
    }

    requestTextCompletionStream(): Promise<never> {
        throw new Error('Method not implemented.');
    }

    createTrainingPrompt(): Promise<string> {
        throw new Error('Method not implemented.');
    }

    startTraining(): Promise<TrainingJob> {
        throw new Error('Method not implemented.');
    }

    cancelTraining(): Promise<TrainingJob> {
        throw new Error('Method not implemented.');
    }

    getTrainingJob(_jobId: string): Promise<TrainingJob> {
        throw new Error('Method not implemented.');
    }

    async createPrompt(segments: PromptSegment[], _opts: ExecutionOptions): Promise<PromptSegment[]> {
        return segments;
    }

    execute(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<ExecutionResponse<PromptSegment[]>> {
        if (isFakeAnswerModel(options.model)) {
            // The path of a real driver: requestTextCompletion, then the validation of the result.
            return super.execute(segments, options, signal);
        }
        switch (options.model) {
            case TestDriverModels.executionError:
                return this.executeError(segments, options);
            case TestDriverModels.validationError:
                return this.executeValidationError(segments, options);
            default:
                throwError(`[test driver] Unknown model: ${options.model}`, segments);
        }
    }
    async stream(
        segments: PromptSegment[],
        options: ExecutionOptions,
        signal?: AbortSignal,
    ): Promise<CompletionStream<PromptSegment[]>> {
        if (isFakeAnswerModel(options.model)) {
            // canStream is false: the stream falls back on execute, and sends the whole answer as one chunk.
            return super.stream(segments, options, signal);
        }
        switch (options.model) {
            case TestDriverModels.executionError:
                return new TestErrorCompletionStream(segments, options);
            case TestDriverModels.validationError:
                return new TestValidationErrorCompletionStream(segments, options);
            default:
                throwError(`[test driver] Unknown model: ${options.model}`, segments);
        }
    }

    protected canStream(): Promise<boolean> {
        return Promise.resolve(false);
    }

    async listTrainableModels(): Promise<AIModel[]> {
        return [];
    }

    async listModels(): Promise<AIModel[]> {
        return [
            {
                id: TestDriverModels.executionError,
                name: 'Execution Error',
                type: ModelType.Test,
                provider: this.provider,
                status: AIModelStatus.Available,
                description: 'Test execution errors',
                tags: [],
            },
            {
                id: TestDriverModels.validationError,
                name: 'Validation Error',
                type: ModelType.Test,
                provider: this.provider,
                status: AIModelStatus.Available,
                description: 'Test validation errors',
                tags: [],
            },
            ...Object.entries(FAKE_ANSWER_MODELS).map(([id, fake]) => ({
                id,
                name: fake.name,
                type: ModelType.Test,
                provider: this.provider,
                status: AIModelStatus.Available,
                description: fake.description,
                tags: [],
            })),
        ];
    }
    validateConnection(): Promise<boolean> {
        throw new Error('Method not implemented.');
    }
    generateEmbeddings(): Promise<EmbeddingsResult> {
        throw new Error('Method not implemented.');
    }

    // ============== execution error ==================
    async executeError(
        segments: PromptSegment[],
        _options: ExecutionOptions,
    ): Promise<ExecutionResponse<PromptSegment[]>> {
        await sleep(1000);
        throwError('Testing stream completion error.', segments);
    }
    // ============== validation error ==================
    async executeValidationError(
        segments: PromptSegment[],
        _options: ExecutionOptions,
    ): Promise<ExecutionResponse<PromptSegment[]>> {
        await sleep(3000);
        return createValidationErrorCompletion(segments);
    }
}
