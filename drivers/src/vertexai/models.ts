import { AIModel, Completion, PromptSegment, ExecutionOptions, CompletionChunkObject } from "@llumiverse/core";
import { VertexAIDriver, trimModelName } from "./index.js";
import { GeminiModelDefinition } from "./models/gemini.js";
import { ClaudeModelDefinition } from "./models/claude.js";
import { LLamaModelDefinition } from "./models/llama.js";

export interface ModelDefinition<PromptT = any> {
    model: AIModel;
    versions?: string[]; // the versions of the model that are available. ex: ['001', '002']
    createPrompt: (driver: VertexAIDriver, segments: PromptSegment[], options: ExecutionOptions) => Promise<PromptT>;
    requestTextCompletion: (driver: VertexAIDriver, prompt: PromptT, options: ExecutionOptions) => Promise<Completion>;
    requestTextCompletionStream: (driver: VertexAIDriver, prompt: PromptT, options: ExecutionOptions) => Promise<AsyncIterable<CompletionChunkObject>>;
    preValidationProcessing?(result: Completion, options: ExecutionOptions): { result: Completion, options: ExecutionOptions };
}

export function getModelDefinition(model: string): ModelDefinition {
    const splits = model.split("/");

    // Handle both formats: "publishers/anthropic/models/..." and "locations/.../publishers/anthropic/models/..."
    let publisher: string | undefined;
    let modelName: string;

    const publisherIndex = splits.indexOf("publishers");
    if (publisherIndex !== -1 && publisherIndex + 1 < splits.length) {
        publisher = splits[publisherIndex + 1];
        modelName = trimModelName(splits[splits.length - 1]);
    } else {
        // Fallback to old logic for backward compatibility
        publisher = splits[1];
        modelName = trimModelName(splits[splits.length - 1]);
    }

    if (publisher?.includes("anthropic")) {
        return new ClaudeModelDefinition(modelName);
    } else if (publisher?.includes("google")) {
        return new GeminiModelDefinition(modelName);
    } else if (publisher?.includes("meta")) {
        return new LLamaModelDefinition(modelName);
    }

    //Fallback, assume it is Gemini.
    return new GeminiModelDefinition(modelName);
}