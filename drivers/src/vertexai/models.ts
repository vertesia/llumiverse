import { AIModel, Completion, CompletionChunkObject, PromptOptions, PromptSegment, ExecutionOptions } from "@llumiverse/core";
import { VertexAIDriver , trimModelName} from "./index.js";
import { GeminiModelDefinition } from "./models/gemini.js";
import { ClaudeModelDefinition } from "./models/claude.js";

export interface ModelDefinition<PromptT = any> {
    model: AIModel;
    versions?: string[]; // the versions of the model that are available. ex: ['001', '002']
    createPrompt: (driver: VertexAIDriver, segments: PromptSegment[], options: PromptOptions) => Promise<PromptT>;
    requestTextCompletion: (driver: VertexAIDriver, prompt: PromptT, options: ExecutionOptions) => Promise<Completion>;
    requestTextCompletionStream: (driver: VertexAIDriver, prompt: PromptT, options: ExecutionOptions) => Promise<AsyncIterable<CompletionChunkObject>>;
    preValidationProcessing?(result: Completion, options: ExecutionOptions): { result: Completion, options: ExecutionOptions };
}

export function getModelDefinition(model: string): ModelDefinition {
    const splits = model.split("/");
    const publisher = splits[1];
    const modelName = trimModelName(splits[splits.length - 1]);
    
    if (publisher?.includes("anthropic")) {
        return new ClaudeModelDefinition(modelName);
    } else if (publisher?.includes("google")) {
        return new GeminiModelDefinition(modelName);
    }

    //Fallback, assume it is Gemini.
    return new GeminiModelDefinition(modelName);
}