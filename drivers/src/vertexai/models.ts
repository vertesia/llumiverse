import { AIModel, Completion, CompletionChunkObject, ExecutionOptions, PromptOptions, PromptSegment } from "@llumiverse/core";
import { VertexAIDriver , trimModelName} from "./index.js";
import { GeminiModelDefinition } from "./models/gemini.js";
import { ClaudeModelDefinition } from "./models/claude.js";

export interface ModelDefinition<PromptT = any> {
    model: AIModel;
    versions?: string[]; // the versions of the model that are available. ex: ['001', '002']
    createPrompt: (driver: VertexAIDriver, segments: PromptSegment[], options: PromptOptions) => Promise<PromptT>;
    requestCompletion: (driver: VertexAIDriver, prompt: PromptT, options: ExecutionOptions) => Promise<Completion>;
    requestCompletionStream: (driver: VertexAIDriver, promp: PromptT, options: ExecutionOptions) => Promise<AsyncIterable<CompletionChunkObject>>;
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