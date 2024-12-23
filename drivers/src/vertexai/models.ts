import { AIModel, Completion, CompletionChunkObject, ExecutionOptions, ModelType, PromptOptions, PromptSegment } from "@llumiverse/core";
import { VertexAIDriver } from "./index.js";
import { GeminiModelDefinition } from "./models/gemini.js";




export interface ModelDefinition<PromptT = any> {
    model: AIModel;
    versions?: string[]; // the versions of the model that are available. ex: ['001', '002']
    createPrompt: (driver: VertexAIDriver, segments: PromptSegment[], options: PromptOptions) => Promise<PromptT>;
    requestCompletion: (driver: VertexAIDriver, prompt: PromptT, options: ExecutionOptions) => Promise<Completion>;
    requestCompletionStream: (driver: VertexAIDriver, promp: PromptT, options: ExecutionOptions) => Promise<AsyncIterable<CompletionChunkObject>>;
}

export function getModelName(model: string) {
    const i = model.lastIndexOf('@');
    return i > -1 ? model.substring(0, i) : model;
}

export function getModelDefinition(model: string): ModelDefinition {
    const modelName = getModelName(model);
    const def = Models[modelName];
    if (!def) {
        throw new Error(`Unknown model ${model}`);
    }
    return def;
}

export function getAIModels() {
    return Object.values(Models).map(m => m.model);
}

// Builtin models. VertexAI doesn't provide an API to list models. so we have to hardcode them here.
export const BuiltinModels: AIModel<string>[] = [
    {
        id: "gemini-1.5-flash-001",
        name: "Gemini Pro 1.5 Flash 001",
        provider: "vertexai",
        owner: "google",
        type: ModelType.MultiModal,
        can_stream: true,
        is_multimodal: true
    },
    {
        id: "gemini-1.5-flash-002",
        name: "Gemini Pro 1.5 Flash 002",
        provider: "vertexai",
        owner: "google",
        type: ModelType.MultiModal,
        can_stream: true,
        is_multimodal: true
    },
    {
        id: "gemini-1.5-flash",
        name: "Gemini Pro 1.5 Flash",
        provider: "vertexai",
        owner: "google",
        type: ModelType.MultiModal,
        can_stream: true,
        is_multimodal: true
    },
    {
        id: "gemini-1.5-pro-001",
        name: "Gemini Pro 1.5 Pro 001",
        provider: "vertexai",
        owner: "google",
        type: ModelType.MultiModal,
        can_stream: true,
        is_multimodal: true
    },
    {
        id: "gemini-1.5-pro-002",
        name: "Gemini Pro 1.5 Pro 002",
        provider: "vertexai",
        owner: "google",
        type: ModelType.MultiModal,
        can_stream: true,
        is_multimodal: true
    },
    {
        id: "gemini-1.5-pro",
        name: "Gemini Pro 1.5 Pro",
        provider: "vertexai",
        owner: "google",
        type: ModelType.MultiModal,
        can_stream: true,
        is_multimodal: true
    },
    {
        id: "gemini-1.0-pro",
        name: "Gemini Pro 1.0",
        provider: "vertexai",
        owner: "google",
        type: ModelType.Text,
        can_stream: true,
    },
]

// Must be updated when adding new models
const Models: Record<string, ModelDefinition> = {
    "gemini-1.5-flash-002": new GeminiModelDefinition("gemini-1.5-flash-002"),
    "gemini-1.5-flash-001": new GeminiModelDefinition("gemini-1.5-flash-001"),
    "gemini-1.5-flash": new GeminiModelDefinition("gemini-1.5-flash"),
    "gemini-1.5-pro-002": new GeminiModelDefinition("gemini-1.5-pro-002"),
    "gemini-1.5-pro-001": new GeminiModelDefinition("gemini-1.5-pro-001"),
    "gemini-1.5-pro": new GeminiModelDefinition("gemini-1.5-pro"),
    "gemini-1.0-pro": new GeminiModelDefinition(),
}