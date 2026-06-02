import type { JSONObject, JSONSchema } from '@llumiverse/core';

export interface VertexInlineData {
    data?: string;
    mimeType?: string;
}

export interface VertexFileData {
    fileUri?: string;
    mimeType?: string;
}

export interface VertexFunctionCall {
    name?: string;
    args?: JSONObject;
}

export interface VertexFunctionResponse {
    name?: string;
    response?: unknown;
}

export interface VertexPart {
    text?: string;
    inlineData?: VertexInlineData;
    fileData?: VertexFileData;
    functionCall?: VertexFunctionCall;
    functionResponse?: VertexFunctionResponse;
    thoughtSignature?: string;
}

export interface VertexContent {
    role?: string;
    parts?: VertexPart[];
}

export interface GenerateContentPrompt {
    contents: VertexContent[];
    system?: VertexContent;
}

export interface VertexSafetySetting {
    category: string;
    threshold: string;
}

export interface VertexFunctionDeclaration {
    name: string;
    description?: string;
    parametersJsonSchema?: unknown;
}

export interface VertexTool {
    functionDeclarations: VertexFunctionDeclaration[];
}

export interface VertexThinkingConfig {
    includeThoughts?: boolean;
    thinkingBudget?: number;
    thinkingLevel?: string;
}

export interface VertexGenerateContentConfig {
    candidateCount?: number;
    responseMimeType?: string;
    responseJsonSchema?: JSONSchema;
    temperature?: number;
    topP?: number;
    topK?: number;
    maxOutputTokens?: number;
    stopSequences?: string[];
    presencePenalty?: number;
    frequencyPenalty?: number;
    seed?: number;
    thinkingConfig?: VertexThinkingConfig;
    responseModalities?: string[];
    imageConfig?: {
        imageSize?: string;
        aspectRatio?: string;
        personGeneration?: string;
        prominentPeople?: string;
        outputMimeType?: string;
        outputCompressionQuality?: number;
    };
}

export interface VertexGenerateContentPayload {
    contents: VertexContent[];
    systemInstruction?: VertexContent;
    safetySettings?: VertexSafetySetting[];
    tools?: VertexTool[];
    toolConfig?: {
        functionCallingConfig: {
            mode: 'AUTO';
        };
    };
    labels?: Record<string, string>;
    generationConfig: VertexGenerateContentConfig;
}

export interface VertexGenerateContentUsageMetadata {
    totalTokenCount?: number;
    promptTokenCount?: number;
    cachedContentTokenCount?: number;
    candidatesTokenCount?: number;
    thoughtsTokenCount?: number;
    toolUsePromptTokenCount?: number;
}

export interface VertexGenerateContentCandidate {
    finishReason?: string;
    finishMessage?: string;
    content?: VertexContent;
    safetyRatings?: unknown[];
}

export interface VertexGenerateContentResponse {
    candidates?: VertexGenerateContentCandidate[];
    promptFeedback?: {
        blockReason?: string;
        blockReasonMessage?: string;
    };
    usageMetadata?: VertexGenerateContentUsageMetadata;
}

export interface VertexListModelsResponse {
    models?: VertexListedModel[];
    publisherModels?: VertexListedModel[];
    nextPageToken?: string;
}

export interface VertexListedModel {
    name?: string;
    displayName?: string;
    versionId?: string;
}
