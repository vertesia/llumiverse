/**
 * Formatters for converting request data to JSONL format for batch processing.
 * Each model type has its own expected input format for batch operations.
 * Uses the same request formats as the regular inference pipeline.
 */

import type { PromptSegment } from "@llumiverse/common";

// ============== Gemini JSONL Format ==============

/**
 * Gemini batch input format.
 * @see https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction
 */
interface GeminiBatchInputLine {
    request: {
        contents: Array<{
            role: "user" | "model";
            parts: Array<{ text: string }>;
        }>;
    };
}

/**
 * Converts PromptSegment role to Gemini role.
 */
function toGeminiRole(role: string): "user" | "model" {
    if (role === "assistant") return "model";
    return "user";
}

/**
 * Formats messages for Gemini batch processing.
 * @param messages - The conversation messages (same format as regular inference)
 * @returns JSONL line string
 */
export function formatGeminiRequest(messages: PromptSegment[]): string {
    const contents = messages
        .filter(msg => msg.role !== "system") // Gemini handles system separately
        .map(msg => ({
            role: toGeminiRole(msg.role),
            parts: [{ text: msg.content }],
        }));

    // Handle system message as instruction in first user message
    const systemMsg = messages.find(msg => msg.role === "system");
    if (systemMsg && contents.length > 0 && contents[0].role === "user") {
        contents[0].parts.unshift({ text: systemMsg.content });
    }

    const line: GeminiBatchInputLine = {
        request: { contents },
    };

    return JSON.stringify(line);
}

// ============== Claude JSONL Format ==============

/**
 * Claude batch input format for Vertex AI batchPredictionJobs.
 * @see https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
 */
interface ClaudeBatchInputLine {
    custom_id: string;
    request: {
        messages: Array<{
            role: "user" | "assistant";
            content: string;
        }>;
        anthropic_version: string;
        max_tokens: number;
        system?: string;
    };
}

/**
 * Formats messages for Claude batch processing.
 * @param messages - The conversation messages (same format as regular inference)
 * @param customId - Unique identifier for this request in the batch (required for Claude batch API)
 * @param maxTokens - Maximum tokens to generate (default: 1024)
 * @returns JSONL line string
 */
export function formatClaudeRequest(
    messages: PromptSegment[],
    customId: string,
    maxTokens: number = 1024
): string {
    // Extract system message
    const systemMsg = messages.find(msg => msg.role === "system");

    // Convert non-system messages
    const claudeMessages = messages
        .filter(msg => msg.role !== "system" && msg.role !== "tool")
        .map(msg => ({
            role: msg.role === "assistant" ? "assistant" as const : "user" as const,
            content: msg.content,
        }));

    const line: ClaudeBatchInputLine = {
        custom_id: customId,
        request: {
            messages: claudeMessages,
            anthropic_version: "vertex-2023-10-16",
            max_tokens: maxTokens,
            ...(systemMsg ? { system: systemMsg.content } : {}),
        },
    };

    return JSON.stringify(line);
}

// ============== Embeddings JSONL Format ==============

/**
 * Embeddings batch input format.
 */
interface EmbeddingsBatchInputLine {
    content: string;
}

/**
 * Extracts text content from PromptSegment array.
 */
function extractText(messages: PromptSegment[]): string {
    return messages
        .filter(msg => msg.role === "user")
        .map(msg => msg.content)
        .join(" ");
}

/**
 * Formats messages for embeddings batch processing.
 * @param messages - The conversation messages (same format as regular inference)
 * @returns JSONL line string
 */
export function formatEmbeddingsRequest(messages: PromptSegment[]): string {
    const line: EmbeddingsBatchInputLine = {
        content: extractText(messages),
    };

    return JSON.stringify(line);
}

