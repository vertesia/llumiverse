import { describe, expect, test } from "vitest";
import { Content } from "@google/genai";
import { ExecutionOptions, unwrapConversationArray } from "@llumiverse/core";
import { VertexAIDriver } from "../src";

function extractTextParts(message: { parts?: Array<{ text?: string }> }): string[] {
    if (!message.parts) return [];
    return message.parts.map(p => p.text ?? "").filter(Boolean);
}

describe("VertexAI streaming conversation rebuild", () => {
    test("Gemini streaming path does not duplicate history when prompt already merged", () => {
        const driver = new VertexAIDriver({ project: "test", region: "us-central1" });

        const summary: Content = { role: "model", parts: [{ text: "summary" }] };
        const userMsg: Content = { role: "user", parts: [{ text: "continue" }] };

        const prompt = {
            contents: [summary, userMsg],
        };

        const options = {
            model: "publishers/google/models/gemini-2.5-flash",
            conversation: [summary],
        } as ExecutionOptions;

        const result = [{ type: "text", value: "response" }];

        const conversation = driver.buildStreamingConversation(
            prompt as any,
            result as any,
            undefined,
            options
        ) as unknown;

        const unwrapped = unwrapConversationArray<Content>(conversation) ?? (conversation as Content[]);

        const summaryCount = unwrapped.filter(m =>
            m.role === "model" && extractTextParts(m).includes("summary")
        ).length;

        expect(summaryCount).toBe(1);
        expect(unwrapped.length).toBe(3);
        expect(extractTextParts(unwrapped[2])).toEqual(["response"]);
    });

    test("Claude streaming path includes existing history once", () => {
        const driver = new VertexAIDriver({ project: "test", region: "us-central1" });

        const existing = {
            messages: [
                { role: "assistant", content: [{ type: "text", text: "summary" }] },
            ],
            system: [{ type: "text", text: "system" }],
        };

        const prompt = {
            messages: [
                { role: "user", content: [{ type: "text", text: "continue" }] },
            ],
        };

        const options = {
            model: "publishers/anthropic/models/claude-sonnet-4-5",
            conversation: existing,
        } as ExecutionOptions;

        const result = [{ type: "text", value: "response" }];

        const conversation = driver.buildStreamingConversation(
            prompt as any,
            result as any,
            undefined,
            options
        ) as any;

        expect(conversation.system).toEqual(existing.system);
        expect(conversation.messages.length).toBe(3);
        expect(conversation.messages[0].content[0].text).toBe("summary");
        expect(conversation.messages[1].content[0].text).toBe("continue");
        expect(conversation.messages[2].content[0].text).toBe("response");
    });
});
