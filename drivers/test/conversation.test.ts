/**
 * Tests for multi-turn conversation handling across drivers.
 *
 * These tests verify that:
 * 1. Multi-turn text conversations work correctly
 * 2. Conversations with images work correctly
 * 3. The conversation object survives JSON.stringify/parse (no Uint8Array corruption)
 */

import { AbstractDriver, DataSource, ExecutionOptions, Modalities, PromptRole, PromptSegment } from '@llumiverse/core';
import 'dotenv/config';
import { describe, expect, test } from "vitest";
import { BedrockDriver, OpenAIDriver, VertexAIDriver } from '../src';
import { completionResultToString } from './utils';

const TIMEOUT = 120 * 1000;

interface TestDriver {
    driver: AbstractDriver;
    textModel: string;
    visionModel: string;
    name: string;
}

const drivers: TestDriver[] = [];

if (process.env.GOOGLE_PROJECT_ID && process.env.GOOGLE_REGION) {
    drivers.push({
        name: "vertexai",
        driver: new VertexAIDriver({
            project: process.env.GOOGLE_PROJECT_ID as string,
            region: process.env.GOOGLE_REGION as string,
        }),
        textModel: "publishers/google/models/gemini-3.0-pro-preview",
        visionModel: "publishers/google/models/gemini-3.0-pro-preview",
    });
} else {
    console.warn("VertexAI tests are skipped: GOOGLE_PROJECT_ID environment variable is not set");
}

if (process.env.OPENAI_API_KEY) {
    drivers.push({
        name: "openai",
        driver: new OpenAIDriver({
            apiKey: process.env.OPENAI_API_KEY as string
        }),
        textModel: "gpt-5",
        visionModel: "gpt-5",
    });
} else {
    console.warn("OpenAI tests are skipped: OPENAI_API_KEY environment variable is not set");
}

if (process.env.BEDROCK_REGION) {
    drivers.push({
        name: "bedrock",
        driver: new BedrockDriver({
            region: process.env.BEDROCK_REGION as string,
        }),
        textModel: "us.anthropic.claude-opus-4-20250514-v1:0",
        visionModel: "us.anthropic.claude-opus-4-20250514-v1:0",
    });
} else {
    console.warn("Bedrock tests are skipped: BEDROCK_REGION environment variable is not set");
}

/**
 * DataSource implementation for URL-based images
 */
class ImageUrlSource implements DataSource {
    constructor(public url: string, public mime_type: string = "image/jpeg") { }
    get name() {
        return this.url.split('/').pop() || 'image';
    }
    async getURL(): Promise<string> {
        return this.url;
    }
    async getStream(): Promise<ReadableStream<string | Uint8Array>> {
        const response = await fetch(this.url);
        if (!response.ok) {
            throw new Error(`Failed to fetch image from url: ${this.url}`);
        }
        if (!response.body) {
            throw new Error(`No content from url: ${this.url}`);
        }
        return response.body;
    }
}

function getTextOptions(model: string): ExecutionOptions {
    return {
        model: model,
        model_options: {
            _option_id: "text-fallback",
            max_tokens: 256,
            temperature: 0.3,
        },
        output_modality: Modalities.text,
    };
}

/**
 * Verify that a conversation object can survive JSON serialization.
 * This catches Uint8Array corruption issues where bytes become { "0": 137, "1": 80, ... }
 */
function verifyConversationSerializable(conversation: unknown, driverName: string): void {
    const serialized = JSON.stringify(conversation);
    const deserialized = JSON.parse(serialized);

    // Check that no Uint8Array was corrupted into an object like { "0": 137, "1": 80 }
    // Such corrupted objects would have numeric string keys
    const checkForCorruptedBytes = (obj: unknown, path: string = ''): void => {
        if (obj === null || obj === undefined) return;

        if (typeof obj === 'object' && !Array.isArray(obj)) {
            const keys = Object.keys(obj as Record<string, unknown>);
            // If all keys are numeric strings, this might be a corrupted Uint8Array
            if (keys.length > 10 && keys.every(k => /^\d+$/.test(k))) {
                throw new Error(
                    `[${driverName}] Possible corrupted Uint8Array at ${path}: ` +
                    `Found object with ${keys.length} numeric keys. ` +
                    `This suggests binary data was not properly stripped before serialization.`
                );
            }
            for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
                checkForCorruptedBytes(value, `${path}.${key}`);
            }
        }

        if (Array.isArray(obj)) {
            obj.forEach((item, i) => checkForCorruptedBytes(item, `${path}[${i}]`));
        }
    };

    checkForCorruptedBytes(deserialized, 'conversation');
}

// Skip driver-specific tests if no drivers are configured
const hasDrivers = drivers.length > 0;

// Fallback test suite when no drivers are configured (prevents "no test suite found" error)
describe.skipIf(hasDrivers)("Multi-turn Conversations (no drivers configured)", () => {
    test("skipped - no API keys configured", () => {
        console.warn("All conversation tests skipped: No API keys configured for any driver");
        expect(true).toBe(true);
    });
});

describe.concurrent.skipIf(!hasDrivers).each(drivers)("Driver $name - Multi-turn Conversations", ({ name, driver, textModel, visionModel }) => {

    test(`${name}: multi-turn text conversation (3 turns)`, { timeout: TIMEOUT }, async () => {
        const options = getTextOptions(textModel);

        // Turn 1: Ask a question
        const prompt1: PromptSegment[] = [{
            role: PromptRole.user,
            content: "I'm thinking of a number between 1 and 10. The number is 7. Remember this number."
        }];

        const result1 = await driver.execute(prompt1, options);
        expect(result1.result.length).toBeGreaterThan(0);
        expect(result1.conversation).toBeDefined();

        // Verify conversation is serializable
        verifyConversationSerializable(result1.conversation, name);

        // Simulate storage: serialize and deserialize the conversation
        const storedConversation1 = JSON.parse(JSON.stringify(result1.conversation));

        // Turn 2: Follow-up question
        const prompt2: PromptSegment[] = [{
            role: PromptRole.user,
            content: "What number did I tell you I was thinking of?"
        }];

        const result2 = await driver.execute(prompt2, { ...options, conversation: storedConversation1 });
        expect(result2.result.length).toBeGreaterThan(0);
        const text2 = result2.result.map(completionResultToString).join("");
        expect(text2).toContain("7");
        expect(result2.conversation).toBeDefined();

        // Verify conversation is still serializable
        verifyConversationSerializable(result2.conversation, name);

        // Simulate storage again
        const storedConversation2 = JSON.parse(JSON.stringify(result2.conversation));

        // Turn 3: Another follow-up
        const prompt3: PromptSegment[] = [{
            role: PromptRole.user,
            content: "Add 3 to that number. What is the result?"
        }];

        const result3 = await driver.execute(prompt3, { ...options, conversation: storedConversation2 });
        expect(result3.result.length).toBeGreaterThan(0);
        const text3 = result3.result.map(completionResultToString).join("");
        expect(text3).toContain("10");

        // Final verification
        verifyConversationSerializable(result3.conversation, name);
    });

    test(`${name}: multi-turn conversation with image`, { timeout: TIMEOUT }, async () => {
        const options = getTextOptions(visionModel);

        // Turn 1: Send an image and ask about it
        const imageUrl = "https://upload.wikimedia.org/wikipedia/commons/b/b2/WhiteCat.jpg";
        const prompt1: PromptSegment[] = [{
            role: PromptRole.user,
            content: "What animal is in this image? Please describe it briefly.",
            files: [new ImageUrlSource(imageUrl)]
        }];

        const result1 = await driver.execute(prompt1, options);
        expect(result1.result.length).toBeGreaterThan(0);
        const text1 = result1.result.map(completionResultToString).join("").toLowerCase();
        expect(text1).toContain("cat");
        expect(result1.conversation).toBeDefined();

        // Critical test: Verify conversation is serializable after image
        // This is where Bedrock would fail without the stripBinaryFromConversation fix
        verifyConversationSerializable(result1.conversation, name);

        // Simulate storage: serialize and deserialize the conversation
        const storedConversation1 = JSON.parse(JSON.stringify(result1.conversation));

        // Turn 2: Follow-up question about the image (without sending it again)
        const prompt2: PromptSegment[] = [{
            role: PromptRole.user,
            content: "What color is the animal you just described?"
        }];

        const result2 = await driver.execute(prompt2, { ...options, conversation: storedConversation1 });
        expect(result2.result.length).toBeGreaterThan(0);
        const text2 = result2.result.map(completionResultToString).join("").toLowerCase();
        expect(text2).toContain("white");

        // Verify conversation is still serializable
        verifyConversationSerializable(result2.conversation, name);

        // Simulate storage again
        const storedConversation2 = JSON.parse(JSON.stringify(result2.conversation));

        // Turn 3: Another follow-up
        const prompt3: PromptSegment[] = [{
            role: PromptRole.user,
            content: "Is this a domestic or wild animal?"
        }];

        const result3 = await driver.execute(prompt3, { ...options, conversation: storedConversation2 });
        expect(result3.result.length).toBeGreaterThan(0);
        const text3 = result3.result.map(completionResultToString).join("").toLowerCase();
        expect(text3).toMatch(/domestic|pet|house/);

        // Final verification
        verifyConversationSerializable(result3.conversation, name);
    });

    test(`${name}: conversation with multiple images`, { timeout: TIMEOUT }, async () => {
        const options = getTextOptions(visionModel);

        // Turn 1: Send two images
        const catImageUrl = "https://upload.wikimedia.org/wikipedia/commons/b/b2/WhiteCat.jpg";
        const dogImageUrl = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg";

        const prompt1: PromptSegment[] = [{
            role: PromptRole.user,
            content: "I'm showing you two images. The first is a cat and the second is a dog. Please confirm you can see both animals.",
            files: [
                new ImageUrlSource(catImageUrl),
                new ImageUrlSource(dogImageUrl)
            ]
        }];

        const result1 = await driver.execute(prompt1, options);
        expect(result1.result.length).toBeGreaterThan(0);
        const text1 = result1.result.map(completionResultToString).join("").toLowerCase();
        expect(text1).toMatch(/cat/);
        expect(text1).toMatch(/dog/);
        expect(result1.conversation).toBeDefined();

        // Critical test: Verify conversation with multiple images is serializable
        verifyConversationSerializable(result1.conversation, name);

        // Simulate storage
        const storedConversation1 = JSON.parse(JSON.stringify(result1.conversation));

        // Turn 2: Ask about the images
        const prompt2: PromptSegment[] = [{
            role: PromptRole.user,
            content: "Which of the two animals appears larger in their respective image?"
        }];

        const result2 = await driver.execute(prompt2, { ...options, conversation: storedConversation1 });
        expect(result2.result.length).toBeGreaterThan(0);

        // Verify conversation is still serializable after second turn
        verifyConversationSerializable(result2.conversation, name);
    });
});
