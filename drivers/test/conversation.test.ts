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
import { BedrockDriver, OpenAIDriver, VertexAIDriver, xAIDriver } from '../src';
import { completionResultToString } from './utils';

const TIMEOUT = 120 * 1000;

interface TestDriver {
    driver: AbstractDriver;
    textModel: string;
    visionModel?: string;
    name: string;
}

const drivers: TestDriver[] = [];

if (process.env.GOOGLE_PROJECT_ID && process.env.GOOGLE_REGION) {
    drivers.push({
        name: "vertexai-gemini",
        driver: new VertexAIDriver({
            project: process.env.GOOGLE_PROJECT_ID as string,
            region: process.env.GOOGLE_REGION as string,
        }),
        textModel: "publishers/google/models/gemini-2.5-flash",
        visionModel: "publishers/google/models/gemini-2.5-flash",
    });
    // Also test Claude Sonnet 4.5 on VertexAI
    drivers.push({
        name: "vertexai-claude-sonnet",
        driver: new VertexAIDriver({
            project: process.env.GOOGLE_PROJECT_ID as string,
            region: process.env.GOOGLE_REGION as string,
        }),
        textModel: "publishers/anthropic/models/claude-sonnet-4-5",
        visionModel: "publishers/anthropic/models/claude-sonnet-4-5",
    });
    // Also test Claude Opus 4.6 on VertexAI
    drivers.push({
        name: "vertexai-claude-opus",
        driver: new VertexAIDriver({
            project: process.env.GOOGLE_PROJECT_ID as string,
            region: process.env.GOOGLE_REGION as string,
        }),
        textModel: "publishers/anthropic/models/claude-opus-4-6",
        visionModel: "publishers/anthropic/models/claude-opus-4-6",
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
        name: "bedrock-claude",
        driver: new BedrockDriver({
            region: process.env.BEDROCK_REGION as string,
        }),
        // Claude Sonnet 4 for text and vision
        textModel: "us.anthropic.claude-sonnet-4-20250514-v1:0",
        visionModel: "us.anthropic.claude-sonnet-4-20250514-v1:0",
    });
    drivers.push({
        name: "bedrock-deepseek-v3",
        driver: new BedrockDriver({
            region: process.env.BEDROCK_REGION as string,
        }),
        // DeepSeek V3.2 for text only (no vision support)
        textModel: "deepseek.v3.2",
    });
    drivers.push({
        name: "bedrock-deepseek-r1",
        driver: new BedrockDriver({
            region: process.env.BEDROCK_REGION as string,
        }),
        // DeepSeek R1 for text only (no vision support)
        textModel: "us.deepseek.r1-v1:0",
    });
} else {
    console.warn("Bedrock tests are skipped: BEDROCK_REGION environment variable is not set");
}

if (process.env.XAI_API_KEY) {
    drivers.push({
        name: "xai",
        driver: new xAIDriver({
            apiKey: process.env.XAI_API_KEY as string,
        }),
        textModel: "grok-4-1-fast-reasoning",
        visionModel: "grok-4-1-fast-reasoning",
    });
} else {
    console.warn("xAI tests are skipped: XAI_API_KEY environment variable is not set");
}

/**
 * DataSource implementation that fetches image from URL and provides it as a stream.
 * This simulates how Studio sends images - fetched and converted to base64/bytes.
 */
class ImageUrlSource implements DataSource {
    private cachedBuffer: ArrayBuffer | null = null;

    constructor(public url: string, public mime_type: string = "image/jpeg") { }

    get name() {
        return this.url.split('/').pop() || 'image';
    }

    async getURL(): Promise<string> {
        // Return the original URL - driver will fetch it
        return this.url;
    }

    async getStream(): Promise<ReadableStream<string | Uint8Array>> {
        // Fetch the image and return as stream (this is what Studio does)
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

/**
 * DataSource implementation that provides pre-fetched image data as base64.
 * This more closely simulates how Studio sends images through the conversation.
 */
class Base64ImageSource implements DataSource {
    private base64Data: string;

    constructor(
        base64Data: string,
        public mime_type: string = "image/png",
        private imageName: string = "image"
    ) {
        this.base64Data = base64Data;
    }

    get name() {
        return this.imageName;
    }

    async getURL(): Promise<string> {
        // Return as data URL
        return `data:${this.mime_type};base64,${this.base64Data}`;
    }

    async getStream(): Promise<ReadableStream<string | Uint8Array>> {
        // Convert base64 to Uint8Array and return as stream
        const binaryString = atob(this.base64Data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return new ReadableStream({
            start(controller) {
                controller.enqueue(bytes);
                controller.close();
            }
        });
    }

    static async fromUrl(url: string, mimeType: string = "image/png"): Promise<Base64ImageSource> {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch image from url: ${url}`);
        }
        const arrayBuffer = await response.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        const base64 = btoa(binary);
        const name = url.split('/').pop() || 'image';
        return new Base64ImageSource(base64, mimeType, name);
    }
}

function getTextOptions(model: string): ExecutionOptions {
    // GPT-5 and similar reasoning models don't support temperature
    const isReasoningModel = model.includes("gpt-5") || model.includes("o1") || model.includes("o3");

    return {
        model: model,
        model_options: isReasoningModel ? {
            _option_id: "openai-thinking",
            max_tokens: 1024,
        } : {
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

    test.skipIf(!visionModel)(`${name}: multi-turn conversation with image`, { timeout: TIMEOUT }, async () => {
        const options = getTextOptions(visionModel!);

        // Turn 1: Send an image as base64 (like Studio does)
        // Using Google logo as a simple, accessible test image
        const imageUrl = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png";
        const imageSource = await Base64ImageSource.fromUrl(imageUrl, "image/png");
        const prompt1: PromptSegment[] = [{
            role: PromptRole.user,
            content: "What company logo is shown in this image? Please describe the colors you see.",
            files: [imageSource]
        }];

        const result1 = await driver.execute(prompt1, options);
        expect(result1.result.length).toBeGreaterThan(0);
        const text1 = result1.result.map(completionResultToString).join("").toLowerCase();
        expect(text1).toMatch(/google/i);
        expect(result1.conversation).toBeDefined();

        // Critical test: Verify conversation is serializable after image
        // This is where Bedrock would fail without the stripBinaryFromConversation fix
        verifyConversationSerializable(result1.conversation, name);

        // Simulate storage: serialize and deserialize the conversation
        const storedConversation1 = JSON.parse(JSON.stringify(result1.conversation));

        // Turn 2: Follow-up question about the image (without sending it again)
        const prompt2: PromptSegment[] = [{
            role: PromptRole.user,
            content: "How many colors are in the logo you just described?"
        }];

        const result2 = await driver.execute(prompt2, { ...options, conversation: storedConversation1 });
        expect(result2.result.length).toBeGreaterThan(0);
        const text2 = result2.result.map(completionResultToString).join("");
        // Google logo has 4 colors (blue, red, yellow, green)
        expect(text2).toMatch(/4|four/i);

        // Verify conversation is still serializable
        verifyConversationSerializable(result2.conversation, name);

        // Simulate storage again
        const storedConversation2 = JSON.parse(JSON.stringify(result2.conversation));

        // Turn 3: Another follow-up
        const prompt3: PromptSegment[] = [{
            role: PromptRole.user,
            content: "What is the first letter in that company name?"
        }];

        const result3 = await driver.execute(prompt3, { ...options, conversation: storedConversation2 });
        expect(result3.result.length).toBeGreaterThan(0);
        const text3 = result3.result.map(completionResultToString).join("").toLowerCase();
        expect(text3).toContain("g");

        // Final verification
        verifyConversationSerializable(result3.conversation, name);
    });

    test.skipIf(!visionModel)(`${name}: conversation with multiple images`, { timeout: TIMEOUT }, async () => {
        const options = getTextOptions(visionModel!);

        // Turn 1: Send two images as base64 (like Studio does)
        const googleLogoUrl = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png";
        const githubLogoUrl = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png";

        const [googleImage, githubImage] = await Promise.all([
            Base64ImageSource.fromUrl(googleLogoUrl, "image/png"),
            Base64ImageSource.fromUrl(githubLogoUrl, "image/png")
        ]);

        const prompt1: PromptSegment[] = [{
            role: PromptRole.user,
            content: "I'm showing you two company logos. Please identify both companies.",
            files: [googleImage, githubImage]
        }];

        const result1 = await driver.execute(prompt1, options);
        expect(result1.result.length).toBeGreaterThan(0);
        const text1 = result1.result.map(completionResultToString).join("").toLowerCase();
        expect(text1).toMatch(/google/);
        expect(text1).toMatch(/github/);
        expect(result1.conversation).toBeDefined();

        // Critical test: Verify conversation with multiple images is serializable
        verifyConversationSerializable(result1.conversation, name);

        // Simulate storage
        const storedConversation1 = JSON.parse(JSON.stringify(result1.conversation));

        // Turn 2: Ask about the images
        const prompt2: PromptSegment[] = [{
            role: PromptRole.user,
            content: "Which of the two logos has more colors?"
        }];

        const result2 = await driver.execute(prompt2, { ...options, conversation: storedConversation1 });
        expect(result2.result.length).toBeGreaterThan(0);
        const text2 = result2.result.map(completionResultToString).join("").toLowerCase();
        // Google logo has 4 colors, GitHub logo is monochrome
        expect(text2).toMatch(/google/);

        // Verify conversation is still serializable after second turn
        verifyConversationSerializable(result2.conversation, name);
    });

    /**
     * STREAMING CONVERSATION TESTS
     * These tests verify that streaming (driver.stream()) properly maintains
     * conversation context across multiple turns.
     *
     * This was a bug where streaming didn't include conversation history in the API call,
     * causing the model to lose context between turns.
     */

    test(`${name}: STREAMING multi-turn text conversation (3 turns)`, { timeout: TIMEOUT }, async () => {
        const options = getTextOptions(textModel);

        // Helper to consume stream and get completion
        async function streamToCompletion(stream: AsyncIterable<string>) {
            const chunks: string[] = [];
            for await (const chunk of stream) {
                chunks.push(chunk);
            }
            return chunks.join('');
        }

        // Turn 1: Ask a question using streaming
        const prompt1: PromptSegment[] = [{
            role: PromptRole.user,
            content: "I'm thinking of a secret word. The word is 'elephant'. Remember this word."
        }];

        const stream1 = await driver.stream(prompt1, options);
        const text1 = await streamToCompletion(stream1);
        expect(text1.length).toBeGreaterThan(0);
        expect(stream1.completion).toBeDefined();
        expect(stream1.completion?.conversation).toBeDefined();

        // Verify conversation is serializable
        verifyConversationSerializable(stream1.completion?.conversation, name);

        // Simulate storage: serialize and deserialize the conversation
        const storedConversation1 = JSON.parse(JSON.stringify(stream1.completion?.conversation));

        // Turn 2: Follow-up question using streaming
        const prompt2: PromptSegment[] = [{
            role: PromptRole.user,
            content: "What was the secret word I told you?"
        }];

        const stream2 = await driver.stream(prompt2, { ...options, conversation: storedConversation1 });
        const text2 = await streamToCompletion(stream2);
        expect(text2.length).toBeGreaterThan(0);
        // The model should remember "elephant" from the conversation context
        expect(text2.toLowerCase()).toContain("elephant");
        expect(stream2.completion?.conversation).toBeDefined();

        // Verify conversation is still serializable
        verifyConversationSerializable(stream2.completion?.conversation, name);

        // Simulate storage again
        const storedConversation2 = JSON.parse(JSON.stringify(stream2.completion?.conversation));

        // Turn 3: Another follow-up using streaming
        const prompt3: PromptSegment[] = [{
            role: PromptRole.user,
            content: "How many letters are in that secret word?"
        }];

        const stream3 = await driver.stream(prompt3, { ...options, conversation: storedConversation2 });
        const text3 = await streamToCompletion(stream3);
        expect(text3.length).toBeGreaterThan(0);
        // "elephant" has 8 letters
        expect(text3).toMatch(/8|eight/i);

        // Final verification
        verifyConversationSerializable(stream3.completion?.conversation, name);
    });

    test(`${name}: STREAMING conversation preserves context after JSON serialization`, { timeout: TIMEOUT }, async () => {
        const options = getTextOptions(textModel);

        // Helper to consume stream and get completion
        async function streamToCompletion(stream: AsyncIterable<string>) {
            for await (const _chunk of stream) {
                // Just consume the stream
            }
        }

        // Turn 1: Establish context with streaming
        const prompt1: PromptSegment[] = [{
            role: PromptRole.user,
            content: "My name is TestUser123. Please greet me by name."
        }];

        const stream1 = await driver.stream(prompt1, options);
        await streamToCompletion(stream1);
        expect(stream1.completion?.conversation).toBeDefined();

        // Critical: Simulate how Studio stores conversations (JSON serialize/deserialize)
        const serialized = JSON.stringify(stream1.completion?.conversation);
        const storedConversation = JSON.parse(serialized);

        // Turn 2: Verify context survives serialization in streaming mode
        const prompt2: PromptSegment[] = [{
            role: PromptRole.user,
            content: "What is my name?"
        }];

        const stream2 = await driver.stream(prompt2, { ...options, conversation: storedConversation });
        await streamToCompletion(stream2);

        // Get the result text from completion
        const text2 = stream2.completion?.result.map(completionResultToString).join("") || "";

        // The model should remember "TestUser123" from the serialized conversation
        expect(text2).toContain("TestUser123");
    });

    test(`${name}: STREAMING vs non-streaming produce consistent conversation context`, { timeout: TIMEOUT }, async () => {
        const options = getTextOptions(textModel);

        // Helper to consume stream
        async function streamToCompletion(stream: AsyncIterable<string>) {
            for await (const _chunk of stream) {
                // Just consume
            }
        }

        // First turn with non-streaming
        const prompt1: PromptSegment[] = [{
            role: PromptRole.user,
            content: "Remember this code: ALPHA-7749"
        }];

        const result1 = await driver.execute(prompt1, options);
        const storedConversation1 = JSON.parse(JSON.stringify(result1.conversation));

        // Second turn with streaming - should work with conversation from non-streaming
        const prompt2: PromptSegment[] = [{
            role: PromptRole.user,
            content: "What code did I tell you to remember?"
        }];

        const stream2 = await driver.stream(prompt2, { ...options, conversation: storedConversation1 });
        await streamToCompletion(stream2);

        const text2 = stream2.completion?.result.map(completionResultToString).join("") || "";
        expect(text2).toContain("ALPHA-7749");

        // Third turn back to non-streaming - should work with conversation from streaming
        const storedConversation2 = JSON.parse(JSON.stringify(stream2.completion?.conversation));

        const prompt3: PromptSegment[] = [{
            role: PromptRole.user,
            content: "What were the numbers in that code?"
        }];

        const result3 = await driver.execute(prompt3, { ...options, conversation: storedConversation2 });
        const text3 = result3.result.map(completionResultToString).join("");
        expect(text3).toContain("7749");
    });
});
