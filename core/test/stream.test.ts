import { describe, expect, test } from "vitest";
import { readStreamAsBase64, readStreamAsString, readStreamAsUint8Array } from "../src/stream";

function createStreamFromString(text: string): ReadableStream {
    return new ReadableStream({
        start(controller) {
            const encoder = new TextEncoder();
            const bytes = encoder.encode(text);
            controller.enqueue(bytes);
            controller.close();
        }
    });
}

function createStreamFromChunks(chunks: string[]): ReadableStream {
    let index = 0;
    return new ReadableStream({
        start(controller) {
            const encoder = new TextEncoder();
            for (const chunk of chunks) {
                controller.enqueue(encoder.encode(chunk));
            }
            controller.close();
        }
    });
}

describe('Stream Reading Functions', () => {
    test('readStreamAsString - single chunk', async () => {
        const testString = "Hello, World!";
        const stream = createStreamFromString(testString);
        const result = await readStreamAsString(stream);
        expect(result).toBe(testString);
    });

    test('readStreamAsString - multiple chunks', async () => {
        const chunks = ["Hello", ", ", "World", "!"];
        const expected = chunks.join("");
        const stream = createStreamFromChunks(chunks);
        const result = await readStreamAsString(stream);
        expect(result).toBe(expected);
    });

    test('readStreamAsString - empty stream', async () => {
        const stream = createStreamFromString("");
        const result = await readStreamAsString(stream);
        expect(result).toBe("");
    });

    test('readStreamAsString - unicode characters', async () => {
        const testString = "Hello 🌍! 你好世界";
        const stream = createStreamFromString(testString);
        const result = await readStreamAsString(stream);
        expect(result).toBe(testString);
    });

    test('readStreamAsBase64 - basic encoding', async () => {
        const testString = "Hello, World!";
        const expected = Buffer.from(testString).toString('base64');
        const stream = createStreamFromString(testString);
        const result = await readStreamAsBase64(stream);
        expect(result).toBe(expected);
    });

    test('readStreamAsBase64 - multiple chunks', async () => {
        const chunks = ["Hello", ", ", "World", "!"];
        const fullString = chunks.join("");
        const expected = Buffer.from(fullString).toString('base64');
        const stream = createStreamFromChunks(chunks);
        const result = await readStreamAsBase64(stream);
        expect(result).toBe(expected);
    });

    test('readStreamAsUint8Array - basic functionality', async () => {
        const testString = "Hello, World!";
        const expected = new TextEncoder().encode(testString);
        const stream = createStreamFromString(testString);
        const result = await readStreamAsUint8Array(stream);
        expect(result).toEqual(expected);
    });

    test('readStreamAsUint8Array - multiple chunks', async () => {
        const chunks = ["Hello", ", ", "World", "!"];
        const fullString = chunks.join("");
        const expected = new TextEncoder().encode(fullString);
        const stream = createStreamFromChunks(chunks);
        const result = await readStreamAsUint8Array(stream);
        expect(result).toEqual(expected);
    });

    test('readStreamAsUint8Array - empty stream', async () => {
        const stream = createStreamFromString("");
        const result = await readStreamAsUint8Array(stream);
        expect(result).toEqual(new Uint8Array(0));
    });

    test('readStreamAsUint8Array - binary data', async () => {
        const binaryData = new Uint8Array([0, 1, 2, 255, 128, 64]);
        const stream = new ReadableStream({
            start(controller) {
                controller.enqueue(binaryData);
                controller.close();
            }
        });
        const result = await readStreamAsUint8Array(stream);
        expect(result).toEqual(binaryData);
    });
});