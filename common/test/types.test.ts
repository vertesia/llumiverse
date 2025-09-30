import { describe, expect, test } from "vitest";
import {
    completionResultToString,
    parseCompletionResultsToJson,
    parseCompletionResults,
    type CompletionResult,
    type TextResult,
    type JsonResult,
    type ImageResult
} from "../src/types";

describe('Completion Result Utilities', () => {
    describe('completionResultToString', () => {
        test('should convert text result to string', () => {
            const result: TextResult = { type: "text", value: "Hello World" };
            expect(completionResultToString(result)).toBe("Hello World");
        });

        test('should convert json result to formatted string', () => {
            const result: JsonResult = { type: "json", value: { foo: "bar", num: 42 } };
            const output = completionResultToString(result);
            expect(output).toContain('"foo": "bar"');
            expect(output).toContain('"num": 42');
        });

        test('should convert image result to string', () => {
            const result: ImageResult = { type: "image", value: "data:image/png;base64,..." };
            expect(completionResultToString(result)).toBe("data:image/png;base64,...");
        });
    });

    describe('parseCompletionResultsToJson', () => {
        test('should parse JSON type result', () => {
            const results: CompletionResult[] = [
                { type: "json", value: { products: ["item1", "item2"], count: 2 } }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual({ products: ["item1", "item2"], count: 2 });
        });

        test('should parse first JSON result when multiple exist', () => {
            const results: CompletionResult[] = [
                { type: "json", value: { first: true } },
                { type: "json", value: { second: true } }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual({ first: true });
        });

        test('should parse text result as JSON', () => {
            const results: CompletionResult[] = [
                { type: "text", value: '{"products": ["item1", "item2"], "count": 2}' }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual({ products: ["item1", "item2"], count: 2 });
        });

        test('should concatenate multiple text results before parsing', () => {
            const results: CompletionResult[] = [
                { type: "text", value: '{"products": [' },
                { type: "text", value: '"item1", "item2"' },
                { type: "text", value: '], "count": 2}' }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual({ products: ["item1", "item2"], count: 2 });
        });

        test('should throw error when no results found', () => {
            const results: CompletionResult[] = [];
            expect(() => parseCompletionResultsToJson(results)).toThrow("No JSON result found or failed to parse text");
        });

        test('should throw error when no text results found', () => {
            const results: CompletionResult[] = [
                { type: "image", value: "data:image/png;base64,..." }
            ];
            expect(() => parseCompletionResultsToJson(results)).toThrow("No JSON result found or failed to parse text");
        });

        test('should throw error when text is not valid JSON', () => {
            const results: CompletionResult[] = [
                { type: "text", value: "This is not JSON" }
            ];
            expect(() => parseCompletionResultsToJson(results)).toThrow("No JSON result found or failed to parse text");
        });

        test('should handle complex nested JSON', () => {
            const complexJson = {
                sku: "38603",
                country: "CN",
                description: "Chain support disc",
                hs_code: "8466.93.9885",
                confidence: 0.9,
                nested: {
                    array: [1, 2, 3],
                    object: { key: "value" }
                }
            };
            const results: CompletionResult[] = [
                { type: "text", value: JSON.stringify(complexJson) }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual(complexJson);
        });

        test('should handle text result with surrounding whitespace', () => {
            const results: CompletionResult[] = [
                { type: "text", value: '  \n{"key": "value"}\n  ' }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual({ key: "value" });
        });
    });

    describe('parseCompletionResults', () => {
        test('should parse JSON when available', () => {
            const results: CompletionResult[] = [
                { type: "json", value: { foo: "bar" } }
            ];
            const parsed = parseCompletionResults(results);
            expect(parsed).toEqual({ foo: "bar" });
        });

        test('should parse text as JSON when possible', () => {
            const results: CompletionResult[] = [
                { type: "text", value: '{"foo": "bar"}' }
            ];
            const parsed = parseCompletionResults(results);
            expect(parsed).toEqual({ foo: "bar" });
        });

        test('should fallback to concatenated text when JSON parsing fails', () => {
            const results: CompletionResult[] = [
                { type: "text", value: "Hello" },
                { type: "text", value: "World" }
            ];
            const parsed = parseCompletionResults(results);
            expect(parsed).toBe("HelloWorld");
        });

        test('should use custom separator for text concatenation', () => {
            const results: CompletionResult[] = [
                { type: "text", value: "Hello" },
                { type: "text", value: "World" }
            ];
            const parsed = parseCompletionResults(results, " ");
            expect(parsed).toBe("Hello World");
        });

        test('should handle mixed result types', () => {
            const results: CompletionResult[] = [
                { type: "text", value: "prefix" },
                { type: "image", value: "data:image/png;base64,..." },
                { type: "text", value: "suffix" }
            ];
            const parsed = parseCompletionResults(results, " ");
            expect(parsed).toBe("prefix data:image/png;base64,... suffix");
        });

        test('should handle empty results', () => {
            const results: CompletionResult[] = [];
            const parsed = parseCompletionResults(results);
            expect(parsed).toBe("");
        });

        test('should prefer JSON over text even with separator', () => {
            const results: CompletionResult[] = [
                { type: "text", value: "ignored" },
                { type: "json", value: { key: "value" } }
            ];
            const parsed = parseCompletionResults(results, " ");
            expect(parsed).toEqual({ key: "value" });
        });
    });
});
