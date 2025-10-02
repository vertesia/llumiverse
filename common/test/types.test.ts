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

        test('should strip markdown code blocks with json language identifier', () => {
            const results: CompletionResult[] = [
                { type: "text", value: '```json\n{"guideline_id": "123", "status": "REUSED_EXISTING"}\n```' }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual({ guideline_id: "123", status: "REUSED_EXISTING" });
        });

        test('should strip markdown code blocks without language identifier', () => {
            const results: CompletionResult[] = [
                { type: "text", value: '```\n{"key": "value"}\n```' }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual({ key: "value" });
        });

        test('should strip markdown code blocks with javascript language identifier', () => {
            const results: CompletionResult[] = [
                { type: "text", value: '```javascript\n{"foo": "bar"}\n```' }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual({ foo: "bar" });
        });

        test('should strip markdown code blocks with typescript language identifier', () => {
            const results: CompletionResult[] = [
                { type: "text", value: '```typescript\n{"foo": "bar"}\n```' }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual({ foo: "bar" });
        });

        test('should handle complex JSON wrapped in markdown code blocks', () => {
            const jsonContent = {
                guideline_id: "68dd9ef4e52e9b18c9d87b3e",
                summary: {
                    status: "REUSED_EXISTING",
                    title: "HTS Classification Guidelines",
                    applicable_chapters: ["57", "63", "44"],
                    coverage: {
                        products_analyzed: 622,
                        confidence_level: "MEDIUM"
                    }
                }
            };
            const results: CompletionResult[] = [
                { type: "text", value: `\`\`\`json\n${JSON.stringify(jsonContent, null, 2)}\n\`\`\`` }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual(jsonContent);
        });

        test('should handle markdown code blocks with extra whitespace', () => {
            const results: CompletionResult[] = [
                { type: "text", value: '```json\n\n{"key": "value"}\n\n```' }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual({ key: "value" });
        });

        test('should extract JSON from text with preceding chatter', () => {
            const results: CompletionResult[] = [
                { type: "text", value: 'Here is the result: {"status": "success", "code": 200}' }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual({ status: "success", code: 200 });
        });

        test('should extract JSON from text with markdown code block and chatter', () => {
            const results: CompletionResult[] = [
                { type: "text", value: 'Sure, here you go:\n```json\n{"key": "value"}\n```\nHope this helps!' }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual({ key: "value" });
        });

        test('should extract JSON array from text with chatter', () => {
            const results: CompletionResult[] = [
                { type: "text", value: 'The items are: [{"id": 1}, {"id": 2}]' }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual([{ id: 1 }, { id: 2 }]);
        });

        test('should handle nested objects with braces in strings', () => {
            const results: CompletionResult[] = [
                { type: "text", value: 'Result: {"message": "Format: {value}", "data": {"nested": true}}' }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual({ message: "Format: {value}", data: { nested: true } });
        });

        test('should extract JSON with escaped quotes', () => {
            const results: CompletionResult[] = [
                { type: "text", value: 'Output: {"text": "She said \\"hello\\"", "count": 1}' }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual({ text: 'She said "hello"', count: 1 });
        });

        test('should prioritize first JSON object when multiple exist', () => {
            const results: CompletionResult[] = [
                { type: "text", value: 'First: {"a": 1} and Second: {"b": 2}' }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toEqual({ a: 1 });
        });

        test('should handle real-world example from tariff workflow', () => {
            const realWorldOutput = '```json\n{\n  "guideline_id": "68dd9ef4e52e9b18c9d87b3e",\n  "summary": {\n    "guideline_id": "68dd9ef4e52e9b18c9d87b3e",\n    "status": "REUSED_EXISTING",\n    "title": "HTS Classification Guidelines - Multi-Category Goods from IN"\n  }\n}\n```';
            const results: CompletionResult[] = [
                { type: "text", value: realWorldOutput }
            ];
            const parsed = parseCompletionResultsToJson(results);
            expect(parsed).toHaveProperty('guideline_id', '68dd9ef4e52e9b18c9d87b3e');
            expect(parsed).toHaveProperty('summary');
            expect(parsed.summary).toHaveProperty('status', 'REUSED_EXISTING');
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
