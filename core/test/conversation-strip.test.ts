import { describe, expect, test } from "vitest";
import { stripBinaryFromConversation, stripBase64ImagesFromConversation } from "../src/conversation-utils";

describe('stripBinaryFromConversation', () => {
    test('should strip Uint8Array from nested object (Bedrock toolResult format)', () => {
        const input = {
            messages: [{
                content: [{
                    toolResult: {
                        toolUseId: 'abc123',
                        content: [{
                            text: 'Image loaded'
                        }, {
                            image: {
                                format: 'png',
                                source: { bytes: new Uint8Array([137, 80, 78, 71]) }
                            }
                        }]
                    }
                }]
            }]
        };

        const result = stripBinaryFromConversation(input) as any;

        expect(result.messages[0].content[0].toolResult.content[0].text).toBe('Image loaded');
        expect(result.messages[0].content[0].toolResult.content[1].image.source.bytes)
            .toBe('[Binary data stripped - use tool to fetch again]');
    });

    test('should preserve structure when no Uint8Array present', () => {
        const input = {
            messages: [{
                role: 'user',
                content: [{ text: 'Hello' }]
            }]
        };

        const result = stripBinaryFromConversation(input);
        expect(result).toEqual(input);
    });

    test('should handle null and undefined', () => {
        expect(stripBinaryFromConversation(null)).toBeNull();
        expect(stripBinaryFromConversation(undefined)).toBeUndefined();
    });

    test('should handle primitive types', () => {
        expect(stripBinaryFromConversation('string')).toBe('string');
        expect(stripBinaryFromConversation(123)).toBe(123);
        expect(stripBinaryFromConversation(true)).toBe(true);
    });

    test('should handle arrays with mixed content', () => {
        const input = [
            { bytes: new Uint8Array([1, 2, 3]) },
            { text: 'normal' },
            new Uint8Array([4, 5, 6])
        ];

        const result = stripBinaryFromConversation(input) as any[];

        expect(result[0].bytes).toBe('[Binary data stripped - use tool to fetch again]');
        expect(result[1].text).toBe('normal');
        expect(result[2]).toBe('[Binary data stripped - use tool to fetch again]');
    });

    test('should handle deeply nested structures', () => {
        const input = {
            a: { b: { c: { d: { bytes: new Uint8Array([1]) } } } }
        };

        const result = stripBinaryFromConversation(input) as any;
        expect(result.a.b.c.d.bytes).toBe('[Binary data stripped - use tool to fetch again]');
    });

    test('should handle empty objects and arrays', () => {
        expect(stripBinaryFromConversation({})).toEqual({});
        expect(stripBinaryFromConversation([])).toEqual([]);
    });

    test('should strip Bedrock document bytes', () => {
        const input = {
            document: {
                format: 'pdf',
                name: 'report.pdf',
                source: { bytes: new Uint8Array([0x25, 0x50, 0x44, 0x46]) } // %PDF
            }
        };

        const result = stripBinaryFromConversation(input) as any;
        expect(result.document.source.bytes).toBe('[Binary data stripped - use tool to fetch again]');
        expect(result.document.format).toBe('pdf');
        expect(result.document.name).toBe('report.pdf');
    });

    test('should strip Bedrock video bytes', () => {
        const input = {
            video: {
                format: 'mp4',
                source: { bytes: new Uint8Array([0x00, 0x00, 0x00, 0x20]) }
            }
        };

        const result = stripBinaryFromConversation(input) as any;
        expect(result.video.source.bytes).toBe('[Binary data stripped - use tool to fetch again]');
        expect(result.video.format).toBe('mp4');
    });
});

describe('stripBase64ImagesFromConversation', () => {
    test('should strip data:image base64 URLs (OpenAI format)', () => {
        const input = {
            messages: [{
                content: [{
                    type: 'image_url',
                    image_url: {
                        url: 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD...'
                    }
                }]
            }]
        };

        const result = stripBase64ImagesFromConversation(input) as any;
        expect(result.messages[0].content[0].image_url.url)
            .toBe('[Image data stripped - use tool to fetch again]');
    });

    test('should strip Gemini inlineData with large base64', () => {
        // Generate a base64 string > 1000 chars
        const largeBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='.repeat(20);
        const input = {
            parts: [{
                inlineData: {
                    mimeType: 'image/png',
                    data: largeBase64
                }
            }]
        };

        const result = stripBase64ImagesFromConversation(input) as any;
        expect(result.parts[0].inlineData.data)
            .toBe('[Image data stripped - use tool to fetch again]');
        expect(result.parts[0].inlineData.mimeType).toBe('image/png');
    });

    test('should preserve short inlineData (not images)', () => {
        const input = {
            parts: [{
                inlineData: {
                    mimeType: 'text/plain',
                    data: 'short'
                }
            }]
        };

        const result = stripBase64ImagesFromConversation(input) as any;
        expect(result.parts[0].inlineData.data).toBe('short');
    });

    test('should preserve http URLs', () => {
        const input = {
            image_url: { url: 'https://example.com/image.jpg' }
        };

        const result = stripBase64ImagesFromConversation(input) as any;
        expect(result.image_url.url).toBe('https://example.com/image.jpg');
    });

    test('should handle null and undefined', () => {
        expect(stripBase64ImagesFromConversation(null)).toBeNull();
        expect(stripBase64ImagesFromConversation(undefined)).toBeUndefined();
    });

    test('should preserve non-image data URLs', () => {
        const input = {
            url: 'data:text/plain;base64,SGVsbG8gV29ybGQ='
        };

        const result = stripBase64ImagesFromConversation(input) as any;
        expect(result.url).toBe('data:text/plain;base64,SGVsbG8gV29ybGQ=');
    });

    test('should handle different image types', () => {
        const formats = ['jpeg', 'png', 'gif', 'webp'];
        for (const format of formats) {
            const input = { url: `data:image/${format};base64,abc123` };
            const result = stripBase64ImagesFromConversation(input) as any;
            expect(result.url).toBe('[Image data stripped - use tool to fetch again]');
        }
    });

    test('should handle nested arrays with base64 images', () => {
        const input = {
            content: [[{
                image_url: { url: 'data:image/png;base64,abc' }
            }]]
        };

        const result = stripBase64ImagesFromConversation(input) as any;
        expect(result.content[0][0].image_url.url)
            .toBe('[Image data stripped - use tool to fetch again]');
    });
});
