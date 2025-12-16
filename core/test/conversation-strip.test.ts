import { describe, expect, test } from "vitest";
import { stripBinaryFromConversation, stripBase64ImagesFromConversation } from "../src/conversation-utils";

const IMAGE_PLACEHOLDER = '[Image removed from conversation history]';
const DOCUMENT_PLACEHOLDER = '[Document removed from conversation history]';
const VIDEO_PLACEHOLDER = '[Video removed from conversation history]';

describe('stripBinaryFromConversation', () => {
    test('should replace Bedrock image block with text block', () => {
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
        // Image block should be replaced with text block
        expect(result.messages[0].content[0].toolResult.content[1]).toEqual({ text: IMAGE_PLACEHOLDER });
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

    test('should handle raw Uint8Array in array', () => {
        const input = [
            { text: 'normal' },
            new Uint8Array([4, 5, 6])
        ];

        const result = stripBinaryFromConversation(input) as any[];

        expect(result[0].text).toBe('normal');
        expect(result[1]).toBe(IMAGE_PLACEHOLDER);
    });

    test('should handle empty objects and arrays', () => {
        expect(stripBinaryFromConversation({})).toEqual({});
        expect(stripBinaryFromConversation([])).toEqual([]);
    });

    test('should replace Bedrock document block with text block', () => {
        const input = [{
            document: {
                format: 'pdf',
                name: 'report.pdf',
                source: { bytes: new Uint8Array([0x25, 0x50, 0x44, 0x46]) } // %PDF
            }
        }];

        const result = stripBinaryFromConversation(input) as any[];
        expect(result[0]).toEqual({ text: DOCUMENT_PLACEHOLDER });
    });

    test('should replace Bedrock video block with text block', () => {
        const input = [{
            video: {
                format: 'mp4',
                source: { bytes: new Uint8Array([0x00, 0x00, 0x00, 0x20]) }
            }
        }];

        const result = stripBinaryFromConversation(input) as any[];
        expect(result[0]).toEqual({ text: VIDEO_PLACEHOLDER });
    });

    test('should handle multiple image blocks in content array', () => {
        const input = {
            content: [
                { text: 'First' },
                { image: { format: 'png', source: { bytes: new Uint8Array([1]) } } },
                { text: 'Middle' },
                { image: { format: 'jpg', source: { bytes: new Uint8Array([2]) } } },
                { text: 'Last' }
            ]
        };

        const result = stripBinaryFromConversation(input) as any;

        expect(result.content[0]).toEqual({ text: 'First' });
        expect(result.content[1]).toEqual({ text: IMAGE_PLACEHOLDER });
        expect(result.content[2]).toEqual({ text: 'Middle' });
        expect(result.content[3]).toEqual({ text: IMAGE_PLACEHOLDER });
        expect(result.content[4]).toEqual({ text: 'Last' });
    });
});

describe('stripBase64ImagesFromConversation', () => {
    test('should replace OpenAI image_url block with text block', () => {
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
        // Should be replaced with text block
        expect(result.messages[0].content[0]).toEqual({ type: 'text', text: IMAGE_PLACEHOLDER });
    });

    test('should replace Gemini inlineData block with text block', () => {
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
        // Should be replaced with text block
        expect(result.parts[0]).toEqual({ text: IMAGE_PLACEHOLDER });
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

    test('should preserve http URLs in image_url blocks', () => {
        const input = {
            messages: [{
                content: [{
                    type: 'image_url',
                    image_url: { url: 'https://example.com/image.jpg' }
                }]
            }]
        };

        const result = stripBase64ImagesFromConversation(input) as any;
        // Should NOT be stripped - only base64 data URLs are stripped
        expect(result.messages[0].content[0].image_url.url).toBe('https://example.com/image.jpg');
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

    test('should handle mixed content with images and text', () => {
        const input = {
            content: [
                { type: 'text', text: 'Hello' },
                { type: 'image_url', image_url: { url: 'data:image/png;base64,abc123' } },
                { type: 'text', text: 'World' }
            ]
        };

        const result = stripBase64ImagesFromConversation(input) as any;

        expect(result.content[0]).toEqual({ type: 'text', text: 'Hello' });
        expect(result.content[1]).toEqual({ type: 'text', text: IMAGE_PLACEHOLDER });
        expect(result.content[2]).toEqual({ type: 'text', text: 'World' });
    });
});
