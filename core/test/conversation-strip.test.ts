import { describe, expect, test } from "vitest";
import {
    stripBinaryFromConversation,
    stripBase64ImagesFromConversation,
    stripHeartbeatsFromConversation,
    truncateLargeTextInConversation,
    getConversationMeta,
    setConversationMeta,
    incrementConversationTurn,
    deserializeBinaryFromStorage
} from "../src/conversation-utils";

const IMAGE_PLACEHOLDER = '[Image removed from conversation history]';
const DOCUMENT_PLACEHOLDER = '[Document removed from conversation history]';
const VIDEO_PLACEHOLDER = '[Video removed from conversation history]';

const FORCE_STRIP = { keepForTurns: 0, currentTurn: 0 };

describe('stripBinaryFromConversation', () => {
    test('should replace Bedrock image block with text block when force stripping', () => {
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

        const result = stripBinaryFromConversation(input, FORCE_STRIP) as any;

        expect(result.messages[0].content[0].toolResult.content[0].text).toBe('Image loaded');
        // Image block should be replaced with text block
        expect(result.messages[0].content[0].toolResult.content[1]).toEqual({ text: IMAGE_PLACEHOLDER });
    });

    test('should serialize (not strip) by default when no options provided', () => {
        const input = {
            messages: [{
                content: [{
                    image: {
                        format: 'png',
                        source: { bytes: new Uint8Array([137, 80, 78, 71]) }
                    }
                }]
            }]
        };

        // Default keepForTurns=Infinity means serialize for safe JSON storage
        const result = stripBinaryFromConversation(input) as any;
        expect(result.messages[0].content[0].image.source.bytes._base64).toBeDefined();
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

        const result = stripBinaryFromConversation(input, FORCE_STRIP) as any[];

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

        const result = stripBinaryFromConversation(input, FORCE_STRIP) as any[];
        expect(result[0]).toEqual({ text: DOCUMENT_PLACEHOLDER });
    });

    test('should replace Bedrock video block with text block', () => {
        const input = [{
            video: {
                format: 'mp4',
                source: { bytes: new Uint8Array([0x00, 0x00, 0x00, 0x20]) }
            }
        }];

        const result = stripBinaryFromConversation(input, FORCE_STRIP) as any[];
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

        const result = stripBinaryFromConversation(input, FORCE_STRIP) as any;

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

        const result = stripBase64ImagesFromConversation(input, FORCE_STRIP) as any;
        // Should be replaced with text block
        expect(result.messages[0].content[0]).toEqual({ type: 'text', text: IMAGE_PLACEHOLDER });
    });

    test('should be a no-op by default when no options provided', () => {
        const input = {
            messages: [{
                content: [{
                    type: 'image_url',
                    image_url: { url: 'data:image/png;base64,abc123' }
                }]
            }]
        };

        // Default keepForTurns=Infinity means don't strip
        const result = stripBase64ImagesFromConversation(input) as any;
        expect(result.messages[0].content[0].image_url.url).toBe('data:image/png;base64,abc123');
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

        const result = stripBase64ImagesFromConversation(input, FORCE_STRIP) as any;
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

        const result = stripBase64ImagesFromConversation(input, FORCE_STRIP) as any;
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

        const result = stripBase64ImagesFromConversation(input, FORCE_STRIP) as any;
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

        const result = stripBase64ImagesFromConversation(input, FORCE_STRIP) as any;
        expect(result.url).toBe('data:text/plain;base64,SGVsbG8gV29ybGQ=');
    });

    test('should replace Anthropic base64 image block with text block', () => {
        const input = {
            messages: [{
                role: 'user',
                content: [
                    { type: 'text', text: 'Here is an image' },
                    {
                        type: 'image',
                        source: {
                            type: 'base64',
                            data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk',
                            media_type: 'image/png'
                        }
                    }
                ]
            }]
        };

        const result = stripBase64ImagesFromConversation(input, FORCE_STRIP) as any;

        expect(result.messages[0].content[0]).toEqual({ type: 'text', text: 'Here is an image' });
        expect(result.messages[0].content[1]).toEqual({ type: 'text', text: IMAGE_PLACEHOLDER });
    });

    test('should replace Anthropic base64 document block with text block', () => {
        const input = {
            messages: [{
                role: 'user',
                content: [
                    { type: 'text', text: 'Here is a document' },
                    {
                        type: 'document',
                        source: {
                            type: 'base64',
                            data: 'JVBERi0xLjQKJeLjz9MKMSAwIG9iago=',
                            media_type: 'application/pdf'
                        }
                    }
                ]
            }]
        };

        const result = stripBase64ImagesFromConversation(input, FORCE_STRIP) as any;

        expect(result.messages[0].content[0]).toEqual({ type: 'text', text: 'Here is a document' });
        expect(result.messages[0].content[1]).toEqual({ type: 'text', text: DOCUMENT_PLACEHOLDER });
    });

    test('should not strip Anthropic image block with URL source', () => {
        const input = {
            messages: [{
                role: 'user',
                content: [{
                    type: 'image',
                    source: {
                        type: 'url',
                        url: 'https://example.com/image.png'
                    }
                }]
            }]
        };

        const result = stripBase64ImagesFromConversation(input, FORCE_STRIP) as any;

        // URL-based images should not be stripped
        expect(result.messages[0].content[0].source.url).toBe('https://example.com/image.png');
    });

    test('should handle mixed content with images and text', () => {
        const input = {
            content: [
                { type: 'text', text: 'Hello' },
                { type: 'image_url', image_url: { url: 'data:image/png;base64,abc123' } },
                { type: 'text', text: 'World' }
            ]
        };

        const result = stripBase64ImagesFromConversation(input, FORCE_STRIP) as any;

        expect(result.content[0]).toEqual({ type: 'text', text: 'Hello' });
        expect(result.content[1]).toEqual({ type: 'text', text: IMAGE_PLACEHOLDER });
        expect(result.content[2]).toEqual({ type: 'text', text: 'World' });
    });
});

describe('turn-based stripping', () => {
    describe('conversation metadata', () => {
        test('should return default metadata for objects without metadata', () => {
            const conversation = { messages: [] };
            const meta = getConversationMeta(conversation);
            expect(meta.turnNumber).toBe(0);
        });

        test('should return default metadata for null/undefined', () => {
            expect(getConversationMeta(null)).toEqual({ turnNumber: 0 });
            expect(getConversationMeta(undefined)).toEqual({ turnNumber: 0 });
        });

        test('should set and get metadata correctly', () => {
            const conversation = { messages: [] };
            const updated = setConversationMeta(conversation, { turnNumber: 5 });
            const meta = getConversationMeta(updated);
            expect(meta.turnNumber).toBe(5);
        });

        test('should increment turn number', () => {
            const conversation = { messages: [] };
            const turn1 = incrementConversationTurn(conversation);
            expect(getConversationMeta(turn1).turnNumber).toBe(1);

            const turn2 = incrementConversationTurn(turn1);
            expect(getConversationMeta(turn2).turnNumber).toBe(2);

            const turn3 = incrementConversationTurn(turn2);
            expect(getConversationMeta(turn3).turnNumber).toBe(3);
        });
    });

    describe('keepForTurns option', () => {
        test('should serialize binary data when keepForTurns > currentTurn', () => {
            const input = {
                content: [{
                    image: {
                        format: 'png',
                        source: { bytes: new Uint8Array([137, 80, 78, 71]) }
                    }
                }]
            };

            // Keep for 3 turns, current turn is 1 - should serialize not strip
            const result = stripBinaryFromConversation(input, {
                keepForTurns: 3,
                currentTurn: 1
            }) as any;

            // Should have serialized format (bytes._base64), not placeholder
            expect(result.content[0].image.source.bytes._base64).toBeDefined();
            expect(result.content[0].image.source.bytes._base64).toBe('iVBORw=='); // Base64 of [137, 80, 78, 71]
        });

        test('should strip when currentTurn >= keepForTurns', () => {
            const input = {
                content: [{
                    image: {
                        format: 'png',
                        source: { bytes: new Uint8Array([137, 80, 78, 71]) }
                    }
                }]
            };

            // Keep for 2 turns, current turn is 2 - should strip
            const result = stripBinaryFromConversation(input, {
                keepForTurns: 2,
                currentTurn: 2
            }) as any;

            expect(result.content[0]).toEqual({ text: IMAGE_PLACEHOLDER });
        });

        test('should strip when currentTurn > keepForTurns', () => {
            const input = {
                content: [{
                    image: {
                        format: 'png',
                        source: { bytes: new Uint8Array([137, 80, 78, 71]) }
                    }
                }]
            };

            // Keep for 2 turns, current turn is 5 - should strip
            const result = stripBinaryFromConversation(input, {
                keepForTurns: 2,
                currentTurn: 5
            }) as any;

            expect(result.content[0]).toEqual({ text: IMAGE_PLACEHOLDER });
        });

        test('should strip immediately when keepForTurns is 0', () => {
            const input = {
                content: [{
                    image: {
                        format: 'png',
                        source: { bytes: new Uint8Array([137, 80, 78, 71]) }
                    }
                }]
            };

            const result = stripBinaryFromConversation(input, {
                keepForTurns: 0,
                currentTurn: 0
            }) as any;

            expect(result.content[0]).toEqual({ text: IMAGE_PLACEHOLDER });
        });

        test('should use turn number from metadata when currentTurn not provided', () => {
            let conversation: any = {
                content: [{
                    image: {
                        format: 'png',
                        source: { bytes: new Uint8Array([137, 80, 78, 71]) }
                    }
                }]
            };

            // Set turn number to 1 in metadata
            conversation = setConversationMeta(conversation, { turnNumber: 1 });

            // Keep for 3 turns - should serialize (turn 1 < 3)
            const result = stripBinaryFromConversation(conversation, {
                keepForTurns: 3
            }) as any;

            expect(result.content[0].image.source.bytes._base64).toBeDefined();
        });
    });

    describe('serialization and deserialization', () => {
        test('should serialize Uint8Array to base64 for safe storage', () => {
            const originalBytes = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
            const input = {
                content: [{
                    image: {
                        format: 'png',
                        source: { bytes: originalBytes }
                    }
                }]
            };

            // Serialize for storage
            const serialized = stripBinaryFromConversation(input, {
                keepForTurns: 5,
                currentTurn: 1
            }) as any;

            // Should be base64 encoded (bytes becomes { _base64: ... })
            expect(serialized.content[0].image.source.bytes._base64).toBeDefined();

            // Should survive JSON.stringify/parse
            const jsonString = JSON.stringify(serialized);
            const parsed = JSON.parse(jsonString);
            expect(parsed.content[0].image.source.bytes._base64).toBe(serialized.content[0].image.source.bytes._base64);

            // Should be deserializable back to Uint8Array
            const deserialized = deserializeBinaryFromStorage(parsed) as any;
            expect(deserialized.content[0].image.source.bytes).toBeInstanceOf(Uint8Array);
            expect(Array.from(deserialized.content[0].image.source.bytes)).toEqual(Array.from(originalBytes));
        });

        test('should strip serialized images when threshold exceeded', () => {
            // Start with serialized data (as if it was stored and restored)
            // The serialized format has bytes: { _base64: '...' }
            const serializedConversation = {
                content: [{
                    image: {
                        format: 'png',
                        source: { bytes: { _base64: 'iVBORw0KGgo=' } }
                    }
                }],
                _llumiverse_meta: { turnNumber: 3 }
            };

            // Now strip with threshold of 2 turns (current turn 3 >= 2)
            const result = stripBinaryFromConversation(serializedConversation, {
                keepForTurns: 2,
                currentTurn: 3
            }) as any;

            // Should be stripped to placeholder
            expect(result.content[0]).toEqual({ text: IMAGE_PLACEHOLDER });
        });
    });

    describe('base64 images turn-based stripping', () => {
        test('should keep base64 images when keepForTurns > currentTurn', () => {
            const input = {
                content: [{
                    type: 'image_url',
                    image_url: { url: 'data:image/png;base64,abc123' }
                }]
            };

            // Keep for 3 turns, current turn is 1 - should keep
            const result = stripBase64ImagesFromConversation(input, {
                keepForTurns: 3,
                currentTurn: 1
            }) as any;

            // Should still have the image URL
            expect(result.content[0].image_url.url).toBe('data:image/png;base64,abc123');
        });

        test('should strip base64 images when currentTurn >= keepForTurns', () => {
            const input = {
                content: [{
                    type: 'image_url',
                    image_url: { url: 'data:image/png;base64,abc123' }
                }]
            };

            // Keep for 2 turns, current turn is 3 - should strip
            const result = stripBase64ImagesFromConversation(input, {
                keepForTurns: 2,
                currentTurn: 3
            }) as any;

            expect(result.content[0]).toEqual({ type: 'text', text: IMAGE_PLACEHOLDER });
        });
    });
});

const TEXT_TRUNCATED_MARKER = '\n\n[Content truncated - exceeded token limit]';

describe('truncateLargeTextInConversation', () => {
    test('should not truncate when textMaxTokens is not set', () => {
        const input = {
            content: [{ text: 'a'.repeat(100000) }]
        };

        const result = truncateLargeTextInConversation(input) as any;

        expect(result.content[0].text.length).toBe(100000);
    });

    test('should not truncate when textMaxTokens is 0', () => {
        const input = {
            content: [{ text: 'a'.repeat(100000) }]
        };

        const result = truncateLargeTextInConversation(input, { textMaxTokens: 0 }) as any;

        expect(result.content[0].text.length).toBe(100000);
    });

    test('should truncate text exceeding token limit', () => {
        // 10000 tokens = ~40000 chars
        const longText = 'a'.repeat(50000);
        const input = {
            content: [{ text: longText }]
        };

        const result = truncateLargeTextInConversation(input, { textMaxTokens: 10000 }) as any;

        // Should be truncated to ~40000 chars + marker
        expect(result.content[0].text.length).toBeLessThan(50000);
        expect(result.content[0].text).toContain(TEXT_TRUNCATED_MARKER);
        expect(result.content[0].text.substring(0, 40000)).toBe('a'.repeat(40000));
    });

    test('should not truncate text within token limit', () => {
        const shortText = 'Hello world';
        const input = {
            content: [{ text: shortText }]
        };

        const result = truncateLargeTextInConversation(input, { textMaxTokens: 10000 }) as any;

        expect(result.content[0].text).toBe(shortText);
    });

    test('should truncate nested text in Bedrock tool results', () => {
        const longText = 'x'.repeat(50000);
        const input = {
            messages: [{
                content: [{
                    toolResult: {
                        toolUseId: 'abc123',
                        content: [{
                            text: longText
                        }]
                    }
                }]
            }]
        };

        const result = truncateLargeTextInConversation(input, { textMaxTokens: 10000 }) as any;

        const resultText = result.messages[0].content[0].toolResult.content[0].text;
        expect(resultText.length).toBeLessThan(50000);
        expect(resultText).toContain(TEXT_TRUNCATED_MARKER);
    });

    test('should truncate OpenAI tool message content string', () => {
        const longText = 'y'.repeat(50000);
        const input = {
            messages: [{
                role: 'tool',
                content: longText,
                tool_call_id: 'call_123'
            }]
        };

        const result = truncateLargeTextInConversation(input, { textMaxTokens: 10000 }) as any;

        expect(result.messages[0].content.length).toBeLessThan(50000);
        expect(result.messages[0].content).toContain(TEXT_TRUNCATED_MARKER);
    });

    test('should preserve conversation metadata during truncation', () => {
        const longText = 'z'.repeat(50000);
        const input = {
            content: [{ text: longText }],
            _llumiverse_meta: { turnNumber: 5 }
        };

        const result = truncateLargeTextInConversation(input, { textMaxTokens: 10000 }) as any;

        expect(result._llumiverse_meta).toEqual({ turnNumber: 5 });
        expect(result.content[0].text).toContain(TEXT_TRUNCATED_MARKER);
    });

    test('should handle arrays with mixed short and long text', () => {
        const input = {
            items: [
                { text: 'short' },
                { text: 'w'.repeat(50000) },
                { text: 'also short' }
            ]
        };

        const result = truncateLargeTextInConversation(input, { textMaxTokens: 10000 }) as any;

        expect(result.items[0].text).toBe('short');
        expect(result.items[1].text).toContain(TEXT_TRUNCATED_MARKER);
        expect(result.items[2].text).toBe('also short');
    });

    test('should handle null and undefined', () => {
        expect(truncateLargeTextInConversation(null, { textMaxTokens: 100 })).toBeNull();
        expect(truncateLargeTextInConversation(undefined, { textMaxTokens: 100 })).toBeUndefined();
    });
});

describe('conversation serialization safety', () => {
    test('conversation with binary data should survive JSON roundtrip when serialized', () => {
        const originalConversation = {
            messages: [{
                content: [{
                    image: {
                        format: 'png',
                        source: { bytes: new Uint8Array([137, 80, 78, 71]) }
                    }
                }]
            }]
        };

        // Serialize for storage (keeping for 5 turns)
        const serialized = stripBinaryFromConversation(originalConversation, {
            keepForTurns: 5,
            currentTurn: 1
        });

        // Simulate storage: JSON stringify and parse
        const stored = JSON.stringify(serialized);
        const restored = JSON.parse(stored);

        // The restored conversation should NOT have corrupted numeric keys
        // This was the original bug we fixed
        // The bytes field should be { _base64: ... }, not { "0": 137, "1": 80, ... }
        const bytesObj = restored.messages[0].content[0].image.source.bytes;
        const keys = Object.keys(bytesObj);
        expect(keys).not.toContain('0');
        expect(keys).not.toContain('1');
        expect(keys).toContain('_base64');
    });

    test('conversation with stripped data should survive JSON roundtrip', () => {
        const originalConversation = {
            messages: [{
                content: [{
                    image: {
                        format: 'png',
                        source: { bytes: new Uint8Array([137, 80, 78, 71]) }
                    }
                }]
            }]
        };

        // Strip immediately (explicit keepForTurns: 0)
        const stripped = stripBinaryFromConversation(originalConversation, FORCE_STRIP);

        // Simulate storage: JSON stringify and parse
        const stored = JSON.stringify(stripped);
        const restored = JSON.parse(stored);

        // Should have text placeholder
        expect(restored.messages[0].content[0]).toEqual({ text: IMAGE_PLACEHOLDER });
    });

    test('metadata should be preserved through serialization and stripping', () => {
        let conversation: any = { messages: [] };
        conversation = incrementConversationTurn(conversation);
        conversation = incrementConversationTurn(conversation);

        expect(getConversationMeta(conversation).turnNumber).toBe(2);

        // Serialize for storage
        const serialized = stripBinaryFromConversation(conversation, {
            keepForTurns: 5,
            currentTurn: 2
        });

        // JSON roundtrip
        const restored = JSON.parse(JSON.stringify(serialized));

        // Metadata should be preserved
        expect(getConversationMeta(restored).turnNumber).toBe(2);
    });
});

const HEARTBEAT_PLACEHOLDER = '[Heartbeat removed from conversation history]';
const HEARTBEAT_MSG = '<heartbeat>[System] Workstream status heartbeat:\n- search_engine: running (web_search) — 45s elapsed, 255s remaining\n\nYou may steer, cancel, or launch additional workstreams. If no action needed, simply acknowledge.</heartbeat>';

describe('stripHeartbeatsFromConversation', () => {
    const FORCE_STRIP_HB = { keepForTurns: 0, currentTurn: 0 };

    test('should strip heartbeat-tagged strings when force stripping', () => {
        const input = {
            messages: [{
                role: 'user',
                content: [{ type: 'text', text: HEARTBEAT_MSG }]
            }]
        };

        const result = stripHeartbeatsFromConversation(input, FORCE_STRIP_HB) as any;
        expect(result.messages[0].content[0].text).toBe(HEARTBEAT_PLACEHOLDER);
    });

    test('should preserve heartbeats within keepForTurns threshold', () => {
        const input = {
            messages: [{
                role: 'user',
                content: [{ type: 'text', text: HEARTBEAT_MSG }]
            }]
        };

        // Keep for 3 turns, current turn is 1 - should preserve
        const result = stripHeartbeatsFromConversation(input, {
            keepForTurns: 3,
            currentTurn: 1,
        }) as any;

        expect(result.messages[0].content[0].text).toBe(HEARTBEAT_MSG);
    });

    test('should strip when currentTurn >= keepForTurns', () => {
        const input = {
            messages: [{
                role: 'user',
                content: [{ type: 'text', text: HEARTBEAT_MSG }]
            }]
        };

        const result = stripHeartbeatsFromConversation(input, {
            keepForTurns: 1,
            currentTurn: 2,
        }) as any;

        expect(result.messages[0].content[0].text).toBe(HEARTBEAT_PLACEHOLDER);
    });

    test('should strip with default keepForTurns (1) when currentTurn >= 1', () => {
        const input = {
            content: [{ text: HEARTBEAT_MSG }]
        };

        // Default keepForTurns is 1, currentTurn 1 means strip
        const result = stripHeartbeatsFromConversation(input, { currentTurn: 1 }) as any;
        expect(result.content[0].text).toBe(HEARTBEAT_PLACEHOLDER);
    });

    test('should not strip with default keepForTurns (1) when currentTurn is 0', () => {
        const input = {
            content: [{ text: HEARTBEAT_MSG }]
        };

        // Default keepForTurns is 1, currentTurn 0 means keep
        const result = stripHeartbeatsFromConversation(input, { currentTurn: 0 }) as any;
        expect(result.content[0].text).toBe(HEARTBEAT_MSG);
    });

    test('should never strip when keepForTurns is Infinity', () => {
        const input = {
            content: [{ text: HEARTBEAT_MSG }]
        };

        const result = stripHeartbeatsFromConversation(input, {
            keepForTurns: Infinity,
            currentTurn: 100,
        }) as any;

        expect(result.content[0].text).toBe(HEARTBEAT_MSG);
    });

    test('should preserve non-heartbeat strings', () => {
        const input = {
            messages: [
                { role: 'user', content: [{ type: 'text', text: 'Hello, how are you?' }] },
                { role: 'user', content: [{ type: 'text', text: '[System] Some other system message' }] },
                { role: 'assistant', content: [{ type: 'text', text: 'I am doing well.' }] },
            ]
        };

        const result = stripHeartbeatsFromConversation(input, FORCE_STRIP_HB) as any;

        expect(result.messages[0].content[0].text).toBe('Hello, how are you?');
        expect(result.messages[1].content[0].text).toBe('[System] Some other system message');
        expect(result.messages[2].content[0].text).toBe('I am doing well.');
    });

    test('should handle OpenAI conversation format', () => {
        const input = [
            { role: 'user', content: [{ type: 'text', text: HEARTBEAT_MSG }] },
            { role: 'assistant', content: [{ type: 'text', text: 'Acknowledged.' }] },
        ];

        const result = stripHeartbeatsFromConversation(input, FORCE_STRIP_HB) as any[];

        expect(result[0].content[0].text).toBe(HEARTBEAT_PLACEHOLDER);
        expect(result[1].content[0].text).toBe('Acknowledged.');
    });

    test('should handle Bedrock conversation format', () => {
        const input = {
            messages: [
                { role: 'user', content: [{ text: HEARTBEAT_MSG }] },
                { role: 'assistant', content: [{ text: 'Acknowledged.' }] },
            ]
        };

        const result = stripHeartbeatsFromConversation(input, FORCE_STRIP_HB) as any;

        expect(result.messages[0].content[0].text).toBe(HEARTBEAT_PLACEHOLDER);
        expect(result.messages[1].content[0].text).toBe('Acknowledged.');
    });

    test('should handle Gemini conversation format', () => {
        const input = [
            { role: 'user', parts: [{ text: HEARTBEAT_MSG }] },
            { role: 'model', parts: [{ text: 'Acknowledged.' }] },
        ];

        const result = stripHeartbeatsFromConversation(input, FORCE_STRIP_HB) as any[];

        expect(result[0].parts[0].text).toBe(HEARTBEAT_PLACEHOLDER);
        expect(result[1].parts[0].text).toBe('Acknowledged.');
    });

    test('should handle null and undefined', () => {
        expect(stripHeartbeatsFromConversation(null, FORCE_STRIP_HB)).toBeNull();
        expect(stripHeartbeatsFromConversation(undefined, FORCE_STRIP_HB)).toBeUndefined();
    });

    test('should preserve _llumiverse_meta metadata', () => {
        const input = {
            messages: [{ role: 'user', content: [{ text: HEARTBEAT_MSG }] }],
            _llumiverse_meta: { turnNumber: 5 },
        };

        const result = stripHeartbeatsFromConversation(input, FORCE_STRIP_HB) as any;

        expect(result._llumiverse_meta).toEqual({ turnNumber: 5 });
        expect(result.messages[0].content[0].text).toBe(HEARTBEAT_PLACEHOLDER);
    });

    test('should strip multiple heartbeats while preserving regular messages', () => {
        const input = {
            messages: [
                { role: 'user', content: [{ type: 'text', text: 'Analyze this data' }] },
                { role: 'user', content: [{ type: 'text', text: HEARTBEAT_MSG }] },
                { role: 'assistant', content: [{ type: 'text', text: 'Acknowledged.' }] },
                { role: 'user', content: [{ type: 'text', text: '<heartbeat>[System] Workstream status heartbeat:\n- analyzer: running — 90s elapsed\n\nAcknowledge.</heartbeat>' }] },
                { role: 'assistant', content: [{ type: 'text', text: 'Still waiting.' }] },
                { role: 'assistant', content: [{ type: 'text', text: 'Analysis complete: found 15 results.' }] },
            ]
        };

        const result = stripHeartbeatsFromConversation(input, FORCE_STRIP_HB) as any;

        expect(result.messages[0].content[0].text).toBe('Analyze this data');
        expect(result.messages[1].content[0].text).toBe(HEARTBEAT_PLACEHOLDER);
        expect(result.messages[2].content[0].text).toBe('Acknowledged.');
        expect(result.messages[3].content[0].text).toBe(HEARTBEAT_PLACEHOLDER);
        expect(result.messages[4].content[0].text).toBe('Still waiting.');
        expect(result.messages[5].content[0].text).toBe('Analysis complete: found 15 results.');
    });

    test('should use turn number from metadata when currentTurn not provided', () => {
        let conversation: any = {
            content: [{ text: HEARTBEAT_MSG }]
        };

        // Set turn number to 2 in metadata
        conversation = setConversationMeta(conversation, { turnNumber: 2 });

        // keepForTurns 1, turn 2 from metadata → should strip
        const result = stripHeartbeatsFromConversation(conversation, {
            keepForTurns: 1,
        }) as any;

        expect(result.content[0].text).toBe(HEARTBEAT_PLACEHOLDER);
    });

    test('should only match strings that both start and end with heartbeat tags', () => {
        const input = {
            content: [
                { text: '<heartbeat>partial tag without close' },
                { text: 'no tags at all' },
                { text: 'some text</heartbeat>' },
            ]
        };

        const result = stripHeartbeatsFromConversation(input, FORCE_STRIP_HB) as any;

        // None of these should be stripped - they don't have both opening and closing tags
        expect(result.content[0].text).toBe('<heartbeat>partial tag without close');
        expect(result.content[1].text).toBe('no tags at all');
        expect(result.content[2].text).toBe('some text</heartbeat>');
    });
});
