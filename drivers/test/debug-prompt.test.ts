import type { ConverseRequest } from '@aws-sdk/client-bedrock-runtime';
import type { Content } from '@google/genai';
import { describe, expect, test } from 'vitest';
import { BedrockDriver } from '../src/bedrock/index.js';
import { GroqDriver } from '../src/groq/index.js';
import { formatOpenAIDebugPrompt } from '../src/openai/openai_format.js';
import { type ClaudePrompt, formatClaudeDebugPrompt } from '../src/shared/claude-messages.js';
import { BINARY_TRUNCATED_MARKER } from '../src/shared/debug-prompt.js';
import type { GenerateContentPrompt } from '../src/vertexai/index.js';
import { formatGeminiDebugPrompt } from '../src/vertexai/models/gemini.js';
import type { ImagenPrompt } from '../src/vertexai/models/imagen.js';
import { formatImagenDebugPrompt } from '../src/vertexai/models/imagen.js';

describe('debug prompt binary truncation', () => {
    test('truncates Claude base64 content blocks', () => {
        const prompt: ClaudePrompt = {
            messages: [
                {
                    role: 'user',
                    content: [
                        {
                            type: 'image',
                            source: {
                                type: 'base64',
                                media_type: 'image/png',
                                data: 'a'.repeat(80),
                            },
                        },
                    ],
                },
            ],
        };

        const result = formatClaudeDebugPrompt(prompt);
        const content = result.messages[0].content;
        const originalContent = prompt.messages[0].content;

        expect(
            Array.isArray(content) && content[0].type === 'image' && content[0].source.type === 'base64'
                ? content[0].source.data
                : undefined,
        ).toBe(`aaaaaaaaaa${BINARY_TRUNCATED_MARKER}aaaaaaaaaa`);
        expect(
            Array.isArray(originalContent) &&
                originalContent[0].type === 'image' &&
                originalContent[0].source.type === 'base64'
                ? originalContent[0].source.data
                : undefined,
        ).toBe('a'.repeat(80));
    });

    test('truncates OpenAI response input image data URLs', () => {
        const prompt: Parameters<typeof formatOpenAIDebugPrompt>[0] = [
            {
                role: 'user',
                content: [
                    {
                        type: 'input_image',
                        image_url: `data:image/png;base64,${'b'.repeat(80)}`,
                        detail: 'auto',
                    },
                ],
            },
        ];

        const result = formatOpenAIDebugPrompt(prompt);
        const item = result[0];

        expect(
            'content' in item && Array.isArray(item.content) && item.content[0].type === 'input_image'
                ? item.content[0].image_url
                : undefined,
        ).toBe(`data:image/png;base64,bbbbbbbbbb${BINARY_TRUNCATED_MARKER}bbbbbbbbbb`);
    });

    test('truncates OpenAI response input file data URLs', () => {
        const prompt: Parameters<typeof formatOpenAIDebugPrompt>[0] = [
            {
                role: 'user',
                content: [
                    {
                        type: 'input_file',
                        filename: 'document.pdf',
                        file_data: `data:application/pdf;base64,${'f'.repeat(80)}`,
                    },
                ],
            },
        ];

        const result = formatOpenAIDebugPrompt(prompt);
        const item = result[0];

        expect(
            'content' in item && Array.isArray(item.content) && item.content[0].type === 'input_file'
                ? item.content[0].file_data
                : undefined,
        ).toBe(`data:application/pdf;base64,ffffffffff${BINARY_TRUNCATED_MARKER}ffffffffff`);
    });

    test('truncates Vertex Gemini inlineData payloads', () => {
        const prompt: GenerateContentPrompt = {
            contents: [
                {
                    role: 'user',
                    parts: [
                        {
                            inlineData: {
                                mimeType: 'image/png',
                                data: 'c'.repeat(80),
                            },
                        },
                    ],
                } satisfies Content,
            ],
        };

        const result = formatGeminiDebugPrompt(prompt);

        expect(result.contents[0].parts?.[0].inlineData?.data).toBe(`cccccccccc${BINARY_TRUNCATED_MARKER}cccccccccc`);
    });

    test('truncates Vertex Imagen reference images', () => {
        const prompt: ImagenPrompt = {
            prompt: 'make an image',
            referenceImages: [
                {
                    referenceType: 'REFERENCE_TYPE_RAW',
                    referenceId: 1,
                    referenceImage: {
                        bytesBase64Encoded: 'd'.repeat(80),
                    },
                },
            ],
        };

        const result = formatImagenDebugPrompt(prompt);

        expect(result.referenceImages?.[0].referenceImage?.bytesBase64Encoded).toBe(
            `dddddddddd${BINARY_TRUNCATED_MARKER}dddddddddd`,
        );
    });

    test('truncates Bedrock Converse byte sources', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const prompt: ConverseRequest = {
            modelId: 'anthropic.claude-3-sonnet',
            messages: [
                {
                    role: 'user',
                    content: [
                        {
                            image: {
                                format: 'png',
                                source: {
                                    bytes: Uint8Array.from({ length: 120 }, (_, i) => i),
                                },
                            },
                        },
                    ],
                },
            ],
        };

        const result = driver.formatDebugPrompt(prompt);
        const bytes: unknown = 'modelId' in result ? result.messages?.[0].content?.[0].image?.source?.bytes : undefined;

        expect(typeof bytes).toBe('string');
        expect(bytes).toContain(BINARY_TRUNCATED_MARKER);
    });

    test('truncates Bedrock Converse tool result byte sources', () => {
        const driver = new BedrockDriver({ region: 'us-east-1' });
        const prompt: ConverseRequest = {
            modelId: 'anthropic.claude-3-sonnet',
            messages: [
                {
                    role: 'user',
                    content: [
                        {
                            toolResult: {
                                toolUseId: 'tool-1',
                                content: [
                                    {
                                        image: {
                                            format: 'png',
                                            source: {
                                                bytes: Uint8Array.from({ length: 120 }, (_, i) => i),
                                            },
                                        },
                                    },
                                ],
                            },
                        },
                    ],
                },
            ],
        };

        const result = driver.formatDebugPrompt(prompt);
        const bytes: unknown =
            'modelId' in result
                ? result.messages?.[0].content?.[0].toolResult?.content?.[0].image?.source?.bytes
                : undefined;

        expect(typeof bytes).toBe('string');
        expect(bytes).toContain(BINARY_TRUNCATED_MARKER);
    });

    test('truncates Groq chat image data URLs', () => {
        const driver = new GroqDriver({ apiKey: 'test' });
        const result = driver.formatDebugPrompt({
            _is_openai_chat_completions: true,
            messages: [
                {
                    role: 'user',
                    content: [
                        {
                            type: 'image_url',
                            image_url: {
                                url: `data:image/jpeg;base64,${'e'.repeat(80)}`,
                            },
                        },
                    ],
                },
            ],
        });

        const content = result.messages[0].content;

        expect(Array.isArray(content) && content[0].type === 'image_url' ? content[0].image_url.url : undefined).toBe(
            `data:image/jpeg;base64,eeeeeeeeee${BINARY_TRUNCATED_MARKER}eeeeeeeeee`,
        );
    });
});
