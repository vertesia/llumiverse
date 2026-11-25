import { describe, expect, it } from 'vitest';
import {
    formatOmittedAttachmentNote,
    isToolResultAttachmentSupported,
    normalizeAttachmentMimeType,
    toolResultAttachmentApi,
} from './tool-result-attachments.js';

describe('normalizeAttachmentMimeType', () => {
    it.each([
        [' IMAGE/PNG ', 'image/png'],
        ['image/jpg', 'image/jpeg'],
        ['Image/JPG', 'image/jpeg'],
        ['text/csv; charset=utf-8', 'text/csv'],
        [undefined, ''],
    ])('normalises %j to %j', (input, expected) => {
        expect(normalizeAttachmentMimeType(input)).toBe(expected);
    });
});

describe('isToolResultAttachmentSupported', () => {
    it.each([
        ['image/png', true],
        ['image/jpg', true],
        ['image/webp', true],
        ['application/pdf', true],
        ['text/csv', true],
        ['application/json', true],
        ['image/bmp', false],
        ['image/svg+xml', false],
        ['audio/mpeg', false],
        ['video/mp4', false],
        ['application/zip', false],
        ['', false],
    ])('Claude: %s → %s', (mimeType, expected) => {
        expect(isToolResultAttachmentSupported('claude', mimeType, 'claude-sonnet-5-5')).toBe(expected);
    });

    it.each([
        ['image/bmp', true],
        ['image/tiff', true],
        ['IMAGE/PNG', true],
        ['video/mp4', true],
        ['application/pdf', true],
        ['text/csv', true],
        ['application/json', true],
        ['audio/mpeg', false],
        ['audio/wav', false],
        ['application/zip', false],
        ['application/octet-stream', false],
    ])('Gemini 3: %s → %s', (mimeType, expected) => {
        expect(isToolResultAttachmentSupported('gemini', mimeType, 'gemini-3.5-flash')).toBe(expected);
    });

    it('reads Gemini 3 rules for later and unversioned Gemini models', () => {
        expect(isToolResultAttachmentSupported('gemini', 'image/png', 'publishers/google/models/gemini-4-pro')).toBe(
            true,
        );
        expect(isToolResultAttachmentSupported('gemini', 'image/png', 'gemini-flash-latest')).toBe(true);
    });

    it('keeps no attachment for Gemini 2.x, which ignores functionResponse.parts', () => {
        expect(
            isToolResultAttachmentSupported(
                'gemini',
                'image/png',
                'locations/global/publishers/google/models/gemini-2.5-flash',
            ),
        ).toBe(false);
    });

    it.each([
        ['openai_responses', 'application/pdf', true],
        ['openai_responses', 'image/bmp', false],
        ['openai_responses', 'audio/wav', false],
        ['openai_chat', 'image/png', true],
        ['openai_chat', 'text/plain', true],
        ['openai_chat', 'application/pdf', false],
    ] as const)('%s: %s → %s', (api, mimeType, expected) => {
        expect(isToolResultAttachmentSupported(api, mimeType, 'gpt-5.6')).toBe(expected);
    });
});

describe('toolResultAttachmentApi', () => {
    it.each([
        ['anthropic', 'claude-sonnet-5-5', 'claude'],
        ['vertexai', 'locations/global/publishers/anthropic/models/claude-haiku-5-5', 'claude'],
        ['vertexai', 'locations/global/publishers/google/models/gemini-3.5-flash', 'gemini'],
        ['vertexai', 'publishers/meta/models/llama-4', undefined],
        ['bedrock', 'global.anthropic.claude-haiku-5-5', 'claude'],
        ['bedrock', 'us.amazon.nova-2-lite-v1:0', undefined],
        ['openai', 'gpt-5.6', 'openai_responses'],
        ['groq', 'llama-5', 'openai_chat'],
        ['mistralai', 'mistral-large', undefined],
        ['bedrock_mantle', 'anthropic.claude-haiku-5-5', 'claude'],
        ['bedrock_mantle', 'openai.gpt-5.6', 'openai_responses'],
        ['bedrock_mantle', 'openai.gpt-oss-120b', 'openai_chat'],
        ['bedrock_mantle', 'acme.unknown-1', undefined],
    ])('%s %s → %s', (provider, model, expected) => {
        expect(toolResultAttachmentApi(provider, model)).toBe(expected);
    });

    it('follows the openai_api of an OpenAI-compatible driver, Responses by default as in the factory', () => {
        expect(toolResultAttachmentApi('openai_compatible', 'my-model')).toBe('openai_responses');
        expect(toolResultAttachmentApi('openai_compatible', 'my-model', { openaiApi: 'responses' })).toBe(
            'openai_responses',
        );
        expect(toolResultAttachmentApi('openai_compatible', 'my-model', { openaiApi: 'chat_completions' })).toBe(
            'openai_chat',
        );
    });
});

describe('formatOmittedAttachmentNote', () => {
    it('names the file, its type and the model', () => {
        expect(
            formatOmittedAttachmentNote(
                { name: 'report.wav', mime_type: 'Audio/WAV' },
                'locations/global/publishers/google/models/gemini-3.5-flash',
            ),
        ).toBe(
            '[attachment omitted: report.wav (audio/wav) — gemini-3.5-flash cannot read this type in a tool result]',
        );
    });

    it('says so when the name or type is missing', () => {
        expect(formatOmittedAttachmentNote({}, 'gpt-5.6')).toBe(
            '[attachment omitted: unnamed (unknown type) — gpt-5.6 cannot read this type in a tool result]',
        );
    });
});
