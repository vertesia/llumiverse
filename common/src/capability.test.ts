import { describe, expect, it } from 'vitest';
import { getModelCapabilities, isEmbeddingModel, supportsToolUse } from './capability.js';
import { Providers } from './types.js';

describe('embedding model classification', () => {
    it.each([
        [{ id: 'text-embedding-3-small' }, Providers.openai],
        [{ id: 'text-embedding-ada-002' }, Providers.azure_openai],
        [{ id: 'mistral-embed' }, Providers.mistralai],
        [{ id: 'embedding-model', output_modalities: ['vectors'] }, Providers.xai],
        [{ id: 'chat-model', type: 'embedding' }, Providers.togetherai],
    ] as const)('recognizes an embedding listing: %s', (model, provider) => {
        expect(isEmbeddingModel(model, provider)).toBe(true);
    });

    it('does not classify normal inference models as embeddings', () => {
        expect(isEmbeddingModel({ id: 'gpt-5.6-sol' }, Providers.openai)).toBe(false);
        expect(isEmbeddingModel({ id: 'grok-4.3', output_modalities: ['text'] }, Providers.xai)).toBe(false);
    });

    it('does not hide non-embedding models when capability metadata is incomplete', () => {
        expect(isEmbeddingModel({ id: 'whisper-large-v3' }, Providers.groq)).toBe(false);
        expect(isEmbeddingModel({ id: 'canopylabs/orpheus-v1-english' }, Providers.groq)).toBe(false);
        expect(isEmbeddingModel({ id: 'meta-llama/llama-prompt-guard-2-86m' }, Providers.groq)).toBe(false);
        expect(isEmbeddingModel({ id: 'llama-3.3-70b-versatile' }, Providers.groq)).toBe(false);
        expect(isEmbeddingModel({ id: 'gpt-image-1' }, Providers.openai)).toBe(false);
        expect(isEmbeddingModel({ id: 'amazon.titan-image-generator-v3' }, Providers.bedrock)).toBe(false);
        expect(isEmbeddingModel({ id: 'amazon.nova-reel-v1' }, Providers.bedrock)).toBe(false);
    });
});

describe('xAI Grok tool capabilities', () => {
    it.each(['grok-2', 'grok-3', 'grok-4', 'grok-4-fast-reasoning'])(
        'enables tool use for %s via Providers.xai without setting tool_support_streaming',
        (model) => {
            const caps = getModelCapabilities(model, Providers.xai);
            expect(caps.tool_support).toBe(true);
            expect(caps.tool_support_streaming).toBeUndefined();
            // Streaming agents must still attach tools when the flag is omitted
            expect(supportsToolUse(model, Providers.xai, true)).toBe(true);
            expect(supportsToolUse(model, Providers.xai, false)).toBe(true);
        },
    );

    it('carries verified image input support into current and future Grok 4 models', () => {
        expect(getModelCapabilities('grok-4.20', Providers.xai).input.image).toBe(true);
        expect(getModelCapabilities('grok-4.5', Providers.xai).input.image).toBe(true);
    });

    it('masks unsupported platform audio and video modalities without mutating source metadata', () => {
        const caps = getModelCapabilities('future-audio-video-model', Providers.openai_compatible);
        expect(caps.input.audio).toBe(false);
        expect(caps.input.video).toBe(false);
        expect(caps.output.audio).toBe(false);
        expect(caps.output.video).toBe(false);
    });
});

describe('supportsToolUse streaming default', () => {
    it('falls back to tool_support when tool_support_streaming is unset', () => {
        // Vertex Grok records tool_support but not tool_support_streaming
        expect(getModelCapabilities('grok-3', Providers.vertexai).tool_support).toBe(true);
        expect(getModelCapabilities('grok-3', Providers.vertexai).tool_support_streaming).toBeUndefined();
        expect(supportsToolUse('grok-3', Providers.vertexai, true)).toBe(true);
    });

    it('honors explicit tool_support_streaming: false', () => {
        // Bedrock runtime Llama has tools but not streaming tools
        const caps = getModelCapabilities('meta.llama3-1-70b-instruct-v1:0', Providers.bedrock);
        expect(caps.tool_support).toBe(true);
        expect(caps.tool_support_streaming).toBe(false);
        expect(supportsToolUse('meta.llama3-1-70b-instruct-v1:0', Providers.bedrock, true)).toBe(false);
        expect(supportsToolUse('meta.llama3-1-70b-instruct-v1:0', Providers.bedrock, false)).toBe(true);
    });
});
